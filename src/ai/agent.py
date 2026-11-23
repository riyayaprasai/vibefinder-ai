"""The VibeFinder 2.0 workflow: natural-language request -> grounded, checked recommendations.

This is a deterministic, rule-driven multi-step tool workflow. It is NOT an LLM agent that
chooses its own tools: the step order is fixed in code, but several steps make real decisions
(stop and ask for clarification, stop on a blocked request, run an extra "relaxed" search when
the request conflicts with the catalog, fall back from LLM to offline text when verification
fails). Every step and decision is written to a JSON trace.

Steps
  1. check_input        guardrail on the raw text
  2. parse_request      rules + knowledge base + index (and optional LLM fields)
  3. decide             stop with a clarifying question if nothing musical was understood
  4. apply_exclusions   remove genres / moods the listener said no to
  5. check_feasibility  conflicts, missing genres, thin catalog coverage
  6. retrieve_bonuses   related-genre / related-mood credit and text relevance from the index
  7. rank               Project 3 recommend_songs (score_song + strategy + artist penalty)
  8. relax_conflicts    if a conflict was found, re-rank without the weakest conflicting field
  9. generate           explanation from evidence (offline template or LLM)
 10. verify_output      grounding check; fall back to the offline text if it fails
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.ai.generator import LLMGenerator, OfflineGenerator, build_evidence, evidence_text
from src.ai.guardrails import (WEAK_MATCH_RATIO, check_feasibility, check_input, match_strength,
                               max_possible_score, verify_explanation)
from src.ai.index import build_index
from src.ai.knowledge import KnowledgeBase
from src.ai.parser import llm_parse, parse_request
from src.recommender import load_songs, recommend_songs

SONGS_CSV = Path(__file__).resolve().parents[2] / "data" / "songs.csv"
BONUS_CAPS = {"related_genre": 1.0, "related_mood": 0.5, "text": 0.75}
TEXT_MIN_SIMILARITY = 0.05
RELAX_ORDER = ["energy", "mood", "genre"]  # which conflicting field to drop first


@dataclass
class AgentResult:
    status: str                 # "ok", "clarify", "rejected" or "no_results"
    request: str
    message: str = ""
    parsed: Optional[Dict] = None
    recommendations: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    alternatives: List[Dict] = field(default_factory=list)
    explanation: str = ""
    generator: str = ""
    verification: Optional[Dict] = None
    trace: Dict = field(default_factory=dict)


class Trace:
    """Records each action, its inputs/outputs and any decision taken."""

    def __init__(self, request: str, config: Dict):
        self.data = {"request": request, "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "config": config, "steps": []}
        self._t0 = time.perf_counter()

    def step(self, action: str, output: Dict, decision: str = "", **inputs) -> None:
        self.data["steps"].append({"n": len(self.data["steps"]) + 1, "action": action, "input": inputs,
                                   "output": output, "decision": decision,
                                   "elapsed_ms": round((time.perf_counter() - self._t0) * 1000, 1)})

    def finish(self, status: str) -> Dict:
        self.data["status"] = status
        return self.data

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        return path


class MusicAgent:
    def __init__(self, songs: Optional[List[Dict]] = None, kb: Optional[KnowledgeBase] = None,
                 level: str = "retrieval", backend: str = "offline", llm=None,
                 strategy: str = "default", artist_penalty: float = 0.5):
        self.songs = songs if songs is not None else load_songs(str(SONGS_CSV))
        self.kb = kb if kb is not None else KnowledgeBase.load()
        self.level = level
        self.index = build_index(self.songs, self.kb) if level == "retrieval" else None
        self.genres = sorted({s["genre"] for s in self.songs})
        self.moods = sorted({s["mood"] for s in self.songs})
        self.strategy = strategy
        self.artist_penalty = artist_penalty
        self.backend = backend
        self.llm = llm
        if backend == "ollama" and self.llm is None:
            from src.ai.llm import OllamaClient
            self.llm = OllamaClient()

    # ----- helpers -------------------------------------------------------------------------

    def _bonuses(self, parsed, songs: List[Dict], request: str) -> Dict[int, List]:
        if self.level == "baseline":
            return {}
        bonus: Dict[int, List] = {}
        for s in songs:
            items = []
            if s["genre"] != parsed.profile.get("genre") and s["genre"] in parsed.hint_genres:
                items.append((BONUS_CAPS["related_genre"], f"related genre: {s['genre']} (knowledge base)"))
            if s["mood"] != parsed.profile.get("mood") and s["mood"] in parsed.hint_moods:
                items.append((BONUS_CAPS["related_mood"], f"related mood: {s['mood']} (knowledge base)"))
            if items:
                bonus[s["id"]] = items
        if self.index is not None:
            allowed = {s["id"] for s in songs}
            hits = [h for h in self.index.search(request, source="song", top_k=len(self.songs))
                    if int(h.doc_id.split(":")[1]) in allowed]
            top = hits[0].score if hits else 0.0
            if top >= TEXT_MIN_SIMILARITY:
                for h in hits:
                    points = round(BONUS_CAPS["text"] * h.score / top, 2)
                    if points >= 0.05:
                        bonus.setdefault(int(h.doc_id.split(":")[1]), []).append(
                            (points, f"text relevance {h.score:.2f} (index)"))
        return bonus

    def _rank(self, profile: Dict, songs: List[Dict], k: int, bonus: Dict, max_score: float) -> List[Dict]:
        rows = recommend_songs(profile, songs, k=k, strategy=self.strategy,
                               artist_penalty=self.artist_penalty, bonus=bonus)
        return [{"song": s, "score": score, "explanation": why,
                 "match_strength": match_strength(score, max_score)} for s, score, why in rows]

    def _parse(self, text: str, trace: Trace):
        parsed = parse_request(text, self.kb, self.genres, self.moods, level=self.level, index=self.index)
        trace.step("parse_request", parsed.to_dict(), level=self.level)
        if self.backend == "ollama" and self.llm is not None:
            if not self.llm.available():
                trace.step("llm_parse", {"used": False}, decision="Ollama not reachable; keeping rule-based parse")
                return parsed
            try:
                llm_req = llm_parse(text, self.llm, self.genres, self.moods, few_shot=True)
            except Exception as exc:  # network or model failure must not break the app
                trace.step("llm_parse", {"used": False, "error": str(exc)}, decision="LLM failed; keeping rules")
                return parsed
            if llm_req is None:
                trace.step("llm_parse", {"used": False}, decision="LLM reply invalid; keeping rules")
                return parsed
            added = {}
            for key, value in llm_req.profile.items():
                if key not in parsed.explicit:
                    parsed.profile[key] = value
                    parsed.provenance[key] = "llm-few-shot"
                    parsed.explicit.append(key)
                    added[key] = value
            for g in llm_req.exclude_genres:
                if g not in parsed.exclude_genres:
                    parsed.exclude_genres.append(g)
            parsed.k = parsed.k or llm_req.k
            parsed.parser = "rules+llm"
            trace.step("llm_parse", {"used": True, "llm_fields": llm_req.profile, "added": added},
                       decision="merged LLM fields the rules had not set explicitly")
        return parsed

    # ----- main entry point ------------------------------------------------------------------

    def run(self, request: str, k: int = 5, save_trace=None) -> AgentResult:
        trace = Trace(request, {"level": self.level, "backend": self.backend, "strategy": self.strategy,
                                "artist_penalty": self.artist_penalty, "k": k})

        def done(result: AgentResult) -> AgentResult:
            result.trace = trace.finish(result.status)
            if save_trace:
                trace.save(save_trace)
            return result

        # 1. Input guardrail
        check = check_input(request)
        trace.step("check_input", {"ok": check.ok, "reason": check.reason},
                   decision="continue" if check.ok else "stop: request rejected")
        if not check.ok:
            return done(AgentResult("rejected", request or "", message=check.reason))
        text = check.cleaned

        # 2-3. Parse, then decide whether there is enough to act on
        parsed = self._parse(text, trace)
        if not parsed.has_signal:
            msg = ("I couldn't find a genre, mood, activity or sound description in that request. "
                   "Try something like 'upbeat pop for a workout' or 'calm music for studying'. "
                   f"Genres in the catalog: {', '.join(self.genres)}.")
            trace.step("decide", {"has_signal": False}, decision="stop: ask a clarifying question")
            return done(AgentResult("clarify", text, message=msg, parsed=parsed.to_dict()))
        trace.step("decide", {"has_signal": True}, decision="continue")

        notes = []
        if parsed.k:
            k = parsed.k
        if k > len(self.songs):
            notes.append(f"Asked for {k} songs but the catalog has {len(self.songs)}; showing all.")
            k = len(self.songs)

        # 4. Exclusions
        pool = [s for s in self.songs if s["genre"] not in parsed.exclude_genres
                and s["mood"] not in parsed.exclude_moods]
        trace.step("apply_exclusions", {"remaining": len(pool), "removed": len(self.songs) - len(pool)},
                   exclude_genres=parsed.exclude_genres, exclude_moods=parsed.exclude_moods,
                   decision="continue" if pool else "stop: nothing left")
        if not pool:
            return done(AgentResult("no_results", text, message="Every song was excluded by the request.",
                                    parsed=parsed.to_dict()))
        k = min(k, len(pool))

        # 5. Feasibility
        feas = check_feasibility(parsed, pool, k)
        notes += feas.notes
        trace.step("check_feasibility", {"warnings": feas.warnings, "notes": feas.notes,
                                         "conflicts": [c["fields"] for c in feas.conflicts]},
                   decision="conflict found: will also search with a relaxed request" if feas.conflicts else "no conflict")

        # 6. Retrieval bonuses
        bonus = self._bonuses(parsed, pool, text)
        trace.step("retrieve_bonuses", {"songs_with_bonus": len(bonus), "hint_genres": parsed.hint_genres,
                                        "hint_moods": parsed.hint_moods,
                                        "top_text_matches": [  # the three largest text bonuses
                                            {"song_id": sid, "points": p}
                                            for sid, p in sorted(((sid, pts) for sid, items in bonus.items()
                                                                  for pts, why in items if why.startswith("text")),
                                                                 key=lambda x: -x[1])[:3]]})

        # 7. Rank with the Project 3 engine
        max_score = max_possible_score(parsed.profile, BONUS_CAPS if self.level != "baseline" else {})
        results = self._rank(parsed.profile, pool, k, bonus, max_score)
        warnings = list(feas.warnings)
        if results and results[0]["match_strength"] < WEAK_MATCH_RATIO:
            warnings.append(f"Weak match: the best song earns only {results[0]['match_strength']:.0%} of the "
                            f"points this request could earn.")
        trace.step("rank", {"max_possible": max_score,
                            "top": [{"title": r["song"]["title"], "score": r["score"],
                                     "match_strength": r["match_strength"]} for r in results]},
                   strategy=self.strategy, k=k)

        # 8. Relax conflicting fields and report the best alternative
        alternatives, tried = [], set()
        for conflict in feas.conflicts:
            for drop in [f for f in RELAX_ORDER if f in conflict["fields"] and f not in tried]:
                tried.add(drop)
                relaxed = {key: v for key, v in parsed.profile.items() if key != drop}
                # Partial credit that only exists because of the dropped label is removed too.
                prefix = f"related {drop}"
                relaxed_bonus = {sid: [b for b in items if not b[1].startswith(prefix)] for sid, items in bonus.items()}
                alt = self._rank(relaxed, pool, 1, relaxed_bonus, max_possible_score(relaxed, BONUS_CAPS))
                found = None
                if alt and alt[0]["song"]["id"] != results[0]["song"]["id"] \
                        and alt[0]["song"]["id"] not in {a["song_id"] for a in alternatives}:
                    found = {"dropped": drop, "song_id": alt[0]["song"]["id"], "title": alt[0]["song"]["title"],
                             "artist": alt[0]["song"]["artist"], "score": alt[0]["score"]}
                    alternatives.append(found)
                trace.step("relax_conflict", {"alternative": found}, conflict=conflict["fields"], dropped=drop,
                           decision="offer alternative" if found else "relaxed search gave the same top song")
        for a in alternatives:
            warnings.append(f"If you drop the {a['dropped']} requirement, the top pick would be "
                            f"'{a['title']}' by {a['artist']}.")

        # 9-10. Generate and verify
        evidence = build_evidence(results, self.kb)
        understood = ", ".join(f"{key}={v if not isinstance(v, list) else '/'.join(v)}"
                               for key, v in parsed.profile.items())
        summary = f"Top {len(results)} for \"{text}\" (understood as: {understood or 'general taste'})."
        if parsed.activities:
            summary += f" Activity: {', '.join(parsed.activities)}."
        offline = OfflineGenerator()
        generator = offline
        if self.backend == "ollama" and self.llm is not None and self.llm.available():
            generator = LLMGenerator(self.llm)
        try:
            explanation = generator.generate(text, evidence, warnings + notes, summary)
        except Exception as exc:
            trace.step("generate", {"generator": generator.name, "error": str(exc)}, decision="fall back to offline")
            generator, explanation = offline, offline.generate(text, evidence, warnings + notes, summary)
        else:
            trace.step("generate", {"generator": generator.name, "chars": len(explanation)})
        ev_text = evidence_text(evidence, warnings + notes) + "\n" + summary
        check_out = verify_explanation(explanation, [r["song"] for r in results], self.songs, ev_text)
        decision = "accept"
        if not check_out.ok and generator is not offline:
            decision = "reject LLM text; fall back to offline template"
            generator, explanation = offline, offline.generate(text, evidence, warnings + notes, summary)
            check_out = verify_explanation(explanation, [r["song"] for r in results], self.songs, ev_text)
        trace.step("verify_output", {"ok": check_out.ok, "problems": check_out.problems}, decision=decision)

        return done(AgentResult(
            "ok", text, parsed=parsed.to_dict(), recommendations=results, warnings=warnings, notes=notes,
            alternatives=alternatives, explanation=explanation, generator=generator.name,
            verification={"ok": check_out.ok, "problems": check_out.problems}))

"""Builds the explanation shown to the listener from retrieved evidence.

The evidence for each recommended song is: its catalog facts (source 1: songs.csv), the score
reasons from Project 3's score_song plus retrieval bonuses, and the knowledge-base note for its
genre (source 2: data/knowledge). Two generators use the same evidence:

  * OfflineGenerator - deterministic template that only rearranges evidence sentences. It is
                       NOT a language model; it is the reproducible default.
  * LLMGenerator     - sends the evidence to a local Ollama model and asks for a grounded
                       write-up (retrieval-augmented generation). Its output is verified by
                       guardrails.verify_explanation and replaced by the offline text if it fails.
"""

import re
from typing import Dict, List

from src.ai.knowledge import KnowledgeBase


def _first_sentence(text: str) -> str:
    match = re.match(r"(.+?[.!?])(\s|$)", text or "")
    return match.group(1) if match else (text or "")


def _top_reasons(explanation: str, n: int = 3) -> List[str]:
    """Largest positive contributions from a '; '-joined reasons string."""
    scored = []
    for part in explanation.split("; "):
        m = re.search(r"\(([+-])(\d+\.\d+)\)\s*$", part)
        if m and m.group(1) == "+" and float(m.group(2)) > 0:
            scored.append((float(m.group(2)), part[: m.start()].strip()))
    scored.sort(key=lambda x: -x[0])
    return [text for _, text in scored[:n]]


def build_evidence(results: List[Dict], kb: KnowledgeBase) -> List[Dict]:
    """One evidence record per recommended song."""
    evidence = []
    for rank, row in enumerate(results, 1):
        song = row["song"]
        genre_doc = kb.get("genre", song["genre"])
        evidence.append({
            "rank": rank, "title": song["title"], "artist": song["artist"],
            "genre": song["genre"], "mood": song["mood"], "energy": song["energy"],
            "score": row["score"], "match_strength": row["match_strength"],
            "top_reasons": _top_reasons(row["explanation"]),
            "penalized": "artist repeat penalty" in row["explanation"],
            "knowledge": _first_sentence(genre_doc.description) if genre_doc else "",
            "knowledge_source": f"genres.md#{song['genre']}" if genre_doc else "",
        })
    return evidence


def evidence_text(evidence: List[Dict], warnings: List[str]) -> str:
    """Plain-text evidence block (what an LLM is allowed to use, and what the verifier checks)."""
    lines = []
    for e in evidence:
        lines.append(f"[{e['rank']}] \"{e['title']}\" by {e['artist']} | genre {e['genre']} | mood {e['mood']} "
                     f"| energy {e['energy']} | score {e['score']:.2f} | match strength {e['match_strength']:.2f}")
        lines.append("    reasons: " + "; ".join(e["top_reasons"]))
        if e["knowledge"]:
            lines.append(f"    note ({e['knowledge_source']}): {e['knowledge']}")
    for w in warnings:
        lines.append(f"WARNING: {w}")
    return "\n".join(lines)


class OfflineGenerator:
    name = "offline-template"

    def generate(self, request: str, evidence: List[Dict], warnings: List[str], summary: str) -> str:
        out = [summary]
        for e in evidence:
            reasons = "; ".join(e["top_reasons"]) or "closest overall match"
            line = (f"{e['rank']}. {e['title']} by {e['artist']} ({e['genre']}, {e['mood']}) "
                    f"- match strength {e['match_strength']:.2f}. Strongest reasons: {reasons}.")
            if e["knowledge"]:
                line += f" Genre note: {e['knowledge']}"
            if e["penalized"]:
                line += f" (Artist-repeat penalty applied: {e['artist']} already appears higher in the list.)"
            out.append(line)
        for w in warnings:
            out.append(f"Note: {w}")
        return "\n".join(out)


class LLMGenerator:
    def __init__(self, llm):
        self.llm = llm
        self.name = f"llm:{getattr(llm, 'name', 'unknown')}:{getattr(llm, 'model', '')}"

    def build_prompt(self, request: str, evidence: List[Dict], warnings: List[str]) -> str:
        return "\n".join([
            "You are a music recommendation assistant. Explain the recommendations below to the listener.",
            "Rules: use ONLY the facts in the EVIDENCE block. Mention every song by its exact title, in the",
            "given order. Do not mention any other songs. Do not invent numbers. If there are WARNING lines,",
            "tell the listener about them honestly. Keep it under 150 words.",
            "",
            f"LISTENER REQUEST: {request}",
            "",
            "EVIDENCE:",
            evidence_text(evidence, warnings),
        ])

    def generate(self, request: str, evidence: List[Dict], warnings: List[str], summary: str) -> str:
        return self.llm.complete(self.build_prompt(request, evidence, warnings), temperature=0.2).strip()

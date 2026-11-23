"""Reliability checks around the recommender.

1. check_input       - rejects empty, oversized or instruction-injection requests before parsing.
2. check_feasibility - compares what the listener explicitly asked for with what the catalog
                       actually contains, and reports conflicts instead of silently returning
                       a poor list (the Project 3 model card listed this as a known weakness).
3. match_strength    - score divided by the best score this profile could possibly earn.
                       It is a ratio of real points, NOT a probability or a calibrated confidence.
4. verify_explanation - checks generated text against the evidence: it may only name songs that
                       were recommended, must mention each of them, and may only use numbers
                       that appear in the evidence.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.recommender import (W_ACOUSTIC, W_DANCE, W_DECADE, W_ENERGY, W_GENRE, W_INSTRUMENTAL,
                             W_LIVENESS, W_MOOD, W_POPULARITY, W_TAGS, W_TEMPO, W_VALENCE)

MAX_REQUEST_CHARS = 400
CONFLICT_ENERGY_GAP = 0.3
WEAK_MATCH_RATIO = 0.55
INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts?)",
    r"system prompt", r"you are now", r"disregard (the |all )?(rules|instructions)",
    r"developer mode", r"jailbreak", r"<\s*/?\s*(script|system)\b",
]


@dataclass
class InputCheck:
    ok: bool
    reason: str = ""
    cleaned: str = ""


def check_input(text: Optional[str]) -> InputCheck:
    if text is None or not text.strip():
        return InputCheck(False, "The request is empty. Describe the music you want, e.g. "
                                 "'chill lofi for studying'.")
    cleaned = " ".join(text.split())
    if len(cleaned) > MAX_REQUEST_CHARS:
        return InputCheck(False, f"The request is {len(cleaned)} characters; the limit is "
                                 f"{MAX_REQUEST_CHARS}. Please shorten it.")
    lowered = cleaned.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return InputCheck(False, "The request looks like an instruction to the system rather "
                                     "than a music request, so it was not processed.")
    return InputCheck(True, cleaned=cleaned)


@dataclass
class Feasibility:
    warnings: List[str] = field(default_factory=list)
    conflicts: List[Dict] = field(default_factory=list)  # each: {"fields": [...], "message": str}
    notes: List[str] = field(default_factory=list)


def check_feasibility(parsed, songs: List[Dict], k: int) -> Feasibility:
    """Checks the explicit request against the catalog that remains after exclusions."""
    result = Feasibility()
    prof, explicit = parsed.profile, set(parsed.explicit)

    for g in parsed.unknown_genres:
        alt = ", ".join(parsed.hint_genres[:3]) or "the closest songs by sound"
        result.warnings.append(f"The catalog has no {g} songs; showing related genres instead ({alt}).")
    for m in parsed.unknown_moods:
        result.warnings.append(f"The catalog has no songs labeled '{m}'.")

    genre = prof.get("genre") if "genre" in explicit else None
    mood = prof.get("mood") if "mood" in explicit else None
    energy = prof.get("energy") if "energy" in explicit else None

    if genre:
        n = sum(s["genre"] == genre for s in songs)
        if n < k:
            result.notes.append(f"Only {n} {genre} song(s) exist in the catalog, so the other "
                                f"{k - n} pick(s) come from other genres.")

    def any_song(pred) -> bool:
        return any(pred(s) for s in songs)

    def near(s):
        return abs(s["energy"] - energy) <= CONFLICT_ENERGY_GAP

    if genre and mood and not any_song(lambda s: s["genre"] == genre and s["mood"] == mood):
        result.conflicts.append({"fields": ["genre", "mood"],
                                 "message": f"No song is both {genre} and {mood}."})
    if mood and energy is not None and any_song(lambda s: s["mood"] == mood) \
            and not any_song(lambda s: s["mood"] == mood and near(s)):
        closest = min((s for s in songs if s["mood"] == mood), key=lambda s: abs(s["energy"] - energy))
        result.conflicts.append({"fields": ["mood", "energy"],
                                 "message": f"No {mood} song has energy near {energy}; the closest "
                                            f"is '{closest['title']}' at {closest['energy']}."})
    if genre and energy is not None and any_song(lambda s: s["genre"] == genre) \
            and not any_song(lambda s: s["genre"] == genre and near(s)):
        closest = min((s for s in songs if s["genre"] == genre), key=lambda s: abs(s["energy"] - energy))
        result.conflicts.append({"fields": ["genre", "energy"],
                                 "message": f"No {genre} song has energy near {energy}; the closest "
                                            f"is '{closest['title']}' at {closest['energy']}."})
    for c in result.conflicts:
        result.warnings.append("Conflicting request: " + c["message"])
    return result


def max_possible_score(profile: Dict, bonus_caps: Dict[str, float]) -> float:
    """The best total this profile could earn: core weights, requested extras and bonus caps."""
    total = W_ENERGY + W_VALENCE + W_DANCE + W_ACOUSTIC + W_TEMPO
    total += W_GENRE if "genre" in profile else bonus_caps.get("related_genre", 0.0)
    total += W_MOOD if "mood" in profile else bonus_caps.get("related_mood", 0.0)
    for key, weight in (("popularity", W_POPULARITY), ("release_decade", W_DECADE),
                        ("instrumentalness", W_INSTRUMENTAL), ("liveness", W_LIVENESS),
                        ("mood_tags", W_TAGS)):
        if key in profile:
            total += weight
    total += bonus_caps.get("text", 0.0)
    return round(total, 2)


def match_strength(score: float, max_score: float) -> float:
    return round(max(0.0, score) / max_score, 2) if max_score else 0.0


@dataclass
class Verification:
    ok: bool
    problems: List[str] = field(default_factory=list)


def _numbers(text: str) -> set:
    return {n.rstrip(".") for n in re.findall(r"\d+(?:\.\d+)?", text)}


def verify_explanation(text: str, recommended: List[Dict], catalog: List[Dict], evidence: str) -> Verification:
    problems = []
    lowered = (text or "").lower()
    if not lowered.strip():
        return Verification(False, ["explanation is empty"])
    rec_titles = {s["title"].lower() for s in recommended}
    for s in catalog:
        t = s["title"].lower()
        if t in lowered and t not in rec_titles:
            problems.append(f"mentions '{s['title']}', which was not recommended")
    for s in recommended:
        if s["title"].lower() not in lowered:
            problems.append(f"does not mention recommended song '{s['title']}'")
    allowed = _numbers(evidence) | {str(i) for i in range(0, 11)}
    invented = sorted(_numbers(text) - allowed)
    if invented:
        problems.append(f"uses numbers not found in the evidence: {', '.join(invented)}")
    return Verification(not problems, problems)

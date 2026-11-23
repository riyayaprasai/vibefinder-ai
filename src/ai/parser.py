"""Turns a natural-language request into a Project 3 preference profile.

Three parsing levels, so the evaluation can measure what each layer adds:
  * "baseline"  - only the exact catalog genre / mood labels plus generic cues (energy words,
                  BPM, decade, "instrumental", "top 3"). This is roughly what Project 3 could
                  understand if you typed its own labels.
  * "lookup"    - baseline + knowledge-base aliases (e.g. "sad" -> melancholy, "edm" ->
                  electronic) and activity notes (e.g. "studying" fills energy/tempo/tags).
  * "retrieval" - lookup + the TF-IDF index: knowledge docs whose text is similar to the
                  request add their hints even when no alias was typed.

Every field records where it came from (its provenance), which the agent trace shows.
An optional LLM parser (llm_parse) can replace the rule parser when Ollama is running.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.ai.index import Index
from src.ai.knowledge import KnowledgeBase

LEVELS = ("baseline", "lookup", "retrieval")
NEGATORS = {"no", "not", "without", "except", "avoid", "hate", "dont", "skip", "minus", "nothing"}
RETRIEVAL_MIN_SCORE = 0.18   # knowledge docs below this cosine similarity are ignored
MUSIC_WORDS = {"music", "song", "songs", "track", "tracks", "playlist", "listen", "listening", "vibe",
               "vibes", "beats", "tunes", "soundtrack", "album", "albums", "artist", "artists", "band",
               "bands", "jam", "jams", "genre", "bpm"}
WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10}

ENERGY_CUES: List[Tuple[str, float]] = [
    ("high energy", 0.9), ("high-energy", 0.9), ("energetic", 0.88), ("upbeat", 0.8),
    ("loud", 0.9), ("hard hitting", 0.92), ("banger", 0.9), ("bangers", 0.9),
    ("low energy", 0.25), ("low-energy", 0.25), ("quiet", 0.25), ("soft", 0.3),
    ("gentle", 0.3), ("calm", 0.3), ("peaceful", 0.25), ("medium energy", 0.55),
]
TEMPO_CUES: List[Tuple[str, float]] = [("fast", 145.0), ("slow", 70.0), ("mid tempo", 105.0),
                                       ("midtempo", 105.0)]
POPULARITY_CUES: List[Tuple[str, float]] = [
    ("popular", 85), ("mainstream", 85), ("hits", 85), ("underground", 40), ("obscure", 40),
    ("hidden gems", 40), ("lesser known", 40), ("lesser-known", 40),
]
INSTRUMENTAL_CUES = ["instrumental", "no vocals", "without vocals", "no lyrics", "without lyrics",
                     "no singing"]
ACOUSTIC_YES = ["acoustic", "unplugged"]
ACOUSTIC_NO = ["electric", "not acoustic", "no acoustic"]


@dataclass
class ParsedRequest:
    """The parser's output: a profile for score_song plus everything the agent needs."""
    text: str
    level: str
    profile: Dict = field(default_factory=dict)
    provenance: Dict[str, str] = field(default_factory=dict)
    explicit: List[str] = field(default_factory=list)       # fields the user actually typed
    exclude_genres: List[str] = field(default_factory=list)
    exclude_moods: List[str] = field(default_factory=list)
    hint_genres: List[str] = field(default_factory=list)    # partial-credit genres
    hint_moods: List[str] = field(default_factory=list)     # partial-credit moods
    unknown_genres: List[str] = field(default_factory=list)  # asked for but not in catalog
    unknown_moods: List[str] = field(default_factory=list)
    activities: List[str] = field(default_factory=list)
    retrieved: List[Tuple[str, float]] = field(default_factory=list)
    k: Optional[int] = None
    parser: str = "rules"
    music_intent: bool = False

    @property
    def has_signal(self) -> bool:
        return bool(self.profile or self.exclude_genres or self.hint_genres or self.activities
                    or self.unknown_genres)

    def to_dict(self) -> Dict:
        return {
            "parser": self.parser, "level": self.level, "profile": self.profile,
            "provenance": self.provenance, "explicit": self.explicit,
            "exclude_genres": self.exclude_genres, "exclude_moods": self.exclude_moods,
            "hint_genres": self.hint_genres, "hint_moods": self.hint_moods,
            "unknown_genres": self.unknown_genres, "unknown_moods": self.unknown_moods,
            "activities": self.activities, "retrieved": self.retrieved, "k": self.k,
        }


def normalize(text: str) -> str:
    """Lowercase, drop apostrophes, keep letters/digits/&/-, collapse spaces."""
    text = (text or "").lower().replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9&\-\s]", " ", text)
    return " ".join(text.split())


def _find(phrase: str, text: str) -> List[int]:
    """Token positions where phrase occurs as whole words in normalized text."""
    tokens, target = text.split(), normalize(phrase).split()
    n = len(target)
    return [i for i in range(len(tokens) - n + 1) if tokens[i:i + n] == target] if n else []


def _negated(text: str, pos: int) -> bool:
    tokens = text.split()
    return any(t in NEGATORS for t in tokens[max(0, pos - 2):pos])


def _set(req: ParsedRequest, key: str, value, source: str, explicit: bool = False) -> None:
    """Sets a profile field unless a stronger source already set it."""
    if key in req.profile:
        return
    req.profile[key] = value
    req.provenance[key] = source
    if explicit:
        req.explicit.append(key)


def _match_labels(text: str, names_to_aliases: Dict[str, List[str]]) -> List[Tuple[int, str, bool, str]]:
    """Finds label mentions. Returns (position, label, negated, matched_alias), longest alias first."""
    candidates = sorted(((alias, name) for name, aliases in names_to_aliases.items() for alias in aliases),
                        key=lambda p: -len(p[0].split()))
    taken, found = set(), []
    for alias, name in candidates:
        n = len(normalize(alias).split())
        for pos in _find(alias, text):
            span = set(range(pos, pos + n))
            if span & taken:
                continue
            taken |= span
            found.append((pos, name, _negated(text, pos), alias))
    found.sort()
    return found


def _apply_cues(req: ParsedRequest, text: str) -> None:
    """Generic cues every level understands."""
    for cue, value in ENERGY_CUES:
        hits = [p for p in _find(cue, text) if not _negated(text, p)]
        if hits:
            _set(req, "energy", value, f"cue:'{cue}'", explicit=True)
            break
    bpm = re.search(r"\b(\d{2,3})\s*bpm\b", text)
    if bpm:
        _set(req, "tempo", float(bpm.group(1)), "cue:bpm", explicit=True)
    for cue, value in TEMPO_CUES:
        if [p for p in _find(cue, text) if not _negated(text, p)]:
            _set(req, "tempo", value, f"cue:'{cue}'", explicit=True)
            break
    decade = re.search(r"\b(?:19|20)?([0-9])0s\b", text)
    if decade:
        digit = int(decade.group(1))
        year = (1900 if digit >= 5 else 2000) + digit * 10
        _set(req, "release_decade", year, "cue:decade", explicit=True)
    for cue, value in POPULARITY_CUES:
        if [p for p in _find(cue, text) if not _negated(text, p)]:
            _set(req, "popularity", value, f"cue:'{cue}'", explicit=True)
            break
    if any(_find(c, text) for c in INSTRUMENTAL_CUES):
        _set(req, "instrumentalness", 0.85, "cue:instrumental", explicit=True)
    if any(_find(c, text) for c in ACOUSTIC_NO):
        _set(req, "likes_acoustic", False, "cue:not acoustic", explicit=True)
    elif any(not _negated(text, p) for c in ACOUSTIC_YES for p in _find(c, text)):
        _set(req, "likes_acoustic", True, "cue:acoustic", explicit=True)
    count = re.search(r"\b(?:top\s+)?(\d{1,2}|" + "|".join(WORD_NUMBERS) + r")\s+(?:songs?|tracks?|picks?)\b", text) \
        or re.search(r"\btop\s+(\d{1,2}|" + "|".join(WORD_NUMBERS) + r")\b", text)
    if count:
        raw = count.group(1)
        req.k = WORD_NUMBERS.get(raw) or int(raw)


def _apply_doc_hints(req: ParsedRequest, doc, source: str) -> None:
    """Activity / mood docs fill fields the listener left open, and add partial-credit hints."""
    for key, value in doc.profile.items():
        _set(req, key, value, source)
    if doc.tags:
        _set(req, "mood_tags", list(doc.tags), source)
    for g in doc.genres:
        if g not in req.hint_genres and g not in req.exclude_genres:
            req.hint_genres.append(g)
    for m in doc.moods:
        if m not in req.hint_moods and m not in req.exclude_moods:
            req.hint_moods.append(m)


def parse_request(text: str, kb: KnowledgeBase, catalog_genres: List[str], catalog_moods: List[str],
                  level: str = "retrieval", index: Optional[Index] = None) -> ParsedRequest:
    """Rule-based parser. See module docstring for the three levels."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    norm = normalize(text)
    req = ParsedRequest(text=text, level=level)
    use_kb = level in ("lookup", "retrieval")

    # 1. Genres and moods.
    for kind, catalog_names, out_exclude in (("genre", catalog_genres, req.exclude_genres),
                                             ("mood", catalog_moods, req.exclude_moods)):
        if use_kb:
            vocab = {d.name: [d.name] + d.aliases for d in kb.of_kind(kind)}
        else:
            vocab = {name: [name] for name in catalog_names}
        chosen = None
        for _, name, negated, alias in _match_labels(norm, vocab):
            if negated:
                if name not in out_exclude:
                    out_exclude.append(name)
            elif name not in catalog_names:
                unknown = req.unknown_genres if kind == "genre" else req.unknown_moods
                if name not in unknown:
                    unknown.append(name)
            elif chosen is None:
                chosen = name
                _set(req, kind, name, f"{'alias' if alias != name else 'label'}:'{alias}'", explicit=True)
            else:
                hints = req.hint_genres if kind == "genre" else req.hint_moods
                if name not in hints:
                    hints.append(name)

        if use_kb:
            # A requested label (or one the catalog lacks) brings its related labels as partial credit.
            hints = req.hint_genres if kind == "genre" else req.hint_moods
            unknown = req.unknown_genres if kind == "genre" else req.unknown_moods
            for base in ([req.profile[kind]] if kind in req.profile else []) + unknown:
                for rel in kb.related(kind, base):
                    if rel in catalog_names and rel not in hints and rel not in out_exclude:
                        hints.append(rel)

    # 2. Generic cues.
    _apply_cues(req, norm)

    if not use_kb:
        return req

    # 3. Mood defaults (e.g. "sad" implies low valence) for fields still open.
    if "mood" in req.profile:
        mood_doc = kb.get("mood", req.profile["mood"])
        if mood_doc:
            for key, value in mood_doc.profile.items():
                _set(req, key, value, f"mood-note:{mood_doc.name}")

    # 4. Activities named with an alias.
    vocab = {d.name: [d.name] + d.aliases for d in kb.of_kind("activity")}
    for _, name, negated, _ in _match_labels(norm, vocab):
        if not negated and name not in req.activities:
            req.activities.append(name)
            _apply_doc_hints(req, kb.get("activity", name), f"activity:{name}")

    # 5. Retrieval: knowledge docs similar to the request text. Guardrail: similarity alone may
    #    not decide that a request is about music ("tell me a joke" is weakly similar to the
    #    country note), so retrieval only runs when the request already shows music intent.
    req.music_intent = bool(req.has_signal or set(norm.split()) & MUSIC_WORDS)
    if level == "retrieval" and index is not None and req.music_intent:
        for hit in index.search(text, source="knowledge", top_k=3, min_score=RETRIEVAL_MIN_SCORE):
            kind, name = hit.doc_id.split(":", 1)
            req.retrieved.append((hit.doc_id, hit.score))
            if kind == "activity" and not req.activities:  # at most one activity fills the profile
                req.activities.append(name)
                _apply_doc_hints(req, kb.get("activity", name), f"retrieved:{hit.doc_id}")
            elif kind == "genre" and name in catalog_genres and name not in req.exclude_genres \
                    and name != req.profile.get("genre") and name not in req.hint_genres:
                req.hint_genres.append(name)
            elif kind == "mood" and name in catalog_moods and name not in req.exclude_moods \
                    and name != req.profile.get("mood") and name not in req.hint_moods:
                req.hint_moods.append(name)

    # A hint must never contradict the main label or an exclusion.
    req.hint_genres = [g for g in req.hint_genres if g != req.profile.get("genre") and g not in req.exclude_genres]
    req.hint_moods = [m for m in req.hint_moods if m != req.profile.get("mood") and m not in req.exclude_moods]
    return req


# ---------------------------------------------------------------------------------------------
# Optional LLM parser (used only with the Ollama backend)
# ---------------------------------------------------------------------------------------------

LLM_FIELDS = {"genre": str, "mood": str, "energy": float, "tempo": float, "likes_acoustic": bool,
              "instrumentalness": float, "exclude_genres": list, "k": int}

FEW_SHOT_EXAMPLES = [
    ("upbeat pop for a morning run, nothing acoustic",
     {"genre": "pop", "mood": "happy", "energy": 0.85, "likes_acoustic": False}),
    ("sad piano pieces to cry to, slow please",
     {"genre": "classical", "mood": "melancholy", "energy": 0.25, "tempo": 65}),
    ("3 songs for a house party but no rap",
     {"genre": "electronic", "mood": "euphoric", "energy": 0.85, "exclude_genres": ["hip hop"], "k": 3}),
]


def build_llm_prompt(text: str, catalog_genres: List[str], catalog_moods: List[str], few_shot: bool) -> str:
    lines = [
        "You convert a music request into JSON for a recommender.",
        f"Allowed genre values: {', '.join(sorted(catalog_genres))}.",
        f"Allowed mood values: {', '.join(sorted(catalog_moods))}.",
        "Keys (all optional): genre, mood, energy (0-1), tempo (BPM), likes_acoustic (true/false),",
        "instrumentalness (0-1), exclude_genres (list), k (number of songs).",
        "Leave out any key the request does not imply. Reply with JSON only.",
    ]
    if few_shot:
        lines.append("")
        for example, answer in FEW_SHOT_EXAMPLES:
            lines += [f"Request: {example}", f"JSON: {json.dumps(answer)}", ""]
    lines += [f"Request: {text}", "JSON:"]
    return "\n".join(lines)


def validate_llm_fields(raw: Dict, catalog_genres: List[str], catalog_moods: List[str]) -> Dict:
    """Keeps only well-typed, in-range fields. Anything invented is dropped, not trusted."""
    clean: Dict = {}
    for key, kind in LLM_FIELDS.items():
        if key not in raw or raw[key] is None:
            continue
        value = raw[key]
        try:
            if kind is str:
                value = str(value).lower().strip()
                allowed = catalog_genres if key == "genre" else catalog_moods
                if value not in allowed:
                    continue
            elif kind is float:
                value = float(value)
                if key == "tempo":
                    if not 40 <= value <= 220:
                        continue
                elif not 0.0 <= value <= 1.0:
                    continue
            elif kind is bool:
                if not isinstance(value, bool):
                    continue
            elif kind is int:
                value = int(value)
                if not 1 <= value <= 20:
                    continue
            elif kind is list:
                value = [str(v).lower() for v in value if str(v).lower() in catalog_genres]
                if not value:
                    continue
        except (TypeError, ValueError):
            continue
        clean[key] = value
    return clean


def extract_json(reply: str) -> Optional[Dict]:
    match = re.search(r"\{.*\}", reply or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def llm_parse(text: str, llm, catalog_genres: List[str], catalog_moods: List[str],
              few_shot: bool = True) -> Optional[ParsedRequest]:
    """Asks the LLM for a profile. Returns None when the reply is unusable."""
    reply = llm.complete(build_llm_prompt(text, catalog_genres, catalog_moods, few_shot), json_mode=True)
    data = extract_json(reply)
    if data is None:
        return None
    fields = validate_llm_fields(data, catalog_genres, catalog_moods)
    if not fields:
        return None
    req = ParsedRequest(text=text, level="llm", parser="llm-few-shot" if few_shot else "llm-zero-shot")
    req.exclude_genres = fields.pop("exclude_genres", [])
    req.k = fields.pop("k", None)
    for key, value in fields.items():
        _set(req, key, value, "llm", explicit=True)
    return req

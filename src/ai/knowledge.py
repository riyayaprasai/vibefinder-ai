"""Loads the hand-written knowledge base in data/knowledge/*.md.

Every '## name' section is one document. Lines of the form 'key: value' right under the
heading are structured fields; the remaining text is the description that the index searches
and that explanations quote. The file a section comes from decides its kind:
genres.md -> "genre", moods.md -> "mood", activities.md -> "activity".
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge"
KIND_BY_FILE = {"genres.md": "genre", "moods.md": "mood", "activities.md": "activity"}
LIST_FIELDS = ("aliases", "related", "genres", "moods", "tags")


@dataclass
class KnowledgeDoc:
    """One knowledge-base entry (a genre, a mood or an activity)."""
    kind: str
    name: str
    description: str
    aliases: List[str] = field(default_factory=list)
    related: List[str] = field(default_factory=list)
    genres: List[str] = field(default_factory=list)
    moods: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    profile: Dict = field(default_factory=dict)
    source: str = ""

    @property
    def doc_id(self) -> str:
        return f"{self.kind}:{self.name}"


def _parse_value(raw: str):
    """Turns a profile value into bool, float or str."""
    raw = raw.strip()
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return float(raw)
    except ValueError:
        return raw


def _parse_profile(raw: str) -> Dict:
    """'energy=0.35, likes_acoustic=true' -> {'energy': 0.35, 'likes_acoustic': True}."""
    out = {}
    for part in raw.split(","):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key.strip()] = _parse_value(value)
    return out


def parse_markdown(text: str, kind: str, source: str = "") -> List[KnowledgeDoc]:
    """Parses one knowledge file into documents."""
    docs: List[KnowledgeDoc] = []
    current: Optional[dict] = None

    def flush():
        if current is not None:
            current["description"] = " ".join(current.pop("lines")).strip()
            docs.append(KnowledgeDoc(kind=kind, source=source, **current))

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            flush()
            current = {"name": stripped[3:].strip().lower(), "lines": [], "profile": {}}
            continue
        if current is None or not stripped:
            continue
        key, sep, value = stripped.partition(":")
        key = key.strip().lower()
        if sep and not current["lines"] and (key in LIST_FIELDS or key == "profile"):
            if key == "profile":
                current["profile"] = _parse_profile(value)
            else:
                current[key] = [v.strip().lower() for v in value.split(",") if v.strip()]
        else:
            current["lines"].append(stripped)
    flush()
    return docs


class KnowledgeBase:
    """All knowledge documents, with lookups by kind and name."""

    def __init__(self, docs: List[KnowledgeDoc]):
        self.docs = docs
        self._by_id = {d.doc_id: d for d in docs}

    @classmethod
    def load(cls, directory: Path = KNOWLEDGE_DIR) -> "KnowledgeBase":
        docs: List[KnowledgeDoc] = []
        for filename, kind in KIND_BY_FILE.items():
            path = Path(directory) / filename
            if path.exists():
                docs.extend(parse_markdown(path.read_text(encoding="utf-8"), kind, source=filename))
        return cls(docs)

    def get(self, kind: str, name: str) -> Optional[KnowledgeDoc]:
        return self._by_id.get(f"{kind}:{(name or '').lower()}")

    def of_kind(self, kind: str) -> List[KnowledgeDoc]:
        return [d for d in self.docs if d.kind == kind]

    def related(self, kind: str, name: str) -> List[str]:
        doc = self.get(kind, name)
        return list(doc.related) if doc else []

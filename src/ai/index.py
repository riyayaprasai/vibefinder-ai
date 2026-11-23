"""A small custom TF-IDF index over two sources: the song catalog and the knowledge base.

Written from scratch (no external libraries) so every step is inspectable:
  * tokenize() lowercases, splits on non-letters and applies a light suffix stemmer.
  * Song documents are *enriched*: a song's text is its title, artist, genre, mood and mood
    tags (repeated so they weigh more) plus the knowledge-base description of its genre and
    mood. That lets a query like "cozy rainy afternoon" reach songs whose own fields never
    use those words.
  * Knowledge documents are indexed by name, aliases, tags and description.
Search returns cosine similarity between the query vector and each document vector.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.ai.knowledge import KnowledgeBase

STOPWORDS = set("""
a an and are as at be but by for from have i im in into is it its me my of on or so some
something that the this to with you your want need like give play find music song songs
track tracks please can could would some any just really very get feel feels feeling
""".split())

FIELD_REPEAT = 3  # how many times structured song fields are repeated in the song document


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s", "y"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if len(word) >= 4 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]  # running -> runn -> run
    return word


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-z&]+", (text or "").lower())
    return [_stem(w) for w in words if w not in STOPWORDS and len(w) > 1]


@dataclass
class Hit:
    doc_id: str
    score: float
    source: str  # "song" or "knowledge"


class Index:
    """TF-IDF vectors with cosine similarity."""

    def __init__(self):
        self.vectors: Dict[str, Dict[str, float]] = {}
        self.sources: Dict[str, str] = {}
        self.idf: Dict[str, float] = {}

    def build(self, documents: Dict[str, str], sources: Dict[str, str]) -> "Index":
        counts = {doc_id: Counter(tokenize(text)) for doc_id, text in documents.items()}
        n = len(counts)
        df = Counter(term for c in counts.values() for term in c)
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
        for doc_id, c in counts.items():
            vec = {t: (1 + math.log(tf)) * self.idf[t] for t, tf in c.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.vectors[doc_id] = {t: v / norm for t, v in vec.items()}
        self.sources = dict(sources)
        return self

    def query_vector(self, text: str) -> Dict[str, float]:
        c = Counter(t for t in tokenize(text) if t in self.idf)
        vec = {t: (1 + math.log(tf)) * self.idf[t] for t, tf in c.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, text: str, source: Optional[str] = None, top_k: int = 5,
               min_score: float = 0.0) -> List[Hit]:
        q = self.query_vector(text)
        if not q:
            return []
        hits = []
        for doc_id, vec in self.vectors.items():
            if source and self.sources[doc_id] != source:
                continue
            score = sum(w * vec.get(t, 0.0) for t, w in q.items())
            if score > min_score:
                hits.append(Hit(doc_id, round(score, 4), self.sources[doc_id]))
        hits.sort(key=lambda h: (-h.score, h.doc_id))
        return hits[:top_k]


def song_document(song: Dict, kb: Optional[KnowledgeBase]) -> str:
    """Builds the enriched text for one song."""
    fields = [song["genre"], song["mood"]] + list(song.get("mood_tags") or [])
    parts = [song["title"], song["artist"]] + fields * FIELD_REPEAT
    if kb is not None:
        for kind, name in (("genre", song["genre"]), ("mood", song["mood"])):
            doc = kb.get(kind, name)
            if doc:
                parts.append(doc.description)
    return " ".join(parts)


def knowledge_document(doc) -> str:
    return " ".join([doc.name] * 2 + doc.aliases + doc.tags + [doc.description])


def build_index(songs: List[Dict], kb: Optional[KnowledgeBase]) -> Index:
    """Indexes every song (as 'song:<id>') and, if given, every knowledge doc."""
    documents, sources = {}, {}
    for song in songs:
        key = f"song:{song['id']}"
        documents[key] = song_document(song, kb)
        sources[key] = "song"
    if kb is not None:
        for doc in kb.docs:
            documents[doc.doc_id] = knowledge_document(doc)
            sources[doc.doc_id] = "knowledge"
    return Index().build(documents, sources)

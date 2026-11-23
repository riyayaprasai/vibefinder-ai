"""Few-shot exemplar specialization (offline, no language model, no training).

The 'specialized' parser level adds a small bank of labeled example requests
(data/exemplars.json, 2-3 per label). For a new request, the bank's own TF-IDF index finds the
nearest exemplars; a genre or mood label is copied only if

  * the exemplar's cosine similarity is at least MIN_SIMILARITY, and
  * the label's summed similarity over the top-K neighbours is at least MIN_VOTE.

It only fills genre / mood fields the rule parser left empty. This is nearest-neighbour
classification over a handful of labeled examples ("few-shot" in the sense of few labeled
examples per class). It is NOT few-shot prompting of an LLM and NOT fine-tuning.

The settings below were fixed before the held-out evaluation was run, and were not tuned on it.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.ai.index import Index

EXEMPLARS_PATH = Path(__file__).resolve().parents[2] / "data" / "exemplars.json"
TOP_K = 3
MIN_SIMILARITY = 0.20
MIN_VOTE = 0.25
LABEL_FIELDS = ("genre", "mood")


@dataclass
class Exemplar:
    id: str
    text: str
    labels: Dict[str, str]


class ExemplarBank:
    def __init__(self, exemplars: List[Exemplar]):
        self.exemplars = exemplars
        self._by_id = {e.id: e for e in exemplars}
        self.index = Index().build({e.id: e.text for e in exemplars}, {e.id: "exemplar" for e in exemplars})

    @classmethod
    def load(cls, path: Path = EXEMPLARS_PATH) -> "ExemplarBank":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))["exemplars"]
        return cls([Exemplar(e["id"], e["text"], {f: e[f] for f in LABEL_FIELDS if f in e}) for e in raw])

    def neighbours(self, text: str) -> List[Tuple[str, float]]:
        return [(h.doc_id, h.score) for h in self.index.search(text, top_k=TOP_K, min_score=MIN_SIMILARITY - 1e-9)]

    def predict(self, text: str) -> Dict[str, Dict]:
        """Returns {field: {"value": label, "vote": score, "support": [exemplar ids]}}."""
        votes: Dict[str, Dict[str, float]] = {f: {} for f in LABEL_FIELDS}
        support: Dict[Tuple[str, str], List[str]] = {}
        for ex_id, score in self.neighbours(text):
            for field, label in self._by_id[ex_id].labels.items():
                votes[field][label] = votes[field].get(label, 0.0) + score
                support.setdefault((field, label), []).append(ex_id)
        out = {}
        for field, tally in votes.items():
            if not tally:
                continue
            label, vote = max(tally.items(), key=lambda kv: (kv[1], kv[0]))
            if vote >= MIN_VOTE:
                out[field] = {"value": label, "vote": round(vote, 3), "support": support[(field, label)]}
        return out


_DEFAULT: Optional[ExemplarBank] = None


def default_bank() -> ExemplarBank:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ExemplarBank.load()
    return _DEFAULT

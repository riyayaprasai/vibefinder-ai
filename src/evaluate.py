"""Evaluation harness for VibeFinder 2.0.

Run from the project folder:
    python -m src.evaluate                  # three configurations, summary table
    python -m src.evaluate --verbose        # also prints every case
    python -m src.evaluate --json eval_results.json
    python -m src.evaluate --llm            # adds rules vs zero-shot vs few-shot LLM parsing (needs Ollama)

Every request in data/eval_cases.json goes through the real MusicAgent three times:
  baseline  - exact catalog labels + generic cues only (no knowledge base, no index)
  lookup    - + knowledge-base aliases, related labels and activity notes
  retrieval - + custom TF-IDF index over songs and knowledge (the full system)
All numbers printed are computed here from actual runs; nothing is typed in by hand.
"""

import argparse
import json
import statistics
from pathlib import Path

from tabulate import tabulate

from src.ai.agent import MusicAgent
from src.ai.knowledge import KnowledgeBase
from src.ai.parser import llm_parse, parse_request
from src.recommender import load_songs

ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "data" / "eval_cases.json"
LEVELS = ["baseline", "lookup", "retrieval"]
K = 5
RELEVANCE_PASS = 0.6


def load_cases(path=CASES_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))["cases"]


def is_relevant(song, case) -> bool:
    genres, moods = case.get("acceptable_genres"), case.get("acceptable_moods")
    label_ok = True
    if genres or moods:
        label_ok = song["genre"] in (genres or []) or song["mood"] in (moods or [])
    lo, hi = case.get("energy_range", [0.0, 1.0])
    return label_ok and lo <= song["energy"] <= hi


def parse_accuracy(parsed_profile, expected) -> float:
    if not expected:
        return None
    hits = 0
    for key, value in expected.items():
        got = parsed_profile.get(key)
        hits += (abs(got - value) <= 0.1) if isinstance(value, float) and isinstance(got, (int, float)) else got == value
    return hits / len(expected)


def evaluate_case(agent: MusicAgent, case) -> dict:
    result = agent.run(case["request"], k=K)
    songs = [r["song"] for r in result.recommendations]
    row = {"id": case["id"], "status": result.status, "status_ok": result.status == case["expect_status"],
           "top": [s["title"] for s in songs], "warnings": result.warnings,
           "relevance": None, "exclusion_ok": None, "warning_ok": None, "parse_acc": None,
           "verified": result.verification["ok"] if result.verification else None}
    has_labels = any(key in case for key in ("acceptable_genres", "acceptable_moods", "energy_range"))
    if result.status == "ok" and has_labels:
        row["relevance"] = round(sum(is_relevant(s, case) for s in songs) / len(songs), 2) if songs else 0.0
    elif case["expect_status"] == "ok" and has_labels:
        row["relevance"] = 0.0  # a request the system failed to answer earns no relevance
    if "must_exclude_genres" in case:
        row["exclusion_ok"] = result.status == "ok" and not any(s["genre"] in case["must_exclude_genres"] for s in songs)
    if "expect_warning" in case:
        row["warning_ok"] = bool(result.warnings) == case["expect_warning"]
    if "expect_parse" in case:
        row["parse_acc"] = parse_accuracy((result.parsed or {}).get("profile", {}), case["expect_parse"])
    checks = [row["status_ok"]]
    if row["relevance"] is not None:
        checks.append(row["relevance"] >= RELEVANCE_PASS)
    for key in ("exclusion_ok", "warning_ok"):
        if row[key] is not None:
            checks.append(row[key])
    if row["verified"] is not None:
        checks.append(row["verified"])
    row["passed"] = all(checks)
    return row


def summarize(rows) -> dict:
    def mean(values):
        values = [v for v in values if v is not None]
        return round(statistics.mean(values), 2) if values else None
    return {
        "cases": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "status_accuracy": mean([float(r["status_ok"]) for r in rows]),
        "mean_relevance@5": mean([r["relevance"] for r in rows]),
        "parse_accuracy": mean([r["parse_acc"] for r in rows]),
        "warnings_correct": f"{sum(bool(r['warning_ok']) for r in rows if r['warning_ok'] is not None)}"
                            f"/{sum(r['warning_ok'] is not None for r in rows)}",
        "exclusions_respected": f"{sum(bool(r['exclusion_ok']) for r in rows if r['exclusion_ok'] is not None)}"
                                f"/{sum(r['exclusion_ok'] is not None for r in rows)}",
        "grounding_verified": f"{sum(bool(r['verified']) for r in rows if r['verified'] is not None)}"
                              f"/{sum(r['verified'] is not None for r in rows)}",
    }


def run_llm_comparison(cases, songs, kb) -> None:
    from src.ai.llm import OllamaClient
    client = OllamaClient()
    print("\nLLM PARSING COMPARISON (rules vs zero-shot vs few-shot)")
    if not client.available():
        print(f"  skipped: Ollama model '{client.model}' is not reachable at {client.host}.")
        return
    genres, moods = sorted({s["genre"] for s in songs}), sorted({s["mood"] for s in songs})
    rows, totals = [], {"rules": [], "zero-shot": [], "few-shot": []}
    for case in [c for c in cases if c.get("expect_parse")]:
        rules = parse_request(case["request"], kb, genres, moods, level="lookup").profile
        accs = {"rules": parse_accuracy(rules, case["expect_parse"])}
        for label, few in (("zero-shot", False), ("few-shot", True)):
            req = llm_parse(case["request"], client, genres, moods, few_shot=few)
            accs[label] = parse_accuracy(req.profile if req else {}, case["expect_parse"])
        for key, value in accs.items():
            totals[key].append(value)
        rows.append([case["id"]] + [f"{accs[key]:.2f}" for key in totals])
    print(tabulate(rows, headers=["case"] + list(totals), tablefmt="simple"))
    print("  mean: " + ", ".join(f"{key}={statistics.mean(v):.2f}" for key, v in totals.items()))
    print(f"  model: {client.model}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate VibeFinder 2.0 on predefined requests")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", help="also write all results to this JSON file")
    ap.add_argument("--llm", action="store_true", help="compare zero-shot and few-shot LLM parsing (Ollama)")
    args = ap.parse_args()

    cases = load_cases()
    songs = load_songs(str(ROOT / "data" / "songs.csv"))
    kb = KnowledgeBase.load()
    all_rows, summaries = {}, {}
    for level in LEVELS:
        agent = MusicAgent(songs=songs, kb=kb, level=level, backend="offline")
        all_rows[level] = [evaluate_case(agent, c) for c in cases]
        summaries[level] = summarize(all_rows[level])

    print(f"VibeFinder 2.0 evaluation: {len(cases)} predefined requests, top-{K}, offline backend\n")
    if args.verbose:
        for level in LEVELS:
            print(f"--- {level}")
            print(tabulate([[r["id"], r["status"], "PASS" if r["passed"] else "fail",
                             "-" if r["relevance"] is None else f"{r['relevance']:.2f}",
                             ", ".join(r["top"][:3])] for r in all_rows[level]],
                           headers=["case", "status", "result", "rel@5", "top 3"], tablefmt="simple"))
            print()

    print("PER-CASE RESULT BY CONFIGURATION")
    print(tabulate([[c["id"]] + ["PASS" if all_rows[lv][i]["passed"] else "fail" for lv in LEVELS]
                    + ["-" if all_rows["retrieval"][i]["relevance"] is None else f"{all_rows['retrieval'][i]['relevance']:.2f}"]
                    for i, c in enumerate(cases)],
                   headers=["case"] + LEVELS + ["rel@5 (full)"], tablefmt="simple"))

    print("\nSUMMARY")
    keys = list(summaries["baseline"].keys())
    print(tabulate([[key] + [summaries[lv][key] for lv in LEVELS] for key in keys],
                   headers=["metric"] + LEVELS, tablefmt="simple"))

    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summaries, "cases": all_rows}, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    if args.llm:
        run_llm_comparison(cases, songs, kb)


if __name__ == "__main__":
    main()

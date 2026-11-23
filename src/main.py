"""Command line runner for the Music Recommender Simulation.

Usage (run from the project folder):
    python -m src.main                              # all four profiles, default ranking
    python -m src.main --profile pop --mode genre   # one profile, genre-first ranking
    python -m src.main --mode mood --k 3            # mood-first ranking, top 3
    python -m src.main --artist-penalty 0           # turn the diversity penalty off

Project 4 (VibeFinder 2.0) - natural-language requests through the AI workflow:
    python -m src.main --ask "chill lofi for studying"
    python -m src.main --ask "sad but high energy classical" --show-trace
    python -m src.main --ask "party music, no rap" --save-trace traces/party.json
    python -m src.main --ask "..." --backend ollama  # optional local LLM (see README)
"""

import argparse
from pathlib import Path
from tabulate import tabulate

from src.recommender import load_songs, score_song, recommend_songs, W_MAX
from src.strategy import STRATEGIES

SONGS_CSV = Path(__file__).resolve().parent.parent / "data" / "songs.csv"
DEFAULT_ARTIST_PENALTY = 0.5


PROFILES = [
    {
        "key": "pop",
        "name": "High-Energy Pop Fan",
        "genre": "pop", "mood": "happy", "energy": 0.85, "valence": 0.80,
        "danceability": 0.80, "likes_acoustic": False, "tempo": 125.0,
        "popularity": 80, "release_decade": 2020, "instrumentalness": 0.02,
        "liveness": 0.15, "mood_tags": ["uplifting", "sunny", "danceable"],
    },
    {
        "key": "lofi",
        "name": "Chill Lofi Listener",
        "genre": "lofi", "mood": "chill", "energy": 0.35, "valence": 0.60,
        "danceability": 0.55, "likes_acoustic": True, "tempo": 75.0,
        "popularity": 55, "release_decade": 2020, "instrumentalness": 0.90,
        "liveness": 0.08, "mood_tags": ["calm", "studious", "cozy"],
    },
    {
        "key": "rock",
        "name": "Intense Rock Enthusiast",
        "genre": "rock", "mood": "intense", "energy": 0.90, "valence": 0.45,
        "danceability": 0.65, "likes_acoustic": False, "tempo": 150.0,
        "popularity": 65, "release_decade": 2010, "instrumentalness": 0.05,
        "liveness": 0.30, "mood_tags": ["aggressive", "driving", "powerful"],
    },
    {
        "key": "adversarial",
        "name": "Adversarial: Sad but High-Energy",
        "genre": "classical", "mood": "melancholy", "energy": 0.90, "valence": 0.30,
        "danceability": 0.25, "likes_acoustic": True, "tempo": 65.0,
        "popularity": 45, "release_decade": 1990, "instrumentalness": 0.95,
        "liveness": 0.20, "mood_tags": ["melancholic", "lonely"],
    },
]


def format_profile(profile: dict) -> str:
    """Return a short summary of a profile's preferences."""
    parts = [
        f"genre={profile['genre']}",
        f"mood={profile['mood']}",
        f"energy={profile['energy']}",
        f"acoustic={'yes' if profile.get('likes_acoustic') else 'no'}",
        f"tempo={profile.get('tempo', 120)}bpm",
    ]
    extras = [f"{name}={profile[name]}" for name in
              ("popularity", "release_decade", "instrumentalness", "liveness") if name in profile]
    if profile.get("mood_tags"):
        extras.append("mood_tags=" + "/".join(profile["mood_tags"]))
    return ", ".join(parts + extras)


def print_results(profile: dict, results: list, mode: str, artist_penalty: float) -> None:
    """Print a formatted table of recommendations for one profile."""
    print("=" * 80)
    print(f"  Profile: {profile['name']}")
    print(f"  Preferences: {format_profile(profile)}")
    print(f"  Ranking mode: {mode} | artist penalty: {artist_penalty}")
    print("=" * 80)

    table_data = []
    for rank, (song, score, explanation) in enumerate(results, 1):
        table_data.append([
            rank,
            song["title"],
            song["artist"],
            f"{score:.2f}/{W_MAX:.2f}",
            explanation.replace("; ", "\n"),
        ])

    print(tabulate(
        table_data,
        headers=["Rank", "Title", "Artist", "Score", "Reasons"],
        tablefmt="grid",
        maxcolwidths=[None, 20, 16, None, 70],
    ))
    print()


def print_agent_result(result, show_trace: bool = False) -> None:
    """Prints the output of the Project 4 workflow."""
    print("=" * 80)
    print(f"  Request: {result.request}")
    print(f"  Status:  {result.status}")
    print("=" * 80)
    if result.status != "ok":
        print(result.message)
    else:
        rows = [[i, r["song"]["title"], r["song"]["artist"], f"{r['song']['genre']} / {r['song']['mood']}",
                 f"{r['score']:.2f}", f"{r['match_strength']:.2f}"]
                for i, r in enumerate(result.recommendations, 1)]
        print(tabulate(rows, headers=["Rank", "Title", "Artist", "Genre / Mood", "Score", "Match"],
                       tablefmt="simple"))
        print("\nExplanation (" + result.generator + "):")
        print(result.explanation)
        verified = "passed" if result.verification["ok"] else "FAILED: " + "; ".join(result.verification["problems"])
        print(f"\nGrounding check: {verified}")
    if show_trace:
        print("\nAgent trace:")
        for step in result.trace["steps"]:
            decision = f"  -> {step['decision']}" if step["decision"] else ""
            print(f"  {step['n']:>2}. {step['action']:<18}{decision}")
    print()


def run_ask(args) -> None:
    from src.ai.agent import MusicAgent

    agent = MusicAgent(level=args.level, backend=args.backend, strategy=args.mode,
                       artist_penalty=args.artist_penalty)
    if args.backend == "ollama" and not agent.llm.available():
        print(f"Note: Ollama model '{agent.llm.model}' is not reachable at {agent.llm.host}; "
              "using the offline parser and generator instead.\n")
    result = agent.run(args.ask, k=args.k, save_trace=args.save_trace)
    print_agent_result(result, show_trace=args.show_trace)
    if args.save_trace:
        print(f"Trace saved to {args.save_trace}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Music Recommender Simulation")
    parser.add_argument("--ask", help="natural-language music request (Project 4 AI workflow)")
    parser.add_argument("--backend", choices=["offline", "ollama"], default="offline",
                        help="explanation/parsing backend for --ask (default: offline)")
    parser.add_argument("--level", choices=["baseline", "lookup", "retrieval"], default="retrieval",
                        help="how much of the knowledge layer --ask uses (default: retrieval)")
    parser.add_argument("--show-trace", action="store_true", help="print the agent's step trace")
    parser.add_argument("--save-trace", help="write the agent's JSON trace to this path")
    parser.add_argument("--mode", choices=sorted(STRATEGIES), default="default",
                        help="ranking strategy (default: default)")
    parser.add_argument("--profile", choices=[p["key"] for p in PROFILES] + ["all"], default="all",
                        help="which listener profile to run (default: all)")
    parser.add_argument("--k", type=int, default=5, help="how many songs to show (default: 5)")
    parser.add_argument("--artist-penalty", type=float, default=DEFAULT_ARTIST_PENALTY,
                        help="points taken off per song by an artist already in the list; 0 turns it off")
    args = parser.parse_args()

    if args.ask is not None:
        run_ask(args)
        return

    songs = load_songs(str(SONGS_CSV))
    print(f"\nLoaded {len(songs)} songs from catalog.\n")

    for profile in PROFILES:
        if args.profile not in ("all", profile["key"]):
            continue
        results = recommend_songs(profile, songs, k=args.k, strategy=args.mode,
                                  artist_penalty=args.artist_penalty)
        print_results(profile, results, args.mode, args.artist_penalty)


if __name__ == "__main__":
    main()
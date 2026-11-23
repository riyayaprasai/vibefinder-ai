# AI Interactions Log

---

## Agentic Workflow (SF8)


**What task did you give the agent?**

<!-- Audit the project against the grading rubric, then fix what was missing. The audit found that data/songs.csv had only 10 songs and 10 columns, that Song already had popularity, instrumentalness and liveness fields which were not in the CSV and not used in scoring, and that python -m src.main crashed on an import. The task for this stretch feature was to make five extra attributes real: in the data, in the scoring and in the explanations. -->

**Prompts used:**

<-- Prompt used:

""Please review the attached project codebase and evaluate it against the course rubric, specifically checking our implementation of the core recommender logic, unit tests, and stretch features (SF8 extra attributes, SF9 diversity penalty, and SF10 Strategy design pattern). First, provide your code review, identify any potential bugs or missing pieces, and confirm that pytest passes cleanly."

Exact earlier prompt used to initiate the workflow and fix structural/test errors:

"Perform a rigorous diagnostic review of the codebase against the course assessment rubric. Systematically resolve any namespace import discrepancies and UTF-8/whitespace CSV ingestion bottlenecks to ensure a clean test suite execution (pytest). Concurrently, expand the corpus and scoring matrix to fully operationalize SF8—integrating continuous latent features and multi-valued metadata into both the data layer and similarity heuristics—while codifying and structuring the SF10 Strategy Design Pattern for clean algorithmic decoupling."

**What did the agent generate or change?**

<!-- data/songs.csv expansion: Expanded the dataset from 10 to 20 songs and added 5 new feature columns (popularity, release_decade, mood_tags, instrumentalness, and liveness), bringing the total column count to 15.

src/strategies.py creation: Implemented the Strategy Design Pattern with an abstract base class (RecommendationStrategy), concrete ranking strategies (GenreFirstStrategy, MoodFirstStrategy, EnergyFocusedStrategy, DefaultStrategy), and a factory function (get_strategy).

src/recommender.py updates: Enhanced load_songs to parse new fields cleanly with utf-8-sig encoding, unified scoring logic so Recommender and score_song use a single source of truth, and updated score rounding so individual itemized reasons sum up accurately to the overall score.

src/main.py updates: Updated profiles to define targets for all 15 attributes and modified the CLI output to render explanations for the new feature contributions.

tests/test_recommender.py updates: Added unit tests verifying that each new attribute alters the calculated score, that mood_tags are evaluated by tag overlap ratio, and that missing values are handled gracefully. -->

**What did you verify or fix manually?**

<!-- Fixed CSV Header Matching & Encoding: Resolved a ValueError during initial pytest runs where DictReader failed to recognize required CSV columns due to leading/trailing whitespace and UTF-8 BOM encoding. Updated load_songs in src/recommender.py to use encoding="utf-8-sig" and normalized header string parsing.

Fixed Module Naming (src/strategies.py): Resolved an ImportError / ModuleNotFoundError during test collection where the test suite expected from src.strategies import ... by renaming the strategy file from strategy.py to strategies.py and exposing get_strategy.

Verified Test Coverage: Ran python3 -m pytest in the terminal to verify all 79 test cases passed cleanly without failure.

Verified Strategy Penalties & Scoring: Checked that the diversity penalty (SF9) correctly re-ranks top candidates inside recommend_songs during list construction rather than simply truncating choices after selection. -->

---

## Design Pattern (SF10)

> Document how AI helped you choose or implement a design pattern.

**Which design pattern did you use?**

    <Strategy Pattern was used -->

**How did AI help you brainstorm or implement it?**

<!-- The pattern was already in the project when the audit started (src/strategy.py). The agent reviewed it, kept it, and added tests for it.
Asked the AI how to structure strategy.py using the Strategy Design Pattern with concrete classes like GenreFirstStrategy, MoodFirstStrategy, and EnergyFocusedStrategy. The AI suggested defining an abstract base class (RecommendationStrategy) with a common interface method (score_songs / rank), subclassing it for each specific ranking behavior, and implementing dependency injection so the recommender and main.py can dynamically swap strategies at runtime via a factory helper (get_strategy).-->

**How does the pattern appear in your final code?**

<!-- RecommendationStrategy (abstract base class in src/strategies.py): Defines the common interface with the abstract method score_songs (or rank), establishing a unified contract that all concrete strategies must follow.

Concrete Strategy Classes (src/strategies.py): GenreFirstStrategy, MoodFirstStrategy, EnergyFocusedStrategy, and DefaultStrategy implement specific sorting and prioritization algorithms, overriding the scoring method to tailor recommendations to specific user preferences.

Factory & Dependency Injection (src/strategies.py & src/recommender.py): The get_strategy(name) helper function maps string arguments to strategy instances, and recommendation functions accept the strategy object dynamically. This decouples the core recommender engine from specific sorting rules.

CLI & Experiment Integration (src/main.py & src/experiments.py): Command-line arguments or experiment scripts pass the selected strategy mode into the recommendation pipeline, allowing users to toggle between ranking behaviors seamlessly without modifying the underlying song-loading or data-parsing logic. -->

---
---

# Project 4: VibeFinder 2.0 AI interactions

## Collaboration summary

- **Tool:** Claude (Claude Code desktop app) with access to the Project 4 folder and terminal.
- **Task given:** Recover the Project 3 context from the repository, not from the old chat. Then
  choose and build a Project 4 extension that meets the AI110 rubric, without changing the
  Project 3 repository or overstating what the AI does.
- **What Claude did, in order:**
  1. Cloned Project 3 into a scratch folder and read every source file, the tests and the docs.
     It ran the test suite: **56 passed**.
  2. Checked the machine for an LLM runtime and API keys. It found no Ollama and no key, so it
     made offline mode the default and Ollama optional.
  3. Created this repository by cloning Project 3 into `vibefinder-ai/` and removing the
     `origin` remote, so nothing can be pushed to the Project 3 repository by accident.
  4. Wrote the knowledge base, index, parser, guardrails, generator, agent, evaluation harness and
     44 new tests. Then it ran the CLI, the tests (**100 passed**) and the evaluation.
- **What was verified by running code, not just by reading it:** every sample in `samples/`,
  every trace in `traces/`, and the evaluation numbers in the README and model card.
- **Not verified:** the Ollama path against a real model. It is covered only by unit tests with a
  fake LLM.

## Agentic workflow traces (stretch B)

`MusicAgent.run()` writes one JSON trace per request. Each step records `action`, `input`,
`output`, `decision` and `elapsed_ms`. These saved traces came from real runs (see
`samples/*.txt` for the matching console output):

| Trace | Request | Decisions it shows |
|---|---|---|
| [traces/01_study_lofi.json](traces/01_study_lofi.json) | "chill lofi beats for studying" | Normal path. `parse_request` shows the provenance of each target: label, mood note or activity note. `verify_output` gives `accept`. |
| [traces/02_conflict_sad_loud.json](traces/02_conflict_sad_loud.json) | "sad but really high energy classical" | `check_feasibility` finds two conflicts (mood+energy, genre+energy). Three `relax_conflict` steps follow: two return the same top song, and dropping the genre offers Night Drive Loop. |
| [traces/03_missing_genre_kpop.json](traces/03_missing_genre_kpop.json) | "give me some k-pop" | The parser marks k-pop as missing from the catalog. Related genres (pop, electronic, hip hop) come from the knowledge base. |
| [traces/04_off_topic.json](traces/04_off_topic.json) | "what's the weather tomorrow?" | `decide` gives `stop: ask a clarifying question`. Retrieval was skipped because the request shows no music intent. |
| [traces/05_conflict_alternative_jazz.json](traces/05_conflict_alternative_jazz.json) | "high energy jazz" | Dropping energy gives the same top song. Dropping the genre gives `offer alternative`, which is Neon Pulse. |

Excerpt from `traces/05_conflict_alternative_jazz.json` (step 9):

```json
{
  "n": 9,
  "action": "relax_conflict",
  "input": {"conflict": ["genre", "energy"], "dropped": "genre"},
  "output": {"alternative": {"dropped": "genre", "song_id": 13, "title": "Neon Pulse",
                             "artist": "Volt Atlas", "score": 3.51}},
  "decision": "offer alternative"
}
```

To regenerate a trace:

```bash
python -m src.main --ask "high energy jazz" --k 3 --show-trace --save-trace traces/05_conflict_alternative_jazz.json
```

## Design decisions made with AI, and what was changed

| Proposal | Outcome | Evidence |
|---|---|---|
| Keep `score_song` unchanged and add an optional `bonus` to `recommend_songs` | Accepted | `test_bonus_none_keeps_project3_results_identical` |
| Let retrieval alone decide what a request means | **Revised.** "tell me a joke" returned country songs, so a music-intent guardrail was added | `test_retrieval_needs_music_intent` |
| Keep partial-credit bonuses during the relaxed search | **Revised.** "calm metal" suggested loud rock, so dependent bonuses are now dropped | `samples/05_conflict_alternative_jazz.txt` |
| Change the `dinner-date` expectation so the eval passes | **Rejected.** The catalog ceiling is documented instead | model_card.md, section 11 |
| Call the offline template "RAG" or "an LLM" | **Rejected.** The README uses honest terminology | README, "Honest terminology" |

## Note on the Project 3 log above

Comparing the Project 3 entries with the repository found two statements that don't match the
code. The log says the strategy file was renamed to `src/strategies.py`, but the code and tests use
`src/strategy.py`. The log says 79 tests passed, but the suite has 56 (and the Project 3 model card
also says 56). The Project 3 text above is left as it was submitted. This note records the
difference.

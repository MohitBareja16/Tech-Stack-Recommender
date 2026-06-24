"""
main.py — Orchestrator / CLI entry point for the Tech Stack Recommender.

Wires together the 4-stage IPO pipeline:
  1. Ingestion  : data_loader.load_items()
  2. Vectorize  : vectorizer (build_vocabulary / compute_idf / build_item_matrix /
                              build_user_vector)
  3. Score      : similarity_engine.score_all_items()
  4. Rank       : ranker.rank() or ranker.apply_cold_start_fallback()
  5. Present    : presenter.format_results()

This module is also the single integration test entry point (ARCHITECTURE.md §3.6).
Input validation (≥3 non-blank skills) lives here — it is a pipeline-entry
concern, not a module-internal one.

Usage:
    python main.py                                  # interactive mode
    python main.py --skills "Python" "SQL" "AWS"   # CLI flag mode
    python main.py --dataset data/raw_skills.csv --top-n 5 --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.data_loader import DataLoadError, load_items, normalize_skill, normalize_text
from src.presenter import format_results
from src.ranker import apply_cold_start_fallback, rank
from src.similarity_engine import ScoredItem, is_cold_start, score_all_items
from src.vectorizer import (
    build_item_matrix,
    build_user_vector,
    build_vocabulary,
    compute_idf,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default dataset path relative to this file's location
_DEFAULT_DATASET = str(Path(__file__).parent / "data" / "raw_skills.csv")


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class InputValidationError(Exception):
    """Raised when user-supplied skill input fails pre-pipeline validation."""


# ---------------------------------------------------------------------------
# Core pipeline function (integration-testable unit)
# ---------------------------------------------------------------------------

def recommend(
    user_skills_raw: list[str],
    dataset_path: str = _DEFAULT_DATASET,
    top_n: int = 3,
    verbose: bool = False,
) -> list[ScoredItem]:
    """Run the full recommendation pipeline and return ranked results.

    This is the single function tested end-to-end in TEST_PLAN.md §8.
    It does NOT print anything — the CLI layer handles that.

    Pipeline steps (per ARCHITECTURE.md §3.6):
      1. Normalize + validate user input (≥3 non-blank skills).
      2. Load item corpus (data_loader).
      3. Build vocabulary, IDF, item matrix (vectorizer).
      4. Build user vector (vectorizer).
      5. Score all items (similarity_engine).
      6. Route: Cold Start fallback OR rank+truncate.

    Args:
        user_skills_raw: Raw user skill strings (before normalization).
        dataset_path:    Path to the CSV dataset file.
        top_n:           Maximum results to return (default 3).
        verbose:         Enable debug logging output.

    Returns:
        Ordered list of ScoredItem (length ≤ top_n).

    Raises:
        InputValidationError: If fewer than 3 valid (non-blank) skills are given.
        DataLoadError:        If the dataset cannot be loaded.
        ValueError:           If top_n ≤ 0.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Step 1: Normalize + validate ──────────────────────────────────────
    normalized_skills = [normalize_skill(s) for s in user_skills_raw if normalize_skill(s)]
    if len(normalized_skills) < 3:
        raise InputValidationError(
            f"Please enter at least 3 non-blank skills. "
            f"You provided {len(normalized_skills)} valid skill(s) after normalization. "
            "Blank or whitespace-only inputs do not count (PRD FR-1)."
        )

    # ── Step 2: Load item corpus ───────────────────────────────────────────
    items = load_items(dataset_path)

    # ── Step 3: Build vocabulary, IDF, item matrix ────────────────────────
    vocab = build_vocabulary(items)
    idf = compute_idf(items, vocab)
    item_matrix = build_item_matrix(items, vocab, idf)

    # ── Step 4: Build user vector ─────────────────────────────────────────
    user_vector, oov_terms = build_user_vector(normalized_skills, vocab, idf)

    if oov_terms:
        logger.info("OOV skills not in dataset vocabulary: %s", oov_terms)

    # ── Step 5: Score all items ───────────────────────────────────────────
    scored = score_all_items(user_vector, item_matrix, items, vocab)

    # ── Step 6: Route (Cold Start OR rank) ───────────────────────────────
    if is_cold_start(scored):
        logger.info("Cold Start detected — all scores are 0. Returning popularity fallback.")
        results = apply_cold_start_fallback(items, top_n)
    else:
        results = rank(scored, top_n)

    # Attach metadata for the presenter (used when verbose=True)
    # Store on the first result for convenience (presenter accesses separately)
    results[0].__dict__["_debug_user_vector"] = user_vector
    results[0].__dict__["_debug_vocab"] = vocab
    results[0].__dict__["_debug_oov"] = oov_terms

    return results


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _interactive_input() -> list[str]:
    """Prompt the user interactively for skills (at least 3 required)."""
    print("\n" + "═" * 60)
    print("  🚀  Tech Stack Recommender — DecodeLabs Project 3")
    print("═" * 60)
    print("  Enter your skills/interests one per line.")
    print("  Type 'done' when finished (minimum 3 skills required).\n")

    skills: list[str] = []
    while True:
        try:
            raw = input(f"  Skill #{len(skills) + 1}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            sys.exit(0)

        if raw.lower() == "done":
            if len(skills) < 3:
                print(f"  ⚠  Need at least 3 skills. You've entered {len(skills)} so far.")
                continue
            break
        if not raw:
            print("  ⚠  Blank input ignored.")
            continue
        skills.append(raw)
        print(f"      ✔  Added: {raw}")

    return skills


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tech_stack_recommender",
        description=(
            "Content-based Tech Stack Recommender.\n"
            "Matches your skills to the most relevant job roles using TF-IDF + Cosine Similarity."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skills", "-s",
        nargs="+",
        metavar="SKILL",
        help="One or more skill/interest keywords (minimum 3 required).",
    )
    parser.add_argument(
        "--dataset", "-d",
        default=_DEFAULT_DATASET,
        metavar="PATH",
        help=f"Path to the CSV dataset file (default: {_DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--top-n", "-n",
        type=int,
        default=3,
        metavar="N",
        help="Number of top results to return (default: 3).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug output (shows TF-IDF vector breakdown).",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Determine skills: CLI flag or interactive
    if args.skills:
        raw_skills = args.skills
    else:
        raw_skills = _interactive_input()

    try:
        results = recommend(
            user_skills_raw=raw_skills,
            dataset_path=args.dataset,
            top_n=args.top_n,
            verbose=args.verbose,
        )
    except InputValidationError as exc:
        print(f"\n  ✘  Input Error: {exc}\n")
        sys.exit(1)
    except DataLoadError as exc:
        print(f"\n  ✘  Dataset Error: {exc}\n")
        sys.exit(2)
    except ValueError as exc:
        print(f"\n  ✘  Configuration Error: {exc}\n")
        sys.exit(3)

    # Retrieve debug metadata if attached
    user_vector = results[0].__dict__.pop("_debug_user_vector", None)
    vocab = results[0].__dict__.pop("_debug_vocab", None)
    oov_terms = results[0].__dict__.pop("_debug_oov", None)

    output = format_results(
        results=results,
        verbose=args.verbose,
        user_vector=user_vector,
        vocab=vocab,
        oov_terms=oov_terms,
    )
    print(output)


if __name__ == "__main__":
    main()

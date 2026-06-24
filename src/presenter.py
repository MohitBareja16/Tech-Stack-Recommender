"""
presenter.py — Output formatting + explainability view.

Responsibility:
  - Render the final recommendation list in a human-readable format.
  - Label Cold Start fallback results clearly (FR-6, US-4 AC1).
  - Provide a verbose/debug view for Reviewer persona verification (FR-7, US-5).

Per ARCHITECTURE.md §3.5.
"""

from __future__ import annotations

import numpy as np

from src.similarity_engine import ScoredItem


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEPARATOR = "─" * 60
_FALLBACK_LABEL = "⚠  TRENDING FALLBACK (Cold Start detected)"


# ---------------------------------------------------------------------------
# Main output formatter
# ---------------------------------------------------------------------------

def format_results(
    results: list[ScoredItem],
    verbose: bool = False,
    user_vector: np.ndarray | None = None,
    vocab: list[str] | None = None,
    oov_terms: list[str] | None = None,
) -> str:
    """Return a human-readable string of ranked recommendations.

    Args:
        results:     Ordered list of ScoredItem from ranker.rank() or
                     ranker.apply_cold_start_fallback().
        verbose:     If True, include a debug section with the user vector
                     dimensions (FR-7 / US-5).
        user_vector: Required when verbose=True. The user's TF-IDF vector.
        vocab:       Required when verbose=True. The vocabulary list.
        oov_terms:   OOV terms to mention if any.

    Returns:
        A formatted multi-line string ready for printing to stdout.
    """
    lines: list[str] = []
    lines.append("")
    lines.append(_SEPARATOR)
    lines.append("  🎯  TECH STACK RECOMMENDER — Results")
    lines.append(_SEPARATOR)

    if not results:
        lines.append("  No results returned.")
        lines.append(_SEPARATOR)
        return "\n".join(lines)

    # If any result is a fallback, show the global fallback notice at the top
    if any(r.is_fallback for r in results):
        lines.append(f"  {_FALLBACK_LABEL}")
        lines.append(
            "  None of your skills matched the dataset vocabulary.\n"
            "  Showing the most popular roles as a default suggestion."
        )
        lines.append(_SEPARATOR)

    for rank_idx, item in enumerate(results, start=1):
        lines.append(f"\n  #{rank_idx}  {item.role_name.title()}")

        if item.is_fallback or item.score is None:
            lines.append("      Score  : — (fallback / no similarity computed)")
            lines.append("      Reason : Trending / popularity-based default")
        else:
            lines.append(f"      Score  : {item.score:.4f}")
            if item.matched_skills:
                matched_str = ", ".join(item.matched_skills)
                lines.append(f"      Matched: {matched_str}")
            else:
                lines.append("      Matched: (no overlapping vocabulary terms)")

    lines.append("")
    lines.append(_SEPARATOR)

    # OOV notice
    if oov_terms:
        lines.append(
            f"  ℹ  Out-of-vocabulary terms (not in dataset, zero weight): "
            f"{', '.join(oov_terms)}"
        )
        lines.append(_SEPARATOR)

    # Debug/verbose section
    if verbose and user_vector is not None and vocab is not None:
        lines.append(format_debug_view(user_vector, vocab, results[0]))
        lines.append(_SEPARATOR)

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Debug / verbose view (FR-7, US-5 AC1)
# ---------------------------------------------------------------------------

def format_debug_view(
    user_vector: np.ndarray,
    vocab: list[str],
    top_match: ScoredItem,
) -> str:
    """Return a debug string showing the user TF-IDF vector and top match info.

    Satisfies FR-7 (explainability) and US-5 AC1: a Reviewer can inspect
    the raw vector dimensions to manually verify the cosine math.

    Args:
        user_vector: The user's TF-IDF weight vector (shape: vocab_size).
        vocab:       Vocabulary list (index → term name).
        top_match:   The top-ranked ScoredItem (for context labeling).

    Returns:
        A multi-line debug string.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("  ── DEBUG / VERBOSE VIEW ──────────────────────────────")
    lines.append(f"  Top match: {top_match.role_name.title()}")
    if top_match.score is not None:
        lines.append(f"  Cosine score: {top_match.score:.6f}")
    lines.append("")
    lines.append("  User TF-IDF Vector (non-zero dimensions only):")

    nonzero_found = False
    for idx, term in enumerate(vocab):
        weight = float(user_vector[idx])
        if weight != 0.0:
            lines.append(f"    [{idx:>3}] {term:<30} weight = {weight:.6f}")
            nonzero_found = True

    if not nonzero_found:
        lines.append("    (all dimensions are zero — Cold Start condition)")

    if top_match.matched_skills:
        lines.append("")
        lines.append(f"  Matched skills driving the score: {', '.join(top_match.matched_skills)}")

    lines.append("  ────────────────────────────────────────────────────")
    return "\n".join(lines)

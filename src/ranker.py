"""
ranker.py — Sorting + Filtering stage (the "Output" part of IPO).

Responsibility:
  - Sort ScoredItem list by descending score, with alphabetical tie-break.
  - Truncate to top_n results.
  - Apply Cold Start popularity fallback when all scores are zero.

Per ARCHITECTURE.md §3.4 and PRD FR-5, FR-6, §10.2.
"""

from __future__ import annotations

from src.data_loader import ItemRecord
from src.similarity_engine import ScoredItem


def rank(scored_items: list[ScoredItem], top_n: int) -> list[ScoredItem]:
    """Sort and truncate *scored_items* to the top *top_n* results.

    Sort key (PRD §10.2, NFR-2):
      Primary:   descending cosine score  → ``-score``
      Secondary: ascending role name      → ``role_name`` (alphabetical)

    The secondary sort guarantees deterministic output when scores tie
    within float tolerance (ARCHITECTURE.md §3.4).

    Args:
        scored_items: All scored items from ``similarity_engine.score_all_items()``.
        top_n:        Maximum number of results to return (must be > 0).

    Returns:
        Sorted list of at most *top_n* ScoredItem objects.

    Raises:
        ValueError: If *top_n* ≤ 0 (PRD Edge Cases table).
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be a positive integer; got {top_n!r}.")

    sorted_items = sorted(
        scored_items,
        key=lambda item: (-(item.score or 0.0), item.role_name),
    )
    return sorted_items[:top_n]


def apply_cold_start_fallback(
    items: list[ItemRecord],
    top_n: int,
) -> list[ScoredItem]:
    """Return a popularity-ordered fallback list when Cold Start is detected.

    Called only when ``similarity_engine.is_cold_start()`` returns True (FR-6).
    Items are ordered by ``popularity_rank`` ascending (rank 1 = most popular).
    Each returned ScoredItem is tagged ``is_fallback=True`` and ``score=None``
    so the presenter can label the output clearly (US-4 AC1).

    Args:
        items:  Full item corpus (list of ItemRecord).
        top_n:  Maximum number of fallback results to return.

    Returns:
        List of ScoredItem ordered by popularity_rank, length ≤ top_n.

    Raises:
        ValueError: If *top_n* ≤ 0.
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be a positive integer; got {top_n!r}.")

    sorted_items = sorted(items, key=lambda item: item.popularity_rank)
    fallback: list[ScoredItem] = []
    for item in sorted_items[:top_n]:
        fallback.append(ScoredItem(
            role_name=item.role_name,
            score=None,
            matched_skills=[],
            popularity_rank=item.popularity_rank,
            is_fallback=True,
        ))
    return fallback

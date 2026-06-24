"""
similarity_engine.py — Scoring stage of the IPO pipeline (Process, part 2).

Responsibility:
  - Compute Cosine Similarity between two vectors (from scratch — no sklearn).
  - Score the user vector against every item in the corpus.
  - Detect the Cold Start condition (all-zero scores).

Math reference: ALGORITHM_SPEC.md §2.4, §3.9, §5.
Architecture reference: ARCHITECTURE.md §3.3, §6 (vectorized implementation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.data_loader import ItemRecord


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ScoredItem:
    """A job role together with its similarity score against the user vector.

    Attributes:
        role_name:       Human-readable job title.
        score:           Cosine similarity in [0, 1], or None for fallback rows.
        matched_skills:  Vocabulary terms with non-zero weight in BOTH the user
                         vector and this item's vector (explainability — FR-7).
        popularity_rank: Passthrough from ItemRecord; used for fallback ordering.
        is_fallback:     True only when this result comes from the Cold Start
                         popularity fallback path (FR-6 / US-4).
    """
    role_name: str
    score: float | None
    matched_skills: list[str] = field(default_factory=list)
    popularity_rank: int = 0
    is_fallback: bool = False


# ---------------------------------------------------------------------------
# Core similarity function (from scratch)
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cosine Similarity between vectors *a* and *b*.

    Formula (ALGORITHM_SPEC.md §2.4):
        cos(θ) = (a · b) / (||a|| * ||b||)

    Zero-vector guard: returns 0.0 if either norm is zero, implementing the
    Cold Start math from the deck ("vector of zeros → zero score for every
    item") without raising a ZeroDivisionError.

    Args:
        a: NumPy array of float64.
        b: NumPy array of float64, same shape as *a*.

    Returns:
        Float in [0.0, 1.0].  TF-IDF weights are non-negative so the result
        is always ≥ 0 (PRD FR-4).
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Bulk scoring (vectorized)
# ---------------------------------------------------------------------------

def score_all_items(
    user_vector: np.ndarray,
    item_matrix: np.ndarray,
    item_records: list[ItemRecord],
    vocab: list[str],
) -> list[ScoredItem]:
    """Score the user vector against every row in *item_matrix*.

    Uses a single vectorized matrix multiply for performance (ARCHITECTURE.md §6),
    satisfying NFR-3 (10k items in < 2s) without a Python-level loop over items.

    Matched skills are computed per-item after the bulk score — they identify
    vocabulary terms whose TF-IDF weight is non-zero in BOTH the user and item
    vectors (explainability requirement — FR-7 / US-3 AC1).

    Args:
        user_vector:  Shape ``(vocab_size,)``.
        item_matrix:  Shape ``(num_items, vocab_size)``.
        item_records: Parallel list of ItemRecord objects.
        vocab:        Sorted vocabulary list (maps index → term string).

    Returns:
        List of ScoredItem, one per item, in the same order as *item_records*.
        No filtering or sorting — full scan, per PRD FR-4.
    """
    user_norm = float(np.linalg.norm(user_vector))
    item_norms = np.linalg.norm(item_matrix, axis=1)  # shape (num_items,)

    # Vectorized dot products: shape (num_items,)
    dots = item_matrix @ user_vector

    # Compute cosine scores — safe divide: only divide where denom > 0,
    # avoiding RuntimeWarning from numpy evaluating dots/denom at zero positions.
    denom = item_norms * user_norm
    safe_mask = denom > 0.0
    scores = np.zeros(len(item_records), dtype=np.float64)
    scores[safe_mask] = dots[safe_mask] / denom[safe_mask]

    # Build ScoredItem list with matched_skills per item
    vocab_arr = np.array(vocab)
    user_nonzero = user_vector > 0  # boolean mask: user has weight here

    results: list[ScoredItem] = []
    for i, record in enumerate(item_records):
        item_row = item_matrix[i]
        item_nonzero = item_row > 0
        # Matched = terms non-zero in BOTH vectors
        overlap_mask = user_nonzero & item_nonzero
        matched = vocab_arr[overlap_mask].tolist()

        results.append(ScoredItem(
            role_name=record.role_name,
            score=float(scores[i]),
            matched_skills=matched,
            popularity_rank=record.popularity_rank,
            is_fallback=False,
        ))

    return results


# ---------------------------------------------------------------------------
# Cold Start detection
# ---------------------------------------------------------------------------

def is_cold_start(scored_items: list[ScoredItem]) -> bool:
    """Return True if every item has a score of exactly 0.0 (ARCHITECTURE.md §3.3).

    This is the direct programmatic translation of the deck's Cold Start slide:
    a user vector of zeros produces zero similarity against every item.
    The ``similarity_engine.is_cold_start()`` result gates the routing decision
    in ``ranker.py``.

    Args:
        scored_items: Output of ``score_all_items()``.

    Returns:
        True if all scores are 0.0; False if at least one score > 0.
    """
    return all(item.score == 0.0 for item in scored_items)

"""
vectorizer.py — Vector Mapping + TF-IDF (the "Process" stage, part 1).

Responsibility:
  - Build a shared vocabulary from the item corpus (corpus-fixed, per PRD §10.4).
  - Compute Term Frequency (TF) for any token list.
  - Compute Inverse Document Frequency (IDF) across the corpus.
  - Build the item TF-IDF matrix (shape: num_items × vocab_size).
  - Build the user TF-IDF vector and report OOV terms.

Math reference: ALGORITHM_SPEC.md §2 — all formulas implemented from scratch;
no sklearn TfidfVectorizer is used (ARCHITECTURE.md Principle #2).

NumPy is used for array storage and arithmetic only.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from src.data_loader import ItemRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vocabulary construction
# ---------------------------------------------------------------------------

def build_vocabulary(items: list[ItemRecord]) -> list[str]:
    """Return a sorted list of unique skill tokens found across all items.

    The vocabulary is derived **from the item corpus only** (PRD §10.4).
    User-input skills that are absent here are OOV and never added.

    Args:
        items: Non-empty list of ItemRecord objects (already normalized).

    Returns:
        Alphabetically sorted list of unique skill token strings.
        Sorted order ensures reproducible vector indices (NFR-2).
    """
    token_set: set[str] = set()
    for item in items:
        token_set.update(item.skills)
    return sorted(token_set)


# ---------------------------------------------------------------------------
# TF computation
# ---------------------------------------------------------------------------

def compute_tf(tokens: list[str], vocab: list[str]) -> np.ndarray:
    """Compute a Term Frequency vector for *tokens* over *vocab*.

    Formula (ALGORITHM_SPEC.md §2.1):
        TF(t, d) = count(t in d) / total_terms(d)

    The denominator is ``len(tokens)`` — the raw input length INCLUDING any
    OOV tokens (per ALGORITHM_SPEC.md §3.7 and PRD §10.1 rationale).

    Args:
        tokens: List of (already-normalized) token strings.  May include
                tokens not present in *vocab* (OOV); they do not contribute
                to any dimension but count toward the denominator.
        vocab:  The fixed corpus vocabulary (list of strings, sorted).

    Returns:
        NumPy array of shape ``(len(vocab),)`` with float64 values in [0, 1].
        Returns a zero vector if *tokens* is empty.
    """
    tf = np.zeros(len(vocab), dtype=np.float64)
    if not tokens:
        return tf

    total = len(tokens)
    # Build a fast token→count lookup
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1

    for idx, term in enumerate(vocab):
        tf[idx] = counts.get(term, 0) / total

    return tf


# ---------------------------------------------------------------------------
# IDF computation
# ---------------------------------------------------------------------------

def compute_idf(items: list[ItemRecord], vocab: list[str]) -> np.ndarray:
    """Compute the Inverse Document Frequency vector over the corpus.

    Formula (ALGORITHM_SPEC.md §2.2):
        IDF(t) = log( N / df(t) )

    where N = number of items, df(t) = number of items containing term t
    at least once.

    Because the vocabulary is built FROM the corpus, every term has df ≥ 1,
    so log(N/0) cannot occur (no smoothing required — see ALGORITHM_SPEC.md
    §2.2 implementation note).

    Args:
        items: Full item corpus (list of ItemRecord).
        vocab: Sorted vocabulary list (from ``build_vocabulary``).

    Returns:
        NumPy array of shape ``(len(vocab),)`` with float64 IDF values.
    """
    n = len(items)
    idf = np.zeros(len(vocab), dtype=np.float64)

    # Precompute document frequency for each vocab term
    for idx, term in enumerate(vocab):
        df = sum(1 for item in items if term in item.skills)
        # df ≥ 1 guaranteed by vocabulary construction — guard anyway
        if df > 0:
            idf[idx] = math.log(n / df)
        else:
            idf[idx] = 0.0  # Defensive: term appeared in vocab but no item? → 0 weight

    return idf


# ---------------------------------------------------------------------------
# Item matrix construction
# ---------------------------------------------------------------------------

def build_item_matrix(
    items: list[ItemRecord],
    vocab: list[str],
    idf: np.ndarray,
) -> np.ndarray:
    """Build the TF-IDF item matrix.

    Each row is a TF-IDF vector for one item:
        row[i] = compute_tf(items[i].skills, vocab) * idf

    Args:
        items: Full item corpus.
        vocab: Sorted vocabulary list.
        idf:   IDF vector of shape ``(len(vocab),)``.

    Returns:
        NumPy array of shape ``(len(items), len(vocab))`` with float64 values.
    """
    matrix = np.zeros((len(items), len(vocab)), dtype=np.float64)
    for i, item in enumerate(items):
        tf = compute_tf(item.skills, vocab)
        matrix[i] = tf * idf
    return matrix


# ---------------------------------------------------------------------------
# User vector construction
# ---------------------------------------------------------------------------

def build_user_vector(
    user_skills: list[str],
    vocab: list[str],
    idf: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build the TF-IDF vector for the user's skill input.

    The user's input is treated as a document for TF purposes (ALGORITHM_SPEC.md
    §3.7).  OOV terms are logged and reported but never added to *vocab*
    (PRD §10.4 / ARCHITECTURE.md §3.2 critical invariant).

    Args:
        user_skills: Normalized user input tokens (may include OOV terms).
        vocab:       Fixed corpus vocabulary (NOT mutated).
        idf:         IDF vector of shape ``(len(vocab),)``.

    Returns:
        Tuple of:
          - user_vector: NumPy array shape ``(len(vocab),)`` — TF-IDF weights.
          - oov_terms:   List of user input tokens not found in *vocab*.
    """
    vocab_set = set(vocab)
    oov_terms = [t for t in user_skills if t not in vocab_set]

    if oov_terms:
        logger.info(
            "Out-of-vocabulary user skills (excluded from vector): %s",
            oov_terms,
        )

    tf = compute_tf(user_skills, vocab)
    user_vector = tf * idf
    return user_vector, oov_terms

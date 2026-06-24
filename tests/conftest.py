"""
conftest.py — Shared pytest fixtures for the Tech Stack Recommender test suite.

Fixtures defined here are available to all test modules without importing.
References: TEST_PLAN.md §2 (Mini Corpus fixture).
"""

import csv
import os
import tempfile

import numpy as np
import pytest

from src.data_loader import ItemRecord, normalize_skill
from src.similarity_engine import ScoredItem


# ---------------------------------------------------------------------------
# Mini corpus fixture (ALGORITHM_SPEC.md §3.1 canonical 3-item corpus)
# ---------------------------------------------------------------------------

@pytest.fixture
def mini_items() -> list[ItemRecord]:
    """Three-item corpus matching ALGORITHM_SPEC.md §3.1."""
    return [
        ItemRecord(role_name="data scientist",    skills=["python", "sql", "machine_learning"],    popularity_rank=1),
        ItemRecord(role_name="devops engineer",   skills=["aws", "docker", "kubernetes"],           popularity_rank=2),
        ItemRecord(role_name="backend developer", skills=["java", "python", "sql"],                 popularity_rank=3),
    ]


@pytest.fixture
def mini_vocab() -> list[str]:
    """Canonical 7-term vocabulary from ALGORITHM_SPEC.md §3.2."""
    return ["aws", "docker", "java", "kubernetes", "machine_learning", "python", "sql"]


@pytest.fixture
def mini_idf(mini_items, mini_vocab) -> np.ndarray:
    """Pre-computed IDF for the mini corpus. Values from ALGORITHM_SPEC.md §3.4."""
    from src.vectorizer import compute_idf
    return compute_idf(mini_items, mini_vocab)


@pytest.fixture
def mini_item_matrix(mini_items, mini_vocab, mini_idf) -> np.ndarray:
    """Pre-computed item TF-IDF matrix for the mini corpus."""
    from src.vectorizer import build_item_matrix
    return build_item_matrix(mini_items, mini_vocab, mini_idf)


@pytest.fixture
def user_skills_canonical() -> list[str]:
    """User input from ALGORITHM_SPEC.md §3.7 (pre-normalization)."""
    return ["Python", "Machine Learning", "Statistics"]


@pytest.fixture
def user_vector_canonical(mini_vocab, mini_idf, user_skills_canonical) -> tuple[np.ndarray, list[str]]:
    """User TF-IDF vector for the canonical user input."""
    from src.vectorizer import build_user_vector
    normalized = [normalize_skill(s) for s in user_skills_canonical]
    return build_user_vector(normalized, mini_vocab, mini_idf)


# ---------------------------------------------------------------------------
# Temporary CSV helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_csv_path(tmp_path):
    """Factory fixture: returns a function that writes a CSV and returns its path."""

    def _make_csv(rows: list[dict], header=None) -> str:
        if header is None:
            header = ["role_name", "skills", "popularity_rank"]
        path = tmp_path / "test_dataset.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        return str(path)

    return _make_csv


@pytest.fixture
def sample_dataset_path() -> str:
    """Path to the real sample dataset (data/raw_skills.csv)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "data", "raw_skills.csv")


# ---------------------------------------------------------------------------
# Scored item helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def scored_items_normal() -> list[ScoredItem]:
    """Sample scored items with distinct scores (not cold start)."""
    return [
        ScoredItem(role_name="data scientist",    score=0.9448, popularity_rank=1),
        ScoredItem(role_name="backend developer", score=0.1135, popularity_rank=3),
        ScoredItem(role_name="devops engineer",   score=0.0,    popularity_rank=2),
    ]


@pytest.fixture
def scored_items_cold_start() -> list[ScoredItem]:
    """Sample scored items where every score is 0.0 (cold start condition)."""
    return [
        ScoredItem(role_name="data scientist",    score=0.0, popularity_rank=1),
        ScoredItem(role_name="devops engineer",   score=0.0, popularity_rank=2),
        ScoredItem(role_name="backend developer", score=0.0, popularity_rank=3),
    ]

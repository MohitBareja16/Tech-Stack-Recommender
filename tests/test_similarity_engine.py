"""
test_similarity_engine.py — Unit tests for src/similarity_engine.py.
Coverage: TC-SIM-01 through TC-SIM-07 (TEST_PLAN.md §5).

All expected values from ALGORITHM_SPEC.md §3.9 and §3.10.
"""

import numpy as np
import pytest

from src.similarity_engine import ScoredItem, cosine_similarity, is_cold_start, score_all_items

FLOAT_TOL = 1e-4


# ---------------------------------------------------------------------------
# TC-SIM-01 to TC-SIM-03: cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_canonical_value_data_scientist_vs_user(self):
        # ALGORITHM_SPEC.md §3.9 — spec uses rounded intermediates (0.3662, 0.1352)
        # so the hand-computed value (0.9448) differs slightly from full-precision.
        # Using rel=1e-3 (0.1%) to accommodate that without masking real bugs.
        A = np.array([0, 0, 0, 0, 0.3662, 0.1352, 0.1352], dtype=np.float64)
        B = np.array([0, 0, 0, 0, 0.3662, 0.1352, 0],       dtype=np.float64)
        result = cosine_similarity(A, B)
        assert result == pytest.approx(0.9448, rel=1e-3)

    def test_zero_vector_a_returns_zero(self):
        # TC-SIM-02: Cold Start guard — zero user vector
        A = np.zeros(7, dtype=np.float64)
        B = np.array([0.3662, 0.1352, 0.1352, 0, 0, 0, 0], dtype=np.float64)
        assert cosine_similarity(A, B) == 0.0

    def test_zero_vector_b_returns_zero(self):
        A = np.array([0.3662, 0.1352, 0.1352, 0, 0, 0, 0], dtype=np.float64)
        B = np.zeros(7, dtype=np.float64)
        assert cosine_similarity(A, B) == 0.0

    def test_both_zero_vectors_returns_zero(self):
        # TC-SIM-03
        assert cosine_similarity(np.zeros(3), np.zeros(3)) == 0.0

    def test_identical_vectors_returns_one(self):
        v = np.array([0.5, 0.3, 0.2], dtype=np.float64)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=FLOAT_TOL)

    def test_orthogonal_vectors_returns_zero(self):
        # Completely non-overlapping → dot product = 0
        A = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        B = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        assert cosine_similarity(A, B) == pytest.approx(0.0, abs=FLOAT_TOL)

    def test_returns_float(self):
        v = np.array([1.0, 0.5], dtype=np.float64)
        result = cosine_similarity(v, v)
        assert isinstance(result, float)

    def test_result_in_zero_one_range(self):
        # TF-IDF weights are non-negative → cosine in [0,1]
        A = np.random.rand(10).astype(np.float64)
        B = np.random.rand(10).astype(np.float64)
        result = cosine_similarity(A, B)
        assert 0.0 <= result <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# TC-SIM-04 to TC-SIM-05: score_all_items
# ---------------------------------------------------------------------------

class TestScoreAllItems:
    def test_data_scientist_score(self, mini_items, mini_vocab, mini_item_matrix, user_vector_canonical):
        user_vec, _ = user_vector_canonical
        scored = score_all_items(user_vec, mini_item_matrix, mini_items, mini_vocab)
        # Index 0 = Data Scientist
        # ALGORITHM_SPEC.md §3.9 — hand-computed with rounded intermediates (TF=0.3333,
        # IDF=0.4055 etc.), so full-precision result (0.94496) differs by ~0.00016.
        # rel=1e-3 (0.1%) is still tight enough to catch real scoring bugs.
        assert scored[0].score == pytest.approx(0.9448, rel=1e-3)

    def test_devops_engineer_score_is_zero(self, mini_items, mini_vocab, mini_item_matrix, user_vector_canonical):
        # No overlap between user and DevOps vector → score = 0
        user_vec, _ = user_vector_canonical
        scored = score_all_items(user_vec, mini_item_matrix, mini_items, mini_vocab)
        devops = next(s for s in scored if s.role_name == "devops engineer")
        assert devops.score == pytest.approx(0.0, abs=FLOAT_TOL)

    def test_backend_developer_score(self, mini_items, mini_vocab, mini_item_matrix, user_vector_canonical):
        # From ALGORITHM_SPEC.md §3.10
        user_vec, _ = user_vector_canonical
        scored = score_all_items(user_vec, mini_item_matrix, mini_items, mini_vocab)
        backend = next(s for s in scored if s.role_name == "backend developer")
        # ALGORITHM_SPEC.md §3.10: spec shows 0.1135 (rounded), full-precision = 0.11328.
        # Diff = 0.00022. rel=1e-3 only gives ±1.1e-4; use abs=3e-4 to cover the gap.
        assert backend.score == pytest.approx(0.1135, abs=3e-4)

    def test_returns_all_items(self, mini_items, mini_vocab, mini_item_matrix, user_vector_canonical):
        user_vec, _ = user_vector_canonical
        scored = score_all_items(user_vec, mini_item_matrix, mini_items, mini_vocab)
        assert len(scored) == 3  # full scan, no filtering

    def test_matched_skills_correct(self, mini_items, mini_vocab, mini_item_matrix, user_vector_canonical):
        # TC-SIM-05: matched skills for Data Scientist should include python and machine_learning
        user_vec, _ = user_vector_canonical
        scored = score_all_items(user_vec, mini_item_matrix, mini_items, mini_vocab)
        ds = next(s for s in scored if s.role_name == "data scientist")
        assert "python" in ds.matched_skills
        assert "machine_learning" in ds.matched_skills

    def test_matched_skills_excludes_zero_weight_terms(self, mini_items, mini_vocab, mini_item_matrix, user_vector_canonical):
        user_vec, _ = user_vector_canonical
        scored = score_all_items(user_vec, mini_item_matrix, mini_items, mini_vocab)
        ds = next(s for s in scored if s.role_name == "data scientist")
        # "sql" has zero weight in user vector → should NOT be in matched_skills
        assert "sql" not in ds.matched_skills

    def test_returns_scored_item_instances(self, mini_items, mini_vocab, mini_item_matrix, user_vector_canonical):
        user_vec, _ = user_vector_canonical
        scored = score_all_items(user_vec, mini_item_matrix, mini_items, mini_vocab)
        assert all(isinstance(s, ScoredItem) for s in scored)

    def test_is_fallback_false_by_default(self, mini_items, mini_vocab, mini_item_matrix, user_vector_canonical):
        user_vec, _ = user_vector_canonical
        scored = score_all_items(user_vec, mini_item_matrix, mini_items, mini_vocab)
        assert all(not s.is_fallback for s in scored)


# ---------------------------------------------------------------------------
# TC-SIM-06 to TC-SIM-07: is_cold_start
# ---------------------------------------------------------------------------

class TestIsColdStart:
    def test_all_zeros_is_cold_start(self, scored_items_cold_start):
        assert is_cold_start(scored_items_cold_start) is True

    def test_any_nonzero_is_not_cold_start(self, scored_items_normal):
        assert is_cold_start(scored_items_normal) is False

    def test_single_nonzero_is_not_cold_start(self):
        items = [
            ScoredItem(role_name="Role A", score=0.0),
            ScoredItem(role_name="Role B", score=0.0),
            ScoredItem(role_name="Role C", score=0.001),  # tiny but nonzero
        ]
        assert is_cold_start(items) is False

    def test_empty_list_is_cold_start(self):
        # Edge: no items → vacuously all-zero → True (safe default)
        assert is_cold_start([]) is True

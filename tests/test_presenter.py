"""
test_presenter.py — Unit tests for src/presenter.py.
Coverage: TC-PRES-01 through TC-PRES-03 (TEST_PLAN.md §7).
"""

import numpy as np
import pytest

from src.presenter import format_debug_view, format_results
from src.similarity_engine import ScoredItem


# ---------------------------------------------------------------------------
# TC-PRES-01: format_results — basic output
# ---------------------------------------------------------------------------

class TestFormatResults:
    def test_contains_role_name(self):
        items = [ScoredItem(role_name="data scientist", score=0.9448)]
        output = format_results(items)
        assert "data scientist" in output.lower()

    def test_contains_score(self):
        items = [ScoredItem(role_name="data scientist", score=0.9448)]
        output = format_results(items)
        # Score rounded to 4 dp → "0.9448" appears
        assert "0.9448" in output or "0.94" in output

    def test_multiple_items_appear(self):
        items = [
            ScoredItem(role_name="data scientist",    score=0.9448),
            ScoredItem(role_name="backend developer", score=0.1135),
        ]
        output = format_results(items)
        assert "data scientist" in output.lower()
        assert "backend developer" in output.lower()

    def test_matched_skills_appear(self):
        items = [ScoredItem(role_name="data scientist", score=0.9448,
                            matched_skills=["python", "machine_learning"])]
        output = format_results(items)
        assert "python" in output
        assert "machine_learning" in output

    def test_empty_results_no_crash(self):
        output = format_results([])
        assert isinstance(output, str)
        assert "No results" in output

    def test_returns_string(self):
        items = [ScoredItem(role_name="data scientist", score=0.9448)]
        assert isinstance(format_results(items), str)

    def test_rank_numbers_appear(self):
        items = [
            ScoredItem(role_name="data scientist",    score=0.9448),
            ScoredItem(role_name="backend developer", score=0.1135),
        ]
        output = format_results(items)
        assert "#1" in output
        assert "#2" in output


# ---------------------------------------------------------------------------
# TC-PRES-02: format_results — Cold Start / fallback labeling
# ---------------------------------------------------------------------------

class TestFormatResultsFallback:
    def test_fallback_label_appears(self):
        items = [ScoredItem(role_name="data scientist", score=None, is_fallback=True, popularity_rank=1)]
        output = format_results(items)
        assert "fallback" in output.lower()

    def test_no_fallback_label_for_normal_results(self):
        items = [ScoredItem(role_name="data scientist", score=0.9448, is_fallback=False)]
        output = format_results(items)
        assert "fallback" not in output.lower()

    def test_fallback_score_shown_as_dash_or_none(self):
        items = [ScoredItem(role_name="data scientist", score=None, is_fallback=True)]
        output = format_results(items)
        # Should not show a numeric score for a fallback item
        assert "0.0000" not in output


# ---------------------------------------------------------------------------
# TC-PRES-03: format_debug_view
# ---------------------------------------------------------------------------

class TestFormatDebugView:
    def test_contains_role_name(self, mini_vocab, mini_idf, user_vector_canonical):
        user_vec, _ = user_vector_canonical
        top_match = ScoredItem(role_name="data scientist", score=0.9448,
                               matched_skills=["python", "machine_learning"])
        output = format_debug_view(user_vec, mini_vocab, top_match)
        assert "data scientist" in output.lower()

    def test_contains_score(self, mini_vocab, user_vector_canonical):
        user_vec, _ = user_vector_canonical
        top_match = ScoredItem(role_name="data scientist", score=0.9448)
        output = format_debug_view(user_vec, mini_vocab, top_match)
        assert "0.9448" in output or "0.94" in output

    def test_contains_nonzero_vector_dimension(self, mini_vocab, user_vector_canonical):
        user_vec, _ = user_vector_canonical
        top_match = ScoredItem(role_name="data scientist", score=0.9448)
        output = format_debug_view(user_vec, mini_vocab, top_match)
        # python and machine_learning should appear as non-zero dims
        assert "python" in output or "machine_learning" in output

    def test_zero_vector_shows_cold_start_message(self, mini_vocab, mini_idf):
        zero_vec = np.zeros(len(mini_vocab))
        top_match = ScoredItem(role_name="data scientist", score=None, is_fallback=True)
        output = format_debug_view(zero_vec, mini_vocab, top_match)
        assert "zero" in output.lower()

    def test_returns_string(self, mini_vocab, user_vector_canonical):
        user_vec, _ = user_vector_canonical
        top_match = ScoredItem(role_name="data scientist", score=0.9448)
        result = format_debug_view(user_vec, mini_vocab, top_match)
        assert isinstance(result, str)

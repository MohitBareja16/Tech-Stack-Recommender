"""
test_ranker.py — Unit tests for src/ranker.py.
Coverage: TC-RANK-01 through TC-RANK-06 (TEST_PLAN.md §6).
"""

import pytest

from src.data_loader import ItemRecord
from src.ranker import apply_cold_start_fallback, rank
from src.similarity_engine import ScoredItem


# ---------------------------------------------------------------------------
# TC-RANK-01: Descending sort by score
# ---------------------------------------------------------------------------

class TestRankSorting:
    def test_descending_score_order(self):
        items = [
            ScoredItem(role_name="c role", score=0.3),
            ScoredItem(role_name="a role", score=0.9),
            ScoredItem(role_name="b role", score=0.6),
        ]
        result = rank(items, top_n=3)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_canonical_order_data_scientist_first(self, scored_items_normal):
        result = rank(scored_items_normal, top_n=3)
        assert result[0].role_name == "data scientist"

    def test_backend_developer_second(self, scored_items_normal):
        result = rank(scored_items_normal, top_n=3)
        assert result[1].role_name == "backend developer"

    def test_devops_engineer_last(self, scored_items_normal):
        result = rank(scored_items_normal, top_n=3)
        assert result[2].role_name == "devops engineer"


# ---------------------------------------------------------------------------
# TC-RANK-02: Tie-breaking by role name (alphabetical ascending)
# ---------------------------------------------------------------------------

class TestRankTieBreaking:
    def test_alphabetical_tie_break(self):
        items = [
            ScoredItem(role_name="zebra role",  score=0.5),
            ScoredItem(role_name="alpha role",  score=0.5),
            ScoredItem(role_name="middle role", score=0.5),
        ]
        result = rank(items, top_n=3)
        names = [r.role_name for r in result]
        assert names == sorted(names)

    def test_alpha_before_zebra(self):
        items = [
            ScoredItem(role_name="zebra role", score=0.5),
            ScoredItem(role_name="alpha role", score=0.5),
        ]
        result = rank(items, top_n=2)
        assert result[0].role_name == "alpha role"
        assert result[1].role_name == "zebra role"

    def test_score_still_primary_sort_key(self):
        items = [
            ScoredItem(role_name="alpha role", score=0.1),   # alpha but low score
            ScoredItem(role_name="zebra role", score=0.9),   # zebra but high score
        ]
        result = rank(items, top_n=2)
        assert result[0].role_name == "zebra role"  # score wins over alpha order


# ---------------------------------------------------------------------------
# TC-RANK-03: top_n truncation
# ---------------------------------------------------------------------------

class TestRankTruncation:
    def test_returns_exactly_top_n(self, scored_items_normal):
        result = rank(scored_items_normal, top_n=2)
        assert len(result) == 2

    def test_returns_top_n_equals_one(self, scored_items_normal):
        result = rank(scored_items_normal, top_n=1)
        assert len(result) == 1
        assert result[0].role_name == "data scientist"


# ---------------------------------------------------------------------------
# TC-RANK-04: Fewer items than top_n
# ---------------------------------------------------------------------------

class TestRankFewerItemsThanTopN:
    def test_returns_all_available(self, scored_items_normal):
        # 3 items, top_n=5 → return all 3
        result = rank(scored_items_normal, top_n=5)
        assert len(result) == 3

    def test_no_error_on_top_n_larger_than_dataset(self, scored_items_normal):
        result = rank(scored_items_normal, top_n=100)
        assert len(result) == len(scored_items_normal)


# ---------------------------------------------------------------------------
# TC-RANK-05: top_n ≤ 0 raises ValueError
# ---------------------------------------------------------------------------

class TestRankInvalidTopN:
    def test_top_n_zero_raises(self, scored_items_normal):
        with pytest.raises(ValueError):
            rank(scored_items_normal, top_n=0)

    def test_top_n_negative_raises(self, scored_items_normal):
        with pytest.raises(ValueError):
            rank(scored_items_normal, top_n=-1)

    def test_top_n_minus_one_raises(self, scored_items_normal):
        with pytest.raises(ValueError, match="positive integer"):
            rank(scored_items_normal, top_n=-1)


# ---------------------------------------------------------------------------
# TC-RANK-06: apply_cold_start_fallback
# ---------------------------------------------------------------------------

class TestApplyColdStartFallback:
    def test_ordered_by_popularity_rank(self, mini_items):
        result = apply_cold_start_fallback(mini_items, top_n=3)
        ranks = [r.popularity_rank for r in result]
        assert ranks == sorted(ranks)

    def test_first_item_is_most_popular(self, mini_items):
        result = apply_cold_start_fallback(mini_items, top_n=3)
        assert result[0].role_name == "data scientist"  # popularity_rank=1

    def test_all_is_fallback_true(self, mini_items):
        result = apply_cold_start_fallback(mini_items, top_n=3)
        assert all(r.is_fallback for r in result)

    def test_all_scores_are_none(self, mini_items):
        result = apply_cold_start_fallback(mini_items, top_n=3)
        assert all(r.score is None for r in result)

    def test_truncates_to_top_n(self, mini_items):
        result = apply_cold_start_fallback(mini_items, top_n=2)
        assert len(result) == 2

    def test_fewer_items_than_top_n(self, mini_items):
        result = apply_cold_start_fallback(mini_items, top_n=10)
        assert len(result) == len(mini_items)

    def test_top_n_zero_raises(self, mini_items):
        with pytest.raises(ValueError):
            apply_cold_start_fallback(mini_items, top_n=0)

    def test_returns_scored_items(self, mini_items):
        result = apply_cold_start_fallback(mini_items, top_n=3)
        assert all(isinstance(r, ScoredItem) for r in result)

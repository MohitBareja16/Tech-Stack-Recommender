"""
test_integration.py — End-to-end and edge case tests for main.recommend().
Coverage: TC-INT-01 through TC-INT-08, TC-EDGE-01 through TC-EDGE-05 (TEST_PLAN.md §8–9).
"""

import pytest

from main import InputValidationError, recommend
from src.similarity_engine import ScoredItem


# ---------------------------------------------------------------------------
# TC-INT-01: Happy path — correct top-1 result
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_returns_list(self, sample_dataset_path):
        results = recommend(["Python", "Machine Learning", "Statistics"], dataset_path=sample_dataset_path)
        assert isinstance(results, list)

    def test_returns_top_3_by_default(self, sample_dataset_path):
        results = recommend(["Python", "Machine Learning", "Statistics"], dataset_path=sample_dataset_path)
        assert len(results) == 3

    def test_top_result_is_relevant(self, sample_dataset_path):
        results = recommend(["Python", "Machine Learning", "Statistics"], dataset_path=sample_dataset_path)
        top_role = results[0].role_name.lower()
        # Should be data scientist or ml engineer — both are plausible top matches
        assert "scientist" in top_role or "engineer" in top_role or "analyst" in top_role

    def test_results_sorted_descending(self, sample_dataset_path):
        results = recommend(["Python", "Machine Learning", "Statistics"], dataset_path=sample_dataset_path)
        scores = [r.score for r in results if r.score is not None]
        assert scores == sorted(scores, reverse=True)

    def test_all_results_are_scored_items(self, sample_dataset_path):
        results = recommend(["Python", "Machine Learning", "Statistics"], dataset_path=sample_dataset_path)
        assert all(isinstance(r, ScoredItem) for r in results)

    def test_top_result_score_above_zero(self, sample_dataset_path):
        results = recommend(["Python", "Machine Learning", "Statistics"], dataset_path=sample_dataset_path)
        assert results[0].score is not None and results[0].score > 0


# ---------------------------------------------------------------------------
# TC-INT-02: DevOps-adjacent skills surface relevant roles
# ---------------------------------------------------------------------------

class TestDevOpsQuery:
    def test_aws_docker_kubernetes_no_crash(self, sample_dataset_path):
        results = recommend(["aws", "docker", "kubernetes"], dataset_path=sample_dataset_path)
        assert 1 <= len(results) <= 3

    def test_devops_role_in_top_results(self, sample_dataset_path):
        results = recommend(["aws", "docker", "kubernetes"], dataset_path=sample_dataset_path)
        role_names = [r.role_name.lower() for r in results]
        # DevOps Engineer or Cloud Architect should surface
        has_relevant = any("devops" in n or "cloud" in n for n in role_names)
        assert has_relevant


# ---------------------------------------------------------------------------
# TC-INT-03: Fewer than 3 valid skills → InputValidationError
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_two_skills_raises(self, sample_dataset_path):
        with pytest.raises(InputValidationError):
            recommend(["Python", "SQL"], dataset_path=sample_dataset_path)

    def test_one_skill_raises(self, sample_dataset_path):
        with pytest.raises(InputValidationError):
            recommend(["Python"], dataset_path=sample_dataset_path)

    def test_zero_skills_raises(self, sample_dataset_path):
        with pytest.raises(InputValidationError):
            recommend([], dataset_path=sample_dataset_path)

    def test_error_message_mentions_minimum(self, sample_dataset_path):
        with pytest.raises(InputValidationError, match="3"):
            recommend(["Python", "SQL"], dataset_path=sample_dataset_path)


# ---------------------------------------------------------------------------
# TC-INT-04: Blank skills filtered before count
# ---------------------------------------------------------------------------

class TestBlankSkillFiltering:
    def test_blank_skills_not_counted(self, sample_dataset_path):
        # 2 blanks + 1 real = only 1 valid skill → should raise
        with pytest.raises(InputValidationError):
            recommend(["", "  ", "Python"], dataset_path=sample_dataset_path)

    def test_whitespace_only_not_counted(self, sample_dataset_path):
        with pytest.raises(InputValidationError):
            recommend(["   ", "\t", "Python"], dataset_path=sample_dataset_path)


# ---------------------------------------------------------------------------
# TC-INT-05: Cold Start fallback triggered
# ---------------------------------------------------------------------------

class TestColdStartFallback:
    def test_all_oov_returns_fallback(self, sample_dataset_path):
        results = recommend(["Photography", "Painting", "Sculpture"], dataset_path=sample_dataset_path)
        assert all(r.is_fallback for r in results)

    def test_fallback_ordered_by_popularity(self, sample_dataset_path):
        results = recommend(["Photography", "Painting", "Sculpture"], dataset_path=sample_dataset_path)
        ranks = [r.popularity_rank for r in results]
        assert ranks == sorted(ranks)

    def test_fallback_scores_are_none(self, sample_dataset_path):
        results = recommend(["Photography", "Painting", "Sculpture"], dataset_path=sample_dataset_path)
        assert all(r.score is None for r in results)


# ---------------------------------------------------------------------------
# TC-INT-06: top_n=1 returns exactly 1 result
# ---------------------------------------------------------------------------

class TestTopNParameter:
    def test_top_n_one(self, sample_dataset_path):
        results = recommend(["Python", "Machine Learning", "Statistics"],
                             dataset_path=sample_dataset_path, top_n=1)
        assert len(results) == 1

    def test_top_n_five(self, sample_dataset_path):
        results = recommend(["Python", "Machine Learning", "Statistics"],
                             dataset_path=sample_dataset_path, top_n=5)
        assert len(results) == 5

    def test_top_n_larger_than_dataset(self, sample_dataset_path):
        # Dataset has 10 items; top_n=100 should return all 10
        results = recommend(["Python", "Machine Learning", "Statistics"],
                             dataset_path=sample_dataset_path, top_n=100)
        assert len(results) == 10


# ---------------------------------------------------------------------------
# TC-INT-07: Determinism — same input → same output
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_two_runs_identical(self, sample_dataset_path):
        skills = ["Python", "Machine Learning", "Statistics"]
        run1 = recommend(skills, dataset_path=sample_dataset_path)
        run2 = recommend(skills, dataset_path=sample_dataset_path)
        assert [(r.role_name, r.score) for r in run1] == [(r.role_name, r.score) for r in run2]


# ---------------------------------------------------------------------------
# TC-INT-08: Verbose mode no crash
# ---------------------------------------------------------------------------

class TestVerboseMode:
    def test_verbose_true_no_crash(self, sample_dataset_path):
        results = recommend(["Python", "Machine Learning", "Statistics"],
                             dataset_path=sample_dataset_path, verbose=True)
        assert results is not None


# ---------------------------------------------------------------------------
# TC-EDGE tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_top_n_zero_raises(self, sample_dataset_path):
        # TC-EDGE-03
        with pytest.raises(ValueError):
            recommend(["Python", "Machine Learning", "Statistics"],
                      dataset_path=sample_dataset_path, top_n=0)

    def test_top_n_negative_raises(self, sample_dataset_path):
        with pytest.raises(ValueError):
            recommend(["Python", "Machine Learning", "Statistics"],
                      dataset_path=sample_dataset_path, top_n=-5)

    def test_duplicate_skills_increase_tf(self, sample_dataset_path):
        # TC-EDGE-02: PRD §10.1 — duplicates count for TF
        # python×3 should heavily bias toward python-heavy roles
        results = recommend(["Python", "python", "PYTHON"], dataset_path=sample_dataset_path)
        assert results[0].score is not None and results[0].score > 0
        top_role = results[0].role_name.lower()
        # Any Python-heavy role should win
        python_roles = {"data scientist", "backend developer", "full stack developer",
                        "data analyst", "ml engineer"}
        assert top_role in python_roles

    def test_mixed_case_normalization(self, sample_dataset_path):
        # "PYTHON" and "python" and "Python" should all match the same vocab term
        results_upper = recommend(["PYTHON", "SQL", "PANDAS"], dataset_path=sample_dataset_path)
        results_lower = recommend(["python", "sql", "pandas"], dataset_path=sample_dataset_path)
        assert [(r.role_name, r.score) for r in results_upper] == [(r.role_name, r.score) for r in results_lower]

    def test_nonexistent_dataset_raises_data_load_error(self):
        from src.data_loader import DataLoadError
        with pytest.raises(DataLoadError):
            recommend(["Python", "SQL", "Pandas"], dataset_path="/no/such/file.csv")

    def test_matched_skills_present_for_top_result(self, sample_dataset_path):
        # US-3 AC1: top result must include matched_skills for explainability
        results = recommend(["python", "sql", "pandas"], dataset_path=sample_dataset_path)
        assert isinstance(results[0].matched_skills, list)
        assert len(results[0].matched_skills) > 0

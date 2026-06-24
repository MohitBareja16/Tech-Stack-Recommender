"""
test_data_loader.py — Unit tests for src/data_loader.py.
Coverage: TC-DL-01 through TC-DL-06 (TEST_PLAN.md §3).
"""

import pytest

from src.data_loader import (
    DataLoadError,
    ItemRecord,
    load_items,
    normalize_skill,
    normalize_text,
    validate_row,
)


# ---------------------------------------------------------------------------
# TC-DL-05: normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_strips_leading_trailing_whitespace(self):
        assert normalize_text("  Python  ") == "python"

    def test_lowercases_input(self):
        assert normalize_text("MACHINE LEARNING") == "machine learning"

    def test_already_normalized(self):
        assert normalize_text("aws") == "aws"

    def test_collapses_internal_whitespace(self):
        assert normalize_text("deep   learning") == "deep learning"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_non_string_coerced(self):
        # TC-DL-06: numeric input coerced to string
        assert normalize_text(42) == "42"

    def test_none_coerced(self):
        assert normalize_text(None) == "none"


class TestNormalizeSkill:
    def test_space_to_underscore(self):
        assert normalize_skill("Machine Learning") == "machine_learning"

    def test_strips_and_lowercases(self):
        assert normalize_skill("  AWS  ") == "aws"

    def test_multiple_spaces(self):
        # \s+ matches the entire whitespace run and replaces it with one _
        assert normalize_skill("deep   learning") == "deep_learning"

    def test_already_underscored(self):
        assert normalize_skill("machine_learning") == "machine_learning"

    def test_non_string_coerced(self):
        assert normalize_skill(42) == "42"


# ---------------------------------------------------------------------------
# TC-DL-01: Happy-path load
# ---------------------------------------------------------------------------

class TestLoadItemsHappyPath:
    def test_returns_correct_count(self, tmp_csv_path):
        path = tmp_csv_path([
            {"role_name": "Data Scientist",    "skills": "python|sql",       "popularity_rank": "1"},
            {"role_name": "DevOps Engineer",   "skills": "aws|docker",       "popularity_rank": "2"},
            {"role_name": "Backend Developer", "skills": "java|python|sql",  "popularity_rank": "3"},
        ])
        items = load_items(path)
        assert len(items) == 3

    def test_returns_item_records(self, tmp_csv_path):
        path = tmp_csv_path([
            {"role_name": "Data Scientist", "skills": "python|sql|machine_learning", "popularity_rank": "1"},
        ])
        items = load_items(path)
        assert all(isinstance(item, ItemRecord) for item in items)

    def test_role_name_normalized(self, tmp_csv_path):
        path = tmp_csv_path([
            {"role_name": "  Data Scientist  ", "skills": "python", "popularity_rank": "1"},
        ])
        items = load_items(path)
        assert items[0].role_name == "data scientist"

    def test_skills_normalized_and_split(self, tmp_csv_path):
        path = tmp_csv_path([
            {"role_name": "Role", "skills": "Python|SQL|Machine_Learning", "popularity_rank": "1"},
        ])
        items = load_items(path)
        assert "python" in items[0].skills
        assert "sql" in items[0].skills

    def test_popularity_rank_parsed(self, tmp_csv_path):
        path = tmp_csv_path([
            {"role_name": "Role", "skills": "python", "popularity_rank": "5"},
        ])
        items = load_items(path)
        assert items[0].popularity_rank == 5


# ---------------------------------------------------------------------------
# TC-DL-02: File not found
# ---------------------------------------------------------------------------

class TestLoadItemsFileNotFound:
    def test_raises_data_load_error(self):
        with pytest.raises(DataLoadError, match="not found"):
            load_items("/nonexistent/path/dataset.csv")

    def test_error_message_contains_path(self):
        bad_path = "/does/not/exist.csv"
        with pytest.raises(DataLoadError, match="does/not/exist"):
            load_items(bad_path)


# ---------------------------------------------------------------------------
# TC-DL-03: Empty file (header only)
# ---------------------------------------------------------------------------

class TestLoadItemsEmptyFile:
    def test_raises_data_load_error_on_header_only(self, tmp_csv_path):
        path = tmp_csv_path([])  # header written, no data rows
        with pytest.raises(DataLoadError):
            load_items(path)


# ---------------------------------------------------------------------------
# TC-DL-04: Row with blank skills skipped
# ---------------------------------------------------------------------------

class TestLoadItemsSkipsInvalidRows:
    def test_skips_blank_skills_row(self, tmp_csv_path):
        path = tmp_csv_path([
            {"role_name": "Valid Role 1",  "skills": "python|sql",  "popularity_rank": "1"},
            {"role_name": "Invalid Role",  "skills": "",             "popularity_rank": "2"},
            {"role_name": "Valid Role 2",  "skills": "aws|docker",  "popularity_rank": "3"},
        ])
        items = load_items(path)
        assert len(items) == 2
        role_names = [i.role_name for i in items]
        assert "invalid role" not in role_names

    def test_skips_blank_role_name(self, tmp_csv_path):
        path = tmp_csv_path([
            {"role_name": "",              "skills": "python|sql",  "popularity_rank": "1"},
            {"role_name": "Valid Role",    "skills": "aws|docker",  "popularity_rank": "2"},
        ])
        items = load_items(path)
        assert len(items) == 1

    def test_skips_invalid_popularity_rank(self, tmp_csv_path):
        path = tmp_csv_path([
            {"role_name": "Good Role",  "skills": "python", "popularity_rank": "1"},
            {"role_name": "Bad Rank",   "skills": "python", "popularity_rank": "notanint"},
        ])
        items = load_items(path)
        assert len(items) == 1

    def test_skips_zero_popularity_rank(self, tmp_csv_path):
        path = tmp_csv_path([
            {"role_name": "Good Role",  "skills": "python", "popularity_rank": "1"},
            {"role_name": "Zero Rank",  "skills": "python", "popularity_rank": "0"},
        ])
        items = load_items(path)
        assert len(items) == 1


# ---------------------------------------------------------------------------
# validate_row unit tests
# ---------------------------------------------------------------------------

class TestValidateRow:
    def test_valid_row_returns_true(self):
        assert validate_row({"role_name": "Dev", "skills": "python|sql", "popularity_rank": "1"})

    def test_blank_role_name_returns_false(self):
        assert not validate_row({"role_name": "   ", "skills": "python", "popularity_rank": "1"})

    def test_blank_skills_returns_false(self):
        assert not validate_row({"role_name": "Dev", "skills": "", "popularity_rank": "1"})

    def test_skills_all_pipes_returns_false(self):
        assert not validate_row({"role_name": "Dev", "skills": "|||", "popularity_rank": "1"})

    def test_negative_rank_returns_false(self):
        assert not validate_row({"role_name": "Dev", "skills": "python", "popularity_rank": "-1"})

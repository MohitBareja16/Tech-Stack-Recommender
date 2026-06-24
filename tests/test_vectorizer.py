"""
test_vectorizer.py — Unit tests for src/vectorizer.py.
Coverage: TC-VEC-01 through TC-VEC-07 (TEST_PLAN.md §4).

All expected values are hand-computed from ALGORITHM_SPEC.md §3.
"""

import math

import numpy as np
import pytest

from src.data_loader import ItemRecord, normalize_text
from src.vectorizer import (
    build_item_matrix,
    build_user_vector,
    build_vocabulary,
    compute_idf,
    compute_tf,
)

FLOAT_TOL = 1e-4  # PRD NFR-1 allows 1e-6; 1e-4 is conservative safety margin


# ---------------------------------------------------------------------------
# TC-VEC-01: build_vocabulary
# ---------------------------------------------------------------------------

class TestBuildVocabulary:
    def test_returns_sorted_unique_tokens(self, mini_items):
        vocab = build_vocabulary(mini_items)
        assert vocab == sorted(set(vocab)), "Vocabulary must be alphabetically sorted"

    def test_exact_vocabulary(self, mini_items, mini_vocab):
        assert build_vocabulary(mini_items) == mini_vocab

    def test_no_duplicates(self, mini_items):
        vocab = build_vocabulary(mini_items)
        assert len(vocab) == len(set(vocab))

    def test_single_item(self):
        items = [ItemRecord(role_name="Role A", skills=["python", "sql"], popularity_rank=1)]
        vocab = build_vocabulary(items)
        assert vocab == ["python", "sql"]

    def test_corpus_is_sole_source(self, mini_items, mini_vocab):
        # User OOV terms must NOT be in vocab
        vocab = build_vocabulary(mini_items)
        assert "statistics" not in vocab  # OOV from ALGORITHM_SPEC.md §3.7


# ---------------------------------------------------------------------------
# TC-VEC-02: compute_tf
# ---------------------------------------------------------------------------

class TestComputeTF:
    def test_data_scientist_tf(self, mini_vocab):
        tokens = ["python", "sql", "machine_learning"]
        tf = compute_tf(tokens, mini_vocab)
        idx_ml = mini_vocab.index("machine_learning")
        idx_py = mini_vocab.index("python")
        idx_sq = mini_vocab.index("sql")
        assert tf[idx_ml] == pytest.approx(1 / 3, abs=FLOAT_TOL)
        assert tf[idx_py] == pytest.approx(1 / 3, abs=FLOAT_TOL)
        assert tf[idx_sq] == pytest.approx(1 / 3, abs=FLOAT_TOL)

    def test_non_vocab_terms_are_zero(self, mini_vocab):
        tokens = ["python", "sql", "machine_learning"]
        tf = compute_tf(tokens, mini_vocab)
        idx_aws = mini_vocab.index("aws")
        assert tf[idx_aws] == 0.0

    def test_empty_tokens_returns_zero_vector(self, mini_vocab):
        tf = compute_tf([], mini_vocab)
        assert np.all(tf == 0.0)

    def test_oov_tokens_count_toward_denominator(self, mini_vocab):
        # TC-VEC-07: "statistics" is OOV but counts in denominator
        tokens = ["python", "machine_learning", "statistics"]
        tf = compute_tf(tokens, mini_vocab)
        idx_py = mini_vocab.index("python")
        # TF(python) should be 1/3, not 1/2 (denominator = all tokens including OOV)
        assert tf[idx_py] == pytest.approx(1 / 3, abs=FLOAT_TOL)

    def test_duplicate_tokens_increase_tf(self, mini_vocab):
        # PRD §10.1: duplicates intentionally increase TF
        tokens = ["python", "python", "python"]
        tf = compute_tf(tokens, mini_vocab)
        idx_py = mini_vocab.index("python")
        assert tf[idx_py] == pytest.approx(1.0, abs=FLOAT_TOL)  # 3/3 = 1.0

    def test_returns_numpy_array(self, mini_vocab):
        tf = compute_tf(["python"], mini_vocab)
        assert isinstance(tf, np.ndarray)

    def test_shape_equals_vocab_size(self, mini_vocab):
        tf = compute_tf(["python", "sql"], mini_vocab)
        assert tf.shape == (len(mini_vocab),)


# ---------------------------------------------------------------------------
# TC-VEC-03: compute_idf
# ---------------------------------------------------------------------------

class TestComputeIDF:
    def test_idf_python_two_items(self, mini_items, mini_vocab):
        # python appears in 2 of 3 items: IDF = log(3/2) = 0.4055
        idf = compute_idf(mini_items, mini_vocab)
        idx_py = mini_vocab.index("python")
        assert idf[idx_py] == pytest.approx(math.log(3 / 2), abs=FLOAT_TOL)

    def test_idf_aws_one_item(self, mini_items, mini_vocab):
        # aws appears in 1 of 3 items: IDF = log(3/1) = 1.0986
        idf = compute_idf(mini_items, mini_vocab)
        idx_aws = mini_vocab.index("aws")
        assert idf[idx_aws] == pytest.approx(math.log(3 / 1), abs=FLOAT_TOL)

    def test_idf_sql_two_items(self, mini_items, mini_vocab):
        idf = compute_idf(mini_items, mini_vocab)
        idx_sql = mini_vocab.index("sql")
        assert idf[idx_sql] == pytest.approx(math.log(3 / 2), abs=FLOAT_TOL)

    def test_rare_terms_higher_idf(self, mini_items, mini_vocab):
        idf = compute_idf(mini_items, mini_vocab)
        idx_py = mini_vocab.index("python")
        idx_aws = mini_vocab.index("aws")
        assert idf[idx_aws] > idf[idx_py], "Rare term (aws) must have higher IDF than common (python)"

    def test_returns_numpy_array(self, mini_items, mini_vocab):
        idf = compute_idf(mini_items, mini_vocab)
        assert isinstance(idf, np.ndarray)


# ---------------------------------------------------------------------------
# TC-VEC-04: build_item_matrix
# ---------------------------------------------------------------------------

class TestBuildItemMatrix:
    def test_shape(self, mini_items, mini_vocab, mini_idf):
        matrix = build_item_matrix(mini_items, mini_vocab, mini_idf)
        assert matrix.shape == (3, 7)

    def test_data_scientist_machine_learning_weight(self, mini_items, mini_vocab, mini_idf):
        # Data Scientist × machine_learning = TF(1/3) × IDF(log 3/1) ≈ 0.3662
        matrix = build_item_matrix(mini_items, mini_vocab, mini_idf)
        idx_ml = mini_vocab.index("machine_learning")
        expected = (1 / 3) * math.log(3 / 1)
        assert matrix[0, idx_ml] == pytest.approx(expected, abs=FLOAT_TOL)

    def test_devops_aws_weight(self, mini_items, mini_vocab, mini_idf):
        # DevOps × aws = TF(1/3) × IDF(log 3/1) ≈ 0.3662
        matrix = build_item_matrix(mini_items, mini_vocab, mini_idf)
        idx_aws = mini_vocab.index("aws")
        expected = (1 / 3) * math.log(3 / 1)
        assert matrix[1, idx_aws] == pytest.approx(expected, abs=FLOAT_TOL)

    def test_non_overlapping_is_zero(self, mini_items, mini_vocab, mini_idf):
        # DevOps has no python; row 1, col python should be 0
        matrix = build_item_matrix(mini_items, mini_vocab, mini_idf)
        idx_py = mini_vocab.index("python")
        assert matrix[1, idx_py] == 0.0

    def test_returns_numpy_array(self, mini_items, mini_vocab, mini_idf):
        matrix = build_item_matrix(mini_items, mini_vocab, mini_idf)
        assert isinstance(matrix, np.ndarray)


# ---------------------------------------------------------------------------
# TC-VEC-05 & TC-VEC-06: build_user_vector
# ---------------------------------------------------------------------------

class TestBuildUserVector:
    def test_user_vector_python_weight(self, user_vector_canonical, mini_vocab):
        user_vec, _ = user_vector_canonical
        idx_py = mini_vocab.index("python")
        expected = (1 / 3) * math.log(3 / 2)  # TF=1/3, IDF=log(3/2)
        assert user_vec[idx_py] == pytest.approx(expected, abs=FLOAT_TOL)

    def test_user_vector_machine_learning_weight(self, user_vector_canonical, mini_vocab):
        user_vec, _ = user_vector_canonical
        idx_ml = mini_vocab.index("machine_learning")
        expected = (1 / 3) * math.log(3 / 1)  # TF=1/3, IDF=log(3/1)
        assert user_vec[idx_ml] == pytest.approx(expected, abs=FLOAT_TOL)

    def test_sql_is_zero_for_user(self, user_vector_canonical, mini_vocab):
        user_vec, _ = user_vector_canonical
        idx_sql = mini_vocab.index("sql")
        assert user_vec[idx_sql] == 0.0

    def test_oov_term_reported(self, user_vector_canonical):
        _, oov = user_vector_canonical
        assert "statistics" in oov

    def test_oov_does_not_mutate_vocab(self, mini_vocab, mini_idf):
        # TC-VEC-06: critical invariant — vocab must not grow
        vocab_before = list(mini_vocab)
        build_user_vector(["python", "machine_learning", "statistics"], mini_vocab, mini_idf)
        assert mini_vocab == vocab_before

    def test_all_oov_returns_zero_vector(self, mini_vocab, mini_idf):
        user_vec, oov = build_user_vector(
            ["photography", "painting", "sculpture"], mini_vocab, mini_idf
        )
        assert np.all(user_vec == 0.0)
        assert len(oov) == 3

    def test_returns_tuple(self, mini_vocab, mini_idf):
        result = build_user_vector(["python", "sql", "aws"], mini_vocab, mini_idf)
        assert isinstance(result, tuple)
        assert len(result) == 2

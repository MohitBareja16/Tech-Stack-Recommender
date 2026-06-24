# Test Plan
## Tech Stack Recommender — Project 3

**Companion to:** `PRD.md §9`, `ARCHITECTURE.md §3`, `ALGORITHM_SPEC.md §3`
**Framework:** `pytest` | **Coverage target:** 100% of qualification criteria (PRD NFR-7)

---

## 1. Test Philosophy

- **Unit tests** isolate each module function; pass hand-computed expected values
  from `ALGORITHM_SPEC.md §3` (satisfies PRD NFR-1).
- **Integration tests** exercise the full `main.recommend()` pipeline end-to-end
  (satisfies PRD NFR-7's "every concept has a locatable code artifact + passing test").
- **No mocking of math.** TF, IDF, and cosine calculations are asserted against
  hand-computed values, not mocked return values.
- **Float tolerance:** `pytest.approx` with `abs=1e-4` throughout (NFR-1 allows 1e-6;
  `1e-4` gives margin for intermediate rounding without losing meaningful precision).

---

## 2. Fixture: Mini Corpus (from `ALGORITHM_SPEC.md §3.1`)

All unit tests share this fixture (defined in `conftest.py`):

```python
MINI_ITEMS = [
    ItemRecord(role_name="Data Scientist",    skills=["python","sql","machine_learning"],    popularity_rank=1),
    ItemRecord(role_name="DevOps Engineer",   skills=["aws","docker","kubernetes"],           popularity_rank=2),
    ItemRecord(role_name="Backend Developer", skills=["java","python","sql"],                 popularity_rank=3),
]
MINI_VOCAB = ["aws", "docker", "java", "kubernetes", "machine_learning", "python", "sql"]
```

---

## 3. Unit Tests — `data_loader.py`

### TC-DL-01: Happy-path CSV load
- **Given:** a valid `raw_skills.csv` with 3 well-formed rows
- **When:** `load_items(path)` is called
- **Then:** returns `list[ItemRecord]` of length 3, all fields populated

### TC-DL-02: File not found
- **Given:** path that does not exist
- **When:** `load_items(path)` is called
- **Then:** raises `DataLoadError` (not `FileNotFoundError`) with the path in the message

### TC-DL-03: Empty file (header only)
- **Given:** CSV with a header row and no data rows
- **When:** `load_items(path)` is called
- **Then:** raises `DataLoadError`

### TC-DL-04: Row with blank skills skipped
- **Given:** CSV with 2 valid rows and 1 row where `skills` is empty string `""`
- **When:** `load_items(path)` is called
- **Then:** returns list of length 2; no crash; warning emitted

### TC-DL-05: normalize_text
- **Given:** input strings `"  Python  "`, `"MACHINE LEARNING"`, `"aws"`
- **When:** `normalize_text(s)` is called
- **Then:** returns `"python"`, `"machine learning"`, `"aws"` respectively

### TC-DL-06: Non-string skill coercion
- **Given:** a row where a skill token is the integer `42`
- **When:** loaded and normalized
- **Then:** coerced to `"42"`, no crash

---

## 4. Unit Tests — `vectorizer.py`

### TC-VEC-01: build_vocabulary — correct union, alphabetically sorted
- **Given:** `MINI_ITEMS`
- **When:** `build_vocabulary(MINI_ITEMS)`
- **Then:** returns `["aws","docker","java","kubernetes","machine_learning","python","sql"]`

### TC-VEC-02: compute_tf — Data Scientist vector
- **Given:** tokens = `["python","sql","machine_learning"]`, vocab = `MINI_VOCAB`
- **When:** `compute_tf(tokens, MINI_VOCAB)`
- **Then:**
  - `tf[4]` (machine_learning) `≈ 0.3333` (±1e-4)
  - `tf[5]` (python) `≈ 0.3333`
  - `tf[6]` (sql) `≈ 0.3333`
  - all others `== 0.0`

### TC-VEC-03: compute_idf — matches hand-computed values
- **Given:** `MINI_ITEMS`, `MINI_VOCAB`
- **When:** `compute_idf(MINI_ITEMS, MINI_VOCAB)`
- **Then:**
  - `idf[5]` (python, df=2) `≈ log(3/2) = 0.4055` (±1e-4)
  - `idf[0]` (aws, df=1) `≈ log(3/1) = 1.0986` (±1e-4)

### TC-VEC-04: build_item_matrix — shape and spot-check
- **Given:** `MINI_ITEMS`, `MINI_VOCAB`, IDF from TC-VEC-03
- **When:** `build_item_matrix(MINI_ITEMS, MINI_VOCAB, idf)`
- **Then:**
  - Shape is `(3, 7)`
  - Row 0, col 4 (Data Scientist × machine_learning) `≈ 0.3662` (±1e-4)
  - Row 1, col 0 (DevOps × aws) `≈ 0.3662`

### TC-VEC-05: build_user_vector — in-vocab terms weighted correctly
- **Given:** user_skills = `["Python","Machine Learning","Statistics"]` (pre-normalization),
  vocab = `MINI_VOCAB`, IDF from TC-VEC-03
- **When:** `build_user_vector(user_skills, vocab, idf)`
- **Then:**
  - `user_vector[5]` (python) `≈ 0.1352` (±1e-4) — TF=1/3 × IDF=0.4055
  - `user_vector[4]` (machine_learning) `≈ 0.3662` — TF=1/3 × IDF=1.0986
  - `user_vector[6]` (sql) `== 0.0`
  - OOV list contains `"statistics"`

### TC-VEC-06: build_user_vector — OOV does NOT mutate vocab
- **Given:** same inputs as TC-VEC-05
- **When:** `build_user_vector(user_skills, vocab, idf)` is called
- **Then:** `len(vocab)` is still `7` after the call

### TC-VEC-07: compute_tf — TF denominator uses ALL input tokens (including OOV)
- **Given:** tokens = `["python", "machine_learning", "statistics"]` (statistics is OOV but still in list)
- **When:** `compute_tf(tokens, MINI_VOCAB)` — with `total_terms = len(tokens) = 3`
- **Then:** `tf[5]` (python) `≈ 1/3 ≈ 0.3333`, confirming OOV terms count toward denominator

---

## 5. Unit Tests — `similarity_engine.py`

### TC-SIM-01: cosine_similarity — canonical hand-computed value
- **Given:** `A = [0,0,0,0,0.3662,0.1352,0.1352]`, `B = [0,0,0,0,0.3662,0.1352,0]`
- **When:** `cosine_similarity(A, B)`
- **Then:** result `≈ 0.9448` (±1e-4) — exact value from `ALGORITHM_SPEC.md §3.9`

### TC-SIM-02: cosine_similarity — zero vector guard (Cold Start math)
- **Given:** `A = [0,0,0,0,0,0,0]`, `B = [0.3662,0.1352,0.1352,0,0,0,0]`
- **When:** `cosine_similarity(A, B)`
- **Then:** returns `0.0` (not NaN, not exception)

### TC-SIM-03: cosine_similarity — both zero vectors
- **Given:** `A = [0,0,0]`, `B = [0,0,0]`
- **When:** `cosine_similarity(A, B)`
- **Then:** returns `0.0`

### TC-SIM-04: score_all_items — correct scores for mini corpus
- **Given:** `user_vector` from TC-VEC-05, `item_matrix` from TC-VEC-04, `MINI_ITEMS`
- **When:** `score_all_items(user_vector, item_matrix, MINI_ITEMS)`
- **Then:**
  - `scored[0].score ≈ 0.9448` (Data Scientist)
  - `scored[1].score == 0.0` (DevOps Engineer)
  - `scored[2].score ≈ 0.1135` (Backend Developer, from §3.10)

### TC-SIM-05: score_all_items — matched_skills populated correctly
- **Given:** same as TC-SIM-04
- **When:** `score_all_items(...)`
- **Then:** `scored[0].matched_skills` contains `"python"` and `"machine_learning"` (the non-zero overlap terms)

### TC-SIM-06: is_cold_start — all zeros → True
- **Given:** `scored_items` where every `.score == 0.0`
- **When:** `is_cold_start(scored_items)`
- **Then:** returns `True`

### TC-SIM-07: is_cold_start — any nonzero → False
- **Given:** `scored_items` with at least one `.score > 0.0`
- **When:** `is_cold_start(scored_items)`
- **Then:** returns `False`

---

## 6. Unit Tests — `ranker.py`

### TC-RANK-01: rank — descending order by score
- **Given:** `scored_items` with scores `[0.9448, 0.1135, 0.0]` (Data Scientist, Backend, DevOps)
- **When:** `rank(scored_items, top_n=3)`
- **Then:** result order is Data Scientist → Backend Developer → DevOps Engineer

### TC-RANK-02: rank — tie-breaking by role name ascending
- **Given:** two items with identical score `0.5`: `"Zebra Role"` and `"Alpha Role"`
- **When:** `rank([...], top_n=2)`
- **Then:** `"Alpha Role"` appears before `"Zebra Role"`

### TC-RANK-03: rank — top_n truncation
- **Given:** 3 scored items, `top_n=2`
- **When:** `rank(scored_items, top_n=2)`
- **Then:** returns exactly 2 items

### TC-RANK-04: rank — fewer items than top_n
- **Given:** 2 scored items, `top_n=5`
- **When:** `rank(scored_items, top_n=5)`
- **Then:** returns 2 items (no padding, no error)

### TC-RANK-05: rank — top_n ≤ 0 raises ValueError
- **Given:** `top_n=0`
- **When:** `rank(scored_items, top_n=0)`
- **Then:** raises `ValueError`

### TC-RANK-06: apply_cold_start_fallback — ordered by popularity_rank
- **Given:** `MINI_ITEMS` (popularity_ranks: Data Scientist=1, DevOps=2, Backend=3), `top_n=3`
- **When:** `apply_cold_start_fallback(MINI_ITEMS, top_n=3)`
- **Then:**
  - Returns 3 items in order: Data Scientist → DevOps Engineer → Backend Developer
  - Each item has `is_fallback=True`
  - Each item has `score=None`

---

## 7. Unit Tests — `presenter.py`

### TC-PRES-01: format_results — contains role name and score
- **Given:** a list of `ScoredItem` with one item (role=`"Data Scientist"`, score=`0.9448`)
- **When:** `format_results([...], verbose=False)`
- **Then:** returned string contains `"Data Scientist"` and `"0.94"` (or equivalent rounded representation)

### TC-PRES-02: format_results — fallback label appears for cold-start items
- **Given:** a `ScoredItem` with `is_fallback=True`, `score=None`
- **When:** `format_results([...], verbose=False)`
- **Then:** output contains the word `"fallback"` or `"FALLBACK"` (case-insensitive)

### TC-PRES-03: format_debug_view — contains vector dimension info
- **Given:** `user_vector` from TC-VEC-05, `vocab = MINI_VOCAB`, `top_match = scored[0]`
- **When:** `format_debug_view(user_vector, vocab, top_match)`
- **Then:** output contains at least the top match's role name and some numeric representation of the vector

---

## 8. Integration Tests — `main.recommend()`

These use the full sample dataset (`data/raw_skills.csv`) and exercise the complete pipeline.

### TC-INT-01: Happy-path — correct top-1 result
- **Given:** `user_skills = ["Python", "Machine Learning", "Statistics"]`
- **When:** `recommend(user_skills, dataset_path, top_n=3)`
- **Then:**
  - Returns a list of 3 results
  - First result role is `"Data Scientist"` (or `"ML Engineer"`) with score > 0
  - Results are in descending score order

### TC-INT-02: AC1 from US-1 — DevOps Engineer surfaces for cloud/automation skills
- **Given:** `user_skills = ["Python", "Cloud", "Automation"]`
  (Note: "Cloud" and "Automation" may be OOV depending on dataset — test verifies graceful handling)
- **When:** `recommend(user_skills, dataset_path, top_n=3)`
- **Then:** no crash; returns between 1 and 3 results; sorted descending

### TC-INT-03: Fewer than 3 valid skills → InputValidationError
- **Given:** `user_skills = ["Python", "SQL"]` (only 2 non-blank)
- **When:** `recommend(user_skills, dataset_path)`
- **Then:** raises `InputValidationError`

### TC-INT-04: Blank skills filtered before count
- **Given:** `user_skills = ["", "  ", "Python"]`
- **When:** `recommend(user_skills, dataset_path)`
- **Then:** raises `InputValidationError` (only 1 valid skill after normalization)

### TC-INT-05: Cold Start fallback triggered (US-4)
- **Given:** `user_skills = ["Photography", "Painting", "Sculpture"]` (all OOV)
- **When:** `recommend(user_skills, dataset_path, top_n=3)`
- **Then:**
  - Returns 3 results ordered by `popularity_rank`
  - Every result has `is_fallback=True`

### TC-INT-06: top_n=1 returns exactly 1 result
- **Given:** valid skills, `top_n=1`
- **When:** `recommend(user_skills, dataset_path, top_n=1)`
- **Then:** returns a list of length 1

### TC-INT-07: Determinism — same input produces same output
- **Given:** same `user_skills` and same `dataset_path`
- **When:** `recommend()` called twice
- **Then:** both calls return byte-identical results

### TC-INT-08: Verbose/debug mode returns debug string (US-5 AC1)
- **Given:** valid skills, `verbose=True`
- **When:** `recommend(user_skills, dataset_path, verbose=True)`
- **Then:** no crash; debug output accessible (captured from presenter)

---

## 9. Edge Case Tests

### TC-EDGE-01: Dataset with exactly 1 role, top_n=3
- Returns list of length 1 (no padding)

### TC-EDGE-02: User inputs all-duplicate skills after normalization
- `["Python","python","PYTHON"]` → TF(python) = 3/3 = 1.0 per PRD §10.1
- Verify TF calculation uses raw count (not deduplicated)

### TC-EDGE-03: `top_n=-1` raises ValueError
- `recommend(valid_skills, path, top_n=-1)` → `ValueError`

### TC-EDGE-04: Role name tie in score — alphabetical order guaranteed
- Construct two items with identical scores; verify alphabetical ordering

### TC-EDGE-05: Dataset row with blank `role_name` skipped
- CSV has a row where `role_name` is `""` — that row must not appear in output

---

## 10. Test File Layout

```
tests/
├── conftest.py             # shared fixtures: MINI_ITEMS, MINI_VOCAB, tmp CSV helpers
├── test_data_loader.py     # TC-DL-*
├── test_vectorizer.py      # TC-VEC-*
├── test_similarity_engine.py  # TC-SIM-*
├── test_ranker.py          # TC-RANK-*
├── test_presenter.py       # TC-PRES-*
└── test_integration.py     # TC-INT-*, TC-EDGE-*
```

---

## 11. Running Tests

```bash
# From project root:
pytest tests/ -v

# With coverage report:
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a single module:
pytest tests/test_vectorizer.py -v
```

**All tests must be green before submitting for DecodeLabs review (PRD §9 — 100% pass rate).**

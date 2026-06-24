# Architecture Document
## Tech Stack Recommender — Project 3 (Content-Based Filtering Engine)

**Companion to:** `PRD.md` | **See also:** `ALGORITHM_SPEC.md`, `DATA_SCHEMA.md`, `TEST_PLAN.md`

---

## 1. Architectural Principles

1. **IPO separation is structural, not just conceptual.** Ingestion,
   Processing (vectorization + scoring), and Output (sort/filter) live in
   physically separate modules with no cross-cutting state, per PRD NFR-6.
2. **Math from scratch, I/O from libraries.** TF, IDF, and Cosine Similarity
   must be implemented manually (per PRD Risks table) so the trainee
   demonstrates the math. `numpy`/`pandas` are permitted for array storage
   and CSV parsing only.
3. **No hidden global state.** The vocabulary, IDF table, and item vectors
   are computed once at load time and passed explicitly between functions
   — never re-derived ad hoc mid-pipeline (protects NFR-2 Determinism).
4. **Fail loud, not silent.** Malformed data, OOV terms, and cold-start
   conditions are surfaces as explicit signals in the return value/logs —
   never swallowed.

---

## 2. High-Level Component Diagram

```
                         ┌────────────────────────┐
                         │   raw_skills.csv        │
                         │  (Item Dataset)         │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │   data_loader.py        │
                         │  - parse CSV            │
                         │  - validate rows        │
                         │  - normalize text       │
                         └────────────┬────────────┘
                                      │  List[ItemRecord]
                                      ▼
                         ┌────────────────────────┐
       user skills  ───► │   vectorizer.py         │
       (CLI input)       │  - build vocabulary     │
                         │  - compute TF (items)   │
                         │  - compute IDF (corpus) │
                         │  - compute TF-IDF matrix│
                         │  - compute user vector  │
                         └────────────┬────────────┘
                                      │  (item_matrix, user_vector, vocab)
                                      ▼
                         ┌────────────────────────┐
                         │   similarity_engine.py  │
                         │  - cosine_similarity()  │
                         │  - score every item     │
                         │  - cold-start detection │
                         └────────────┬────────────┘
                                      │  List[ScoredItem]
                                      ▼
                         ┌────────────────────────┐
                         │   ranker.py             │
                         │  - sort desc by score   │
                         │  - tie-break by name    │
                         │  - truncate to top_n    │
                         └────────────┬────────────┘
                                      │  List[Recommendation]
                                      ▼
                         ┌────────────────────────┐
                         │   presenter.py          │
                         │  - format output        │
                         │  - explainability view  │
                         │  - fallback labeling    │
                         └────────────┬────────────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │  CLI / Output  │
                              └───────────────┘
```

---

## 3. Module Specifications

### 3.1 `data_loader.py`
**Responsibility:** Ingestion stage of the IPO pipeline.

| Function | Signature | Behavior |
|---|---|---|
| `load_items(path: str) -> list[ItemRecord]` | Reads CSV, raises `FileNotFoundError`/`DataLoadError` with actionable message if missing/empty | Required |
| `validate_row(row: dict) -> bool` | Returns False (and logs a warning) for rows with missing/blank `skills` field | Required |
| `normalize_text(s: str) -> str` | Lowercase + strip + collapse internal whitespace | Shared utility, reused by both item and user-input normalization (PRD §10.3) |

**`ItemRecord` shape:**
```python
@dataclass
class ItemRecord:
    role_name: str
    skills: list[str]       # normalized, de-duplicated within the row
    popularity_rank: int    # used only for Cold Start fallback ordering
```

**Validation rules enforced here (traceable to PRD Edge Cases table):**
- Empty file → `DataLoadError` raised immediately, not an empty list.
- Row with blank `skills` → skipped + warning logged, pipeline continues.
- Skill strings normalized before they ever reach the vectorizer.

---

### 3.2 `vectorizer.py`
**Responsibility:** Vector Mapping + TF-IDF (the "Process" half, part 1).

| Function | Signature | Behavior |
|---|---|---|
| `build_vocabulary(items: list[ItemRecord]) -> list[str]` | Union of all normalized skill tokens across the **item corpus only** (PRD §10.4 — corpus is the fixed source of truth) | Required |
| `compute_tf(tokens: list[str], vocab: list[str]) -> np.ndarray` | `count(t)/len(tokens)` per vocab dimension; works for both an item's skill list and the user's input list | Required |
| `compute_idf(items: list[ItemRecord], vocab: list[str]) -> np.ndarray` | `log(N / df(t))` per `ALGORITHM_SPEC.md §2` | Required |
| `build_item_matrix(items, vocab, idf) -> np.ndarray` | Shape `(num_items, vocab_size)`, each row = `TF(item) * IDF` | Required |
| `build_user_vector(user_skills: list[str], vocab, idf) -> tuple[np.ndarray, list[str]]` | Returns the weighted user vector **and** the list of OOV terms (terms in user input not in `vocab`) for transparency (FR-7) | Required |

**Critical invariant:** `build_user_vector` MUST NOT mutate `vocab`. OOV
terms are reported, never added. This enforces PRD §10.4.

---

### 3.3 `similarity_engine.py`
**Responsibility:** Scoring (the "Process" half, part 2).

| Function | Signature | Behavior |
|---|---|---|
| `cosine_similarity(a: np.ndarray, b: np.ndarray) -> float` | `dot(a,b) / (norm(a)*norm(b))`; **must return 0.0 (not NaN/error) when either norm is 0** | Required — this single guard is what correctly implements Cold Start math per the deck's "vector of zeros" example |
| `score_all_items(user_vector, item_matrix, item_records) -> list[ScoredItem]` | Computes cosine similarity of `user_vector` against every row of `item_matrix` | Required, full scan — no pre-filtering |
| `is_cold_start(scored_items: list[ScoredItem]) -> bool` | True if **every** score is exactly 0.0 | Required — feeds the fallback decision in `ranker.py` |

**`ScoredItem` shape:**
```python
@dataclass
class ScoredItem:
    role_name: str
    score: float
    matched_skills: list[str]   # vocab terms with nonzero weight in BOTH user & item vector
    popularity_rank: int
```

---

### 3.4 `ranker.py`
**Responsibility:** Sorting + Filtering (output truncation).

| Function | Signature | Behavior |
|---|---|---|
| `rank(scored_items, top_n: int) -> list[ScoredItem]` | Sort by `(-score, role_name)` for deterministic tie-break (PRD §10.2); slice to `top_n`; if fewer items exist than `top_n`, return all | Required |
| `apply_cold_start_fallback(items: list[ItemRecord], top_n: int) -> list[ScoredItem]` | Returns items ordered by `popularity_rank` ascending, each tagged `is_fallback=True`, `score=None` | Required, called only when `similarity_engine.is_cold_start()` is True |

**Validation guard:** `top_n <= 0` raises `ValueError` before any work is done (PRD Edge Cases).

---

### 3.5 `presenter.py`
**Responsibility:** Output formatting + explainability view.

| Function | Signature | Behavior |
|---|---|---|
| `format_results(results: list[ScoredItem], verbose: bool=False) -> str` | Human-readable table: role, score (or "fallback"), matched skills | Required |
| `format_debug_view(user_vector, vocab, top_match: ScoredItem) -> str` | Prints raw vector dimensions + dot product breakdown for the #1 match | Required for FR-7 / US-5 |

---

### 3.6 `main.py` (orchestrator / CLI entry point)
**Responsibility:** Wires the pipeline together; owns input validation
(minimum-3-skills rule) since that is a pipeline-entry concern, not a
module-internal one.

```python
def recommend(user_skills_raw: list[str], dataset_path: str, top_n: int = 3, verbose: bool = False) -> list[ScoredItem]:
    # 1. Normalize + validate (>=3 non-blank skills) -> InputValidationError if violated
    # 2. data_loader.load_items(dataset_path)
    # 3. vectorizer.build_vocabulary / compute_idf / build_item_matrix
    # 4. vectorizer.build_user_vector
    # 5. similarity_engine.score_all_items
    # 6. if similarity_engine.is_cold_start(...): ranker.apply_cold_start_fallback(...)
    #    else: ranker.rank(...)
    # 7. presenter.format_results(...)  [printed by CLI layer, not returned by this function]
```

This function is the **single integration point** tested end-to-end in
`TEST_PLAN.md §5` (integration tests), while §3.1–§3.5 are unit-tested in
isolation.

---

## 4. Data Flow Summary (per PRD §6 contract)

```
raw_skills.csv ──► ItemRecord[] ──► vocabulary, IDF table ──► item TF-IDF matrix
                                                   ▲
user skills[] ─────────────────────────────────────┘
                                                   │
                                                   ▼
                                        user TF-IDF vector
                                                   │
                          cosine_similarity(user_vector, item_matrix)
                                                   │
                                                   ▼
                                        ScoredItem[] (all items)
                                                   │
                              is_cold_start? ──Yes──► popularity fallback
                                    │No
                                    ▼
                         sort(-score, name) + truncate(top_n)
                                                   │
                                                   ▼
                                     Recommendation[] (final output)
```

---

## 5. Technology Choices

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.10+ | Matches the deck's vocabulary examples (`Python`, `TensorFlow` cubes in slide imagery) and is standard for the training track |
| Numeric ops | `numpy` | Allowed for array storage/dot-products per Architectural Principle #2 — NOT for `TfidfVectorizer` or `cosine_similarity` from `sklearn`, which would bypass the learning objective |
| CSV parsing | Python `csv` module or `pandas.read_csv` | I/O only, not similarity logic |
| Interface | CLI (argparse or simple `input()` loop) | Matches "Step 1: Ingestion... script must accept user inputs"; a web UI is an optional stretch goal, not required by PRD scope |
| Testing | `pytest` | Standard, supports the hand-computed-value assertions required by NFR-1 |

---

## 6. Performance Approach (NFR-3)

- `item_matrix` is a single `numpy` 2D array of shape `(num_items, vocab_size)`.
- Cosine similarity against **all** items is computed as one vectorized
  operation: `dots = item_matrix @ user_vector`, `norms = np.linalg.norm(item_matrix, axis=1)`,
  with zero-norm rows masked to a 0.0 score (vs. divide-by-zero) — this is
  the **vectorized equivalent** of the per-item loop described in the deck's
  "Step 2: Scoring" slide, satisfying both the conceptual requirement
  (loop through every item) and the performance requirement (NFR-3).
- This keeps the engine at O(n·v) with a single matrix multiply rather than
  an interpreted Python-level loop over thousands of items.

---

## 7. Error Handling Strategy

| Error Condition | Exception / Signal | Caught By |
|---|---|---|
| Dataset file missing or empty | `DataLoadError` | `main.py`, surfaced to CLI with actionable message |
| Fewer than 3 valid skills | `InputValidationError` | `main.py`, before pipeline runs |
| `top_n <= 0` | `ValueError` | `ranker.py` |
| All-OOV input | *(not an exception)* — handled as a first-class flow via `is_cold_start()` | `similarity_engine.py` → `ranker.py` |
| Malformed CSV row | Logged warning, row skipped | `data_loader.py` |

No bare `except:` blocks anywhere in the pipeline — every catch is
type-specific, per general engineering hygiene and to keep NFR-7
(auditability) honest.

---

## 8. Extension Points (explicitly out of scope for THIS milestone, per PRD §2.3, but architected so they don't require a rewrite)

- **Collaborative filtering:** could be added as a parallel scorer module
  that consumes the same `ItemRecord` shape, blended later via a weighted
  ensemble — not built now.
- **Synonym/embedding matching:** would replace `build_vocabulary`'s exact-
  string matching with a semantic lookup — explicitly deferred, since the
  deck frames this as "neural" territory for a future project.
- **Web UI:** `presenter.py`'s output is already a separate layer from the
  scoring pipeline, so a Flask/FastAPI wrapper could call `main.recommend()`
  without touching the math modules.
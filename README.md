<div align="center">

# 🎯 Tech Stack Recommender

### DecodeLabs Industrial Training Kit — Project 3: AI Recommendation Logic

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-only%20I%2FO-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![pytest](https://img.shields.io/badge/Tests-143%20passing-34D399?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org)
[![License](https://img.shields.io/badge/License-MIT-6366f1?style=for-the-badge)](LICENSE)

**A production-quality content-based recommendation engine built entirely from first principles.**  
TF-IDF vectorization + Cosine Similarity — zero ML libraries, full explainability.

[Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Architecture](#-architecture) · [Algorithm](#-algorithm-deep-dive) · [Testing](#-testing) · [CLI Reference](#-cli-reference)

</div>

---

## 📋 Table of Contents

1. [Background](#-background)
2. [Quick Start](#-quick-start)
3. [Features](#-features)
4. [How It Works](#-how-it-works)
5. [Project Structure](#-project-structure)
6. [Architecture](#-architecture)
7. [Algorithm Deep-Dive](#-algorithm-deep-dive)
8. [Dataset](#-dataset)
9. [CLI Reference](#-cli-reference)
10. [Testing](#-testing)
11. [Design Decisions](#-design-decisions)
12. [Edge Cases Handled](#-edge-cases-handled)
13. [Tech Stack](#-tech-stack)
14. [Documentation](#-documentation)

---

## 🏢 Background

This project is **Milestone 3** of the DecodeLabs Industrial Training Kit (Batch 2026).

> *"Bridge the gap between raw user data and relevant content through pure algorithmic logic."*

Before trainees build complex neural collaborative filtering models, they must master the fundamental art of **matching user profiles with item attributes using pure similarity logic**. This project proves you can implement the same core mechanism that powers Netflix/Amazon-style personalization — from scratch, without black-box libraries.

The concrete assignment: a **Tech Stack Recommender** that takes a user's raw skills and career interests and maps them to the most relevant **job roles** (e.g., Data Scientist, DevOps Engineer, Backend Developer) using job roles as the "items" in a content-based recommendation engine.

**What this proves:**
- Mastery of the **IPO model** (Input → Process → Output)
- Mastery of **vector mapping** of qualitative data into a shared numerical vocabulary space
- Mastery of **TF-IDF weighting** to differentiate generic vs. specific terms
- Mastery of **Cosine Similarity** as the scoring function and *why* it is chosen over Euclidean Distance
- Awareness of, and mitigation strategy for, the **Cold Start Problem**

---

## ⚡ Quick Start

### Prerequisites

- Python 3.11+
- pip

### Installation & Run

```bash
# 1. Clone or download the project
git clone <your-repo-url>
cd Project3Decodelabs

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run interactively (guided prompts)
python main.py

# 4. Or pass skills directly via CLI
python main.py --skills "Python" "Machine Learning" "SQL" "Statistics"

# 5. Run the full test suite
pytest tests/ -v

# 6. Open the visual dashboard in your browser
# Windows:
start dashboard/index.html
# macOS:
open dashboard/index.html
# Linux:
xdg-open dashboard/index.html
```

### Example Output

```
────────────────────────────────────────────────────────────
  🎯  TECH STACK RECOMMENDER — Results
────────────────────────────────────────────────────────────

  #1  Data Scientist
      Score  : 0.9196
      Matched: python, machine_learning, statistics, sql, pandas

  #2  Data Analyst
      Score  : 0.8312
      Matched: python, sql, statistics

  #3  Ml Engineer
      Score  : 0.7841
      Matched: python, machine_learning

────────────────────────────────────────────────────────────
  ℹ  Out-of-vocabulary terms (zero weight): (none)
```

---

## ✨ Features

| Feature | Description |
|---|---|
| **From-scratch TF-IDF** | Term Frequency + Inverse Document Frequency computed manually — no `sklearn.TfidfVectorizer` |
| **From-scratch Cosine Similarity** | Vectorized dot-product implementation — no `sklearn.metrics.pairwise` |
| **Cold Start Handling** | Detects all-OOV input and returns a labeled popularity-ranked fallback — never silently shows "Score: 0.00" |
| **Explainability** | Every recommendation shows which specific skills caused the match (`matched_skills`) |
| **Debug / Verbose Mode** | `--verbose` flag prints raw TF-IDF vector dimensions for reviewer verification |
| **Configurable Top-N** | Return any number of results, not hardcoded to 3 |
| **Input Validation** | Rejects < 3 valid skills, blank inputs, invalid `top_n` values |
| **Deterministic Output** | Alphabetical tie-breaking ensures identical output across runs (NFR-2) |
| **Modular Pipeline** | 4 independent stages, each unit-testable in isolation |
| **143 passing tests** | Unit + integration test suite with hand-computed expected values |

---

## 🔍 How It Works

```
User input: ["Python", "Machine Learning", "Statistics"]
       │
       ▼  normalize_skill() → ["python", "machine_learning", "statistics"]
       │
       ▼  data_loader.py  ──  loads raw_skills.csv → ItemRecord[]
       │
       ▼  vectorizer.py
       │     ├─ build_vocabulary()    → ["aws", "docker", ..., "sql"]  (corpus-fixed)
       │     ├─ compute_idf()         → IDF weights per term
       │     ├─ build_item_matrix()   → (10 × vocab_size) TF-IDF matrix
       │     └─ build_user_vector()   → user TF-IDF vector + OOV report
       │
       ▼  similarity_engine.py
       │     ├─ score_all_items()     → cosine similarity vs every role (full scan)
       │     └─ is_cold_start()       → True if all scores == 0.0
       │
       ▼  ranker.py
       │     ├─ rank()                → sort by (-score, role_name), slice top_n
       │     └─ apply_cold_start_fallback()  → popularity-ranked fallback if OOV
       │
       ▼  presenter.py
             └─ format_results()      → human-readable output + optional debug view

Output: Top-3 roles with score, matched skills, fallback flag
```

---

## 📁 Project Structure

```
Project3Decodelabs/
│
├── main.py                        # CLI entry point + pipeline orchestrator
├── requirements.txt               # numpy, pytest, pytest-cov
│
├── data/
│   └── raw_skills.csv             # Item dataset: 10 job roles, controlled vocab
│
├── src/                           # Core pipeline modules
│   ├── __init__.py
│   ├── data_loader.py             # Stage 1 — Ingestion (CSV → ItemRecord[])
│   ├── vectorizer.py              # Stage 2 — TF-IDF vectorization (from scratch)
│   ├── similarity_engine.py       # Stage 3 — Cosine Similarity scoring (from scratch)
│   ├── ranker.py                  # Stage 4 — Sort, truncate, cold-start fallback
│   └── presenter.py               # Output formatting + debug/verbose view
│
├── tests/                         # Full test suite (143 tests)
│   ├── conftest.py                # Shared fixtures: mini corpus, tmp CSV helpers
│   ├── test_data_loader.py        # TC-DL-* (29 tests)
│   ├── test_vectorizer.py         # TC-VEC-* (29 tests)
│   ├── test_similarity_engine.py  # TC-SIM-* (20 tests)
│   ├── test_ranker.py             # TC-RANK-* (22 tests)
│   ├── test_presenter.py          # TC-PRES-* (15 tests)
│   └── test_integration.py        # TC-INT-* + TC-EDGE-* (28 tests)
│
├── dashboard/                     # Interactive visual dashboard (browser)
│   ├── index.html
│   ├── style.css
│   ├── data.js                    # Browser-side TF-IDF engine mirror
│   └── app.js                     # UI controller
│
└── docs/
    ├── PRD.md                     # Product Requirements Document (v1.1)
    ├── ARCHITECTURE.md            # Module architecture + data flow diagrams
    ├── ALGORITHM_SPEC.md          # TF-IDF + Cosine worked examples (hand-computed)
    ├── DATA_SCHEMA.md             # CSV format, validation rules, vocab policy
    └── TEST_PLAN.md               # Full test case specification (60+ cases)
```

---

## 🏗 Architecture

The pipeline follows strict **IPO (Input → Process → Output)** separation. Each module has one responsibility and is independently unit-testable.

```
┌─────────────────────┐
│   raw_skills.csv    │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────┐
│   data_loader.py     │  Stage 1: INGESTION
│  - parse CSV         │  Raises DataLoadError on missing/empty file
│  - validate rows     │  Skips malformed rows (logs warning, continues)
│  - normalize text    │  normalize_skill(): lowercase + spaces→underscores
└──────────┬───────────┘
           │  List[ItemRecord]
           ▼
┌──────────────────────┐
│   vectorizer.py      │  Stage 2: VECTORIZE (Process, part 1)
│  - build_vocabulary  │  Corpus-fixed vocabulary (user cannot inject new terms)
│  - compute_tf        │  count(t)/total_terms — OOV terms count toward denominator
│  - compute_idf       │  log(N / df(t)) — no smoothing needed (vocab ⊆ corpus)
│  - build_item_matrix │  Shape: (num_items × vocab_size)
│  - build_user_vector │  Returns vector + OOV list (never mutates vocab)
└──────────┬───────────┘
           │  (item_matrix, user_vector, vocab, idf)
           ▼
┌──────────────────────┐
│ similarity_engine.py │  Stage 3: SCORE (Process, part 2)
│  - cosine_similarity │  dot(A,B) / (‖A‖·‖B‖), returns 0.0 on zero norm
│  - score_all_items   │  Vectorized matrix multiply — O(n·v) single pass
│  - is_cold_start     │  True if every score == 0.0
└──────────┬───────────┘
           │  List[ScoredItem]
           ▼
┌──────────────────────┐
│    ranker.py         │  Stage 4a: SORT + FILTER
│  - rank()            │  Sort key: (-score, role_name) for deterministic output
│  - cold_start_fallback│ popularity_rank ordering, is_fallback=True tagged
└──────────┬───────────┘
           │  List[ScoredItem] (top_n)
           ▼
┌──────────────────────┐
│   presenter.py       │  Stage 4b: OUTPUT
│  - format_results()  │  Role + score + matched_skills + fallback label
│  - format_debug_view │  Raw vector dimensions for reviewer verification
└──────────────────────┘
```

---

## 📐 Algorithm Deep-Dive

All math is implemented from scratch in `src/vectorizer.py` and `src/similarity_engine.py`. No `sklearn`, no `gensim`, no hidden abstractions.

### Term Frequency (TF)

```
TF(term, document) = count(term in document) / total_terms(document)
```

- **Denominator** = raw input length, **including OOV terms** (per `ALGORITHM_SPEC.md §3.7`)
- **Duplicate skills intentionally increase TF** (PRD §10.1) — typing "Python" twice is a stronger signal
- Applied to both item skill lists and the user's input list

### Inverse Document Frequency (IDF)

```
IDF(term) = log( N / df(term) )
```

- `N` = total number of job roles in the corpus
- `df(term)` = number of roles whose skill list contains that term
- Computed **once at load time** across the corpus only (vocabulary is corpus-fixed — PRD §10.4)
- No smoothing needed: vocabulary is built FROM the corpus, so `df ≥ 1` is guaranteed

**Effect:** Generic skills that appear across many roles (e.g., `python`, `sql`) get penalized with lower IDF weights. Rare/specific skills (e.g., `penetration_testing`, `mlops`) get higher weights.

### TF-IDF Weight

```
weight(term, document) = TF(term, document) × IDF(term)
```

### Cosine Similarity

```
cos(θ) = (A · B) / (‖A‖ × ‖B‖)
```

- **Zero-vector guard:** returns `0.0` if either norm is zero — no `ZeroDivisionError`
- **Why cosine over Euclidean?** Euclidean distance is sensitive to vector magnitude. A role with 10 skill tags has a larger-magnitude vector than one with 3, even if their skill *proportions* are identical. Cosine normalizes by `‖A‖·‖B‖`, measuring orientation only — the relative "shape" of the skill profile, not its size.
- **Why cosine over raw Jaccard/binary overlap?** Binary overlap treats `"python"` (generic, appears in 8/10 roles) identically to `"pytorch"` (specific, appears in 1/10). TF-IDF weights differentiate them.

### Worked Example (from `ALGORITHM_SPEC.md §3`)

**User input:** `["Python", "Machine Learning", "Statistics"]`  
**After normalization:** `["python", "machine_learning", "statistics"]` (statistics is OOV)

| Role | Score | Calculation |
|---|---|---|
| **Data Scientist** | **0.9449** | High overlap on `python` + `machine_learning` (high IDF) |
| Backend Developer | 0.1133 | Only `python` overlaps (lower-IDF shared term) |
| DevOps Engineer | 0.0000 | Zero vector overlap → dot product = 0 |

### Cold Start Handling

When **all user skills are out-of-vocabulary**, the user vector is `[0,0,...,0]`. Every cosine score mathematically resolves to `0.0` (zero dot product). Instead of presenting three "Score: 0.00" rows as fake recommendations:

1. `similarity_engine.is_cold_start()` detects the condition
2. `ranker.apply_cold_start_fallback()` returns items ordered by `popularity_rank`
3. The output is clearly labeled `⚠ TRENDING FALLBACK` — not a personalized match

---

## 📊 Dataset

The item corpus is `data/raw_skills.csv` — 10 job roles with pipe-delimited skill tags and popularity ranks.

| Role | Skills (sample) | Rank |
|---|---|---|
| Data Scientist | python, sql, machine_learning, statistics, pandas, numpy, data_visualization | 1 |
| Backend Developer | java, python, sql, rest_api, databases, git, spring_boot | 2 |
| DevOps Engineer | aws, docker, kubernetes, linux, ci_cd, terraform, git | 3 |
| Frontend Developer | javascript, html, css, react, typescript, ui_ux, git | 4 |
| Full Stack Developer | javascript, python, react, node.js, sql, rest_api, git | 5 |
| ML Engineer | python, machine_learning, deep_learning, tensorflow, pytorch, numpy, mlops | 6 |
| Cloud Architect | aws, azure, gcp, kubernetes, terraform, networking, security | 7 |
| Cybersecurity Analyst | networking, security, linux, firewalls, penetration_testing, cryptography | 8 |
| Data Analyst | sql, python, data_visualization, excel, tableau, statistics, pandas | 9 |
| Mobile Developer | kotlin, swift, java, react_native, ui_ux, git, rest_api | 10 |

**CSV format:** `role_name,skills,popularity_rank` where skills are `|`-delimited.

**Vocabulary policy:** All skill tokens are lowercase, underscore-joined (e.g., `machine_learning`). Multi-word user inputs like `"Machine Learning"` are normalized to `"machine_learning"` by `normalize_skill()`. Synonym mapping is explicitly out of scope (deferred to neural/embedding matching in a future project).

**Item Cold Start robustness:** Because this engine is purely content-based, a new role can be added to the CSV with only its skill tags and it immediately participates in scoring — no retraining, no warm-up period.

---

## 💻 CLI Reference

```
usage: tech_stack_recommender [-h] [--skills SKILL [SKILL ...]]
                              [--dataset PATH] [--top-n N] [--verbose]
```

| Flag | Short | Default | Description |
|---|---|---|---|
| `--skills` | `-s` | *(interactive)* | One or more skill keywords. Minimum 3 non-blank required. |
| `--dataset` | `-d` | `data/raw_skills.csv` | Path to the CSV item dataset. |
| `--top-n` | `-n` | `3` | Number of top results to return. Must be ≥ 1. |
| `--verbose` | `-v` | `False` | Print raw TF-IDF vector dimensions + debug info for reviewer verification. |

### Examples

```bash
# Interactive mode (guided prompts)
python main.py

# Explicit skills via flags
python main.py --skills "Python" "Machine Learning" "SQL" "Statistics" "Pandas"

# DevOps profile, top 5 results
python main.py --skills "AWS" "Docker" "Kubernetes" --top-n 5

# Cold Start demo (all skills OOV → trending fallback)
python main.py --skills "Photography" "Painting" "Sculpture"

# Verbose debug mode — shows TF-IDF vector for reviewer verification
python main.py --skills "Python" "Deep Learning" "PyTorch" --verbose

# Custom dataset
python main.py --skills "Python" "SQL" "Pandas" --dataset /path/to/my_roles.csv

# View the interactive dashboard
# Windows: start dashboard/index.html
# macOS:   open dashboard/index.html
# Linux:   xdg-open dashboard/index.html
```

### Input Validation

| Condition | Behavior |
|---|---|
| Fewer than 3 non-blank skills | `InputValidationError` raised; pipeline does not run |
| Blank / whitespace-only strings | Filtered out before counting toward the 3-minimum |
| `top_n ≤ 0` | `ValueError` raised |
| Dataset file missing | `DataLoadError` raised with the path shown |
| Dataset file empty (header only) | `DataLoadError` raised |

---

## 🧪 Testing

The project ships with **143 unit and integration tests** that must all pass before DecodeLabs review (PRD §9 — 100% pass rate required).

### Run the suite

```bash
# Full suite (verbose)
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Single module
pytest tests/test_vectorizer.py -v

# By keyword (e.g., all cold-start tests)
pytest tests/ -v -k cold_start

# Stop on first failure
pytest tests/ -v -x
```

### Test Coverage Matrix

| File | Cases | What it covers |
|---|---|---|
| `test_data_loader.py` | 29 | CSV loading, row validation, `normalize_text`, `normalize_skill`, error handling |
| `test_vectorizer.py` | 29 | Vocabulary, TF, IDF, item matrix, user vector, OOV, vocab immutability |
| `test_similarity_engine.py` | 20 | Cosine formula, zero-vector guard, full scan, matched_skills, cold-start detection |
| `test_ranker.py` | 22 | Sort order, tie-breaking, truncation, popularity fallback, `top_n` validation |
| `test_presenter.py` | 15 | Output formatting, fallback labeling, debug view |
| `test_integration.py` | 28 | End-to-end pipeline, all acceptance criteria, determinism, edge cases |
| **Total** | **143** | **All PRD qualification criteria** |

### Key test assertions (from `ALGORITHM_SPEC.md` worked example)

```python
# Hand-computed cosine similarity (±1e-3 rel tolerance, spec uses rounded intermediates)
assert cosine_similarity(data_scientist_vector, user_vector) ≈ 0.9449

# IDF values match formula exactly
assert idf["python"]  ≈ log(3/2) = 0.4055   # appears in 2/3 items
assert idf["aws"]     ≈ log(3/1) = 1.0986   # appears in 1/3 items

# Cold Start produces fallback, not misleading zero scores
results = recommend(["Photography", "Painting", "Sculpture"], ...)
assert all(r.is_fallback for r in results)
assert all(r.score is None for r in results)

# Determinism (NFR-2)
run1 = recommend(skills, path)
run2 = recommend(skills, path)
assert [(r.role_name, r.score) for r in run1] == [(r.role_name, r.score) for r in run2]
```

---

## 🎯 Design Decisions

These decisions are documented in `PRD.md §10` with full rationale.

| Decision | Policy | Rationale |
|---|---|---|
| **Duplicate skills increase TF** | `["Python","Python","Python"]` → TF=1.0 | Matches the literal TF formula; repeated input = stronger signal (PRD §10.1) |
| **Score ties broken alphabetically** | Sort key: `(-score, role_name)` | Guarantees deterministic output across runs (PRD §10.2, NFR-2) |
| **Vocabulary is corpus-fixed** | User input cannot inject new vocab terms | Prevents IDF instability across queries; OOV terms logged but not added (PRD §10.4) |
| **Normalization: spaces → underscores for skills** | `"Machine Learning"` → `"machine_learning"` | Matches the dataset's controlled vocabulary token format (`ALGORITHM_SPEC.md §3.7`) |
| **OOV denominator includes all input tokens** | `TF("python", ["python","ml","statistics"]) = 1/3` | The literal deck formula uses `total_terms(d)` = raw input length (ALGORITHM_SPEC.md §3.7) |
| **No sklearn, no gensim** | All TF-IDF + cosine math hand-coded | Proves understanding of the math (ARCHITECTURE.md Principle #2) |

---

## ⚠️ Edge Cases Handled

| Edge Case | Behavior |
|---|---|
| Fewer than 3 valid (non-blank) inputs | `InputValidationError`; pipeline does not run |
| Blank / whitespace-only skill strings | Filtered and not counted toward the 3-minimum |
| Duplicate skills in input | Allowed — increase TF weight (documented behavior per PRD §10.1) |
| OOV user skill (not in dataset vocabulary) | Logged + included in OOV report; zero weight; does not crash |
| All input skills OOV | Cold Start fallback triggered; results labeled `is_fallback=True` |
| Dataset has fewer rows than `top_n` | Returns all available rows; no padding or error |
| Empty/missing dataset file | `DataLoadError` with actionable message |
| CSV row with blank/missing skills | Row skipped with logged warning; pipeline continues |
| CSV row with invalid `popularity_rank` | Row skipped with logged warning; pipeline continues |
| `top_n ≤ 0` | `ValueError` before any pipeline work |
| Non-string skill input | Coerced to string via `normalize_skill()`; does not crash |
| Tied similarity scores | Stable secondary sort by role name (alphabetical ascending) |

---

## 🔧 Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Matches the training track standard |
| Numeric ops | `numpy` | Array storage + dot products only — **not** for similarity logic |
| CSV parsing | Python `csv` module | Standard library, no extra dependency |
| Testing | `pytest` + `pytest-cov` | Industry standard; supports parametrized assertions against hand-computed values |
| Similarity math | **From scratch** | The entire learning objective — TF, IDF, Cosine Similarity coded manually |
| UI dashboard | Vanilla HTML/CSS/JS | No framework needed; self-contained browser demo |

**Explicitly NOT used (per `ARCHITECTURE.md` Principle #2):**
- `sklearn.TfidfVectorizer` — would hide the TF-IDF math
- `sklearn.metrics.pairwise.cosine_similarity` — would hide the cosine math
- Any collaborative filtering library
- Any neural/embedding library

---

## 📚 Documentation

All companion documents live in `docs/`:

| Document | Purpose |
|---|---|
| [`docs/PRD.md`](docs/PRD.md) | Product Requirements Document — all functional/non-functional requirements, acceptance criteria, edge cases, open decisions |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module architecture, data flow diagrams, component contracts, error handling strategy |
| [`docs/ALGORITHM_SPEC.md`](docs/ALGORITHM_SPEC.md) | Step-by-step TF-IDF + Cosine formulas with a fully worked 3-item example — the canonical fixture for unit tests |
| [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md) | CSV format, column definitions, validation rules, controlled vocabulary policy |
| [`docs/TEST_PLAN.md`](docs/TEST_PLAN.md) | Full test case specification (60+ cases with expected values) |

---

## 🏆 Qualification Criteria Checklist

Per DecodeLabs' badge unlock requirements:

- [x] **IPO model** — Ingestion / Processing / Output are physically separate modules
- [x] **Vector mapping** — qualitative skill strings → numeric TF-IDF vectors
- [x] **TF-IDF weighting** — TF and IDF computed from scratch, both item and user vectors
- [x] **Cosine Similarity** — implemented from scratch with magnitude-invariance rationale
- [x] **Cold Start awareness** — detected, labeled, routed to popularity fallback
- [x] **Explainability** — `matched_skills` on every result; `--verbose` debug view
- [x] **Modular** — 4 pipeline stages, each independently unit-testable (NFR-6)
- [x] **100% test pass rate** — 143 tests green
- [x] **Demo runs** — 3+ distinct sample queries produce distinct, sensible Top-3 outputs

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

**DecodeLabs Industrial Training Kit · Batch 2026 · Project 3**  
*Content-Based Filtering · TF-IDF · Cosine Similarity · Built from scratch*

</div>

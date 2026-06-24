# Data Schema Document
## Tech Stack Recommender — `raw_skills.csv`

**Companion to:** `PRD.md §6`, `ARCHITECTURE.md §3.1` | **See also:** `ALGORITHM_SPEC.md`

---

## 1. File Overview

The item dataset (`raw_skills.csv`) is the **static corpus** powering the recommendation engine.
It defines every candidate job role and its associated skill vocabulary.

- **Format:** UTF-8 CSV with a header row.
- **Location (default):** `data/raw_skills.csv` (configurable via CLI `--dataset` flag).
- **Immutable during a session:** The vocabulary and IDF table are fixed at load time (PRD §10.4).
  A new role can be added between sessions by appending a row and re-running the engine.

---

## 2. Column Definitions

| Column | Type | Required | Notes |
|---|---|---|---|
| `role_name` | string | **Yes** | Human-readable job role label (e.g., `"Data Scientist"`). Displayed in output. Must be non-empty after stripping whitespace. |
| `skills` | string | **Yes** | Pipe-delimited (`\|`) list of skill tokens for this role (e.g., `"python\|sql\|machine_learning"`). Normalized to lowercase + trimmed during ingestion (PRD §10.3). Must be non-empty. |
| `popularity_rank` | int | **Yes** | Integer rank (lower = more popular/default). Used **only** for Cold Start fallback ordering (FR-6). Must be a positive integer. Roles with rank 1 appear first in the fallback list. |

### Why pipe-delimited skills?
Commas inside skill names (e.g., `"C, C++"`) would break naive CSV parsing.
Pipe (`|`) is guaranteed absent from standard tech skill names and survives round-trips
through `csv.reader` and `pandas.read_csv` without quoting issues.

---

## 3. Vocabulary / Controlled Vocabulary Policy

Per PRD §13 Risk table: free-text skill fields risk vocabulary fragmentation
(e.g., `"Web Design"` vs `"Frontend Development"` as synonyms). This schema
mitigates that risk by using a **controlled skill token vocabulary**:

- All skill tokens are **lowercase, single-word or underscore-joined** (e.g., `machine_learning`, not `Machine Learning`).
- The ingestion layer normalizes at load time (lowercases + strips whitespace + collapses spaces to underscores within a token where applicable).
- **Synonym mapping is explicitly out of scope** for this milestone (PRD §2.3).
  The dataset author is responsible for consistent token spelling.

### Canonical token examples
| Raw author input | Normalized token |
|---|---|
| `Python` | `python` |
| `Machine Learning` | `machine_learning` |
| `AWS` | `aws` |
| `Node.js` | `node.js` |
| ` SQL ` | `sql` |

---

## 4. Sample Dataset (minimum viable corpus — 10 roles)

The dataset must contain **≥ 5 distinct roles** per PRD §13 to produce observable
score differentiation. This schema ships with **10 roles** for richer testing.

```
role_name,skills,popularity_rank
Data Scientist,python|sql|machine_learning|statistics|pandas|numpy|data_visualization,1
Backend Developer,java|python|sql|rest_api|databases|git|spring_boot,2
DevOps Engineer,aws|docker|kubernetes|linux|ci_cd|terraform|git,3
Frontend Developer,javascript|html|css|react|typescript|ui_ux|git,4
Full Stack Developer,javascript|python|react|node.js|sql|rest_api|git,5
ML Engineer,python|machine_learning|deep_learning|tensorflow|pytorch|numpy|mlops,6
Cloud Architect,aws|azure|gcp|kubernetes|terraform|networking|security,7
Cybersecurity Analyst,networking|security|linux|firewalls|penetration_testing|cryptography,8
Data Analyst,sql|python|data_visualization|excel|tableau|statistics|pandas,9
Mobile Developer,kotlin|swift|java|react_native|ui_ux|git|rest_api,10
```

---

## 5. Validation Rules (enforced by `data_loader.py`)

| Rule | Enforced By | Failure Behavior |
|---|---|---|
| File must exist at the given path | `load_items()` | Raises `DataLoadError` with path shown |
| File must have ≥ 1 data row (beyond header) | `load_items()` | Raises `DataLoadError` |
| `role_name` must be non-empty | `validate_row()` | Row skipped, warning logged |
| `skills` column must be non-empty | `validate_row()` | Row skipped, warning logged |
| At least 1 skill token non-blank after splitting on `\|` | `validate_row()` | Row skipped, warning logged |
| `popularity_rank` must be a positive integer | `validate_row()` | Row skipped, warning logged |
| Skill tokens normalized (lowercase + strip) before ingestion | `normalize_text()` | Applied silently; original CSV unmodified |

---

## 6. Item Cold Start Robustness

Per PRD FR-6 (Item Cold Start): because this engine is content-based,
**a new role can be added to `raw_skills.csv` with only its skill tags**
(no user interaction history required) and it will immediately participate
in vocabulary-building and vector scoring on the next run.
This is the architectural proof that content-based filtering is inherently
robust to Item Cold Start — no retraining, no warm-up period.

---

## 7. Extension Notes (out of scope for this milestone)

- **Multi-value weights:** future versions could add a `skill_weights` column
  so an expert-curated importance score augments the TF-IDF weight.
- **Category tags:** a `category` column (e.g., `"Engineering"`, `"Data"`)
  could support pre-filtering before similarity scoring.
- Both extensions require only a schema update and a `data_loader.py`
  change — no changes to `vectorizer.py`, `similarity_engine.py`, or `ranker.py`.

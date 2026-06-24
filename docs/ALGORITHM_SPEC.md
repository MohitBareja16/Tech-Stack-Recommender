# Algorithm Specification
## TF-IDF Vectorization + Cosine Similarity — Worked Reference

**Companion to:** `ARCHITECTURE.md §3.2–3.3` | **Source concepts:** training deck slides
"Bridging the Language Barrier," "The Limitation of Binary Overlap," "Upgrading Feature
Extraction with TF-IDF," "Cosine Similarity: The Industry Standard"

This document exists so that an implementer (or grader) can verify the code against
**hand-computed expected values**, satisfying PRD NFR-1.

---

## 1. Why not binary overlap / Jaccard alone?

Per the deck's "Limitation of Binary Overlap" slide: counting raw tag overlap (1=present,
0=absent) treats a generic tag (e.g., "code") identically to a highly specific one (e.g.,
"tensors"). Two unrelated profiles can both show "3 overlapping tags" and look equally
similar even though one overlap is meaningless. TF-IDF fixes this by **down-weighting**
tags that appear in many items and **up-weighting** tags that are rare/specific.

---

## 2. Step-by-step formulas

### 2.1 Term Frequency (TF)
For a document `d` (either an item's skill list or the user's input list) and term `t`:

```
TF(t, d) = count(t appears in d) / (total terms in d)
```

### 2.2 Inverse Document Frequency (IDF)
Computed once, across the **item corpus only** (per Architecture §3.2 — corpus is fixed,
user input never alters IDF):

```
IDF(t) = log( N / df(t) )
```//
- `N` = total number of items in the corpus
- `df(t)` = number of items whose skill list contains `t` at least once

> **Implementation note:** to avoid `log(N/0)` / division-by-zero for a term that, by
> construction, only ever comes from the corpus itself, `df(t) >= 1` is guaranteed for
> every vocabulary term (vocabulary is built FROM the corpus — see Architecture §3.2).
> No smoothing constant is required for this reason, but implementers MAY add Laplace
> smoothing (`+1` to numerator and denominator) as a documented, optional defensive
> measure — not a requirement.

### 2.3 TF-IDF weight
```
weight(t, d) = TF(t, d) * IDF(t)
```

### 2.4 Cosine Similarity
For vectors **A** (user) and **B** (item), both in the shared vocabulary space:

```
cos(θ) = (A · B) / (||A|| * ||B||)
```

**Zero-vector guard (Cold Start):** if `||A|| == 0` or `||B|| == 0`, define
`cos(θ) = 0.0` by convention (rather than raising a divide-by-zero error). This is the
direct implementation of the deck's "Cold Start" slide: *"Multiplying any item vector by
a vector of zeros results in zero."*

---

## 3. Worked Example

### 3.1 Mini corpus (3 items)

| Item | Skills (already normalized) |
|---|---|
| Data Scientist | python, sql, machine_learning |
| DevOps Engineer | aws, docker, kubernetes |
| Backend Developer | java, python, sql |

### 3.2 Vocabulary (alphabetical, fixed order for reproducibility)
```
["aws", "docker", "java", "kubernetes", "machine_learning", "python", "sql"]
```
(7 dimensions, `N = 3` items)

### 3.3 Document Frequency (df) per term
| term | df(t) |
|---|---|
| aws | 1 |
| docker | 1 |
| java | 1 |
| kubernetes | 1 |
| machine_learning | 1 |
| python | 2 |
| sql | 2 |

### 3.4 IDF per term — `IDF(t) = log(3 / df(t))` (natural log)
| term | IDF |
|---|---|
| aws | log(3/1) = 1.0986 |
| docker | 1.0986 |
| java | 1.0986 |
| kubernetes | 1.0986 |
| machine_learning | 1.0986 |
| python | log(3/2) = 0.4055 |
| sql | 0.4055 |

This is exactly the deck's intended effect: **python/sql** (appear in 2/3 items) get a
*lower* weight than **aws/docker/java/kubernetes/machine_learning** (each unique to 1
item) — generic-within-this-corpus terms are penalized, rare/specific ones rewarded.

### 3.5 TF for "Data Scientist" (3 terms total, each appears once)
`TF = 1/3 = 0.3333` for python, sql, machine_learning; `0` for the other 4 terms.

### 3.6 TF-IDF vector for "Data Scientist"
```
python:            0.3333 * 0.4055 = 0.1352
sql:               0.3333 * 0.4055 = 0.1352
machine_learning:  0.3333 * 1.0986 = 0.3662
(all others = 0)

vector = [0, 0, 0, 0, 0.3662, 0.1352, 0.1352]
  order: [aws, docker, java, kubernetes, machine_learning, python, sql]
```

### 3.7 User input: `["Python", "Machine Learning", "Statistics"]`
After normalization: `["python", "machine_learning", "statistics"]`.
- `"statistics"` is **OOV** (not in vocabulary) → logged, contributes 0 dimensions.
- Valid terms for TF purposes: 2 out of 3 raw inputs map into vocab
  (`python`, `machine_learning`); per Architecture §3.2, TF denominator uses
  the **count of all input tokens including OOV** (`3`), matching the
  literal deck formula `count(t)/total_terms(d)` where `total_terms` = the
  user's raw input length, not the post-filter length.
  - `TF(python) = 1/3 = 0.3333`
  - `TF(machine_learning) = 1/3 = 0.3333`
  - `TF(statistics)` → not represented (no vocabulary dimension exists for it)

### 3.8 User TF-IDF vector (reusing corpus IDF table)
```
python:           0.3333 * 0.4055 = 0.1352
machine_learning: 0.3333 * 1.0986 = 0.3662
(all others = 0)

user_vector = [0, 0, 0, 0, 0.3662, 0.1352, 0]
  order: [aws, docker, java, kubernetes, machine_learning, python, sql]
```

### 3.9 Cosine similarity vs. "Data Scientist"
```
A = [0, 0, 0, 0, 0.3662, 0.1352, 0.1352]   (Data Scientist)
B = [0, 0, 0, 0, 0.3662, 0.1352, 0]         (User)

A · B = (0.3662*0.3662) + (0.1352*0.1352) + (0.1352*0)
      = 0.1341 + 0.0183 + 0
      = 0.1524

||A|| = sqrt(0.3662^2 + 0.1352^2 + 0.1352^2) = sqrt(0.1341+0.0183+0.0183) = sqrt(0.1707) = 0.4132
||B|| = sqrt(0.3662^2 + 0.1352^2)            = sqrt(0.1341+0.0183)         = sqrt(0.1524) = 0.3904

cos(θ) = 0.1524 / (0.4132 * 0.3904) = 0.1524 / 0.1613 = 0.9448
```

**Expected test assertion:** `cosine_similarity(user_vector, data_scientist_vector) ≈ 0.9448` (±1e-4).

### 3.10 Cosine similarity vs. "DevOps Engineer" and "Backend Developer"
- DevOps Engineer vector = `[0.3662, 0.3662, 0, 0.3662, 0, 0, 0]` → dot product with
  user vector = 0 (zero orientation overlap) → **score = 0.0**.
- Backend Developer vector: skills = java, python, sql →
  `TF=1/3` each → weights: java `0.3662`, python `0.1352`, sql `0.1352`.
  Dot with user = `(0.1352*0.1352) = 0.0183`.
  `||Backend|| = sqrt(0.3662^2+0.1352^2+0.1352^2) = 0.4132`.
  `cos(θ) = 0.0183 / (0.4132*0.3904) = 0.0183/0.1613 = 0.1135`.

### 3.11 Expected ranked output (top_n=3)
| Rank | Role | Score |
|---|---|---|
| 1 | Data Scientist | 0.9448 |
| 2 | Backend Developer | 0.1135 |
| 3 | DevOps Engineer | 0.0000 |

This worked example is the **canonical fixture** referenced by `TEST_PLAN.md`.

---

## 4. Why Cosine over Euclidean (design rationale, for the explainability requirement)

Per the deck's "Why Euclidean Distance Fails at Scale" slide: Euclidean distance is
sensitive to vector *magnitude*. A job role described with many tags (long skill list)
will have a larger-magnitude vector even if its *direction* (the relative proportion of
skills) is nearly identical to a sparser profile — Euclidean would incorrectly judge them
as dissimilar. Cosine Similarity normalizes by vector length (`||A||*||B||` in the
denominator), so it measures **orientation/direction only** — i.e., the relative
"shape" of the skill profile, not its raw size. This is why the architecture mandates
Cosine Similarity as the **only** scoring function (Architecture §3.3) rather than
offering Euclidean as a togglable alternative.

---

## 5. Cold Start — math walkthrough

If a user enters `["Photography", "Painting", "Sculpture"]` against the mini corpus
above, all 3 terms are OOV. The resulting user vector is `[0,0,0,0,0,0,0]` — literally
the zero vector described in the deck's Cold Start slide. Every cosine similarity
computation against this vector returns `0.0` for every item (guarded division, §2.4),
which `similarity_engine.is_cold_start()` detects (Architecture §3.3) and routes to the
`popularity_rank`-based fallback list instead of presenting three "0.00 score" results as
genuine personalized matches.
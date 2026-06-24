# Product Requirements Document (PRD)
## Project 3 — AI Recommendation Logic: "Tech Stack Recommender"
**Org:** DecodeLabs Industrial Training Kit | **Batch:** 2026
**Doc owner:** AI Engineering Trainee | **Status:** Draft v1.1 (post gap-review)
**Related docs:** `ARCHITECTURE.md`, `ALGORITHM_SPEC.md`, `DATA_SCHEMA.md`, `TEST_PLAN.md`, `README.md`

---

## 1. Background & Problem Statement

DecodeLabs' Project 3 milestone requires trainees to build a **content-based
recommendation engine** before progressing to neural collaborative filtering
in later tracks. The source material (training deck) frames the problem as:

> "Bridge the gap between raw user data and relevant content through pure
> algorithmic logic" — using **Input → Process → Output (IPO)** architecture,
> **TF-IDF feature weighting**, and **Cosine Similarity** scoring.

The concrete capstone assignment is a **Tech Stack Recommender**: a tool that
takes a user's raw skills and career interests and maps them to the most
relevant **job role** (e.g., Data Scientist, DevOps Engineer, Backend
Developer), using job roles as the "items" in a recommendation engine.

### Why this matters (business framing from source material)
Recommendation engines combat "choice overload" and are the same core
mechanism that powers Netflix/Amazon-style personalization — connecting
users to relevant outcomes before they explicitly articulate them. This
project proves the trainee can build that mechanism from first principles,
without relying on a pre-built ML library to do the similarity math for them.

---

## 2. Goals

### 2.1 Primary Goal
Build a working, explainable, content-based recommendation system that:
1. Accepts user skill/interest input.
2. Matches those inputs against a structured item dataset using
   similarity logic (TF-IDF vectors + Cosine Similarity).
3. Returns a ranked, truncated list of the most relevant job roles.

### 2.2 Learning Goals (qualification criteria, from source deck)
- Demonstrate mastery of the **IPO model** (Input–Process–Output).
- Demonstrate mastery of **vector mapping** of qualitative data into a
  shared numerical vocabulary space.
- Demonstrate mastery of **TF-IDF weighting** to differentiate generic vs.
  specific terms.
- Demonstrate mastery of **Cosine Similarity** as the scoring function and
  articulate *why* it is chosen over Euclidean Distance and over raw
  binary/Jaccard overlap.
- Demonstrate awareness of, and a mitigation strategy for, the **Cold Start
  Problem**.

### 2.3 Non-Goals (explicitly out of scope)
- Collaborative filtering (user-user or item-item behavioral patterns).
  The source material is explicit: *"Project 3 focuses exclusively on
  Content-Based Filtering... avoiding the need for massive historical user
  datasets."*
- Neural/embedding-based semantic matching (deferred to a later project per
  the deck's framing — "before you build complex neural collaborative
  filtering models").
- Real user authentication, persistence across sessions, or a production
  database. This is a training milestone, not a deployed product.
- Multi-language support. English skill terms only.

---

## 3. Users & Personas

| Persona | Description | Need |
|---|---|---|
| **The Learner (primary user)** | Trainee running the tool to validate their own implementation | Clear, deterministic, debuggable output that proves the similarity math works |
| **The Reviewer (grader)** | DecodeLabs mentor verifying the qualification criteria | Ability to inspect intermediate steps (vectors, scores) — not just final output, to verify "quality," per the Qualification Criteria slide |
| **The End User (simulated)** | A job-seeker entering skills | A simple, fast, "talk to me in plain English" interface returning believable role recommendations |

---

## 4. Functional Requirements

### FR-1: Input Capture (Ingestion)
- The system **MUST** accept a minimum of **3 user skill/interest inputs**
  (explicit requirement from "Pipeline Steps 1 & 2" slide: *"your script must
  accept a minimum of three user inputs to ensure sufficient data density
  for accurate matching"*).
- The system **MUST** support more than 3 inputs (no fixed upper cap, but
  see NFR-3 for a practical bound).
- The system **MUST** validate that each input is a non-empty string and
  reject (with a clear message) inputs that are blank/whitespace-only.
- The system **MUST** normalize input casing and surrounding whitespace
  (e.g., `" Python "` → `"python"`) before vectorization, because raw
  string mismatches break the shared-vocabulary requirement called out on
  the "Bridging the Language Barrier" slide.

### FR-2: Vocabulary & Vector Mapping
- The system **MUST** build a single shared vocabulary derived from the
  union of: (a) every skill tag across all job roles in the item dataset,
  and (b) the user's input skills.
- Any user-input skill **not present** in the item dataset's vocabulary
  **MUST** be handled gracefully (see FR-6 — Out-of-Vocabulary policy) rather
  than silently dropped without explanation, and rather than crashing.
- The system **MUST** map both user input and each item's skill list into
  numeric vectors over this shared vocabulary.

### FR-3: TF-IDF Weighting
- The system **MUST** compute **Term Frequency (TF)** per item:
  `TF(t, d) = count(t in d) / total_terms(d)`.
- The system **MUST** compute **Inverse Document Frequency (IDF)** across
  the full item corpus: `IDF(t) = log(N / df(t))`, where `N` = total number
  of items (job roles) and `df(t)` = number of items containing term `t`.
- The system **MUST** weight each vector dimension as `TF × IDF`.
- The system **MUST** treat the user's input as a "document" for the
  purpose of generating its own TF vector (frequency of each skill in the
  user's input list), reusing the corpus-derived IDF weights.

### FR-4: Similarity Scoring (Process)
- The system **MUST** use **Cosine Similarity** — not Euclidean Distance,
  not raw Jaccard/binary overlap — as the final scoring function, per the
  explicit rationale in the deck (magnitude-invariance, "industry
  standard").
- The system **MUST** compute a similarity score for the user vector
  against **every** item vector in the dataset (full scan, no
  pre-filtering that could silently exclude valid items).
- Scores **MUST** fall in `[0, 1]` given TF-IDF's non-negative output (the
  deck notes the practical cosine range sits in `[0,1]`, not the full
  `[-1,1]`, because TF-IDF weights cannot be negative).

### FR-5: Ranking & Output (Sorting + Filtering)
- The system **MUST** sort items by descending similarity score.
- The system **MUST** truncate to a **Top-N list** (default N=3, per the
  "Tech Stack Recommender" capstone spec: *"returns the Top 3 most relevant
  career paths"*). N **MUST** be configurable, not hardcoded as a magic
  number buried in logic.
- The system **MUST** display, for each recommended item: role name, final
  similarity score (rounded for readability), and the matched/overlapping
  skills (for explainability — this also satisfies the Reviewer persona's
  need to verify quality).
- If the dataset has fewer items than N, the system **MUST** return all
  available items rather than erroring.

### FR-6: Cold Start Handling
- **User Cold Start:** If 100% of a user's input skills are out-of-vocabulary
  (zero-overlap with every item), every cosine score will mathematically
  resolve to 0 (zero vector ⋅ anything = 0; this is mathematically expected
  per the "Cold Start" slide, not a bug). In this case the system **MUST**
  detect the zero-vector / all-zero-score condition and respond with a
  **Trending Fallback**: a popularity/default ranked list (e.g., dataset
  order or a curated `popularity_rank` column), clearly labeled as a
  fallback, instead of presenting all-zero scores as if they were
  meaningful matches.
- **Item Cold Start:** Not directly testable by the end user in this scope
  (item set is static per session), but the data layer **MUST** be designed
  so a new item with only metadata (no interaction history) can be added to
  the CSV and immediately be eligible for matching — proving the
  content-based approach is inherently robust to this case, per the deck.

### FR-7: Explainability / Transparency
- The system **SHOULD** expose (via a verbose/debug mode or a documented
  intermediate output) the constructed user vector and the top-matching
  item vectors, so a Reviewer can manually verify the cosine math.
- The system **SHOULD** log (not silently swallow) which user-input skills
  were out-of-vocabulary, with a message rather than failing silently.

---

## 5. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 (Correctness) | Cosine similarity, TF, and IDF calculations must be unit-testable against hand-computed expected values (see `TEST_PLAN.md`). No floating point discrepancy beyond 1e-6 tolerance. |
| NFR-2 (Determinism) | Given identical input + identical dataset, output must be byte-for-byte identical across runs. No randomness anywhere in the pipeline. |
| NFR-3 (Performance) | Must score against a dataset of up to 10,000 items in under 2 seconds on a standard laptop CPU, using vectorized operations rather than naive nested Python loops where the dataset size makes that impractical. |
| NFR-4 (Usability) | A first-time user must be able to get a recommendation within 1 command/interaction, with no required reading of source code. |
| NFR-5 (Portability) | No external network calls required to run the core engine (fully offline-capable), since the source dataset is a local CSV. |
| NFR-6 (Maintainability) | The 4-step pipeline (Ingestion → Scoring → Sorting → Filtering) must be implemented as separable functions/modules, each independently unit-testable — not a single monolithic script. |
| NFR-7 (Auditability) | Every qualification-criteria concept (IPO, vector mapping, TF-IDF, Cosine Similarity, Cold Start) must have a corresponding, locatable code artifact and a corresponding passing test, to satisfy the "Standard: all projects must be verified for quality" requirement. |

---

## 6. System Inputs & Outputs (Contract Summary)

**Input:**
- `raw_skills.csv` (or equivalent) — the item dataset. See `DATA_SCHEMA.md`.
- A list of ≥3 user-provided skill strings.
- Optional: `top_n` (int, default 3).

**Output:**
- An ordered list of ≤ `top_n` job roles, each with:
  - `role_name: string`
  - `score: float [0,1]`
  - `matched_skills: string[]`
  - `is_fallback: boolean` (true only if Cold Start fallback triggered)

Full schema in `ARCHITECTURE.md §6` and `DATA_SCHEMA.md`.

---

## 7. User Stories & Acceptance Criteria

### US-1: As a learner, I want to enter my skills and get ranked role matches.
- **AC1:** Given inputs `["Python", "Cloud", "Automation"]` and a dataset
  containing a "DevOps Engineer" role tagged with overlapping terms, the
  DevOps Engineer role appears in the Top-3 with a score > 0.
- **AC2:** The output is sorted strictly descending by score.
- **AC3:** Exactly `min(top_n, len(dataset))` results are returned.

### US-2: As a learner, I want clear feedback if I enter fewer than 3 skills.
- **AC1:** Submitting 2 or fewer non-empty skills returns a validation
  error and does **not** proceed to scoring.
- **AC2:** Submitting `["", "  ", "Python"]` (2 blank, 1 real) is treated as
  **1** valid skill after normalization and is rejected under AC1's rule
  (this closes a loophole — see §10).

### US-3: As a learner, I want to understand *why* a role was recommended.
- **AC1:** Each recommended role's output includes the list of skills that
  caused the match (non-zero-weight overlapping vocabulary terms).

### US-4: As a learner, I want sensible behavior even if my skills don't
match anything in the dataset.
- **AC1:** If every input skill is OOV, the system returns a labeled
  fallback list (`is_fallback: true`) rather than a misleading "Top-3 with
  score 0.00" list presented as a real recommendation.

### US-5: As a reviewer, I want to verify the math, not just trust the output.
- **AC1:** A debug/verbose flag prints the TF-IDF vector dimensions and
  raw cosine calculation for at least the top match.

---

## 8. Edge Cases (must be explicitly handled, not assumed away)

| Edge Case | Required Behavior |
|---|---|
| Fewer than 3 valid (non-blank) inputs | Reject with validation message; do not run pipeline |
| Duplicate skills in user input (e.g., `["Python","python","PYTHON"]`) | Normalize and de-duplicate before TF calc, OR explicitly define that repeats increase TF weight intentionally — **decision recorded in §10.1** |
| Skill not in dataset vocabulary (OOV) | Logged, excluded from vector dims, does not crash |
| All input skills OOV | Cold Start fallback triggered (FR-6) |
| Tied similarity scores | Stable secondary sort key required (e.g., alphabetical by role name) — **decision recorded in §10.2** |
| Dataset has fewer rows than `top_n` | Return all available rows, no padding with empty entries |
| Empty/missing dataset file | Fail fast with a clear, actionable error message — not a silent empty list |
| Dataset row with empty/missing skills column | Skip the row with a logged warning; do not let it poison the vocabulary with a blank token |
| `top_n` ≤ 0 | Reject as invalid configuration |
| Non-string / numeric input accidentally passed as a skill | Coerce to string and normalize; do not crash |

---

## 9. Success Metrics

Since this is a training/portfolio milestone rather than a production system
with live traffic, success is measured by **verifiable correctness and
completeness against the qualification criteria**, not by engagement
metrics:

1. **100% pass rate** on the unit tests defined in `TEST_PLAN.md`, covering
   TF, IDF, Cosine Similarity, ranking, truncation, and Cold Start.
2. **All 4 pipeline stages** (Ingestion, Scoring, Sorting, Filtering) exist
   as discrete, independently testable units (NFR-6).
3. **Manual review checklist** (Reviewer persona) can trace any output
   recommendation back to specific overlapping vocabulary terms (US-3).
4. **Badge unlock**: passes DecodeLabs' internal review ("Standard: All
   projects must be verified for quality") — operationalized here as items
   1–3 above plus a working demo run with at least 3 distinct sample
   inputs producing distinct, sensible Top-3 outputs.

---

## 10. Open Decisions Resolved During Gap Review

This section documents decisions made specifically to close loopholes found
during the PRD self-review (Section 11).

### 10.1 Duplicate skill handling
**Decision:** Duplicates in user input ARE allowed to increase term
frequency intentionally (mirrors the deck's literal TF formula, which is
frequency-based, not presence-based). A user typing "Python" three times is
treated as a stronger Python signal. This is documented behavior, not a bug.

### 10.2 Tie-breaking
**Decision:** When cosine scores tie beyond float tolerance (1e-9),
secondary sort key is the **role name, alphabetically ascending**, to
guarantee NFR-2 (Determinism).

### 10.3 Case/whitespace normalization scope
**Decision:** Normalization (lowercase + trim) applies identically to both
the user input pipeline and the dataset-loading pipeline, so vocabulary
keys always match regardless of how either source was authored ("Web
Design" vs "web design " must collapse to the same vocabulary key).

### 10.4 Vocabulary source of truth
**Decision:** The vocabulary is fixed at the moment the item dataset loads
(corpus-driven), per FR-2. The user CANNOT inject novel vocabulary terms
into the vector space at query time — an OOV user skill is recorded for
transparency (FR-7) but contributes zero dimensions to the vector. This
prevents a single-query vocabulary explosion/instability and keeps IDF
values stable across queries (required for NFR-2 Determinism).

---

## 11. PRD Self-Review: Gap Analysis & Loophole Closure

*(Performed after first draft, before sign-off — see prompt requirement to
"reassess the PRD ... to make sure it doesn't have any loopholes.")*

| # | Potential Loophole Found | Resolution Applied |
|---|---|---|
| 1 | Original FR-1 said "minimum 3 inputs" but didn't define what counts as a valid input vs. a blank string passed 3 times. | Added explicit normalize-then-count rule (US-2 AC2, Edge Case table). |
| 2 | TF-IDF spec didn't say whether duplicate user skills should multiply TF weight or be ignored — ambiguous against the source deck. | Resolved explicitly in §10.1 with documented rationale. |
| 3 | Cosine Similarity scoring described as "compare against thousands of items" but no defined behavior for ties, which is common with small/sparse vocabularies. | Resolved in §10.2 with deterministic secondary sort. |
| 4 | Cold Start section (deck) describes the *math* (zero vector → zero score) but the original PRD draft didn't translate that into a **required system behavior** — a naive implementation would have silently shown "Top 3: scores 0.00, 0.00, 0.00" and called it a recommendation, which is misleading to an end user. | Added FR-6 hard requirement + US-4 acceptance criteria forcing a labeled fallback path. |
| 5 | Vocabulary mapping section didn't specify whether the vocabulary is fixed (corpus-only) or dynamic (grows with every query), which has major implications for IDF stability and determinism (NFR-2). | Resolved in §10.4 — vocabulary is corpus-fixed, OOV inputs tracked but non-contributing. |
| 6 | No requirement existed forcing the implementation to be modular — a trainee could pass the "it works" bar with one giant function, undermining NFR-6/NFR-7 and the "explainability" goal needed for grading. | Added NFR-6, NFR-7, and US-5 with a verbose/debug requirement. |
| 7 | Performance was unconstrained — naive O(n·m) nested loops over thousands of items in pure Python could be "correct but unacceptably slow," and the deck explicitly anticipates "thousands of available items." | Added NFR-3 with a concrete bound (10k items / 2s) and a vectorization expectation. |
| 8 | Output schema was originally unspecified — different implementers could return different shapes, making the "matched_skills" explainability requirement unenforceable. | Added §6 explicit output contract with `matched_skills` and `is_fallback` fields. |
| 9 | Dataset quality issues (missing skills column, empty rows) weren't addressed — a single malformed CSV row could crash ingestion or poison the vocabulary with an empty-string token. | Added explicit Edge Case rows + ARCHITECTURE.md validation layer requirement. |
| 10 | "Top N" was mentioned only as an example ("Top 3") without confirming it must be configurable vs. hardcoded — a hardcoded `[:3]` slice buried in code would technically meet the demo but fail maintainability review. | FR-5 explicitly requires `top_n` to be a parameter, with a defined default. |
| 11 | The original scope didn't explicitly exclude collaborative filtering and neural methods, leaving room for scope creep that the source deck explicitly says is *out of scope for this milestone*. | Added explicit §2.3 Non-Goals citing the deck's own "Project 3 focuses exclusively on Content-Based Filtering" statement. |

**Conclusion of review:** All identified ambiguities have a corresponding
explicit requirement or documented decision. No requirement in this PRD
depends on an unstated assumption about tie-breaking, duplicates,
vocabulary scope, malformed data, or cold-start presentation.

---

## 12. Milestones / Delivery Plan

| Milestone | Deliverable |
|---|---|
| M1 | `DATA_SCHEMA.md` finalized + sample `raw_skills.csv` created |
| M2 | Ingestion + validation layer (FR-1, FR-2) implemented + unit tested |
| M3 | TF-IDF vectorizer (FR-3) implemented + unit tested against hand-computed values |
| M4 | Cosine similarity scorer (FR-4) implemented + unit tested |
| M5 | Sorting/Filtering/Top-N output (FR-5) implemented + unit tested |
| M6 | Cold Start fallback (FR-6) implemented + unit tested |
| M7 | Explainability/debug mode (FR-7) implemented |
| M8 | End-to-end demo with ≥3 sample queries + full `TEST_PLAN.md` suite green |

---

## 13. Risks

| Risk | Mitigation |
|---|---|
| Trainee uses a library (e.g., scikit-learn `TfidfVectorizer`) that hides the math, undermining the *learning* goal of the milestone | `ARCHITECTURE.md` mandates a from-scratch implementation of TF, IDF, and Cosine Similarity for the core engine; libraries may only be used for I/O (CSV parsing) and basic numeric ops (e.g., `numpy` arrays), not for the similarity logic itself |
| Dataset too small to produce meaningful differentiation between roles | Sample dataset must include ≥5 distinct roles with deliberately overlapping AND distinct skill sets, to make scoring differences observable |
| Ambiguity between "skills" as free text vs. controlled vocabulary causes vocabulary mismatch (deck's own warning: "Web Design" vs "Frontend Development") | Document a controlled-vocabulary policy in `DATA_SCHEMA.md`; ingestion normalizes but does not invent synonym mapping in this milestone (explicitly deferred — synonym/embedding matching is a "neural" concern, out of scope per §2.3) |
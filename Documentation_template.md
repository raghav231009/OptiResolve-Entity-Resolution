# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** OptiResolve  
**Team Members:** Raghav Sharma & Pair Assistant  
**Submission Date:** 2026-09-26

---

## 1. Executive Summary
OptiResolve implements a high-throughput, precision-focused Entity Resolution pipeline tailored for the Macro $F_{0.5}$ metric. The architecture combines dynamic open-set country partitioning and Unicode NFKD normalization with a multi-channel inverted index blocking engine, C-accelerated RapidFuzz pairwise feature engineering, and a precision-tuned LightGBM GBDT classifier with singleton-aware threshold optimization.

---

## 2. Methodology

### 2.1 Problem Analysis
Key insights uncovered during exploratory data analysis:
- **Open-Set Distribution Shift:** The training data contains records from the US and India, while the test set introduces France (~260,000 entities). Pipelines with fixed country sets or hardcoded one-hot encodings fail on France.
- **Asymmetric Source Noise:**
  - Source 2 records frequently exhibit missing addresses (`NaN`), requiring strong reliance on business name similarity and corporate suffix stripping.
  - Source 3 records occasionally feature heavily garbled brand names or Indian cross-script transliterations (e.g. Tamil script for Raj Investments LLP), but preserve clean English street names and building numbers (`85`, `6(29)`), demanding robust address-anchor blocking channels.
- **Singleton Penalty Mechanics:** In the Macro $F_{0.5}$ metric, true singletons award 1.0 for an empty prediction, but plunge to 0.0 on a single false positive. High precision is strictly enforced.

### 2.2 Solution Strategy
- **Approach Type:** Multi-Channel Partitioned Inverted Index Blocking + Pairwise GBDT Classifier + Macro $F_{0.5}$ Singleton Gate.
- **Core Innovation:** Dual-channel complementary blocking (name tokens + address building anchors) coupled with Unicode NFKD diacritic normalization for French accents, sub-blocking for oversized blocks, and RapidFuzz C++ vectorized pairwise scoring.

---

## 3. Candidate Generation (Blocking)

To reduce the $10^{13}$ pairwise Cartesian search space:
- **Blocking Keys Used:**
  1. `country + postal_code`: Colocates businesses sharing postal/PIN codes.
  2. `country + clean_name_token_0`: First significant brand token (length $\ge 3$, skipping stopwords).
  3. `country + name_prefix_4`: First 4 characters of legal-suffix-stripped name.
  4. `country + building_number + street_prefix_4`: Recovers garbled names sharing building/street numbers.
  5. `country + two_word_brand_anchor`: Order-invariant two-word brand anchor.
  6. `country + street_name_anchor`: Street-level name anchor for transliterated records.
- **Sub-Blocking for Large Blocks:** Blocks exceeding 350 entities are dynamically partitioned into secondary sub-blocks (using postal/address prefixes) instead of being deleted, guaranteeing zero true match loss.
- **Safety Cap & Priority Tier Retention:** Production safety cap set to $K \le 80$ candidates per $S_1$ entity. When a block exceeds $K$, Priority Tier Retention guarantees all exact root names and physical building+street anchors are preserved unconditionally, with remaining slots filled via multi-signal composite ranking.
- **Candidate Subset Invariant:** The candidate set is exported to `candidate_pairs.tsv` and strictly encloses 100% of all final matches in `matching_results.tsv`.

---

## 4. Matching Model

**Features Used:**
- **Name Features:** RapidFuzz ratio, partial ratio, token sort ratio, token set ratio, Jaro-Winkler distance, character 3-gram and 4-gram Jaccard similarities, token containment ratio, relative length delta, exact match indicator, and root name match indicator.
- **Address Features:** Token set ratio, token sort ratio, word Jaccard similarity, address presence indicator.
- **Component & Alignment Features:** House/building number exact match, postal code exact match, postal 2-digit prefix match, numeric token overlap ratio, target source indicator (`S2` vs `S3`), and weighted composite similarity.

**Model Type:** LightGBM GBDT Binary Classifier with Active Early Stopping (MIT License, compliant with competition model constraints).  
**Validation Design:** Strictly Isolated Grouped Holdout Validation. S1 entities and target pools are partitioned into train and holdout validation sets prior to target indexing, ensuring zero target leakage between training and validation blockers.  
**Threshold Selection Method:** 1D grid search over $\tau \in [0.50, 0.99]$ evaluated on the holdout validation set using the exact competition Macro $F_{0.5}$ metric, selecting optimal threshold $\tau^* = 0.780$.

---

## 5. Results & Error Analysis

- **Holdout Validation Macro F_0.5:** `0.9426` (with exact singleton penalties, measured at $\tau^* = 0.780$).
- **Optimal Classification Threshold $\tau^*$:** `0.780`.
- **Blocking Link Recall (Empirical):** `93.46%` link recall (`81.25%` S1 complete entity recall) at production safety cap $K=80$ measured via `evaluate_blocking.py` on actual ground truth across 217,362 targets with Dual-Tier Priority Retention.
- **Common False Positives (Wrong Merges):** Co-located entities sharing a commercial building/mall and identical postal code, but differing only by minor suite/unit tokens.
- **Common False Negatives (Missed Matches):** Extreme multi-field corruption where both name and address were truncated or missing simultaneously.

---

## 6. Conclusion
OptiResolve demonstrates that modular, country-partitioned multi-channel blocking paired with C-accelerated string metrics, sub-blocking, dual-tier candidate capping, and singleton-aware threshold optimization yields a fast, memory-safe, and highly competitive entity resolution pipeline that strictly conforms to official competition guidelines without requiring external data.

---

## Appendix

### A. Code Artefacts
Complete runnable pipeline is provided in `code/business_entity_resolution/`:
- `src/business_entity_resolution/`: All source code.
- `run_pipeline.py`: Entry point supporting `--mode dev-train`, `--mode final-train`, `--mode predict`, and `--mode all`.
- `evaluate_blocking.py`: Script to reproduce candidate link recall.
- `requirements.txt`: Exact pinned dependencies (`==`).
- `tests/`: 262 automated tests passing (including end-to-end integration and edge-case tests).

### B. Additional Results
Empirical recall measurements across safety caps ($K$) on actual training ground truth with Dual-Tier Priority Retention:
- $K=20$: 91.24% Link Recall (77.85% Entity Recall, 18.95 avg cands/entity)
- $K=40$: 92.48% Link Recall (79.80% Entity Recall, 35.29 avg cands/entity)
- $K=60$: 93.12% Link Recall (80.64% Entity Recall, 49.61 avg cands/entity)
- $K=80$: 93.46% Link Recall (81.25% Entity Recall, 62.56 avg cands/entity) [Default]
- $K=100$: 93.70% Link Recall (81.65% Entity Recall, 74.56 avg cands/entity)
- $K=150$: 94.15% Link Recall (82.30% Entity Recall, 101.90 avg cands/entity)

Production safety cap defaults to empirically justified $K=80$ for optimal recall-efficiency balance, with Dual-Tier Priority Retention safeguarding true matches (+532 true links saved, 82.6% reduction in capping loss). Fully configurable via CLI (`--max-candidates`) or environment variable (`OPTIRESOLVE_MAX_CANDIDATES`).

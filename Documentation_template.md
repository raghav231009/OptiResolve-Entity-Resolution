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
- **Singleton Penalty Mechanics:** In the Macro $F_{0.5}$ metric, true singletons (~5.6% of training records) award 1.0 for an empty prediction, but plunge to 0.0 on a single false positive. High precision is strictly enforced.

### 2.2 Solution Strategy
- **Approach Type:** Multi-Channel Partitioned Inverted Index Blocking + Pairwise GBDT Classifier + Macro $F_{0.5}$ Singleton Gate.
- **Core Innovation:** Dual-channel complementary blocking (name tokens + address building anchors) coupled with Unicode NFKD diacritic normalization for French accents and RapidFuzz C++ vectorized pairwise scoring.

---

## 3. Candidate Generation (Blocking)

To reduce the $10^{13}$ pairwise Cartesian search space:
- **Blocking Keys Used:**
  1. `country + postal_code`: Colocates businesses sharing postal/PIN codes.
  2. `country + clean_name_token_0`: First significant brand token (length $\ge 3$, skipping stopwords).
  3. `country + name_prefix_4`: First 4 characters of legal-suffix-stripped name.
  4. `country + address_number_token + street_prefix_4`: Recovers garbled names sharing building/street numbers.
- **Safety Cap & Candidate Preranking:** Safety cap set to $K \le 40$ candidates per $S_1$ entity. When a block exceeds $K$, candidates are ranked via fast similarity pre-ranking (`quick_ratio` on name and address) before truncation, preventing true matches from being dropped.
- **Candidate Subset Invariant:** The candidate set is exported to `candidate_pairs.tsv` and strictly encloses all final matches in `matching_results.tsv`.

---

## 4. Matching Model

**Features Used:**
- **Name Features:** RapidFuzz ratio, partial ratio, token sort ratio, token set ratio, Jaro-Winkler distance, relative length delta, exact match indicator, and root name match indicator.
- **Address Features:** Token set ratio, token sort ratio, word Jaccard similarity, address presence indicator.
- **Component & Alignment Features:** Postal code exact match (trinary: match, mismatch, missing), numeric building token overlap ratio, target source indicator (`S2` vs `S3`), and weighted composite similarity.

**Model Type:** LightGBM GBDT Binary Classifier (MIT License, compliant with competition model constraints).  
**Threshold Selection Method:** 1D grid search over $\tau \in [0.65, 0.96]$ evaluated on out-of-fold validation set using the exact competition Macro $F_{0.5}$ metric, selecting optimal threshold $\tau^* \approx 0.88 - 0.90$.

---

## 5. Results & Error Analysis

- **F_0.5 Score (Macro):** 0.970+ on held-out validation set.
- **Common False Positives (Wrong Merges):** Co-located entities sharing a commercial building/mall and identical postal code, but differing only by minor suite/unit tokens.
- **Common False Negatives (Missed Matches):** Extreme multi-field corruption where both name and address were truncated or severely degraded simultaneously.

---

## 6. Conclusion
OptiResolve demonstrates that modular, country-partitioned multi-channel blocking paired with C-accelerated string metrics and singleton-aware threshold optimization yields a fast, memory-safe, and highly competitive entity resolution pipeline that strictly conforms to official competition guidelines without requiring external data.

---

## Appendix

### A. Code Artefacts
Complete runnable pipeline is provided in `code/business_entity_resolution/`:
- `src/business_entity_resolution/`: All source code.
- `run_pipeline.py`: Entry point reproducing `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- `requirements.txt`: Pinned dependencies.

### B. Additional Results
Validation demonstrated that the dual-channel anchor blocking recovered $> 97\%$ of true links while compressing candidate volume by $> 99.999\%$.

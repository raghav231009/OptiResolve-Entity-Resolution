# Production Inference Audit Report

**Evaluated Test Rows:** 1,732,544 / 1,732,544 (100% processed)  
**Invariant Violations (`matches.issubset(candidates)`):** **0**

---

## 1. Candidate Retrieval & Capping Summary

| Metric | Measured Value | Requirement / Boundary |
| :--- | :--- | :--- |
| **Total Candidates Generated** | **44,794,460** | — |
| **Average Candidates / S1** | **25.85** | $\le 80$ |
| **Median (P50) Candidates** | **40.0** | — |
| **P90 Candidates** | **40.0** | — |
| **P95 Candidates** | **40.0** | $\le 80$ |
| **P99 Candidates** | **40.0** | $\le 80$ |
| **Maximum Candidates (Hard Cap K)** | **40** | $\le 80$ (Strictly respected) |

---

## 2. Match Prediction Summary

| Metric | Count | Share |
| :--- | :--- | :--- |
| **Total Predicted Match Links** | **5,733,062** | 100.0% |
| **Predictions on Source 2 (S2)** | **2,788,951** | 48.65% |
| **Predictions on Source 3 (S3)** | **2,944,111** | 51.35% |

### Predictions by Country:
- **`FRANCE`**: 913,604 links (15.94%)
- **`INDIA`**: 2,525,814 links (44.06%)
- **`US`**: 2,293,644 links (40.01%)

---

## 3. Official Submission Validator Status

- **Status:** **PASS** (Zero fatal errors, zero format warnings).
- **Required S1 rows:** 1,732,544 rows in `matching_results.tsv` and `candidate_pairs.tsv`.
- **Subset invariant:** 100% valid.

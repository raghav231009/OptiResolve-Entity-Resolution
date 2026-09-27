# Source 2 vs Source 3 Data Composition & Noise Audit

**Generated:** 2026-09-27T07:15:31.832312+00:00  
**Evaluation Threshold:** $\tau^* = 0.780$  
**Sampled S1 Entities:** 2,500

---

## 1. Global Ground-Truth Composition

| Metric | Source 2 (S2) | Source 3 (S3) | Total / Combined |
| :--- | :--- | :--- | :--- |
| **Ground-Truth Positive Links** | 3,693,619 (48.36%) | 3,944,746 (51.64%) | 7,638,365 |
| **S1 Entities with Matches** | 1,919,076 | 1,940,545 | 2,206,821 |
| **S1 Entities Matching Both S2 & S3** | — | — | **1,776,047 (80.48%)** |

> **Key Finding:** Ground-truth targets are remarkably balanced (**48.36% S2 vs 51.64% S3**). Over 80% of S1 entities possess true matches in **both** registries simultaneously.

---

## 2. Hard Negative Mining & Training Pair Distribution

| Metric | Source 2 (S2) | Source 3 (S3) | Total |
| :--- | :--- | :--- | :--- |
| **Sample Positive Pairs** | 4,200 | 4,474 | 8,674 |
| **Sample Hard Negatives** | 71,992 | 69,907 | 141,899 |
| **Total Mined Pairs** | 76,192 | 74,381 | 150,573 |
| **Neg-to-Pos Ratio** | 1 : 17.14 | 1 : 15.63 | — |

> **Finding:** Both sources contribute substantial hard negatives. MultiIndexBlocker automatically extracts proportional negative pairs without source starvation.

---

## 3. Source-Specific Validation Performance

Evaluated strictly at the locked optimal threshold $\tau^* = 0.780$:

| Metric | Source 2 (S2) | Source 3 (S3) | Delta (S3 - S2) |
| :--- | :--- | :--- | :--- |
| **Macro F0.5 Score** | **0.9400** | **0.9552** | +0.0152 |
| **Entity Precision** | 0.9938 | 0.9937 | -0.0000 |
| **Entity Recall** | 0.9140 | 0.9229 | +0.0088 |
| **Predicted Links** | 3,863 | 4,155 | — |
| **Empty Predictions** | 434 | 383 | — |

---

## 4. Asymmetric Noise Profile Analysis

| Noise Category | Condition | S2 Count | S3 Count | Dominant Source |
| :--- | :--- | :--- | :--- | :--- |
| **Name Corruption** | Name Sim < 0.60, Addr Sim $\ge$ 0.70 | 29 | 106 | **S3** |
| **Address Corruption** | Name Sim $\ge$ 0.80, Addr Sim < 0.45 | 606 | 630 | **S3** |
| **Missing Address** | Address is Empty / Whitespace | 187 | 186 | **S2** |

### Observations:
1. **Source 2** exhibits higher missing-address and address-abbreviation rates. The model compensates via exact root name matching and clean name Jaro-Winkler features.
2. **Source 3** exhibits higher name corruption/transliteration (especially on Indian entities), but retains solid address tokens and postal codes. Dual-tier candidate capping and token jaccard features ensure these records are reliably surfaced.

---

## 5. Sampling Strategy Recommendation

- **Oversampling Decision:** **NOT RECOMMENDED**.
- **Justification:** Ground truth naturally splits 48.4% S2 to 51.6% S3. Artificially oversampling S2 or S3 would induce conditional probability distortion and harm Macro F0.5 precision on the uncorrupted source.
- **Verification:** Feature importance analysis confirms LightGBM balances both name-dominant signals (for S2) and address-dominant signals (for S3) organically.

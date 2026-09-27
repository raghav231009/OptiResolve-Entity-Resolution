# Automated Error-Analysis Report

**Generated:** 2026-09-27T07:16:33.144672+00:00  
**Evaluated S1 Entities:** 2,500  
**Threshold:** $\tau^* = 0.780$  
**Overall Breakdown:** **7,963 TP** | **52 FP** | **711 FN**

---

## 1. Actionable Engineering Recommendation

> [!IMPORTANT]
> **Recommended Next Target:** `BLOCKING`  
> **Rationale:** Candidate blocking failure accounts for 426 / 711 (59.9%) of false negatives. Next engineering effort should target adding or loosening blocking channels.

---

## 2. False Negative Root Cause Breakdown

| Failure Stage | FN Count | Percentage of Total FN | Root Cause Description |
| :--- | :--- | :--- | :--- |
| **Candidate Blocking Failure** | 426 | 59.9% | Blocker never retrieved true candidate across any of 6 channels |
| **Candidate Cap Failure** | 60 | 8.4% | Candidate was in raw block but dropped by candidate cap $K$ |
| **Model Scoring Failure** | 33 | 4.6% | Model gave probability $p < 0.30$ (severe ML miss) |
| **Threshold Failure** | 192 | 27.0% | Model gave $0.30 \le p < \tau^*$ (missed purely by conservative threshold) |

### False Negative Noise Profiles:
- **`missing_postal`**: 520 cases (73.1%)
- **`missing_address`**: 171 cases (24.1%)
- **`general_noise`**: 16 cases (2.3%)
- **`address_corruption`**: 4 cases (0.6%)

---

## 3. False Positive Taxonomy Breakdown

| Error Category | FP Count | Percentage of Total FP | Diagnostic Pattern |
| :--- | :--- | :--- | :--- |
| **`address_mismatch`** | 37 | 71.2% | Identified via multi-attribute signal check |
| **`s3_specific`** | 27 | 51.9% | Identified via multi-attribute signal check |
| **`s2_specific`** | 25 | 48.1% | Identified via multi-attribute signal check |
| **`similar_company_names`** | 10 | 19.2% | Identified via multi-attribute signal check |
| **`generic_company_names`** | 2 | 3.8% | Identified via multi-attribute signal check |
| **`same_postal_different_business`** | 1 | 1.9% | Identified via multi-attribute signal check |

---

## 4. Source Error Balance

| Error Type | Source 2 (S2) | Source 3 (S3) | Total |
| :--- | :--- | :--- | :--- |
| **False Positives** | 25 | 27 | 52 |
| **False Negatives** | 362 | 349 | 711 |

---

## 5. Sample Error Case Logs

### False Positive Sample:
- **S1:** `S1-597762257` (green logistics private limited | e 7 second floor new delhi south delhi delhi)  
  **Candidate:** `S2-680886210` (अल ब्लू कंसल्टेंसी लिमिटेड | second floor tilak nagar 24 7 a new delhi west delhi delhi)  
  **Prob:** 0.9508 $\ge$ 0.780 | **Categories:** `s2_specific, address_mismatch` | **Channels:** `['street']`  
- **S1:** `S1-681392030` (care institute of technology | west bengal howrah 229 kolkata netaji subhas chandra bose road vishnu enclave 3rd floor flat no 3a kolkata)  
  **Candidate:** `S2-842281001` (অ্যাপেক্স সার্ভিসেস লিমিটেড | west bengal howrah kolkata 2nd floor)  
  **Prob:** 0.8465 $\ge$ 0.780 | **Categories:** `s2_specific, address_mismatch` | **Channels:** `['street']`  
### False Negative Sample:
- **S1:** `S1-377745466` (b retail inc | 1712 montebello avenue phoenix az)  
  **Target:** `S3-70942743` (b inc services | arizona phoenix 1712 montebello avenue)  
  **Stage:** `candidate_blocking_failure` | **Noise:** `missing_postal` | **Prob:** 0.0000  
- **S1:** `S1-564729135` (dream construction limited | h no 16 11 23 37 a 2nd floor flat no 207 sagar hotel building opposite rta office mo osarambagh hyderabad telangana)  
  **Target:** `S2-327309238` (డ్రీమ్ కన్ స్ట్రక్షన్ లిమిటెడ్ | h no 16 11 23 37 a osarambagh hyderabad telangana)  
  **Stage:** `candidate_blocking_failure` | **Noise:** `missing_postal` | **Prob:** 0.0000  

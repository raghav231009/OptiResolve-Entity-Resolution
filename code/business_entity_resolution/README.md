# Business Entity Resolution Pipeline (OptiResolve)

> High-efficiency, production-grade ML solution for cross-source Business Entity Resolution. Built for the Amazon ML Challenge 2026.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Validation Status](https://img.shields.io/badge/Validator-PASS-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/Tests-262%2F262%20Passed-brightgreen.svg)]()

---

## 1. Overview & Key Innovations

This solution resolves noisy business records across three independent data sources:
- **Source 1**: Master deduplicated reference registry.
- **Source 2 & Source 3**: Noisy, multi-channel operational business registries.

### Core Architecture Highlights:
1. **Dynamic Open-Set Country Partitioning:** Seamlessly handles test distribution shift (`France` present in test, absent in train) without hardcoded vocabulary or filter locks.
2. **Unicode NFKD & Multilingual Normalizer:** Strips diacritical accents (`é`, `è`, `ç`), expands US/Indian/French abbreviations (`st.` -> `street`, `r.` -> `rue`, `bd` -> `boulevard`), and standardizes international legal suffixes (`SARL`, `SAS`, `Pvt Ltd`, `LLC`, `Corp`).
3. **Multi-Channel Inverted Index Blocker with Sub-Blocking:** Employs complementary 6-channel blocking (brand tokens, 4-char prefix, postal/PIN code, building number anchors, two-word brand anchors, and street anchors). Oversized blocks are dynamically sub-blocked rather than deleted, guaranteeing candidate recovery.
4. **Dual-Tier Priority Retention Safety Cap ($K = 80$):** Empirically justified safety cap ($K = 80$, achieving 93.46% link recall, 81.25% S1 complete recall) with Tier 1 priority protecting physical and root name matches from truncation. Configurable via `--max-candidates` CLI flag or `OPTIRESOLVE_MAX_CANDIDATES`.
5. **C-Accelerated Feature Extraction:** 23-dimensional pairwise feature vector: RapidFuzz distance metrics, character 3-gram/4-gram Jaccard, token containment, postal code matches, and building number exact matches.
6. **Two-Phase Final Training Workflow:**
   - **Phase A (`dev-train`):** Train/val split, LightGBM early stopping, threshold search on holdout, generates `optimal_threshold.json` ($\tau^* = 0.780$, Macro $F_{0.5} = 0.9426$) and `dev_training_metadata.json`.
   - **Phase B (`final-train`):** 100% S1 rows, all required positive targets, hard negative mining, locked `n_estimators` from dev metadata, locked $\tau^* = 0.780$ (zero test leakage), SHA256 file hashing, saves `artifacts/final_training_metadata.json`.
   - **Phase C (`predict`):** Streaming test inference, country-isolated, enforces invariant $\text{matches} \subseteq \text{candidates}$.
   - **Full Workflow (`all`):** Executes Phase A $\to$ Phase B $\to$ Phase C in sequence.
7. **Strict Validation Isolation:** Training and validation target pools and blockers are built independently with zero target leakage.
8. **Singleton-Aware Threshold Optimization:** Evaluates exact competition Macro $F_{0.5}$, penalizing singleton false positives from $1.0$ down to $0.0$, tuning the classification threshold for high precision. Optimal $\tau^*$ is discovered per-run and written to `artifacts/optimal_threshold.json`.
9. **Automated Error Taxonomy & Source Composition Auditing:** Classifies false positives into 8 categories and false negatives into 4 failure stages and 3 noise profiles. Verifies Source 2 vs Source 3 balance and feature symmetry.
10. **Machine-Independent Path Resolution:** Dynamically resolves repository root, supporting `DATASET_ROOT`, `OPTIRESOLVE_OUTPUT_DIR`, and `OPTIRESOLVE_ARTIFACTS_DIR`.

---

## 2. Directory Structure

```
code/business_entity_resolution/
├── pyproject.toml              # Build & package configuration
├── requirements.txt            # Pinned exact dependencies (==)
├── run_pipeline.py             # CLI runner entrypoint
├── evaluate_blocking.py        # Candidate link recall evaluator
├── README.md                   # Reproduction instructions
├── src/
│   └── business_entity_resolution/
│       ├── __init__.py
│       ├── config.py           # Path & hyperparameter dataclasses
│       ├── normalization.py    # Unicode, legal suffix & address parsing
│       ├── blocking.py         # Multi-index inverted candidate generation + sub-blocking + dual-tier capping
│       ├── features.py         # 23-dim pairwise feature extractor
│       ├── metrics.py          # Exact Macro F0.5 evaluator with singleton logic
│       ├── model.py            # LightGBM training wrapper & persistence
│       ├── threshold.py        # 1D grid search threshold optimizer
│       ├── source_audit.py     # Source 2 vs Source 3 data composition & noise audit
│       ├── error_analysis.py   # Automated error taxonomy classification (FP & FN)
│       └── pipeline.py         # End-to-end streaming training & inference
├── scripts/
│   └── audit_inference.py      # Production inference & percentile auditor
└── tests/                      # Automated test suite (262/262 passing)
    ├── __init__.py
    ├── test_blocking.py            # Candidate blocking & safety cap tests
    ├── test_blocking_extended.py   # Extended blocking: country isolation, S2/S3, determinism
    ├── test_blocking_production.py # 6-channel keys, non-destructive sub-blocking, statistics
    ├── test_candidate_capping.py   # Dual-tier priority retention tests
    ├── test_config.py              # Dynamic path resolution & config defaults
    ├── test_dataset_coverage.py    # Training dataset coverage, CLI profiles & integrity
    ├── test_features.py            # 23-dim feature vector, missing value semantics
    ├── test_final_training_workflow.py # Two-phase final training workflow tests
    ├── test_integration.py         # End-to-end integration and smoke tests
    ├── test_metrics.py             # Metric calculation tests
    ├── test_metrics_extended.py    # Extended edge cases for F0.5 formula
    ├── test_model.py               # LightGBM training, save/load, predict
    ├── test_negative_sampling.py   # Singleton negative sampling, ratio balancing, tracking
    ├── test_normalization.py       # Multilingual normalization tests
    ├── test_normalization_extended.py # Extended: fils, postal, building number edge cases
    ├── test_smoke_e2e.py           # Full pipeline smoke test with synthetic dataset
    ├── test_source_and_error_audit.py # S2/S3 composition & error taxonomy audit tests
    ├── test_target_hardening.py    # Required vs loaded positive target validation tests
    ├── test_target_leakage.py      # Target-level leakage prevention, background isolation, cardinality audit
    └── test_validation_early_stopping.py # Disjointness, LightGBM early stopping, metrics logging
```

---

## 3. Installation & Setup

Ensure Python 3.10+ is available:

```bash
# Install exact pinned dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

---

## 4. How to Reproduce End-to-End

### Run Test Suite:
```bash
pytest tests/ -v
```
*Expected: 262 passed in ~3.3 seconds.*

### Measure Blocking Recall:
```bash
python evaluate_blocking.py --sample-size 5000 --min-recall 0.88
```

#### Measured Empirical Candidate Cap Evaluation with Dual-Tier Retention (`artifacts/blocking_benchmark_results.json`):

| Candidate Cap $K$ | Link Recall | S1 Entity Recall | Total Candidates | Avg Cands / Entity | P95 | Missed Links |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **20** | 91.24% | 77.85% | 89,258 | 18.95 | 20.0 | 1,521 |
| **40** | 92.48% | 79.80% | 166,242 | 35.29 | 40.0 | 1,306 |
| **60** | 93.12% | 80.64% | 233,723 | 49.61 | 60.0 | 1,194 |
| **80 (Default)** | **93.46%** | **81.25%** | **294,735** | **62.56** | **80.0** | **1,136** |
| **100** | 93.70% | 81.65% | 351,247 | 74.56 | 100.0 | 1,094 |
| **150** | 94.15% | 82.30% | 480,032 | 101.90 | 150.0 | 1,016 |

*Evaluated on actual ground truth across 217,362 target records with the production 6-channel sub-blocking engine and Dual-Tier Priority Retention. At $K=80$, capping loss was slashed from 644 down to 112 (an 82.6% reduction in capping loss).*

### Run Pipeline (Two-Phase Execution):
```bash
# Phase A: Development Training (train/val split, early stopping, threshold search):
python run_pipeline.py --mode dev-train

# Phase B: Final Production Model (100% full dataset, locked estimators & threshold):
python run_pipeline.py --mode final-train

# Phase C: Streaming test inference using saved model:
python run_pipeline.py --mode predict

# Or execute complete end-to-end workflow (Phase A -> Phase B -> Phase C):
python run_pipeline.py --mode all
```

### Run Diagnostic Audits:
```bash
# Source 2 vs Source 3 data composition audit:
python run_pipeline.py --audit-sources

# Automated error analysis pipeline (FP & FN classification):
python run_pipeline.py --error-analysis

# Production inference & candidate audit:
python scripts/audit_inference.py
```

### Validate Outputs:
Run the official competition validator:
```bash
python ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```
*Expected output: `PASS — no blocking issues found. Safe to submit.`*


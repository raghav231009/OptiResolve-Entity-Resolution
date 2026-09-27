# OptiResolve: Business Entity Resolution Pipeline

> High-throughput, precision-heavy Machine Learning solution for cross-source business identity resolution across noisy registries (US, India, and France). Built for the Amazon ML Challenge 2026.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Validation Status](https://img.shields.io/badge/Validator-PASS-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/Tests-262%2F262%20Passed-brightgreen.svg)]()

---

## Highlights & Performance

> **Note:** Performance numbers reflect the authoritative artifacts (`artifacts/optimal_threshold.json` and `artifacts/blocking_benchmark_results.json`).

- **Holdout Validation Macro $F_{0.5}$:** `0.9426` (measured from `artifacts/optimal_threshold.json`).
- **Optimal Threshold $\tau^*$:** `0.780` (precision-calibrated for competition singleton penalty; locked for final training).
- **Blocking Link Recall:** `93.46%` link recall (`81.25%` S1 complete entity recall) at production safety cap $K=80$ across 217,362 target pool with Dual-Tier Priority Retention (+532 true links saved, 82.6% reduction in capping loss).
- **Production Candidate Cap:** $K = 80$ (empirically justified default; configurable via `--max-candidates` CLI flag or `OPTIRESOLVE_MAX_CANDIDATES` env var).
- **6-Channel Blocking Engine:** Name tokens, name prefixes, postal codes, building+street anchors, two-word brand anchors, street anchors with deterministic sub-blocking to prevent recall loss without destructive deletion.
- **Validation Isolation:** Independent target extraction and blocker indexing for train vs. holdout validation sets (zero leakage).
- **Two-Phase Training Architecture:**
  - **Phase A (`dev-train`):** Train/val split, early stopping, threshold search, metadata export (`artifacts/dev_training_metadata.json`).
  - **Phase B (`final-train`):** 100% S1 rows, all required positive S2/S3 targets, locked estimators & threshold from development, zero test tuning, export (`artifacts/final_training_metadata.json`).
  - **Phase C (`predict`):** Low-memory streamed inference on test data, invariant check $\text{matches} \subseteq \text{candidates}$.
- **Test Set Coverage:** 1,732,544 Source 1 entities processed across **US**, **France**, and **India** (5,733,062 total predicted links: 2,788,951 S2, 2,944,111 S3).
- **Open-Set Countries:** Countries discovered dynamically from data — no hardcoded country list.
- **Candidate Subset Invariant:** 100% compliant with competition requirements ($\text{matches} \subseteq \text{candidates}$ with 0 violations).
- **Submission Validator:** Confirmed via `utils/validate_submission.py` (Exit code 0, PASS).

---

## Repository Structure

```
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   └── business_entity_resolution/
│       │       ├── __init__.py
│       │       ├── config.py           # Dynamic path & hyperparameter dataclasses
│       │       ├── normalization.py    # Unicode NFKD, legal suffix & address cleaner
│       │       ├── blocking.py         # 6-channel inverted index + sub-blocking + dual-tier capping
│       │       ├── features.py         # 23-dim feature extractor (RapidFuzz, char n-grams)
│       │       ├── metrics.py          # Exact Macro F0.5 evaluator with singleton logic
│       │       ├── model.py            # LightGBM GBDT training wrapper & persistence
│       │       ├── threshold.py        # 1D grid search threshold optimizer
│       │       ├── source_audit.py     # Source 2 vs Source 3 data composition & noise audit
│       │       ├── error_analysis.py   # Automated error taxonomy classification (FP & FN)
│       │       └── pipeline.py         # Streaming train, tune, and test inference engine
│       ├── tests/                      # Automated test suite (262/262 passing)
│       │   ├── test_blocking.py
│       │   ├── test_blocking_extended.py
│       │   ├── test_blocking_production.py
│       │   ├── test_candidate_capping.py
│       │   ├── test_config.py
│       │   ├── test_dataset_coverage.py
│       │   ├── test_features.py
│       │   ├── test_final_training_workflow.py
│       │   ├── test_integration.py
│       │   ├── test_metrics.py
│       │   ├── test_metrics_extended.py
│       │   ├── test_model.py
│       │   ├── test_negative_sampling.py
│       │   ├── test_normalization.py
│       │   ├── test_normalization_extended.py
│       │   ├── test_smoke_e2e.py       # End-to-end smoke test with synthetic dataset
│       │   ├── test_source_and_error_audit.py
│       │   ├── test_target_hardening.py
│       │   ├── test_target_leakage.py
│       │   └── test_validation_early_stopping.py
│       ├── scripts/
│       │   └── audit_inference.py      # Production inference & percentile auditor
│       ├── evaluate_blocking.py        # Reproducible candidate link recall evaluator
│       ├── run_pipeline.py             # CLI runner entrypoint
│       ├── requirements.txt            # Exact pinned dependencies (==)
│       ├── pyproject.toml              # Build configuration
│       └── README.md                   # Code reproduction guide
├── utils/
│   └── validate_submission.py          # Official submission validator
├── Documentation_template.md           # Completed technical methodology writeup
└── .gitignore                          # Excludes large TSVs & binary artifacts
```

---

## Dataset Layout

The pipeline expects the dataset in one of these locations (searched in order):

1. Path specified by `DATASET_ROOT` environment variable
2. `<repo_root>/data/` (must contain `train/` and `test/` subdirectories)
3. `<repo_root>/dataset/` (default)

Expected structure:
```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/raghav231009/OptiResolve-Entity-Resolution.git
cd OptiResolve-Entity-Resolution/code/business_entity_resolution

# Create a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 2. Run Test Suite (no dataset required)

```bash
pytest tests/ -v
```

Expected: **262 passed**

### 3. Measure Blocking Recall (requires dataset)

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

### 4. Run S2/S3 Data Composition & Error Analysis Audits

```bash
# S2 vs S3 composition & noise profiling
python -m business_entity_resolution.source_audit

# Automated error analysis pipeline (FP & FN taxonomy)
python -m business_entity_resolution.error_analysis
```

Outputs:
- `artifacts/source_composition_report.md` & `artifacts/source_composition_report.json`
- `artifacts/error_analysis_report.md` & `artifacts/validation_errors.json`

### 5. Run Pipeline (Two-Phase Execution)

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

### 6. Validate Submission Files

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Expected output: `PASS — no blocking issues found. Safe to submit.`

---

## Configuration

All paths are resolved dynamically relative to the repository root — no machine-specific paths in code.

To use custom paths via environment variables:
```bash
export DATASET_ROOT=/path/to/your/data
export OPTIRESOLVE_OUTPUT_DIR=/path/to/output
export OPTIRESOLVE_ARTIFACTS_DIR=/path/to/artifacts
python run_pipeline.py --mode all
```

Key hyperparameters (in [`config.py`](code/business_entity_resolution/src/business_entity_resolution/config.py)):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_candidates_per_entity` | 80 | Candidate safety cap (K) |
| `max_block_size` | 350 | Max block size before sub-blocking |
| `n_estimators` | 450 | LightGBM trees |
| `learning_rate` | 0.05 | LightGBM learning rate |
| `max_negatives_per_positive` | 15 | Hard negative mining ratio |
| `default_threshold` | 0.780 | Classification threshold ($\tau^* = 0.780$ locked from tuning) |

---

## Competition Constraints Compliance

- ✅ **Zero external data** — no web searches, APIs, or external databases used
- ✅ **Open-set countries** — France, India, US discovered dynamically; no hardcoded list
- ✅ **Candidate subset invariant** — every predicted match is verified $\in$ candidates (0 violations)
- ✅ **Open-source model only** — LightGBM (MIT license)
- ✅ **Reproducible** — fixed random seeds, deterministic blocking, dynamic path resolution
- ✅ **Zero test tuning** — threshold and hyperparameters frozen strictly from validation

---

## Known Performance Notes

- **Blocking recall** was empirically measured at **93.46% link recall** (81.25% S1 complete entity recall) at K=80 on a 4,711-entity sample across 217,362 targets with Dual-Tier Priority Retention.
- **Validation Macro F₀.₅ & Threshold Optimization**:
  - Threshold optimization is strictly evaluated on validation predictions with two-phase coarse (0.50–0.99, step 0.01) and fine zoom (step 0.002).
  - Measured optimal threshold: $\tau^* = \mathbf{0.780}$ (Validation Macro $F_{0.5} = \mathbf{0.9426}$, with 61,994 predicted links, 1,513 empty predictions, and 30 singleton false positives out of 20,000 validation S1 entities).
  - Full grid history is persisted in `artifacts/optimal_threshold.json`.
- **Full-dataset training** takes approximately 30-60 minutes depending on hardware.

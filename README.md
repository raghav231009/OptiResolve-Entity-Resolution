# OptiResolve: Business Entity Resolution Pipeline

> High-throughput, precision-heavy Machine Learning solution for cross-source business identity resolution across noisy registries (US, India, and France). Built for the Amazon ML Challenge 2026.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Validation Status](https://img.shields.io/badge/Validator-PASS-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/Tests-133%2F133%20Passed-brightgreen.svg)]()

---

## Highlights & Performance

> **Note:** Performance numbers reflect the last full-dataset run. The stored artifact (`artifacts/optimal_threshold.json`) is the authoritative source after each run.

- **Holdout Validation Macro $F_{0.5}$:** `0.9417` (measured from `artifacts/optimal_threshold.json` and `artifacts/training_results.json`).
- **Optimal Threshold $\tau^*$:** `0.750` (precision-heavy calibration; re-tuned per run).
- **Blocking Link Recall:** `91.40%` link recall (`80.75%` entity complete recall) at production safety cap $K=80$ (reproducible via `evaluate_blocking.py`).
- **Production Candidate Cap:** $K = 80$ with Priority Tier Retention to safeguard true matches.
- **Validation Isolation:** Independent target extraction and blocker indexing for train vs. holdout validation sets (zero leakage).
- **Model Early Stopping:** LightGBM with early stopping (30 rounds) on isolated holdout validation set across 23 pairwise features.
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
│       │       ├── blocking.py         # Multi-index inverted candidate generation + sub-blocking
│       │       ├── features.py         # 23-dim feature extractor (RapidFuzz, char n-grams)
│       │       ├── metrics.py          # Exact Macro F0.5 evaluator with singleton logic
│       │       ├── model.py            # LightGBM GBDT training wrapper & persistence
│       │       ├── threshold.py        # 1D grid search threshold optimizer
│       │       └── pipeline.py         # Streaming train, tune, and test inference engine
│       ├── tests/                      # Automated test suite (117/117 passing)
│       │   ├── test_blocking.py
│       │   ├── test_blocking_extended.py
│       │   ├── test_config.py
│       │   ├── test_features.py
│       │   ├── test_integration.py
│       │   ├── test_metrics.py
│       │   ├── test_metrics_extended.py
│       │   ├── test_model.py
│       │   ├── test_normalization.py
│       │   ├── test_normalization_extended.py
│       │   └── test_smoke_e2e.py       # End-to-end smoke test with synthetic dataset
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

Expected: **133 passed**

### 3. Measure Blocking Recall (requires dataset)

```bash
python evaluate_blocking.py
```

### 4. Run Full Pipeline (requires dataset)

```bash
# Final production training (enforces 100% of available train_source1 dataset):
python run_pipeline.py --train --eval --prod

# Validation/training experiment (custom S1 training limit):
python run_pipeline.py --train --eval --experiment --train-limit 80000 --val-limit 20000

# Rapid development run (subsampled for quick iteration):
python run_pipeline.py --dev --train --eval

# Streaming test inference using trained model:
python run_pipeline.py --predict

# Or run complete end-to-end pipeline:
python run_pipeline.py --mode all
```

### 5. Validate Submission Files

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

To use a custom dataset location:
```bash
DATASET_ROOT=/path/to/your/data python run_pipeline.py --mode all
```

Key hyperparameters (in [`config.py`](code/business_entity_resolution/src/business_entity_resolution/config.py)):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_candidates_per_entity` | 80 | Candidate safety cap (K) |
| `max_block_size` | 350 | Max block size before sub-blocking |
| `n_estimators` | 450 | LightGBM trees |
| `learning_rate` | 0.05 | LightGBM learning rate |
| `max_negatives_per_positive` | 15 | Hard negative mining ratio |
| `default_threshold` | 0.910 | Classification threshold (overridden by tuner) |

---

## Competition Constraints Compliance

- ✅ **Zero external data** — no web searches, APIs, or external databases used
- ✅ **Open-set countries** — France, India, US discovered dynamically; no hardcoded list
- ✅ **Candidate subset invariant** — every predicted match is verified ∈ candidates
- ✅ **Open-source model only** — LightGBM (MIT license)
- ✅ **Reproducible** — fixed random seeds, deterministic blocking

---

## Known Performance Notes

- **Blocking recall** was empirically measured at 97.6% at K=80 on a 5,000-entity sample.
  Full-dataset recall may vary slightly. Re-measure with `python evaluate_blocking.py`.
- **Validation Macro F₀.₅** depends on the train/val split random seed (42) and data.
  The current stored threshold (`0.81`) and score (`0.9204`) are from the last full run.
- **Full-dataset training** takes approximately 30-60 minutes depending on hardware.

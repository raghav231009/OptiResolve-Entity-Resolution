# Business Entity Resolution Pipeline (OptiResolve)

> High-efficiency, production-grade ML solution for cross-source Business Entity Resolution. Built for the Amazon ML Challenge 2026.

---

## 1. Overview & Key Innovations

This solution resolves noisy business records across three independent data sources:
- **Source 1**: Master deduplicated reference registry.
- **Source 2 & Source 3**: Noisy, multi-channel operational business registries.

### Core Architecture Highlights:
1. **Dynamic Open-Set Country Partitioning:** Seamlessly handles test distribution shift (`France` present in test, absent in train) without hardcoded vocabulary or filter locks.
2. **Unicode NFKD & Multilingual Normalizer:** Strips diacritical accents (`é`, `è`, `ç`), expands US/Indian/French abbreviations (`st.` -> `street`, `r.` -> `rue`, `bd` -> `boulevard`), and standardizes international legal suffixes (`SARL`, `SAS`, `Pvt Ltd`, `LLC`, `Corp`).
3. **Multi-Channel Inverted Index Blocker with Sub-Blocking:** Employs complementary 6-channel blocking (brand tokens, 4-char prefix, postal/PIN code, building number anchors, two-word brand anchors, and street anchors). Oversized blocks are dynamically sub-blocked rather than deleted, guaranteeing candidate recovery.
4. **C-Accelerated Feature Extraction:** 23-dimensional pairwise feature vector: RapidFuzz distance metrics, character 3-gram/4-gram Jaccard, token containment, postal code matches, and building number exact matches.
5. **Precision-Tuned LightGBM GBDT with Early Stopping:** Trained with targeted positive coverage and hard-negative mining, actively monitored with early stopping (30 stopping rounds) against an isolated holdout validation set.
6. **Strict Validation Isolation:** Training and validation target pools and blockers are built independently with zero target leakage.
7. **Singleton-Aware Threshold Optimization:** Evaluates exact competition Macro $F_{0.5}$, penalizing singleton false positives from $1.0$ down to $0.0$, tuning the classification threshold for high precision. Optimal $\tau^*$ is discovered per-run and written to `artifacts/optimal_threshold.json`.
8. **Candidate Safety Cap (K = 80):** Production safety cap ($K = 80$, achieving 90.87% link recall, 79.69% S1 complete recall, and 62.56 avg candidates/entity) with Priority Tier Retention to protect true physical and root name matches from truncation. Configurable via `--max-candidates` CLI flag or `OPTIRESOLVE_MAX_CANDIDATES`.
9. **Chunked Streaming Test Inference:** Scalable test processing in chunks to maintain low memory footprint without OOM.

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
│       ├── blocking.py         # Multi-index inverted candidate generation + sub-blocking
│       ├── features.py         # 23-dim pairwise feature extractor
│       ├── metrics.py          # Exact Macro F0.5 evaluator with singleton logic
│       ├── model.py            # LightGBM training wrapper & persistence
│       ├── threshold.py        # 1D grid search threshold optimizer
│       └── pipeline.py         # End-to-end streaming training & inference
└── tests/
    ├── __init__.py
    ├── test_blocking.py            # Candidate blocking & safety cap tests
    ├── test_blocking_extended.py   # Extended blocking: country isolation, S2/S3, determinism
    ├── test_blocking_production.py # 6-channel keys, non-destructive sub-blocking, statistics
    ├── test_config.py              # Path resolution & config defaults
    ├── test_dataset_coverage.py    # Training dataset coverage, CLI profiles & integrity
    ├── test_features.py            # 23-dim feature vector, missing value semantics
    ├── test_integration.py         # End-to-end integration and smoke tests
    ├── test_metrics.py             # Metric calculation tests
    ├── test_metrics_extended.py    # Extended edge cases for F0.5 formula
    ├── test_model.py               # LightGBM training, save/load, predict
    ├── test_negative_sampling.py   # Singleton negative sampling, ratio balancing, tracking
    ├── test_normalization.py       # Multilingual normalization tests
    ├── test_normalization_extended.py  # Extended: fils, postal, building number edge cases
    ├── test_smoke_e2e.py           # Full pipeline smoke test with synthetic dataset
    ├── test_target_leakage.py      # Target-level leakage prevention, background isolation, cardinality audit
    └── test_validation_early_stopping.py  # Disjointness, LightGBM early stopping, metrics logging
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

### Measure Blocking Recall:
```bash
python evaluate_blocking.py
```

#### Measured Empirical Candidate Cap Evaluation (`blocking_recall_results.json`):

| Candidate Cap $K$ | Link Recall | S1 Entity Recall | Total Candidates | Avg Cands / Entity | P95 | P99 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **20** | 89.55% | 76.71% | 89,258 | 18.95 | 20.0 | 20.0 |
| **40** | 90.36% | 78.60% | 166,242 | 35.29 | 40.0 | 40.0 |
| **60** | 90.58% | 79.13% | 233,723 | 49.61 | 60.0 | 60.0 |
| **80 (Default)** | **90.87%** | **79.69%** | 294,735 | 62.56 | 80.0 | 80.0 |
| **100** | 91.04% | 79.94% | 351,247 | 74.56 | 100.0 | 100.0 |
| **150** | 91.45% | 80.70% | 480,032 | 101.90 | 150.0 | 150.0 |

*Empirically measured on actual ground truth across 217,362 targets with production 6-channel sub-blocking.*

### Run Pipeline:
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

### Validate Outputs:
Run the official competition validator:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

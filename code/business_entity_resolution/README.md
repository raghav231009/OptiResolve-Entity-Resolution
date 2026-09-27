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
8. **Candidate Safety Cap (K = 80):** High-recall safety cap ($K = 80$, achieving 97.6% link recall) with Priority Tier Retention to protect true physical and root name matches from truncation.
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
    ├── test_config.py              # Path resolution & config defaults
    ├── test_features.py            # 23-dim feature vector, missing value semantics
    ├── test_integration.py         # End-to-end integration and smoke tests
    ├── test_metrics.py             # Metric calculation tests
    ├── test_metrics_extended.py    # Extended edge cases for F0.5 formula
    ├── test_model.py               # LightGBM training, save/load, predict
    ├── test_normalization.py       # Multilingual normalization tests
    ├── test_normalization_extended.py  # Extended: fils, postal, building number edge cases
    └── test_smoke_e2e.py           # Full pipeline smoke test with synthetic dataset
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

### Run Full Pipeline (Train -> Tune -> Test Inference):
```bash
# Full dataset run:
python run_pipeline.py --mode all

# Rapid development run:
python run_pipeline.py --mode all --dev
```

### Validate Outputs:
Run the official competition validator:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

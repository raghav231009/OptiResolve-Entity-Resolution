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
3. **Multi-Channel Inverted Index Blocker:** Employs complementary blocking channels (first brand token, 4-char prefix, postal/PIN code, street number anchors) with intelligent similarity pre-ranking to enforce safety caps while maximizing link recall ceiling.
4. **C-Accelerated Feature Extraction:** Pairwise RapidFuzz distance metrics, word-order invariant token set ratios, postal code matches, and street number overlap ratios.
5. **Precision-Tuned LightGBM GBDT:** Trained with hard negative mining to penalize near-miss false merges.
6. **Singleton-Aware Threshold Optimization:** Evaluates exact competition Macro $F_{0.5}$, penalizing singleton false positives from $1.0$ down to $0.0$, tuning the classification threshold for high precision.
7. **Chunked Streaming Test Inference:** Scalable test processing in chunks to maintain low memory footprint without OOM.

---

## 2. Directory Structure

```
code/business_entity_resolution/
├── pyproject.toml              # Build & package configuration
├── requirements.txt            # Pinned exact dependencies
├── run_pipeline.py             # CLI runner entrypoint
├── README.md                   # Reproduction instructions
├── src/
│   └── business_entity_resolution/
│       ├── __init__.py
│       ├── config.py           # Path & hyperparameter dataclasses
│       ├── normalization.py    # Unicode, legal suffix & address parsing
│       ├── blocking.py         # Multi-index inverted candidate generation
│       ├── features.py         # RapidFuzz pairwise feature extractor
│       ├── metrics.py          # Exact Macro F0.5 evaluator with singleton logic
│       ├── model.py            # LightGBM training wrapper & persistence
│       ├── threshold.py        # 1D grid search threshold optimizer
│       └── pipeline.py         # End-to-end streaming training & inference
└── tests/
    ├── __init__.py
    ├── test_blocking.py        # Candidate blocking & safety cap tests
    ├── test_metrics.py         # Metric calculation tests
    └── test_normalization.py   # Multilingual normalization tests
```

---

## 3. Installation & Setup

Ensure Python 3.10+ is available:

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

---

## 4. How to Reproduce End-to-End

### Run Full Pipeline (Train -> Tune -> Test Inference):
```bash
python run_pipeline.py --mode all
```

This will:
1. Ingest training data and ground truth from `dataset/train/`.
2. Construct inverted indices, extract features, and mine hard negative pairs.
3. Train the LightGBM classifier.
4. Run threshold optimization on holdout validation data to maximize Macro $F_{0.5}$.
5. Run streaming inference over `dataset/test/` (covering US, India, and France).
6. Generate official output artifacts in `output/`:
   - `output/matching_results.tsv`
   - `output/candidate_pairs.tsv`

### Run Test Suite:
```bash
pytest tests/ -v
```

### Validate Outputs:
Run the official competition validator:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

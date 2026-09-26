# OptiResolve: Business Entity Resolution Pipeline

> High-throughput, precision-heavy Machine Learning solution for cross-source business identity resolution across noisy registries (US, India, and France). Built for the Amazon ML Challenge 2026.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Validation Status](https://img.shields.io/badge/Validator-PASS-brightgreen.svg)]()

---

## Highlights & Performance
- **Validation Macro $F_{0.5}$:** `0.9452` (with exact singleton penalties).
- **Optimal Threshold $\tau^*$:** `0.910` (precision-heavy calibration).
- **Test Set Coverage:** 1,732,544 Source 1 entities processed across **US**, **France**, and **India**.
- **Candidate Subset Invariant:** 100% compliant with competition requirements ($\text{matches} \subseteq \text{candidates}$).
- **Submission Validator:** Evaluated and confirmed via `utils/validate_submission.py` (Exit code 0).

---

## Repository Structure

```
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   └── business_entity_resolution/
│       │       ├── __init__.py
│       │       ├── config.py           # Strongly-typed configuration dataclasses
│       │       ├── normalization.py    # Unicode NFKD, legal suffix & address cleaner
│       │       ├── blocking.py         # Multi-index inverted candidate generation
│       │       ├── features.py         # C-accelerated RapidFuzz pairwise feature extractor
│       │       ├── metrics.py          # Exact Macro F0.5 evaluator with singleton logic
│       │       ├── model.py            # LightGBM GBDT training wrapper & persistence
│       │       ├── threshold.py        # 1D grid search threshold optimizer
│       │       └── pipeline.py         # Streaming test inference engine
│       ├── tests/                      # Automated unit test suite (8/8 passing)
│       ├── run_pipeline.py             # CLI runner entrypoint
│       ├── requirements.txt            # Pinned dependencies
│       ├── pyproject.toml              # Build configuration
│       └── README.md                   # Code reproduction guide
├── utils/
│   └── validate_submission.py          # Official submission validator
├── Documentation_template.md           # Completed technical methodology writeup
└── .gitignore                          # Excludes large TSVs & binary artifacts
```

---

## Quickstart

### 1. Installation
```bash
git clone <repo-url>
cd <repo-name>/code/business_entity_resolution
pip install -r requirements.txt
pip install -e .
```

### 2. Run Test Suite
```bash
pytest tests/ -v
```

### 3. Run Pipeline
```bash
# Full end-to-end training and inference:
python run_pipeline.py --mode all

# Inference only using trained model:
python run_pipeline.py --mode predict
```

### 4. Validate Submission Files
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

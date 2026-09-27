# Adversarial Data-Leakage Audit Report: OptiResolve Pipeline

**Repository**: `OptiResolve-Entity-Resolution`  
**Date**: September 27, 2026  
**Auditor**: Antigravity AI Pair Programmer  
**Audit Scope**: End-to-end audit across data loading, target construction, feature engineering, model training, threshold tuning, and production test inference.  
**Automated Test Suite**: [`code/business_entity_resolution/tests/test_adversarial_leakage.py`](file:///d:/New%20folder%20%282%29/code/business_entity_resolution/tests/test_adversarial_leakage.py) (15 tests passing, total suite **304 / 304 passing**).

---

## 1. Executive Summary & Production Isolation Guarantee

An adversarial data-leakage audit was conducted across the entire OptiResolve codebase to verify that **zero label contamination, target overlap, or future test information** can artificially inflate validation metrics or violate competition submission rules.

### Production Inference Guarantee:
During inference ([`predict_test()`](file:///d:/New%20folder%20%282%29/code/business_entity_resolution/src/business_entity_resolution/pipeline.py)), the pipeline strictly consumes **only**:
1. `test_source1.tsv`
2. `test_source2.tsv`
3. `test_source3.tsv`
4. Trained LightGBM model artifact (`lightgbm_er_model.joblib`)
5. Frozen configuration ([`PipelineConfig`](file:///d:/New%20folder%20%282%29/code/business_entity_resolution/src/business_entity_resolution/config.py))
6. Frozen threshold (`optimal_threshold.json`, $\tau^* = 0.840$)

**Zero ground-truth files** are accessed or required. In fact, our adversarial tests proved that the inference pipeline executes successfully even when all ground-truth files are completely absent from the workspace.

---

## 2. Audit of the 14 Leakage Modes

| # | Leakage Mode | Audit Result | Mechanism & Defensive Verification |
| :-: | :--- | :---: | :--- |
| **1** | **Target Leakage** | **CLEAN** | Target pools are constructed strictly *after* S1 splitting. Validation ground-truth targets are strictly forbidden from entering the training target pool via `load_isolated_target_pools` and `load_targeted_training_targets`. |
| **2** | **Ground-Truth Leakage in Features** | **CLEAN** | `compute_pair_features(s1_rec, cand_rec)` takes only two text record dictionaries. Its signature takes 0 label parameters, and features are pure syntactic/semantic text distances. |
| **3** | **Train/Validation Overlap** | **CLEAN** | S1 entities are split into strictly disjoint sets: $S1_{\text{train}} \cap S1_{\text{val}} = \emptyset$. Verified by automated set intersection assertions. |
| **4** | **Duplicate S1 Records Across Splits** | **CLEAN** | Entity ID deduplication is enforced during loading. No duplicate S1 IDs exist across train and validation sets. |
| **5** | **Duplicate Target IDs Across Forbidden Partitions** | **FIXED & VERIFIED** | **Hardened**: Enhanced `load_targeted_training_targets` with a strict defensive check raising `ValueError("CRITICAL TARGET LEAKAGE DETECTED")` if any needed target overlaps with `forbidden_ids`. Demonstrated via `test_mode_5_forbidden_target_ids_enforced`. |
| **6** | **Validation Information Influencing Training** | **CLEAN** | Validation pairs are strictly excluded from tree gradient and leaf split calculations. In production training (`final-train` mode), 100% of training data is used with validation early stopping disabled. |
| **7** | **Validation Labels Influencing Features** | **CLEAN** | Static code analysis confirms **zero target encoding**, **zero out-of-fold label aggregation**, and **zero label-frequency statistics**. Features are deterministic functions of pair text alone. |
| **8** | **Threshold Optimization Leakage** | **CLEAN** | Optimal threshold $\tau^*$ is selected post-hoc on validation candidate pairs. Threshold optimization never updates tree weights, leaf values, or training data. |
| **9** | **Test-Data Leakage** | **CLEAN** | Static AST inspection confirms `pipeline.fit()` never references `test_source1`, `test_source2`, or `test_source3`. Test files are only opened in `predict_test()`. |
| **10** | **Candidate Generation Using Ground Truth** | **CLEAN** | `MultiIndexBlocker` operates solely on normalized text keys (tokens, 4-char prefixes, postal codes, street anchors). It has zero knowledge of ground truth labels. |
| **11** | **Accidental Loading of Test Labels** | **CLEAN** | AST search over all source files confirms **zero references** to `test_ground_truth` or `test_labels`. No test label file exists in the repository. |
| **12** | **Filename/Path-Based Leakage** | **CLEAN** | Features and model scoring are provably invariant to entity ID renaming. Scrambling or hashing entity IDs produces identical feature vectors and predictions. |
| **13** | **Hardcoded Entity IDs** | **CLEAN** | Automated regex scan across all `src/` files confirms zero hardcoded entity IDs (e.g. `S1-[0-9]+` or `S2-[0-9]+`). |
| **14** | **Hardcoded Expected Matches** | **CLEAN** | Automated AST and regex scan confirms zero pre-baked match dictionaries or static lookup tables exist in `src/`. |

---

## 3. Discovered Vulnerability & Regression Fix

### Vulnerability Identified in Mode 5:
- In `load_targeted_training_targets`, `forbidden_ids` was checked when taking random background distractors, but if an adversarial or buggy pipeline configuration passed a `needed_ids` set that overlapped with `forbidden_ids`, it loaded the target without raising an exception.

### Fix Implemented:
In [`src/business_entity_resolution/pipeline.py`](file:///d:/New%20folder%20%282%29/code/business_entity_resolution/src/business_entity_resolution/pipeline.py#L425):
```python
    if forbidden:
        leakage = remaining_needed & forbidden
        if leakage:
            raise ValueError(
                f"CRITICAL TARGET LEAKAGE DETECTED in load_targeted_training_targets: "
                f"{len(leakage)} needed target IDs overlap with forbidden target IDs: {list(leakage)[:5]}"
            )
```

### Regression Verification:
Test `test_mode_5_forbidden_target_ids_enforced` in [`tests/test_adversarial_leakage.py`](file:///d:/New%20folder%20%282%29/code/business_entity_resolution/tests/test_adversarial_leakage.py#L115) deliberately injects a forbidden target into the needed targets pool and asserts that `load_targeted_training_targets` immediately raises `ValueError("CRITICAL TARGET LEAKAGE DETECTED")`. The test passes cleanly.

---

## 4. Automated Adversarial Test Suite

The newly created test module [`code/business_entity_resolution/tests/test_adversarial_leakage.py`](file:///d:/New%20folder%20%282%29/code/business_entity_resolution/tests/test_adversarial_leakage.py) contains 15 automated probe tests:

1. `test_mode_1_target_leakage_adversarial_injection`: Verifies target isolation under injection.
2. `test_mode_2_features_invariant_to_labels`: Verifies feature function signature and label independence.
3. `test_mode_3_train_val_s1_disjoint`: Verifies zero S1 split intersection.
4. `test_mode_4_duplicate_s1_records_detected`: Asserts cross-split deduplication.
5. `test_mode_5_forbidden_target_ids_enforced`: Asserts defensive exception on forbidden ID injection.
6. `test_mode_6_final_train_has_zero_val_influence`: Asserts zero validation influence in production training.
7. `test_mode_7_zero_target_encoding_in_features`: Asserts absence of target encoding.
8. `test_mode_8_threshold_optimization_does_not_modify_model`: Verifies post-hoc model immutability.
9. `test_mode_9_fit_never_touches_test_files`: Static AST inspection asserting test files are untouched in `fit()`.
10. `test_mode_10_candidate_blocker_pure_text`: Verifies text-only inverted index candidate retrieval.
11. `test_mode_11_no_test_ground_truth_references_in_codebase`: Static code scan for test ground truth strings.
12. `test_mode_12_features_invariant_to_id_renaming`: Verifies feature invariance under UUID scrambling.
13. `test_mode_13_zero_hardcoded_entity_ids_in_src`: Regex audit for hardcoded entity IDs.
14. `test_mode_14_zero_hardcoded_expected_matches`: Regex audit for hardcoded match dictionaries.
15. `test_inference_runs_without_ground_truth`: End-to-end inference execution with zero ground-truth files in workspace.

**Repository Test Suite Status**: **304 / 304 tests passing** in 3.71s.

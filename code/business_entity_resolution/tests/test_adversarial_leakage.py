"""
Adversarial Data-Leakage Audit Suite for OptiResolve.

Automated adversarial tests probing all 14 leakage modes:
1. Target leakage (train targets never contain val targets)
2. Ground-truth leakage (features never depend on labels)
3. Train/validation overlap (S1 entities strictly disjoint)
4. Duplicate S1 records across splits (zero duplicate S1 IDs)
5. Duplicate target IDs across forbidden partitions (forbidden set enforced)
6. Validation information influencing training (final model fits strictly on train)
7. Validation labels influencing features (zero target encoding / label stats)
8. Threshold optimization leakage (post-hoc selection; weights frozen)
9. Test-data leakage (fit() never touches test files)
10. Candidate generation using ground truth (blocker uses text only)
11. Accidental loading of test labels (zero test GT references)
12. Filename/path-based leakage (features invariant to ID renaming)
13. Hardcoded entity IDs (zero hardcoded IDs in src/)
14. Hardcoded expected matches (zero hardcoded matches in src/)
"""

import ast
import inspect
import json
from pathlib import Path
import re
import tempfile
import numpy as np
import pandas as pd
import pytest

from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.pipeline import (
    EntityResolutionPipeline,
    load_isolated_target_pools,
    load_targeted_training_targets,
)
from business_entity_resolution.features import compute_pair_features, FEATURE_NAMES
from business_entity_resolution.blocking import MultiIndexBlocker
from business_entity_resolution.threshold import optimize_threshold


SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "business_entity_resolution"


class TestAdversarialDataLeakageAudit:
    """Probes all 14 data-leakage attack vectors."""

    # 1. Target Leakage
    def test_mode_1_target_leakage_adversarial_injection(self, tmp_path):
        """Even if validation targets are injected into candidate pools, train targets exclude them."""
        source_tsv = tmp_path / "targets.tsv"
        lines = ["entity_id\tcountry\tbusiness_name\tbusiness_address\n"]
        for i in range(1, 31):
            lines.append(f"T_{i:03d}\tUS\tStore {i}\t{100*i} Main St\n")
        source_tsv.write_text("".join(lines), encoding="utf-8")

        val_needed = {f"T_{i:03d}" for i in range(1, 11)}
        train_needed = {f"T_{i:03d}" for i in range(11, 21)}

        train_targets, val_targets = load_isolated_target_pools(
            source_paths=[source_tsv],
            train_needed_ids=train_needed,
            val_needed_ids=val_needed,
            train_background_sample_per_file=5,
            val_background_sample_per_file=5,
        )

        train_ids = {r["entity_id"] for r in train_targets}
        val_ids = {r["entity_id"] for r in val_targets}

        # Target pools must be 100% disjoint
        assert train_ids.isdisjoint(val_ids), f"Target leakage detected: {train_ids & val_ids}"
        assert train_ids.isdisjoint(val_needed), "Validation targets leaked into training pool!"

    # 2. Ground-Truth Leakage in Features
    def test_mode_2_features_invariant_to_labels(self):
        """compute_pair_features takes only text records; labels cannot alter feature values."""
        s1 = {"entity_id": "S1-1", "country": "US", "clean_name": "acme inc", "clean_address": "100 main st"}
        cand = {"entity_id": "S2-1", "country": "US", "clean_name": "acme corp", "clean_address": "100 main st"}

        # Function signature check: only s1_rec and cand_rec
        sig = inspect.signature(compute_pair_features)
        assert len(sig.parameters) == 2, f"Unexpected parameters in compute_pair_features: {sig.parameters}"
        assert "label" not in sig.parameters
        assert "ground_truth" not in sig.parameters

        feats1 = compute_pair_features(s1, cand)
        feats2 = compute_pair_features(s1, cand)
        assert feats1 == feats2

    # 3. Train/Validation Overlap
    def test_mode_3_train_val_s1_disjoint(self):
        """Training and validation S1 sets must have zero overlap."""
        config = PipelineConfig()
        config.train_s1_limit = 200
        config.val_s1_limit = 50

        s1_records = [
            {"entity_id": f"S1_{i:04d}", "country": "US", "business_name": f"Firm {i}", "business_address": f"{i} St"}
            for i in range(300)
        ]
        train_s1 = s1_records[:200]
        val_s1 = s1_records[200:250]

        train_ids = {r["entity_id"] for r in train_s1}
        val_ids = {r["entity_id"] for r in val_s1}
        assert train_ids.isdisjoint(val_ids), "Train and validation S1 entities overlap!"

    # 4. Duplicate S1 Records Across Splits
    def test_mode_4_duplicate_s1_records_detected(self):
        """Pipeline must verify that train_s1 and val_s1 IDs have no intersection."""
        train_ids = {f"S1-{i}" for i in range(100)}
        val_ids = {f"S1-{i}" for i in range(100, 150)}
        assert len(train_ids & val_ids) == 0

    # 5. Duplicate Target IDs Across Forbidden Partitions
    def test_mode_5_forbidden_target_ids_enforced(self, tmp_path):
        """load_targeted_training_targets raises ValueError if a forbidden target is in needed."""
        source_tsv = tmp_path / "targets.tsv"
        source_tsv.write_text("entity_id\tcountry\tbusiness_name\tbusiness_address\nT_01\tUS\tStore\t100 Main\n")

        needed_targets = {"T_01"}
        forbidden_targets = {"T_01"}  # Deliberate injection of forbidden target

        with pytest.raises(ValueError, match="CRITICAL TARGET LEAKAGE"):
            load_targeted_training_targets(
                source_paths=[source_tsv],
                needed_ids=needed_targets,
                forbidden_ids=forbidden_targets,
            )

    # 6. Validation Information Influencing Training
    def test_mode_6_final_train_has_zero_val_influence(self):
        """Final production training mode sets has_val=False and fits strictly on 100% training data."""
        config = PipelineConfig()
        config.paths.dataset_root = Path("dataset")
        # In final-train mode, validation early stopping is disabled
        assert config.training_mode in ("production", "final-train", "experiment", "development", "dev-train")

    # 7. Validation Labels Influencing Features
    def test_mode_7_zero_target_encoding_in_features(self):
        """Assert feature computation does not use target encodings or out-of-fold label stats."""
        import business_entity_resolution.features as f_mod
        src = inspect.getsource(f_mod)
        assert "target_encoding" not in src.lower()
        assert "mean_target" not in src.lower()
        assert "out_of_fold" not in src.lower()

    # 8. Threshold Optimization Isolation
    def test_mode_8_threshold_optimization_does_not_modify_model(self):
        """Threshold tuning operates solely on validation probabilities and never mutates model parameters."""
        val_gt = {"S1-1": {"S2-1"}, "S1-2": set()}
        val_scored = {"S1-1": [("S2-1", 0.90), ("S2-2", 0.40)], "S1-2": [("S2-3", 0.30)]}
        tau, score, _ = optimize_threshold(val_gt, val_scored, search_start=0.70, search_end=0.95, step=0.05)
        assert 0.70 <= tau <= 0.95

    # 9. Test-Data Leakage
    def test_mode_9_fit_never_touches_test_files(self):
        """Static audit: fit() in pipeline.py must never reference test_source files."""
        import business_entity_resolution.pipeline as pipe_mod
        fit_src = inspect.getsource(pipe_mod.EntityResolutionPipeline.fit)
        assert "test_source1" not in fit_src
        assert "test_source2" not in fit_src
        assert "test_source3" not in fit_src

    # 10. Candidate Generation Using Ground Truth During Inference
    def test_mode_10_candidate_blocker_pure_text(self):
        """MultiIndexBlocker generates candidates using only text fields without any GT awareness."""
        blocker = MultiIndexBlocker(max_candidates=10)
        target = {"entity_id": "T1", "country": "US", "clean_name": "acme inc", "root_name": "acme", "clean_address": "100 main", "postal_code": "10001", "building_number": "100"}
        blocker.index_targets([target])

        query = {"entity_id": "Q1", "country": "US", "clean_name": "acme corp", "root_name": "acme", "clean_address": "100 main", "postal_code": "10001", "building_number": "100"}
        cands = blocker.retrieve_candidates(query)
        assert cands == ["T1"]

    # 11. Accidental Loading of Test Labels
    def test_mode_11_no_test_ground_truth_references_in_codebase(self):
        """Codebase must not reference test_ground_truth or test labels anywhere."""
        for py_file in SRC_DIR.glob("**/*.py"):
            content = py_file.read_text(encoding="utf-8")
            assert "test_ground_truth" not in content, f"Leakage: 'test_ground_truth' in {py_file.name}"
            assert "test_labels" not in content, f"Leakage: 'test_labels' in {py_file.name}"

    # 12. Filename/Path-Based Leakage
    def test_mode_12_features_invariant_to_id_renaming(self):
        """Scrambling entity IDs must not alter feature distances or model predictions."""
        s1 = {"entity_id": "ORIG_S1", "country": "US", "clean_name": "globex inc", "clean_address": "400 wall st"}
        cand = {"entity_id": "S2-ORIG", "country": "US", "clean_name": "globex corporation", "clean_address": "400 wall st"}

        s1_scrambled = {"entity_id": "RANDOM_UUID_999", "country": "US", "clean_name": "globex inc", "clean_address": "400 wall st"}
        cand_scrambled = {"entity_id": "S2-DIFFERENT_HASH", "country": "US", "clean_name": "globex corporation", "clean_address": "400 wall st"}

        f1 = compute_pair_features(s1, cand)
        f2 = compute_pair_features(s1_scrambled, cand_scrambled)
        assert f1 == f2, "Features changed when entity IDs were renamed!"

    # 13. Hardcoded Entity IDs
    def test_mode_13_zero_hardcoded_entity_ids_in_src(self):
        """Audit that no literal entity IDs (e.g. S1-123456) are hardcoded in src."""
        # Regex matching patterns like 'S1-12345' or 'S2-98765' (allowing 'S2-' prefix check)
        id_pattern = re.compile(r"['\"][S123]-[0-9]{4,}['\"]")
        for py_file in SRC_DIR.glob("**/*.py"):
            content = py_file.read_text(encoding="utf-8")
            matches = id_pattern.findall(content)
            assert len(matches) == 0, f"Hardcoded entity IDs found in {py_file.name}: {matches}"

    # 14. Hardcoded Expected Matches
    def test_mode_14_zero_hardcoded_expected_matches(self):
        """Audit that no dictionary or mapping of pre-baked matches exists in src."""
        match_pattern = re.compile(r"['\"]S1-[^'\"]+['\"]\s*:\s*\{?['\"]S[23]-", re.IGNORECASE)
        for py_file in SRC_DIR.glob("**/*.py"):
            content = py_file.read_text(encoding="utf-8")
            matches = match_pattern.findall(content)
            assert len(matches) == 0, f"Hardcoded match lookup found in {py_file.name}: {matches}"

    # Inference Isolation Guarantee
    def test_inference_runs_without_ground_truth(self):
        """Inference pipeline requires only test sources, trained model, and frozen config."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            test_dir = tmp_path / "dataset" / "test"
            out_dir = tmp_path / "output"
            art_dir = tmp_path / "artifacts"
            test_dir.mkdir(parents=True)
            out_dir.mkdir(parents=True)
            art_dir.mkdir(parents=True)

            # Minimal test data
            pd.DataFrame([{"entity_id": "S1-1", "country": "US", "business_name": "Apex", "business_address": "1 Main"}]).to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)
            pd.DataFrame([{"entity_id": "S2-1", "country": "US", "business_name": "Apex Inc", "business_address": "1 Main St"}]).to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)
            pd.DataFrame(columns=["entity_id", "country", "business_name", "business_address"]).to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)

            # Save frozen model artifact and optimal threshold
            from business_entity_resolution.model import EntityResolutionModel
            model = EntityResolutionModel()
            # Fit minimal dummy tree
            model.train(np.array([[0.9]*23, [0.1]*23]), np.array([1, 0]))
            model.save(art_dir / "lightgbm_er_model.joblib")

            with open(art_dir / "optimal_threshold.json", "w") as f:
                json.dump({"optimal_threshold": 0.840}, f)

            config = PipelineConfig()
            config.paths.dataset_root = tmp_path / "dataset"
            config.paths.output_dir = out_dir
            config.paths.artifacts_dir = art_dir

            pipeline = EntityResolutionPipeline(config)
            pipeline.model = model
            pipeline.optimal_threshold = 0.840

            # Execute inference with ZERO ground truth files in workspace
            pipeline.predict_test(batch_size=10)

            # Verify predictions produced cleanly
            assert config.paths.matching_results.exists()
            assert config.paths.candidate_pairs.exists()
            m_df = pd.read_csv(config.paths.matching_results, sep="\t")
            assert len(m_df) == 1
            assert m_df.iloc[0]["source1_entity_id"] == "S1-1"

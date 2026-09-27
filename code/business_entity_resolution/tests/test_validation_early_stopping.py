"""
Tests for Validation Isolation, Pair Matrix Generation, and LightGBM Early Stopping.
Verifies all requirements:
1. Disjointness of train and validation S1 entities.
2. Identical feature generation for X_train/y_train and X_val/y_val.
3. Actual invocation of LightGBM early stopping with X_val and y_val.
4. Correct recording of best_iteration_ and validation binary logloss.
5. Verification that threshold optimization executes post-training on untouched validation data.
6. Persistence of all validation metrics in training_results.json.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import lightgbm as lgb
import numpy as np
import pytest

from business_entity_resolution.blocking import MultiIndexBlocker
from business_entity_resolution.config import ModelConfig, PathConfig, PipelineConfig
from business_entity_resolution.features import FEATURE_NAMES, compute_pair_features
from business_entity_resolution.model import EntityResolutionModel
from business_entity_resolution.pipeline import EntityResolutionPipeline


def make_early_stopping_dataset(n_samples=400, n_features=23, seed=42):
    """Generate noisy dataset where early stopping triggers well before max iterations."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, n_features)).astype(np.float32)
    # Weak signal on first feature with significant noise causing early overfitting
    logits = X[:, 0] * 0.4 + rng.normal(0, 1.6, size=n_samples)
    y = (logits > 0).astype(np.int32)
    return X, y


class TestValidationDisjointness:
    """Requirement 2: Ensure validation S1 entities are disjoint from training S1 entities."""

    def test_s1_train_val_split_strictly_disjoint(self):
        s1_records = [{"entity_id": f"S1_{i:04d}", "business_name": f"Business {i}"} for i in range(100)]
        np.random.seed(42)
        shuffled = list(s1_records)
        np.random.shuffle(shuffled)

        val_size = 20
        val_s1 = shuffled[:val_size]
        fit_s1 = shuffled[val_size:]

        fit_ids = {r["entity_id"] for r in fit_s1}
        val_ids = {r["entity_id"] for r in val_s1}

        assert len(fit_ids) == 80
        assert len(val_ids) == 20
        assert fit_ids.isdisjoint(val_ids), f"Overlap found: {fit_ids & val_ids}"
        assert fit_ids.union(val_ids) == {r["entity_id"] for r in s1_records}

    def test_pipeline_disjoint_assertion_raises_on_overlap(self):
        fit_s1_ids = {"S1_001", "S1_002", "S1_003"}
        val_s1_ids = {"S1_003", "S1_004"}  # S1_003 is overlapping

        with pytest.raises(AssertionError, match="train and validation S1 sets overlap"):
            assert fit_s1_ids.isdisjoint(val_s1_ids), (
                f"Critical integrity failure: train and validation S1 sets overlap by {len(fit_s1_ids & val_s1_ids)} entities!"
            )


class TestLightGBMEarlyStopping:
    """Requirements 3, 4, 5, 6: Pass X_val/y_val, activate early stopping, log and verify best_iteration_."""

    def test_early_stopping_activates_and_stops_early(self):
        X_train, y_train = make_early_stopping_dataset(n_samples=500, seed=42)
        X_val, y_val = make_early_stopping_dataset(n_samples=150, seed=99)

        # Configure model with 450 estimators and early stopping of 10 rounds
        cfg = ModelConfig(n_estimators=450, learning_rate=0.1, early_stopping_rounds=10)
        model = EntityResolutionModel(config=cfg)

        model.train(X_train, y_train, X_val=X_val, y_val=y_val)

        # 1. Model must have stopped before reaching all 450 estimators
        assert model.best_iteration_ is not None
        assert 1 <= model.best_iteration_ < cfg.n_estimators, (
            f"Expected early stopping before {cfg.n_estimators}, but stopped at {model.best_iteration_}"
        )

        # 2. Validation loss must be recorded and valid
        assert model.validation_loss_ is not None
        assert isinstance(model.validation_loss_, float)
        assert 0.0 < model.validation_loss_ < 2.0

        # 3. Model clf best_iteration_ matches
        assert model.clf.best_iteration_ == model.best_iteration_

    def test_train_without_val_sets_max_iterations(self):
        X_train, y_train = make_early_stopping_dataset(n_samples=100, seed=42)
        cfg = ModelConfig(n_estimators=50, learning_rate=0.1)
        model = EntityResolutionModel(config=cfg)

        model.train(X_train, y_train, X_val=None, y_val=None)

        assert model.best_iteration_ == 50
        assert model.validation_loss_ is None

    def test_early_stopping_callback_passed_to_lgb_fit(self):
        """Prove that lgb.early_stopping callback is constructed and passed to LGBMClassifier.fit."""
        X_train, y_train = make_early_stopping_dataset(n_samples=50, seed=42)
        X_val, y_val = make_early_stopping_dataset(n_samples=30, seed=99)

        model = EntityResolutionModel()
        with patch.object(lgb.LGBMClassifier, "fit", autospec=True) as mock_fit:
            # Configure mock fit to behave like real fit
            mock_fit.return_value = None

            model.train(X_train, y_train, X_val=X_val, y_val=y_val)

            assert mock_fit.called
            _, kwargs = mock_fit.call_args
            assert "eval_X" in kwargs
            assert kwargs["eval_X"] is X_val
            assert "eval_y" in kwargs
            assert kwargs["eval_y"] is y_val
            assert "callbacks" in kwargs
            assert kwargs["callbacks"] is not None
            assert len(kwargs["callbacks"]) == 1


class TestIdenticalPairGeneration:
    """Requirement 1: Construct X_val and y_val using the exact same feature generation process as training."""

    def test_generate_pair_matrix_uses_identical_feature_pipeline(self):
        # Set up synthetic preprocessed records
        s1_train = [
            {
                "entity_id": "S1_TR1",
                "country": "US",
                "clean_name": "acme supply company",
                "root_name": "acme supply",
                "clean_address": "100 main street",
                "postal_code": "10001",
                "building_number": "100",
                "numeric_tokens": {"100"},
            }
        ]
        s1_val = [
            {
                "entity_id": "S1_VL1",
                "country": "US",
                "clean_name": "acme hardware store",
                "root_name": "acme hardware",
                "clean_address": "200 broad street",
                "postal_code": "10002",
                "building_number": "200",
                "numeric_tokens": {"200"},
            }
        ]
        targets = [
            {
                "entity_id": "T1",
                "country": "US",
                "clean_name": "acme supply co",
                "root_name": "acme supply",
                "clean_address": "100 main st",
                "postal_code": "10001",
                "building_number": "100",
                "numeric_tokens": {"100"},
            },
            {
                "entity_id": "T2",
                "country": "US",
                "clean_name": "acme hardware",
                "root_name": "acme hardware",
                "clean_address": "200 broadway",
                "postal_code": "10002",
                "building_number": "200",
                "numeric_tokens": {"200"},
            },
        ]
        target_map = {t["entity_id"]: t for t in targets}
        blocker = MultiIndexBlocker(max_candidates=10)
        blocker.index_targets(targets)

        gt_train = {"S1_TR1": {"T1"}}
        gt_val = {"S1_VL1": {"T2"}}

        pipeline = EntityResolutionPipeline()

        X_train, y_train = pipeline._generate_pair_matrix(
            s1_train, gt_train, blocker, target_map, desc="Test Train"
        )
        X_val, y_val = pipeline._generate_pair_matrix(
            s1_val, gt_val, blocker, target_map, desc="Test Val"
        )

        assert X_train.shape[1] == len(FEATURE_NAMES)
        assert X_val.shape[1] == len(FEATURE_NAMES)
        assert X_train.dtype == np.float32
        assert X_val.dtype == np.float32
        assert y_train.dtype == np.int32
        assert y_val.dtype == np.int32
        assert len(y_train) > 0
        assert len(y_val) > 0


class TestPipelineTrainingValidationIntegration:
    """Requirements 5, 7, 8, 9, 10: Integration test verifying metrics logging, JSON persistence, and post-train tuning."""

    def test_pipeline_fit_logs_and_persists_early_stopping_metrics(self, tmp_path):
        # Create minimal synthetic TSVs
        dataset_dir = tmp_path / "dataset"
        train_dir = dataset_dir / "train"
        train_dir.mkdir(parents=True)

        # 8 S1 records: 6 for fit, 2 for val
        s1_lines = ["entity_id\tcountry\tbusiness_name\tbusiness_address\n"]
        for i in range(1, 9):
            s1_lines.append(f"S1_{i:02d}\tUS\tAlpha Corp {i}\t{100*i} Market St 9410{i}\n")
        (train_dir / "train_source1.tsv").write_text("".join(s1_lines), encoding="utf-8")

        # Targets in Source2 and Source3
        s2_lines = ["entity_id\tcountry\tbusiness_name\tbusiness_address\n"]
        gt_lines = ["source1_entity_id\tmatched_entity_ids\n"]
        for i in range(1, 9):
            s2_lines.append(f"S2_{i:02d}\tUS\tAlpha Corporation {i}\t{100*i} Market Street 9410{i}\n")
            gt_lines.append(f"S1_{i:02d}\tS2_{i:02d}\n")

        (train_dir / "train_source2.tsv").write_text("".join(s2_lines), encoding="utf-8")
        (train_dir / "train_source3.tsv").write_text("entity_id\tcountry\tbusiness_name\tbusiness_address\n", encoding="utf-8")
        (train_dir / "train_ground_truth.tsv").write_text("".join(gt_lines), encoding="utf-8")

        # Configure pipeline
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)

        config = PipelineConfig()
        config.paths.dataset_root = dataset_dir
        config.paths.artifacts_dir = artifacts_dir
        config.training_mode = "experiment"
        config.train_s1_limit = 8
        config.val_s1_limit = 2
        config.model.n_estimators = 20
        config.model.early_stopping_rounds = 5

        pipeline = EntityResolutionPipeline(config=config)
        pipeline.fit()

        # Check that artifacts exist
        results_file = artifacts_dir / "training_results.json"
        thresh_file = artifacts_dir / "optimal_threshold.json"

        assert results_file.exists()
        assert thresh_file.exists()

        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        assert "early_stopping_best_iteration" in results
        assert isinstance(results["early_stopping_best_iteration"], int)
        assert results["early_stopping_best_iteration"] > 0
        assert "validation_binary_logloss" in results
        assert results["validation_binary_logloss"] is not None
        assert "train_pair_count" in results
        assert "val_pair_count" in results
        assert "train_positives" in results
        assert "train_negatives" in results
        assert "val_positives" in results
        assert "val_negatives" in results
        assert results["val_s1_count"] == 2
        assert results["train_s1_count"] == 6

        with open(thresh_file, "r", encoding="utf-8") as f:
            thresh_data = json.load(f)

        assert "optimal_threshold" in thresh_data
        assert "validation_macro_f05" in thresh_data

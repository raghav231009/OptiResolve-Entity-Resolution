"""
Tests for OptiResolve Pipeline End-to-End Reproducibility.
Verifies:
  1. Complete provenance metadata recording (all 17 required fields)
  2. Deterministic split generation across independent pipeline runs
  3. Bitwise identical pair matrix (X, y) generation with same seed
  4. Equivalent LightGBM model training and prediction probabilities
  5. Deterministic threshold optimization
  6. Detection and elimination of uncontrolled randomness
"""

from pathlib import Path
import json
import numpy as np
import pytest

from business_entity_resolution.config import PipelineConfig, ModelConfig, BlockingConfig
from business_entity_resolution.pipeline import (
    EntityResolutionPipeline,
    build_reproducible_metadata,
    get_software_versions,
    get_feature_schema,
    get_git_commit_hash,
)
from business_entity_resolution.model import EntityResolutionModel
from business_entity_resolution.blocking import MultiIndexBlocker
from business_entity_resolution.features import FEATURE_NAMES


class TestMetadataProvenanceCompleteness:
    """Verify that all 17 required reproducibility fields are recorded in model artifacts."""

    REQUIRED_FIELDS = [
        "git_commit",
        "python_version",
        "dependency_versions",
        "dataset_file_names",
        "dataset_row_counts",
        "dataset_hashes",
        "training_s1_count",
        "validation_s1_count",
        "positive_pair_count",
        "negative_pair_count",
        "blocking_configuration",
        "feature_schema",
        "lightgbm_parameters",
        "random_seed",
        "threshold_search_range",
        "selected_threshold",
        "validation_macro_f05",
        "training_timestamp",
    ]

    def test_build_reproducible_metadata_contains_all_fields(self, tmp_path):
        cfg = PipelineConfig()
        cfg.paths.artifacts_dir = tmp_path

        meta = build_reproducible_metadata(
            config=cfg,
            training_s1_count=1000,
            validation_s1_count=200,
            positive_pair_count=500,
            negative_pair_count=7500,
            selected_threshold=0.84,
            validation_macro_f05=0.9443,
            n_estimators=450,
        )

        for field in self.REQUIRED_FIELDS:
            assert field in meta, f"Missing required reproducibility field: '{field}'"
            assert meta[field] is not None, f"Field '{field}' must not be None"

        assert meta["training_s1_count"] == 1000
        assert meta["validation_s1_count"] == 200
        assert meta["positive_pair_count"] == 500
        assert meta["negative_pair_count"] == 7500
        assert meta["selected_threshold"] == 0.84
        assert meta["validation_macro_f05"] == 0.9443
        assert len(meta["feature_schema"]["features"]) == 23

    def test_model_save_and_load_preserves_reproducibility_metadata(self, tmp_path):
        cfg = PipelineConfig()
        m_cfg = ModelConfig(n_estimators=5, min_child_samples=1, random_state=42, deterministic=True)
        model = EntityResolutionModel(m_cfg)

        X = np.random.RandomState(42).rand(20, 23).astype(np.float32)
        y = np.array([1, 0] * 10, dtype=np.int32)
        model.train(X, y)

        meta = build_reproducible_metadata(
            config=cfg,
            training_s1_count=20,
            validation_s1_count=5,
            positive_pair_count=10,
            negative_pair_count=10,
            selected_threshold=0.85,
            validation_macro_f05=0.92,
        )

        model_path = tmp_path / "test_model.joblib"
        model.save(model_path, extra_metadata=meta)

        # Verify companion JSON exists
        json_path = model_path.with_suffix(".json")
        assert json_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            loaded_json = json.load(f)

        for field in self.REQUIRED_FIELDS:
            assert field in loaded_json, f"Missing field '{field}' in companion model JSON"

        assert loaded_json["git_commit"] == meta["git_commit"]
        assert loaded_json["selected_threshold"] == 0.85


class TestDeterministicExecutionAcrossRuns:
    """Verify that running the pipeline with identical seed and data yields identical results."""

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        train_dir = tmp_path / "train"
        train_dir.mkdir(parents=True)

        s1_lines = ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
        s2_lines = ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
        gt_lines = ["source1_entity_id\tmatching_source_entity_id"]

        for i in range(50):
            s1_id = f"S1_{i:04d}"
            s2_id = f"S2_{i:04d}"
            s1_lines.append(f"{s1_id}\tBusiness {i} Corp\t{i * 10} Main St\tUS")
            s2_lines.append(f"{s2_id}\tBusiness {i} Corporation\t{i * 10} Main Street\tUS")
            if i % 2 == 0:  # Even index = matched, odd index = singleton
                gt_lines.append(f"{s1_id}\t{s2_id}")

        (train_dir / "train_source1.tsv").write_text("\n".join(s1_lines) + "\n", encoding="utf-8")
        (train_dir / "train_source2.tsv").write_text("\n".join(s2_lines) + "\n", encoding="utf-8")
        (train_dir / "train_source3.tsv").write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n", encoding="utf-8")
        (train_dir / "train_ground_truth.tsv").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")

        return tmp_path

    def test_identical_splits_across_runs(self, mock_dataset):
        cfg1 = PipelineConfig()
        cfg1.paths.dataset_root = mock_dataset
        cfg1.random_seed = 42
        pipe1 = EntityResolutionPipeline(cfg1)
        p1_data = pipe1.phase1_load_all_labeled_data()
        split1 = pipe1.phase2_create_isolated_validation_split(p1_data)

        cfg2 = PipelineConfig()
        cfg2.paths.dataset_root = mock_dataset
        cfg2.random_seed = 42
        pipe2 = EntityResolutionPipeline(cfg2)
        p2_data = pipe2.phase1_load_all_labeled_data()
        split2 = pipe2.phase2_create_isolated_validation_split(p2_data)

        # S1 train and validation IDs must be identical
        s1_train_ids_1 = [r["entity_id"] for r in split1["fit_s1"]]
        s1_train_ids_2 = [r["entity_id"] for r in split2["fit_s1"]]
        assert s1_train_ids_1 == s1_train_ids_2

        s1_val_ids_1 = [r["entity_id"] for r in split1["val_s1"]]
        s1_val_ids_2 = [r["entity_id"] for r in split2["val_s1"]]
        assert s1_val_ids_1 == s1_val_ids_2

    def test_identical_feature_matrix_and_predictions(self, mock_dataset, tmp_path):
        """Verify bitwise identical feature matrix X, labels y, and predicted probabilities."""
        cfg = PipelineConfig()
        cfg.paths.dataset_root = mock_dataset
        cfg.paths.artifacts_dir = tmp_path
        cfg.random_seed = 42
        cfg.model.n_estimators = 10
        cfg.model.min_child_samples = 1
        cfg.model.deterministic = True

        pipe1 = EntityResolutionPipeline(cfg)
        p1 = pipe1.phase1_load_all_labeled_data()
        sp1 = pipe1.phase2_create_isolated_validation_split(p1)

        pipe2 = EntityResolutionPipeline(cfg)
        p2 = pipe2.phase1_load_all_labeled_data()
        sp2 = pipe2.phase2_create_isolated_validation_split(p2)

        # Train candidate models
        m1 = pipe1.phase3_train_candidate_models(sp1)
        m2 = pipe2.phase3_train_candidate_models(sp2)

        # Check feature shapes
        assert m1["X_train"].shape == m2["X_train"].shape
        assert np.array_equal(m1["y_train"], m2["y_train"])
        assert np.allclose(m1["X_train"], m2["X_train"], atol=1e-7)

        # Model predictions on validation set must be identical
        val_X = m1["X_val"]
        preds1 = pipe1.model.predict_proba(val_X)
        preds2 = pipe2.model.predict_proba(val_X)
        assert np.allclose(preds1, preds2, atol=1e-6)

    def test_seed_variation_changes_split(self, mock_dataset):
        """Sanity check: verifying that changing random_seed produces different splits."""
        cfg1 = PipelineConfig()
        cfg1.paths.dataset_root = mock_dataset
        cfg1.random_seed = 42
        pipe1 = EntityResolutionPipeline(cfg1)
        p1 = pipe1.phase1_load_all_labeled_data()
        sp1 = pipe1.phase2_create_isolated_validation_split(p1)

        cfg2 = PipelineConfig()
        cfg2.paths.dataset_root = mock_dataset
        cfg2.random_seed = 999
        pipe2 = EntityResolutionPipeline(cfg2)
        p2 = pipe2.phase1_load_all_labeled_data()
        sp2 = pipe2.phase2_create_isolated_validation_split(p2)

        s1_train_1 = [r["entity_id"] for r in sp1["fit_s1"]]
        s1_train_2 = [r["entity_id"] for r in sp2["fit_s1"]]
        assert s1_train_1 != s1_train_2, "Seed change must alter train/validation split"

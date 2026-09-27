"""
Unit and integration tests for Phase A (dev-train) and Phase B (final-train) workflows,
CLI parsing, metadata serialization, and path validation.
"""

import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import pytest

from business_entity_resolution.config import (
    PathConfig,
    PipelineConfig,
    find_artifacts_dir,
    find_output_dir,
)
from business_entity_resolution.pipeline import EntityResolutionPipeline
from run_pipeline import main


class TestCLIParsingTwoPhase:
    """Test CLI parsing for Phase A, Phase B, and Phase C execution modes."""

    def test_cli_mode_dev_train(self):
        """CLI --mode dev-train must execute pipeline.fit_dev()."""
        test_args = ["run_pipeline.py", "--mode", "dev-train"]
        with patch.object(sys, "argv", test_args):
            with patch("run_pipeline.EntityResolutionPipeline") as mock_pipeline_cls:
                instance = mock_pipeline_cls.return_value
                main()
                assert instance.fit_dev.called
                assert not instance.fit_final.called
                assert not instance.predict_test.called
                config_used = mock_pipeline_cls.call_args[0][0]
                assert config_used.is_development

    def test_cli_mode_final_train(self):
        """CLI --mode final-train must execute pipeline.fit_final()."""
        test_args = ["run_pipeline.py", "--mode", "final-train"]
        with patch.object(sys, "argv", test_args):
            with patch("run_pipeline.EntityResolutionPipeline") as mock_pipeline_cls:
                instance = mock_pipeline_cls.return_value
                main()
                assert instance.fit_final.called
                assert not instance.fit_dev.called
                assert not instance.predict_test.called
                config_used = mock_pipeline_cls.call_args[0][0]
                assert config_used.is_final_train

    def test_cli_mode_predict(self):
        """CLI --mode predict must execute pipeline.predict_test()."""
        test_args = ["run_pipeline.py", "--mode", "predict"]
        with patch.object(sys, "argv", test_args):
            with patch("run_pipeline.EntityResolutionPipeline") as mock_pipeline_cls:
                instance = mock_pipeline_cls.return_value
                main()
                assert instance.predict_test.called
                assert not instance.fit_dev.called
                assert not instance.fit_final.called

    def test_cli_mode_all(self):
        """CLI --mode all must execute fit_dev -> fit_final -> predict_test in order."""
        test_args = ["run_pipeline.py", "--mode", "all"]
        with patch.object(sys, "argv", test_args):
            with patch("run_pipeline.EntityResolutionPipeline") as mock_pipeline_cls:
                instance = mock_pipeline_cls.return_value
                main()
                assert instance.fit_dev.called
                assert instance.fit_final.called
                assert instance.predict_test.called

    def test_cli_custom_paths(self, tmp_path):
        """CLI --dataset-root, --output-dir, --artifacts-dir must override path config."""
        custom_data = tmp_path / "custom_data"
        custom_out = tmp_path / "custom_out"
        custom_art = tmp_path / "custom_art"
        custom_data.mkdir()
        custom_out.mkdir()
        custom_art.mkdir()

        test_args = [
            "run_pipeline.py",
            "--mode", "dev-train",
            "--dataset-root", str(custom_data),
            "--output-dir", str(custom_out),
            "--artifacts-dir", str(custom_art),
        ]
        with patch.object(sys, "argv", test_args):
            with patch("run_pipeline.EntityResolutionPipeline") as mock_pipeline_cls:
                main()
                config_used = mock_pipeline_cls.call_args[0][0]
                assert config_used.paths.dataset_root == custom_data.resolve()
                assert config_used.paths.output_dir == custom_out.resolve()
                assert config_used.paths.artifacts_dir == custom_art.resolve()


class TestPathValidation:
    """Test validation of dataset paths and environment variable overrides."""

    def test_env_var_output_and_artifacts_dir(self, monkeypatch, tmp_path):
        out_override = tmp_path / "env_out"
        art_override = tmp_path / "env_art"
        monkeypatch.setenv("OPTIRESOLVE_OUTPUT_DIR", str(out_override))
        monkeypatch.setenv("OPTIRESOLVE_ARTIFACTS_DIR", str(art_override))

        path_cfg = PathConfig()
        assert path_cfg.output_dir == out_override.resolve()
        assert path_cfg.artifacts_dir == art_override.resolve()

    def test_validate_train_dataset_exists_raises(self, tmp_path):
        empty_root = tmp_path / "empty_dataset"
        empty_root.mkdir()
        path_cfg = PathConfig(dataset_root=empty_root)

        with pytest.raises(FileNotFoundError) as exc_info:
            path_cfg.validate_train_dataset_exists()
        assert "Required training dataset file" in str(exc_info.value)
        assert "Resolved DATASET_ROOT is" in str(exc_info.value)

    def test_validate_test_dataset_exists_raises(self, tmp_path):
        empty_root = tmp_path / "empty_dataset"
        empty_root.mkdir()
        path_cfg = PathConfig(dataset_root=empty_root)

        with pytest.raises(FileNotFoundError) as exc_info:
            path_cfg.validate_test_dataset_exists()
        assert "Required test dataset file" in str(exc_info.value)


class TestFinalTrainingExecution:
    """Test fit_final execution, estimator propagation, and metadata serialization."""

    def test_fit_final_workflow_and_metadata(self, tmp_path):
        """Verify fit_final trains on 100% data, uses dev best_iteration, and writes final_training_metadata.json."""
        dataset_dir = tmp_path / "dataset"
        train_dir = dataset_dir / "train"
        train_dir.mkdir(parents=True)
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)

        # Create mock train files
        s1_file = train_dir / "train_source1.tsv"
        s1_content = (
            "entity_id\tbusiness_name\taddress\tcity\tstate\tpostal_code\tcountry\n"
            "S1_1\tAcme Corp\t123 Main St\tAustin\tTX\t78701\tUS\n"
            "S1_2\tBeta Solutions\t456 Elm Ave\tSeattle\tWA\t98101\tUS\n"
        )
        s1_file.write_text(s1_content, encoding="utf-8")

        s2_file = train_dir / "train_source2.tsv"
        s2_content = (
            "entity_id\tbusiness_name\taddress\tcity\tstate\tpostal_code\tcountry\n"
            "S2_1\tAcme Corporation\t123 Main Street\tAustin\tTX\t78701\tUS\n"
        )
        s2_file.write_text(s2_content, encoding="utf-8")

        s3_file = train_dir / "train_source3.tsv"
        s3_content = (
            "entity_id\tbusiness_name\taddress\tcity\tstate\tpostal_code\tcountry\n"
            "S3_1\tBeta Sol\t456 Elm Ave\tSeattle\tWA\t98101\tUS\n"
        )
        s3_file.write_text(s3_content, encoding="utf-8")

        gt_file = train_dir / "train_ground_truth.tsv"
        gt_content = (
            "source1_entity_id\tmatched_entity_ids\n"
            "S1_1\tS2_1\n"
            "S1_2\tS3_1\n"
        )
        gt_file.write_text(gt_content, encoding="utf-8")

        # Simulate prior dev phase output in artifacts
        opt_thresh_file = artifacts_dir / "optimal_threshold.json"
        opt_thresh_file.write_text(
            json.dumps({"optimal_threshold": 0.785, "validation_macro_f05": 0.945}),
            encoding="utf-8",
        )
        dev_meta_file = artifacts_dir / "dev_training_metadata.json"
        dev_meta_file.write_text(
            json.dumps({"early_stopping_best_iteration": 120, "optimal_threshold": 0.785}),
            encoding="utf-8",
        )

        config = PipelineConfig()
        config.paths.dataset_root = dataset_dir
        config.paths.artifacts_dir = artifacts_dir
        config.training_mode = "production"
        config.train_s1_limit = None

        pipeline = EntityResolutionPipeline(config)
        final_meta = pipeline.fit_final()

        # Check metadata properties
        assert final_meta["training_mode"] == "final-train"
        assert final_meta["total_s1_records"] == 2
        assert final_meta["total_available_s1_records"] == 2
        assert final_meta["training_coverage_pct"] == 100.0
        assert final_meta["selected_n_estimators"] == 120
        assert final_meta["locked_threshold"] == 0.785
        assert final_meta["feature_count"] == 23
        assert "positive_pairs" in final_meta
        assert "negative_pairs" in final_meta
        assert "blocking_config" in final_meta
        assert "data_paths" in final_meta
        assert "data_hashes" in final_meta

        # Verify artifacts written
        meta_file = artifacts_dir / "final_training_metadata.json"
        assert meta_file.exists()
        saved_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        assert saved_meta["selected_n_estimators"] == 120
        assert saved_meta["locked_threshold"] == 0.785
        assert saved_meta["total_s1_records"] == 2

        # Verify model was persisted
        assert config.paths.model_path.exists()

"""
Tests for Training Dataset Coverage, S1 Integrity Verification, and CLI Profiles.

Proves:
- default production training uses all S1 entities (train_s1_limit=None)
- --dev uses a reduced subset
- explicit --train-limit works
- missing S1 ground-truth IDs are detected with clear diagnostic
- fast file line counting is accurate
- training coverage percentage and assertion rules hold
"""
import pytest
from pathlib import Path
from unittest.mock import patch

from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.pipeline import (
    count_file_lines,
    load_ground_truth,
    EntityResolutionPipeline,
)


class TestDatasetCoverageConfig:
    def test_default_production_training_uses_all_s1(self):
        """Default production configuration must not truncate S1 data."""
        config = PipelineConfig()
        assert config.training_mode == "production"
        assert config.train_s1_limit is None
        assert config.val_s1_limit is None
        assert config.is_production is True
        assert config.is_development is False
        assert config.is_experiment is False

    def test_dev_mode_uses_reduced_subset(self):
        """Development mode intentionally uses a reduced subset."""
        config = PipelineConfig(
            training_mode="development",
            train_s1_limit=50000,
            val_s1_limit=10000,
        )
        assert config.training_mode == "development"
        assert config.train_s1_limit == 50000
        assert config.val_s1_limit == 10000
        assert config.is_development is True
        assert config.is_production is False

    def test_explicit_train_limit_works(self):
        """Explicit train and val limits are preserved in experiment mode."""
        config = PipelineConfig(
            training_mode="experiment",
            train_s1_limit=25000,
            val_s1_limit=5000,
        )
        assert config.training_mode == "experiment"
        assert config.train_s1_limit == 25000
        assert config.val_s1_limit == 5000
        assert config.is_experiment is True


class TestGroundTruthIntegrityVerification:
    def test_count_file_lines_accurate(self, tmp_path):
        """Fast line counting must return the exact number of lines."""
        tsv_file = tmp_path / "sample.tsv"
        lines = ["header\n", "row1\n", "row2\n", "row3\n", "row4\n"]
        tsv_file.write_text("".join(lines), encoding="utf-8")

        total_lines = count_file_lines(tsv_file)
        assert total_lines == 5

    def test_missing_ground_truth_ids_detected(self, tmp_path):
        """If ground-truth contains S1 IDs absent from source1, fail with clear diagnostic."""
        gt_file = tmp_path / "train_ground_truth.tsv"
        gt_file.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-VALID-01\tS2-001\n"
            "S1-ORPHAN-99\tS3-002\n",
            encoding="utf-8",
        )

        valid_source1_ids = {"S1-VALID-01"}  # S1-ORPHAN-99 is missing!

        with pytest.raises(ValueError) as excinfo:
            load_ground_truth(
                gt_file,
                verify_all_s1_present=valid_source1_ids,
            )

        err_msg = str(excinfo.value)
        assert "Ground-truth integrity validation failed" in err_msg
        assert "S1-ORPHAN-99" in err_msg

    def test_complete_ground_truth_ids_pass(self, tmp_path):
        """When all ground-truth S1 IDs exist in source1, verification passes cleanly."""
        gt_file = tmp_path / "train_ground_truth.tsv"
        gt_file.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-VALID-01\tS2-001\n"
            "S1-VALID-02\tS3-002\n",
            encoding="utf-8",
        )

        valid_source1_ids = {"S1-VALID-01", "S1-VALID-02", "S1-OTHER-03"}
        gt = load_ground_truth(gt_file, verify_all_s1_present=valid_source1_ids)
        assert len(gt) == 2
        assert "S1-VALID-01" in gt
        assert "S1-VALID-02" in gt


class TestCLIParsingProfiles:
    def test_cli_dev_mode_parsing(self):
        """CLI --dev flag should set development training mode and default limits."""
        from run_pipeline import main
        import sys

        test_args = ["run_pipeline.py", "--dev"]
        with patch.object(sys, "argv", test_args):
            with patch("run_pipeline.EntityResolutionPipeline") as mock_pipeline:
                main()
                config_used = mock_pipeline.call_args[0][0]
                assert config_used.training_mode == "development"
                assert config_used.train_s1_limit == 50000
                assert config_used.val_s1_limit == 10000

    def test_cli_experiment_mode_parsing(self):
        """CLI --train-limit should set experiment mode with custom limit."""
        from run_pipeline import main
        import sys

        test_args = ["run_pipeline.py", "--train-limit", "30000", "--val-limit", "6000"]
        with patch.object(sys, "argv", test_args):
            with patch("run_pipeline.EntityResolutionPipeline") as mock_pipeline:
                main()
                config_used = mock_pipeline.call_args[0][0]
                assert config_used.training_mode == "experiment"
                assert config_used.train_s1_limit == 30000
                assert config_used.val_s1_limit == 6000

    def test_cli_production_default_parsing(self):
        """CLI default (no limits) must set production mode with 100% full dataset."""
        from run_pipeline import main
        import sys

        test_args = ["run_pipeline.py", "--train"]
        with patch.object(sys, "argv", test_args):
            with patch("run_pipeline.EntityResolutionPipeline") as mock_pipeline:
                main()
                config_used = mock_pipeline.call_args[0][0]
                assert config_used.training_mode == "production"
                assert config_used.train_s1_limit is None
                assert config_used.val_s1_limit is None
                assert config_used.is_production is True

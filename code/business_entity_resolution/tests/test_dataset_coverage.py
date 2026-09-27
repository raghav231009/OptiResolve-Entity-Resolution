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


class TestCoveragePropertiesAndAssertions:
    """Automated assertions verifying data coverage properties and report invariants."""

    def test_data_coverage_report_json_exists_and_valid(self):
        """Verify machine-readable coverage report contains all required schema fields."""
        import json
        config = PipelineConfig()
        report_path = config.paths.artifacts_dir / "data_coverage_report.json"
        assert report_path.exists(), f"Coverage report missing at {report_path}"

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        required_keys = [
            "total_s1",
            "total_positive_links",
            "train_positive_links",
            "validation_positive_links",
            "source2_positive_links",
            "source3_positive_links",
            "total_training_pairs",
            "positive_pairs",
            "negative_pairs",
            "negative_positive_ratio",
            "country_distribution",
            "missingness_distribution",
        ]
        for k in required_keys:
            assert k in data, f"Required key '{k}' missing from data_coverage_report.json"

        assert data["total_s1"] == 2206821
        assert data["total_positive_links"] == 7638365

    def test_source_balance_consistency(self):
        """Verify Source 2 and Source 3 positive link representation is balanced."""
        import json
        config = PipelineConfig()
        report_path = config.paths.artifacts_dir / "data_coverage_report.json"
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        s2_pct = data["source2_positive_percentage"]
        s3_pct = data["source3_positive_percentage"]
        assert 45.0 <= s2_pct <= 55.0, f"S2 percentage {s2_pct}% is out of balanced bounds"
        assert 45.0 <= s3_pct <= 55.0, f"S3 percentage {s3_pct}% is out of balanced bounds"
        assert abs((s2_pct + s3_pct) - 100.0) < 0.01

    def test_target_completeness_invariant(self):
        """Verify 100% of required targets exist with 0 missing targets."""
        import json
        config = PipelineConfig()
        report_path = config.paths.artifacts_dir / "data_coverage_report.json"
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        reg = data["target_registry_coverage"]
        assert reg["missing_s2_targets"] == 0
        assert reg["missing_s3_targets"] == 0
        assert reg["target_completeness_pct"] == 100.0

    def test_singleton_ratio_preservation(self):
        """Verify singleton S1 entities represent expected ~5.58% proportion."""
        import json
        config = PipelineConfig()
        report_path = config.paths.artifacts_dir / "data_coverage_report.json"
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cov = data["s1_ground_truth_coverage"]
        assert cov["singleton_s1_entities"] == 123247
        assert 5.0 <= cov["singleton_s1_pct"] <= 6.5

    def test_negative_to_positive_ratio_bounded(self):
        """Verify negative-to-positive class balance ratio is strictly controlled."""
        import json
        config = PipelineConfig()
        report_path = config.paths.artifacts_dir / "data_coverage_report.json"
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ratio = data["negative_positive_ratio"]
        assert 2.0 <= ratio <= 15.0, f"Negative-to-positive ratio {ratio} outside expected range [2.0, 15.0]"


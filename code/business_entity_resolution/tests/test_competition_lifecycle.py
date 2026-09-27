"""
Comprehensive test suite for the OptiResolve 10-Phase Competition-Production Lifecycle.

Verifies:
  Phase 1: Load all labeled training data.
  Phase 2: Create an isolated validation split (disjoint S1 and disjoint target pools).
  Phase 3: Train candidate models using training S1 entities only.
  Phase 4: Evaluate on validation S1 entities (exact Macro F0.5).
  Phase 5: Select feature, blocking, model configurations & threshold.
  Phase 6: Freeze all selected hyperparameters.
  Phase 7: Train FINAL production classifier using 100% data with ZERO validation labels used.
  Phase 8: Persist all 8 production artifacts.
  Phase 9: Run blind test inference.
  Phase 10: Validate output submission files.
"""

from collections import Counter
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import pytest

from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.features import FEATURE_NAMES
from business_entity_resolution.pipeline import (
    EntityResolutionPipeline,
    compute_dataset_statistics,
    get_feature_schema,
    get_git_commit_hash,
    get_software_versions,
    validate_submission_output,
)
from run_pipeline import main


class TestLifecycleCLI:
    """Test CLI parsing for lifecycle and validation execution modes."""

    def test_cli_mode_lifecycle(self):
        """CLI --mode lifecycle must execute pipeline.run_competition_lifecycle()."""
        test_args = ["run_pipeline.py", "--mode", "lifecycle"]
        with patch.object(sys, "argv", test_args):
            with patch("run_pipeline.EntityResolutionPipeline") as mock_cls:
                instance = mock_cls.return_value
                main()
                assert instance.run_competition_lifecycle.called

    def test_cli_flag_lifecycle(self):
        """CLI --lifecycle must execute pipeline.run_competition_lifecycle()."""
        test_args = ["run_pipeline.py", "--lifecycle"]
        with patch.object(sys, "argv", test_args):
            with patch("run_pipeline.EntityResolutionPipeline") as mock_cls:
                instance = mock_cls.return_value
                main()
                assert instance.run_competition_lifecycle.called

    def test_cli_flag_validate(self):
        """CLI --validate must execute pipeline.phase10_validate_output()."""
        test_args = ["run_pipeline.py", "--validate"]
        with patch.object(sys, "argv", test_args):
            with patch("run_pipeline.EntityResolutionPipeline") as mock_cls:
                instance = mock_cls.return_value
                main()
                assert instance.phase10_validate_output.called

    def test_cli_mode_validate(self):
        """CLI --mode validate must execute pipeline.phase10_validate_output()."""
        test_args = ["run_pipeline.py", "--mode", "validate"]
        with patch.object(sys, "argv", test_args):
            with patch("run_pipeline.EntityResolutionPipeline") as mock_cls:
                instance = mock_cls.return_value
                main()
                assert instance.phase10_validate_output.called


class TestHelperFunctions:
    """Test lifecycle utilities for software versions, git commit, schema, and statistics."""

    def test_git_commit_hash(self):
        commit = get_git_commit_hash()
        assert isinstance(commit, str)
        assert len(commit) > 0

    def test_software_versions(self):
        versions = get_software_versions()
        assert "python" in versions
        assert "lightgbm" in versions
        assert "pandas" in versions
        assert "numpy" in versions
        assert "sklearn" in versions
        assert "rapidfuzz" in versions

    def test_feature_schema(self):
        schema = get_feature_schema()
        assert schema["schema_version"] == "2.0"
        assert schema["feature_count"] == len(FEATURE_NAMES)
        assert schema["feature_count"] == 23
        assert len(schema["features"]) == 23
        for idx, feat in enumerate(schema["features"]):
            assert feat["index"] == idx
            assert feat["name"] == FEATURE_NAMES[idx]
            assert "category" in feat
            assert "dtype" in feat

    def test_compute_dataset_statistics(self, tmp_path):
        cfg = PipelineConfig()
        cfg.paths.dataset_root = tmp_path
        stats = compute_dataset_statistics(cfg)
        assert "timestamp" in stats
        assert "splits" in stats
        assert "files" in stats


class TestSubmissionOutputValidation:
    """Test Phase 10 independent verification of submission TSVs."""

    def test_validate_submission_output_success(self, tmp_path):
        s1_file = tmp_path / "test_source1.tsv"
        s1_file.write_text("entity_id\tbusiness_name\tcountry\nS1_1\tA Corp\tUS\nS1_2\tB LLC\tUS\n", encoding="utf-8")

        m_file = tmp_path / "matching_results.tsv"
        m_file.write_text("source1_entity_id\tmatched_entity_ids\nS1_1\tS2_1\nS1_2\t\n", encoding="utf-8")

        c_file = tmp_path / "candidate_pairs.tsv"
        c_file.write_text("source1_entity_id\tcandidate_entity_ids\nS1_1\tS2_1,S3_1\nS1_2\tS2_2\n", encoding="utf-8")

        rep_file = tmp_path / "report.json"
        rep = validate_submission_output(m_file, c_file, s1_file, report_path=rep_file)

        assert rep["status"] == "PASSED"
        assert rep["total_test_s1_entities"] == 2
        assert rep["candidate_subset_violations"] == 0
        assert rep["total_predicted_links"] == 1
        assert rep["empty_match_entities"] == 1
        assert rep_file.exists()

    def test_validate_submission_output_header_failure(self, tmp_path):
        s1_file = tmp_path / "test_source1.tsv"
        s1_file.write_text("entity_id\nS1_1\n", encoding="utf-8")

        m_file = tmp_path / "matching_results.tsv"
        m_file.write_text("bad_col1\tbad_col2\nS1_1\tS2_1\n", encoding="utf-8")

        c_file = tmp_path / "candidate_pairs.tsv"
        c_file.write_text("source1_entity_id\tcandidate_entity_ids\nS1_1\tS2_1\n", encoding="utf-8")

        with pytest.raises(AssertionError) as exc_info:
            validate_submission_output(m_file, c_file, s1_file)
        assert "Invalid matching header" in str(exc_info.value)

    def test_validate_submission_output_subset_violation(self, tmp_path):
        s1_file = tmp_path / "test_source1.tsv"
        s1_file.write_text("entity_id\nS1_1\n", encoding="utf-8")

        m_file = tmp_path / "matching_results.tsv"
        m_file.write_text("source1_entity_id\tmatched_entity_ids\nS1_1\tS2_99\n", encoding="utf-8")

        c_file = tmp_path / "candidate_pairs.tsv"
        c_file.write_text("source1_entity_id\tcandidate_entity_ids\nS1_1\tS2_1,S2_2\n", encoding="utf-8")

        with pytest.raises(AssertionError) as exc_info:
            validate_submission_output(m_file, c_file, s1_file)
        assert "Candidate subset invariant violated" in str(exc_info.value)

    def test_validate_submission_output_row_count_mismatch(self, tmp_path):
        s1_file = tmp_path / "test_source1.tsv"
        s1_file.write_text("entity_id\nS1_1\nS1_2\n", encoding="utf-8")

        m_file = tmp_path / "matching_results.tsv"
        m_file.write_text("source1_entity_id\tmatched_entity_ids\nS1_1\tS2_1\n", encoding="utf-8")

        c_file = tmp_path / "candidate_pairs.tsv"
        c_file.write_text("source1_entity_id\tcandidate_entity_ids\nS1_1\tS2_1\n", encoding="utf-8")

        with pytest.raises(AssertionError) as exc_info:
            validate_submission_output(m_file, c_file, s1_file)
        assert "Row count mismatch" in str(exc_info.value)


class TestFullCompetitionLifecycleExecution:
    """Test sequential execution of all 10 phases on isolated synthetic mock data."""

    @pytest.fixture
    def mock_lifecycle_environment(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        train_dir = dataset_dir / "train"
        test_dir = dataset_dir / "test"
        train_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        # Train files
        s1_train = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S1_1\tAlpha Logistics\t100 Commerce Way\tUS\n"
            "S1_2\tBeta Pharma\t200 Health Ave\tUS\n"
            "S1_3\tGamma Tech\t300 Cyber Blvd\tUS\n"
            "S1_4\tDelta Singleton\t400 Lone St\tUS\n"
        )
        (train_dir / "train_source1.tsv").write_text(s1_train, encoding="utf-8")

        s2_train = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S2_1\tAlpha Logistics Inc\t100 Commerce Way\tUS\n"
            "S2_3\tGamma Technology Corp\t300 Cyber Blvd\tUS\n"
        )
        (train_dir / "train_source2.tsv").write_text(s2_train, encoding="utf-8")

        s3_train = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S3_2\tBeta Pharma LLC\t200 Health Ave\tUS\n"
        )
        (train_dir / "train_source3.tsv").write_text(s3_train, encoding="utf-8")

        gt_train = (
            "source1_entity_id\tmatched_entity_ids\n"
            "S1_1\tS2_1\n"
            "S1_2\tS3_2\n"
            "S1_3\tS2_3\n"
        )
        (train_dir / "train_ground_truth.tsv").write_text(gt_train, encoding="utf-8")

        # Test files
        s1_test = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S1_T1\tAlpha Logistics\t100 Commerce Way\tUS\n"
            "S1_T2\tEpsilon Unknown\t500 Mystery Rd\tUS\n"
        )
        (test_dir / "test_source1.tsv").write_text(s1_test, encoding="utf-8")

        s2_test = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S2_T1\tAlpha Logistics Inc\t100 Commerce Way\tUS\n"
        )
        (test_dir / "test_source2.tsv").write_text(s2_test, encoding="utf-8")

        s3_test = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S3_T1\tZeta Corp\t999 Other St\tUS\n"
        )
        (test_dir / "test_source3.tsv").write_text(s3_test, encoding="utf-8")

        cfg = PipelineConfig()
        cfg.paths.dataset_root = dataset_dir
        cfg.paths.artifacts_dir = artifacts_dir
        cfg.paths.output_dir = output_dir
        cfg.model.n_estimators = 10
        cfg.model.early_stopping_rounds = 5
        cfg.training_mode = "production"
        cfg.train_s1_limit = None
        cfg.val_s1_limit = 2

        return cfg, artifacts_dir, output_dir

    def test_run_competition_lifecycle_end_to_end(self, mock_lifecycle_environment):
        cfg, artifacts_dir, output_dir = mock_lifecycle_environment
        pipeline = EntityResolutionPipeline(cfg)

        result = pipeline.run_competition_lifecycle(skip_test_inference=False, validate_outputs=True)

        assert result["status"] == "SUCCESS"
        assert "frozen_config" in result
        assert "persisted_artifacts" in result
        assert "validation_report" in result

        # Verify all 8 Phase 8 artifacts exist
        assert (artifacts_dir / "lightgbm_er_model.joblib").exists()
        assert (artifacts_dir / "frozen_pipeline_config.json").exists()
        assert (artifacts_dir / "optimal_threshold.json").exists()
        assert (artifacts_dir / "production_training_metadata.json").exists()
        assert (artifacts_dir / "final_training_metadata.json").exists()
        assert (artifacts_dir / "feature_schema.json").exists()
        assert (artifacts_dir / "dataset_statistics.json").exists()
        assert (artifacts_dir / "software_versions.json").exists()
        assert (artifacts_dir / "git_commit_hash.txt").exists()

        # Verify Phase 9 output TSVs exist
        assert (output_dir / "matching_results.tsv").exists()
        assert (output_dir / "candidate_pairs.tsv").exists()

        # Verify Phase 10 validation report exists and passed
        assert (artifacts_dir / "submission_validation_report.json").exists()
        report = json.loads((artifacts_dir / "submission_validation_report.json").read_text(encoding="utf-8"))
        assert report["status"] == "PASSED"
        assert report["candidate_subset_violations"] == 0
        assert report["total_lines_validated"] == 2

    def test_phase7_validation_independence(self, mock_lifecycle_environment):
        """
        Verify that Phase 7 trains final classifier without passing validation sets or labels.
        Ensures validation independence guarantee.
        """
        cfg, artifacts_dir, _ = mock_lifecycle_environment
        pipeline = EntityResolutionPipeline(cfg)

        p1 = pipeline.phase1_load_all_labeled_data()
        p2 = pipeline.phase2_create_isolated_validation_split(p1)
        p3 = pipeline.phase3_train_candidate_models(p2)
        p4 = pipeline.phase4_evaluate_validation(p2)
        p5 = pipeline.phase5_select_configurations(p3, p4)
        p6 = pipeline.phase6_freeze_hyperparameters(p5)

        with patch.object(pipeline.model, "train", wraps=pipeline.model.train) as mock_train:
            p7 = pipeline.phase7_train_final_classifier(p6)
            assert mock_train.called
            call_kwargs = mock_train.call_args[1]
            # Must strictly be called with X_val=None and y_val=None
            assert call_kwargs.get("X_val") is None
            assert call_kwargs.get("y_val") is None
            assert p7["s1_selected"] == 4  # 100% full dataset

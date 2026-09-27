"""
Tests for production inference checkpointing, atomic recovery, file reconciliation,
and candidate-subset invariant preservation across crash/resume cycles.
"""

import json
from pathlib import Path
import pytest

from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.pipeline import (
    EntityResolutionPipeline,
    reconcile_output_files,
    validate_submission_output,
)


class TestFileReconciliation:
    """Test crash recovery file reconciliation between matching and candidate output TSVs."""

    def test_reconcile_identical_files(self, tmp_path):
        m_file = tmp_path / "matching.tsv"
        c_file = tmp_path / "candidate.tsv"

        m_file.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1_1\tS2_1\n"
            "S1_2\t\n"
            "S1_3\tS3_1\n",
            encoding="utf-8",
        )
        c_file.write_text(
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1_1\tS2_1\n"
            "S1_2\tS2_2\n"
            "S1_3\tS3_1,S3_2\n",
            encoding="utf-8",
        )

        valid_len, processed_ids = reconcile_output_files(m_file, c_file)
        assert valid_len == 3
        assert processed_ids == {"S1_1", "S1_2", "S1_3"}

    def test_reconcile_truncated_candidate_file(self, tmp_path):
        """Simulate crash where matching.tsv wrote 3 rows but candidate.tsv only wrote 2 rows."""
        m_file = tmp_path / "matching.tsv"
        c_file = tmp_path / "candidate.tsv"

        m_file.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1_1\tS2_1\n"
            "S1_2\t\n"
            "S1_3\tS3_1\n",
            encoding="utf-8",
        )
        c_file.write_text(
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1_1\tS2_1\n"
            "S1_2\tS2_2\n",
            encoding="utf-8",
        )

        valid_len, processed_ids = reconcile_output_files(m_file, c_file)
        assert valid_len == 2
        assert processed_ids == {"S1_1", "S1_2"}

        # Both files must now have exactly 2 data rows
        m_lines = m_file.read_text(encoding="utf-8").strip().split("\n")
        c_lines = c_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(m_lines) == 3  # 1 header + 2 rows
        assert len(c_lines) == 3

    def test_reconcile_desynced_ids(self, tmp_path):
        """Simulate ID desynchronization at row 3."""
        m_file = tmp_path / "matching.tsv"
        c_file = tmp_path / "candidate.tsv"

        m_file.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1_1\tS2_1\n"
            "S1_2\t\n"
            "S1_3\tS3_1\n",
            encoding="utf-8",
        )
        c_file.write_text(
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1_1\tS2_1\n"
            "S1_2\tS2_2\n"
            "S1_DIFFERENT\tS3_1\n",
            encoding="utf-8",
        )

        valid_len, processed_ids = reconcile_output_files(m_file, c_file)
        assert valid_len == 2
        assert processed_ids == {"S1_1", "S1_2"}


class TestInferenceCrashAndResume:
    """Test full crash and resume workflow on mock dataset."""

    @pytest.fixture
    def mock_inference_env(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        test_dir = dataset_dir / "test"
        test_dir.mkdir(parents=True)
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        # 4 Test S1 entities (2 US, 2 India)
        s1_content = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S1_US_1\tAlpha Corp\t100 Main St\tUS\n"
            "S1_US_2\tBeta LLC\t200 Oak Ave\tUS\n"
            "S1_IN_1\tGamma Ltd\t300 Pine Rd\tIndia\n"
            "S1_IN_2\tDelta Pvt\t400 Maple Dr\tIndia\n"
        )
        (test_dir / "test_source1.tsv").write_text(s1_content, encoding="utf-8")

        s2_content = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S2_US_1\tAlpha Corporation\t100 Main St\tUS\n"
            "S2_IN_1\tGamma Enterprises Ltd\t300 Pine Rd\tIndia\n"
        )
        (test_dir / "test_source2.tsv").write_text(s2_content, encoding="utf-8")

        s3_content = (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S3_US_1\tBeta Tech LLC\t200 Oak Ave\tUS\n"
            "S3_IN_1\tDelta Solutions Pvt\t400 Maple Dr\tIndia\n"
        )
        (test_dir / "test_source3.tsv").write_text(s3_content, encoding="utf-8")

        cfg = PipelineConfig()
        cfg.paths.dataset_root = dataset_dir
        cfg.paths.artifacts_dir = artifacts_dir
        cfg.paths.output_dir = output_dir

        # Train and save a lightweight dummy model so inference can run or reload
        import numpy as np
        from business_entity_resolution.model import EntityResolutionModel
        from business_entity_resolution.config import ModelConfig
        m_cfg = ModelConfig(min_child_samples=1, n_estimators=2)
        model = EntityResolutionModel(m_cfg)
        dummy_X = np.random.RandomState(42).rand(10, 23)
        dummy_y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        model.train(dummy_X, dummy_y)
        model.save(cfg.paths.model_path)

        return cfg, artifacts_dir, output_dir, test_dir

    def test_checkpoint_resume_no_duplicates_or_omissions(self, mock_inference_env):
        cfg, artifacts_dir, output_dir, test_dir = mock_inference_env
        pipeline = EntityResolutionPipeline(cfg)

        # Step 1: Simulate interrupted run (process only 1 entity per country)
        res1 = pipeline.predict_test(
            batch_size=1,
            resume=False,
            max_entities_per_country=1,
        )

        assert res1["total_s1_processed"] == 2  # 1 from India, 1 from US
        ckpt_file = artifacts_dir / "inference_checkpoint.json"
        assert ckpt_file.exists()

        # Step 2: Resume inference with NO limit (process remaining entities)
        res2 = pipeline.predict_test(
            batch_size=1,
            resume=True,
            max_entities_per_country=None,
        )

        assert res2["total_s1_processed"] == 4  # All 4 entities completed

        # Step 3: Run Phase 10 output validation
        rep = validate_submission_output(
            matching_tsv=cfg.paths.matching_results,
            candidate_tsv=cfg.paths.candidate_pairs,
            test_s1_tsv=test_dir / "test_source1.tsv",
        )

        assert rep["status"] == "PASSED"
        assert rep["total_test_s1_entities"] == 4
        assert rep["total_lines_validated"] == 4
        assert rep["candidate_subset_violations"] == 0
        assert rep["unique_matching_s1_count"] == 4
        assert rep["unique_candidate_s1_count"] == 4

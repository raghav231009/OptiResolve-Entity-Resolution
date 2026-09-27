"""
Unit and Integration Tests for Ground-Truth Target Loading Hardening.

Verifies:
1. Zero missing targets passes cleanly.
2. One missing target fails loudly in production mode with TargetPoolVerificationError.
3. One missing target allows continuation in development mode only with explicit flag.
4. Multiple missing targets are categorized and reported by source (Source 2 vs Source 3).
5. Malformed target IDs are detected, categorized, and flagged.
6. Diagnostic JSON file is saved with complete structured breakdown.
7. Production PipelineConfig prohibits allow_missing_targets=True.
8. Integration with load_isolated_target_pools and load_targeted_training_targets.
"""

import json
from pathlib import Path
import pytest

from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.pipeline import (
    TargetPoolVerificationError,
    load_isolated_target_pools,
    load_targeted_training_targets,
    verify_target_pool_completeness,
)


class TestTargetPoolVerification:
    """Requirement 1, 2, 3, 4, 5, 6: verify_target_pool_completeness unit tests."""

    def test_zero_missing_targets_passes_cleanly(self, tmp_path):
        """When all required target IDs are present, verification passes without error."""
        required = {"S2-100", "S2-101", "S3-200", "S3-201"}
        loaded_records = [
            {"entity_id": "S2-100", "country": "US", "business_name": "Alpha Corp", "business_address": "1 Main St"},
            {"entity_id": "S2-101", "country": "US", "business_name": "Beta LLC", "business_address": "2 Main St"},
            {"entity_id": "S3-200", "country": "US", "business_name": "Gamma Inc", "business_address": "3 Main St"},
            {"entity_id": "S3-201", "country": "US", "business_name": "Delta Co", "business_address": "4 Main St"},
        ]
        diag_path = tmp_path / "diag.json"

        result = verify_target_pool_completeness(
            required_ids=required,
            loaded_records=loaded_records,
            allow_missing=False,
            diagnostic_path=diag_path,
            pool_label="Test Zero Missing",
        )

        assert result["required_target_count"] == 4
        assert result["loaded_target_count"] == 4
        assert result["missing_target_count"] == 0
        assert len(result["missing_target_ids"]) == 0
        assert not diag_path.exists()

    def test_one_missing_target_fails_in_production(self, tmp_path):
        """In production (allow_missing=False), a single missing target ID fails loudly."""
        required = {"S2-100", "S2-101", "S3-200"}
        loaded_records = [
            {"entity_id": "S2-100", "country": "US", "business_name": "Alpha Corp", "business_address": "1 Main St"},
            {"entity_id": "S3-200", "country": "US", "business_name": "Gamma Inc", "business_address": "3 Main St"},
        ]  # S2-101 is missing!
        diag_path = tmp_path / "missing_one.json"

        with pytest.raises(TargetPoolVerificationError) as exc_info:
            verify_target_pool_completeness(
                required_ids=required,
                loaded_records=loaded_records,
                allow_missing=False,
                diagnostic_path=diag_path,
                pool_label="Test One Missing",
            )

        err_msg = str(exc_info.value)
        assert "S2-101" in err_msg
        assert "Missing 1 / 3 required target IDs" in err_msg
        assert diag_path.exists(), "Diagnostic file should be saved on missing targets!"

        with open(diag_path, "r", encoding="utf-8") as f:
            diag_data = json.load(f)
        assert diag_data["missing_target_count"] == 1
        assert "S2-101" in diag_data["missing_target_ids"]
        assert diag_data["missing_by_source"]["S2"]["count"] == 1
        assert diag_data["missing_by_source"]["S3"]["count"] == 0

    def test_one_missing_target_allowed_in_dev_mode(self, tmp_path):
        """In development mode with explicit allow_missing=True, warning is logged and execution proceeds."""
        required = {"S2-100", "S2-101"}
        loaded_records = [
            {"entity_id": "S2-100", "country": "US", "business_name": "Alpha Corp", "business_address": "1 Main St"},
        ]
        diag_path = tmp_path / "dev_missing.json"

        # Does not raise!
        result = verify_target_pool_completeness(
            required_ids=required,
            loaded_records=loaded_records,
            allow_missing=True,
            diagnostic_path=diag_path,
            pool_label="Test Dev Mode",
        )

        assert result["missing_target_count"] == 1
        assert "S2-101" in result["missing_target_ids"]
        assert diag_path.exists()

    def test_multiple_missing_targets_reported_by_source(self, tmp_path):
        """Multiple missing targets are categorized accurately across Source 2 and Source 3."""
        required = {
            "S2-001", "S2-002", "S2-003",
            "S3-001", "S3-002", "S3-003", "S3-004",
        }
        # Only S2-001 and S3-001 are loaded
        loaded_records = [
            {"entity_id": "S2-001", "country": "US", "business_name": "A", "business_address": "1 A St"},
            {"entity_id": "S3-001", "country": "US", "business_name": "B", "business_address": "2 B St"},
        ]
        diag_path = tmp_path / "multiple_missing.json"

        with pytest.raises(TargetPoolVerificationError):
            verify_target_pool_completeness(
                required_ids=required,
                loaded_records=loaded_records,
                allow_missing=False,
                diagnostic_path=diag_path,
                pool_label="Test Multiple Missing",
            )

        with open(diag_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["missing_target_count"] == 5
        assert data["missing_by_source"]["S2"]["count"] == 2
        assert set(data["missing_by_source"]["S2"]["sample"]) == {"S2-002", "S2-003"}
        assert data["missing_by_source"]["S2"]["expected_file"] == "train_source2.tsv"

        assert data["missing_by_source"]["S3"]["count"] == 3
        assert set(data["missing_by_source"]["S3"]["sample"]) == {"S3-002", "S3-003", "S3-004"}
        assert data["missing_by_source"]["S3"]["expected_file"] == "train_source3.tsv"

    def test_malformed_target_ids_detected_and_categorized(self, tmp_path):
        """Malformed target IDs (non-standard prefix, spaces, empty) are detected and categorized."""
        required = {
            "S2-100",
            "INVALID_NO_PREFIX",
            "S2-WITH SPACE",
            "S4-99999",
            "123456",
        }
        loaded_records = [
            {"entity_id": "S2-100", "country": "US", "business_name": "A", "business_address": "1 St"},
        ]
        diag_path = tmp_path / "malformed.json"

        with pytest.raises(TargetPoolVerificationError):
            verify_target_pool_completeness(
                required_ids=required,
                loaded_records=loaded_records,
                allow_missing=False,
                diagnostic_path=diag_path,
                pool_label="Test Malformed",
                strict_prefix=True,
            )

        with open(diag_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        malformed_list = data["malformed_target_ids"]
        assert "INVALID_NO_PREFIX" in malformed_list
        assert "S2-WITH SPACE" in malformed_list
        assert "S4-99999" in malformed_list
        assert "123456" in malformed_list
        assert data["missing_by_source"]["malformed"]["count"] == 4


class TestPipelineConfigHardening:
    """Requirement 2 & 3: Production configuration prohibits allow_missing_targets."""

    def test_production_training_mode_rejects_allow_missing(self):
        with pytest.raises(ValueError) as exc_info:
            PipelineConfig(training_mode="production", allow_missing_targets=True)
        assert "Production training violation" in str(exc_info.value)

    def test_development_mode_allows_explicit_flag(self):
        config = PipelineConfig(training_mode="development", allow_missing_targets=True)
        assert config.allow_missing_targets is True
        assert config.is_development is True


class TestTargetLoadersIntegration:
    """Integration tests with load_isolated_target_pools and load_targeted_training_targets."""

    def test_load_isolated_target_pools_fails_on_missing_target(self, tmp_path):
        """load_isolated_target_pools raises TargetPoolVerificationError if a train target is missing."""
        source_tsv = tmp_path / "train_source2.tsv"
        lines = [
            "entity_id\tcountry\tbusiness_name\tbusiness_address\n",
            "S2-001\tUS\tAlpha Corp\t1 Main St\n",
            "S2-002\tUS\tBeta LLC\t2 Main St\n",
            "S2-003\tUS\tGamma Inc\t3 Main St\n",
        ]
        source_tsv.write_text("".join(lines), encoding="utf-8")

        train_needed = {"S2-001", "S2-002", "S2-MISSING"}
        val_needed = {"S2-003"}

        diag_dir = tmp_path / "diagnostics"

        with pytest.raises(TargetPoolVerificationError):
            load_isolated_target_pools(
                source_paths=[source_tsv],
                train_needed_ids=train_needed,
                val_needed_ids=val_needed,
                train_background_sample_per_file=10,
                val_background_sample_per_file=10,
                allow_missing_targets=False,
                diagnostic_dir=diag_dir,
            )

        assert (diag_dir / "missing_targets_train.json").exists()

    def test_load_isolated_target_pools_allows_missing_in_dev_mode(self, tmp_path):
        """load_isolated_target_pools proceeds when allow_missing_targets=True in dev mode."""
        source_tsv = tmp_path / "train_source2.tsv"
        lines = [
            "entity_id\tcountry\tbusiness_name\tbusiness_address\n",
            "S2-001\tUS\tAlpha Corp\t1 Main St\n",
            "S2-002\tUS\tBeta LLC\t2 Main St\n",
        ]
        source_tsv.write_text("".join(lines), encoding="utf-8")

        train_needed = {"S2-001", "S2-MISSING"}
        val_needed = {"S2-002"}

        diag_dir = tmp_path / "diagnostics_dev"

        train_targets, val_targets = load_isolated_target_pools(
            source_paths=[source_tsv],
            train_needed_ids=train_needed,
            val_needed_ids=val_needed,
            train_background_sample_per_file=10,
            val_background_sample_per_file=10,
            allow_missing_targets=True,
            diagnostic_dir=diag_dir,
        )

        assert len(train_targets) >= 1
        assert len(val_targets) >= 1
        assert (diag_dir / "missing_targets_train.json").exists()

    def test_load_targeted_training_targets_fails_on_missing_target(self, tmp_path):
        """load_targeted_training_targets raises TargetPoolVerificationError if target is missing."""
        source_tsv = tmp_path / "train_source2.tsv"
        lines = [
            "entity_id\tcountry\tbusiness_name\tbusiness_address\n",
            "S2-001\tUS\tAlpha Corp\t1 Main St\n",
        ]
        source_tsv.write_text("".join(lines), encoding="utf-8")

        needed = {"S2-001", "S2-MISSING-TARGET"}
        diag_dir = tmp_path / "targeted_diag"

        with pytest.raises(TargetPoolVerificationError):
            load_targeted_training_targets(
                source_paths=[source_tsv],
                needed_ids=needed,
                allow_missing_targets=False,
                diagnostic_dir=diag_dir,
            )

        assert (diag_dir / "missing_targets.json").exists()

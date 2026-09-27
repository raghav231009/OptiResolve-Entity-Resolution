"""
Tests for Independent Adversarial Submission Validator.
Ensures the independent validator detects every adversarial flaw without false positives or negatives.
"""

from pathlib import Path
import pytest

from business_entity_resolution.independent_validator import run_adversarial_audit


class TestIndependentAdversarialValidator:
    """Test matrix of adversarial test cases for independent submission validation."""

    @pytest.fixture
    def mock_env(self, tmp_path):
        # S1 test file: 3 entities (1 US, 1 France, 1 India)
        s1 = tmp_path / "test_s1.tsv"
        s1.write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S1_US_1\tAcme Corp\t123 Main St\tUS\n"
            "S1_FR_1\tSociete Generale\t10 Rue de Paris\tFrance\n"
            "S1_IN_1\tTata Sons\tBombay House\tIndia\n",
            encoding="utf-8",
        )

        # S2 test file: 3 targets
        s2 = tmp_path / "test_s2.tsv"
        s2.write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S2_US_1\tAcme Corporation\t123 Main St\tUS\n"
            "S2_FR_1\tSociete Generale SA\t10 Rue de Paris\tFrance\n"
            "S2_IN_1\tTata Sons Pvt Ltd\tBombay House\tIndia\n",
            encoding="utf-8",
        )

        # S3 test file: 3 targets
        s3 = tmp_path / "test_s3.tsv"
        s3.write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S3_US_1\tAcme Inc\t123 Main St\tUS\n"
            "S3_FR_1\tSG Group\t10 Rue de Paris\tFrance\n"
            "S3_IN_1\tTata Enterprises\tBombay House\tIndia\n",
            encoding="utf-8",
        )

        # Valid baseline outputs
        matching = tmp_path / "matching.tsv"
        matching.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1_US_1\tS2_US_1,S3_US_1\n"
            "S1_FR_1\tS2_FR_1\n"
            "S1_IN_1\t\n",  # empty match representation
            encoding="utf-8",
        )

        candidates = tmp_path / "candidates.tsv"
        candidates.write_text(
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1_US_1\tS2_US_1,S3_US_1\n"
            "S1_FR_1\tS2_FR_1,S3_FR_1\n"
            "S1_IN_1\tS2_IN_1,S3_IN_1\n",
            encoding="utf-8",
        )

        return s1, s2, s3, matching, candidates

    def test_valid_submission_passes(self, mock_env):
        s1, s2, s3, matching, candidates = mock_env
        res = run_adversarial_audit(s1, s2, s3, matching, candidates, max_k_candidates=80)
        assert res["status"] == "PASSED"
        assert res["total_violation_count"] == 0
        assert res["statistics"]["total_rows_matching_tsv"] == 3
        assert res["statistics"]["empty_match_entities"] == 1
        assert res["statistics"]["total_predicted_matches"] == 3

    def test_detects_malformed_header(self, mock_env):
        s1, s2, s3, matching, candidates = mock_env
        matching.write_text("s1_id\tmatches\nS1_US_1\tS2_US_1\n", encoding="utf-8")
        res = run_adversarial_audit(s1, s2, s3, matching, candidates)
        assert res["status"] == "FAILED"
        assert res["violations"]["matching_header_violation"] is not None

    def test_detects_duplicate_s1(self, mock_env):
        s1, s2, s3, matching, candidates = mock_env
        matching.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1_US_1\tS2_US_1\n"
            "S1_US_1\tS2_US_1\n"
            "S1_FR_1\tS2_FR_1\n",
            encoding="utf-8",
        )
        res = run_adversarial_audit(s1, s2, s3, matching, candidates)
        assert res["status"] == "FAILED"
        assert res["violations"]["duplicate_s1_in_matching"] >= 1

    def test_detects_missing_s1(self, mock_env):
        s1, s2, s3, matching, candidates = mock_env
        matching.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1_US_1\tS2_US_1\n"
            "S1_FR_1\tS2_FR_1\n",  # S1_IN_1 missing
            encoding="utf-8",
        )
        candidates.write_text(
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1_US_1\tS2_US_1\n"
            "S1_FR_1\tS2_FR_1\n",
            encoding="utf-8",
        )
        res = run_adversarial_audit(s1, s2, s3, matching, candidates)
        assert res["status"] == "FAILED"
        assert res["violations"]["missing_s1_in_matching"] == 1
        assert res["violations"]["missing_s1_in_candidates"] == 1

    def test_detects_invalid_target_id(self, mock_env):
        s1, s2, s3, matching, candidates = mock_env
        matching.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1_US_1\tS2_US_1,S99_NONEXISTENT\n"
            "S1_FR_1\tS2_FR_1\n"
            "S1_IN_1\t\n",
            encoding="utf-8",
        )
        candidates.write_text(
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1_US_1\tS2_US_1,S99_NONEXISTENT\n"
            "S1_FR_1\tS2_FR_1\n"
            "S1_IN_1\tS2_IN_1\n",
            encoding="utf-8",
        )
        res = run_adversarial_audit(s1, s2, s3, matching, candidates)
        assert res["status"] == "FAILED"
        assert res["violations"]["invalid_target_ids_in_matching"] == 1

    def test_detects_candidate_subset_invariant_violation(self, mock_env):
        s1, s2, s3, matching, candidates = mock_env
        # Match S3_US_1 is NOT in candidates
        matching.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1_US_1\tS2_US_1,S3_US_1\n"
            "S1_FR_1\tS2_FR_1\n"
            "S1_IN_1\t\n",
            encoding="utf-8",
        )
        candidates.write_text(
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1_US_1\tS2_US_1\n"  # S3_US_1 omitted from candidate
            "S1_FR_1\tS2_FR_1\n"
            "S1_IN_1\tS2_IN_1\n",
            encoding="utf-8",
        )
        res = run_adversarial_audit(s1, s2, s3, matching, candidates)
        assert res["status"] == "FAILED"
        assert res["violations"]["candidate_subset_violations"] == 1

    def test_detects_cross_country_match(self, mock_env):
        s1, s2, s3, matching, candidates = mock_env
        # Match US entity S1_US_1 to India target S2_IN_1
        matching.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1_US_1\tS2_IN_1\n"
            "S1_FR_1\tS2_FR_1\n"
            "S1_IN_1\t\n",
            encoding="utf-8",
        )
        candidates.write_text(
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1_US_1\tS2_IN_1\n"
            "S1_FR_1\tS2_FR_1\n"
            "S1_IN_1\tS2_IN_1\n",
            encoding="utf-8",
        )
        res = run_adversarial_audit(s1, s2, s3, matching, candidates)
        assert res["status"] == "FAILED"
        assert res["violations"]["cross_country_matches"] >= 1

    def test_detects_forbidden_null_literals(self, mock_env):
        s1, s2, s3, matching, candidates = mock_env
        matching.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1_US_1\tNaN\n"
            "S1_FR_1\tS2_FR_1\n"
            "S1_IN_1\t\n",
            encoding="utf-8",
        )
        res = run_adversarial_audit(s1, s2, s3, matching, candidates)
        assert res["status"] == "FAILED"
        assert res["violations"]["forbidden_null_literals"] >= 1

    def test_detects_candidate_count_exceeded(self, mock_env):
        s1, s2, s3, matching, candidates = mock_env
        # S1_US_1 has 2 candidates, set max_k_candidates=1
        res = run_adversarial_audit(s1, s2, s3, matching, candidates, max_k_candidates=1)
        assert res["status"] == "FAILED"
        assert res["violations"]["candidate_count_exceeded"] >= 1

    def test_detects_row_order_desync(self, mock_env):
        s1, s2, s3, matching, candidates = mock_env
        # Swap order in candidates
        candidates.write_text(
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1_FR_1\tS2_FR_1\n"
            "S1_US_1\tS2_US_1\n"
            "S1_IN_1\tS2_IN_1\n",
            encoding="utf-8",
        )
        res = run_adversarial_audit(s1, s2, s3, matching, candidates)
        assert res["status"] == "FAILED"
        assert res["violations"]["order_desync_count"] >= 1

"""
Tests for Source 2 vs Source 3 Data Composition Audit and Automated Error Analysis Pipeline.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.error_analysis import (
    classify_false_negative_noise,
    classify_false_positive,
    extract_suite,
    run_error_analysis,
)
from business_entity_resolution.features import FEATURE_NAMES, compute_pair_features
from business_entity_resolution.model import EntityResolutionModel
from business_entity_resolution.source_audit import (
    compute_feature_summary,
    identify_source,
    run_source_composition_audit,
)


class TestSourceIdentification:
    def test_identify_source(self):
        assert identify_source("S2-12345") == "S2"
        assert identify_source("S3-98765") == "S3"
        assert identify_source("S2_001") == "S2"
        assert identify_source("S3_999") == "S3"
        assert identify_source("OTHER_01") == "Unknown"

    def test_compute_feature_summary(self):
        mat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        names = ["feat1", "feat2"]
        summary = compute_feature_summary(mat, names)
        assert "feat1" in summary
        assert "feat2" in summary
        assert summary["feat1"]["mean"] == 3.0
        assert summary["feat1"]["min"] == 1.0
        assert summary["feat1"]["max"] == 5.0
        assert summary["feat2"]["median"] == 4.0


class TestErrorClassificationTaxonomy:
    def test_extract_suite(self):
        assert extract_suite("123 Main St, Suite 400") == "400"
        assert extract_suite("500 Elm Ave Ste B-12") == "B-12"
        assert extract_suite("45 Broadway Apt 3F") == "3F"
        assert extract_suite("100 Market St Unit 12") == "12"
        assert extract_suite("100 Market St") is None

    def test_classify_false_positive_same_building_diff_business(self):
        s1 = {"clean_name": "STARBUCKS", "clean_address": "100 MAIN ST", "entity_id": "S1_1"}
        cand = {"clean_name": "SUBWAY", "clean_address": "100 MAIN ST", "entity_id": "S2_2"}
        # feats: name_jaccard=0.0, name_jw=0.2, ..., addr_jaccard=0.8, ..., same_bldg=1.0, same_postal=1.0
        feats = [0.0] * len(FEATURE_NAMES)
        feats[0] = 0.0   # name jaccard
        feats[1] = 0.2   # name jw
        feats[7] = 0.8   # addr jaccard
        feats[12] = 1.0  # same bldg
        feats[13] = 1.0  # same postal

        cats = classify_false_positive(s1, cand, feats, prob=0.85, threshold=0.78)
        assert "same_building_different_business" in cats
        assert "s2_specific" in cats

    def test_classify_false_positive_similar_company_names(self):
        s1 = {"clean_name": "METRO LOGISTICS", "clean_address": "100 MAIN ST", "entity_id": "S1_1"}
        cand = {"clean_name": "METRO LOGISTICS CORP", "clean_address": "800 BROADWAY AVE", "entity_id": "S3_3"}
        feats = [0.0] * len(FEATURE_NAMES)
        feats[0] = 0.85  # name jaccard
        feats[1] = 0.95  # name jw
        feats[7] = 0.10  # addr jaccard
        feats[12] = 0.0  # same bldg
        feats[13] = 0.0  # same postal

        cats = classify_false_positive(s1, cand, feats, prob=0.82, threshold=0.78)
        assert "similar_company_names" in cats
        assert "s3_specific" in cats

    def test_classify_false_positive_generic_names(self):
        s1 = {"clean_name": "GLOBAL RESTAURANT", "clean_address": "100 MAIN ST", "entity_id": "S1_1"}
        cand = {"clean_name": "GLOBAL RESTAURANT", "clean_address": "500 ELM AVE", "entity_id": "S2_2"}
        feats = [0.0] * len(FEATURE_NAMES)
        feats[0] = 0.9
        feats[1] = 0.9
        feats[7] = 0.2

        cats = classify_false_positive(s1, cand, feats, prob=0.80, threshold=0.78)
        assert "generic_company_names" in cats

    def test_classify_false_negative_noise_profiles(self):
        s1 = {"clean_name": "ALPHA BIOTECH", "postal_code": "90210", "clean_address": "123 MAIN ST"}

        # Missing address
        cand_no_addr = {"clean_name": "ALPHA BIOTECH", "postal_code": "90210", "clean_address": ""}
        feats = [0.9, 0.95] + [0.0] * 21
        assert classify_false_negative_noise(s1, cand_no_addr, feats) == "missing_address"

        # Missing postal
        cand_no_post = {"clean_name": "ALPHA BIOTECH", "postal_code": "", "clean_address": "123 MAIN ST"}
        assert classify_false_negative_noise(s1, cand_no_post, feats) == "missing_postal"

        # Name corruption (transliteration / noisy name, strong address)
        cand_corrupt_name = {"clean_name": "ALPH BTECH", "postal_code": "90210", "clean_address": "123 MAIN ST"}
        feats_corr_name = [0.2, 0.4] + [0.0] * 5 + [0.85] + [0.0] * 15
        assert classify_false_negative_noise(s1, cand_corrupt_name, feats_corr_name) == "name_corruption"

        # Address corruption (clean name, corrupted address)
        cand_corrupt_addr = {"clean_name": "ALPHA BIOTECH", "postal_code": "90210", "clean_address": "77 NO ROAD"}
        feats_corr_addr = [0.9, 0.95] + [0.0] * 5 + [0.1] + [0.0] * 15
        assert classify_false_negative_noise(s1, cand_corrupt_addr, feats_corr_addr) == "address_corruption"


class TestEndToEndAuditPipelines:
    """Test full execution of source audit and error analysis on mock data."""

    def test_run_source_audit_and_error_analysis(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        train_dir = dataset_dir / "train"
        train_dir.mkdir(parents=True)
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)

        # Create mock train files with S2 and S3 records
        s1_file = train_dir / "train_source1.tsv"
        s1_file.write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S1-1\tStarbucks Coffee\t123 Main St Ste 100\tUS\n"
            "S1-2\tSubway Sandwiches\t456 Elm Ave\tUS\n",
            encoding="utf-8",
        )

        s2_file = train_dir / "train_source2.tsv"
        s2_file.write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S2-101\tStarbucks\t123 Main St\tUS\n"
            "S2-102\tSubway Sandwiches\t456 Elm Ave\tUS\n",
            encoding="utf-8",
        )

        s3_file = train_dir / "train_source3.tsv"
        s3_file.write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S3-201\tStarbucks Store\t123 Main Street\tUS\n"
            "S3-202\tSubway\t456 Elm Ave\tUS\n",
            encoding="utf-8",
        )

        gt_file = train_dir / "train_ground_truth.tsv"
        gt_file.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-1\tS2-101,S3-201\n"
            "S1-2\tS2-102,S3-202\n",
            encoding="utf-8",
        )

        # Pre-train and serialize a mock model
        model = EntityResolutionModel()
        model.config.n_estimators = 10
        X_mock = np.ones((4, len(FEATURE_NAMES)), dtype=np.float32)
        y_mock = np.array([1, 1, 0, 0], dtype=np.int32)
        model.train(X_mock, y_mock)
        model_path = artifacts_dir / "lightgbm_er_model.joblib"
        model.save(model_path)

        thresh_file = artifacts_dir / "optimal_threshold.json"
        thresh_file.write_text(json.dumps({"optimal_threshold": 0.50, "validation_macro_f05": 0.95}), encoding="utf-8")

        config = PipelineConfig()
        config.paths.dataset_root = dataset_dir
        config.paths.artifacts_dir = artifacts_dir

        # 1. Run source composition audit
        audit_res = run_source_composition_audit(config, s1_limit=2)
        assert "global_ground_truth" in audit_res
        assert audit_res["global_ground_truth"]["s2_positive_links"] == 2
        assert audit_res["global_ground_truth"]["s3_positive_links"] == 2
        assert "source_specific_validation_metrics" in audit_res
        assert "noise_distribution_cases" in audit_res

        assert (artifacts_dir / "source_composition_report.json").exists()
        assert (artifacts_dir / "source_composition_report.md").exists()

        # 2. Run error analysis pipeline
        err_res = run_error_analysis(config, val_sample_limit=2)
        assert "recommended_next_engineering_target" in err_res
        assert "fn_failure_stages_breakdown" in err_res
        assert "fp_categories_breakdown" in err_res

        assert (artifacts_dir / "validation_errors.json").exists()
        assert (artifacts_dir / "error_analysis_report.md").exists()

"""
Extended Feature Engineering Tests.
Verifies the exact 23-dimensional feature vector, missing-value semantics,
and edge case handling in compute_pair_features.
"""

import pytest
from business_entity_resolution.features import compute_pair_features, FEATURE_NAMES


def make_record(
    entity_id="S2-001",
    country="US",
    clean_name="acme logistics inc",
    root_name="acme logistics",
    clean_address="123 main street new york",
    postal_code="10001",
    building_number="123",
    numeric_tokens=None,
):
    return {
        "entity_id": entity_id,
        "country": country,
        "clean_name": clean_name,
        "root_name": root_name,
        "clean_address": clean_address,
        "postal_code": postal_code,
        "building_number": building_number,
        "numeric_tokens": numeric_tokens if numeric_tokens is not None else {"123"},
    }


class TestFeatureVector:
    def test_feature_count_is_23(self):
        s1 = make_record(entity_id="S1-001")
        cand = make_record(entity_id="S2-001")
        feats = compute_pair_features(s1, cand)
        assert len(feats) == 23, f"Expected 23 features, got {len(feats)}"

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == 23

    def test_perfect_match_scores(self):
        s1 = make_record(entity_id="S1-001")
        cand = make_record(entity_id="S2-001")
        feats = compute_pair_features(s1, cand)
        feat_dict = dict(zip(FEATURE_NAMES, feats))
        # Name exact match should be 1.0 for identical clean_names
        assert feat_dict["name_exact_match"] == 1.0
        # Root name exact match should be 1.0
        assert feat_dict["root_name_exact_match"] == 1.0
        # Postal exact match should be 1.0
        assert feat_dict["postal_exact_match"] == 1.0
        # House number match should be 1.0
        assert feat_dict["house_number_match"] == 1.0
        # Country match should be 1.0
        assert feat_dict["country_match"] == 1.0

    def test_missing_address_semantics(self):
        """Missing address fields should give 0.0 for addr features and addr_present_both=0."""
        s1 = make_record(entity_id="S1-001", clean_address="", building_number="")
        cand = make_record(entity_id="S2-001", clean_address="")
        feats = compute_pair_features(s1, cand)
        feat_dict = dict(zip(FEATURE_NAMES, feats))
        assert feat_dict["addr_present_both"] == 0.0
        assert feat_dict["addr_ratio"] == 0.0

    def test_missing_building_number_sentinel(self):
        """Missing building number should return -1.0 (missing sentinel)."""
        s1 = make_record(entity_id="S1-001", building_number="")
        cand = make_record(entity_id="S2-001", building_number="")
        feats = compute_pair_features(s1, cand)
        feat_dict = dict(zip(FEATURE_NAMES, feats))
        assert feat_dict["house_number_match"] == -1.0

    def test_missing_postal_code_sentinel(self):
        """Missing postal code should return -1.0 for postal features."""
        s1 = make_record(entity_id="S1-001", postal_code="")
        cand = make_record(entity_id="S2-001", postal_code="")
        feats = compute_pair_features(s1, cand)
        feat_dict = dict(zip(FEATURE_NAMES, feats))
        assert feat_dict["postal_exact_match"] == -1.0
        assert feat_dict["postal_prefix_match"] == -1.0

    def test_cross_country_flag(self):
        """country_match should be 0.0 for different countries."""
        s1 = make_record(entity_id="S1-001", country="US")
        cand = make_record(entity_id="S2-001", country="FRANCE")
        feats = compute_pair_features(s1, cand)
        feat_dict = dict(zip(FEATURE_NAMES, feats))
        assert feat_dict["country_match"] == 0.0

    def test_s3_source_flag(self):
        """target_source_is_s3 should be 1.0 for S3- prefixed candidates."""
        s1 = make_record(entity_id="S1-001")
        cand = make_record(entity_id="S3-001")
        feats = compute_pair_features(s1, cand)
        feat_dict = dict(zip(FEATURE_NAMES, feats))
        assert feat_dict["target_source_is_s3"] == 1.0

    def test_s2_source_flag(self):
        """target_source_is_s3 should be 0.0 for S2- prefixed candidates."""
        s1 = make_record(entity_id="S1-001")
        cand = make_record(entity_id="S2-001")
        feats = compute_pair_features(s1, cand)
        feat_dict = dict(zip(FEATURE_NAMES, feats))
        assert feat_dict["target_source_is_s3"] == 0.0

    def test_no_nan_values(self):
        """Feature vector must contain no NaN/None values."""
        import math
        s1 = make_record(entity_id="S1-001", clean_address="", building_number="", postal_code="")
        cand = make_record(entity_id="S2-001", clean_address="", building_number="", postal_code="")
        feats = compute_pair_features(s1, cand)
        for i, v in enumerate(feats):
            assert not math.isnan(v), f"NaN at feature index {i} ({FEATURE_NAMES[i]})"

    def test_postal_prefix_match(self):
        """postal_prefix_match checks first 2 digits of postal code."""
        s1 = make_record(entity_id="S1-001", postal_code="10001")
        cand = make_record(entity_id="S2-001", postal_code="10099")
        feats = compute_pair_features(s1, cand)
        feat_dict = dict(zip(FEATURE_NAMES, feats))
        assert feat_dict["postal_prefix_match"] == 1.0  # both start with "10"
        assert feat_dict["postal_exact_match"] == 0.0

    def test_feature_values_in_range(self):
        """Most features should be in [-1.0, 1.0] range."""
        s1 = make_record(entity_id="S1-001")
        cand = make_record(entity_id="S2-001", clean_name="completely different", root_name="different")
        feats = compute_pair_features(s1, cand)
        for i, v in enumerate(feats):
            assert -1.0 <= v <= 1.0, f"Feature {FEATURE_NAMES[i]} = {v} out of range [-1, 1]"

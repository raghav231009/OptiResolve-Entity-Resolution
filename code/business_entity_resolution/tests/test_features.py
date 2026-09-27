"""
Feature Engineering Engine & LightGBM Feature Interface Tests.
Verifies:
1. Exact 23-dimensional feature vector matching FEATURE_NAMES.
2. Deterministic behavior across runs.
3. Explicit missing-value semantics (-1 sentinels for structural features, 0.0 for distances).
4. Range validation across all 23 features.
5. Targeted edge cases:
   - exact match
   - typo
   - abbreviation
   - missing address
   - missing postal
   - different building number
   - same building number
   - S2 candidate
   - S3 candidate
   - transliteration-like corruption
6. LightGBM integration: receives and verifies exactly 23 columns.
"""

import math
import numpy as np
import pytest
from business_entity_resolution.features import compute_pair_features, FEATURE_NAMES
from business_entity_resolution.model import EntityResolutionModel


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
        "numeric_tokens": numeric_tokens if numeric_tokens is not None else ({"123"} if building_number else set()),
    }


class TestFeatureSpecification:
    """Verifies that the feature vector exactly matches the 23-feature specification."""

    def test_feature_names_length_is_23(self):
        assert len(FEATURE_NAMES) == 23

    def test_feature_vector_length_equals_feature_names_length(self):
        s1 = make_record(entity_id="S1-001")
        cand = make_record(entity_id="S2-001")
        feats = compute_pair_features(s1, cand)
        assert len(feats) == len(FEATURE_NAMES) == 23

    def test_feature_determinism(self):
        """Every feature must have deterministic behavior on identical inputs."""
        s1 = make_record(entity_id="S1-001", clean_name="apollo health care", root_name="apollo health")
        cand = make_record(entity_id="S2-002", clean_name="apollo healthcare ltd", root_name="apollo health")
        feats1 = compute_pair_features(s1, cand)
        feats2 = compute_pair_features(s1, cand)
        assert feats1 == feats2

    def test_all_values_are_floats_no_nans(self):
        s1 = make_record(entity_id="S1-001", clean_address="", building_number="", postal_code="")
        cand = make_record(entity_id="S2-001", clean_address="", building_number="", postal_code="")
        feats = compute_pair_features(s1, cand)
        for i, val in enumerate(feats):
            assert isinstance(val, (float, int)), f"Feature {FEATURE_NAMES[i]} is not numeric: {type(val)}"
            assert not math.isnan(val), f"Feature {FEATURE_NAMES[i]} is NaN"
            assert not math.isinf(val), f"Feature {FEATURE_NAMES[i]} is Inf"


class TestFeatureCategories:
    """Tests targeted scenarios required by specification."""

    def test_exact_match(self):
        """Exact match scenario: identical clean names, addresses, postal, and building number."""
        s1 = make_record(
            entity_id="S1-001",
            clean_name="acme logistics inc",
            root_name="acme logistics",
            clean_address="100 broadway new york",
            postal_code="10001",
            building_number="100",
        )
        cand = make_record(
            entity_id="S2-001",
            clean_name="acme logistics inc",
            root_name="acme logistics",
            clean_address="100 broadway new york",
            postal_code="10001",
            building_number="100",
        )
        feats = compute_pair_features(s1, cand)
        fd = dict(zip(FEATURE_NAMES, feats))

        assert fd["name_exact_match"] == 1.0
        assert fd["root_name_exact_match"] == 1.0
        assert fd["name_ratio"] == 1.0
        assert fd["addr_ratio"] == 1.0
        assert fd["addr_present_both"] == 1.0
        assert fd["house_number_match"] == 1.0
        assert fd["postal_exact_match"] == 1.0
        assert fd["postal_prefix_match"] == 1.0
        assert fd["numeric_token_overlap"] == 1.0
        assert fd["country_match"] == 1.0
        assert fd["combined_weighted_sim"] == 1.0

    def test_typo_resilience(self):
        """Typo scenario: minor character insertions or deletions."""
        s1 = make_record(clean_name="acme logistics international", root_name="acme logistics international")
        cand = make_record(clean_name="acm logistcs internatonal", root_name="acm logistcs internatonal")
        feats = compute_pair_features(s1, cand)
        fd = dict(zip(FEATURE_NAMES, feats))

        assert fd["name_exact_match"] == 0.0
        assert fd["name_ratio"] > 0.85
        assert fd["name_jaro_winkler"] > 0.88
        assert fd["name_char3_jaccard"] > 0.50

    def test_abbreviation(self):
        """Abbreviation scenario: expanded tokens vs abbreviated tokens."""
        s1 = make_record(clean_name="international business machines", root_name="international business machines")
        cand = make_record(clean_name="ibm", root_name="ibm")
        feats = compute_pair_features(s1, cand)
        fd = dict(zip(FEATURE_NAMES, feats))

        assert fd["name_exact_match"] == 0.0
        assert fd["name_len_diff_ratio"] > 0.80

        # Sub-token containment scenario
        s1_c = make_record(clean_name="starbucks coffee company", root_name="starbucks coffee")
        cand_c = make_record(clean_name="starbucks", root_name="starbucks")
        feats_c = compute_pair_features(s1_c, cand_c)
        fd_c = dict(zip(FEATURE_NAMES, feats_c))
        assert fd_c["name_token_containment"] == 1.0

    def test_missing_address(self):
        """Missing address scenario: address features evaluate to 0.0 and addr_present_both to 0.0."""
        s1 = make_record(clean_address="", building_number="")
        cand = make_record(clean_address="100 market street", building_number="100")
        feats = compute_pair_features(s1, cand)
        fd = dict(zip(FEATURE_NAMES, feats))

        assert fd["addr_present_both"] == 0.0
        assert fd["addr_ratio"] == 0.0
        assert fd["addr_token_sort_ratio"] == 0.0
        assert fd["addr_token_set_ratio"] == 0.0
        assert fd["addr_jaccard"] == 0.0
        assert fd["house_number_match"] == -1.0  # missing sentinel

    def test_missing_postal(self):
        """Missing postal scenario: structural postal features return -1.0 sentinel."""
        s1 = make_record(postal_code="")
        cand = make_record(postal_code="75008")
        feats = compute_pair_features(s1, cand)
        fd = dict(zip(FEATURE_NAMES, feats))

        assert fd["postal_exact_match"] == -1.0
        assert fd["postal_prefix_match"] == -1.0

    def test_same_building_number(self):
        """Same building number scenario: house_number_match returns 1.0."""
        s1 = make_record(building_number="450", numeric_tokens={"450"})
        cand = make_record(building_number="450", numeric_tokens={"450"})
        feats = compute_pair_features(s1, cand)
        fd = dict(zip(FEATURE_NAMES, feats))

        assert fd["house_number_match"] == 1.0

    def test_different_building_number(self):
        """Different building number scenario: house_number_match returns 0.0 (not -1.0)."""
        s1 = make_record(building_number="450", numeric_tokens={"450"})
        cand = make_record(building_number="452", numeric_tokens={"452"})
        feats = compute_pair_features(s1, cand)
        fd = dict(zip(FEATURE_NAMES, feats))

        assert fd["house_number_match"] == 0.0

    def test_s2_candidate(self):
        """S2 candidate scenario: target_source_is_s3 returns 0.0."""
        s1 = make_record(entity_id="S1-001")
        cand = make_record(entity_id="S2-999")
        feats = compute_pair_features(s1, cand)
        fd = dict(zip(FEATURE_NAMES, feats))

        assert fd["target_source_is_s3"] == 0.0

    def test_s3_candidate(self):
        """S3 candidate scenario: target_source_is_s3 returns 1.0."""
        s1 = make_record(entity_id="S1-001")
        cand = make_record(entity_id="S3-888")
        feats = compute_pair_features(s1, cand)
        fd = dict(zip(FEATURE_NAMES, feats))

        assert fd["target_source_is_s3"] == 1.0

    def test_transliteration_like_corruption(self):
        """Transliteration / phonetic variation scenario (e.g. Indian/French phonetics)."""
        s1 = make_record(clean_name="sharma enterprizes", root_name="sharma enterprizes")
        cand = make_record(clean_name="sarma enterprises", root_name="sarma enterprises")
        feats = compute_pair_features(s1, cand)
        fd = dict(zip(FEATURE_NAMES, feats))

        assert fd["name_exact_match"] == 0.0
        assert fd["name_jaro_winkler"] > 0.88
        assert fd["name_char3_jaccard"] > 0.50
        assert fd["name_token_sort_ratio"] > 0.80


class TestRangeValidation:
    """Validates that all features fall within their bounded mathematical ranges."""

    def test_range_across_diverse_records(self):
        pairs = [
            (make_record(clean_name="apple", postal_code="94016", building_number="1"), make_record(clean_name="apple inc", postal_code="94016", building_number="1")),
            (make_record(clean_name="target", clean_address="", postal_code="", building_number=""), make_record(clean_name="walmart", clean_address="", postal_code="", building_number="")),
            (make_record(clean_name="", country="US"), make_record(clean_name="", country="FRANCE")),
            (make_record(clean_name="a", postal_code="1"), make_record(clean_name="z", postal_code="2")),
        ]

        sentinel_features = {"house_number_match", "postal_exact_match", "postal_prefix_match"}

        for s1, cand in pairs:
            feats = compute_pair_features(s1, cand)
            fd = dict(zip(FEATURE_NAMES, feats))
            for name, val in fd.items():
                if name in sentinel_features:
                    assert val in (-1.0, 0.0, 1.0), f"Sentinel feature {name} has unexpected value: {val}"
                else:
                    assert 0.0 <= val <= 1.0, f"Feature {name} = {val} out of bounds [0, 1]"


class TestLightGBMFeatureContract:
    """Verifies that EntityResolutionModel enforces exactly 23 columns."""

    def test_model_train_accepts_23_columns(self):
        model = EntityResolutionModel()
        X_train = np.random.rand(50, 23).astype(np.float32)
        y_train = np.random.randint(0, 2, size=50).astype(np.int32)
        X_val = np.random.rand(10, 23).astype(np.float32)
        y_val = np.random.randint(0, 2, size=10).astype(np.int32)

        # Should execute without dimension assertion error
        model.train(X_train, y_train, X_val=X_val, y_val=y_val)
        assert model.clf is not None

        # Predict should also enforce 23 columns
        X_test = np.random.rand(5, 23).astype(np.float32)
        probs = model.predict_proba(X_test)
        assert len(probs) == 5

    def test_model_train_rejects_wrong_column_count(self):
        model = EntityResolutionModel()
        # 22 columns instead of 23
        X_train_bad = np.random.rand(50, 22).astype(np.float32)
        y_train = np.random.randint(0, 2, size=50).astype(np.int32)

        with pytest.raises(AssertionError, match="Feature dimension mismatch"):
            model.train(X_train_bad, y_train)

    def test_model_predict_rejects_wrong_column_count(self):
        model = EntityResolutionModel()
        X_train = np.random.rand(50, 23).astype(np.float32)
        y_train = np.random.randint(0, 2, size=50).astype(np.int32)
        model.train(X_train, y_train)

        # 24 columns instead of 23
        X_test_bad = np.random.rand(5, 24).astype(np.float32)
        with pytest.raises(AssertionError, match="Feature dimension mismatch"):
            model.predict_proba(X_test_bad)

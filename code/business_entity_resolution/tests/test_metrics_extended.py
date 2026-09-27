"""
Extended Metrics Tests.
Verifies exact F0.5 formula, edge cases, and macro averaging correctness.
"""

import pytest
from business_entity_resolution.metrics import compute_entity_f05, compute_macro_f05


class TestEntityF05:
    def test_singleton_correct(self):
        assert compute_entity_f05(set(), set()) == 1.0

    def test_singleton_false_positive(self):
        assert compute_entity_f05(set(), {"S2-001"}) == 0.0

    def test_non_singleton_missed_entirely(self):
        assert compute_entity_f05({"S2-001"}, set()) == 0.0

    def test_perfect_match(self):
        assert compute_entity_f05({"S2-001", "S3-002"}, {"S2-001", "S3-002"}) == 1.0

    def test_one_fp(self):
        """One FP: P=2/3, R=1.0 -> F0.5 = 1.25 * P * R / (0.25*P + R)"""
        gt = {"S2-001", "S3-002"}
        pred = {"S2-001", "S3-002", "S2-999"}
        score = compute_entity_f05(gt, pred)
        p = 2 / 3
        r = 1.0
        expected = 1.25 * p * r / (0.25 * p + r)
        assert abs(score - expected) < 1e-6

    def test_one_fn(self):
        """One FN: P=1.0, R=0.5 -> F0.5 = 1.25 * 1 * 0.5 / (0.25*1 + 0.5) = 0.8333"""
        gt = {"S2-001", "S3-002"}
        pred = {"S2-001"}
        score = compute_entity_f05(gt, pred)
        expected = 1.25 * 1.0 * 0.5 / (0.25 * 1.0 + 0.5)
        assert abs(score - expected) < 1e-6

    def test_complete_miss_non_singleton(self):
        """Predict wrong entity entirely."""
        score = compute_entity_f05({"S2-001"}, {"S3-999"})
        assert score == 0.0

    def test_formula_matches_official_example(self):
        """Verify official PDF example: pred=[S2-00047, S2-00193, S3-00812], gt=[S2-00047, S3-00812]"""
        pred = {"S2-00047", "S2-00193", "S3-00812"}
        gt = {"S2-00047", "S3-00812"}
        score = compute_entity_f05(gt, pred)
        # P=2/3, R=1.0
        p, r = 2 / 3, 1.0
        expected = 1.25 * p * r / (0.25 * p + r)
        assert abs(score - expected) < 1e-4

    def test_gt_superset_of_pred(self):
        """GT is larger than pred: partial recall."""
        gt = {"S2-001", "S2-002", "S2-003"}
        pred = {"S2-001", "S2-002"}
        score = compute_entity_f05(gt, pred)
        tp = 2
        p = 2 / 2  # 1.0
        r = 2 / 3
        expected = 1.25 * p * r / (0.25 * p + r)
        assert abs(score - expected) < 1e-6


class TestMacroF05:
    def test_empty_ground_truth(self):
        assert compute_macro_f05({}, {}) == 0.0

    def test_all_singletons_correct(self):
        gt = {"S1-1": set(), "S1-2": set()}
        preds = {"S1-1": set(), "S1-2": set()}
        assert compute_macro_f05(gt, preds) == 1.0

    def test_mixed_scores(self):
        gt = {
            "S1-1": {"S2-1"},        # perfect -> 1.0
            "S1-2": set(),           # singleton correct -> 1.0
            "S1-3": set(),           # singleton wrong -> 0.0
            "S1-4": {"S2-2", "S3-2"},  # partial
        }
        preds = {
            "S1-1": {"S2-1"},
            "S1-2": set(),
            "S1-3": {"S2-99"},
            "S1-4": {"S2-2"},       # P=1.0, R=0.5
        }
        # S1-4: F0.5 = 1.25*1*0.5/(0.25*1+0.5) = 0.625/0.75
        s1_4 = 1.25 * 1.0 * 0.5 / (0.25 * 1.0 + 0.5)
        expected = (1.0 + 1.0 + 0.0 + s1_4) / 4.0
        assert abs(compute_macro_f05(gt, preds) - expected) < 1e-6

    def test_missing_prediction_treated_as_empty(self):
        """If an S1 entity has no prediction entry, treat as empty prediction."""
        gt = {"S1-1": {"S2-1"}, "S1-2": {"S2-2"}}
        preds = {"S1-1": {"S2-1"}}  # S1-2 missing from predictions
        # S1-2 gets score 0 (non-empty GT, empty prediction)
        expected = (1.0 + 0.0) / 2.0
        assert abs(compute_macro_f05(gt, preds) - expected) < 1e-6

    def test_duplicate_prediction_ids_handled(self):
        """Duplicate IDs in prediction set must not inflate TP count."""
        gt = {"S1-1": {"S2-001"}}
        preds = {"S1-1": {"S2-001"}}  # sets deduplicate automatically
        assert compute_macro_f05(gt, preds) == 1.0

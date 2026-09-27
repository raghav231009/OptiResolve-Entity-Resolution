"""
Threshold Optimization and Metric Verification Tests.
Covers:
  - Exact competition metric evaluation
  - All singletons
  - No predictions (all empty)
  - Perfect predictions
  - False positives (singleton and non-singleton)
  - False negatives (partial and total)
  - Mixed entities with exact arithmetic verification
  - Two-phase coarse and fine grid sweep
  - Five recorded metrics: threshold, macro_f05, predicted_links, empty_predictions, singleton_false_positives
  - Reproducibility and deterministic tie-breaking
"""

from typing import Dict, List, Set, Tuple
import pytest
import numpy as np

from business_entity_resolution.metrics import compute_entity_f05, compute_macro_f05
from business_entity_resolution.threshold import (
    ThresholdMetric,
    evaluate_threshold,
    optimize_threshold,
)


class TestCompetitionMetricVerification:
    """Requirement 1, 2, 3: Reimplement/verify exact competition metric and singleton rules."""

    def test_singleton_empty_gt_and_empty_pred_equals_one(self):
        """empty GT + empty prediction = 1"""
        assert compute_entity_f05(set(), set()) == 1.0

    def test_singleton_empty_gt_and_non_empty_pred_equals_zero(self):
        """empty GT + non-empty prediction = 0"""
        assert compute_entity_f05(set(), {"S2-001"}) == 0.0
        assert compute_entity_f05(set(), {"S2-001", "S3-002"}) == 0.0

    def test_non_empty_gt_and_empty_pred_equals_zero(self):
        """non-empty GT + empty prediction = 0"""
        assert compute_entity_f05({"S2-001"}, set()) == 0.0
        assert compute_entity_f05({"S2-001", "S3-002"}, set()) == 0.0

    def test_otherwise_calculates_entity_f05(self):
        """
        Verify exact F_0.5 formula:
        F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
        """
        gt = {"S2-10", "S2-20"}
        pred = {"S2-10", "S2-30"}  # TP=1, FP=1, FN=1 -> P=0.5, R=0.5
        score = compute_entity_f05(gt, pred)
        p, r = 0.5, 0.5
        expected = (1.25 * p * r) / (0.25 * p + r)  # 0.3125 / 0.625 = 0.5
        assert abs(score - expected) < 1e-6
        assert abs(score - 0.5) < 1e-6

    def test_macro_average_across_s1_entities(self):
        """Macro-average across all S1 entities in ground truth."""
        gt = {
            "S1-1": set(),
            "S1-2": {"S2-1"},
            "S1-3": {"S2-2"},
        }
        pred = {
            "S1-1": set(),          # 1.0
            "S1-2": {"S2-1"},       # 1.0
            "S1-3": set(),          # 0.0
        }
        score = compute_macro_f05(gt, pred)
        assert abs(score - (1.0 + 1.0 + 0.0) / 3.0) < 1e-6


class TestAllSingleton:
    """Test behavior when all validation entities are singletons."""

    def test_all_singletons_perfect_empty(self):
        val_gt = {f"S1-{i}": set() for i in range(10)}
        val_scored: Dict[str, List[Tuple[str, float]]] = {f"S1-{i}": [] for i in range(10)}

        eval_res = evaluate_threshold(val_gt, val_scored, 0.80)
        assert eval_res["macro_f05"] == 1.0
        assert eval_res["predicted_links"] == 0
        assert eval_res["empty_predictions"] == 10
        assert eval_res["singleton_false_positives"] == 0

    def test_all_singletons_with_false_positives_eliminated_by_threshold(self):
        """
        Low threshold causes singleton false positives;
        higher threshold eliminates candidates and restores Macro F0.5 = 1.0.
        """
        val_gt = {
            "S1-1": set(),
            "S1-2": set(),
            "S1-3": set(),
        }
        # Candidates scored below 0.85
        val_scored = {
            "S1-1": [("S2-1", 0.60)],
            "S1-2": [("S2-2", 0.72)],
            "S1-3": [("S2-3", 0.80)],
        }

        # At tau=0.55: all 3 have false positives -> Macro F0.5 = 0.0
        low_res = evaluate_threshold(val_gt, val_scored, 0.55)
        assert low_res["macro_f05"] == 0.0
        assert low_res["singleton_false_positives"] == 3
        assert low_res["predicted_links"] == 3
        assert low_res["empty_predictions"] == 0

        # At tau=0.75: S1-1 and S1-2 are empty (1.0 each), S1-3 has FP (0.0) -> Macro F0.5 = 2/3
        mid_res = evaluate_threshold(val_gt, val_scored, 0.75)
        assert abs(mid_res["macro_f05"] - 2.0 / 3.0) < 1e-6
        assert mid_res["singleton_false_positives"] == 1
        assert mid_res["predicted_links"] == 1
        assert mid_res["empty_predictions"] == 2

        # At tau=0.85: all empty -> Macro F0.5 = 1.0
        high_res = evaluate_threshold(val_gt, val_scored, 0.85)
        assert high_res["macro_f05"] == 1.0
        assert high_res["singleton_false_positives"] == 0
        assert high_res["predicted_links"] == 0
        assert high_res["empty_predictions"] == 3

        # Threshold optimization should find tau >= 0.81
        best_tau, best_score, history = optimize_threshold(
            val_gt, val_scored, search_start=0.50, search_end=0.99, step=0.05
        )
        assert best_score == 1.0
        assert best_tau >= 0.80


class TestNoPredictions:
    """Test behavior when no predictions are made (all empty)."""

    def test_no_predictions_statistics(self):
        val_gt = {
            "S1-1": set(),          # singleton -> gets 1.0
            "S1-2": set(),          # singleton -> gets 1.0
            "S1-3": {"S2-1"},       # non-singleton -> gets 0.0
            "S1-4": {"S2-2", "S3-2"}, # non-singleton -> gets 0.0
        }
        val_scored: Dict[str, List[Tuple[str, float]]] = {k: [] for k in val_gt}

        res = evaluate_threshold(val_gt, val_scored, 0.70)
        assert res["predicted_links"] == 0
        assert res["empty_predictions"] == 4
        assert res["singleton_false_positives"] == 0
        # 2 singletons correct (1.0), 2 non-singletons missed (0.0) -> 2/4 = 0.50
        assert res["macro_f05"] == 0.50


class TestPerfectPredictions:
    """Test behavior when predictions perfectly match ground truth."""

    def test_perfect_predictions_statistics(self):
        val_gt = {
            "S1-1": {"S2-1"},
            "S1-2": set(),
            "S1-3": {"S2-3", "S3-3"},
        }
        # Scored with high confidence
        val_scored = {
            "S1-1": [("S2-1", 0.95)],
            "S1-2": [],
            "S1-3": [("S2-3", 0.92), ("S3-3", 0.90)],
        }

        res = evaluate_threshold(val_gt, val_scored, 0.85)
        assert res["macro_f05"] == 1.0
        assert res["predicted_links"] == 3
        assert res["empty_predictions"] == 1  # S1-2 is empty
        assert res["singleton_false_positives"] == 0


class TestFalsePositives:
    """Test false positives on both non-singletons and singletons."""

    def test_non_singleton_false_positives(self):
        val_gt = {"S1-1": {"S2-1"}}
        val_scored = {"S1-1": [("S2-1", 0.95), ("S2-FP", 0.80)]}

        # At tau=0.75: TP=1, FP=1 -> P=0.5, R=1.0 -> F0.5 = 1.25*0.5*1/(0.25*0.5+1) = 0.625/1.125 = 5/9
        res_fp = evaluate_threshold(val_gt, val_scored, 0.75)
        expected_fp = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0)
        assert abs(res_fp["macro_f05"] - expected_fp) < 1e-6
        assert res_fp["predicted_links"] == 2
        assert res_fp["singleton_false_positives"] == 0

        # At tau=0.85: FP filtered out -> P=1.0, R=1.0 -> F0.5 = 1.0
        res_clean = evaluate_threshold(val_gt, val_scored, 0.85)
        assert res_clean["macro_f05"] == 1.0
        assert res_clean["predicted_links"] == 1

    def test_singleton_false_positive_count(self):
        val_gt = {
            "S1-1": set(),
            "S1-2": set(),
            "S1-3": {"S2-1"},
        }
        val_scored = {
            "S1-1": [("S2-99", 0.70)],  # FP
            "S1-2": [],                  # true empty
            "S1-3": [("S2-1", 0.90)],    # true match
        }
        res = evaluate_threshold(val_gt, val_scored, 0.65)
        assert res["singleton_false_positives"] == 1
        assert res["predicted_links"] == 2
        assert res["empty_predictions"] == 1


class TestFalseNegatives:
    """Test false negatives: partial recall and missed non-singletons."""

    def test_partial_false_negative(self):
        val_gt = {"S1-1": {"S2-1", "S2-2"}}
        val_scored = {"S1-1": [("S2-1", 0.95), ("S2-2", 0.65)]}

        # At tau=0.80: S2-2 is not predicted -> TP=1, FP=0, FN=1 -> P=1.0, R=0.5
        # F0.5 = 1.25*1*0.5 / (0.25*1+0.5) = 0.625/0.75 = 5/6
        res = evaluate_threshold(val_gt, val_scored, 0.80)
        expected = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5)
        assert abs(res["macro_f05"] - expected) < 1e-6
        assert res["predicted_links"] == 1
        assert res["empty_predictions"] == 0

    def test_total_false_negative_counts_as_empty(self):
        val_gt = {"S1-1": {"S2-1"}}
        val_scored = {"S1-1": [("S2-1", 0.55)]}

        # At tau=0.70: nothing predicted -> non-singleton + empty pred = 0.0
        res = evaluate_threshold(val_gt, val_scored, 0.70)
        assert res["macro_f05"] == 0.0
        assert res["empty_predictions"] == 1
        assert res["predicted_links"] == 0


class TestMixedEntities:
    """Test mixed dataset with singletons, non-singletons, FPs, and FNs."""

    def test_mixed_exact_macro_calculation(self):
        val_gt = {
            "S1-1": set(),                  # Singleton correct -> 1.0
            "S1-2": set(),                  # Singleton with FP -> 0.0
            "S1-3": {"S2-1"},               # Perfect -> 1.0
            "S1-4": {"S2-2"},               # Non-singleton with FP: {S2-2, S2-99} -> 5/9 = 0.55556
            "S1-5": {"S2-3", "S3-3"},       # Non-singleton with FN: {S2-3} -> 5/6 = 0.83333
            "S1-6": {"S2-4"},               # Non-singleton complete FN -> 0.0
        }
        val_scored = {
            "S1-1": [],
            "S1-2": [("S2-98", 0.85)],
            "S1-3": [("S2-1", 0.90)],
            "S1-4": [("S2-2", 0.90), ("S2-99", 0.80)],
            "S1-5": [("S2-3", 0.90)],
            "S1-6": [],
        }

        res = evaluate_threshold(val_gt, val_scored, 0.75)
        # Expected scores:
        # S1-1: 1.0
        # S1-2: 0.0 (singleton FP)
        # S1-3: 1.0 (perfect)
        # S1-4: (1.25*0.5*1)/(0.25*0.5+1) = 0.625/1.125 = 5/9
        # S1-5: (1.25*1*0.5)/(0.25*1+0.5) = 0.625/0.75 = 5/6
        # S1-6: 0.0 (complete miss)
        s4 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0)
        s5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5)
        expected_macro = (1.0 + 0.0 + 1.0 + s4 + s5 + 0.0) / 6.0

        assert abs(res["macro_f05"] - expected_macro) < 1e-6
        assert res["predicted_links"] == 5  # S1-2(1) + S1-3(1) + S1-4(2) + S1-5(1)
        assert res["empty_predictions"] == 2  # S1-1 and S1-6
        assert res["singleton_false_positives"] == 1  # S1-2


class TestTwoPhaseGridOptimization:
    """Test two-phase coarse + fine grid threshold optimization."""

    def test_two_phase_fine_resolution(self):
        """
        Construct pairs where optimal threshold lies at 0.754 (between 0.75 and 0.76).
        Fine search should discover this finer threshold.
        """
        # Ground truth: 1 non-singleton with true match at 0.755, distractor at 0.752
        val_gt = {
            "S1-1": {"S2-1"},
            "S1-2": set(),
        }
        val_scored = {
            "S1-1": [("S2-1", 0.755), ("S2-distractor", 0.752)],
            "S1-2": [],
        }

        best_tau, best_score, history = optimize_threshold(
            val_gt,
            val_scored,
            search_start=0.50,
            search_end=0.99,
            step=0.02,
            fine_step=0.002,
            fine_window=0.03,
        )

        # At tau=0.754: S2-1 (0.755 >= 0.754) is selected, S2-distractor (0.752 < 0.754) is dropped
        # Result: S1-1 perfect (1.0), S1-2 singleton perfect (1.0) -> Macro F0.5 = 1.0
        assert best_score == 1.0
        assert 0.753 <= best_tau <= 0.755

        # Check that fine search generated records with all 5 metrics
        for tau, metric in history.items():
            assert isinstance(metric, float)
            assert hasattr(metric, "predicted_links")
            assert hasattr(metric, "empty_predictions")
            assert hasattr(metric, "singleton_false_positives")
            assert "macro_f05" in metric
            assert "threshold" in metric

    def test_threshold_metric_backward_compatibility(self):
        """ThresholdMetric must behave as float, dict, and object with attributes."""
        record = {
            "threshold": 0.88,
            "macro_f05": 0.942,
            "predicted_links": 1500,
            "empty_predictions": 300,
            "singleton_false_positives": 12,
        }
        m = ThresholdMetric(record)
        # Float behavior
        assert isinstance(m, float)
        assert abs(float(m) - 0.942) < 1e-6
        assert m > 0.90
        # Dict behavior
        assert m["predicted_links"] == 1500
        assert m["singleton_false_positives"] == 12
        assert m.get("empty_predictions") == 300
        # Attribute behavior
        assert m.predicted_links == 1500
        assert m.empty_predictions == 300
        assert m.singleton_false_positives == 12
        assert m.threshold == 0.88
        # Serialization
        d = m.to_dict()
        assert d["threshold"] == 0.88


class TestReproducibility:
    """Requirement 12: Verify threshold optimization is reproducible and deterministic."""

    def test_optimization_reproducibility(self):
        val_gt = {
            "S1-1": {"S2-1"},
            "S1-2": set(),
            "S1-3": {"S2-3"},
            "S1-4": set(),
        }
        val_scored = {
            "S1-1": [("S2-1", 0.85), ("S2-X", 0.72)],
            "S1-2": [("S2-Y", 0.60)],
            "S1-3": [("S2-3", 0.91)],
            "S1-4": [],
        }

        tau1, score1, history1 = optimize_threshold(val_gt, val_scored)
        tau2, score2, history2 = optimize_threshold(val_gt, val_scored)

        assert tau1 == tau2
        assert score1 == score2
        assert len(history1) == len(history2)
        for t in history1:
            assert t in history2
            assert abs(history1[t] - history2[t]) < 1e-9
            assert history1[t].predicted_links == history2[t].predicted_links
            assert history1[t].singleton_false_positives == history2[t].singleton_false_positives

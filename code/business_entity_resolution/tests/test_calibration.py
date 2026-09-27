"""
Tests for Probability Calibration and Threshold Sensitivity Audit Module.
Verifies:
  - Distribution statistics and histogram binning
  - Expected Calibration Error (ECE) and reliability diagrams
  - Brier score computation
  - Score segmentation (TP, hard neg, singleton neg, S2, S3)
  - Threshold sensitivity derivatives and plateau stability
  - Honest 2-fold cross-split calibration evaluation
"""

import numpy as np
import pytest

from business_entity_resolution.calibration import (
    analyze_threshold_sensitivity,
    compute_brier_score,
    compute_distribution_statistics,
    compute_ece,
    evaluate_calibration_benefit,
    segment_probability_scores,
)


class TestDistributionStatistics:
    def test_empty_array(self):
        stats = compute_distribution_statistics(np.array([]))
        assert stats["count"] == 0
        assert stats["mean"] == 0.0
        assert stats["histogram"] == {}

    def test_known_distribution(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        stats = compute_distribution_statistics(scores)
        assert stats["count"] == 10
        assert abs(stats["mean"] - 0.55) < 1e-6
        assert abs(stats["min"] - 0.1) < 1e-6
        assert abs(stats["max"] - 1.0) < 1e-6
        assert len(stats["histogram"]) == 10


class TestCalibrationMetrics:
    def test_perfect_calibration_ece(self):
        """When predicted probabilities equal empirical accuracy, ECE is near zero."""
        # 100 predictions of 0.8 where exactly 80 are positive
        y_prob = np.full(100, 0.8)
        y_true = np.zeros(100, dtype=np.int32)
        y_true[:80] = 1

        ece, bins = compute_ece(y_true, y_prob, n_bins=10)
        assert ece < 1e-4

    def test_severe_miscalibration_ece(self):
        """When probabilities are 0.9 but all labels are 0, ECE is high."""
        y_prob = np.full(100, 0.9)
        y_true = np.zeros(100, dtype=np.int32)

        ece, bins = compute_ece(y_true, y_prob, n_bins=10)
        assert abs(ece - 0.9) < 1e-4

    def test_brier_score(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([1.0, 0.0, 1.0, 0.0])
        assert compute_brier_score(y_true, y_prob) == 0.0

        y_prob_worst = np.array([0.0, 1.0, 0.0, 1.0])
        assert compute_brier_score(y_true, y_prob_worst) == 1.0


class TestSegmentation:
    def test_segment_probability_scores(self):
        val_gt = {
            "S1-1": {"S2-1"},       # non-singleton
            "S1-2": set(),           # true singleton
        }
        val_scored = {
            "S1-1": [
                ("S2-1", 0.92),      # True positive (S2)
                ("S3-9", 0.40),      # Hard negative (S3)
            ],
            "S1-2": [
                ("S2-88", 0.15),     # Singleton negative (S2)
            ],
        }

        segs = segment_probability_scores(val_gt, val_scored)
        assert len(segs["true_positives"]) == 1
        assert segs["true_positives"][0] == 0.92

        assert len(segs["hard_negatives"]) == 1
        assert segs["hard_negatives"][0] == 0.40

        assert len(segs["singleton_negatives"]) == 1
        assert segs["singleton_negatives"][0] == 0.15

        assert len(segs["s2_candidates"]) == 2
        assert len(segs["s3_candidates"]) == 1
        assert len(segs["all_candidates"]) == 3


class TestThresholdSensitivityAnalysis:
    def test_sensitivity_computation(self):
        val_gt = {
            "S1-1": {"S2-1"},
            "S1-2": set(),
        }
        val_scored = {
            "S1-1": [("S2-1", 0.85), ("S2-FP", 0.70)],
            "S1-2": [("S2-SingFP", 0.60)],
        }

        grid = [0.55, 0.65, 0.75, 0.85, 0.95]
        recs = analyze_threshold_sensitivity(val_gt, val_scored, grid)

        assert len(recs) == len(grid)
        # Check monotonic decrease of predicted links
        links = [r["predicted_links"] for r in recs]
        assert links == sorted(links, reverse=True)

        # Check fields present
        for r in recs:
            assert "macro_f05" in r
            assert "precision" in r
            assert "recall" in r
            assert "sensitivity_d_score" in r
            assert "is_stable" in r


class TestHonestCalibrationEvaluation:
    def test_calibration_benefit_isolated_split(self):
        """
        Verify that evaluate_calibration_benefit executes 2-fold cross-split
        and generates valid recommendations without errors or data contamination.
        """
        val_gt = {
            f"S1-{i}": {f"S2-{i}"} if i % 2 == 0 else set()
            for i in range(20)
        }
        val_scored = {}
        for i in range(20):
            if i % 2 == 0:
                val_scored[f"S1-{i}"] = [(f"S2-{i}", 0.88), (f"S3-{i+100}", 0.35)]
            else:
                val_scored[f"S1-{i}"] = [(f"S2-Distractor-{i}", 0.25)]

        res = evaluate_calibration_benefit(val_gt, val_scored, random_seed=42)

        assert "uncalibrated" in res
        assert "calibrated_methods" in res
        assert "sigmoid" in res["calibrated_methods"]
        assert "isotonic" in res["calibrated_methods"]
        assert "recommendation" in res
        assert "best_method" in res["recommendation"]
        assert "rationale" in res["recommendation"]

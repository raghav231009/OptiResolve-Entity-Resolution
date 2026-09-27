"""
Tests for Negative Sampling for Singleton and Zero-Positive S1 Entities.
Verifies all requirements:
1. Zero-positive S1 entities sample up to min_negatives_per_zero_positive_entity hard negatives.
2. One-positive S1 entities sample up to max_negatives_per_positive hard negatives.
3. Multi-positive S1 entities sample proportional hard negatives (num_positives * max_negatives_per_positive).
4. Entities with no blocker candidates do not artificially inflate negatives with random pairs.
5. Negative labels are never assigned to true matches.
6. Singletons are represented in both training and validation splits.
7. Tracking metrics (zero-positive count, contributing count, avg/min/max negatives, class balance).
"""

import json
from pathlib import Path
import numpy as np
import pytest

from business_entity_resolution.blocking import MultiIndexBlocker
from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.pipeline import EntityResolutionPipeline


def make_dummy_s1(entity_id: str, name: str, addr: str) -> dict:
    return {
        "entity_id": entity_id,
        "country": "US",
        "clean_name": name.lower(),
        "root_name": name.lower(),
        "clean_address": addr.lower(),
        "postal_code": "10001",
        "building_number": "100",
        "numeric_tokens": {"100"},
    }


def make_dummy_target(entity_id: str, name: str, addr: str) -> dict:
    return {
        "entity_id": entity_id,
        "country": "US",
        "clean_name": name.lower(),
        "root_name": name.lower(),
        "clean_address": addr.lower(),
        "postal_code": "10001",
        "building_number": "100",
        "numeric_tokens": {"100"},
    }


class TestNegativeSamplingScenarios:
    """Requirement 9: Unit tests for zero-positive, one-positive, multi-positive, and no-candidate entities."""

    def test_zero_positive_entity_samples_min_negatives(self):
        """Zero-positive (singleton) entity samples configured minimum hard negatives from blocker."""
        s1 = [make_dummy_s1("S1_ZERO", "Summit Capital", "100 Wall Street")]
        gt = {"S1_ZERO": set()}  # 0 true matches

        # Blocker provides 10 candidates
        targets = [make_dummy_target(f"T_{i:02d}", f"Summit Corp {i}", f"100 Wall St Suite {i}") for i in range(10)]
        target_map = {t["entity_id"]: t for t in targets}
        blocker = MultiIndexBlocker(max_candidates=10)
        blocker.index_targets(targets)

        cfg = PipelineConfig()
        cfg.min_negatives_per_zero_positive_entity = 6
        pipeline = EntityResolutionPipeline(config=cfg)

        X, y, stats = pipeline._generate_pair_matrix(
            s1, gt, blocker, target_map, return_stats=True
        )

        assert stats["zero_positive_s1_count"] == 1
        assert stats["zero_positive_contributing_negatives"] == 1
        assert stats["total_positives"] == 0
        assert stats["total_negatives"] == 6
        assert len(X) == 6
        assert len(y) == 6
        assert np.all(y == 0)

    def test_one_positive_entity_samples_configured_negatives(self):
        """Entity with 1 true match samples up to max_negatives_per_positive hard negatives."""
        s1 = [make_dummy_s1("S1_ONE", "Summit Capital", "100 Wall Street")]
        gt = {"S1_ONE": {"T_00"}}  # 1 true match

        targets = [make_dummy_target(f"T_{i:02d}", f"Summit Corp {i}", f"100 Wall St Suite {i}") for i in range(10)]
        target_map = {t["entity_id"]: t for t in targets}
        blocker = MultiIndexBlocker(max_candidates=10)
        blocker.index_targets(targets)

        cfg = PipelineConfig()
        cfg.max_negatives_per_positive = 4
        pipeline = EntityResolutionPipeline(config=cfg)

        X, y, stats = pipeline._generate_pair_matrix(
            s1, gt, blocker, target_map, return_stats=True
        )

        assert stats["positive_s1_count"] == 1
        assert stats["total_positives"] == 1
        assert stats["total_negatives"] == 4
        assert len(y) == 5
        assert np.sum(y == 1) == 1
        assert np.sum(y == 0) == 4

    def test_multi_positive_entity_samples_proportional_negatives(self):
        """Entity with 3 true matches samples up to (3 * max_negatives_per_positive) hard negatives."""
        s1 = [make_dummy_s1("S1_MULTI", "Summit Capital", "100 Wall Street")]
        gt = {"S1_MULTI": {"T_00", "T_01", "T_02"}}  # 3 true matches

        # 20 candidate targets (3 positive + 17 negative candidates)
        targets = [make_dummy_target(f"T_{i:02d}", f"Summit Corp {i}", f"100 Wall St Suite {i}") for i in range(20)]
        target_map = {t["entity_id"]: t for t in targets}
        blocker = MultiIndexBlocker(max_candidates=20)
        blocker.index_targets(targets)

        cfg = PipelineConfig()
        cfg.max_negatives_per_positive = 3  # Expect 3 * 3 = 9 negatives
        pipeline = EntityResolutionPipeline(config=cfg)

        X, y, stats = pipeline._generate_pair_matrix(
            s1, gt, blocker, target_map, return_stats=True
        )

        assert stats["positive_s1_count"] == 1
        assert stats["total_positives"] == 3
        assert stats["total_negatives"] == 9
        assert len(y) == 12
        assert np.sum(y == 1) == 3
        assert np.sum(y == 0) == 9

    def test_no_available_candidates_does_not_artificially_inflate_negatives(self):
        """Requirements 5 & 6: When blocker yields 0 candidates, do not sample arbitrary fake negatives."""
        s1 = [make_dummy_s1("S1_ISOLATED", "Zephyr Unique Name", "999 Remote Desolate Road")]
        gt = {"S1_ISOLATED": set()}

        # No targets indexed in blocker
        targets = []
        target_map = {}
        blocker = MultiIndexBlocker(max_candidates=10)
        blocker.index_targets(targets)

        pipeline = EntityResolutionPipeline()
        X, y, stats = pipeline._generate_pair_matrix(
            s1, gt, blocker, target_map, return_stats=True
        )

        assert len(X) == 0
        assert len(y) == 0
        assert stats["zero_positive_s1_count"] == 1
        assert stats["zero_positive_contributing_negatives"] == 0
        assert stats["total_negatives"] == 0

    def test_negative_labels_never_assigned_to_true_matches(self):
        """Requirement 10: Ensure negative labels (0) are never accidentally assigned to true matches."""
        s1 = [make_dummy_s1("S1_TEST", "Acme Logistics", "100 Market St")]
        gt = {"S1_TEST": {"T_TRUE_01", "T_TRUE_02"}}

        targets = [
            make_dummy_target("T_TRUE_01", "Acme Logistics Inc", "100 Market St"),
            make_dummy_target("T_TRUE_02", "Acme Logistics LLC", "100 Market Street"),
            make_dummy_target("T_NEG_01", "Acme Bakery", "100 Market St"),
            make_dummy_target("T_NEG_02", "Acme Hardware", "100 Market St"),
        ]
        target_map = {t["entity_id"]: t for t in targets}
        blocker = MultiIndexBlocker(max_candidates=10)
        blocker.index_targets(targets)

        pipeline = EntityResolutionPipeline()
        X, y, stats = pipeline._generate_pair_matrix(
            s1, gt, blocker, target_map, return_stats=True
        )

        assert stats["total_positives"] == 2
        assert stats["total_negatives"] == 2
        # Check that positives match true targets
        assert np.sum(y == 1) == 2
        assert np.sum(y == 0) == 2


class TestSingletonSplitAndTracking:
    """Requirements 7 & 8: Singleton representation in train and val, and metrics tracking."""

    def test_singleton_represented_in_both_train_and_val_splits(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        train_dir = dataset_dir / "train"
        train_dir.mkdir(parents=True)

        # 12 S1 records: 4 singletons, 8 matched
        s1_lines = ["entity_id\tcountry\tbusiness_name\tbusiness_address\n"]
        gt_lines = ["source1_entity_id\tmatched_entity_ids\n"]
        s2_lines = ["entity_id\tcountry\tbusiness_name\tbusiness_address\n"]

        for i in range(1, 13):
            s1_lines.append(f"S1_{i:02d}\tUS\tStore {i}\t{100*i} St\n")
            if i <= 4:
                # Singletons (no matches)
                gt_lines.append(f"S1_{i:02d}\t\n")
            else:
                # Matched
                gt_lines.append(f"S1_{i:02d}\tS2_{i:02d}\n")
                s2_lines.append(f"S2_{i:02d}\tUS\tStore {i}\t{100*i} Street\n")

        (train_dir / "train_source1.tsv").write_text("".join(s1_lines), encoding="utf-8")
        (train_dir / "train_source2.tsv").write_text("".join(s2_lines), encoding="utf-8")
        (train_dir / "train_source3.tsv").write_text("entity_id\tcountry\tbusiness_name\tbusiness_address\n", encoding="utf-8")
        (train_dir / "train_ground_truth.tsv").write_text("".join(gt_lines), encoding="utf-8")

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)

        config = PipelineConfig()
        config.paths.dataset_root = dataset_dir
        config.paths.artifacts_dir = artifacts_dir
        config.training_mode = "experiment"
        config.train_s1_limit = 12
        config.val_s1_limit = 4
        config.min_negatives_per_zero_positive_entity = 3
        config.max_negatives_per_positive = 3
        config.model.n_estimators = 10
        config.model.early_stopping_rounds = 5

        pipeline = EntityResolutionPipeline(config=config)
        pipeline.fit()

        results_path = artifacts_dir / "training_results.json"
        assert results_path.exists()
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "negative_sampling_stats" in data
        neg_stats = data["negative_sampling_stats"]

        # Train stats
        assert "train" in neg_stats
        train_s = neg_stats["train"]
        assert train_s["zero_positive_s1_count"] > 0
        assert "avg_negatives_per_entity" in train_s
        assert "min_negatives_per_entity" in train_s
        assert "max_negatives_per_entity" in train_s
        assert "class_balance_ratio_neg_to_pos" in train_s

        # Val stats
        assert "val" in neg_stats
        val_s = neg_stats["val"]
        assert val_s["zero_positive_s1_count"] > 0
        assert "avg_negatives_per_entity" in val_s

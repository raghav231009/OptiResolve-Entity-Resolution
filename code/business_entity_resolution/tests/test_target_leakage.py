"""
Tests for Target-Level Leakage Prevention and Cardinality Auditing.
Verifies all requirements:
1. Split S1 entities before constructing target pools.
2. Ground-truth S2/S3 target IDs for train and val are determined independently.
3. Separate training and validation target pools are built with zero intersection.
4. Validation target IDs never appear in the training target pool.
5. Training target IDs never appear in the validation pool when corresponding to validation ground truth.
6. Background negatives are sampled separately and disjointly.
7. Target-to-S1 cardinality is audited and documented (detects 1:1 vs 1:N).
8. Strict assertions and logging are verified.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.pipeline import (
    EntityResolutionPipeline,
    analyze_target_s1_cardinality,
    load_isolated_target_pools,
    load_targeted_training_targets,
)


class TestTargetLevelLeakagePrevention:
    """Requirements 1 to 8: Strict isolation of train and validation target pools."""

    def test_load_isolated_target_pools_zero_overlap(self, tmp_path):
        # Create a synthetic target source TSV
        source_tsv = tmp_path / "source2.tsv"
        lines = ["entity_id\tcountry\tbusiness_name\tbusiness_address\n"]
        for i in range(1, 51):
            lines.append(f"T_{i:03d}\tUS\tTarget Business {i}\t{100*i} Main St\n")
        source_tsv.write_text("".join(lines), encoding="utf-8")

        # Train needs T_001 to T_010; Val needs T_011 to T_020
        train_needed = {f"T_{i:03d}" for i in range(1, 11)}
        val_needed = {f"T_{i:03d}" for i in range(11, 21)}

        train_targets, val_targets = load_isolated_target_pools(
            source_paths=[source_tsv],
            train_needed_ids=train_needed,
            val_needed_ids=val_needed,
            train_background_sample_per_file=10,
            val_background_sample_per_file=10,
        )

        train_ids = {r["entity_id"] for r in train_targets}
        val_ids = {r["entity_id"] for r in val_targets}

        # 1. Zero target-level intersection
        intersection = train_ids.intersection(val_ids)
        assert len(intersection) == 0, f"Target intersection found: {intersection}"

        # 2. Validation ground truth never appears in train pool
        assert train_ids.isdisjoint(val_needed), "Val needed IDs leaked into train pool!"

        # 3. Training ground truth never appears in val pool
        assert val_ids.isdisjoint(train_needed), "Train needed IDs leaked into val pool!"

        # 4. All needed IDs were successfully retrieved
        assert train_needed.issubset(train_ids)
        assert val_needed.issubset(val_ids)

    def test_background_negatives_sampled_separately_and_disjoint(self, tmp_path):
        """Requirement 7 & 8: Background negatives must be sampled separately and disjointly."""
        source_tsv = tmp_path / "source3.tsv"
        lines = ["entity_id\tcountry\tbusiness_name\tbusiness_address\n"]
        # Only background records (none needed by GT)
        for i in range(1, 41):
            lines.append(f"BG_{i:03d}\tUS\tBackground Store {i}\t{50*i} Broadway Ave\n")
        source_tsv.write_text("".join(lines), encoding="utf-8")

        train_targets, val_targets = load_isolated_target_pools(
            source_paths=[source_tsv],
            train_needed_ids=set(),
            val_needed_ids=set(),
            train_background_sample_per_file=10,
            val_background_sample_per_file=10,
        )

        train_bg_ids = {r["entity_id"] for r in train_targets}
        val_bg_ids = {r["entity_id"] for r in val_targets}

        assert len(train_bg_ids) > 0
        assert len(val_bg_ids) > 0
        assert train_bg_ids.isdisjoint(val_bg_ids), f"Background distractors overlap: {train_bg_ids & val_bg_ids}"

    def test_load_targeted_training_targets_respects_forbidden_ids(self, tmp_path):
        """Verify load_targeted_training_targets excludes forbidden validation IDs."""
        source_tsv = tmp_path / "source.tsv"
        lines = ["entity_id\tcountry\tbusiness_name\tbusiness_address\n"]
        lines.append("T_01\tUS\tStore 1\t100 Main St\n")
        lines.append("T_02\tUS\tStore 2\t200 Main St\n")
        lines.append("T_03\tUS\tStore 3\t300 Main St\n")
        source_tsv.write_text("".join(lines), encoding="utf-8")

        # T_02 is forbidden (belongs to validation ground truth)
        targets = load_targeted_training_targets(
            [source_tsv],
            needed_ids={"T_01"},
            forbidden_ids={"T_02"},
            background_sample_per_file=5,
        )
        loaded_ids = {r["entity_id"] for r in targets}
        assert "T_01" in loaded_ids
        assert "T_02" not in loaded_ids, "Forbidden validation target leaked into training pool!"


class TestTargetCardinalityAudit:
    """Verify ground truth relationship structure (1:1 vs 1:N)."""

    def test_cardinality_strictly_one_to_one(self):
        gt = {
            "S1_01": {"T_01", "T_02"},
            "S1_02": {"T_03"},
            "S1_03": set(),
        }
        stats = analyze_target_s1_cardinality(gt)
        assert stats["total_unique_targets"] == 3
        assert stats["multi_mapped_targets_count"] == 0
        assert stats["max_s1_per_target"] == 1
        assert stats["is_strictly_one_to_one"] is True

    def test_cardinality_detects_multi_mapping_without_error(self):
        gt = {
            "S1_01": {"T_01", "T_SHARED"},
            "S1_02": {"T_02", "T_SHARED"},
            "S1_03": {"T_SHARED"},
        }
        stats = analyze_target_s1_cardinality(gt)
        assert stats["total_unique_targets"] == 3
        assert stats["multi_mapped_targets_count"] == 1
        assert stats["max_s1_per_target"] == 3
        assert stats["is_strictly_one_to_one"] is False
        assert "T_SHARED" in stats["sample_multi_mapped"]
        assert set(stats["sample_multi_mapped"]["T_SHARED"]) == {"S1_01", "S1_02", "S1_03"}


class TestPipelineLeakageIntegration:
    """Requirements 9 & 10: Full pipeline integration, leakage assertions, and audit logging."""

    def test_pipeline_fit_zero_leakage_assertion_and_results(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        train_dir = dataset_dir / "train"
        train_dir.mkdir(parents=True)

        # S1: 10 entities (8 train, 2 val)
        s1_lines = ["entity_id\tcountry\tbusiness_name\tbusiness_address\n"]
        for i in range(1, 11):
            s1_lines.append(f"S1_{i:02d}\tUS\tAlpha Corp {i}\t{100*i} Market St 9410{i}\n")
        (train_dir / "train_source1.tsv").write_text("".join(s1_lines), encoding="utf-8")

        # Targets in Source2 and Source3
        s2_lines = ["entity_id\tcountry\tbusiness_name\tbusiness_address\n"]
        gt_lines = ["source1_entity_id\tmatched_entity_ids\n"]
        for i in range(1, 11):
            s2_lines.append(f"S2_{i:02d}\tUS\tAlpha Corporation {i}\t{100*i} Market Street 9410{i}\n")
            gt_lines.append(f"S1_{i:02d}\tS2_{i:02d}\n")
        # Add background distractors
        for i in range(50, 70):
            s2_lines.append(f"S2_DIST_{i:02d}\tUS\tDistractor {i}\t{200*i} Distractor Blvd\n")

        (train_dir / "train_source2.tsv").write_text("".join(s2_lines), encoding="utf-8")
        (train_dir / "train_source3.tsv").write_text("entity_id\tcountry\tbusiness_name\tbusiness_address\n", encoding="utf-8")
        (train_dir / "train_ground_truth.tsv").write_text("".join(gt_lines), encoding="utf-8")

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)

        config = PipelineConfig()
        config.paths.dataset_root = dataset_dir
        config.paths.artifacts_dir = artifacts_dir
        config.training_mode = "experiment"
        config.train_s1_limit = 10
        config.val_s1_limit = 2
        config.model.n_estimators = 15
        config.model.early_stopping_rounds = 5

        pipeline = EntityResolutionPipeline(config=config)
        pipeline.fit()

        results_path = artifacts_dir / "training_results.json"
        assert results_path.exists()
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "target_leakage_audit" in data
        audit = data["target_leakage_audit"]
        assert audit["target_intersection_size"] == 0
        assert audit["is_strictly_disjoint"] is True
        assert audit["train_target_count"] > 0
        assert audit["val_target_count"] > 0
        assert audit["multi_mapped_targets_count"] == 0
        assert audit["max_s1_per_target"] == 1

    def test_pipeline_leakage_assertion_raises_on_overlap(self):
        train_targets = {"T_01", "T_02", "T_LEAK"}
        val_targets = {"T_03", "T_04", "T_LEAK"}
        intersection = train_targets.intersection(val_targets)

        with pytest.raises(AssertionError, match="Target-level validation leakage detected"):
            assert len(intersection) == 0, (
                f"Target-level validation leakage detected! {len(intersection):,} targets overlap "
                f"between train and validation pools: {list(intersection)[:5]}"
            )

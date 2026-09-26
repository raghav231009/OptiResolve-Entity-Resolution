#!/usr/bin/env python3
"""
Blocking Recall Evaluation Script.
Empirically measures and validates candidate link recall across various candidate caps K.
"""

import sys
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd
from tqdm import tqdm

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from business_entity_resolution.blocking import MultiIndexBlocker
from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.pipeline import (
    load_and_preprocess_file,
    load_ground_truth,
    load_targeted_training_targets,
)


def evaluate_blocking_recall(
    sample_size: int = 5000,
    k_values: List[int] = [20, 40, 60, 80, 100],
) -> Dict[int, float]:
    """Empirically evaluate link recall across multiple candidate safety caps K."""
    config = PipelineConfig()
    print(f"Loading {sample_size:,} S1 entities from {config.paths.train_source1}...")
    s1_records = load_and_preprocess_file(config.paths.train_source1, nrows=sample_size)
    s1_ids = {r["entity_id"] for r in s1_records}

    print(f"Loading ground truth for sample entities...")
    gt = load_ground_truth(config.paths.train_ground_truth, s1_ids)

    needed_targets = set()
    total_true_links = 0
    for s1_id, matches in gt.items():
        needed_targets.update(matches)
        total_true_links += len(matches)

    print(f"Total True Links to recover: {total_true_links:,} across {len(needed_targets):,} unique target records.")

    print("Loading target records from Source 2 and Source 3...")
    targets = load_targeted_training_targets(
        [config.paths.train_source2, config.paths.train_source3],
        needed_ids=needed_targets,
        background_sample_per_file=50000,
    )

    results = {}
    for k in k_values:
        print(f"\nEvaluating Blocker with safety cap K={k}...")
        blocker = MultiIndexBlocker(max_candidates=k)
        blocker.index_targets(targets)
        blocker.prune_large_blocks()

        recovered_links = 0
        for s1_rec in tqdm(s1_records, desc=f"Evaluating K={k}"):
            s1_id = s1_rec["entity_id"]
            true_matches = gt.get(s1_id, set())
            if not true_matches:
                continue

            candidates = set(blocker.retrieve_candidates(s1_rec))
            recovered_links += len(true_matches & candidates)

        recall = (recovered_links / total_true_links) if total_true_links > 0 else 0.0
        results[k] = recall
        print(f"Cap K={k:3d} -> Recovered: {recovered_links:,}/{total_true_links:,} -> Link Recall: {recall * 100:.2f}%")

    print("\n" + "=" * 50)
    print("FINAL BLOCKING RECALL SUMMARY:")
    print("=" * 50)
    for k, rec in results.items():
        print(f"  K = {k:3d}  |  Link Recall: {rec * 100:.2f}%")
    print("=" * 50)
    return results


if __name__ == "__main__":
    evaluate_blocking_recall()

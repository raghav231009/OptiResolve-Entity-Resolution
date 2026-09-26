#!/usr/bin/env python3
"""
Blocking Recall Evaluation Script.
Empirically measures and validates candidate link recall across various candidate caps K
using the EXACT production blocker configuration (max_block_size, min_token_len,
name_prefix_len) to ensure measurements represent real pipeline behavior.

Results are written to artifacts/blocking_recall_results.json for reproducibility.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd
from tqdm import tqdm

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from business_entity_resolution.blocking import MultiIndexBlocker
from business_entity_resolution.config import PipelineConfig, BlockingConfig
from business_entity_resolution.pipeline import (
    load_and_preprocess_file,
    load_ground_truth,
    load_targeted_training_targets,
)


def evaluate_blocking_recall(
    sample_size: int = 5000,
    k_values: List[int] = [20, 40, 60, 80, 100],
) -> Dict[int, float]:
    """
    Empirically evaluate link recall across multiple candidate safety caps K.
    Uses the EXACT production BlockingConfig parameters for all evaluations.
    """
    config = PipelineConfig()
    blocking_cfg = config.blocking  # Production blocking parameters

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
        background_sample_per_file=80000,
    )

    results = {}
    detailed_results = []

    for k in k_values:
        print(f"\nEvaluating Blocker with safety cap K={k} (production config: max_block_size={blocking_cfg.max_block_size})...")

        # Use EXACT production blocker parameters — only override max_candidates
        blocker = MultiIndexBlocker(
            max_candidates=k,
            min_token_len=blocking_cfg.min_token_len,
            name_prefix_len=blocking_cfg.name_prefix_len,
            max_block_size=blocking_cfg.max_block_size,
        )
        blocker.index_targets(targets)
        blocker.prune_large_blocks()

        recovered_links = 0
        entities_with_complete_recall = 0
        entities_evaluated = 0

        for s1_rec in tqdm(s1_records, desc=f"Evaluating K={k}"):
            s1_id = s1_rec["entity_id"]
            true_matches = gt.get(s1_id, set())
            if not true_matches:
                continue

            entities_evaluated += 1
            candidates = set(blocker.retrieve_candidates(s1_rec))
            hits = len(true_matches & candidates)
            recovered_links += hits
            if hits == len(true_matches):
                entities_with_complete_recall += 1

        recall = (recovered_links / total_true_links) if total_true_links > 0 else 0.0
        entity_recall = (entities_with_complete_recall / entities_evaluated) if entities_evaluated > 0 else 0.0
        results[k] = recall

        detail = {
            "K": k,
            "link_recall": round(recall, 6),
            "link_recall_pct": f"{recall * 100:.2f}%",
            "entity_complete_recall": round(entity_recall, 6),
            "recovered_links": recovered_links,
            "total_true_links": total_true_links,
            "entities_with_complete_recall": entities_with_complete_recall,
            "entities_evaluated": entities_evaluated,
        }
        detailed_results.append(detail)
        print(f"Cap K={k:3d} -> Recovered: {recovered_links:,}/{total_true_links:,} -> Link Recall: {recall * 100:.2f}% | Entity Complete Recall: {entity_recall * 100:.2f}%")

    print("\n" + "=" * 60)
    print("FINAL BLOCKING RECALL SUMMARY:")
    print(f"  Production max_block_size: {blocking_cfg.max_block_size}")
    print(f"  Production min_token_len:  {blocking_cfg.min_token_len}")
    print(f"  Production name_prefix_len: {blocking_cfg.name_prefix_len}")
    print(f"  Sample Size (S1 entities): {sample_size:,}")
    print(f"  Total Targets Loaded:      {len(targets):,}")
    print("=" * 60)
    for d in detailed_results:
        print(f"  K = {d['K']:3d}  |  Link Recall: {d['link_recall_pct']}  |  Entity Complete: {d['entity_complete_recall'] * 100:.2f}%")
    print("=" * 60)

    # Persist results for reproducibility (Issue #10)
    output_path = config.paths.artifacts_dir / "blocking_recall_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": sample_size,
        "total_targets_loaded": len(targets),
        "production_config": {
            "max_block_size": blocking_cfg.max_block_size,
            "min_token_len": blocking_cfg.min_token_len,
            "name_prefix_len": blocking_cfg.name_prefix_len,
        },
        "results": detailed_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults persisted to {output_path}")

    return results


if __name__ == "__main__":
    evaluate_blocking_recall()

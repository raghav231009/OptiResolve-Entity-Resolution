#!/usr/bin/env python3
"""
Empirical Blocking Recall Evaluation Engine.
Evaluates candidate caps K in [20, 40, 60, 80, 100, 150] on actual training ground truth
using the EXACT production blocker configuration (6 channels, sub-blocking, priority tier retention).

Calculates for each K:
- link recall
- S1 entity recall (complete entity recall)
- candidate count (total candidates generated across evaluated entities)
- average candidates/entity
- P95 candidates/entity
- P99 candidates/entity

Results are written to blocking_recall_results.json and artifacts/blocking_recall_results.json.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set
import numpy as np
from tqdm import tqdm

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from business_entity_resolution.blocking import MultiIndexBlocker
from business_entity_resolution.config import PipelineConfig, BlockingConfig, PROJECT_ROOT
from business_entity_resolution.pipeline import (
    load_and_preprocess_file,
    load_ground_truth,
    load_targeted_training_targets,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_blocking_recall(
    sample_size: int = 5000,
    k_values: List[int] = [20, 40, 60, 80, 100, 150],
    background_sample_per_file: int = 100000,
    output_file: Path = None,
) -> Dict[str, Any]:
    """
    Empirically evaluate blocking recall across candidate safety caps K on actual ground truth.
    Uses the EXACT production blocker with all 6 channels and deterministic sub-blocking.
    """
    config = PipelineConfig()
    blocking_cfg = config.blocking

    logger.info(f"Loading {sample_size:,} S1 entities from {config.paths.train_source1}...")
    s1_records = load_and_preprocess_file(config.paths.train_source1, nrows=sample_size)
    s1_ids = {r["entity_id"] for r in s1_records}

    logger.info("Loading ground truth for sample entities...")
    gt = load_ground_truth(config.paths.train_ground_truth, s1_ids)

    needed_targets: Set[str] = set()
    total_true_links = 0
    matched_s1_records = []

    for s1_rec in s1_records:
        s1_id = s1_rec["entity_id"]
        matches = gt.get(s1_id, set())
        if matches:
            needed_targets.update(matches)
            total_true_links += len(matches)
            matched_s1_records.append(s1_rec)

    total_evaluated_entities = len(matched_s1_records)
    logger.info(
        f"Evaluation scope: {total_evaluated_entities:,} matched S1 entities ({len(s1_records) - total_evaluated_entities:,} singletons), "
        f"{total_true_links:,} true links to recover across {len(needed_targets):,} unique target entities."
    )

    logger.info("Loading target records from Source 2 and Source 3 with background distractors...")
    targets = load_targeted_training_targets(
        [config.paths.train_source2, config.paths.train_source3],
        needed_ids=needed_targets,
        background_sample_per_file=background_sample_per_file,
    )
    logger.info(f"Target pool loaded: {len(targets):,} total targets.")

    # Instantiate production blocker with max K to build indices once
    max_k = max(k_values)
    logger.info(
        f"Building production 6-channel blocker index (max_block_size={blocking_cfg.max_block_size}, "
        f"sub_block_threshold={blocking_cfg.sub_block_threshold}, min_token_len={blocking_cfg.min_token_len}, "
        f"name_prefix_len={blocking_cfg.name_prefix_len})..."
    )
    blocker = MultiIndexBlocker(
        max_candidates=max_k,
        min_token_len=blocking_cfg.min_token_len,
        name_prefix_len=blocking_cfg.name_prefix_len,
        max_block_size=blocking_cfg.max_block_size,
        sub_block_threshold=blocking_cfg.sub_block_threshold,
    )
    blocker.index_targets(targets)
    blocker.prune_large_blocks()

    detailed_results = []

    for k in k_values:
        logger.info(f"\n--- Evaluating Candidate Cap K={k} ---")
        blocker.max_candidates = k
        blocker.max_candidates_per_entity = k
        blocker.retrieval_stats = {
            "total_queries": 0,
            "total_candidates_before_cap": 0,
            "total_candidates_after_cap": 0,
            "max_candidates_before_cap": 0,
            "max_candidates_after_cap": 0,
        }

        recovered_links = 0
        entities_with_complete_recall = 0
        cand_counts_per_entity = []

        for s1_rec in tqdm(matched_s1_records, desc=f"Evaluating K={k}"):
            s1_id = s1_rec["entity_id"]
            true_matches = gt.get(s1_id, set())

            candidates = blocker.retrieve_candidates(s1_rec)
            cand_len = len(candidates)
            cand_counts_per_entity.append(cand_len)

            cand_set = set(candidates)
            hits = len(true_matches & cand_set)
            recovered_links += hits
            if hits == len(true_matches):
                entities_with_complete_recall += 1

        link_recall = (recovered_links / total_true_links) if total_true_links > 0 else 0.0
        entity_recall = (entities_with_complete_recall / total_evaluated_entities) if total_evaluated_entities > 0 else 0.0
        total_candidates = int(np.sum(cand_counts_per_entity))
        avg_candidates = float(np.mean(cand_counts_per_entity)) if cand_counts_per_entity else 0.0
        p95_candidates = float(np.percentile(cand_counts_per_entity, 95)) if cand_counts_per_entity else 0.0
        p99_candidates = float(np.percentile(cand_counts_per_entity, 99)) if cand_counts_per_entity else 0.0

        detail = {
            "K": k,
            "link_recall": round(link_recall, 6),
            "link_recall_pct": f"{link_recall * 100:.2f}%",
            "entity_recall": round(entity_recall, 6),
            "entity_recall_pct": f"{entity_recall * 100:.2f}%",
            "recovered_links": recovered_links,
            "total_true_links": total_true_links,
            "entities_with_complete_recall": entities_with_complete_recall,
            "entities_evaluated": total_evaluated_entities,
            "candidate_count": total_candidates,
            "average_candidates_per_entity": round(avg_candidates, 2),
            "p95_candidates_per_entity": round(p95_candidates, 2),
            "p99_candidates_per_entity": round(p99_candidates, 2),
        }
        detailed_results.append(detail)
        logger.info(
            f"K={k:3d} | Link Recall: {detail['link_recall_pct']} | S1 Entity Recall: {detail['entity_recall_pct']} | "
            f"Avg Cands: {detail['average_candidates_per_entity']} | P95: {detail['p95_candidates_per_entity']} | P99: {detail['p99_candidates_per_entity']}"
        )

    # Print summary table
    print("\n" + "=" * 92)
    print(f"{'K':<5} | {'Link Recall':<12} | {'Entity Recall':<14} | {'Total Cands':<12} | {'Avg/Entity':<11} | {'P95':<8} | {'P99':<8}")
    print("-" * 92)
    for d in detailed_results:
        print(
            f"{d['K']:<5} | {d['link_recall_pct']:<12} | {d['entity_recall_pct']:<14} | "
            f"{d['candidate_count']:<12,d} | {d['average_candidates_per_entity']:<11.2f} | "
            f"{d['p95_candidates_per_entity']:<8.1f} | {d['p99_candidates_per_entity']:<8.1f}"
        )
    print("=" * 92 + "\n")

    output_data = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": sample_size,
        "matched_entities_evaluated": total_evaluated_entities,
        "total_true_links": total_true_links,
        "total_targets_loaded": len(targets),
        "production_config": {
            "max_block_size": blocking_cfg.max_block_size,
            "sub_block_threshold": blocking_cfg.sub_block_threshold,
            "min_token_len": blocking_cfg.min_token_len,
            "name_prefix_len": blocking_cfg.name_prefix_len,
        },
        "results": detailed_results,
    }

    # Save to artifacts/blocking_recall_results.json and root blocking_recall_results.json
    paths_to_save = [
        config.paths.artifacts_dir / "blocking_recall_results.json",
        PROJECT_ROOT / "blocking_recall_results.json",
    ]
    if output_file:
        paths_to_save.append(Path(output_file))

    for p in paths_to_save:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to: {p}")

    return output_data


def main():
    parser = argparse.ArgumentParser(description="Empirical Blocking Recall Evaluation")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of S1 records to sample for evaluation")
    parser.add_argument("--k-values", nargs="+", type=int, default=[20, 40, 60, 80, 100, 150], help="List of K values to evaluate")
    parser.add_argument("--background-sample", type=int, default=100000, help="Background targets per target file")
    parser.add_argument("--output", type=str, default=None, help="Optional output path")
    args = parser.parse_args()

    evaluate_blocking_recall(
        sample_size=args.sample_size,
        k_values=args.k_values,
        background_sample_per_file=args.background_sample,
        output_file=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()

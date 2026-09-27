#!/usr/bin/env python3
"""
Candidate Capping & Ranking Strategy Audit for OptiResolve.

Measures and compares:
  1. Arbitrary top-K
  2. Legacy current ranking
  3. Audited Tiered ranking (Priority Tier 1 + Deterministic Cap + Balanced Address-Aware Tier 2)

Evaluates:
  - Link Recall across K in [20, 40, 60, 80, 100, 150]
  - Complete Entity Recall
  - Exact count of ground-truth matches lost due to candidate capping
  - Statistical comparison before vs after
"""

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Set
import numpy as np
from tqdm import tqdm

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from business_entity_resolution.blocking import MultiIndexBlocker
from business_entity_resolution.config import PipelineConfig, PROJECT_ROOT
from business_entity_resolution.pipeline import (
    load_and_preprocess_file,
    load_ground_truth,
    load_targeted_training_targets,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_capping_audit(
    sample_size: int = 5000,
    k_values: List[int] = [20, 40, 60, 80, 100, 150],
    strategies: List[str] = ["arbitrary", "current", "tiered"],
    background_sample_per_file: int = 100000,
    output_dir: Path = None,
) -> Dict[str, Any]:
    config = PipelineConfig()
    blocking_cfg = config.blocking
    if output_dir is None:
        output_dir = config.paths.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("============================================================")
    logger.info("OPTISOLVE CANDIDATE CAPPING & RANKING AUDIT")
    logger.info(f"Sample Size: {sample_size:,} S1 entities")
    logger.info(f"K Values:    {k_values}")
    logger.info(f"Strategies:  {strategies}")
    logger.info("============================================================")

    # 1. Load S1 sample & ground truth
    s1_records = load_and_preprocess_file(config.paths.train_source1, nrows=sample_size)
    s1_ids = {r["entity_id"] for r in s1_records}
    gt = load_ground_truth(config.paths.train_ground_truth, s1_ids)

    needed_targets: Set[str] = set()
    total_true_links = 0
    matched_s1_records: List[dict] = []

    for s1_rec in s1_records:
        s1_id = s1_rec["entity_id"]
        matches = gt.get(s1_id, set())
        if matches:
            needed_targets.update(matches)
            total_true_links += len(matches)
            matched_s1_records.append(s1_rec)

    total_matched_entities = len(matched_s1_records)
    logger.info(
        f"Matched S1 Entities: {total_matched_entities:,} | "
        f"True Ground-Truth Links to Recover: {total_true_links:,}"
    )

    # 2. Load target pool
    targets = load_targeted_training_targets(
        [config.paths.train_source2, config.paths.train_source3],
        needed_ids=needed_targets,
        background_sample_per_file=background_sample_per_file,
    )
    logger.info(f"Target Pool Loaded: {len(targets):,} records.")

    # 3. Build unified inverted index once
    blocker = MultiIndexBlocker(
        max_candidates=max(k_values),
        min_token_len=blocking_cfg.min_token_len,
        name_prefix_len=blocking_cfg.name_prefix_len,
        max_block_size=blocking_cfg.max_block_size,
        sub_block_threshold=blocking_cfg.sub_block_threshold,
    )
    blocker.index_targets(targets)
    blocker.prune_large_blocks()

    audit_results: Dict[str, Dict[int, Dict[str, Any]]] = {strat: {} for strat in strategies}
    comparison_rows = []

    for strat in strategies:
        logger.info(f"\n============================================================")
        logger.info(f"EVALUATING CAPPING STRATEGY: '{strat.upper()}'")
        logger.info(f"============================================================")
        blocker.capping_strategy = strat

        for k in k_values:
            blocker.max_candidates = k
            blocker.max_candidates_per_entity = k
            blocker.retrieval_stats = {
                "total_queries": 0,
                "total_candidates_before_cap": 0,
                "total_candidates_after_cap": 0,
                "max_candidates_before_cap": 0,
                "max_candidates_after_cap": 0,
                "capping_events": 0,
                "tier1_retained": 0,
                "tier2_retained": 0,
                "tier1_overflow_events": 0,
            }

            recovered_links = 0
            complete_recall_entities = 0
            lost_to_capping = 0  # Present in uncapped set, but dropped after cap
            uncapped_total_hits = 0

            for s1_rec in tqdm(matched_s1_records, desc=f"Strategy: {strat} | K={k}"):
                s1_id = s1_rec["entity_id"]
                true_matches = gt.get(s1_id, set())
                n_true = len(true_matches)

                cands, uncapped = blocker.retrieve_candidates_with_uncapped(s1_rec)
                cand_set = set(cands)

                hits = len(true_matches & cand_set)
                recovered_links += hits

                uncapped_hits = len(true_matches & uncapped)
                uncapped_total_hits += uncapped_hits

                # True matches that were retrieved across the 6 channels, but dropped by the cap
                lost = len(true_matches & uncapped) - hits
                lost_to_capping += max(0, lost)

                if hits == n_true:
                    complete_recall_entities += 1

            link_recall = recovered_links / total_true_links if total_true_links > 0 else 0.0
            entity_recall = complete_recall_entities / total_matched_entities if total_matched_entities > 0 else 0.0
            uncapped_recall = uncapped_total_hits / total_true_links if total_true_links > 0 else 0.0

            stats = blocker.get_retrieval_statistics()

            strat_k_metrics = {
                "strategy": strat,
                "k": k,
                "total_true_links": total_true_links,
                "recovered_links": recovered_links,
                "link_recall": round(link_recall, 4),
                "link_recall_pct": round(link_recall * 100, 2),
                "complete_recall_entities": complete_recall_entities,
                "entity_recall": round(entity_recall, 4),
                "entity_recall_pct": round(entity_recall * 100, 2),
                "uncapped_recall": round(uncapped_recall, 4),
                "lost_to_capping": lost_to_capping,
                "capping_loss_pct": round((lost_to_capping / total_true_links) * 100, 2) if total_true_links > 0 else 0.0,
                "capping_events": stats.get("capping_events", 0),
                "tier1_retained": stats.get("tier1_retained", 0),
                "tier2_retained": stats.get("tier2_retained", 0),
                "tier1_overflow_events": stats.get("tier1_overflow_events", 0),
                "avg_candidates_after_cap": stats.get("avg_candidates_after_cap", 0.0),
            }

            audit_results[strat][k] = strat_k_metrics
            comparison_rows.append(strat_k_metrics)

            logger.info(
                f"[{strat.upper()}] K={k:3d} -> Link Recall: {link_recall * 100:6.2f}% | "
                f"Entity Recall: {entity_recall * 100:6.2f}% | "
                f"Lost to Capping: {lost_to_capping:4d} ({strat_k_metrics['capping_loss_pct']:.2f}%)"
            )

    # 4. Summary Table Comparison
    logger.info("\n" + "=" * 80)
    logger.info("CANDIDATE CAPPING STRATEGY COMPARISON SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'K':>4} | {'Arbitrary Recall':>17} | {'Legacy Recall':>14} | {'Tiered Recall':>14} | {'Recall Gain':>12} | {'Links Saved':>12}")
    logger.info("-" * 80)

    summary_comparison = []
    for k in k_values:
        arb_rec = audit_results.get("arbitrary", {}).get(k, {}).get("link_recall_pct", 0.0)
        cur_rec = audit_results.get("current", {}).get(k, {}).get("link_recall_pct", 0.0)
        tier_rec = audit_results.get("tiered", {}).get(k, {}).get("link_recall_pct", 0.0)

        cur_lost = audit_results.get("current", {}).get(k, {}).get("lost_to_capping", 0)
        tier_lost = audit_results.get("tiered", {}).get(k, {}).get("lost_to_capping", 0)

        links_saved = cur_lost - tier_lost
        gain_pct = round(tier_rec - cur_rec, 2)

        summary_comparison.append({
            "k": k,
            "arbitrary_link_recall_pct": arb_rec,
            "legacy_link_recall_pct": cur_rec,
            "tiered_link_recall_pct": tier_rec,
            "recall_gain_pct": gain_pct,
            "links_saved_vs_legacy": links_saved,
            "legacy_lost_to_capping": cur_lost,
            "tiered_lost_to_capping": tier_lost,
        })

        logger.info(
            f"{k:4d} | {arb_rec:16.2f}% | {cur_rec:13.2f}% | {tier_rec:13.2f}% | "
            f"{gain_pct:+11.2f}% | {links_saved:+11d}"
        )
    logger.info("=" * 80)

    # 5. Export JSON & CSV Artifacts
    full_output = {
        "metadata": {
            "sample_size": sample_size,
            "total_matched_entities": total_matched_entities,
            "total_true_links": total_true_links,
            "k_values": k_values,
            "strategies": strategies,
        },
        "summary_comparison": summary_comparison,
        "detailed_results_by_strategy": audit_results,
    }

    json_path = output_dir / "candidate_capping_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    logger.info(f"Saved audit JSON to: {json_path}")

    csv_path = output_dir / "candidate_capping_comparison.csv"
    if comparison_rows:
        fieldnames = list(comparison_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_rows)
        logger.info(f"Saved comparison CSV to: {csv_path}")

    return full_output


def main():
    parser = argparse.ArgumentParser(description="Candidate Capping and Ranking Strategy Audit")
    parser.add_argument("--sample-size", type=int, default=5000, help="S1 sample size")
    parser.add_argument("--k-values", nargs="+", type=int, default=[20, 40, 60, 80, 100, 150])
    parser.add_argument("--strategies", nargs="+", type=str, default=["arbitrary", "current", "tiered"])
    parser.add_argument("--background-sample", type=int, default=100000)
    args = parser.parse_args()

    run_capping_audit(
        sample_size=args.sample_size,
        k_values=args.k_values,
        strategies=args.strategies,
        background_sample_per_file=args.background_sample,
    )


if __name__ == "__main__":
    main()

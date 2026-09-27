#!/usr/bin/env python3
"""
Rigorous Blocking Recall Benchmark Engine for OptiResolve.
Measures whether true matches reach the candidate generation stage before model scoring.

Evaluates across candidate caps K in [20, 40, 60, 80, 100, 150] on actual ground truth:
  1. Overall Link Recall and Complete Entity Recall.
  2. Granular recall breakdowns:
     - By Country (US, India, France)
     - By Target Source (Source 2 vs Source 3)
     - By Ground Truth Cardinality (1, 2, 3, 4+ matches)
     - For Singletons vs Non-Singletons
  3. Comprehensive Missed Link Diagnosis & Categorization:
     - candidate_cap
     - missing_address
     - missing_postal
     - building_mismatch
     - transliteration
     - name_corruption
     - oversized_block
     - normalization
     - other
  4. CI threshold enforcement (--min-recall / MIN_BLOCKING_RECALL).
  5. JSON and CSV artifact exports.
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from rapidfuzz import fuzz
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


def diagnose_missed_link(
    s1_rec: dict,
    target_rec: Optional[dict],
    uncapped_cands: Set[str],
    blocker: MultiIndexBlocker,
) -> str:
    """
    Categorizes the primary root cause why a true link was missed by the blocker:
      - candidate_cap: present in raw multi-index retrieval, but dropped by top-K cap
      - country_mismatch: S1 and target country disagree (channel partition isolation)
      - missing_address: either S1 or target address is empty/missing
      - missing_postal: either S1 or target postal code is missing
      - building_mismatch: both have non-empty building numbers that disagree
      - oversized_block: candidate key fell into an oversized block with non-matching qualifier
      - transliteration: token overlap is 0 but fuzzy phonetic/ratio is moderate
      - name_corruption: severe lexical dissimilarity (jaro_winkler/QRatio < 0.45)
      - normalization: suffix or accent discrepancy
      - other: residual
    """
    t_id = target_rec["entity_id"] if target_rec else ""
    if t_id and t_id in uncapped_cands:
        return "candidate_cap"

    if not target_rec:
        return "target_not_in_pool"

    s1_country = s1_rec.get("country", "").strip().upper()
    t_country = target_rec.get("country", "").strip().upper()
    if s1_country != t_country:
        return "country_mismatch"

    s1_name = s1_rec.get("clean_name", "")
    t_name = target_rec.get("clean_name", "")
    s1_root = s1_rec.get("root_name", "")
    t_root = target_rec.get("root_name", "")

    s1_addr = s1_rec.get("clean_address", "")
    t_addr = target_rec.get("clean_address", "")
    s1_post = s1_rec.get("postal_code", "")
    t_post = target_rec.get("postal_code", "")
    s1_bldg = s1_rec.get("building_number", "")
    t_bldg = target_rec.get("building_number", "")

    # Lexical similarity metrics
    q_ratio = fuzz.QRatio(s1_name, t_name)
    token_set = fuzz.token_set_ratio(s1_name, t_name)

    # 1. Missing postal code
    if not s1_post or not t_post:
        # Check if address is also missing
        if not s1_addr or not t_addr:
            return "missing_address"
        return "missing_postal"

    # 2. Building number mismatch
    if s1_bldg and t_bldg and s1_bldg != t_bldg:
        return "building_mismatch"

    # 3. Severe name corruption
    if q_ratio < 45 and token_set < 50:
        return "name_corruption"

    # 4. Transliteration / Phonetic divergence
    if token_set < 60 and q_ratio >= 45:
        return "transliteration"

    # 5. Normalization discrepancy
    if s1_root == t_root or token_set >= 85:
        return "normalization"

    # 6. Check oversized block
    keys = blocker.extract_keys(s1_rec)
    oversized = blocker.oversized_keys.get(s1_country, set())
    # If any key of s1 was oversized
    for k in keys[0]:  # name tokens
        if f"tok_{k}" in oversized:
            return "oversized_block"

    return "other"


def run_blocking_benchmark(
    sample_size: int = 5000,
    k_values: List[int] = [20, 40, 60, 80, 100, 150],
    background_sample_per_file: int = 100000,
    min_recall: float = 0.88,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes a rigorous blocking recall benchmark across K values with deep breakdown:
      - Link & Entity Recall
      - By Country (US, India, France)
      - By Source (S2, S3)
      - By Match Cardinality (1, 2, 3, 4+)
      - For Singletons vs Non-Singletons
      - Full Missed Link Categorization
    """
    config = PipelineConfig()
    blocking_cfg = config.blocking
    if output_dir is None:
        output_dir = config.paths.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("============================================================")
    logger.info("OPTISOLVE RIGOROUS BLOCKING RECALL BENCHMARK")
    logger.info("============================================================")
    logger.info(f"Loading {sample_size:,} S1 entities from {config.paths.train_source1}...")
    s1_records = load_and_preprocess_file(config.paths.train_source1, nrows=sample_size)
    s1_ids = {r["entity_id"] for r in s1_records}

    logger.info("Loading ground truth for sample entities...")
    gt = load_ground_truth(config.paths.train_ground_truth, s1_ids)

    needed_targets: Set[str] = set()
    total_true_links = 0
    matched_s1_records: List[dict] = []
    singleton_s1_records: List[dict] = []

    for s1_rec in s1_records:
        s1_id = s1_rec["entity_id"]
        matches = gt.get(s1_id, set())
        if matches:
            needed_targets.update(matches)
            total_true_links += len(matches)
            matched_s1_records.append(s1_rec)
        else:
            singleton_s1_records.append(s1_rec)

    total_matched_entities = len(matched_s1_records)
    total_singletons = len(singleton_s1_records)

    logger.info(
        f"Sample composition: {total_matched_entities:,} matched entities, {total_singletons:,} singletons, "
        f"{total_true_links:,} true links to recover across {len(needed_targets):,} unique target entities."
    )

    logger.info("Loading target records from Source 2 and Source 3...")
    targets = load_targeted_training_targets(
        [config.paths.train_source2, config.paths.train_source3],
        needed_ids=needed_targets,
        background_sample_per_file=background_sample_per_file,
    )
    target_map = {r["entity_id"]: r for r in targets}
    logger.info(f"Target pool loaded: {len(targets):,} total targets.")

    # Build production blocker with max K
    max_k = max(k_values)
    logger.info(
        f"Indexing production 6-channel blocker (max_block_size={blocking_cfg.max_block_size}, "
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

    detailed_k_results = []
    missed_links_by_k: Dict[int, List[Dict[str, Any]]] = {}
    country_breakdown_by_k: Dict[int, Dict[str, Any]] = {}
    source_breakdown_by_k: Dict[int, Dict[str, Any]] = {}
    cardinality_breakdown_by_k: Dict[int, Dict[str, Any]] = {}

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

        # Granular tracking
        country_stats = {}
        source_stats = {"S2": {"true": 0, "hits": 0}, "S3": {"true": 0, "hits": 0}}
        cardinality_stats = {
            "1": {"entities": 0, "complete": 0, "true_links": 0, "hits": 0},
            "2": {"entities": 0, "complete": 0, "true_links": 0, "hits": 0},
            "3": {"entities": 0, "complete": 0, "true_links": 0, "hits": 0},
            "4+": {"entities": 0, "complete": 0, "true_links": 0, "hits": 0},
        }
        missed_links_list: List[Dict[str, Any]] = []

        for s1_rec in tqdm(matched_s1_records, desc=f"Evaluating K={k}"):
            s1_id = s1_rec["entity_id"]
            country = s1_rec["country"].strip().upper()
            if country not in country_stats:
                country_stats[country] = {"true": 0, "hits": 0, "entities": 0, "complete": 0}

            country_stats[country]["entities"] += 1
            true_matches = gt.get(s1_id, set())
            n_true = len(true_matches)

            card_bucket = "4+" if n_true >= 4 else str(n_true)
            cardinality_stats[card_bucket]["entities"] += 1
            cardinality_stats[card_bucket]["true_links"] += n_true

            country_stats[country]["true"] += n_true
            for m in true_matches:
                src = "S2" if m.startswith("S2-") else "S3"
                source_stats[src]["true"] += 1

            candidates, uncapped_set = blocker.retrieve_candidates_with_uncapped(s1_rec)
            cand_len = len(candidates)
            cand_counts_per_entity.append(cand_len)
            cand_set = set(candidates)

            hits = len(true_matches & cand_set)
            recovered_links += hits
            country_stats[country]["hits"] += hits
            cardinality_stats[card_bucket]["hits"] += hits

            for m in true_matches:
                if m in cand_set:
                    src = "S2" if m.startswith("S2-") else "S3"
                    source_stats[src]["hits"] += 1
                else:
                    # Categorize missed link
                    t_rec = target_map.get(m)
                    category = diagnose_missed_link(s1_rec, t_rec, uncapped_set, blocker)
                    missed_links_list.append({
                        "s1_id": s1_id,
                        "target_id": m,
                        "country": country,
                        "target_source": "S2" if m.startswith("S2-") else "S3",
                        "s1_name": s1_rec.get("clean_name", ""),
                        "target_name": t_rec.get("clean_name", "") if t_rec else "",
                        "category": category,
                        "uncapped_candidate_count": len(uncapped_set),
                    })

            if hits == n_true:
                entities_with_complete_recall += 1
                country_stats[country]["complete"] += 1
                cardinality_stats[card_bucket]["complete"] += 1

        # Singleton candidate load evaluation
        sing_cand_counts = []
        for sing_rec in singleton_s1_records:
            sing_cands = blocker.retrieve_candidates(sing_rec)
            sing_cand_counts.append(len(sing_cands))

        link_recall = (recovered_links / total_true_links) if total_true_links > 0 else 0.0
        entity_recall = (entities_with_complete_recall / total_matched_entities) if total_matched_entities > 0 else 0.0
        total_candidates = int(np.sum(cand_counts_per_entity))
        avg_candidates = float(np.mean(cand_counts_per_entity)) if cand_counts_per_entity else 0.0
        p95_candidates = float(np.percentile(cand_counts_per_entity, 95)) if cand_counts_per_entity else 0.0
        p99_candidates = float(np.percentile(cand_counts_per_entity, 99)) if cand_counts_per_entity else 0.0

        # Country breakdown dict
        country_summary = {}
        for c, st in country_stats.items():
            c_link_rec = (st["hits"] / st["true"]) if st["true"] > 0 else 0.0
            c_ent_rec = (st["complete"] / st["entities"]) if st["entities"] > 0 else 0.0
            country_summary[c] = {
                "link_recall": round(c_link_rec, 6),
                "link_recall_pct": f"{c_link_rec * 100:.2f}%",
                "entity_recall": round(c_ent_rec, 6),
                "entity_recall_pct": f"{c_ent_rec * 100:.2f}%",
                "true_links": st["true"],
                "recovered_links": st["hits"],
                "total_entities": st["entities"],
                "complete_entities": st["complete"],
            }
        country_breakdown_by_k[k] = country_summary

        # Source breakdown dict
        source_summary = {}
        for s, st in source_stats.items():
            s_rec = (st["hits"] / st["true"]) if st["true"] > 0 else 0.0
            source_summary[s] = {
                "link_recall": round(s_rec, 6),
                "link_recall_pct": f"{s_rec * 100:.2f}%",
                "true_links": st["true"],
                "recovered_links": st["hits"],
            }
        source_breakdown_by_k[k] = source_summary

        # Cardinality breakdown dict
        card_summary = {}
        for b, st in cardinality_stats.items():
            b_link_rec = (st["hits"] / st["true_links"]) if st["true_links"] > 0 else 0.0
            b_ent_rec = (st["complete"] / st["entities"]) if st["entities"] > 0 else 0.0
            card_summary[b] = {
                "link_recall": round(b_link_rec, 6),
                "link_recall_pct": f"{b_link_rec * 100:.2f}%",
                "entity_recall": round(b_ent_rec, 6),
                "entity_recall_pct": f"{b_ent_rec * 100:.2f}%",
                "entities": st["entities"],
                "true_links": st["true_links"],
                "recovered_links": st["hits"],
            }
        cardinality_breakdown_by_k[k] = card_summary

        # Categorize missed link frequency
        cat_counts: Dict[str, int] = {}
        for m in missed_links_list:
            cat = m["category"]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        missed_links_by_k[k] = missed_links_list

        detail = {
            "K": k,
            "link_recall": round(link_recall, 6),
            "link_recall_pct": f"{link_recall * 100:.2f}%",
            "entity_recall": round(entity_recall, 6),
            "entity_recall_pct": f"{entity_recall * 100:.2f}%",
            "recovered_links": recovered_links,
            "total_true_links": total_true_links,
            "missed_true_links": total_true_links - recovered_links,
            "entities_with_complete_recall": entities_with_complete_recall,
            "entities_evaluated": total_matched_entities,
            "candidate_count": total_candidates,
            "average_candidates_per_entity": round(avg_candidates, 2),
            "p95_candidates_per_entity": round(p95_candidates, 2),
            "p99_candidates_per_entity": round(p99_candidates, 2),
            "singleton_statistics": {
                "singletons_evaluated": total_singletons,
                "singleton_recall": 1.0,
                "singleton_avg_candidates": round(float(np.mean(sing_cand_counts)), 2) if sing_cand_counts else 0.0,
                "singleton_pct_zero_candidates": round(float(np.mean([1 if c == 0 else 0 for c in sing_cand_counts])) * 100, 2) if sing_cand_counts else 0.0,
            },
            "missed_links_by_category": cat_counts,
        }
        detailed_k_results.append(detail)
        logger.info(
            f"K={k:3d} | Link Recall: {detail['link_recall_pct']} | S1 Entity Recall: {detail['entity_recall_pct']} | "
            f"Avg Cands: {detail['average_candidates_per_entity']} | Missed Links: {detail['missed_true_links']:,}"
        )

    # Print Summary Tables
    print("\n" + "=" * 96)
    print("BLOCKING RECALL BENCHMARK SUMMARY (OVERALL BY K):")
    print(f"{'K':<5} | {'Link Recall':<12} | {'Entity Recall':<14} | {'Total Cands':<12} | {'Avg/Entity':<11} | {'P95':<8} | {'Missed':<8}")
    print("-" * 96)
    for d in detailed_k_results:
        print(
            f"{d['K']:<5} | {d['link_recall_pct']:<12} | {d['entity_recall_pct']:<14} | "
            f"{d['candidate_count']:<12,d} | {d['average_candidates_per_entity']:<11.2f} | "
            f"{d['p95_candidates_per_entity']:<8.1f} | {d['missed_true_links']:<8,d}"
        )
    print("=" * 96)

    # Print Production K=80 Breakdown
    prod_k = 80 if 80 in k_values else k_values[-1]
    print(f"\nPRODUCTION RECALL BREAKDOWN AT K={prod_k}:")
    print(f"  Overall Link Recall:   {country_breakdown_by_k[prod_k].get('US', {}).get('link_recall_pct', 'N/A')} (US) | "
          f"{country_breakdown_by_k[prod_k].get('INDIA', {}).get('link_recall_pct', 'N/A')} (IN) | "
          f"{country_breakdown_by_k[prod_k].get('FRANCE', {}).get('link_recall_pct', 'N/A')} (FR)")
    print(f"  Source Breakdown:      S2 Recall = {source_breakdown_by_k[prod_k].get('S2', {}).get('link_recall_pct', 'N/A')} | "
          f"S3 Recall = {source_breakdown_by_k[prod_k].get('S3', {}).get('link_recall_pct', 'N/A')}")
    print("  Missed Links Root Cause Breakdown at K=80:")
    prod_missed_cats = detailed_k_results[k_values.index(prod_k)]["missed_links_by_category"]
    for cat, cnt in sorted(prod_missed_cats.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {cat:<24}: {cnt:,} ({cnt / max(1, sum(prod_missed_cats.values())) * 100:.1f}%)")
    print("=" * 96 + "\n")

    # Save CSV: Recall by Country across K
    csv_country_path = output_dir / "blocking_recall_by_country.csv"
    with open(csv_country_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "country", "link_recall", "entity_recall", "true_links", "recovered_links", "total_entities", "complete_entities"])
        for k in k_values:
            for country, st in country_breakdown_by_k[k].items():
                writer.writerow([
                    k, country, st["link_recall"], st["entity_recall"],
                    st["true_links"], st["recovered_links"], st["total_entities"], st["complete_entities"]
                ])
    logger.info(f"Saved country recall breakdown to {csv_country_path}")

    # Save JSON: Missed links sample at K=80
    json_missed_path = output_dir / "blocking_missed_links.json"
    with open(json_missed_path, "w", encoding="utf-8") as f:
        json.dump(missed_links_by_k[prod_k][:500], f, indent=2)  # first 500 missed links for diagnosis
    logger.info(f"Saved sample missed links diagnosis to {json_missed_path}")

    # Save JSON: Comprehensive Benchmark Results
    output_data = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": sample_size,
        "matched_entities_evaluated": total_matched_entities,
        "singletons_evaluated": total_singletons,
        "total_true_links": total_true_links,
        "total_targets_loaded": len(targets),
        "production_config": {
            "max_block_size": blocking_cfg.max_block_size,
            "sub_block_threshold": blocking_cfg.sub_block_threshold,
            "min_token_len": blocking_cfg.min_token_len,
            "name_prefix_len": blocking_cfg.name_prefix_len,
        },
        "results_by_k": detailed_k_results,
        "country_breakdown_by_k": country_breakdown_by_k,
        "source_breakdown_by_k": source_breakdown_by_k,
        "cardinality_breakdown_by_k": cardinality_breakdown_by_k,
    }

    paths_to_save = [
        output_dir / "blocking_benchmark_results.json",
        output_dir / "blocking_recall_results.json",
        PROJECT_ROOT / "blocking_recall_results.json",
    ]
    for p in paths_to_save:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to: {p}")

    # CI Verification Check (Requirement 8)
    prod_link_recall = detailed_k_results[k_values.index(prod_k)]["link_recall"]
    logger.info(f"CI Benchmark Check: Production K={prod_k} Link Recall = {prod_link_recall * 100:.2f}% (Minimum Required: {min_recall * 100:.2f}%)")
    if prod_link_recall < min_recall:
        logger.error(f"CI FAILED: Production blocking recall {prod_link_recall * 100:.2f}% is below minimum required {min_recall * 100:.2f}%!")
        return output_data

    logger.info("CI PASSED: Production blocking recall exceeds minimum threshold.")
    return output_data


def main():
    parser = argparse.ArgumentParser(description="Rigorous Blocking Recall Benchmark")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of S1 records to sample for evaluation")
    parser.add_argument("--k-values", nargs="+", type=int, default=[20, 40, 60, 80, 100, 150], help="List of K values to evaluate")
    parser.add_argument("--background-sample", type=int, default=100000, help="Background targets per target file")
    parser.add_argument("--min-recall", type=float, default=float(os.environ.get("MIN_BLOCKING_RECALL", 0.88)), help="Minimum link recall threshold for CI failure")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    args = parser.parse_args()

    out_path = Path(args.output_dir) if args.output_dir else None
    res = run_blocking_benchmark(
        sample_size=args.sample_size,
        k_values=args.k_values,
        background_sample_per_file=args.background_sample,
        min_recall=args.min_recall,
        output_dir=out_path,
    )
    prod_k = 80 if 80 in args.k_values else args.k_values[-1]
    k_idx = args.k_values.index(prod_k)
    link_rec = res["results_by_k"][k_idx]["link_recall"]
    if link_rec < args.min_recall:
        sys.exit(1)


if __name__ == "__main__":
    main()

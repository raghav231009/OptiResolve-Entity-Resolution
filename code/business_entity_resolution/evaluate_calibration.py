#!/usr/bin/env python3
"""
Threshold and Probability Calibration Audit Engine for OptiResolve.
Evaluates:
  1. Probability distributions for TP, Hard Negatives, Singleton Negatives, S2, and S3.
  2. Brier score and Expected Calibration Error (ECE) with reliability diagrams.
  3. Threshold sensitivity and local Lipschitz stability around the optimum.
  4. Honest 2-fold cross-split evaluation of Platt (Sigmoid) and Isotonic calibration.
  5. Exports JSON and CSV artifacts for reproducible reporting.
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from tqdm import tqdm

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from business_entity_resolution.blocking import MultiIndexBlocker
from business_entity_resolution.config import PipelineConfig, PROJECT_ROOT
from business_entity_resolution.features import compute_pair_features
from business_entity_resolution.metrics import compute_macro_f05
from business_entity_resolution.model import EntityResolutionModel
from business_entity_resolution.threshold import evaluate_threshold, optimize_threshold
from business_entity_resolution.calibration import (
    compute_distribution_statistics,
    compute_ece,
    compute_brier_score,
    segment_probability_scores,
    analyze_threshold_sensitivity,
    evaluate_calibration_benefit,
)
from business_entity_resolution.pipeline import (
    load_and_preprocess_file,
    load_ground_truth,
    load_targeted_training_targets,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_calibration_audit(
    val_s1_limit: int = 20000,
    train_s1_limit: int = 80000,
    background_sample_per_file: int = 40000,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    config = PipelineConfig()
    if output_dir is None:
        output_dir = config.paths.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("============================================================")
    logger.info("OPTISOLVE PROBABILITY CALIBRATION & THRESHOLD AUDIT")
    logger.info("============================================================")

    # 1. Load S1 Records and Ground Truth
    total_load = train_s1_limit + val_s1_limit
    logger.info(f"Loading {total_load:,} S1 entities from {config.paths.train_source1}...")
    s1_records = load_and_preprocess_file(config.paths.train_source1, nrows=total_load)
    all_s1_ids = {r["entity_id"] for r in s1_records}

    logger.info("Loading ground truth...")
    gt = load_ground_truth(config.paths.train_ground_truth, all_s1_ids)

    # Replicate exact split from pipeline.fit
    np.random.seed(config.random_seed)
    all_s1_list = list(s1_records)
    singleton_s1 = [r for r in all_s1_list if len(gt.get(r["entity_id"], set())) == 0]
    matched_s1 = [r for r in all_s1_list if len(gt.get(r["entity_id"], set())) > 0]
    np.random.shuffle(singleton_s1)
    np.random.shuffle(matched_s1)

    val_size = min(val_s1_limit, max(1, len(all_s1_list) - 1))
    val_singletons = max(1, min(len(singleton_s1) - 1, int(round(val_size * (len(singleton_s1) / len(all_s1_list))))))
    val_matched = min(len(matched_s1) - 1, val_size - val_singletons)
    val_s1 = singleton_s1[:val_singletons] + matched_s1[:val_matched]

    val_s1_ids = {r["entity_id"] for r in val_s1}
    val_gt = {s1_id: gt.get(s1_id, set()) for s1_id in val_s1_ids}

    logger.info(f"Holdout Validation Set: {len(val_s1):,} S1 entities ({val_singletons:,} singletons, {val_matched:,} matched)")
    logger.info(f"Total validation ground-truth links: {sum(len(m) for m in val_gt.values()):,}")

    # 2. Load Validation Targets
    val_needed_targets = set()
    for s1_id in val_s1_ids:
        val_needed_targets.update(val_gt[s1_id])

    logger.info(f"Loading validation target pool ({len(val_needed_targets):,} required targets)...")
    val_targets = load_targeted_training_targets(
        [config.paths.train_source2, config.paths.train_source3],
        needed_ids=val_needed_targets,
        background_sample_per_file=background_sample_per_file,
    )
    val_target_map = {r["entity_id"]: r for r in val_targets}
    logger.info(f"Validation target pool loaded: {len(val_targets):,} records")

    # 3. Index Blocker
    logger.info("Indexing validation blocker (exact production blocker)...")
    blocker = MultiIndexBlocker(
        max_candidates=config.blocking.max_candidates_per_entity,
        min_token_len=config.blocking.min_token_len,
        name_prefix_len=config.blocking.name_prefix_len,
        max_block_size=config.blocking.max_block_size,
        sub_block_threshold=config.blocking.sub_block_threshold,
    )
    blocker.index_targets(val_targets)
    blocker.prune_large_blocks()

    # 4. Load Trained LightGBM Model
    logger.info(f"Loading trained LightGBM model from {config.paths.model_path}...")
    model = EntityResolutionModel(config.model)
    model.load(config.paths.model_path)

    # 5. Generate Validation Predictions
    logger.info("Scoring validation candidate pairs...")
    val_scored_pairs: Dict[str, List[Tuple[str, float]]] = {}
    total_val_pairs = 0

    for idx, s1_rec in enumerate(val_s1):
        s1_id = s1_rec["entity_id"]
        cands = blocker.retrieve_candidates(s1_rec)
        valid_cands = [c for c in cands if c in val_target_map]
        if not valid_cands:
            val_scored_pairs[s1_id] = []
            continue

        cand_feats = [compute_pair_features(s1_rec, val_target_map[c]) for c in valid_cands]
        X_cand = np.array(cand_feats, dtype=np.float32)
        probs = model.predict_proba(X_cand)
        val_scored_pairs[s1_id] = list(zip(valid_cands, probs.tolist()))
        total_val_pairs += len(valid_cands)

        if (idx + 1) % 5000 == 0 or (idx + 1) == len(val_s1):
            logger.info(f"Scored {idx + 1:,}/{len(val_s1):,} validation entities ({total_val_pairs:,} pairs)")

    # 6. Analyze Probability Distributions
    logger.info("Segmenting probability distributions (TP, Hard Negatives, Singleton Negatives, S2, S3)...")
    segments = segment_probability_scores(val_gt, val_scored_pairs)

    dist_stats = {}
    for seg_name, scores in segments.items():
        dist_stats[seg_name] = compute_distribution_statistics(scores)

    # Extract flat pairs for ECE and Brier score
    all_probs = segments["all_candidates"]
    all_labels = []
    for s1_id, cands in val_scored_pairs.items():
        true_m = val_gt.get(s1_id, set())
        for cid, _ in cands:
            all_labels.append(1 if cid in true_m else 0)
    all_labels_arr = np.array(all_labels, dtype=np.int32)

    ece_value, reliability_bins = compute_ece(all_labels_arr, all_probs, n_bins=10)
    brier = compute_brier_score(all_labels_arr, all_probs)

    logger.info(f"Overall Expected Calibration Error (ECE): {ece_value:.4f}")
    logger.info(f"Overall Brier Score:                    {brier:.4f}")

    # 7. Threshold Sensitivity Analysis
    logger.info("Evaluating threshold sensitivity across grid [0.50 .. 0.95] and around optimum...")
    coarse_grid = [round(t, 2) for t in np.arange(0.50, 0.96, 0.05).tolist()]
    fine_grid = [round(t, 3) for t in np.arange(0.74, 0.825, 0.005).tolist()]
    full_grid = sorted(list(set(coarse_grid + fine_grid)))

    sensitivity_records = analyze_threshold_sensitivity(val_gt, val_scored_pairs, full_grid)

    # Find optimum and sensitivity around it
    best_rec = max(sensitivity_records, key=lambda r: r["macro_f05"])
    best_tau = best_rec["threshold"]

    # Compute local Lipschitz constant in [best_tau - 0.02, best_tau + 0.02]
    neighborhood_recs = [r for r in sensitivity_records if abs(r["threshold"] - best_tau) <= 0.02 + 1e-4]
    max_local_sensitivity = max(abs(r["sensitivity_d_score"]) for r in neighborhood_recs) if neighborhood_recs else 0.0

    logger.info("=" * 60)
    logger.info(f"THRESHOLD SENSITIVITY SUMMARY:")
    logger.info(f"  Optimal Threshold:            tau* = {best_tau:.3f}")
    logger.info(f"  Optimal Macro F0.5:           {best_rec['macro_f05']:.4f}")
    logger.info(f"  Predicted Links at tau*:      {best_rec['predicted_links']:,}")
    logger.info(f"  Empty Predictions at tau*:    {best_rec['empty_predictions']:,}")
    logger.info(f"  Singleton FPs at tau*:        {best_rec['singleton_false_positives']:,}")
    logger.info(f"  Local Max Sensitivity |dF/d|: {max_local_sensitivity:.4f}")
    logger.info(f"  Plateau Stability:            {'STABLE (smooth plateau)' if max_local_sensitivity <= 0.15 else 'SENSITIVE (steep peak)'}")
    logger.info("=" * 60)

    # 8. Honest Calibration Evaluation (2-Fold Disjoint Validation Split)
    logger.info("Evaluating honest calibration benefit via 2-fold cross-split...")
    calib_bench = evaluate_calibration_benefit(val_gt, val_scored_pairs, random_seed=config.random_seed)

    logger.info(f"Calibration Evaluation Recommendation: {calib_bench['recommendation']['best_method'].upper()}")
    logger.info(f"Rationale: {calib_bench['recommendation']['rationale']}")

    # 9. Save CSV and JSON Artifacts
    # CSV 1: Threshold Sensitivity Table
    csv_sensitivity_path = output_dir / "threshold_sensitivity.csv"
    with open(csv_sensitivity_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "macro_f05",
                "precision",
                "recall",
                "predicted_links",
                "true_positives",
                "empty_predictions",
                "singleton_false_positives",
                "sensitivity_d_score",
                "is_stable",
            ],
        )
        writer.writeheader()
        for r in sensitivity_records:
            writer.writerow(r)
    logger.info(f"Saved threshold sensitivity table to {csv_sensitivity_path}")

    # CSV 2: Probability Distribution Summary Table
    csv_dist_path = output_dir / "probability_distribution_summary.csv"
    with open(csv_dist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["segment", "count", "mean", "std", "min", "p10", "p25", "p50_median", "p75", "p90", "p95", "p99", "max"])
        for seg_name, st in dist_stats.items():
            writer.writerow([
                seg_name,
                st["count"],
                f"{st['mean']:.4f}",
                f"{st['std']:.4f}",
                f"{st['min']:.4f}",
                f"{st['p10']:.4f}",
                f"{st['p25']:.4f}",
                f"{st['p50_median']:.4f}",
                f"{st['p75']:.4f}",
                f"{st['p90']:.4f}",
                f"{st['p95']:.4f}",
                f"{st['p99']:.4f}",
                f"{st['max']:.4f}",
            ])
    logger.info(f"Saved probability distribution summary to {csv_dist_path}")

    # JSON: Full Calibration Audit Results
    audit_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "validation_s1_count": len(val_s1),
        "validation_candidate_pair_count": total_val_pairs,
        "calibration_metrics": {
            "expected_calibration_error_ece": round(float(ece_value), 5),
            "brier_score": round(float(brier), 5),
            "reliability_diagram_bins": reliability_bins,
        },
        "distribution_statistics": dist_stats,
        "threshold_sensitivity": {
            "optimal_threshold": float(best_tau),
            "optimal_macro_f05": float(best_rec["macro_f05"]),
            "optimal_predicted_links": int(best_rec["predicted_links"]),
            "optimal_empty_predictions": int(best_rec["empty_predictions"]),
            "optimal_singleton_false_positives": int(best_rec["singleton_false_positives"]),
            "max_local_sensitivity_around_optimum": round(float(max_local_sensitivity), 4),
            "is_plateau_stable": bool(max_local_sensitivity <= 0.15),
            "records": sensitivity_records,
        },
        "honest_calibration_benchmark": calib_bench,
    }

    json_path = output_dir / "calibration_audit_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    logger.info(f"Saved full calibration audit results to {json_path}")

    return audit_results


def main():
    parser = argparse.ArgumentParser(description="OptiResolve Calibration and Threshold Sensitivity Audit")
    parser.add_argument("--val-limit", type=int, default=20000, help="Number of holdout validation S1 entities")
    parser.add_argument("--train-limit", type=int, default=80000, help="Number of training S1 entities to split from")
    parser.add_argument("--bg-sample", type=int, default=40000, help="Target background sample per file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for JSON/CSV results")
    args = parser.parse_args()

    out_path = Path(args.output_dir) if args.output_dir else None
    run_calibration_audit(
        val_s1_limit=args.val_limit,
        train_s1_limit=args.train_limit,
        background_sample_per_file=args.bg_sample,
        output_dir=out_path,
    )


if __name__ == "__main__":
    main()

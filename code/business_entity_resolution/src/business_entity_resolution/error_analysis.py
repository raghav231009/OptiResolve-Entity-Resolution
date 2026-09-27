"""
Automated Error-Analysis Pipeline for Business Entity Resolution.

Performs structured error taxonomy classification on validation predictions:
- False Positives:
    * same building / different business
    * same postal / different business
    * similar company names
    * generic company names
    * suite mismatch
    * address mismatch
    * S2-specific
    * S3-specific
- False Negatives:
    * candidate blocking failure (never retrieved by any channel)
    * candidate cap failure (retrieved in raw block but dropped by K cap)
    * model scoring failure (retrieved but model p < 0.30)
    * threshold failure (retrieved and scored moderately but 0.30 <= p < tau*)
    * noise profile: name corruption, transliteration, missing address, missing postal, address corruption

Saves structured error records and actionable engineering diagnostics.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .blocking import MultiIndexBlocker
from .config import PipelineConfig
from .features import FEATURE_NAMES, compute_pair_features
from .model import EntityResolutionModel
from .normalization import (
    clean_business_name,
    extract_building_number,
    extract_postal_code,
    normalize_address,
)

logger = logging.getLogger(__name__)

GENERIC_BUSINESS_TERMS = {
    "RESTAURANT", "HOLDINGS", "HOTEL", "SERVICES", "TRADING", "ENTERPRISES",
    "CORP", "INC", "LTD", "LLC", "PVT", "GROUP", "SOLUTIONS", "VENTURES",
    "COMPANY", "SYSTEMS", "GLOBAL", "INDIA", "USA", "FRANCE", "SHOP", "STORE",
    "SUPERMARKET", "CAFE", "BAR", "PHARMACY", "CLINIC", "HOSPITAL", "SALON",
}

SUITE_REGEX = re.compile(r"\b(?:SUITE|STE|APT|UNIT|ROOM|RM|FLOOR|FL|BLDG|BUILDING)\s*#?\s*([A-Z0-9\-]+)\b", re.IGNORECASE)


def extract_suite(address: str) -> Optional[str]:
    """Extract suite/unit identifier from address string."""
    m = SUITE_REGEX.search(address)
    return m.group(1).upper() if m else None


def classify_false_positive(
    s1_rec: dict,
    cand_rec: dict,
    feats: List[float],
    prob: float,
    threshold: float,
) -> List[str]:
    """
    Classify a false positive link into one or more categories:
    - same_building_different_business
    - same_postal_different_business
    - similar_company_names
    - generic_company_names
    - suite_mismatch
    - address_mismatch
    - s2_specific
    - s3_specific
    """
    categories = []
    cid = cand_rec["entity_id"]
    source = "S2" if cid.startswith("S2") else ("S3" if cid.startswith("S3") else "Unknown")
    categories.append(f"{source.lower()}_specific")

    name_jaccard = feats[0]
    name_jw = feats[1]
    addr_jaccard = feats[7]
    same_bldg = (feats[12] == 1.0)
    same_postal = (feats[13] == 1.0)

    # 1. Same building / different business
    if same_bldg and addr_jaccard >= 0.70 and name_jw < 0.65:
        categories.append("same_building_different_business")

    # 2. Same postal / different business
    if same_postal and name_jw < 0.60 and not same_bldg:
        categories.append("same_postal_different_business")

    # 3. Similar company names
    if name_jw >= 0.85 and (addr_jaccard < 0.40 and not same_bldg):
        categories.append("similar_company_names")

    # 4. Generic company names
    s1_name_words = set(s1_rec.get("clean_name", "").upper().split())
    cand_name_words = set(cand_rec.get("clean_name", "").upper().split())
    overlap = s1_name_words & cand_name_words
    if overlap and overlap.issubset(GENERIC_BUSINESS_TERMS):
        categories.append("generic_company_names")

    # 5. Suite mismatch
    s1_suite = extract_suite(s1_rec.get("clean_address", ""))
    cand_suite = extract_suite(cand_rec.get("clean_address", ""))
    if s1_suite and cand_suite and s1_suite != cand_suite and same_bldg:
        categories.append("suite_mismatch")

    # 6. Address mismatch
    if addr_jaccard < 0.35 and not same_bldg and not same_postal:
        categories.append("address_mismatch")

    return categories


def classify_false_negative_noise(
    s1_rec: dict,
    cand_rec: dict,
    feats: List[float],
) -> str:
    """Identify the dominant noise profile responsible for missing a true link."""
    name_jaccard = feats[0]
    name_jw = feats[1]
    addr_jaccard = feats[7]
    cand_addr = cand_rec.get("clean_address", "").strip()
    s1_postal = s1_rec.get("postal_code", "").strip()
    cand_postal = cand_rec.get("postal_code", "").strip()

    if not cand_addr or len(cand_addr) < 4:
        return "missing_address"
    if not s1_postal or not cand_postal:
        return "missing_postal"
    if name_jw < 0.55 and addr_jaccard >= 0.70:
        return "name_corruption"
    if name_jw >= 0.80 and addr_jaccard < 0.40:
        return "address_corruption"
    if 0.50 <= name_jw < 0.75 and name_jaccard < 0.30:
        return "transliteration"

    return "general_noise"


def run_error_analysis(
    config: Optional[PipelineConfig] = None,
    val_sample_limit: Optional[int] = 3000,
    export_json: bool = True,
    export_md: bool = True,
) -> Dict[str, Any]:
    """
    Execute end-to-end automated error analysis on validation data.
    Analyzes false positives and false negatives, saves full diagnostic records,
    and identifies the next engineering priority.
    """
    config = config or PipelineConfig()
    artifacts_dir = config.paths.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("============================================================")
    logger.info("STARTING AUTOMATED ERROR-ANALYSIS PIPELINE")
    logger.info("============================================================")

    # 1. Load model and optimal threshold
    model = EntityResolutionModel()
    if config.paths.model_path.exists():
        model.load(config.paths.model_path)
    else:
        raise FileNotFoundError(f"Model file not found at {config.paths.model_path}. Train model before error analysis.")

    tau = config.default_threshold
    thresh_file = artifacts_dir / "optimal_threshold.json"
    if thresh_file.exists():
        try:
            with open(thresh_file, "r", encoding="utf-8") as f:
                tau = float(json.load(f).get("optimal_threshold", tau))
        except Exception:
            pass

    logger.info(f"Using evaluation threshold tau* = {tau:.3f}")

    # 2. Load validation sample
    from .pipeline import (
        load_and_preprocess_file,
        load_ground_truth,
        load_isolated_target_pools,
    )

    limit = val_sample_limit or 3000
    s1_records = load_and_preprocess_file(config.paths.train_source1, nrows=limit)
    s1_ids = {r["entity_id"] for r in s1_records}
    gt = load_ground_truth(config.paths.train_ground_truth, s1_ids_filter=s1_ids)

    needed_targets = set()
    for matches in gt.values():
        needed_targets.update(matches)

    # Load targets
    logger.info(f"Loading targets for {len(s1_records):,} validation S1 entities ({len(needed_targets):,} positive targets)...")
    train_targets, val_targets = load_isolated_target_pools(
        [config.paths.train_source2, config.paths.train_source3],
        train_needed_ids=needed_targets,
        val_needed_ids=set(),
        train_background_sample_per_file=35000,
        val_background_sample_per_file=0,
        allow_missing_targets=True,
    )
    target_map = {r["entity_id"]: r for r in train_targets}

    # 3. Build Blocker
    blocker = MultiIndexBlocker(
        max_candidates=config.blocking.max_candidates_per_entity,
        min_token_len=config.blocking.min_token_len,
        name_prefix_len=config.blocking.name_prefix_len,
        max_block_size=config.blocking.max_block_size,
        sub_block_threshold=config.blocking.sub_block_threshold,
        capping_strategy=config.blocking.capping_strategy,
    )
    blocker.index_targets(train_targets)
    blocker.prune_large_blocks()

    # 4. Error classification structures
    fp_records: List[Dict[str, Any]] = []
    fn_records: List[Dict[str, Any]] = []

    fp_category_counts: Dict[str, int] = {}
    fn_stage_counts: Dict[str, int] = {
        "candidate_blocking_failure": 0,
        "candidate_cap_failure": 0,
        "model_scoring_failure": 0,
        "threshold_failure": 0,
    }
    fn_noise_counts: Dict[str, int] = {}
    source_fp_counts = {"S2": 0, "S3": 0}
    source_fn_counts = {"S2": 0, "S3": 0}

    total_true_positives = 0
    total_eval_pairs = 0

    for s1_rec in tqdm(s1_records, desc="Classifying Validation Errors"):
        s1_id = s1_rec["entity_id"]
        true_matches = gt.get(s1_id, set())

        # Retrieve candidates with channel attribution and uncapped pool
        retained, uncapped, cand_channels = blocker.retrieve_candidates_with_channel_attribution(s1_rec)
        retained_set = set(retained)

        # A. Score retained candidates with model
        cands_in_map = [c for c in retained if c in target_map]
        cand_probs: Dict[str, float] = {}
        cand_feats_map: Dict[str, List[float]] = {}

        if cands_in_map:
            feats_list = [compute_pair_features(s1_rec, target_map[c]) for c in cands_in_map]
            probs = model.predict_proba(np.array(feats_list, dtype=np.float32)).tolist()
            for cid, p, f_vec in zip(cands_in_map, probs, feats_list):
                cand_probs[cid] = float(p)
                cand_feats_map[cid] = f_vec

        # B. Identify False Positives: prob >= tau but not in true_matches
        for cid, p in cand_probs.items():
            total_eval_pairs += 1
            if p >= tau:
                if cid in true_matches:
                    total_true_positives += 1
                else:
                    cand_rec = target_map[cid]
                    f_vec = cand_feats_map[cid]
                    src = "S2" if cid.startswith("S2") else "S3"
                    source_fp_counts[src] += 1

                    categories = classify_false_positive(s1_rec, cand_rec, f_vec, p, tau)
                    for cat in categories:
                        fp_category_counts[cat] = fp_category_counts.get(cat, 0) + 1

                    fp_records.append({
                        "error_type": "false_positive",
                        "s1_id": s1_id,
                        "candidate_id": cid,
                        "source": src,
                        "categories": categories,
                        "normalized_name_s1": s1_rec.get("clean_name", ""),
                        "normalized_name_cand": cand_rec.get("clean_name", ""),
                        "normalized_address_s1": s1_rec.get("clean_address", ""),
                        "normalized_address_cand": cand_rec.get("clean_address", ""),
                        "postal_s1": s1_rec.get("postal_code", ""),
                        "postal_cand": cand_rec.get("postal_code", ""),
                        "building_s1": s1_rec.get("building_number", ""),
                        "building_cand": cand_rec.get("building_number", ""),
                        "feature_vector": f_vec,
                        "model_probability": round(p, 4),
                        "threshold": tau,
                        "ground_truth_label": 0,
                        "blocking_channels": cand_channels.get(cid, []),
                    })

        # C. Identify False Negatives: in true_matches but (not retrieved / not retained / prob < tau)
        for tm in true_matches:
            if tm not in target_map:
                continue

            src = "S2" if tm.startswith("S2") else "S3"
            target_rec = target_map[tm]
            tm_feats = compute_pair_features(s1_rec, target_rec)
            tm_noise = classify_false_negative_noise(s1_rec, target_rec, tm_feats)

            if tm not in uncapped:
                # Stage 1: Candidate Blocking Failure
                fn_stage = "candidate_blocking_failure"
                p = 0.0
            elif tm not in retained_set:
                # Stage 2: Candidate Cap Failure (in uncapped, but dropped by K)
                fn_stage = "candidate_cap_failure"
                p = float(model.predict_proba(np.array([tm_feats], dtype=np.float32))[0])
            else:
                p = cand_probs.get(tm, 0.0)
                if p < 0.30:
                    # Stage 3: Model Scoring Failure
                    fn_stage = "model_scoring_failure"
                elif p < tau:
                    # Stage 4: Threshold Failure
                    fn_stage = "threshold_failure"
                else:
                    # Successfully predicted true positive
                    continue

            fn_noise_counts[tm_noise] = fn_noise_counts.get(tm_noise, 0) + 1
            fn_stage_counts[fn_stage] += 1
            source_fn_counts[src] += 1

            fn_records.append({
                "error_type": "false_negative",
                "s1_id": s1_id,
                "candidate_id": tm,
                "source": src,
                "failure_stage": fn_stage,
                "noise_profile": tm_noise,
                "normalized_name_s1": s1_rec.get("clean_name", ""),
                "normalized_name_cand": target_rec.get("clean_name", ""),
                "normalized_address_s1": s1_rec.get("clean_address", ""),
                "normalized_address_cand": target_rec.get("clean_address", ""),
                "postal_s1": s1_rec.get("postal_code", ""),
                "postal_cand": target_rec.get("postal_code", ""),
                "building_s1": s1_rec.get("building_number", ""),
                "building_cand": target_rec.get("building_number", ""),
                "feature_vector": tm_feats,
                "model_probability": round(p, 4),
                "threshold": tau,
                "ground_truth_label": 1,
                "blocking_channels": cand_channels.get(tm, []),
            })

    total_fp = len(fp_records)
    total_fn = len(fn_records)

    # 5. Determine recommended next improvement target based on empirical error profile
    if fn_stage_counts["candidate_blocking_failure"] > (total_fn * 0.40):
        rec_target = "blocking"
        rec_rationale = (
            f"Candidate blocking failure accounts for {fn_stage_counts['candidate_blocking_failure']:,} / {total_fn:,} "
            f"({fn_stage_counts['candidate_blocking_failure'] / max(1, total_fn) * 100:.1f}%) of false negatives. "
            f"Next engineering effort should target adding or loosening blocking channels."
        )
    elif fn_stage_counts["candidate_cap_failure"] > (total_fn * 0.25):
        rec_target = "blocking_capping"
        rec_rationale = (
            f"Candidate cap K drops {fn_stage_counts['candidate_cap_failure']:,} true matches before ML scoring. "
            f"Next engineering effort should target pre-ranking priority weights or expanding cap K."
        )
    elif fn_stage_counts["threshold_failure"] > (total_fn * 0.45) and total_fp < (total_fn * 0.50):
        rec_target = "threshold"
        rec_rationale = (
            f"Threshold failure accounts for {fn_stage_counts['threshold_failure']:,} false negatives with high model scores "
            f"(0.30 <= p < tau*). Next engineering effort should evaluate lowering threshold tau*."
        )
    elif fn_stage_counts["model_scoring_failure"] > (total_fn * 0.35):
        rec_target = "features_model"
        rec_rationale = (
            f"Model scoring failure accounts for {fn_stage_counts['model_scoring_failure']:,} false negatives (p < 0.30). "
            f"Next engineering effort should target non-linear interaction features or deeper trees."
        )
    else:
        rec_target = "normalization"
        rec_rationale = (
            f"Errors are distributed across address and name corruption. "
            f"Next engineering effort should target country-specific normalization rules."
        )

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "s1_evaluated": len(s1_records),
        "locked_threshold": tau,
        "total_true_positives": total_true_positives,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
        "fp_categories_breakdown": fp_category_counts,
        "fn_failure_stages_breakdown": fn_stage_counts,
        "fn_noise_profiles_breakdown": fn_noise_counts,
        "source_breakdown": {
            "false_positives": source_fp_counts,
            "false_negatives": source_fn_counts,
        },
        "recommended_next_engineering_target": {
            "target": rec_target,
            "rationale": rec_rationale,
        },
    }

    # 6. Save artifacts
    if export_json:
        diag_path = artifacts_dir / "validation_errors.json"
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": summary,
                "sample_false_positives": fp_records[:200],
                "sample_false_negatives": fn_records[:200],
            }, f, indent=2)
        logger.info(f"Detailed error analysis records saved to {diag_path}")

    if export_md:
        md_path = artifacts_dir / "error_analysis_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(_format_error_report_md(summary, fp_records[:5], fn_records[:5]))
        logger.info(f"Error analysis markdown report saved to {md_path}")

    return summary


def _format_error_report_md(
    summary: Dict[str, Any],
    sample_fps: List[Dict[str, Any]],
    sample_fns: List[Dict[str, Any]],
) -> str:
    """Format error analysis results into an actionable markdown audit report."""
    rec = summary["recommended_next_engineering_target"]
    stages = summary["fn_failure_stages_breakdown"]
    fp_cats = summary["fp_categories_breakdown"]
    noise = summary["fn_noise_profiles_breakdown"]

    md = f"""# Automated Error-Analysis Report

**Generated:** {summary['timestamp']}  
**Evaluated S1 Entities:** {summary['s1_evaluated']:,}  
**Threshold:** $\\tau^* = {summary['locked_threshold']:.3f}$  
**Overall Breakdown:** **{summary['total_true_positives']:,} TP** | **{summary['total_false_positives']:,} FP** | **{summary['total_false_negatives']:,} FN**

---

## 1. Actionable Engineering Recommendation

> [!IMPORTANT]
> **Recommended Next Target:** `{rec['target'].upper()}`  
> **Rationale:** {rec['rationale']}

---

## 2. False Negative Root Cause Breakdown

| Failure Stage | FN Count | Percentage of Total FN | Root Cause Description |
| :--- | :--- | :--- | :--- |
| **Candidate Blocking Failure** | {stages['candidate_blocking_failure']:,} | {stages['candidate_blocking_failure'] / max(1, summary['total_false_negatives']) * 100:.1f}% | Blocker never retrieved true candidate across any of 6 channels |
| **Candidate Cap Failure** | {stages['candidate_cap_failure']:,} | {stages['candidate_cap_failure'] / max(1, summary['total_false_negatives']) * 100:.1f}% | Candidate was in raw block but dropped by candidate cap $K$ |
| **Model Scoring Failure** | {stages['model_scoring_failure']:,} | {stages['model_scoring_failure'] / max(1, summary['total_false_negatives']) * 100:.1f}% | Model gave probability $p < 0.30$ (severe ML miss) |
| **Threshold Failure** | {stages['threshold_failure']:,} | {stages['threshold_failure'] / max(1, summary['total_false_negatives']) * 100:.1f}% | Model gave $0.30 \\le p < \\tau^*$ (missed purely by conservative threshold) |

### False Negative Noise Profiles:
"""
    for n_type, count in sorted(noise.items(), key=lambda x: -x[1]):
        md += f"- **`{n_type}`**: {count:,} cases ({count / max(1, summary['total_false_negatives']) * 100:.1f}%)\n"

    md += f"""
---

## 3. False Positive Taxonomy Breakdown

| Error Category | FP Count | Percentage of Total FP | Diagnostic Pattern |
| :--- | :--- | :--- | :--- |
"""
    for cat, count in sorted(fp_cats.items(), key=lambda x: -x[1]):
        md += f"| **`{cat}`** | {count:,} | {count / max(1, summary['total_false_positives']) * 100:.1f}% | Identified via multi-attribute signal check |\n"

    md += f"""
---

## 4. Source Error Balance

| Error Type | Source 2 (S2) | Source 3 (S3) | Total |
| :--- | :--- | :--- | :--- |
| **False Positives** | {summary['source_breakdown']['false_positives']['S2']:,} | {summary['source_breakdown']['false_positives']['S3']:,} | {summary['total_false_positives']:,} |
| **False Negatives** | {summary['source_breakdown']['false_negatives']['S2']:,} | {summary['source_breakdown']['false_negatives']['S3']:,} | {summary['total_false_negatives']:,} |

---

## 5. Sample Error Case Logs

### False Positive Sample:
"""
    for fp in sample_fps[:2]:
        md += f"""- **S1:** `{fp['s1_id']}` ({fp['normalized_name_s1']} | {fp['normalized_address_s1']})  
  **Candidate:** `{fp['candidate_id']}` ({fp['normalized_name_cand']} | {fp['normalized_address_cand']})  
  **Prob:** {fp['model_probability']:.4f} $\\ge$ {fp['threshold']:.3f} | **Categories:** `{', '.join(fp['categories'])}` | **Channels:** `{fp['blocking_channels']}`  
"""

    md += """### False Negative Sample:
"""
    for fn in sample_fns[:2]:
        md += f"""- **S1:** `{fn['s1_id']}` ({fn['normalized_name_s1']} | {fn['normalized_address_s1']})  
  **Target:** `{fn['candidate_id']}` ({fn['normalized_name_cand']} | {fn['normalized_address_cand']})  
  **Stage:** `{fn['failure_stage']}` | **Noise:** `{fn['noise_profile']}` | **Prob:** {fn['model_probability']:.4f}  
"""

    return md


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_error_analysis()

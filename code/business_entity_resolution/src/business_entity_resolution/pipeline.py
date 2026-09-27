"""
End-to-End Execution Pipeline for Business Entity Resolution.
Supports full dataset training, early stopping validation, threshold optimization,
and streamed inference for test sets.
"""

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from tqdm import tqdm

from .blocking import MultiIndexBlocker
from .config import PipelineConfig
from .features import compute_pair_features, FEATURE_NAMES
from .metrics import compute_macro_f05
from .model import EntityResolutionModel
from .normalization import (
    clean_business_name,
    extract_building_number,
    extract_numeric_tokens,
    extract_postal_code,
    normalize_address,
)
from .threshold import optimize_threshold

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def preprocess_record(row: dict) -> dict:
    """Normalize a single raw record dict."""
    eid = str(row["entity_id"]).strip()
    country = str(row.get("country", "")).strip().upper()
    raw_name = str(row.get("business_name", "")) if pd.notna(row.get("business_name")) else ""
    raw_addr = str(row.get("business_address", "")) if pd.notna(row.get("business_address")) else ""

    clean_name, root_name = clean_business_name(raw_name)
    clean_addr = normalize_address(raw_addr)
    postal = extract_postal_code(raw_addr)
    bldg_num = extract_building_number(clean_addr, postal)
    num_tokens = extract_numeric_tokens(clean_addr)

    return {
        "entity_id": eid,
        "country": country,
        "clean_name": clean_name,
        "root_name": root_name,
        "clean_address": clean_addr,
        "postal_code": postal,
        "building_number": bldg_num,
        "numeric_tokens": num_tokens,
    }


def load_and_preprocess_file(filepath: Path, nrows: Optional[int] = None) -> List[dict]:
    """Read TSV and preprocess all rows."""
    logger.info(f"Loading {filepath.name} (limit={nrows})...")
    df = pd.read_csv(filepath, sep="\t", nrows=nrows, dtype=str)
    records = []
    for row in df.to_dict("records"):
        records.append(preprocess_record(row))
    logger.info(f"Loaded {len(records):,} records from {filepath.name}")
    return records


def count_file_lines(filepath: Path) -> int:
    """Fast line count for a TSV file without loading entire content into memory."""
    if not filepath.exists():
        return 0
    with open(filepath, "rb") as f:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1024 * 1024), b""))


def load_ground_truth(
    filepath: Path,
    s1_ids_filter: Optional[Set[str]] = None,
    verify_all_s1_present: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """Load ground truth mapping {s1_id: set(target_ids)} with integrity verification."""
    logger.info(f"Loading ground truth from {filepath.name}...")
    df = pd.read_csv(filepath, sep="\t", dtype=str)
    gt: Dict[str, Set[str]] = {}
    gt_s1_ids: Set[str] = set()

    for _, row in df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        gt_s1_ids.add(s1)
        if s1_ids_filter is not None and s1 not in s1_ids_filter:
            continue
        col_name = "matched_entity_ids" if "matched_entity_ids" in df.columns else "ground_truth_target_ids"
        raw_matches = row.get(col_name, "")
        if pd.isna(raw_matches) or not str(raw_matches).strip():
            gt[s1] = set()
        else:
            # Handle brackets/quotes if formatted as string representation of list
            clean_str = str(raw_matches).strip("[]'\" ")
            gt[s1] = {m.strip(" '\"") for m in clean_str.split(",") if m.strip(" '\"")}

    # Requirement 8: If ground-truth IDs are missing from train_source1, fail with clear diagnostic
    if verify_all_s1_present is not None:
        missing_ids = gt_s1_ids - verify_all_s1_present
        if missing_ids:
            sample_missing = sorted(list(missing_ids))[:5]
            raise ValueError(
                f"Ground-truth integrity validation failed: {len(missing_ids):,} S1 entity IDs in "
                f"{filepath.name} do not exist in train_source1! Examples: {sample_missing}"
            )

    logger.info(f"Ground truth loaded for {len(gt):,} S1 entities (Total GT S1 IDs in file: {len(gt_s1_ids):,}).")
    return gt


def analyze_target_s1_cardinality(ground_truth: Dict[str, Set[str]]) -> Dict[str, Any]:
    """
    Analyze ground truth relationship cardinality between S1 and target operational entities (S2/S3).
    Detects whether targets legitimately map to multiple S1 entities or are strictly 1-to-1.
    """
    target_to_s1: Dict[str, List[str]] = {}
    for s1_id, targets in ground_truth.items():
        for t_id in targets:
            if t_id not in target_to_s1:
                target_to_s1[t_id] = [s1_id]
            else:
                target_to_s1[t_id].append(s1_id)

    multi_mapped = {t: s1s for t, s1s in target_to_s1.items() if len(s1s) > 1}
    max_s1_per_target = max((len(s1s) for s1s in target_to_s1.values()), default=0)

    stats = {
        "total_unique_targets": len(target_to_s1),
        "multi_mapped_targets_count": len(multi_mapped),
        "max_s1_per_target": max_s1_per_target,
        "is_strictly_one_to_one": len(multi_mapped) == 0,
        "sample_multi_mapped": {k: multi_mapped[k] for k in list(multi_mapped.keys())[:5]} if multi_mapped else {},
    }
    return stats


def load_isolated_target_pools(
    source_paths: List[Path],
    train_needed_ids: Set[str],
    val_needed_ids: Set[str],
    train_background_sample_per_file: int = 120000,
    val_background_sample_per_file: int = 60000,
    chunksize: int = 100000,
) -> Tuple[List[dict], List[dict]]:
    """
    Stream through operational target sources (Source 2 and Source 3) and construct
    two strictly isolated target pools: one for training and one for validation.

    Guarantees:
    1. Zero ground-truth leakage: validation target IDs NEVER appear in the training target pool.
    2. Zero training target IDs appear in the validation pool when corresponding to validation ground truth.
    3. Background negatives are sampled separately and disjointly (even vs odd slices).
    4. Memory-efficient streaming pass without redundant file reads.
    """
    train_targets: List[dict] = []
    val_targets: List[dict] = []

    remaining_train_needed = set(train_needed_ids)
    remaining_val_needed = set(val_needed_ids)
    all_needed_ids = train_needed_ids | val_needed_ids

    for path in source_paths:
        logger.info(f"Isolated targeted loading from {path.name}...")
        train_bg_loaded = 0
        val_bg_loaded = 0

        for chunk in pd.read_csv(path, sep="\t", chunksize=chunksize, dtype=str):
            # 1. Extract Train True Targets
            if remaining_train_needed:
                train_mask = chunk["entity_id"].isin(remaining_train_needed)
                train_rows = chunk[train_mask]
                for r in train_rows.to_dict("records"):
                    train_targets.append(preprocess_record(r))
                    remaining_train_needed.discard(r["entity_id"])
            else:
                train_mask = pd.Series(False, index=chunk.index)

            # 2. Extract Validation True Targets (Strictly disjoint from train true targets)
            if remaining_val_needed:
                val_mask = chunk["entity_id"].isin(remaining_val_needed)
                val_rows = chunk[val_mask]
                for r in val_rows.to_dict("records"):
                    val_targets.append(preprocess_record(r))
                    remaining_val_needed.discard(r["entity_id"])
            else:
                val_mask = pd.Series(False, index=chunk.index)

            # 3. Disjoint Background Distractor Sampling
            # Exclude ANY ground truth target ID (from either train or val)
            non_gt_mask = ~train_mask & ~val_mask & ~chunk["entity_id"].isin(all_needed_ids)
            distractor_rows = chunk[non_gt_mask]

            if len(distractor_rows) > 0 and (
                train_bg_loaded < train_background_sample_per_file
                or val_bg_loaded < val_background_sample_per_file
            ):
                # Partition distractors: even indices to train, odd indices to val
                even_slice = distractor_rows.iloc[0::2]
                odd_slice = distractor_rows.iloc[1::2]

                if train_bg_loaded < train_background_sample_per_file and len(even_slice) > 0:
                    take_train = min(len(even_slice), min(10000, train_background_sample_per_file - train_bg_loaded))
                    for r in even_slice.iloc[:take_train].to_dict("records"):
                        train_targets.append(preprocess_record(r))
                    train_bg_loaded += take_train

                if val_bg_loaded < val_background_sample_per_file and len(odd_slice) > 0:
                    take_val = min(len(odd_slice), min(5000, val_background_sample_per_file - val_bg_loaded))
                    for r in odd_slice.iloc[:take_val].to_dict("records"):
                        val_targets.append(preprocess_record(r))
                    val_bg_loaded += take_val

    logger.info(
        f"Isolated target loading complete: {len(train_targets):,} train targets "
        f"(Missing true: {len(remaining_train_needed)}), {len(val_targets):,} val targets "
        f"(Missing true: {len(remaining_val_needed)})."
    )
    return train_targets, val_targets


def load_targeted_training_targets(
    source_paths: List[Path],
    needed_ids: Set[str],
    forbidden_ids: Optional[Set[str]] = None,
    background_sample_per_file: int = 150000,
    chunksize: int = 100000,
) -> List[dict]:
    """
    Stream through target source files, ensuring all true matching target records
    are loaded, plus background distractors for negative mining.
    Allows optional forbidden_ids to prevent target-level leakage.
    """
    targets = []
    remaining_needed = set(needed_ids)
    forbidden = set(forbidden_ids) if forbidden_ids else set()

    for path in source_paths:
        logger.info(f"Targeted loading from {path.name}...")
        bg_loaded = 0
        for chunk in pd.read_csv(path, sep="\t", chunksize=chunksize, dtype=str):
            mask_needed = chunk["entity_id"].isin(remaining_needed)
            needed_rows = chunk[mask_needed]
            for r in needed_rows.to_dict("records"):
                targets.append(preprocess_record(r))
                remaining_needed.discard(r["entity_id"])

            if bg_loaded < background_sample_per_file:
                other_rows = chunk[~mask_needed]
                if forbidden:
                    other_rows = other_rows[~other_rows["entity_id"].isin(forbidden)]
                take_n = min(len(other_rows), min(10000, background_sample_per_file - bg_loaded))
                for r in other_rows.iloc[:take_n].to_dict("records"):
                    targets.append(preprocess_record(r))
                bg_loaded += take_n  # increment once per chunk-slice, not per record

    logger.info(f"Targeted loading complete: {len(targets):,} records loaded (Missing true targets: {len(remaining_needed)}).")
    return targets


class EntityResolutionPipeline:
    """High-efficiency entity resolution training, tuning, and prediction pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.blocker = MultiIndexBlocker(
            max_candidates=self.config.blocking.max_candidates_per_entity,
            min_token_len=self.config.blocking.min_token_len,
            name_prefix_len=self.config.blocking.name_prefix_len,
            max_block_size=self.config.blocking.max_block_size,
        )
        self.model = EntityResolutionModel(self.config.model)
        self.optimal_threshold = self.config.default_threshold

        thresh_path = self.config.paths.artifacts_dir / "optimal_threshold.json"
        if thresh_path.exists():
            try:
                with open(thresh_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.optimal_threshold = float(data.get("optimal_threshold", self.config.default_threshold))
                    logger.info(f"Loaded existing optimal threshold: {self.optimal_threshold:.3f}")
            except Exception as e:
                logger.warning(f"Could not load optimal threshold: {e}")

    def _generate_pair_matrix(
        self,
        s1_records: List[Dict[str, Any]],
        ground_truth: Dict[str, Set[str]],
        blocker: MultiIndexBlocker,
        target_map: Dict[str, Dict[str, Any]],
        desc: str = "Building Pairs",
        return_stats: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
        """
        Generate pairwise feature matrix (X) and binary labels (y) using candidate
        blocking retrieval, true match pairing, and hard-negative mining.

        Supports balanced negative sampling for both matched and zero-positive/singleton entities:
        - For entities with >=1 positive: mines up to (num_positives * max_negatives_per_positive) hard negatives.
        - For entities with 0 positives: mines up to min_negatives_per_zero_positive_entity hard negatives.
        - Strictly pulls candidates from the production blocker; never samples arbitrary random pairs.
        - Strictly prevents true positive matches from receiving negative labels.
        """
        X_list, y_list = [], []
        max_neg_per_pos = self.config.max_negatives_per_positive
        min_neg_zero_pos = self.config.min_negatives_per_zero_positive_entity

        zero_positive_s1_count = 0
        zero_positive_contributing = 0
        positive_s1_count = 0
        positive_contributing = 0
        negatives_per_entity: List[int] = []

        for s1_rec in tqdm(s1_records, desc=desc):
            s1_id = s1_rec["entity_id"]
            true_matches = ground_truth.get(s1_id, set())
            num_pos = len(true_matches)
            candidates = blocker.retrieve_candidates(s1_rec)

            if num_pos == 0:
                zero_positive_s1_count += 1
                target_negatives = min_neg_zero_pos
            else:
                positive_s1_count += 1
                target_negatives = num_pos * max_neg_per_pos

            # Positives
            for m_id in true_matches:
                if m_id in target_map:
                    feats = compute_pair_features(s1_rec, target_map[m_id])
                    X_list.append(feats)
                    y_list.append(1)

            # Mined Hard Negatives from blocker candidates
            # Requirement 5 & 6: Prefer blocker candidates; do not sample arbitrary negatives
            # Requirement 10: Ensure negative labels are never accidentally assigned to true matches
            neg_count = 0
            for c_id in candidates:
                if c_id not in true_matches and c_id in target_map:
                    feats = compute_pair_features(s1_rec, target_map[c_id])
                    X_list.append(feats)
                    y_list.append(0)
                    neg_count += 1
                    if neg_count >= target_negatives:
                        break

            negatives_per_entity.append(neg_count)
            if num_pos == 0 and neg_count > 0:
                zero_positive_contributing += 1
            elif num_pos > 0 and neg_count > 0:
                positive_contributing += 1

        X = np.array(X_list, dtype=np.float32) if X_list else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        y = np.array(y_list, dtype=np.int32) if y_list else np.empty(0, dtype=np.int32)

        total_positives = int(np.sum(y == 1))
        total_negatives = int(np.sum(y == 0))
        avg_negatives = float(np.mean(negatives_per_entity)) if negatives_per_entity else 0.0
        min_negatives = int(min(negatives_per_entity)) if negatives_per_entity else 0
        max_negatives = int(max(negatives_per_entity)) if negatives_per_entity else 0

        stats = {
            "total_s1_entities": len(s1_records),
            "zero_positive_s1_count": zero_positive_s1_count,
            "zero_positive_contributing_negatives": zero_positive_contributing,
            "positive_s1_count": positive_s1_count,
            "positive_contributing_negatives": positive_contributing,
            "avg_negatives_per_entity": round(avg_negatives, 2),
            "min_negatives_per_entity": min_negatives,
            "max_negatives_per_entity": max_negatives,
            "total_positives": total_positives,
            "total_negatives": total_negatives,
            "class_balance_ratio_neg_to_pos": round(total_negatives / total_positives, 2) if total_positives > 0 else 0.0,
        }

        if return_stats:
            return X, y, stats
        return X, y

    def fit(self):
        """Train pipeline with strict holdout validation and early stopping."""
        logger.info("Starting Pipeline Training Phase...")

        # 1. Inspect total available S1 rows in train_source1
        total_available_s1 = count_file_lines(self.config.paths.train_source1) - 1

        is_production_full = self.config.is_production and self.config.train_s1_limit is None
        effective_limit = None if is_production_full else self.config.train_s1_limit

        train_s1 = load_and_preprocess_file(
            self.config.paths.train_source1,
            nrows=effective_limit,
        )
        s1_selected = len(train_s1)
        coverage_pct = (s1_selected / total_available_s1 * 100.0) if total_available_s1 > 0 else 0.0

        # Requirement 6: Assert that final production training uses 100% of available train_source1 rows
        if is_production_full:
            assert s1_selected == total_available_s1, (
                f"Production training assertion failed: final training must use 100% of available "
                f"train_source1 rows! Selected {s1_selected:,} of {total_available_s1:,} ({coverage_pct:.2f}%)."
            )
        elif self.config.is_production and self.config.train_s1_limit is not None:
            logger.warning(
                f"Production training limit explicitly overridden by user: {s1_selected:,} / {total_available_s1:,} ({coverage_pct:.2f}%)"
            )

        all_s1_ids = {r["entity_id"] for r in train_s1}

        # 2. Load Ground Truth and verify S1 ID completeness
        # Requirement 7 & 8: Verify all ground-truth S1 IDs exist; fail with clear diagnostic if missing
        verify_ids = all_s1_ids if is_production_full else None
        gt = load_ground_truth(
            self.config.paths.train_ground_truth,
            s1_ids_filter=all_s1_ids,
            verify_all_s1_present=verify_ids,
        )

        s1_with_gt = sum(1 for sid in all_s1_ids if sid in gt and len(gt[sid]) > 0)
        positive_links = sum(len(gt[sid]) for sid in all_s1_ids if sid in gt)

        # Ground Truth target-to-S1 relationship cardinality audit
        cardinality_stats = analyze_target_s1_cardinality(gt)
        logger.info(
            f"Ground truth target cardinality audit: {cardinality_stats['total_unique_targets']:,} unique targets, "
            f"max S1 per target = {cardinality_stats['max_s1_per_target']}, "
            f"multi-mapped targets = {cardinality_stats['multi_mapped_targets_count']} "
            f"({'strictly 1:1' if cardinality_stats['is_strictly_one_to_one'] else 'supports 1:N mapping'})."
        )

        # Requirement 5: Explicitly log coverage metrics
        logger.info("============================================================")
        logger.info("TRAINING DATASET COVERAGE AUDIT:")
        logger.info(f"  Training Mode:                 {self.config.training_mode.upper()}")
        logger.info(f"  Total S1 Rows Available:       {total_available_s1:,}")
        logger.info(f"  S1 Rows Selected:              {s1_selected:,}")
        logger.info(f"  Training Coverage:             {coverage_pct:.2f}%")
        logger.info(f"  S1 Entities with Ground Truth: {s1_with_gt:,}")
        logger.info(f"  Total Positive Links:          {positive_links:,}")
        logger.info("============================================================")

        # 3. SPLIT S1 FIRST into train and holdout validation sets (Prevents Validation Leakage)
        all_s1_list = list(train_s1)
        np.random.seed(self.config.random_seed)

        # Requirement 8: Ensure singleton entities are represented in both training and validation
        singleton_s1 = [r for r in all_s1_list if len(gt.get(r["entity_id"], set())) == 0]
        matched_s1 = [r for r in all_s1_list if len(gt.get(r["entity_id"], set())) > 0]
        np.random.shuffle(singleton_s1)
        np.random.shuffle(matched_s1)

        if self.config.val_s1_limit is not None:
            val_size = min(self.config.val_s1_limit, max(1, len(all_s1_list) - 1))
        else:
            val_size = max(500, len(all_s1_list) // 5)

        if len(singleton_s1) > 0 and len(matched_s1) > 0 and val_size > 1:
            val_singletons = max(1, min(len(singleton_s1) - 1, int(round(val_size * (len(singleton_s1) / len(all_s1_list))))))
            val_matched = min(len(matched_s1) - 1, val_size - val_singletons)
            val_s1 = singleton_s1[:val_singletons] + matched_s1[:val_matched]
            fit_s1 = singleton_s1[val_singletons:] + matched_s1[val_matched:]
        else:
            shuffled = list(all_s1_list)
            np.random.shuffle(shuffled)
            val_s1 = shuffled[:val_size]
            fit_s1 = shuffled[val_size:]

        fit_s1_ids = {r["entity_id"] for r in fit_s1}
        val_s1_ids = {r["entity_id"] for r in val_s1}
        assert fit_s1_ids.isdisjoint(val_s1_ids), (
            f"Critical integrity failure: train and validation S1 sets overlap by {len(fit_s1_ids & val_s1_ids)} entities!"
        )

        fit_sing_count = sum(1 for r in fit_s1 if len(gt.get(r["entity_id"], set())) == 0)
        val_sing_count = sum(1 for r in val_s1 if len(gt.get(r["entity_id"], set())) == 0)
        logger.info(
            f"S1 train/val split: {len(fit_s1):,} train ({fit_sing_count:,} singletons), "
            f"{len(val_s1):,} val ({val_sing_count:,} singletons)."
        )

        val_gt = {r["entity_id"]: gt.get(r["entity_id"], set()) for r in val_s1}
        fit_gt = {r["entity_id"]: gt.get(r["entity_id"], set()) for r in fit_s1}

        # Separate target IDs for training and validation independently (Strict Validation Isolation)
        fit_needed_ids = set()
        for matches in fit_gt.values():
            fit_needed_ids.update(matches)

        val_needed_ids = set()
        for matches in val_gt.values():
            val_needed_ids.update(matches)

        logger.info(f"Independent target ID extraction: {len(fit_needed_ids):,} train targets, {len(val_needed_ids):,} val targets.")

        # 4 & 5. Build Separate, Leakage-Free Target Pools for Train and Validation
        logger.info("Loading isolated training and validation target pools...")
        train_targets, val_targets = load_isolated_target_pools(
            [self.config.paths.train_source2, self.config.paths.train_source3],
            train_needed_ids=fit_needed_ids,
            val_needed_ids=val_needed_ids,
            train_background_sample_per_file=120000,
            val_background_sample_per_file=60000,
        )

        train_target_ids = {r["entity_id"] for r in train_targets}
        val_target_ids = {r["entity_id"] for r in val_targets}
        intersection = train_target_ids.intersection(val_target_ids)

        # Requirement 9: Log train target IDs, validation target IDs, intersection size
        logger.info("============================================================")
        logger.info("TARGET-LEVEL VALIDATION LEAKAGE AUDIT:")
        logger.info(f"  Training Target IDs:         {len(train_target_ids):,}")
        logger.info(f"  Validation Target IDs:       {len(val_target_ids):,}")
        logger.info(f"  Target Intersection Size:    {len(intersection):,}")
        logger.info(f"  Multi-Mapped Target Count:   {cardinality_stats['multi_mapped_targets_count']:,}")
        logger.info(f"  Max S1 Entities per Target:  {cardinality_stats['max_s1_per_target']}")
        logger.info("============================================================")

        # Requirement 10: Assert the intended separation
        assert len(intersection) == 0, (
            f"Target-level validation leakage detected! {len(intersection):,} targets overlap "
            f"between train and validation pools: {list(intersection)[:5]}"
        )
        assert train_target_ids.isdisjoint(val_needed_ids), (
            f"Critical target leakage: validation ground-truth targets found in training target pool!"
        )
        assert val_target_ids.isdisjoint(fit_needed_ids), (
            f"Critical target leakage: training ground-truth targets found in validation target pool!"
        )

        # Build isolated training and validation blockers
        train_blocker = MultiIndexBlocker(
            max_candidates=self.config.blocking.max_candidates_per_entity,
            min_token_len=self.config.blocking.min_token_len,
            name_prefix_len=self.config.blocking.name_prefix_len,
            max_block_size=self.config.blocking.max_block_size,
            sub_block_threshold=self.config.blocking.sub_block_threshold,
        )
        train_blocker.index_targets(train_targets)
        train_blocker.prune_large_blocks()
        train_target_map = {r["entity_id"]: r for r in train_targets}

        val_blocker = MultiIndexBlocker(
            max_candidates=self.config.blocking.max_candidates_per_entity,
            min_token_len=self.config.blocking.min_token_len,
            name_prefix_len=self.config.blocking.name_prefix_len,
            max_block_size=self.config.blocking.max_block_size,
            sub_block_threshold=self.config.blocking.sub_block_threshold,
        )
        val_blocker.index_targets(val_targets)
        val_blocker.prune_large_blocks()
        val_target_map = {r["entity_id"]: r for r in val_targets}

        # 6. Build Training and Validation Pair Matrices using identical feature extraction
        logger.info("Generating training pairs using train_blocker...")
        X_train, y_train, train_stats = self._generate_pair_matrix(
            fit_s1, fit_gt, train_blocker, train_target_map, desc="Building Train Pairs", return_stats=True
        )
        train_blocker.log_retrieval_statistics(logger)

        logger.info("Generating validation pairs for model early stopping using val_blocker...")
        X_val, y_val, val_stats = self._generate_pair_matrix(
            val_s1, val_gt, val_blocker, val_target_map, desc="Building Val Pairs", return_stats=True
        )
        val_blocker.log_retrieval_statistics(logger)

        train_pos = int(train_stats["total_positives"])
        train_neg = int(train_stats["total_negatives"])
        val_pos = int(val_stats["total_positives"])
        val_neg = int(val_stats["total_negatives"])

        # Requirement 7: Explicitly log negative sampling and class balance metrics
        logger.info("============================================================")
        logger.info("NEGATIVE SAMPLING & CLASS BALANCE REPORT:")
        logger.info("  Training Set:")
        logger.info(f"    Total S1 Entities:                  {train_stats['total_s1_entities']:,}")
        logger.info(f"    Zero-Positive (Singleton) Entities: {train_stats['zero_positive_s1_count']:,}")
        logger.info(f"    Zero-Positives Contributing Negs:   {train_stats['zero_positive_contributing_negatives']:,}")
        logger.info(f"    Matched Entities Contributing Negs: {train_stats['positive_contributing_negatives']:,}")
        logger.info(f"    Average Negatives / Entity:         {train_stats['avg_negatives_per_entity']}")
        logger.info(f"    Min / Max Negatives / Entity:       {train_stats['min_negatives_per_entity']} / {train_stats['max_negatives_per_entity']}")
        logger.info(f"    Positives: {train_pos:,} | Negatives: {train_neg:,} (Ratio: 1:{train_stats['class_balance_ratio_neg_to_pos']})")
        logger.info("  Validation Set:")
        logger.info(f"    Total S1 Entities:                  {val_stats['total_s1_entities']:,}")
        logger.info(f"    Zero-Positive (Singleton) Entities: {val_stats['zero_positive_s1_count']:,}")
        logger.info(f"    Zero-Positives Contributing Negs:   {val_stats['zero_positive_contributing_negatives']:,}")
        logger.info(f"    Average Negatives / Entity:         {val_stats['avg_negatives_per_entity']}")
        logger.info(f"    Min / Max Negatives / Entity:       {val_stats['min_negatives_per_entity']} / {val_stats['max_negatives_per_entity']}")
        logger.info(f"    Positives: {val_pos:,} | Negatives: {val_neg:,} (Ratio: 1:{val_stats['class_balance_ratio_neg_to_pos']})")
        logger.info("============================================================")

        # 7. Train LightGBM model WITH active validation early stopping
        logger.info("Fitting LightGBM classifier with early stopping...")
        self.model.train(X_train, y_train, X_val=X_val, y_val=y_val)

        best_iter = self.model.best_iteration_
        val_logloss = self.model.validation_loss_

        logger.info("============================================================")
        logger.info("LIGHTGBM EARLY STOPPING TRAINING REPORT:")
        logger.info(f"  Training Pair Count:         {len(X_train):,}")
        logger.info(f"  Validation Pair Count:       {len(X_val):,}")
        logger.info(f"  Training Positives:          {train_pos:,}")
        logger.info(f"  Training Negatives:          {train_neg:,}")
        logger.info(f"  Validation Positives:        {val_pos:,}")
        logger.info(f"  Validation Negatives:        {val_neg:,}")
        logger.info(f"  Configured n_estimators:     {self.config.model.n_estimators}")
        logger.info(f"  Best Iteration:              {best_iter}")
        logger.info(f"  Validation Binary Logloss:   {val_logloss if val_logloss is not None else 'N/A'}")
        logger.info("============================================================")

        # Requirement 6: Verify the model actually stopped at best_iteration_
        if len(X_val) > 0 and best_iter is not None:
            assert best_iter > 0, "Model best_iteration_ must be a positive integer"
            assert best_iter <= self.config.model.n_estimators, (
                f"Model best_iteration_ ({best_iter}) exceeds configured "
                f"n_estimators ({self.config.model.n_estimators})"
            )
            logger.info(
                f"Verified: model stopped at best_iteration_ = {best_iter} "
                f"(configured max n_estimators = {self.config.model.n_estimators})"
            )

        # Feature importances
        importances = self.model.get_feature_importances()
        logger.info(f"Top 6 Discriminative Features: {list(importances.items())[:6]}")

        # Save model
        self.model.save(self.config.paths.model_path)
        logger.info(f"Model serialized to {self.config.paths.model_path}")

        # 8. Threshold Optimization on Unseen Validation Set using isolated val_blocker
        logger.info("Scoring validation candidates for challenge Macro F_0.5 threshold tuning...")
        val_scored_pairs: Dict[str, List[Tuple[str, float]]] = {}
        for s1_rec in tqdm(val_s1, desc="Scoring Val Candidates"):
            s1_id = s1_rec["entity_id"]
            candidates = val_blocker.retrieve_candidates(s1_rec)
            valid_cands = [c for c in candidates if c in val_target_map]
            if not valid_cands:
                val_scored_pairs[s1_id] = []
                continue

            cand_feats = [compute_pair_features(s1_rec, val_target_map[c]) for c in valid_cands]
            X_cand = np.array(cand_feats, dtype=np.float32)
            probs = self.model.predict_proba(X_cand)
            val_scored_pairs[s1_id] = list(zip(valid_cands, probs.tolist()))

        logger.info("Optimizing threshold for challenge Macro F_0.5...")
        best_tau, best_score, history = optimize_threshold(
            val_gt,
            val_scored_pairs,
            search_start=self.config.threshold_search_start,
            search_end=self.config.threshold_search_end,
            step=self.config.threshold_search_step,
            fine_step=getattr(self.config, "threshold_fine_step", 0.002),
            fine_window=getattr(self.config, "threshold_fine_window", 0.03),
        )
        self.optimal_threshold = best_tau

        best_metric = history.get(best_tau)
        best_links = int(best_metric["predicted_links"]) if best_metric else 0
        best_empty = int(best_metric["empty_predictions"]) if best_metric else 0
        best_singleton_fp = int(best_metric["singleton_false_positives"]) if best_metric else 0

        thresh_path = self.config.paths.artifacts_dir / "optimal_threshold.json"
        with open(thresh_path, "w", encoding="utf-8") as f:
            json.dump({
                "optimal_threshold": float(best_tau),
                "validation_macro_f05": float(best_score),
                "predicted_links": best_links,
                "empty_predictions": best_empty,
                "singleton_false_positives": best_singleton_fp,
                "search_history": [
                    metric.to_dict() if hasattr(metric, "to_dict") else dict(metric)
                    for t, metric in sorted(history.items(), key=lambda x: x[0])
                ],
            }, f, indent=2)
        logger.info(
            f"==> OPTIMAL THRESHOLD LOCKED: tau* = {best_tau:.3f} with Validation Macro F0.5 = {best_score:.4f}, "
            f"predicted_links = {best_links}, empty_predictions = {best_empty}, "
            f"singleton_false_positives = {best_singleton_fp} (Saved to {thresh_path})"
        )

        # Persist comprehensive training results for reproducibility (Issue #15)
        results_path = self.config.paths.artifacts_dir / "training_results.json"
        training_results = {
            "validation_macro_f05": float(best_score),
            "optimal_threshold": float(best_tau),
            "optimal_threshold_metrics": {
                "threshold": float(best_tau),
                "macro_f05": float(best_score),
                "predicted_links": best_links,
                "empty_predictions": best_empty,
                "singleton_false_positives": best_singleton_fp,
            },
            "threshold_search_history": {str(k): float(v) for k, v in history.items()},
            "threshold_search_records": [
                metric.to_dict() if hasattr(metric, "to_dict") else dict(metric)
                for t, metric in sorted(history.items(), key=lambda x: x[0])
            ],
            "train_matrix_shape": list(X_train.shape),
            "train_pair_count": int(len(X_train)),
            "train_positives": train_pos,
            "train_negatives": train_neg,
            "val_matrix_shape": list(X_val.shape),
            "val_pair_count": int(len(X_val)),
            "val_positives": val_pos,
            "val_negatives": val_neg,
            "train_s1_count": len(fit_s1),
            "val_s1_count": len(val_s1),
            "early_stopping_best_iteration": int(best_iter if best_iter is not None else self.config.model.n_estimators),
            "validation_binary_logloss": float(val_logloss) if val_logloss is not None else None,
            "stopped_early": bool(best_iter is not None and best_iter < self.config.model.n_estimators),
            "training_coverage": {
                "training_mode": self.config.training_mode,
                "total_s1_available": int(total_available_s1),
                "s1_selected": int(s1_selected),
                "coverage_percentage": float(coverage_pct),
                "s1_entities_with_gt": int(s1_with_gt),
                "positive_links": int(positive_links),
                "negative_pairs": train_neg,
            },
            "feature_importances": {k: float(v) for k, v in importances.items()},
            "target_leakage_audit": {
                "train_target_count": int(len(train_target_ids)),
                "val_target_count": int(len(val_target_ids)),
                "target_intersection_size": int(len(intersection)),
                "is_strictly_disjoint": bool(len(intersection) == 0),
                "multi_mapped_targets_count": int(cardinality_stats["multi_mapped_targets_count"]),
                "max_s1_per_target": int(cardinality_stats["max_s1_per_target"]),
            },
            "negative_sampling_stats": {
                "train": train_stats,
                "val": val_stats,
                "min_negatives_per_zero_positive_entity": int(self.config.min_negatives_per_zero_positive_entity),
                "max_negatives_per_positive": int(self.config.max_negatives_per_positive),
            },
            "blocking_config": {
                "max_candidates_per_entity": self.config.blocking.max_candidates_per_entity,
                "max_block_size": self.config.blocking.max_block_size,
                "min_token_len": self.config.blocking.min_token_len,
                "name_prefix_len": self.config.blocking.name_prefix_len,
            },
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(training_results, f, indent=2)
        logger.info(f"Training results persisted to {results_path}")

    def predict_test(self, batch_size: int = 50000):
        """
        Run inference over official test set partitioned by country (France, US, India).
        Enforces candidate subset invariant and streams output TSVs with low memory footprint.
        """
        logger.info("Starting Scalable Test Prediction Phase...")
        out_matching = self.config.paths.matching_results
        out_candidates = self.config.paths.candidate_pairs
        out_matching.parent.mkdir(parents=True, exist_ok=True)

        # Clear/initialize files with official headers
        with open(out_matching, "w", encoding="utf-8") as f_m, open(out_candidates, "w", encoding="utf-8") as f_c:
            f_m.write("source1_entity_id\tmatched_entity_ids\n")
            f_c.write("source1_entity_id\tcandidate_entity_ids\n")

        # Dynamically discover countries from test_source1
        logger.info("Discovering open-set countries in test_source1.tsv...")
        df_countries = pd.read_csv(self.config.paths.test_source1, sep="\t", usecols=["country"])
        countries = sorted([str(c).strip().upper() for c in df_countries["country"].dropna().unique()])
        logger.info(f"Test Set Countries Discovered: {countries}")
        del df_countries
        gc.collect()

        total_s1_processed = 0
        total_matches_predicted = 0

        # Process country by country to keep memory minimal
        for country in countries:
            logger.info(f"\n>>> PROCESSING COUNTRY: {country} <<<")

            country_targets: List[dict] = []
            target_map: Dict[str, dict] = {}

            for src_path in [self.config.paths.test_source2, self.config.paths.test_source3]:
                logger.info(f"Streaming {src_path.name} for country={country}...")
                for chunk in pd.read_csv(src_path, sep="\t", chunksize=200000, dtype=str):
                    sub = chunk[chunk["country"].astype(str).str.strip().str.upper() == country]
                    for r in sub.to_dict("records"):
                        prep = preprocess_record(r)
                        country_targets.append(prep)
                        target_map[prep["entity_id"]] = prep

            logger.info(f"Loaded {len(country_targets):,} target records for country={country}.")

            logger.info(f"Building blocker index for {country}...")
            blocker = MultiIndexBlocker(
                max_candidates=self.config.blocking.max_candidates_per_entity,
                min_token_len=self.config.blocking.min_token_len,
                name_prefix_len=self.config.blocking.name_prefix_len,
                max_block_size=self.config.blocking.max_block_size,
                sub_block_threshold=self.config.blocking.sub_block_threshold,
            )
            blocker.index_targets(country_targets)
            blocker.prune_large_blocks()

            logger.info(f"Streaming {self.config.paths.test_source1.name} for country={country}...")
            s1_stream = pd.read_csv(self.config.paths.test_source1, sep="\t", chunksize=batch_size, dtype=str)

            country_s1_count = 0
            for chunk_idx, df_chunk in enumerate(s1_stream):
                sub_s1 = df_chunk[df_chunk["country"].astype(str).str.strip().str.upper() == country]
                if sub_s1.empty:
                    continue

                s1_records = [preprocess_record(r) for r in sub_s1.to_dict("records")]
                s1_candidates_map = {}
                batch_pair_feats = []
                batch_pair_keys = []

                for s1_rec in s1_records:
                    s1_id = s1_rec["entity_id"]
                    candidates = blocker.retrieve_candidates(s1_rec)
                    s1_candidates_map[s1_id] = candidates

                    for cid in candidates:
                        if cid in target_map:
                            feats = compute_pair_features(s1_rec, target_map[cid])
                            batch_pair_feats.append(feats)
                            batch_pair_keys.append((s1_id, cid))

                matched_per_s1 = {}
                if batch_pair_feats:
                    X_batch = np.array(batch_pair_feats, dtype=np.float32)
                    probs = self.model.predict_proba(X_batch)
                    for (s1_id, cid), p in zip(batch_pair_keys, probs):
                        if p >= self.optimal_threshold:
                            if s1_id not in matched_per_s1:
                                matched_per_s1[s1_id] = []
                            matched_per_s1[s1_id].append(cid)
                            total_matches_predicted += 1

                matching_lines = []
                candidate_lines = []
                for s1_rec in s1_records:
                    s1_id = s1_rec["entity_id"]
                    cands = s1_candidates_map.get(s1_id, [])
                    matches = matched_per_s1.get(s1_id, [])

                    # Invariant Check: matched IDs must be a strict subset of candidate IDs
                    assert set(matches).issubset(set(cands)), f"Invariant violation on {s1_id}: match not in candidates!"

                    candidate_lines.append(f"{s1_id}\t{','.join(cands)}\n")
                    matching_lines.append(f"{s1_id}\t{','.join(matches)}\n")

                with open(out_matching, "a", encoding="utf-8") as f_m, open(out_candidates, "a", encoding="utf-8") as f_c:
                    f_m.writelines(matching_lines)
                    f_c.writelines(candidate_lines)

                country_s1_count += len(s1_records)
                total_s1_processed += len(s1_records)
                logger.info(f"[{country}] Processed {country_s1_count:,} S1 entities...")

            blocker.log_retrieval_statistics(logger)
            del country_targets, target_map, blocker
            gc.collect()

        logger.info("\n============================================================")
        logger.info("TEST INFERENCE FINISHED SUCCESSFULLY:")
        logger.info(f"  Total S1 Entities Processed: {total_s1_processed:,}")
        logger.info(f"  Total Links Predicted:       {total_matches_predicted:,}")
        logger.info(f"  Matching Output:             {out_matching}")
        logger.info(f"  Candidate Output:            {out_candidates}")
        logger.info("============================================================\n")

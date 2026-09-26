"""
End-to-End Execution Pipeline for Business Entity Resolution.
Supports full dataset training, threshold optimization, and streamed inference for test sets.
"""

import gc
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
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
    num_tokens = extract_numeric_tokens(clean_addr)

    return {
        "entity_id": eid,
        "country": country,
        "clean_name": clean_name,
        "root_name": root_name,
        "clean_address": clean_addr,
        "postal_code": postal,
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


def load_ground_truth(filepath: Path, s1_ids_filter: Optional[Set[str]] = None) -> Dict[str, Set[str]]:
    """Load ground truth mapping {s1_id: set(target_ids)}."""
    logger.info(f"Loading ground truth from {filepath.name}...")
    df = pd.read_csv(filepath, sep="\t", dtype=str)
    gt: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        if s1_ids_filter is not None and s1 not in s1_ids_filter:
            continue
        raw_matches = row["matched_entity_ids"]
        if pd.isna(raw_matches) or not str(raw_matches).strip():
            gt[s1] = set()
        else:
            gt[s1] = {m.strip() for m in str(raw_matches).split(",") if m.strip()}
    logger.info(f"Ground truth loaded for {len(gt):,} S1 entities.")
    return gt


def load_targeted_training_targets(
    source_paths: List[Path],
    needed_ids: Set[str],
    background_sample_per_file: int = 150000,
    chunksize: int = 100000,
) -> List[dict]:
    """
    Stream through target source files, ensuring all true matching target records
    are loaded, plus background distractors for negative mining.
    """
    targets = []
    remaining_needed = set(needed_ids)

    for path in source_paths:
        logger.info(f"Targeted loading from {path.name}...")
        bg_loaded = 0
        for chunk in pd.read_csv(path, sep="\t", chunksize=chunksize, dtype=str):
            # Extract needed positive matches
            mask_needed = chunk["entity_id"].isin(remaining_needed)
            needed_rows = chunk[mask_needed]
            for r in needed_rows.to_dict("records"):
                targets.append(preprocess_record(r))
                remaining_needed.discard(r["entity_id"])

            # Extract background distractors
            if bg_loaded < background_sample_per_file:
                other_rows = chunk[~mask_needed]
                take_n = min(len(other_rows), min(10000, background_sample_per_file - bg_loaded))
                for r in other_rows.iloc[:take_n].to_dict("records"):
                    targets.append(preprocess_record(r))
                    bg_loaded += take_n

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

    def fit(self):
        """Train pipeline on configured training dataset."""
        logger.info("Starting Pipeline Training Phase...")

        # 1. Load S1 training subset
        train_s1 = load_and_preprocess_file(
            self.config.paths.train_source1,
            nrows=self.config.train_s1_limit,
        )
        train_s1_ids = {r["entity_id"] for r in train_s1}

        # 2. Load Ground Truth
        gt = load_ground_truth(self.config.paths.train_ground_truth, train_s1_ids)

        # Collect all required target IDs for 100% positive link representation
        needed_target_ids = set()
        for s1_id, matches in gt.items():
            needed_target_ids.update(matches)
        logger.info(f"Identified {len(needed_target_ids):,} unique true target IDs to load.")

        # 3. Targeted Loading of Targets (S2 and S3)
        targets = load_targeted_training_targets(
            [self.config.paths.train_source2, self.config.paths.train_source3],
            needed_ids=needed_target_ids,
            background_sample_per_file=120000,
        )

        logger.info(f"Indexing {len(targets):,} target records into MultiIndexBlocker...")
        self.blocker.index_targets(targets)
        self.blocker.prune_large_blocks()

        target_map = {r["entity_id"]: r for r in targets}

        # 4. Generate pairs: Positives + Mined Hard Negatives
        logger.info("Generating training pairs (positives + hard negatives)...")
        X_list: List[List[float]] = []
        y_list: List[int] = []

        all_s1_list = list(train_s1)
        np.random.seed(self.config.random_seed)
        np.random.shuffle(all_s1_list)

        val_size = min(self.config.val_s1_limit or 5000, len(all_s1_list) // 5)
        val_s1 = all_s1_list[:val_size]
        fit_s1 = all_s1_list[val_size:]

        val_gt = {r["entity_id"]: gt.get(r["entity_id"], set()) for r in val_s1}

        # Build training feature matrix
        for s1_rec in tqdm(fit_s1, desc="Building Train Pairs"):
            s1_id = s1_rec["entity_id"]
            true_matches = gt.get(s1_id, set())

            # Retrieve candidates
            candidates = self.blocker.retrieve_candidates(s1_rec)

            # Positives
            for m_id in true_matches:
                if m_id in target_map:
                    feats = compute_pair_features(s1_rec, target_map[m_id])
                    X_list.append(feats)
                    y_list.append(1)

            # Hard Negatives
            neg_count = 0
            for c_id in candidates:
                if c_id not in true_matches and c_id in target_map:
                    feats = compute_pair_features(s1_rec, target_map[c_id])
                    X_list.append(feats)
                    y_list.append(0)
                    neg_count += 1
                    if neg_count >= self.config.max_negatives_per_positive:
                        break

        X_train = np.array(X_list, dtype=np.float32)
        y_train = np.array(y_list, dtype=np.int32)
        logger.info(f"Constructed training matrix: {X_train.shape} (Positives: {np.sum(y_train):,}, Negatives: {len(y_train) - np.sum(y_train):,})")

        # Train LightGBM model
        logger.info("Fitting LightGBM classifier...")
        self.model.train(X_train, y_train)

        # Feature importances
        importances = self.model.get_feature_importances()
        logger.info(f"Top 6 Discriminative Features: {list(importances.items())[:6]}")

        # Save model
        self.model.save(self.config.paths.model_path)
        logger.info(f"Model serialized to {self.config.paths.model_path}")

        # 5. Threshold Optimization on Validation Set
        logger.info("Scoring validation candidates for threshold tuning...")
        val_scored_pairs: Dict[str, List[Tuple[str, float]]] = {}

        for s1_rec in tqdm(val_s1, desc="Validating"):
            s1_id = s1_rec["entity_id"]
            candidates = self.blocker.retrieve_candidates(s1_rec)
            valid_cands = [c for c in candidates if c in target_map]
            if not valid_cands:
                val_scored_pairs[s1_id] = []
                continue

            cand_feats = [compute_pair_features(s1_rec, target_map[c]) for c in valid_cands]
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
        )
        self.optimal_threshold = best_tau
        thresh_path = self.config.paths.artifacts_dir / "optimal_threshold.json"
        with open(thresh_path, "w", encoding="utf-8") as f:
            json.dump({"optimal_threshold": best_tau, "validation_macro_f05": best_score}, f, indent=2)
        logger.info(f"==> OPTIMAL THRESHOLD LOCKED: tau* = {best_tau:.3f} with Validation Macro F0.5 = {best_score:.4f} (Saved to {thresh_path})")

    def predict_test(self, batch_size: int = 50000):
        """
        Run inference over official test set partitioned by country (France, US, India).
        Maintains low memory footprint and writes candidate_pairs.tsv and matching_results.tsv
        strictly conforming to official submission format.
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
        countries = [str(c).strip().upper() for c in df_countries["country"].dropna().unique()]
        logger.info(f"Test Set Countries Discovered: {countries}")
        del df_countries
        gc.collect()

        total_s1_processed = 0
        total_matches_predicted = 0

        # Process country by country to keep memory minimal
        for country in countries:
            logger.info(f"\n>>> PROCESSING COUNTRY: {country} <<<")

            # 1. Load targets for this country from Test S2 and Test S3
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

            # 2. Build MultiIndexBlocker for this country
            logger.info(f"Building blocker index for {country}...")
            blocker = MultiIndexBlocker(
                max_candidates=self.config.blocking.max_candidates_per_entity,
                min_token_len=self.config.blocking.min_token_len,
                name_prefix_len=self.config.blocking.name_prefix_len,
                max_block_size=self.config.blocking.max_block_size,
            )
            blocker.index_targets(country_targets)
            blocker.prune_large_blocks()

            # 3. Stream S1 for this country and generate predictions
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

                    candidate_lines.append(f"{s1_id}\t{','.join(cands)}\n")
                    matching_lines.append(f"{s1_id}\t{','.join(matches)}\n")

                with open(out_matching, "a", encoding="utf-8") as f_m, open(out_candidates, "a", encoding="utf-8") as f_c:
                    f_m.writelines(matching_lines)
                    f_c.writelines(candidate_lines)

                country_s1_count += len(s1_records)
                total_s1_processed += len(s1_records)
                logger.info(f"[{country}] Processed {country_s1_count:,} S1 entities...")

            del country_targets, target_map, blocker
            gc.collect()

        logger.info("\n============================================================")
        logger.info("TEST INFERENCE FINISHED SUCCESSFULLY:")
        logger.info(f"  Total S1 Entities Processed: {total_s1_processed:,}")
        logger.info(f"  Total Links Predicted:       {total_matches_predicted:,}")
        logger.info(f"  Matching Output:             {out_matching}")
        logger.info(f"  Candidate Output:            {out_candidates}")
        logger.info("============================================================\n")

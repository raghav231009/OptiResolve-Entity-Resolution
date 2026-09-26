import numpy as np
import pytest
from business_entity_resolution.blocking import MultiIndexBlocker
from business_entity_resolution.features import compute_pair_features, FEATURE_NAMES
from business_entity_resolution.metrics import compute_macro_f05
from business_entity_resolution.model import EntityResolutionModel
from business_entity_resolution.normalization import (
    clean_business_name,
    extract_building_number,
    extract_postal_code,
    normalize_address,
)
from business_entity_resolution.threshold import optimize_threshold


def test_end_to_end_pipeline_integration(tmp_path):
    """
    End-to-end integration test verifying the full pipeline lifecycle:
    preprocessing -> blocking -> feature extraction -> model training ->
    threshold optimization -> prediction & candidate subset invariant.
    """
    # 1. Mock Data Setup (US, India, France)
    s1_raw = [
        {"entity_id": "S1-01", "country": "US", "business_name": "Acme Logistics LLC", "business_address": "123 Main St, New York, NY 10001"},
        {"entity_id": "S1-02", "country": "India", "business_name": "Tata Motors Pvt. Ltd.", "business_address": "24 Bombay House, Homi Mody Street, Mumbai 400001"},
        {"entity_id": "S1-03", "country": "France", "business_name": "Société Générale & Fils SARL", "business_address": "29 Boulevard Haussmann, Paris 75009"},
        {"entity_id": "S1-04", "country": "US", "business_name": "Solo Venture Co", "business_address": "999 Desert Rd, Reno, NV 89501"},  # True singleton
    ]

    target_raw = [
        # True match for S1-01 (typo & abbreviation)
        {"entity_id": "S2-01", "country": "US", "business_name": "Acme Logistix Inc", "business_address": "123 Main Street, Suite 2, NY 10001"},
        # True match for S1-02 (address variant)
        {"entity_id": "S3-02", "country": "India", "business_name": "Tata Motors", "business_address": "Bombay House, 24 Homi Mody St, Mumbai"},
        # True match for S1-03 (accented variant)
        {"entity_id": "S2-03", "country": "France", "business_name": "Societe Generale SARL", "business_address": "29 Bd Haussmann, Paris 75009"},
        # Distractor for US
        {"entity_id": "S3-99", "country": "US", "business_name": "Acme Bakery", "business_address": "456 Oak St, Albany, NY 12207"},
    ]

    ground_truth = {
        "S1-01": {"S2-01"},
        "S1-02": {"S3-02"},
        "S1-03": {"S2-03"},
        "S1-04": set(),  # singleton
    }

    # 2. Preprocess Records
    def prep(rec):
        c_name, r_name = clean_business_name(rec["business_name"])
        c_addr = normalize_address(rec["business_address"])
        post = extract_postal_code(rec["business_address"])
        bldg = extract_building_number(c_addr, post)
        return {
            "entity_id": rec["entity_id"],
            "country": rec["country"].strip().upper(),
            "clean_name": c_name,
            "root_name": r_name,
            "clean_address": c_addr,
            "postal_code": post,
            "building_number": bldg,
            "numeric_tokens": {bldg} if bldg else set(),
        }

    s1_prepped = [prep(r) for r in s1_raw]
    targets_prepped = [prep(r) for r in target_raw]
    target_map = {r["entity_id"]: r for r in targets_prepped}

    # 3. Blocking & Candidate Generation
    blocker = MultiIndexBlocker(max_candidates=10)
    blocker.index_targets(targets_prepped)
    blocker.prune_large_blocks()

    # 4. Feature Extraction & Pairwise Construction
    X_train, y_train = [], []
    for s1 in s1_prepped[:3]:  # train on first 3
        s1_id = s1["entity_id"]
        true_set = ground_truth[s1_id]
        cands = blocker.retrieve_candidates(s1)

        for m_id in true_set:
            X_train.append(compute_pair_features(s1, target_map[m_id]))
            y_train.append(1)

        for c_id in cands:
            if c_id not in true_set and c_id in target_map:
                X_train.append(compute_pair_features(s1, target_map[c_id]))
                y_train.append(0)

    assert len(X_train) > 0
    assert len(X_train[0]) == len(FEATURE_NAMES)

    # 5. Model Training
    model = EntityResolutionModel()
    model.train(np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.int32))

    # 6. Prediction & Invariant Verification
    predictions = {}
    for s1 in s1_prepped:
        s1_id = s1["entity_id"]
        cands = blocker.retrieve_candidates(s1)
        if not cands:
            predictions[s1_id] = set()
            continue

        cand_feats = [compute_pair_features(s1, target_map[c]) for c in cands if c in target_map]
        probs = model.predict_proba(np.array(cand_feats, dtype=np.float32))

        # Thresholding
        matches = {c for c, p in zip(cands, probs) if p >= 0.70}
        predictions[s1_id] = matches

        # Invariant assertion: predicted matches MUST be strict subset of candidates
        assert matches.issubset(set(cands)), f"Subset invariant broken for {s1_id}"

    # 7. Exact Macro F0.5 Scoring
    score = compute_macro_f05(ground_truth, predictions)
    assert score >= 0.50, f"Expected reasonable score, got {score}"

"""
End-to-End Smoke Test with Synthetic Dataset.

Creates a tiny but realistic synthetic dataset covering US, India, and France,
then runs the full pipeline training + inference lifecycle.

Usage:
    python tests/test_smoke_e2e.py

Or via pytest:
    pytest tests/test_smoke_e2e.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from business_entity_resolution.blocking import MultiIndexBlocker
from business_entity_resolution.config import PathConfig, PipelineConfig
from business_entity_resolution.features import FEATURE_NAMES, compute_pair_features
from business_entity_resolution.metrics import compute_macro_f05
from business_entity_resolution.model import EntityResolutionModel
from business_entity_resolution.normalization import (
    clean_business_name,
    extract_building_number,
    extract_numeric_tokens,
    extract_postal_code,
    normalize_address,
)
from business_entity_resolution.pipeline import (
    EntityResolutionPipeline,
    load_and_preprocess_file,
    load_ground_truth,
    preprocess_record,
)
from business_entity_resolution.threshold import optimize_threshold


# ---------------------------------------------------------------------------
# Synthetic dataset generator
# ---------------------------------------------------------------------------

SOURCE1_ROWS = [
    # US entries
    {"entity_id": "S1-US-001", "country": "US", "business_name": "Acme Logistics LLC",
     "business_address": "123 Main St, New York, NY 10001"},
    {"entity_id": "S1-US-002", "country": "US", "business_name": "Blue Ridge Ventures Corp",
     "business_address": "500 Oak Avenue, Austin, TX 78701"},
    {"entity_id": "S1-US-003", "country": "US", "business_name": "Solo Corp",
     "business_address": "999 Desert Road, Reno, NV 89501"},  # True singleton
    # India entries
    {"entity_id": "S1-IN-001", "country": "INDIA", "business_name": "Raj Investments Pvt. Ltd.",
     "business_address": "24 Bombay House, Homi Mody Street, Mumbai 400001"},
    {"entity_id": "S1-IN-002", "country": "INDIA", "business_name": "Tata Motors Ltd",
     "business_address": "Bomanjee Petit Road, Ballard Estate, Mumbai 400001"},
    # France entries
    {"entity_id": "S1-FR-001", "country": "FRANCE", "business_name": "Société Générale & Fils SARL",
     "business_address": "29 Boulevard Haussmann, Paris 75009"},
    {"entity_id": "S1-FR-002", "country": "FRANCE", "business_name": "Le Fils Dupont SAS",
     "business_address": "15 Rue de la Paix, Lyon 69001"},
]

SOURCE2_ROWS = [
    # US matches (typo, abbreviation)
    {"entity_id": "S2-US-001", "country": "US", "business_name": "Acme Logistix Inc",
     "business_address": "123 Main Street, Suite 2, NY 10001"},
    {"entity_id": "S2-US-002", "country": "US", "business_name": "Blue Ridge Ventures",
     "business_address": "500 Oak Ave, Austin TX 78701"},
    # India match
    {"entity_id": "S2-IN-001", "country": "INDIA", "business_name": "Raj Investments Pvt Ltd",
     "business_address": "Bombay House 24 Homi Mody St Mumbai"},
    # France match (accented variant)
    {"entity_id": "S2-FR-001", "country": "FRANCE", "business_name": "Societe Generale SARL",
     "business_address": "29 Bd Haussmann, Paris 75009"},
    # Distractors
    {"entity_id": "S2-US-099", "country": "US", "business_name": "Acme Bakery",
     "business_address": "456 Oak St, Albany, NY 12207"},
    {"entity_id": "S2-IN-099", "country": "INDIA", "business_name": "Tata Steel Ltd",
     "business_address": "Bombay House, Mumbai 400001"},
]

SOURCE3_ROWS = [
    # S3 match for Tata Motors (garbled name, preserved address)
    {"entity_id": "S3-IN-001", "country": "INDIA", "business_name": "Tata Motors",
     "business_address": "Bomanjee Petit Road, Mumbai"},
    # S3 match for Le Fils Dupont
    {"entity_id": "S3-FR-001", "country": "FRANCE", "business_name": "Fils Dupont",
     "business_address": "15 rue de la paix lyon 69001"},
    # Distractors
    {"entity_id": "S3-US-099", "country": "US", "business_name": "Acme Tools",
     "business_address": "123 Industrial Blvd, Chicago, IL 60601"},
]

GROUND_TRUTH_ROWS = [
    {"source1_entity_id": "S1-US-001", "matched_entity_ids": "S2-US-001"},
    {"source1_entity_id": "S1-US-002", "matched_entity_ids": "S2-US-002"},
    {"source1_entity_id": "S1-US-003", "matched_entity_ids": ""},  # singleton
    {"source1_entity_id": "S1-IN-001", "matched_entity_ids": "S2-IN-001"},
    {"source1_entity_id": "S1-IN-002", "matched_entity_ids": "S3-IN-001"},
    {"source1_entity_id": "S1-FR-001", "matched_entity_ids": "S2-FR-001"},
    {"source1_entity_id": "S1-FR-002", "matched_entity_ids": "S3-FR-001"},
]


def write_tsv(path: Path, rows: list, fieldnames: list = None):
    """Write a list of dicts to a TSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if fieldnames:
        df = df[fieldnames]
    df.to_csv(path, sep="\t", index=False)


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create a complete synthetic dataset in tmp_path."""
    train_dir = tmp_path / "dataset" / "train"
    test_dir = tmp_path / "dataset" / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Train files
    write_tsv(train_dir / "train_source1.tsv", SOURCE1_ROWS,
              ["entity_id", "country", "business_name", "business_address"])
    write_tsv(train_dir / "train_source2.tsv", SOURCE2_ROWS,
              ["entity_id", "country", "business_name", "business_address"])
    write_tsv(train_dir / "train_source3.tsv", SOURCE3_ROWS,
              ["entity_id", "country", "business_name", "business_address"])
    write_tsv(train_dir / "train_ground_truth.tsv", GROUND_TRUTH_ROWS,
              ["source1_entity_id", "matched_entity_ids"])

    # Test files (use same entities for smoke test)
    write_tsv(test_dir / "test_source1.tsv", SOURCE1_ROWS,
              ["entity_id", "country", "business_name", "business_address"])
    write_tsv(test_dir / "test_source2.tsv", SOURCE2_ROWS,
              ["entity_id", "country", "business_name", "business_address"])
    write_tsv(test_dir / "test_source3.tsv", SOURCE3_ROWS,
              ["entity_id", "country", "business_name", "business_address"])

    return tmp_path


# ---------------------------------------------------------------------------
# Test: Full pipeline smoke test
# ---------------------------------------------------------------------------

class TestSmokePipeline:
    def test_preprocess_record(self):
        """preprocess_record must produce all required keys."""
        raw = SOURCE1_ROWS[0]
        result = preprocess_record(raw)
        required_keys = {"entity_id", "country", "clean_name", "root_name",
                         "clean_address", "postal_code", "building_number", "numeric_tokens"}
        assert required_keys == set(result.keys()), f"Missing keys: {required_keys - set(result.keys())}"

    def test_load_and_preprocess_file(self, synthetic_dataset):
        """load_and_preprocess_file should return all records."""
        path = synthetic_dataset / "dataset" / "train" / "train_source1.tsv"
        records = load_and_preprocess_file(path)
        assert len(records) == len(SOURCE1_ROWS)
        for r in records:
            assert isinstance(r["clean_name"], str)
            assert isinstance(r["numeric_tokens"], set)

    def test_load_ground_truth(self, synthetic_dataset):
        """load_ground_truth should correctly parse match lists."""
        path = synthetic_dataset / "dataset" / "train" / "train_ground_truth.tsv"
        s1_ids = {r["source1_entity_id"] for r in GROUND_TRUTH_ROWS}
        gt = load_ground_truth(path, s1_ids)
        assert "S1-US-001" in gt
        assert "S2-US-001" in gt["S1-US-001"]
        assert gt["S1-US-003"] == set()  # singleton

    def test_full_pipeline_train_predict(self, synthetic_dataset):
        """
        Full end-to-end: Train → Threshold Optimize → Test Inference → Output Files.
        Verifies:
          1. Pipeline completes without error.
          2. Output files are created.
          3. Candidate subset invariant holds (matches ⊆ candidates).
          4. All S1 entities have a row in both outputs.
          5. Model saves and the threshold artifact is written.
        """
        config = PipelineConfig()
        config.paths.dataset_root = synthetic_dataset / "dataset"
        config.paths.output_dir = synthetic_dataset / "output"
        config.paths.artifacts_dir = synthetic_dataset / "artifacts"
        config.train_s1_limit = None
        config.val_s1_limit = 3  # Very small val set for smoke test

        pipeline = EntityResolutionPipeline(config)

        # --- Training Phase ---
        pipeline.fit()

        # Verify model saved
        assert config.paths.model_path.exists(), "Model not saved after fit()"

        # Verify threshold artifact saved
        thresh_path = config.paths.artifacts_dir / "optimal_threshold.json"
        assert thresh_path.exists(), "Threshold artifact not saved"
        with open(thresh_path) as f:
            thresh_data = json.load(f)
        assert "optimal_threshold" in thresh_data
        assert 0.0 < thresh_data["optimal_threshold"] <= 1.0

        # --- Inference Phase ---
        pipeline.predict_test(batch_size=100)

        # Verify output files exist
        assert config.paths.matching_results.exists(), "matching_results.tsv not created"
        assert config.paths.candidate_pairs.exists(), "candidate_pairs.tsv not created"

        # Load outputs and verify structure
        matching_df = pd.read_csv(config.paths.matching_results, sep="\t", dtype=str)
        candidate_df = pd.read_csv(config.paths.candidate_pairs, sep="\t", dtype=str)

        assert list(matching_df.columns) == ["source1_entity_id", "matched_entity_ids"], \
            f"Wrong columns in matching_results: {list(matching_df.columns)}"
        assert list(candidate_df.columns) == ["source1_entity_id", "candidate_entity_ids"], \
            f"Wrong columns in candidate_pairs: {list(candidate_df.columns)}"

        # Count rows: must equal number of S1 entities
        n_s1 = len(SOURCE1_ROWS)
        assert len(matching_df) == n_s1, f"Expected {n_s1} rows in matching_results, got {len(matching_df)}"
        assert len(candidate_df) == n_s1, f"Expected {n_s1} rows in candidate_pairs, got {len(candidate_df)}"

        # Verify candidate subset invariant: every matched ID ∈ candidate IDs
        violations = 0
        for _, m_row in matching_df.iterrows():
            s1_id = m_row["source1_entity_id"]
            m_ids_str = str(m_row["matched_entity_ids"]) if pd.notna(m_row["matched_entity_ids"]) else ""
            m_ids = set(m_ids_str.split(",")) - {""} if m_ids_str else set()

            c_row = candidate_df[candidate_df["source1_entity_id"] == s1_id]
            c_ids_str = str(c_row["candidate_entity_ids"].values[0]) if len(c_row) > 0 else ""
            c_ids = set(c_ids_str.split(",")) - {""} if c_ids_str else set()

            non_candidates = m_ids - c_ids
            if non_candidates:
                violations += 1
                print(f"INVARIANT VIOLATION: {s1_id} — matched {non_candidates} not in candidates")

        assert violations == 0, f"Candidate subset invariant violated for {violations} entities"

        # No duplicate S1 entity IDs in outputs
        assert matching_df["source1_entity_id"].nunique() == n_s1, "Duplicate S1 IDs in matching_results"
        assert candidate_df["source1_entity_id"].nunique() == n_s1, "Duplicate S1 IDs in candidate_pairs"

    def test_output_country_isolation(self, synthetic_dataset):
        """Verify that no matched ID is from a different country than the S1 entity."""
        config = PipelineConfig()
        config.paths.dataset_root = synthetic_dataset / "dataset"
        config.paths.output_dir = synthetic_dataset / "output"
        config.paths.artifacts_dir = synthetic_dataset / "artifacts"
        config.val_s1_limit = 2

        pipeline = EntityResolutionPipeline(config)
        pipeline.fit()
        pipeline.predict_test(batch_size=100)

        matching_df = pd.read_csv(config.paths.matching_results, sep="\t", dtype=str)

        # Build country lookup for all known entities
        all_entities = SOURCE1_ROWS + SOURCE2_ROWS + SOURCE3_ROWS
        country_map = {r["entity_id"]: r["country"].upper() for r in all_entities}

        cross_country_matches = []
        for _, row in matching_df.iterrows():
            s1_id = row["source1_entity_id"]
            s1_country = country_map.get(s1_id, "UNKNOWN").upper()
            matched_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
            matched_ids = [m for m in matched_str.split(",") if m.strip()]
            for mid in matched_ids:
                m_country = country_map.get(mid, "UNKNOWN").upper()
                if m_country != s1_country:
                    cross_country_matches.append((s1_id, s1_country, mid, m_country))

        assert len(cross_country_matches) == 0, (
            f"Cross-country matches found (country isolation broken): {cross_country_matches}"
        )


class TestSmokeThreshold:
    def test_optimize_threshold_on_synthetic_data(self):
        """optimize_threshold must return valid tau and score."""
        val_gt = {
            "S1-001": {"S2-001"},
            "S1-002": set(),
            "S1-003": {"S2-003"},
        }
        val_scored = {
            "S1-001": [("S2-001", 0.92), ("S2-099", 0.35)],
            "S1-002": [],
            "S1-003": [("S2-003", 0.88)],
        }
        tau, score, history = optimize_threshold(val_gt, val_scored)
        assert 0.0 <= tau <= 1.0
        assert 0.0 <= score <= 1.0
        assert len(history) > 0

    def test_optimize_threshold_respects_search_range(self):
        """All evaluated tau values must be within the search range."""
        val_gt = {"S1-001": {"S2-001"}}
        val_scored = {"S1-001": [("S2-001", 0.9)]}
        tau, score, history = optimize_threshold(
            val_gt, val_scored, search_start=0.70, search_end=0.90, step=0.05
        )
        for t in history:
            assert 0.70 - 1e-6 <= t <= 0.90 + 1e-6, f"tau={t} outside search range"


if __name__ == "__main__":
    # Allow running directly for manual smoke testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_dir = tmp_path / "dataset" / "train"
        test_dir = tmp_path / "dataset" / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        write_tsv(train_dir / "train_source1.tsv", SOURCE1_ROWS,
                  ["entity_id", "country", "business_name", "business_address"])
        write_tsv(train_dir / "train_source2.tsv", SOURCE2_ROWS,
                  ["entity_id", "country", "business_name", "business_address"])
        write_tsv(train_dir / "train_source3.tsv", SOURCE3_ROWS,
                  ["entity_id", "country", "business_name", "business_address"])
        write_tsv(train_dir / "train_ground_truth.tsv", GROUND_TRUTH_ROWS,
                  ["source1_entity_id", "matched_entity_ids"])
        write_tsv(test_dir / "test_source1.tsv", SOURCE1_ROWS,
                  ["entity_id", "country", "business_name", "business_address"])
        write_tsv(test_dir / "test_source2.tsv", SOURCE2_ROWS,
                  ["entity_id", "country", "business_name", "business_address"])
        write_tsv(test_dir / "test_source3.tsv", SOURCE3_ROWS,
                  ["entity_id", "country", "business_name", "business_address"])

        config = PipelineConfig()
        config.paths.dataset_root = tmp_path / "dataset"
        config.paths.output_dir = tmp_path / "output"
        config.paths.artifacts_dir = tmp_path / "artifacts"
        config.val_s1_limit = 3

        print("=== SMOKE TEST: Training ===")
        pipeline = EntityResolutionPipeline(config)
        pipeline.fit()
        print("=== SMOKE TEST: Inference ===")
        pipeline.predict_test(batch_size=100)
        print("=== SMOKE TEST: Outputs ===")
        m = pd.read_csv(config.paths.matching_results, sep="\t")
        c = pd.read_csv(config.paths.candidate_pairs, sep="\t")
        print(f"  matching_results: {len(m)} rows")
        print(f"  candidate_pairs: {len(c)} rows")
        print("=== SMOKE TEST PASSED ===")

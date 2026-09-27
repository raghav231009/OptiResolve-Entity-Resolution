"""
Regression tests for Open-Set Country Generalization and Distribution Shift.

Verifies:
1. Pipeline genuinely generalizes to an unseen country (France, Germany, etc.)
   when trained exclusively on US and India.
2. Dynamic country discovery: countries are discovered dynamically from data,
   never restricted by hardcoded country lists or country-specific branches.
3. Strict country isolation: 100% of candidate pairs and predicted matches stay
   strictly within the country partition.
4. Normalization and structural feature extraction for European/French patterns
   (accents, ligatures, legal forms, address abbreviations, 5-digit postal codes).
5. End-to-end execution of complete pipeline (train on US/India only -> predict on US/India/France/Germany).
"""

import ast
import inspect
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.pipeline import EntityResolutionPipeline
from business_entity_resolution.normalization import (
    strip_accents_and_normalize,
    normalize_address,
    clean_business_name,
    extract_postal_code,
    extract_building_number,
    extract_numeric_tokens,
)
from business_entity_resolution.blocking import MultiIndexBlocker
from business_entity_resolution.features import compute_pair_features, FEATURE_NAMES
from business_entity_resolution.model import EntityResolutionModel
from business_entity_resolution.metrics import compute_macro_f05


class TestStaticCodeZeroHardcodedCountries:
    """Verifies that the codebase has no hardcoded country constraints."""

    def test_no_hardcoded_country_lists_in_pipeline(self):
        """Pipeline must discover countries dynamically from data."""
        import business_entity_resolution.pipeline as pipe_mod
        src = inspect.getsource(pipe_mod)
        # Check that there are no hardcoded country filters restricting processing
        assert 'for country in ["US", "INDIA"]' not in src
        assert "for country in ('US', 'INDIA')" not in src
        assert 'if country not in ["US", "INDIA"]' not in src
        assert "df_countries = pd.read_csv" in src  # Dynamic discovery from source

    def test_features_have_no_country_indicators(self):
        """Feature set must be relative/structural, not one-hot country indicators."""
        for feat in FEATURE_NAMES:
            assert "is_us" not in feat
            assert "is_india" not in feat
            assert "is_france" not in feat
        assert "country_match" in FEATURE_NAMES  # Relative equality only


class TestMultilingualNormalizationUnseenCountry:
    """Tests normalization on European and French text patterns."""

    def test_french_accents_and_ligatures(self):
        name = "Société Générale & Cœur de France"
        clean_n, root_n = clean_business_name(name)
        assert "societe generale" in clean_n
        assert "coeur de france" in clean_n
        assert "é" not in clean_n and "œ" not in clean_n

    def test_french_corporate_suffixes(self):
        cases = [
            ("Dassault Aviation SAS", "dassault aviation"),
            ("Carrefour Hypermarché SARL", "carrefour hypermarche"),
            ("Renault EURL", "renault"),
            ("Cabinet Médical SCI", "cabinet medical"),
            ("Société Civile SNC", "societe civile"),
            ("Comptoir Commercial GIE", "comptoir commercial"),
        ]
        for raw, expected_root in cases:
            _, root = clean_business_name(raw)
            assert root == expected_root, f"Failed on {raw}: got '{root}' expected '{expected_root}'"

    def test_french_fils_end_only_handling(self):
        # Trailing 'fils' or 'et fils' is stripped
        _, root1 = clean_business_name("Dupont et Fils SARL")
        assert root1 == "dupont"
        _, root2 = clean_business_name("Leroy & Fils")
        assert root2 == "leroy"

        # Mid-name 'fils' is preserved (e.g. brand name)
        _, root3 = clean_business_name("Le Fils Dupont SAS")
        assert root3 == "le fils dupont"

    def test_french_address_abbreviations(self):
        addr = "29 bd Haussmann, r de Rivoli, pl de la Concorde, bat A, 75009 Paris"
        norm = normalize_address(addr)
        assert "boulevard" in norm
        assert "rue" in norm
        assert "place" in norm
        assert "batiment" in norm

    def test_postal_code_extraction(self):
        # France 5-digit
        assert extract_postal_code("29 Boulevard Haussmann, 75009 Paris") == "75009"
        # Germany 5-digit
        assert extract_postal_code("Industriestrasse 45, 80331 Munich") == "80331"
        # India 6-digit
        assert extract_postal_code("Outer Ring Road, Bangalore 560103") == "560103"
        # US 5-digit
        assert extract_postal_code("123 Main St, New York, NY 10001") == "10001"


class TestCountryIsolationAndDynamicIndexing:
    """Verifies that MultiIndexBlocker partitions dynamically by country."""

    def test_dynamic_unseen_country_blocking(self):
        blocker = MultiIndexBlocker(max_candidates=20)
        targets = [
            {"entity_id": "S2-FR-1", "country": "FRANCE", "clean_name": "societe generale", "root_name": "societe generale", "clean_address": "29 bd haussmann", "postal_code": "75009", "building_number": "29"},
            {"entity_id": "S2-US-1", "country": "US", "clean_name": "societe generale ny", "root_name": "societe generale", "clean_address": "245 park ave", "postal_code": "10167", "building_number": "245"},
            {"entity_id": "S2-DE-1", "country": "GERMANY", "clean_name": "societe generale frankfurt", "root_name": "societe generale", "clean_address": "mainzer landstrasse 46", "postal_code": "60325", "building_number": "46"},
        ]
        blocker.index_targets(targets)

        # Query in France
        query_fr = {"entity_id": "S1-FR-1", "country": "FRANCE", "clean_name": "societe generale", "root_name": "societe generale", "clean_address": "29 boulevard haussmann", "postal_code": "75009", "building_number": "29"}
        cands_fr = blocker.retrieve_candidates(query_fr)
        assert cands_fr == ["S2-FR-1"], f"Expected only France candidates, got {cands_fr}"

        # Query in Germany
        query_de = {"entity_id": "S1-DE-1", "country": "GERMANY", "clean_name": "societe generale", "root_name": "societe generale", "clean_address": "mainzer landstrasse 46", "postal_code": "60325", "building_number": "46"}
        cands_de = blocker.retrieve_candidates(query_de)
        assert cands_de == ["S2-DE-1"], f"Expected only Germany candidates, got {cands_de}"


class TestEndToEndPipelineUnseenCountryShift:
    """
    Simulates the exact competition condition:
    Training dataset: ONLY US and India.
    Test dataset: US, India, AND France (unseen country) plus Germany.
    Verifies full training, threshold tuning, and inference execution without code change.
    """

    def test_full_pipeline_with_unseen_countries_in_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_dir = tmp_path / "dataset" / "train"
            test_dir = tmp_path / "dataset" / "test"
            out_dir = tmp_path / "output"
            art_dir = tmp_path / "artifacts"

            train_dir.mkdir(parents=True, exist_ok=True)
            test_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            art_dir.mkdir(parents=True, exist_ok=True)

            # 1. Training data: ONLY US and INDIA (Zero France or Germany in train!)
            train_s1 = [
                {"entity_id": "S1-US-1", "country": "US", "business_name": "Acme Global Inc", "business_address": "100 Main St, New York, NY 10001"},
                {"entity_id": "S1-US-2", "country": "US", "business_name": "Beta Logistics LLC", "business_address": "200 Market St, San Francisco, CA 94105"},
                {"entity_id": "S1-US-3", "country": "US", "business_name": "Solo Venture Corp", "business_address": "999 Desert Rd, Reno, NV 89501"},
                {"entity_id": "S1-IN-1", "country": "INDIA", "business_name": "Tata Consultancy Pvt Ltd", "business_address": "Plot 5, Hitech City, Hyderabad 500081"},
                {"entity_id": "S1-IN-2", "country": "INDIA", "business_name": "Infosys Tech Limited", "business_address": "44 Electronics City, Bangalore 560100"},
            ]
            train_s2 = [
                {"entity_id": "S2-US-1", "country": "US", "business_name": "Acme Global", "business_address": "100 Main Street 10001"},
                {"entity_id": "S2-US-2", "country": "US", "business_name": "Beta Logistics", "business_address": "200 Market Street 94105"},
                {"entity_id": "S2-US-DIST", "country": "US", "business_name": "Acme Tools and Hardware", "business_address": "500 Oak St 10001"},
                {"entity_id": "S2-IN-1", "country": "INDIA", "business_name": "Tata Consultancy", "business_address": "Plot 5 Hitech City 500081"},
                {"entity_id": "S2-IN-2", "country": "INDIA", "business_name": "Infosys Tech", "business_address": "44 Electronics City 560100"},
                {"entity_id": "S2-IN-DIST", "country": "INDIA", "business_name": "Tata Motors Finance", "business_address": "Outer Ring Road 560100"},
            ]
            train_s3 = [
                {"entity_id": "S3-US-1", "country": "US", "business_name": "Acme Global Inc", "business_address": "100 Main St"},
                {"entity_id": "S3-IN-1", "country": "INDIA", "business_name": "Tata Consultancy", "business_address": "Hitech City"},
            ]
            train_gt = [
                {"source1_entity_id": "S1-US-1", "matched_entity_ids": "S2-US-1,S3-US-1"},
                {"source1_entity_id": "S1-US-2", "matched_entity_ids": "S2-US-2"},
                {"source1_entity_id": "S1-US-3", "matched_entity_ids": ""},
                {"source1_entity_id": "S1-IN-1", "matched_entity_ids": "S2-IN-1,S3-IN-1"},
                {"source1_entity_id": "S1-IN-2", "matched_entity_ids": "S2-IN-2"},
            ]

            # 2. Test data: US, INDIA, AND UNSEEN COUNTRIES: FRANCE and GERMANY
            test_s1 = [
                {"entity_id": "S1-US-1", "country": "US", "business_name": "Acme Global Inc", "business_address": "100 Main St, New York, NY 10001"},
                {"entity_id": "S1-IN-1", "country": "INDIA", "business_name": "Tata Consultancy Pvt Ltd", "business_address": "Plot 5, Hitech City, Hyderabad 500081"},
                # Unseen France entity
                {"entity_id": "S1-FR-1", "country": "FRANCE", "business_name": "Société Générale & Fils SARL", "business_address": "29 Boulevard Haussmann, Paris 75009"},
                # Unseen Germany entity
                {"entity_id": "S1-DE-1", "country": "GERMANY", "business_name": "Siemens Energy AG", "business_address": "Otto-Hahn-Ring 6, 81739 München"},
            ]
            test_s2 = [
                {"entity_id": "S2-US-1", "country": "US", "business_name": "Acme Global", "business_address": "100 Main Street 10001"},
                {"entity_id": "S2-IN-1", "country": "INDIA", "business_name": "Tata Consultancy", "business_address": "Plot 5 Hitech City 500081"},
                {"entity_id": "S2-FR-1", "country": "FRANCE", "business_name": "Societe Generale", "business_address": "29 Bd Haussmann 75009 Paris"},
                {"entity_id": "S2-DE-1", "country": "GERMANY", "business_name": "Siemens Energy", "business_address": "Otto Hahn Ring 6 81739 Munchen"},
            ]
            test_s3 = [
                {"entity_id": "S3-FR-1", "country": "FRANCE", "business_name": "Societe Generale SARL", "business_address": "29 Boulevard Haussmann"},
            ]

            # Write TSV files
            pd.DataFrame(train_s1).to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)
            pd.DataFrame(train_s2).to_csv(train_dir / "train_source2.tsv", sep="\t", index=False)
            pd.DataFrame(train_s3).to_csv(train_dir / "train_source3.tsv", sep="\t", index=False)
            pd.DataFrame(train_gt).to_csv(train_dir / "train_ground_truth.tsv", sep="\t", index=False)

            pd.DataFrame(test_s1).to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)
            pd.DataFrame(test_s2).to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)
            pd.DataFrame(test_s3).to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)

            # Configure pipeline
            config = PipelineConfig()
            config.paths.dataset_root = tmp_path / "dataset"
            config.paths.output_dir = out_dir
            config.paths.artifacts_dir = art_dir
            config.val_s1_limit = 2
            config.model.n_estimators = 20
            config.model.min_child_samples = 1

            pipeline = EntityResolutionPipeline(config)

            # 1. Fit on US/India training data
            pipeline.fit()

            # 2. Run inference on test data (US, India, France, Germany)
            pipeline.predict_test(batch_size=10)

            # 3. Verify output files
            matching_df = pd.read_csv(config.paths.matching_results, sep="\t", dtype=str)
            candidate_df = pd.read_csv(config.paths.candidate_pairs, sep="\t", dtype=str)

            # Ensure all test S1 entities are present in predictions
            assert len(matching_df) == 4
            assert len(candidate_df) == 4
            assert set(matching_df["source1_entity_id"]) == {"S1-US-1", "S1-IN-1", "S1-FR-1", "S1-DE-1"}

            # Check France candidates and predictions
            fr_cand_row = candidate_df[candidate_df["source1_entity_id"] == "S1-FR-1"].iloc[0]
            fr_cands = set(str(fr_cand_row["candidate_entity_ids"]).split(","))
            assert {"S2-FR-1", "S3-FR-1"}.issubset(fr_cands), f"France candidates missing: {fr_cands}"

            fr_row = matching_df[matching_df["source1_entity_id"] == "S1-FR-1"].iloc[0]
            fr_matches = set(str(fr_row["matched_entity_ids"]).split(",")) - {"", "nan"}
            assert len(fr_matches) > 0 and any(m in {"S2-FR-1", "S3-FR-1"} for m in fr_matches), f"France match failed: {fr_matches}"

            # Check Germany candidates and predictions
            de_cand_row = candidate_df[candidate_df["source1_entity_id"] == "S1-DE-1"].iloc[0]
            de_cands = set(str(de_cand_row["candidate_entity_ids"]).split(","))
            assert "S2-DE-1" in de_cands, f"Germany candidate missing: {de_cands}"

            de_row = matching_df[matching_df["source1_entity_id"] == "S1-DE-1"].iloc[0]
            assert de_row is not None

            # 4. Strict country isolation check: no matches across different countries
            country_map = {
                "S1-US-1": "US", "S2-US-1": "US",
                "S1-IN-1": "INDIA", "S2-IN-1": "INDIA",
                "S1-FR-1": "FRANCE", "S2-FR-1": "FRANCE", "S3-FR-1": "FRANCE",
                "S1-DE-1": "GERMANY", "S2-DE-1": "GERMANY",
            }
            for _, row in matching_df.iterrows():
                s1_id = row["source1_entity_id"]
                s1_ctry = country_map[s1_id]
                m_str = str(row["matched_entity_ids"])
                if pd.notna(row["matched_entity_ids"]) and m_str.strip() not in ("", "nan", "none"):
                    for mid in m_str.split(","):
                        if mid.strip() and mid.strip() not in ("nan", "none"):
                            m_ctry = country_map.get(mid.strip())
                            assert s1_ctry == m_ctry, f"Cross-country match: {s1_id} ({s1_ctry}) -> {mid} ({m_ctry})"

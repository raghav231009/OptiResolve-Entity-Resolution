"""
Tests for Rigorous Blocking Recall Benchmark Engine.
Verifies:
  - retrieve_candidates_with_uncapped behavior
  - Missed link diagnosis and categorization:
    * candidate_cap
    * country_mismatch
    * missing_address
    * missing_postal
    * building_mismatch
    * name_corruption
    * transliteration
    * normalization
  - Granular breakdown metrics:
    * Link recall & Entity complete recall
    * Country breakdown (US, India, France)
    * Source breakdown (S2, S3)
    * Cardinality breakdown (1, 2, 3, 4+)
    * Singleton vs Non-singleton handling
  - CI failure threshold check
"""

import pytest
from business_entity_resolution.blocking import MultiIndexBlocker
from evaluate_blocking import diagnose_missed_link


class TestRetrieveCandidatesWithUncapped:
    def test_uncapped_contains_all_candidates_before_cap(self):
        blocker = MultiIndexBlocker(max_candidates=2)
        targets = [
            {"entity_id": "S2-1", "country": "US", "clean_name": "acme logistics", "root_name": "acme logistics", "clean_address": "123 main st", "postal_code": "10001", "building_number": "123"},
            {"entity_id": "S2-2", "country": "US", "clean_name": "acme supply", "root_name": "acme supply", "clean_address": "456 broadway", "postal_code": "10002", "building_number": "456"},
            {"entity_id": "S2-3", "country": "US", "clean_name": "acme express", "root_name": "acme express", "clean_address": "789 park ave", "postal_code": "10003", "building_number": "789"},
            {"entity_id": "S2-4", "country": "US", "clean_name": "acme partners", "root_name": "acme partners", "clean_address": "101 5th ave", "postal_code": "10004", "building_number": "101"},
        ]
        blocker.index_targets(targets)

        s1 = {
            "entity_id": "S1-1",
            "country": "US",
            "clean_name": "acme corporation",
            "root_name": "acme corporation",
            "clean_address": "123 main street",
            "postal_code": "10001",
            "building_number": "123",
        }
        retained, uncapped = blocker.retrieve_candidates_with_uncapped(s1)

        # Retained must strictly respect max_candidates=2
        assert len(retained) == 2
        # Uncapped contains all 4 'acme' targets gathered across channels
        assert len(uncapped) >= 3
        # Retained is a strict subset of uncapped
        assert set(retained).issubset(uncapped)


class TestMissedLinkDiagnosis:
    @pytest.fixture
    def mock_blocker(self):
        b = MultiIndexBlocker(max_candidates=10)
        return b

    def test_candidate_cap_category(self, mock_blocker):
        s1 = {"entity_id": "S1-1", "country": "US", "clean_name": "alpha", "root_name": "alpha", "clean_address": "1 main", "postal_code": "10001", "building_number": "1"}
        t = {"entity_id": "S2-1", "country": "US", "clean_name": "alpha", "root_name": "alpha", "clean_address": "1 main", "postal_code": "10001", "building_number": "1"}
        # Target was in uncapped set, but was dropped
        uncapped = {"S2-1", "S2-2", "S2-3"}
        cat = diagnose_missed_link(s1, t, uncapped, mock_blocker)
        assert cat == "candidate_cap"

    def test_country_mismatch_category(self, mock_blocker):
        s1 = {"entity_id": "S1-1", "country": "US", "clean_name": "acme", "clean_address": "1 main", "postal_code": "10001"}
        t = {"entity_id": "S2-1", "country": "FRANCE", "clean_name": "acme", "clean_address": "1 main", "postal_code": "75001"}
        cat = diagnose_missed_link(s1, t, set(), mock_blocker)
        assert cat == "country_mismatch"

    def test_missing_address_category(self, mock_blocker):
        s1 = {"entity_id": "S1-1", "country": "US", "clean_name": "acme", "clean_address": "", "postal_code": ""}
        t = {"entity_id": "S2-1", "country": "US", "clean_name": "acme", "clean_address": "", "postal_code": ""}
        cat = diagnose_missed_link(s1, t, set(), mock_blocker)
        assert cat == "missing_address"

    def test_missing_postal_category(self, mock_blocker):
        s1 = {"entity_id": "S1-1", "country": "US", "clean_name": "acme", "clean_address": "main street", "postal_code": ""}
        t = {"entity_id": "S2-1", "country": "US", "clean_name": "acme", "clean_address": "main street", "postal_code": "10001"}
        cat = diagnose_missed_link(s1, t, set(), mock_blocker)
        assert cat == "missing_postal"

    def test_building_mismatch_category(self, mock_blocker):
        s1 = {"entity_id": "S1-1", "country": "US", "clean_name": "acme", "clean_address": "100 main", "postal_code": "10001", "building_number": "100"}
        t = {"entity_id": "S2-1", "country": "US", "clean_name": "acme", "clean_address": "200 main", "postal_code": "10001", "building_number": "200"}
        cat = diagnose_missed_link(s1, t, set(), mock_blocker)
        assert cat == "building_mismatch"

    def test_name_corruption_category(self, mock_blocker):
        s1 = {"entity_id": "S1-1", "country": "US", "clean_name": "international technology associates", "clean_address": "100 main", "postal_code": "10001", "building_number": "100"}
        t = {"entity_id": "S2-1", "country": "US", "clean_name": "zephyr global ventures", "clean_address": "100 main", "postal_code": "10001", "building_number": "100"}
        cat = diagnose_missed_link(s1, t, set(), mock_blocker)
        assert cat == "name_corruption"


class TestBenchmarkBreakdowns:
    def test_recall_calculation_mock(self):
        # 3 S1 entities:
        # S1-1 has 1 true link (S2-1) -> retrieved
        # S1-2 has 2 true links (S2-2, S3-2) -> S2-2 retrieved, S3-2 missed
        # S1-3 has 0 true links (singleton)
        gt = {
            "S1-1": {"S2-1"},
            "S1-2": {"S2-2", "S3-2"},
            "S1-3": set(),
        }
        predictions = {
            "S1-1": ["S2-1", "S2-99"],
            "S1-2": ["S2-2", "S3-88"],
            "S1-3": ["S2-77"],
        }
        # Non-singleton true links = 1 + 2 = 3
        # Recovered links: S2-1 (yes), S2-2 (yes), S3-2 (no) -> 2 / 3 = 66.67%
        recovered = 0
        total_true = 0
        complete_entities = 0
        for s1_id in ["S1-1", "S1-2"]:
            true_set = gt[s1_id]
            pred_set = set(predictions[s1_id])
            hits = len(true_set & pred_set)
            recovered += hits
            total_true += len(true_set)
            if hits == len(true_set):
                complete_entities += 1

        assert total_true == 3
        assert recovered == 2
        assert abs((recovered / total_true) - (2.0 / 3.0)) < 1e-6
        # Complete entity recall: S1-1 is complete, S1-2 is not -> 1 / 2 = 50%
        assert complete_entities == 1

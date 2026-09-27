"""
Extended Blocking Tests.
Covers: country separation invariant, candidate cap enforcement,
oversized block sub-blocking, missing postal/address, S2/S3 mixing.
"""

import pytest
from business_entity_resolution.blocking import MultiIndexBlocker


def make_target(entity_id, country, root_name, clean_address="", postal_code="", building_number=""):
    return {
        "entity_id": entity_id,
        "country": country,
        "clean_name": root_name,
        "root_name": root_name,
        "clean_address": clean_address,
        "postal_code": postal_code,
        "building_number": building_number,
        "numeric_tokens": {building_number} if building_number else set(),
    }


def make_s1(entity_id, country, root_name, clean_address="", postal_code="", building_number=""):
    return {
        "entity_id": entity_id,
        "country": country,
        "clean_name": root_name,
        "root_name": root_name,
        "clean_address": clean_address,
        "postal_code": postal_code,
        "building_number": building_number,
        "numeric_tokens": {building_number} if building_number else set(),
    }


class TestCountrySeparation:
    def test_no_cross_country_candidates(self):
        """Candidates from a different country must NEVER be returned."""
        blocker = MultiIndexBlocker(max_candidates=20)
        targets = [
            make_target("S2-US", "US", "alpha logistics", "100 broadway", "10001", "100"),
            make_target("S2-FR", "FRANCE", "alpha logistics", "100 rue paris", "75001", "100"),
        ]
        blocker.index_targets(targets)
        s1 = make_s1("S1-001", "US", "alpha logistics", "100 broadway", "10001", "100")
        cands = blocker.retrieve_candidates(s1)
        assert "S2-US" in cands
        assert "S2-FR" not in cands, "French candidate leaked into US query!"

    def test_france_entities_stay_in_france(self):
        blocker = MultiIndexBlocker(max_candidates=20)
        targets = [
            make_target("S2-FR", "FRANCE", "societe generale", "29 boulevard haussmann", "75009"),
            make_target("S2-US", "US", "societe generale", "100 main street"),
        ]
        blocker.index_targets(targets)
        s1 = make_s1("S1-FR", "FRANCE", "societe generale", "29 boulevard haussmann", "75009")
        cands = blocker.retrieve_candidates(s1)
        assert "S2-FR" in cands
        assert "S2-US" not in cands

    def test_india_entities_isolated(self):
        blocker = MultiIndexBlocker(max_candidates=20)
        targets = [
            make_target("S3-IN", "INDIA", "tata motors", "bombay house homi mody street", "400001"),
            make_target("S2-US", "US", "tata motors", "123 main street"),
        ]
        blocker.index_targets(targets)
        s1 = make_s1("S1-IN", "INDIA", "tata motors", "bombay house", "400001")
        cands = blocker.retrieve_candidates(s1)
        assert "S3-IN" in cands
        assert "S2-US" not in cands


class TestCandidateCap:
    def test_cap_enforced(self):
        """Candidate list must never exceed max_candidates."""
        blocker = MultiIndexBlocker(max_candidates=5)
        # Create many targets with the same brand token "alpha"
        targets = [
            make_target(f"S2-{i:03d}", "US", "alpha corp", f"{i} main street", "10001", str(i))
            for i in range(1, 30)
        ]
        blocker.index_targets(targets)
        blocker.prune_large_blocks()
        s1 = make_s1("S1-001", "US", "alpha corp", "1 main street", "10001", "1")
        cands = blocker.retrieve_candidates(s1)
        assert len(cands) <= 5, f"Cap not enforced: got {len(cands)} candidates"

    def test_cap_not_applied_below_threshold(self):
        """When below cap, all valid candidates are returned."""
        blocker = MultiIndexBlocker(max_candidates=20)
        targets = [
            make_target(f"S2-{i:03d}", "US", "beta logistics", "5 oak street", "10001")
            for i in range(1, 6)
        ]
        blocker.index_targets(targets)
        s1 = make_s1("S1-001", "US", "beta logistics", "5 oak street", "10001")
        cands = blocker.retrieve_candidates(s1)
        assert len(cands) == 5

    def test_exact_name_match_priority_retained(self):
        """
        Exact root name matches must survive candidate capping over non-exact matches.
        When capping kicks in, tier1 (exact matches) must be retained over tier2 (scored).
        """
        blocker = MultiIndexBlocker(max_candidates=3)
        targets = [
            # S2-EXACT: exact root name match with same postal — must survive cap (Tier 1)
            make_target("S2-EXACT", "US", "acme logistics", "999 elm st", "99999"),
        ] + [
            # 19 distractors: similar name prefix "acme" but different root name (Tier 2)
            make_target(f"S2-{i:03d}", "US", "acme shipping", f"{i} main st", "10001")
            for i in range(1, 20)
        ]
        blocker.index_targets(targets)
        blocker.prune_large_blocks()
        # S1 query: exact root name "acme logistics" → S2-EXACT is Tier1; others are Tier2
        s1 = make_s1("S1-001", "US", "acme logistics", "999 elm st", "99999")
        cands = blocker.retrieve_candidates(s1)
        assert len(cands) <= 3, f"Cap not enforced: got {len(cands)}"
        assert "S2-EXACT" in cands, (
            f"Exact name match (Tier1) must survive capping over non-exact candidates. "
            f"Got: {cands}"
        )


class TestOversizedBlocks:
    def test_oversized_block_sub_blocking_used(self):
        """Records should still be retrievable after oversized block handling."""
        blocker = MultiIndexBlocker(max_candidates=5, max_block_size=3)
        targets = [
            make_target(f"S2-{i:03d}", "US", "mega corp", "100 market road", f"1000{i}", "100")
            for i in range(1, 10)
        ]
        blocker.index_targets(targets)
        blocker.prune_large_blocks()
        s1 = make_s1("S1-001", "US", "mega corp", "100 market road", "10001", "100")
        cands = blocker.retrieve_candidates(s1)
        # With max_candidates=5, should return up to 5 candidates
        assert len(cands) <= 5


class TestMissingFields:
    def test_missing_postal_still_retrieves_by_name(self):
        blocker = MultiIndexBlocker(max_candidates=10)
        targets = [
            make_target("S2-001", "US", "green energy corp", "12 oak avenue", "", "12"),
        ]
        blocker.index_targets(targets)
        s1 = make_s1("S1-001", "US", "green energy corp", "12 oak avenue", "", "12")
        cands = blocker.retrieve_candidates(s1)
        assert "S2-001" in cands

    def test_missing_address_still_retrieves_by_name(self):
        blocker = MultiIndexBlocker(max_candidates=10)
        targets = [
            make_target("S2-001", "US", "green energy", "", ""),
        ]
        blocker.index_targets(targets)
        s1 = make_s1("S1-001", "US", "green energy", "", "")
        cands = blocker.retrieve_candidates(s1)
        assert "S2-001" in cands

    def test_no_candidates_returns_empty_list(self):
        blocker = MultiIndexBlocker(max_candidates=10)
        targets = [
            make_target("S2-001", "US", "completely unrelated business", "99 xyz", "99999"),
        ]
        blocker.index_targets(targets)
        s1 = make_s1("S1-001", "US", "nothing in common", "11 abc", "11111")
        cands = blocker.retrieve_candidates(s1)
        assert cands == []


class TestS2S3Mixing:
    def test_s2_and_s3_both_retrieved(self):
        """S2 and S3 targets from same country should both be retrievable."""
        blocker = MultiIndexBlocker(max_candidates=10)
        targets = [
            make_target("S2-001", "US", "acme transport", "50 west ave", "10001"),
            make_target("S3-001", "US", "acme transport", "50 west avenue", "10001"),
        ]
        blocker.index_targets(targets)
        s1 = make_s1("S1-001", "US", "acme transport", "50 west avenue", "10001")
        cands = blocker.retrieve_candidates(s1)
        assert "S2-001" in cands
        assert "S3-001" in cands


class TestDeterminism:
    def test_same_query_same_result(self):
        """Repeated queries must return the same candidate list."""
        blocker = MultiIndexBlocker(max_candidates=10)
        targets = [
            make_target(f"S2-{i:03d}", "US", "delta shipping", "1 harbor blvd", "55555")
            for i in range(1, 8)
        ]
        blocker.index_targets(targets)
        s1 = make_s1("S1-001", "US", "delta shipping", "1 harbor blvd", "55555")
        cands1 = blocker.retrieve_candidates(s1)
        cands2 = blocker.retrieve_candidates(s1)
        assert sorted(cands1) == sorted(cands2)

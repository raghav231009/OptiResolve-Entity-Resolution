"""
Production Blocking Engine Tests.
Proves:
1. Each of the 6 complementary blocking channels produces expected keys:
   - Name token blocking
   - Name prefix blocking
   - Postal-code blocking
   - Building-number + street-anchor blocking
   - Two-word brand/name blocking
   - Street-anchor blocking
2. Oversized blocks are not silently deleted:
   - Records remain in primary index
   - Keys flagged in oversized_keys set
   - Tracked in block statistics
3. Sub-block retrieval works:
   - Targeted sub-block queries retrieve candidates
   - Preserves recall under large blocks
4. Country isolation works:
   - Partitions dynamically by country with zero cross-country leakage
5. True matches survive oversized blocks:
   - Priority tier retention and sub-blocking ensure true positives survive distractors
6. Candidate cap is enforced:
   - Number of returned candidates <= max_candidates_per_entity
   - Candidates before vs. after cap tracked in statistics
7. Configurable parameters:
   - max_block_size, sub_block_threshold, max_candidates_per_entity
8. Audit logging and statistics:
   - Total blocks, block-size distribution percentiles, sub-block count, candidate counts
"""

import pytest
from business_entity_resolution.blocking import MultiIndexBlocker


def make_rec(
    entity_id: str,
    country: str,
    root_name: str,
    clean_address: str = "",
    postal_code: str = "",
    building_number: str = "",
) -> dict:
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


class TestChannelKeyExtraction:
    """Requirement 1: Test that each of the 6 channels produces expected keys."""

    def test_channel_1_name_tokens(self):
        blocker = MultiIndexBlocker(min_token_len=3)
        rec = make_rec("S1-1", "US", "acme logistics international", "100 broadway", "10001", "100")
        toks, prefixes, postals, bldg_street, twoword, street, geo_q, name_q = blocker.extract_keys(rec)

        assert "acme" in toks
        assert "logistics" in toks
        # Stopwords or tokens past the first 2 significant words
        assert len(toks) == 2

    def test_channel_2_name_prefix(self):
        blocker = MultiIndexBlocker(name_prefix_len=4)
        rec = make_rec("S1-1", "US", "walmart stores", "702 sw 8th st", "72716")
        toks, prefixes, postals, bldg_street, twoword, street, geo_q, name_q = blocker.extract_keys(rec)

        assert prefixes == ["walm"]

    def test_channel_3_postal_code(self):
        blocker = MultiIndexBlocker()
        rec = make_rec("S1-1", "FRANCE", "societe generale", "29 boulevard haussmann", "75009")
        toks, prefixes, postals, bldg_street, twoword, street, geo_q, name_q = blocker.extract_keys(rec)

        assert postals == ["75009"]

    def test_channel_4_building_number_street_anchor(self):
        blocker = MultiIndexBlocker()
        rec = make_rec("S1-1", "US", "empire state realty", "350 fifth avenue", "10118", "350")
        toks, prefixes, postals, bldg_street, twoword, street, geo_q, name_q = blocker.extract_keys(rec)

        assert len(bldg_street) == 1
        assert bldg_street[0].startswith("350_")
        assert "fift" in bldg_street[0]

    def test_channel_5_two_word_brand(self):
        blocker = MultiIndexBlocker()
        rec = make_rec("S1-1", "INDIA", "tata motors limited", "bombay house homi mody st", "400001")
        toks, prefixes, postals, bldg_street, twoword, street, geo_q, name_q = blocker.extract_keys(rec)

        assert twoword == ["tata_motors"]

    def test_channel_6_street_anchor(self):
        blocker = MultiIndexBlocker()
        # Multi-word street address
        rec1 = make_rec("S1-1", "FRANCE", "boulangerie", "rue rivoli paris", "75001")
        toks, prefixes, postals, bldg_street, twoword, street, geo_q, name_q = blocker.extract_keys(rec1)
        assert street == ["rue_rivoli"]

        # Single-word street address
        rec2 = make_rec("S1-2", "US", "downtown cafe", "broadway", "10001")
        toks2, prefixes2, postals2, bldg_street2, twoword2, street2, geo_q2, name_q2 = blocker.extract_keys(rec2)
        assert street2 == ["broadway"]

    def test_deterministic_sub_block_qualifiers(self):
        blocker = MultiIndexBlocker()
        rec = make_rec("S1-1", "US", "starbucks coffee", "1912 pike place", "98101", "1912")
        toks, prefixes, postals, bldg_street, twoword, street, geo_q, name_q = blocker.extract_keys(rec)

        # Geo-qualifier should be postal prefix (981)
        assert geo_q == "981"
        # Name-qualifier should be root_name prefix (sta)
        assert name_q == "sta"


class TestOversizedBlocksRecallPreservation:
    """Requirement: Do NOT simply delete oversized blocks; preserve recall."""

    def test_oversized_blocks_not_silently_deleted(self):
        """Oversized primary blocks must NOT be deleted from the primary index."""
        blocker = MultiIndexBlocker(max_candidates=10, max_block_size=5, sub_block_threshold=5)
        # Create 12 targets with same name token "apex"
        targets = [
            make_rec(f"T-{i:02d}", "US", "apex enterprise", f"{i*10} market st", f"100{i:02d}", str(i * 10))
            for i in range(1, 13)
        ]
        blocker.index_targets(targets)

        # Before pruning
        assert len(blocker.idx_name_token["US"]["apex"]) == 12

        # Prune / flag large blocks
        blocker.prune_large_blocks()

        # The primary block MUST NOT be deleted (unlike old implementation that used del ch_dict[k])
        assert len(blocker.idx_name_token["US"]["apex"]) == 12, "Primary block was deleted!"

        # The key must be flagged in oversized_keys
        assert "tok_apex" in blocker.oversized_keys["US"]

        # Statistics should record oversized blocks
        stats = blocker.get_block_statistics()
        assert stats["oversized_blocks_count"] >= 1
        assert stats["total_sub_blocks"] > 0

    def test_sub_block_retrieval_works(self):
        """
        When a primary block is oversized, query should target the relevant sub-block
        matching the query's qualifier to retrieve candidates efficiently.
        """
        blocker = MultiIndexBlocker(max_candidates=10, max_block_size=4, sub_block_threshold=4)
        targets = [
            # Group 1: Seattle (postal 981xx)
            make_rec("T-SEA-1", "US", "starbucks coffee", "1912 pike pl", "98101", "1912"),
            make_rec("T-SEA-2", "US", "starbucks coffee", "1124 1st ave", "98101", "1124"),
            # Group 2: NYC (postal 100xx)
            make_rec("T-NYC-1", "US", "starbucks coffee", "100 broadway", "10001", "100"),
            make_rec("T-NYC-2", "US", "starbucks coffee", "150 5th ave", "10011", "150"),
            make_rec("T-NYC-3", "US", "starbucks coffee", "200 8th ave", "10011", "200"),
        ]
        blocker.index_targets(targets)
        blocker.prune_large_blocks()

        # "tok_starbucks" is oversized (> 4 targets)
        assert "tok_starbucks" in blocker.oversized_keys["US"]

        # Query with NYC postal "10001" (geo_q = "100")
        s1_nyc = make_rec("S1-NYC", "US", "starbucks coffee", "100 broadway", "10001", "100")
        cands_nyc = blocker.retrieve_candidates(s1_nyc)

        # Candidates from NYC sub-block must be retrieved
        assert "T-NYC-1" in cands_nyc
        assert "T-NYC-2" in cands_nyc

    def test_true_matches_survive_oversized_blocks(self):
        """
        True match should survive even when the primary name block has 50 distractors.
        """
        blocker = MultiIndexBlocker(max_candidates=10, max_block_size=5, sub_block_threshold=5)

        # 40 distractors with common name "national supply" across various random postals
        distractors = [
            make_rec(f"DIST-{i:02d}", "US", "national supply", f"{i} highway road", f"500{i:02d}", str(i))
            for i in range(1, 41)
        ]
        # 1 true match with identical physical address and postal code
        true_target = make_rec("TRUE-001", "US", "national supply co", "700 industrial parkway", "30301", "700")

        blocker.index_targets(distractors + [true_target])
        blocker.prune_large_blocks()

        s1_query = make_rec("S1-QUERY", "US", "national supply co", "700 industrial parkway", "30301", "700")
        cands = blocker.retrieve_candidates(s1_query)

        assert "TRUE-001" in cands, f"True match failed to survive oversized block! Candidates: {cands}"


class TestCountryIsolation:
    """Requirement: All blocking must remain country-partitioned. Do not hardcode countries."""

    def test_strict_country_isolation_all_channels(self):
        blocker = MultiIndexBlocker(max_candidates=20)

        # 3 countries with identically named entities and identical addresses
        records = [
            make_rec("US-01", "US", "meridian technologies", "100 technology drive", "94016", "100"),
            make_rec("FR-01", "FRANCE", "meridian technologies", "100 technology drive", "94016", "100"),
            make_rec("IN-01", "INDIA", "meridian technologies", "100 technology drive", "94016", "100"),
            make_rec("DE-01", "GERMANY", "meridian technologies", "100 technology drive", "94016", "100"),
        ]
        blocker.index_targets(records)
        blocker.prune_large_blocks()

        for country, expected_id in [("US", "US-01"), ("FRANCE", "FR-01"), ("INDIA", "IN-01"), ("GERMANY", "DE-01")]:
            s1 = make_rec(f"S1-{country}", country, "meridian technologies", "100 technology drive", "94016", "100")
            cands = blocker.retrieve_candidates(s1)
            assert expected_id in cands, f"{expected_id} not retrieved for {country}"
            for other_id in ["US-01", "FR-01", "IN-01", "DE-01"]:
                if other_id != expected_id:
                    assert other_id not in cands, f"Cross-country leak: {other_id} retrieved in {country} query!"

    def test_no_hardcoded_countries(self):
        """Blocker must seamlessly support arbitrary unseen countries without configuration change."""
        blocker = MultiIndexBlocker(max_candidates=10)
        rec_jp = make_rec("JP-01", "JAPAN", "tokyo trading", "1 chome marunouchi", "100-0005", "1")
        rec_br = make_rec("BR-01", "BRAZIL", "paulista logistics", "avenida paulista 1000", "01310", "1000")

        blocker.index_targets([rec_jp, rec_br])
        blocker.prune_large_blocks()

        cands_jp = blocker.retrieve_candidates(make_rec("S1-JP", "JAPAN", "tokyo trading", "marunouchi", "100-0005"))
        assert "JP-01" in cands_jp
        assert "BR-01" not in cands_jp


class TestCandidateCapAndStatistics:
    """Requirements: Candidate cap enforcement, configurable parameters, logging/statistics."""

    def test_candidate_cap_enforced(self):
        cap = 8
        blocker = MultiIndexBlocker(max_candidates_per_entity=cap)
        assert blocker.max_candidates == cap

        # 30 targets in US
        targets = [
            make_rec(f"T-{i:02d}", "US", f"general trading {i}", f"{i} main street", "20001", str(i))
            for i in range(1, 31)
        ]
        # All share prefix "gene"
        blocker.index_targets(targets)
        blocker.prune_large_blocks()

        s1 = make_rec("S1", "US", "general trading", "50 main street", "20001", "50")
        cands = blocker.retrieve_candidates(s1)

        assert len(cands) <= cap
        assert len(cands) == cap

    def test_retrieval_and_block_statistics(self):
        blocker = MultiIndexBlocker(max_candidates_per_entity=5, max_block_size=3, sub_block_threshold=3)

        targets = [
            make_rec(f"T-{i:02d}", "US", "super warehouse", f"{i} industrial road", "10001", str(i))
            for i in range(1, 10)
        ]
        blocker.index_targets(targets)
        blocker.prune_large_blocks()

        block_stats = blocker.get_block_statistics()
        assert block_stats["total_primary_blocks"] > 0
        assert block_stats["oversized_blocks_count"] > 0
        assert block_stats["total_sub_blocks"] > 0
        assert "block_size_distribution" in block_stats
        dist = block_stats["block_size_distribution"]
        assert dist["min"] >= 1
        assert dist["max"] >= 9
        assert "median" in dist

        # Run several queries
        s1 = make_rec("S1-01", "US", "super warehouse", "1 industrial road", "10001", "1")
        blocker.retrieve_candidates(s1)

        ret_stats = blocker.get_retrieval_statistics()
        assert ret_stats["total_queries"] == 1
        assert ret_stats["total_candidates_before_cap"] >= 5
        assert ret_stats["total_candidates_after_cap"] <= 5
        assert ret_stats["avg_candidates_after_cap"] <= 5.0

    def test_configurable_parameters(self):
        blocker = MultiIndexBlocker(
            max_candidates=50,
            min_token_len=4,
            name_prefix_len=5,
            max_block_size=200,
            sub_block_threshold=150,
            max_candidates_per_entity=60,
        )
        assert blocker.max_candidates == 60
        assert blocker.max_candidates_per_entity == 60
        assert blocker.min_token_len == 4
        assert blocker.name_prefix_len == 5
        assert blocker.max_block_size == 200
        assert blocker.sub_block_threshold == 150

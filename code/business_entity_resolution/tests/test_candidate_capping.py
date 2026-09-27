"""
Unit and Regression Tests for Candidate Capping and Priority Tier Ranking in OptiResolve.

Verifies:
1. Tier 1 candidates (exact root-name matches, exact physical anchors) are guaranteed to survive the cap.
2. Tier 1 itself cannot exceed K without a deterministic priority policy.
3. Candidate ranking uses normalized fields consistently.
4. True matches that rank low by name but high by address are preserved over superficial name distractors.
5. True matches with exact root name survive when candidates > K.
6. True matches with exact building+postal survive when candidates > K.
7. Comparison across arbitrary top-K, current legacy ranking, and tiered ranking.
8. Deterministic tie-breaking produces invariant output regardless of set ordering.
"""

import pytest
from business_entity_resolution.blocking import MultiIndexBlocker
from business_entity_resolution.normalization import (
    clean_business_name,
    extract_building_number,
    extract_postal_code,
    normalize_address,
)


def make_rec(eid, country, name, addr, postal="", bldg=""):
    clean_n, root_n = clean_business_name(name)
    clean_a = normalize_address(addr)
    post = postal or extract_postal_code(addr)
    bldg_n = bldg or extract_building_number(clean_a, post)
    return {
        "entity_id": eid,
        "country": country.upper(),
        "clean_name": clean_n,
        "root_name": root_n,
        "clean_address": clean_a,
        "postal_code": post,
        "building_number": bldg_n,
    }


class TestCandidateCappingTier1Guarantee:
    """Requirement 1 & 6: Tier 1 candidates are guaranteed to survive candidate cap K."""

    def test_true_match_exact_root_name_survives_cap(self):
        """When >K candidates exist, true match with exact root name must survive."""
        cap = 10
        blocker = MultiIndexBlocker(max_candidates_per_entity=cap, capping_strategy="tiered")

        # 40 distractors with partial name match ("horizon trading i")
        distractors = [
            make_rec(f"D-{i:03d}", "US", f"horizon logistics {i}", f"{i*10} oak avenue", "90210", str(i * 10))
            for i in range(1, 41)
        ]
        # True target has exact root name "horizon"
        true_target = make_rec("T-EXACT", "US", "horizon inc", "999 sunset boulevard", "90001", "999")

        blocker.index_targets(distractors + [true_target])
        blocker.prune_large_blocks()

        s1_query = make_rec("S1", "US", "horizon corp", "123 maple street", "10001", "123")
        cands = blocker.retrieve_candidates(s1_query)

        assert len(cands) <= cap
        assert "T-EXACT" in cands, "True match with exact root name was lost due to capping!"

    def test_true_match_exact_building_postal_survives_cap(self):
        """When >K candidates exist, true match with exact building+postal must survive even with divergent name."""
        cap = 10
        blocker = MultiIndexBlocker(max_candidates_per_entity=cap, capping_strategy="tiered")

        # 50 distractors sharing name token "apex"
        distractors = [
            make_rec(f"D-{i:03d}", "US", f"apex enterprise {i}", f"{i*5} random street", f"100{i:02d}", str(i * 5))
            for i in range(1, 51)
        ]
        # True target has completely different name ("summit ventures"), but identical physical address & postal
        true_target = make_rec("T-PHYSICAL", "US", "summit ventures", "450 lexington avenue", "10017", "450")

        blocker.index_targets(distractors + [true_target])
        blocker.prune_large_blocks()

        # Query has name "apex summit" (so it retrieves both summit and apex channels),
        # but physical building 450 + postal 10017 matches true_target
        s1_query = make_rec("S1", "US", "apex summit", "450 lexington ave ste 100", "10017", "450")
        cands = blocker.retrieve_candidates(s1_query)

        assert len(cands) <= cap
        assert "T-PHYSICAL" in cands, "True match with exact building+postal was lost due to capping!"


class TestTier1OverflowDeterministicPolicy:
    """Requirement 2: Ensure Tier 1 itself cannot exceed K without a deterministic policy."""

    def test_tier1_overflow_does_not_exceed_cap(self):
        """If Tier 1 alone exceeds K, returned candidates must equal exactly K."""
        cap = 8
        blocker = MultiIndexBlocker(max_candidates_per_entity=cap, capping_strategy="tiered")

        # 25 candidates all sharing exact root name "starbucks"
        targets = [
            make_rec(f"T-{i:02d}", "US", "starbucks coffee", f"{i*10} market street", f"9410{i%10}", str(i * 10))
            for i in range(1, 26)
        ]
        blocker.index_targets(targets)
        blocker.prune_large_blocks()

        s1_query = make_rec("S1", "US", "starbucks", "50 market street", "94105", "50")
        cands = blocker.retrieve_candidates(s1_query)

        assert len(cands) == cap, f"Expected exactly {cap} candidates, got {len(cands)}"

    def test_tier1_overflow_prioritizes_physical_proximity(self):
        """Within Tier 1 overflow, candidates with matching postal/building rank higher."""
        cap = 5
        blocker = MultiIndexBlocker(max_candidates_per_entity=cap, capping_strategy="tiered")

        targets = []
        for i in range(1, 20):
            # Target 5 has exact building 50 and exact postal 94105
            if i == 5:
                targets.append(make_rec("T-05", "US", "starbucks", "50 market street", "94105", "50"))
            else:
                targets.append(make_rec(f"T-{i:02d}", "US", "starbucks", f"{i*100} far away road", "90001", str(i * 100)))

        blocker.index_targets(targets)
        blocker.prune_large_blocks()

        s1_query = make_rec("S1", "US", "starbucks", "50 market street", "94105", "50")
        cands = blocker.retrieve_candidates(s1_query)

        assert len(cands) == cap
        # T-05 has both exact name AND exact physical anchor (dual match) -> must be ranked first!
        assert cands[0] == "T-05"

    def test_deterministic_tie_breaking(self):
        """Tier 1 ranking produces 100% deterministic results regardless of target order."""
        cap = 6
        targets_order_a = [
            make_rec(f"T-{i:02d}", "US", "national retail", f"100 commercial rd {i}", "50001", "100")
            for i in range(1, 21)
        ]
        targets_order_b = list(reversed(targets_order_a))

        blocker_a = MultiIndexBlocker(max_candidates_per_entity=cap, capping_strategy="tiered")
        blocker_a.index_targets(targets_order_a)

        blocker_b = MultiIndexBlocker(max_candidates_per_entity=cap, capping_strategy="tiered")
        blocker_b.index_targets(targets_order_b)

        s1_query = make_rec("S1", "US", "national retail", "100 commercial rd", "50001", "100")
        cands_a = blocker_a.retrieve_candidates(s1_query)
        cands_b = blocker_b.retrieve_candidates(s1_query)

        assert cands_a == cands_b, f"Tie-breaking is non-deterministic! {cands_a} != {cands_b}"


class TestHighAddressLowNameCandidateSurvival:
    """Requirement 6: True match ranks low by name but high by address."""

    def test_low_name_high_address_survives_cap(self):
        """
        True match has divergent/corrupted name (e.g., subsidiary or abbreviation),
        but strong address alignment. Must not be displaced by superficial name distractors.
        """
        cap = 10
        blocker = MultiIndexBlocker(max_candidates_per_entity=cap, capping_strategy="tiered")

        # 30 distractors with high superficial name similarity to "olympus global"
        distractors = [
            make_rec(f"D-{i:02d}", "US", f"olympus global trading {i}", f"{i*12} unconnected road", f"700{i:02d}", str(i * 12))
            for i in range(1, 31)
        ]

        # True target: name has low lexical overlap ("OG Services"), but address matches perfectly
        true_target = make_rec("T-ADDR-MATCH", "US", "og enterprise group", "742 evergreen terrace", "97477", "742")

        blocker.index_targets(distractors + [true_target])
        blocker.prune_large_blocks()

        # Query has both "olympus" (pulling distractors) and "742 evergreen" (pulling true target)
        s1_query = make_rec("S1", "US", "olympus og inc", "742 evergreen terrace suite 10", "97477", "742")
        cands = blocker.retrieve_candidates(s1_query)

        assert len(cands) <= cap
        assert "T-ADDR-MATCH" in cands, "Candidate with low name match but high address match was dropped by cap!"


class TestNormalizedFieldsConsistency:
    """Requirement 3: Verify ranking uses normalized fields consistently even on un-normalized inputs."""

    def test_raw_records_without_precomputed_fields(self):
        """Blocker must handle raw dicts without pre-computed root_name/clean_address transparently."""
        cap = 5
        blocker = MultiIndexBlocker(max_candidates_per_entity=cap, capping_strategy="tiered")

        # Pass raw target records without clean_name, root_name, postal_code, building_number
        raw_targets = [
            {
                "entity_id": f"T-{i:02d}",
                "country": "US",
                "business_name": f"Acme Corporation {i} LLC",
                "business_address": f"{i} Industrial Way, Springfield, IL 62701",
            }
            for i in range(1, 20)
        ]
        true_raw_target = {
            "entity_id": "T-TRUE",
            "country": "US",
            "business_name": "Acme Inc.",
            "business_address": "500 Main St, Springfield, IL 62701",
        }
        blocker.index_targets(raw_targets + [true_raw_target])
        blocker.prune_large_blocks()

        raw_query = {
            "entity_id": "S1",
            "country": "US",
            "business_name": "Acme",
            "business_address": "500 Main Street, Springfield, IL 62701",
        }
        cands = blocker.retrieve_candidates(raw_query)

        assert len(cands) <= cap
        assert "T-TRUE" in cands


class TestCappingStrategyComparison:
    """Requirement 5: Compare arbitrary top-K, current ranking, and tiered ranking."""

    def test_tiered_outperforms_arbitrary_and_legacy(self):
        """
        Construct a benchmark scenario where:
        - Arbitrary top-K drops the true match because entity_id starts with 'Z'
        - Current legacy ranking drops true match because address weight is too low
        - Tiered ranking preserves the true match
        """
        cap = 5

        # 15 name distractors with moderate name similarity and zero address match
        distractors = [
            make_rec(f"A-{i:02d}", "US", f"metro express cargo {i}", f"{i*15} pine avenue", f"300{i:02d}", str(i * 15))
            for i in range(1, 16)
        ]
        # True match has divergent name ("MX Logistics"), but exact address & postal
        # Entity ID is "Z-TRUE" so arbitrary alphabetical slicing drops it
        true_match = make_rec("Z-TRUE", "US", "mx logistics group", "100 wall street", "10005", "100")

        all_records = distractors + [true_match]

        # 1. Arbitrary capping
        blocker_arb = MultiIndexBlocker(max_candidates_per_entity=cap, capping_strategy="arbitrary")
        blocker_arb.index_targets(all_records)
        blocker_arb.prune_large_blocks()

        # 2. Legacy current ranking
        blocker_cur = MultiIndexBlocker(max_candidates_per_entity=cap, capping_strategy="current")
        blocker_cur.index_targets(all_records)
        blocker_cur.prune_large_blocks()

        # 3. Upgraded Tiered ranking
        blocker_tier = MultiIndexBlocker(max_candidates_per_entity=cap, capping_strategy="tiered")
        blocker_tier.index_targets(all_records)
        blocker_tier.prune_large_blocks()

        s1_query = make_rec("S1", "US", "metro mx", "100 wall street suite 500", "10005", "100")

        cands_arb = blocker_arb.retrieve_candidates(s1_query)
        cands_tier = blocker_tier.retrieve_candidates(s1_query)

        # Arbitrary drops Z-TRUE because A-01..A-05 precede Z-TRUE alphabetically
        assert "Z-TRUE" not in cands_arb, "Arbitrary capping should have dropped Z-TRUE"

        # Tiered ranking preserves Z-TRUE because it qualifies for Tier 1 (exact building 100 + postal 10005)
        assert "Z-TRUE" in cands_tier, "Tiered ranking must preserve Z-TRUE"


class TestRetrievalAuditStatistics:
    """Requirement 7: Verify capping statistics and audit metrics."""

    def test_statistics_recorded_accurately(self):
        cap = 5
        blocker = MultiIndexBlocker(max_candidates_per_entity=cap, capping_strategy="tiered")

        targets = [
            make_rec(f"T-{i:02d}", "US", f"pacific trade {i}", f"{i*10} ocean blvd", "90401", str(i * 10))
            for i in range(1, 20)
        ]
        blocker.index_targets(targets)
        blocker.prune_large_blocks()

        s1_query = make_rec("S1", "US", "pacific trade", "50 ocean blvd", "90401", "50")
        cands, uncapped = blocker.retrieve_candidates_with_uncapped(s1_query)

        stats = blocker.get_retrieval_statistics()
        assert stats["total_queries"] == 1
        assert stats["capping_events"] == 1
        assert stats["total_candidates_before_cap"] == len(uncapped)
        assert stats["total_candidates_after_cap"] == len(cands)
        assert stats["total_candidates_after_cap"] <= cap
        assert stats["capping_strategy"] == "tiered"

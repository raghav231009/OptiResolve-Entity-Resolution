"""
High-Recall Multi-Index Inverted Blocking Engine.
Partitions by open-set country, employs deterministic sub-blocking across all 6 channels
rather than destructive deletion for oversized blocks, and utilizes multi-signal similarity
pre-ranking with Priority Tier Retention to safeguard candidate recall.
"""

from collections import defaultdict
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import rapidfuzz.fuzz as fuzz

from .normalization import (
    clean_business_name,
    extract_building_number,
    extract_postal_code,
    normalize_address,
)

logger = logging.getLogger(__name__)

STOPWORDS = {"the", "and", "dr", "all", "new", "mr", "mrs", "miss", "les", "des", "une"}


class MultiIndexBlocker:
    """
    Multi-channel inverted index for candidate generation.
    Supports dynamic, open-set country partitioning, secondary sub-blocking,
    and priority tier retention during candidate capping.
    """

    def __init__(
        self,
        max_candidates: int = 80,
        min_token_len: int = 3,
        name_prefix_len: int = 4,
        max_block_size: int = 350,
        sub_block_threshold: Optional[int] = None,
        max_candidates_per_entity: Optional[int] = None,
        capping_strategy: str = "tiered",
    ):
        if max_candidates_per_entity is not None:
            max_candidates = max_candidates_per_entity
        self.max_candidates = max_candidates
        self.max_candidates_per_entity = max_candidates
        self.min_token_len = min_token_len
        self.name_prefix_len = name_prefix_len
        self.max_block_size = max_block_size
        self.sub_block_threshold = sub_block_threshold if sub_block_threshold is not None else max_block_size
        self.capping_strategy = capping_strategy

        # Inverted index tables: {country: {key: [entity_id, ...]}}
        # 1. Name token blocking
        self.idx_name_token: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        # 2. Name prefix blocking
        self.idx_name_prefix: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        # 3. Postal-code blocking
        self.idx_postal: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        # 4. Building-number + street-anchor blocking
        self.idx_addr_anchor: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        # 5. Two-word brand/name blocking
        self.idx_two_word: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        # 6. Street-anchor blocking
        self.idx_street_anchor: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        # Sub-blocking tables for large blocks: {country: {f"{channel}_{key}_{sub}": [entity_id, ...]}}
        self.idx_sub_blocks: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.oversized_keys: Dict[str, Set[str]] = defaultdict(set)

        # Fast lookup store for multi-signal pre-ranking: id -> (root_name, clean_name, clean_address, postal, bldg_num)
        self.target_store: Dict[str, Tuple[str, str, str, str, str]] = {}

        # Statistics tracking
        self.last_block_statistics: Dict[str, Any] = {}
        self.retrieval_stats = {
            "total_queries": 0,
            "total_candidates_before_cap": 0,
            "total_candidates_after_cap": 0,
            "max_candidates_before_cap": 0,
            "max_candidates_after_cap": 0,
            "capping_events": 0,
            "tier1_retained": 0,
            "tier2_retained": 0,
            "tier1_overflow_events": 0,
        }
        self.last_candidate_channels: Dict[str, List[str]] = {}

    def extract_keys(
        self, record: dict
    ) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], str, str]:
        """
        Extract 6-channel blocking keys and deterministic sub-block qualifiers:
        1. Name token blocking: significant brand words
        2. Name prefix blocking: 4-character brand prefix
        3. Postal-code blocking: exact postal / PIN code
        4. Building-number + street-anchor blocking: building number + street anchor
        5. Two-word brand/name blocking: two-word compound brand anchor
        6. Street-anchor blocking: street name pair anchor

        Sub-block qualifiers:
        - geo_qualifier: postal 3-char prefix or clean_addr 3-char prefix (for name blocks)
        - name_qualifier: root_name 3-char prefix (for location/address blocks)
        """
        root_name = record.get("root_name", "")
        clean_addr = record.get("clean_address", "")
        postal = record.get("postal_code", "")
        bldg_num = record.get("building_number", "")

        # Consistent normalization fallback
        if not root_name and "business_name" in record:
            _, root_name = clean_business_name(str(record.get("business_name", "")))
        if not clean_addr and "business_address" in record:
            clean_addr = normalize_address(str(record.get("business_address", "")))
        if not postal and "business_address" in record:
            postal = extract_postal_code(str(record.get("business_address", "")))
        if not bldg_num and clean_addr:
            bldg_num = extract_building_number(clean_addr, postal)

        # Channel 1: Significant Name Tokens
        name_words = [tok for tok in root_name.split() if len(tok) >= self.min_token_len and tok not in STOPWORDS]
        name_tokens = name_words[:2] if name_words else []

        # Channel 2: Name Prefix
        prefixes = [root_name[: self.name_prefix_len]] if len(root_name) >= self.name_prefix_len else []

        # Channel 3: Postal Code
        postals = [postal] if postal else []

        # Channel 4: Address Building + First Street Token Prefix Anchor
        addr_words = [w for w in clean_addr.split() if len(w) >= 3 and not w.isdigit() and w not in STOPWORDS]
        addr_anchors = []
        if bldg_num and addr_words:
            first_word_prefix = addr_words[0][:4]
            addr_anchors.append(f"{bldg_num}_{first_word_prefix}")

        # Channel 5: Two-Word Brand Anchor
        two_word_keys = [f"{name_words[0]}_{name_words[1]}"] if len(name_words) >= 2 else []

        # Channel 6: Street Name Anchor
        street_anchors = []
        if len(addr_words) >= 2:
            street_anchors.append(f"{addr_words[0]}_{addr_words[1]}")
        elif len(addr_words) == 1:
            street_anchors.append(addr_words[0])

        # Deterministic sub-block qualifiers:
        # Geo-qualifier for name-based primary blocks
        geo_q = postal[:3] if len(postal) >= 3 else (postal if postal else (clean_addr[:3] if len(clean_addr) >= 3 else "xx"))
        # Name-qualifier for location/address-based primary blocks
        name_q = root_name[:3] if len(root_name) >= 3 else (root_name if root_name else "xx")

        return name_tokens, prefixes, postals, addr_anchors, two_word_keys, street_anchors, geo_q, name_q

    def index_targets(self, records: List[dict]):
        """Build inverted index from Source 2 and Source 3 target records across all 6 channels."""
        for rec in records:
            eid = str(rec["entity_id"]).strip()
            country = str(rec.get("country", "")).strip().upper()
            root_name = rec.get("root_name", "")
            clean_name = rec.get("clean_name", "")
            clean_addr = rec.get("clean_address", "")
            postal = rec.get("postal_code", "")
            bldg_num = rec.get("building_number", "")

            # Ensure normalized fields are populated consistently
            if not root_name and "business_name" in rec:
                clean_name, root_name = clean_business_name(str(rec.get("business_name", "")))
            elif not clean_name and root_name:
                clean_name = root_name
            if not clean_addr and "business_address" in rec:
                clean_addr = normalize_address(str(rec.get("business_address", "")))
            if not postal and "business_address" in rec:
                postal = extract_postal_code(str(rec.get("business_address", "")))
            if not bldg_num and clean_addr:
                bldg_num = extract_building_number(clean_addr, postal)

            self.target_store[eid] = (root_name, clean_name, clean_addr, postal, bldg_num)

            name_tokens, prefixes, postals, addr_anchors, two_words, street_anchors, geo_q, name_q = self.extract_keys(rec)

            # Channel 1: Name Tokens (sub-partitioned by geo_q)
            for tok in name_tokens:
                self.idx_name_token[country][tok].append(eid)
                self.idx_sub_blocks[country][f"tok_{tok}_{geo_q}"].append(eid)

            # Channel 2: Name Prefix (sub-partitioned by geo_q)
            for pref in prefixes:
                self.idx_name_prefix[country][pref].append(eid)
                self.idx_sub_blocks[country][f"pref_{pref}_{geo_q}"].append(eid)

            # Channel 3: Postal Code (sub-partitioned by name_q)
            for post in postals:
                self.idx_postal[country][post].append(eid)
                self.idx_sub_blocks[country][f"post_{post}_{name_q}"].append(eid)

            # Channel 4: Building + Street Anchor (sub-partitioned by name_q)
            for anchor in addr_anchors:
                self.idx_addr_anchor[country][anchor].append(eid)
                self.idx_sub_blocks[country][f"bldg_street_{anchor}_{name_q}"].append(eid)

            # Channel 5: Two-Word Brand (sub-partitioned by geo_q)
            for tw in two_words:
                self.idx_two_word[country][tw].append(eid)
                self.idx_sub_blocks[country][f"twoword_{tw}_{geo_q}"].append(eid)

            # Channel 6: Street Anchor (sub-partitioned by name_q)
            for sa in street_anchors:
                self.idx_street_anchor[country][sa].append(eid)
                self.idx_sub_blocks[country][f"street_{sa}_{name_q}"].append(eid)

    def prune_large_blocks(self):
        """
        Identify oversized primary blocks across all 6 channels and mark them for
        secondary sub-blocking rather than destructive deletion.
        No blocks are ever deleted — oversized blocks are routed to finer
        sub-block partitions at retrieval time to preserve candidate recall.
        """
        threshold = self.sub_block_threshold or self.max_block_size
        self.oversized_keys.clear()

        channel_indices = [
            ("tok", self.idx_name_token),
            ("pref", self.idx_name_prefix),
            ("post", self.idx_postal),
            ("bldg_street", self.idx_addr_anchor),
            ("twoword", self.idx_two_word),
            ("street", self.idx_street_anchor),
        ]

        total_primary = 0
        total_oversized = 0
        all_block_sizes: List[int] = []

        for prefix, ch_dict in channel_indices:
            for country, keys in ch_dict.items():
                for key, entity_list in keys.items():
                    size = len(entity_list)
                    total_primary += 1
                    all_block_sizes.append(size)
                    if size > threshold:
                        self.oversized_keys[country].add(f"{prefix}_{key}")
                        total_oversized += 1

        total_sub_blocks = sum(len(sub_dict) for sub_dict in self.idx_sub_blocks.values())

        # Compute block size distribution statistics
        if all_block_sizes:
            all_block_sizes.sort()
            stats_dist = {
                "min": int(all_block_sizes[0]),
                "p25": int(np.percentile(all_block_sizes, 25)),
                "median": int(np.median(all_block_sizes)),
                "p75": int(np.percentile(all_block_sizes, 75)),
                "p95": int(np.percentile(all_block_sizes, 95)),
                "p99": int(np.percentile(all_block_sizes, 99)),
                "max": int(all_block_sizes[-1]),
                "mean": round(float(np.mean(all_block_sizes)), 2),
            }
        else:
            stats_dist = {"min": 0, "p25": 0, "median": 0, "p75": 0, "p95": 0, "p99": 0, "max": 0, "mean": 0.0}

        self.last_block_statistics = {
            "total_primary_blocks": total_primary,
            "total_sub_blocks": total_sub_blocks,
            "oversized_blocks_count": total_oversized,
            "sub_block_threshold": threshold,
            "block_size_distribution": stats_dist,
        }

        self.log_block_statistics(logger)

    def reset_retrieval_stats(self):
        """Reset cumulative candidate retrieval and capping counters."""
        self.retrieval_stats = {
            "total_queries": 0,
            "total_candidates_before_cap": 0,
            "total_candidates_after_cap": 0,
            "max_candidates_before_cap": 0,
            "max_candidates_after_cap": 0,
            "capping_events": 0,
            "tier1_retained": 0,
            "tier2_retained": 0,
            "tier1_overflow_events": 0,
        }

    def _update_retrieval_stats(self, before_cap: int, after_cap: int):
        self.retrieval_stats["total_queries"] = self.retrieval_stats.get("total_queries", 0) + 1
        self.retrieval_stats["total_candidates_before_cap"] = self.retrieval_stats.get("total_candidates_before_cap", 0) + before_cap
        self.retrieval_stats["total_candidates_after_cap"] = self.retrieval_stats.get("total_candidates_after_cap", 0) + after_cap
        if before_cap > self.retrieval_stats.get("max_candidates_before_cap", 0):
            self.retrieval_stats["max_candidates_before_cap"] = before_cap
        if after_cap > self.retrieval_stats.get("max_candidates_after_cap", 0):
            self.retrieval_stats["max_candidates_after_cap"] = after_cap

    def retrieve_candidates_with_uncapped(self, s1_rec: dict) -> Tuple[List[str], Set[str]]:
        """
        Retrieve candidate IDs for a Source 1 record across all 6 channels,
        returning both:
          1. Retained candidates after priority tier ranking and safety cap K.
          2. Full set of raw candidates gathered across all 6 channels before capping.
        """
        country = s1_rec["country"].strip().upper()
        name_tokens, prefixes, postals, addr_anchors, two_words, street_anchors, geo_q, name_q = self.extract_keys(s1_rec)

        cand_set: Set[str] = set()
        cand_channels: Dict[str, List[str]] = {}

        def query_channel(channel_prefix: str, keys: List[str], primary_idx: Dict[str, List[str]], sub_qualifier: str):
            for k in keys:
                tagged_key = f"{channel_prefix}_{k}"
                if tagged_key in self.oversized_keys[country]:
                    sub_key = f"{channel_prefix}_{k}_{sub_qualifier}"
                    sub_cands = self.idx_sub_blocks[country].get(sub_key, [])
                    if sub_cands:
                        for cid in sub_cands:
                            cand_set.add(cid)
                            cand_channels.setdefault(cid, []).append(channel_prefix)
                    else:
                        # Fallback to bounded sample of primary block to guarantee recall
                        for cid in primary_idx.get(k, [])[: self.max_candidates]:
                            cand_set.add(cid)
                            cand_channels.setdefault(cid, []).append(channel_prefix)
                else:
                    for cid in primary_idx.get(k, []):
                        cand_set.add(cid)
                        cand_channels.setdefault(cid, []).append(channel_prefix)

        # Channel 1: Name Tokens (geo_q)
        query_channel("tok", name_tokens, self.idx_name_token[country], geo_q)
        # Channel 2: Name Prefixes (geo_q)
        query_channel("pref", prefixes, self.idx_name_prefix[country], geo_q)
        # Channel 3: Postal Code (name_q)
        query_channel("post", postals, self.idx_postal[country], name_q)
        # Channel 4: Building + Street Anchor (name_q)
        query_channel("bldg_street", addr_anchors, self.idx_addr_anchor[country], name_q)
        # Channel 5: Two-Word Brand Anchor (geo_q)
        query_channel("twoword", two_words, self.idx_two_word[country], geo_q)
        # Channel 6: Street Name Anchor (name_q)
        query_channel("street", street_anchors, self.idx_street_anchor[country], name_q)

        self.last_candidate_channels = cand_channels
        raw_candidates_uncapped = set(cand_set)
        candidates_before_cap = len(cand_set)

        if not cand_set:
            self._update_retrieval_stats(0, 0)
            return [], set()

        if len(cand_set) <= self.max_candidates:
            self._update_retrieval_stats(candidates_before_cap, len(cand_set))
            return sorted(cand_set), raw_candidates_uncapped

        self.retrieval_stats["capping_events"] = self.retrieval_stats.get("capping_events", 0) + 1

        # Multi-Signal Similarity Pre-Ranking with Priority Tier Retention:
        s1_root = s1_rec.get("root_name", "")
        s1_clean = s1_rec.get("clean_name", "")
        s1_addr = s1_rec.get("clean_address", "")
        s1_post = s1_rec.get("postal_code", "")
        s1_bldg = s1_rec.get("building_number", "")

        # Fallback to runtime normalization if pre-computed fields are missing
        if not s1_root and "business_name" in s1_rec:
            s1_clean, s1_root = clean_business_name(str(s1_rec.get("business_name", "")))
        elif not s1_clean and s1_root:
            s1_clean = s1_root
        if not s1_addr and "business_address" in s1_rec:
            s1_addr = normalize_address(str(s1_rec.get("business_address", "")))
        if not s1_post and "business_address" in s1_rec:
            s1_post = extract_postal_code(str(s1_rec.get("business_address", "")))
        if not s1_bldg and s1_addr:
            s1_bldg = extract_building_number(s1_addr, s1_post)

        if self.capping_strategy == "arbitrary":
            retained = sorted(cand_set)[: self.max_candidates]
            self._update_retrieval_stats(candidates_before_cap, len(retained))
            return retained, raw_candidates_uncapped

        elif self.capping_strategy == "current":
            tier1_guaranteed: List[str] = []
            tier2_scored: List[Tuple[float, str]] = []

            for cid in sorted(cand_set):
                target_tuple = self.target_store.get(cid)
                if target_tuple:
                    c_root, _, c_addr, c_post, c_bldg = target_tuple
                else:
                    c_root, c_addr, c_post, c_bldg = "", "", "", ""

                has_exact_name = bool(s1_root and s1_root == c_root)
                has_exact_physical = bool(s1_bldg and c_bldg and s1_bldg == c_bldg and s1_post and c_post and s1_post == c_post)

                if has_exact_name or has_exact_physical:
                    tier1_guaranteed.append(cid)
                    continue

                name_set_score = fuzz.token_set_ratio(s1_root, c_root)
                name_q_score = fuzz.QRatio(s1_root, c_root)
                addr_score = fuzz.token_set_ratio(s1_addr, c_addr) if (s1_addr and c_addr) else 0.0
                postal_bonus = 20.0 if (s1_post and c_post and s1_post == c_post) else 0.0
                bldg_bonus = 15.0 if (s1_bldg and c_bldg and s1_bldg == c_bldg) else 0.0

                composite_rank_score = (name_set_score * 0.40) + (name_q_score * 0.20) + (addr_score * 0.20) + postal_bonus + bldg_bonus
                tier2_scored.append((composite_rank_score, cid))

            remaining_slots = max(0, self.max_candidates - len(tier1_guaranteed))
            tier2_scored.sort(key=lambda x: (-x[0], x[1]))
            retained = tier1_guaranteed + [cid for _, cid in tier2_scored[:remaining_slots]]
            retained = retained[: self.max_candidates]
            self._update_retrieval_stats(candidates_before_cap, len(retained))
            return retained, raw_candidates_uncapped

        else:  # "tiered" (audited production policy)
            tier1_scored: List[Tuple[float, str]] = []
            tier2_scored: List[Tuple[float, str]] = []

            for cid in sorted(cand_set):
                target_tuple = self.target_store.get(cid)
                if target_tuple:
                    c_root, c_clean, c_addr, c_post, c_bldg = target_tuple
                else:
                    c_root, c_clean, c_addr, c_post, c_bldg = "", "", "", "", ""

                # Tier 1 Qualification:
                # 1. Exact Name Anchor (root name or full clean business name)
                has_exact_root = bool(s1_root and c_root and s1_root == c_root)
                has_exact_clean = bool(s1_clean and c_clean and s1_clean == c_clean)
                has_exact_name = has_exact_root or has_exact_clean

                # 2. Exact Physical Anchor where appropriate
                has_bldg_postal = bool(s1_bldg and c_bldg and s1_bldg == c_bldg and s1_post and c_post and s1_post == c_post)
                addr_token_set = fuzz.token_set_ratio(s1_addr, c_addr) if (s1_addr and c_addr) else 0.0
                has_bldg_street = bool(s1_bldg and c_bldg and s1_bldg == c_bldg and addr_token_set >= 80.0)
                has_postal_street = bool(s1_post and c_post and s1_post == c_post and addr_token_set >= 90.0)
                has_exact_physical = has_bldg_postal or has_bldg_street or has_postal_street

                if has_exact_name or has_exact_physical:
                    # Deterministic Tier 1 priority score
                    tier1_score = 0.0
                    if has_exact_name and has_exact_physical:
                        tier1_score += 2000.0  # Dual anchor match
                    elif has_exact_name:
                        tier1_score += 1000.0
                    else:
                        tier1_score += 1000.0  # Physical anchor match

                    name_token_set = fuzz.token_set_ratio(s1_root, c_root)
                    tier1_score += (addr_token_set * 0.50) + (name_token_set * 0.50)
                    if s1_post and c_post and s1_post == c_post:
                        tier1_score += 30.0
                    if s1_bldg and c_bldg and s1_bldg == c_bldg:
                        tier1_score += 30.0
                    if has_exact_clean:
                        tier1_score += 40.0

                    tier1_scored.append((tier1_score, cid))
                else:
                    # Tier 2: Remaining candidates scored with balanced name & address signals
                    name_token_set = fuzz.token_set_ratio(s1_root, c_root)
                    name_q = fuzz.QRatio(s1_root, c_root)
                    name_sim = (name_token_set * 0.60) + (name_q * 0.40)

                    addr_q = fuzz.QRatio(s1_addr, c_addr) if (s1_addr and c_addr) else 0.0
                    addr_sim = (addr_token_set * 0.70) + (addr_q * 0.30) if (s1_addr and c_addr) else 0.0

                    # Balanced base score: guarantees high address match is never displaced
                    # by superficial name distractors
                    base_score = (max(name_sim, addr_sim) * 0.60) + (min(name_sim, addr_sim) * 0.40)

                    bonus = 0.0
                    if s1_post and c_post and s1_post == c_post:
                        bonus += 25.0
                    if s1_bldg and c_bldg and s1_bldg == c_bldg:
                        bonus += 20.0
                    if addr_token_set >= 80.0:
                        bonus += 15.0

                    tier2_score = base_score + bonus
                    tier2_scored.append((tier2_score, cid))

            # Deterministic ordering: higher score first, tie-break by cid ascending
            tier1_scored.sort(key=lambda x: (-x[0], x[1]))
            tier2_scored.sort(key=lambda x: (-x[0], x[1]))

            # Requirement 1 & 2: Tier 1 candidates are guaranteed to survive up to K.
            # If Tier 1 exceeds K, deterministic policy retains top K Tier 1 candidates.
            if len(tier1_scored) >= self.max_candidates:
                retained = [cid for _, cid in tier1_scored[: self.max_candidates]]
                self.retrieval_stats["tier1_overflow_events"] = self.retrieval_stats.get("tier1_overflow_events", 0) + 1
                self.retrieval_stats["tier1_retained"] = self.retrieval_stats.get("tier1_retained", 0) + len(retained)
            else:
                retained = [cid for _, cid in tier1_scored]
                self.retrieval_stats["tier1_retained"] = self.retrieval_stats.get("tier1_retained", 0) + len(retained)
                remaining_slots = self.max_candidates - len(retained)
                tier2_to_add = [cid for _, cid in tier2_scored[:remaining_slots]]
                retained.extend(tier2_to_add)
                self.retrieval_stats["tier2_retained"] = self.retrieval_stats.get("tier2_retained", 0) + len(tier2_to_add)

            self._update_retrieval_stats(candidates_before_cap, len(retained))
            return retained, raw_candidates_uncapped

    def retrieve_candidates(self, s1_rec: dict) -> List[str]:
        """
        Retrieve candidate IDs for a Source 1 record across all 6 channels.
        If a primary block is oversized, retrieves from targeted sub-blocks.
        Applies Priority Tier Retention during pre-ranking to safeguard true matches.
        """
        return self.retrieve_candidates_with_uncapped(s1_rec)[0]

    def retrieve_candidates_with_channel_attribution(
        self, s1_rec: dict
    ) -> Tuple[List[str], Set[str], Dict[str, List[str]]]:
        """
        Retrieve candidate IDs across all 6 channels with channel attribution.
        Returns: (retained_candidates, raw_uncapped_candidates, candidate_to_channels_map)
        """
        retained, uncapped = self.retrieve_candidates_with_uncapped(s1_rec)
        return retained, uncapped, dict(self.last_candidate_channels)

    def get_block_statistics(self) -> Dict[str, Any]:
        """Return audit statistics of indexed blocks and size distribution."""
        return dict(self.last_block_statistics)

    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Return cumulative statistics on candidate generation before and after cap."""
        queries = self.retrieval_stats["total_queries"]
        before = self.retrieval_stats["total_candidates_before_cap"]
        after = self.retrieval_stats["total_candidates_after_cap"]
        return {
            "total_queries": queries,
            "total_candidates_before_cap": before,
            "total_candidates_after_cap": after,
            "avg_candidates_before_cap": round(before / queries, 2) if queries > 0 else 0.0,
            "avg_candidates_after_cap": round(after / queries, 2) if queries > 0 else 0.0,
            "max_candidates_before_cap": self.retrieval_stats["max_candidates_before_cap"],
            "max_candidates_after_cap": self.retrieval_stats["max_candidates_after_cap"],
            "capping_events": self.retrieval_stats.get("capping_events", 0),
            "tier1_retained": self.retrieval_stats.get("tier1_retained", 0),
            "tier2_retained": self.retrieval_stats.get("tier2_retained", 0),
            "tier1_overflow_events": self.retrieval_stats.get("tier1_overflow_events", 0),
            "capping_strategy": self.capping_strategy,
        }

    def log_block_statistics(self, custom_logger: Optional[logging.Logger] = None):
        """Log structured audit report of primary blocks, oversized blocks, and sub-blocks."""
        target_logger = custom_logger or logger
        stats = self.get_block_statistics()
        if not stats:
            target_logger.info("No block statistics available (index may be empty or not yet pruned).")
            return
        dist = stats.get("block_size_distribution", {})
        target_logger.info("============================================================")
        target_logger.info("BLOCKING ENGINE INDEX AUDIT:")
        target_logger.info(f"  Total Primary Blocks (6 Channels): {stats.get('total_primary_blocks', 0):,}")
        target_logger.info(f"  Oversized Primary Blocks:          {stats.get('oversized_blocks_count', 0):,}")
        target_logger.info(f"  Sub-Block Threshold:               {stats.get('sub_block_threshold', 0):,}")
        target_logger.info(f"  Total Sub-Blocks Constructed:      {stats.get('total_sub_blocks', 0):,}")
        target_logger.info("  Block Size Distribution:")
        target_logger.info(
            f"    Min: {dist.get('min', 0)} | 25th: {dist.get('p25', 0)} | "
            f"Median: {dist.get('median', 0)} | 75th: {dist.get('p75', 0)}"
        )
        target_logger.info(
            f"    95th: {dist.get('p95', 0)} | 99th: {dist.get('p99', 0)} | "
            f"Max: {dist.get('max', 0)} | Mean: {dist.get('mean', 0.0)}"
        )
        target_logger.info("============================================================")

    def log_retrieval_statistics(self, custom_logger: Optional[logging.Logger] = None):
        """Log cumulative candidate retrieval metrics before and after safety cap."""
        target_logger = custom_logger or logger
        stats = self.get_retrieval_statistics()
        target_logger.info("============================================================")
        target_logger.info("CANDIDATE RETRIEVAL & CAPPING AUDIT:")
        target_logger.info(f"  Total Queries Processed:           {stats.get('total_queries', 0):,}")
        target_logger.info(f"  Total Candidates Before Cap:       {stats.get('total_candidates_before_cap', 0):,}")
        target_logger.info(f"  Total Candidates After Cap:        {stats.get('total_candidates_after_cap', 0):,}")
        target_logger.info(f"  Average Candidates Before Cap:     {stats.get('avg_candidates_before_cap', 0.0):.2f}")
        target_logger.info(f"  Average Candidates After Cap:      {stats.get('avg_candidates_after_cap', 0.0):.2f}")
        target_logger.info(f"  Max Candidates Before Cap:         {stats.get('max_candidates_before_cap', 0):,}")
        target_logger.info(f"  Max Candidates After Cap:          {stats.get('max_candidates_after_cap', 0):,}")
        target_logger.info(f"  Queries Requiring Capping:         {stats.get('capping_events', 0):,}")
        target_logger.info(f"  Tier 1 Guaranteed Retained:        {stats.get('tier1_retained', 0):,}")
        target_logger.info(f"  Tier 2 Scored Retained:            {stats.get('tier2_retained', 0):,}")
        target_logger.info(f"  Tier 1 Overflows (Deterministic):  {stats.get('tier1_overflow_events', 0):,}")
        target_logger.info(f"  Configured Cap:                    {self.max_candidates}")
        target_logger.info(f"  Capping Strategy:                  {self.capping_strategy}")
        target_logger.info("============================================================")

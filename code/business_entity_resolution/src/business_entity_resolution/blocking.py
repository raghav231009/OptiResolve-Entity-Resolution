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
    ):
        if max_candidates_per_entity is not None:
            max_candidates = max_candidates_per_entity
        self.max_candidates = max_candidates
        self.max_candidates_per_entity = max_candidates
        self.min_token_len = min_token_len
        self.name_prefix_len = name_prefix_len
        self.max_block_size = max_block_size
        self.sub_block_threshold = sub_block_threshold if sub_block_threshold is not None else max_block_size

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

        # Fast lookup store for multi-signal pre-ranking: id -> (root_name, clean_address, postal, bldg_num)
        self.target_store: Dict[str, Tuple[str, str, str, str]] = {}

        # Statistics tracking
        self.last_block_statistics: Dict[str, Any] = {}
        self.retrieval_stats = {
            "total_queries": 0,
            "total_candidates_before_cap": 0,
            "total_candidates_after_cap": 0,
            "max_candidates_before_cap": 0,
            "max_candidates_after_cap": 0,
        }

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
            eid = rec["entity_id"]
            country = rec["country"].strip().upper()
            root_name = rec["root_name"]
            clean_addr = rec["clean_address"]
            postal = rec.get("postal_code", "")
            bldg_num = rec.get("building_number", "")

            self.target_store[eid] = (root_name, clean_addr, postal, bldg_num)

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

    def _update_retrieval_stats(self, before_cap: int, after_cap: int):
        self.retrieval_stats["total_queries"] += 1
        self.retrieval_stats["total_candidates_before_cap"] += before_cap
        self.retrieval_stats["total_candidates_after_cap"] += after_cap
        if before_cap > self.retrieval_stats["max_candidates_before_cap"]:
            self.retrieval_stats["max_candidates_before_cap"] = before_cap
        if after_cap > self.retrieval_stats["max_candidates_after_cap"]:
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

        def query_channel(channel_prefix: str, keys: List[str], primary_idx: Dict[str, List[str]], sub_qualifier: str):
            for k in keys:
                tagged_key = f"{channel_prefix}_{k}"
                if tagged_key in self.oversized_keys[country]:
                    sub_key = f"{channel_prefix}_{k}_{sub_qualifier}"
                    sub_cands = self.idx_sub_blocks[country].get(sub_key, [])
                    if sub_cands:
                        cand_set.update(sub_cands)
                    else:
                        # Fallback to bounded sample of primary block to guarantee recall
                        cand_set.update(primary_idx.get(k, [])[: self.max_candidates])
                else:
                    cand_set.update(primary_idx.get(k, []))

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

        raw_candidates_uncapped = set(cand_set)
        candidates_before_cap = len(cand_set)

        if not cand_set:
            self._update_retrieval_stats(0, 0)
            return [], set()

        if len(cand_set) <= self.max_candidates:
            self._update_retrieval_stats(candidates_before_cap, len(cand_set))
            return list(cand_set), raw_candidates_uncapped

        # Multi-Signal Similarity Pre-Ranking with Priority Tier Retention:
        s1_name = s1_rec["root_name"]
        s1_addr = s1_rec["clean_address"]
        s1_post = s1_rec.get("postal_code", "")
        s1_bldg = s1_rec.get("building_number", "")

        tier1_guaranteed: List[str] = []
        tier2_scored: List[Tuple[float, str]] = []

        for cid in cand_set:
            c_name, c_addr, c_post, c_bldg = self.target_store.get(cid, ("", "", "", ""))

            # Priority Tier 1: Exact root name match or exact physical building + postal match
            has_exact_name = bool(s1_name and s1_name == c_name)
            has_exact_physical = bool(s1_bldg and c_bldg and s1_bldg == c_bldg and s1_post and c_post and s1_post == c_post)

            if has_exact_name or has_exact_physical:
                tier1_guaranteed.append(cid)
                continue

            # Priority Tier 2: Composite scoring
            name_set_score = fuzz.token_set_ratio(s1_name, c_name)
            name_q_score = fuzz.QRatio(s1_name, c_name)
            addr_score = fuzz.token_set_ratio(s1_addr, c_addr) if (s1_addr and c_addr) else 0.0
            postal_bonus = 20.0 if (s1_post and c_post and s1_post == c_post) else 0.0
            bldg_bonus = 15.0 if (s1_bldg and c_bldg and s1_bldg == c_bldg) else 0.0

            composite_rank_score = (name_set_score * 0.40) + (name_q_score * 0.20) + (addr_score * 0.20) + postal_bonus + bldg_bonus
            tier2_scored.append((composite_rank_score, cid))

        # Always preserve Tier 1 guaranteed candidates
        remaining_slots = max(0, self.max_candidates - len(tier1_guaranteed))
        tier2_scored.sort(key=lambda x: x[0], reverse=True)
        retained = tier1_guaranteed + [cid for _, cid in tier2_scored[:remaining_slots]]
        candidates_after_cap = min(len(retained), self.max_candidates)

        self._update_retrieval_stats(candidates_before_cap, candidates_after_cap)
        return retained[: self.max_candidates], raw_candidates_uncapped

    def retrieve_candidates(self, s1_rec: dict) -> List[str]:
        """
        Retrieve candidate IDs for a Source 1 record across all 6 channels.
        If a primary block is oversized, retrieves from targeted sub-blocks.
        Applies Priority Tier Retention during pre-ranking to safeguard true matches.
        """
        return self.retrieve_candidates_with_uncapped(s1_rec)[0]

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
        target_logger.info(f"  Configured Cap:                    {self.max_candidates}")
        target_logger.info("============================================================")

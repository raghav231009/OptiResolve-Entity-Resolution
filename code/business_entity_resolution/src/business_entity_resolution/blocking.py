"""
High-Recall Multi-Index Inverted Blocking Engine.
Partitions by open-set country, employs sub-blocking instead of hard deletion for large blocks,
and utilizes multi-signal similarity pre-ranking with Priority Tier Retention to safeguard candidate recall.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import rapidfuzz.fuzz as fuzz

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
    ):
        self.max_candidates = max_candidates
        self.min_token_len = min_token_len
        self.name_prefix_len = name_prefix_len
        self.max_block_size = max_block_size

        # Inverted index tables: {country: {key: [entity_id, ...]}}
        self.idx_name_token: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.idx_name_prefix: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.idx_postal: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.idx_addr_anchor: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.idx_two_word: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.idx_street_anchor: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        # Sub-blocking tables for large blocks: {country: {f"{key}_{sub}": [entity_id, ...]}}
        self.idx_sub_blocks: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.oversized_keys: Dict[str, Set[str]] = defaultdict(set)

        # Fast lookup store for multi-signal pre-ranking: id -> (root_name, clean_address, postal, bldg_num)
        self.target_store: Dict[str, Tuple[str, str, str, str]] = {}

    def extract_keys(self, record: dict) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], str]:
        """Extract 6-channel blocking keys and sub-block qualifier from normalized fields."""
        root_name = record.get("root_name", "")
        clean_addr = record.get("clean_address", "")
        postal = record.get("postal_code", "")
        bldg_num = record.get("building_number", "")

        # Channel 1: Significant Name Tokens
        name_tokens = []
        name_words = [tok for tok in root_name.split() if len(tok) >= self.min_token_len and tok not in STOPWORDS]
        if name_words:
            name_tokens = name_words[:2]

        # Channel 2: Name Prefix
        prefixes = []
        if len(root_name) >= self.name_prefix_len:
            prefixes.append(root_name[: self.name_prefix_len])

        # Channel 3: Postal Code
        postals = [postal] if postal else []

        # Channel 4: Address Building + First Street Token Prefix Anchor
        addr_anchors = []
        addr_words = [w for w in clean_addr.split() if len(w) >= 3 and not w.isdigit() and w not in STOPWORDS]
        if bldg_num and addr_words:
            first_word_prefix = addr_words[0][:4]
            addr_anchors.append(f"{bldg_num}_{first_word_prefix}")

        # Channel 5: Two-Word Name Anchor (order invariant for compound brand names)
        two_word_keys = []
        if len(name_words) >= 2:
            two_word_keys.append(f"{name_words[0]}_{name_words[1]}")

        # Channel 6: Street Name Anchor (recovers garbled/transliterated brand names)
        street_anchors = []
        if len(addr_words) >= 2:
            street_anchors.append(f"{addr_words[0]}_{addr_words[1]}")

        # Sub-block qualifier (postal 2-digit prefix or address 2-char prefix)
        sub_qualifier = postal[:2] if postal else (clean_addr[:2] if clean_addr else "xx")

        return name_tokens, prefixes, postals, addr_anchors, two_word_keys, street_anchors, sub_qualifier

    def index_targets(self, records: List[dict]):
        """Build inverted index from Source 2 and Source 3 target records."""
        for rec in records:
            eid = rec["entity_id"]
            country = rec["country"].strip().upper()
            root_name = rec["root_name"]
            clean_addr = rec["clean_address"]
            postal = rec.get("postal_code", "")
            bldg_num = rec.get("building_number", "")

            self.target_store[eid] = (root_name, clean_addr, postal, bldg_num)

            name_tokens, prefixes, postals, addr_anchors, two_words, street_anchors, sub_q = self.extract_keys(rec)

            for tok in name_tokens:
                self.idx_name_token[country][tok].append(eid)
                self.idx_sub_blocks[country][f"tok_{tok}_{sub_q}"].append(eid)

            for pref in prefixes:
                self.idx_name_prefix[country][pref].append(eid)
                self.idx_sub_blocks[country][f"pref_{pref}_{sub_q}"].append(eid)

            for post in postals:
                self.idx_postal[country][post].append(eid)

            for anchor in addr_anchors:
                self.idx_addr_anchor[country][anchor].append(eid)

            for tw in two_words:
                self.idx_two_word[country][tw].append(eid)

            for sa in street_anchors:
                self.idx_street_anchor[country][sa].append(eid)
                self.idx_sub_blocks[country][f"street_{sa}_{sub_q}"].append(eid)

    def prune_large_blocks(self):
        """
        Identify oversized blocks across all channels and mark them for
        secondary sub-blocking rather than destructive deletion.
        No blocks are ever deleted — oversized blocks are routed to finer
        sub-block partitions at retrieval time.
        """
        for country, keys in self.idx_name_token.items():
            for key in list(keys.keys()):
                if len(keys[key]) > self.max_block_size:
                    self.oversized_keys[country].add(f"tok_{key}")

        for country, keys in self.idx_name_prefix.items():
            for key in list(keys.keys()):
                if len(keys[key]) > self.max_block_size:
                    self.oversized_keys[country].add(f"pref_{key}")

        for country, keys in self.idx_street_anchor.items():
            for key in list(keys.keys()):
                if len(keys[key]) > self.max_block_size:
                    self.oversized_keys[country].add(f"street_{key}")

    def retrieve_candidates(self, s1_rec: dict) -> List[str]:
        """
        Retrieve candidate IDs for a Source 1 record across all 6 channels.
        If a primary block is oversized, retrieves from targeted sub-blocks.
        Applies Priority Tier Retention during pre-ranking to safeguard true matches.
        """
        country = s1_rec["country"].strip().upper()
        name_tokens, prefixes, postals, addr_anchors, two_words, street_anchors, sub_q = self.extract_keys(s1_rec)

        cand_set: Set[str] = set()

        # Channel 1: Name Tokens (with sub-block routing for oversized blocks)
        for tok in name_tokens:
            if f"tok_{tok}" in self.oversized_keys[country]:
                cand_set.update(self.idx_sub_blocks[country].get(f"tok_{tok}_{sub_q}", []))
            else:
                cand_set.update(self.idx_name_token[country].get(tok, []))

        # Channel 2: Name Prefixes (with sub-block routing)
        for pref in prefixes:
            if f"pref_{pref}" in self.oversized_keys[country]:
                cand_set.update(self.idx_sub_blocks[country].get(f"pref_{pref}_{sub_q}", []))
            else:
                cand_set.update(self.idx_name_prefix[country].get(pref, []))

        # Channel 3: Postal Code
        for post in postals:
            cand_set.update(self.idx_postal[country].get(post, []))

        # Channel 4: Address Anchor
        for anchor in addr_anchors:
            cand_set.update(self.idx_addr_anchor[country].get(anchor, []))

        # Channel 5: Two-Word Brand Anchor
        for tw in two_words:
            cand_set.update(self.idx_two_word[country].get(tw, []))

        # Channel 6: Street Name Anchor (with sub-block routing for oversized blocks)
        for sa in street_anchors:
            if f"street_{sa}" in self.oversized_keys[country]:
                cand_set.update(self.idx_sub_blocks[country].get(f"street_{sa}_{sub_q}", []))
            else:
                cand_set.update(self.idx_street_anchor[country].get(sa, []))

        if not cand_set:
            return []

        if len(cand_set) <= self.max_candidates:
            return list(cand_set)

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

        return retained[: self.max_candidates]

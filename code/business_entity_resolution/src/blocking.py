"""
High-Recall Multi-Index Inverted Blocking Engine.
Partitions by open-set country, employs sub-blocking instead of hard deletion for large blocks,
and utilizes multi-signal similarity pre-ranking to safeguard candidate recall.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import rapidfuzz.fuzz as fuzz

STOPWORDS = {"the", "and", "dr", "all", "new", "mr", "mrs", "miss", "les", "des", "une"}


class MultiIndexBlocker:
    """
    Multi-channel inverted index for candidate generation.
    Supports dynamic, open-set country partitioning and secondary sub-blocking.
    """

    def __init__(
        self,
        max_candidates: int = 60,
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

        # Sub-blocking tables for large blocks: {country: {f"{key}_{sub}": [entity_id, ...]}}
        self.idx_sub_blocks: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.oversized_keys: Dict[str, Set[str]] = defaultdict(set)

        # Fast lookup store for multi-signal pre-ranking during capping
        self.target_store: Dict[str, Tuple[str, str, str]] = {}  # id -> (root_name, clean_address, postal)

    def extract_keys(self, record: dict) -> Tuple[List[str], List[str], List[str], List[str], str]:
        """Extract multi-channel blocking keys and sub-block qualifier from normalized fields."""
        root_name = record.get("root_name", "")
        clean_addr = record.get("clean_address", "")
        postal = record.get("postal_code", "")
        bldg_num = record.get("building_number", "")

        # Channel 1: Significant Name Tokens
        name_tokens = []
        for tok in root_name.split():
            if len(tok) >= self.min_token_len and tok not in STOPWORDS:
                name_tokens.append(tok)
                if len(name_tokens) >= 2:
                    break

        # Channel 2: Name Prefix
        prefixes = []
        if len(root_name) >= self.name_prefix_len:
            prefixes.append(root_name[: self.name_prefix_len])

        # Channel 3: Postal Code
        postals = [postal] if postal else []

        # Channel 4: Address Anchor (Building Number + First Street Token Prefix)
        addr_anchors = []
        if bldg_num and clean_addr:
            addr_words = [w for w in clean_addr.split() if len(w) >= 3 and not w.isdigit() and w not in STOPWORDS]
            if addr_words:
                first_word_prefix = addr_words[0][:4]
                addr_anchors.append(f"{bldg_num}_{first_word_prefix}")

        # Sub-block qualifier (first char of address or first digit of postal)
        sub_qualifier = postal[:2] if postal else (clean_addr[:2] if clean_addr else "xx")

        return name_tokens, prefixes, postals, addr_anchors, sub_qualifier

    def index_targets(self, records: List[dict]):
        """Build inverted index from Source 2 and Source 3 target records."""
        for rec in records:
            eid = rec["entity_id"]
            country = rec["country"].strip().upper()
            root_name = rec["root_name"]
            clean_addr = rec["clean_address"]
            postal = rec.get("postal_code", "")

            self.target_store[eid] = (root_name, clean_addr, postal)

            name_tokens, prefixes, postals, addr_anchors, sub_q = self.extract_keys(rec)

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

    def prune_large_blocks(self):
        """
        Identify oversized blocks and mark them for secondary sub-blocking
        rather than destructive deletion.
        """
        for country, keys in self.idx_name_token.items():
            for key in list(keys.keys()):
                if len(keys[key]) > self.max_block_size:
                    self.oversized_keys[country].add(f"tok_{key}")

        for country, keys in self.idx_name_prefix.items():
            for key in list(keys.keys()):
                if len(keys[key]) > self.max_block_size:
                    self.oversized_keys[country].add(f"pref_{key}")

    def retrieve_candidates(self, s1_rec: dict) -> List[str]:
        """
        Retrieve candidate IDs for a Source 1 record.
        If a primary block is oversized, retrieves from targeted sub-blocks.
        Applies multi-signal similarity pre-ranking when candidates exceed the safety cap.
        """
        country = s1_rec["country"].strip().upper()
        name_tokens, prefixes, postals, addr_anchors, sub_q = self.extract_keys(s1_rec)

        cand_set: Set[str] = set()

        # Query Name Tokens: use main block if normal-sized, else sub-block
        for tok in name_tokens:
            if f"tok_{tok}" in self.oversized_keys[country]:
                cand_set.update(self.idx_sub_blocks[country].get(f"tok_{tok}_{sub_q}", []))
            else:
                cand_set.update(self.idx_name_token[country].get(tok, []))

        # Query Name Prefixes
        for pref in prefixes:
            if f"pref_{pref}" in self.oversized_keys[country]:
                cand_set.update(self.idx_sub_blocks[country].get(f"pref_{pref}_{sub_q}", []))
            else:
                cand_set.update(self.idx_name_prefix[country].get(pref, []))

        # Query Postal Code
        for post in postals:
            cand_set.update(self.idx_postal[country].get(post, []))

        # Query Address Anchors
        for anchor in addr_anchors:
            cand_set.update(self.idx_addr_anchor[country].get(anchor, []))

        if not cand_set:
            return []

        if len(cand_set) <= self.max_candidates:
            return list(cand_set)

        # Multi-Signal Similarity Pre-Ranking:
        # Blends Token-Set Ratio, QRatio, Address Token Overlap, and Postal Matches
        s1_name = s1_rec["root_name"]
        s1_addr = s1_rec["clean_address"]
        s1_post = s1_rec.get("postal_code", "")

        scored_candidates = []
        for cid in cand_set:
            c_name, c_addr, c_post = self.target_store.get(cid, ("", "", ""))
            name_set_score = fuzz.token_set_ratio(s1_name, c_name)
            name_q_score = fuzz.QRatio(s1_name, c_name)
            addr_score = fuzz.token_set_ratio(s1_addr, c_addr) if (s1_addr and c_addr) else 0.0
            postal_bonus = 15.0 if (s1_post and c_post and s1_post == c_post) else 0.0

            composite_rank_score = (name_set_score * 0.45) + (name_q_score * 0.25) + (addr_score * 0.20) + postal_bonus
            scored_candidates.append((composite_rank_score, cid))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [cid for _, cid in scored_candidates[: self.max_candidates]]

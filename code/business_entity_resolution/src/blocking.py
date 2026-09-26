"""
High-Recall Multi-Index Inverted Blocking Engine.
Partitions by open-set country and indexes complementary name and address channels.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import rapidfuzz.fuzz as fuzz

STOPWORDS = {"the", "and", "dr", "all", "new", "mr", "mrs", "miss", "les", "des", "une"}


class MultiIndexBlocker:
    """
    Multi-channel inverted index for candidate generation.
    Supports dynamic, open-set country partitioning.
    """

    def __init__(
        self,
        max_candidates: int = 40,
        min_token_len: int = 3,
        name_prefix_len: int = 4,
        max_block_size: int = 500,
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

        # Fast lookup store for pre-ranking candidates during capping
        self.target_store: Dict[str, Tuple[str, str]] = {}  # id -> (root_name, clean_address)

    def extract_keys(self, record: dict) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Extract multi-channel blocking keys from normalized fields."""
        root_name = record["root_name"]
        clean_addr = record["clean_address"]
        postal = record["postal_code"]
        num_tokens = record["numeric_tokens"]

        # Channel 1: Significant Name Tokens
        name_tokens = []
        for tok in root_name.split():
            if len(tok) >= self.min_token_len and tok not in STOPWORDS:
                name_tokens.append(tok)
                if len(name_tokens) >= 2:  # take first 2 significant tokens
                    break

        # Channel 2: Name Prefix
        prefixes = []
        if len(root_name) >= self.name_prefix_len:
            prefixes.append(root_name[: self.name_prefix_len])

        # Channel 3: Postal Code
        postals = [postal] if postal else []

        # Channel 4: Address Anchor (Street/Building Number + First Street Token Prefix)
        addr_anchors = []
        if num_tokens and clean_addr:
            addr_words = [w for w in clean_addr.split() if len(w) >= 3 and not w.isdigit() and w not in STOPWORDS]
            if addr_words:
                first_num = sorted(list(num_tokens))[0]
                first_word_prefix = addr_words[0][:4]
                addr_anchors.append(f"{first_num}_{first_word_prefix}")

        return name_tokens, prefixes, postals, addr_anchors

    def index_targets(self, records: List[dict]):
        """Build inverted index from Source 2 and Source 3 target records."""
        for rec in records:
            eid = rec["entity_id"]
            country = rec["country"].strip().upper()
            root_name = rec["root_name"]
            clean_addr = rec["clean_address"]

            self.target_store[eid] = (root_name, clean_addr)

            name_tokens, prefixes, postals, addr_anchors = self.extract_keys(rec)

            for tok in name_tokens:
                self.idx_name_token[country][tok].append(eid)

            for pref in prefixes:
                self.idx_name_prefix[country][pref].append(eid)

            for post in postals:
                self.idx_postal[country][post].append(eid)

            for anchor in addr_anchors:
                self.idx_addr_anchor[country][anchor].append(eid)

    def prune_large_blocks(self):
        """Remove overly broad keys (e.g. ultra-common generic words) to avoid explosive noise."""
        for country_idx in [self.idx_name_token, self.idx_name_prefix, self.idx_postal, self.idx_addr_anchor]:
            for country, keys in country_idx.items():
                for key in list(keys.keys()):
                    if len(keys[key]) > self.max_block_size:
                        del keys[key]

    def retrieve_candidates(self, s1_rec: dict) -> List[str]:
        """Retrieve candidate IDs for a Source 1 record, applying intelligent similarity pre-ranking if capped."""
        country = s1_rec["country"].strip().upper()
        name_tokens, prefixes, postals, addr_anchors = self.extract_keys(s1_rec)

        cand_set: Set[str] = set()

        # Query all channels
        for tok in name_tokens:
            cand_set.update(self.idx_name_token[country].get(tok, []))

        for pref in prefixes:
            cand_set.update(self.idx_name_prefix[country].get(pref, []))

        for post in postals:
            cand_set.update(self.idx_postal[country].get(post, []))

        for anchor in addr_anchors:
            cand_set.update(self.idx_addr_anchor[country].get(anchor, []))

        if not cand_set:
            return []

        # If candidate pool is within safety limit, return directly
        if len(cand_set) <= self.max_candidates:
            return list(cand_set)

        # Smart pre-ranking: Sort by fast quick_ratio to protect true matches from arbitrary truncation
        s1_name = s1_rec["root_name"]
        s1_addr = s1_rec["clean_address"]

        scored_candidates = []
        for cid in cand_set:
            c_name, c_addr = self.target_store.get(cid, ("", ""))
            name_score = fuzz.QRatio(s1_name, c_name)
            addr_score = fuzz.QRatio(s1_addr, c_addr) if (s1_addr and c_addr) else 0.0
            total_score = name_score * 0.65 + addr_score * 0.35
            scored_candidates.append((total_score, cid))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [cid for _, cid in scored_candidates[: self.max_candidates]]

"""
Feature Engineering Engine for Entity Resolution.
Computes fast, C-accelerated string metrics, character n-gram similarities,
and explicit component-level address alignments.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import rapidfuzz.distance.JaroWinkler as jw
import rapidfuzz.fuzz as fuzz

FEATURE_NAMES = [
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "name_char3_jaccard",
    "name_char4_jaccard",
    "name_token_containment",
    "name_len_diff_ratio",
    "name_exact_match",
    "root_name_exact_match",
    "addr_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_jaccard",
    "addr_present_both",
    "house_number_match",
    "postal_exact_match",
    "postal_prefix_match",
    "numeric_token_overlap",
    "target_source_is_s3",
    "country_match",
    "combined_weighted_sim",
]


def compute_char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Compute character n-gram Jaccard similarity."""
    s1 = s1 or ""
    s2 = s2 or ""
    if not s1 or not s2:
        return 0.0
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 else 0.0
    ngrams1 = {s1[i : i + n] for i in range(len(s1) - n + 1)}
    ngrams2 = {s2[i : i + n] for i in range(len(s2) - n + 1)}
    inter = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    return inter / union if union > 0 else 0.0


def compute_token_jaccard(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute token Jaccard similarity."""
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_token_containment(tokens1: List[str], tokens2: List[str]) -> float:
    """1.0 if tokens of shorter string are a subset of the longer string."""
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 or not set2:
        return 0.0
    shorter, longer = (set1, set2) if len(set1) <= len(set2) else (set2, set1)
    return 1.0 if shorter.issubset(longer) else 0.0


def compute_pair_features(s1_rec: dict, cand_rec: dict) -> List[float]:
    """
    Extract a deterministic, robust 23-dimensional numerical feature vector for an (S1, Candidate) pair:
    - NAME (1-11): ratio, partial_ratio, token_sort_ratio, token_set_ratio, jaro_winkler,
                   char3_jaccard, char4_jaccard, token_containment, len_diff_ratio,
                   name_exact_match, root_name_exact_match
    - ADDRESS (12-16): addr_ratio, addr_token_sort_ratio, addr_token_set_ratio, addr_jaccard, addr_present_both
    - STRUCTURAL (17-20): house_number_match, postal_exact_match, postal_prefix_match, numeric_token_overlap
    - META (21-22): target_source_is_s3, country_match
    - COMPOSITE (23): combined_weighted_sim
    """
    s1_name = str(s1_rec.get("clean_name", "") or "")
    s1_root = str(s1_rec.get("root_name", "") or "")
    s1_addr = str(s1_rec.get("clean_address", "") or "")
    s1_post = str(s1_rec.get("postal_code", "") or "")
    s1_bldg = str(s1_rec.get("building_number", "") or "")
    s1_nums = set(s1_rec.get("numeric_tokens", set()) or set())
    s1_country = str(s1_rec.get("country", "") or "").strip().upper()

    c_id = str(cand_rec.get("entity_id", "") or "")
    c_name = str(cand_rec.get("clean_name", "") or "")
    c_root = str(cand_rec.get("root_name", "") or "")
    c_addr = str(cand_rec.get("clean_address", "") or "")
    c_post = str(cand_rec.get("postal_code", "") or "")
    c_bldg = str(cand_rec.get("building_number", "") or "")
    c_nums = set(cand_rec.get("numeric_tokens", set()) or set())
    c_country = str(cand_rec.get("country", "") or "").strip().upper()

    # 1. Name String Distances & Lexical Features
    n_ratio = float(fuzz.ratio(s1_name, c_name)) / 100.0
    n_partial = float(fuzz.partial_ratio(s1_name, c_name)) / 100.0
    n_sort = float(fuzz.token_sort_ratio(s1_name, c_name)) / 100.0
    n_set = float(fuzz.token_set_ratio(s1_name, c_name)) / 100.0
    n_jw = float(jw.similarity(s1_name, c_name))

    # Character n-gram similarities (robust against typos & transliterations)
    n_char3 = compute_char_ngram_jaccard(s1_name, c_name, n=3)
    n_char4 = compute_char_ngram_jaccard(s1_name, c_name, n=4)
    n_contain = compute_token_containment(s1_root.split(), c_root.split())

    max_len = max(len(s1_name), len(c_name), 1)
    n_len_diff = abs(len(s1_name) - len(c_name)) / max_len
    n_exact = 1.0 if (s1_name and s1_name == c_name) else 0.0
    root_exact = 1.0 if (s1_root and s1_root == c_root) else 0.0

    # 2. Address Distances (handling missing values cleanly)
    has_both_addr = bool(s1_addr and c_addr)
    if has_both_addr:
        a_ratio = fuzz.ratio(s1_addr, c_addr) / 100.0
        a_sort = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0
        a_set = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0
        a_jaccard = compute_token_jaccard(s1_addr.split(), c_addr.split())
    else:
        a_ratio = 0.0
        a_sort = 0.0
        a_set = 0.0
        a_jaccard = 0.0

    addr_present = 1.0 if has_both_addr else 0.0

    # 3. Explicit Component-Level Matches
    # House/Building number match
    if s1_bldg and c_bldg:
        house_match = 1.0 if s1_bldg == c_bldg else 0.0
    else:
        house_match = -1.0  # missing building signal

    # Postal exact and prefix matches
    if s1_post and c_post:
        p_exact = 1.0 if s1_post == c_post else 0.0
        p_prefix = 1.0 if s1_post[:2] == c_post[:2] else 0.0
    else:
        p_exact = -1.0
        p_prefix = -1.0

    # Numeric token overlap (plot numbers, units)
    if s1_nums and c_nums:
        num_inter = len(s1_nums & c_nums)
        num_union = len(s1_nums | c_nums)
        num_overlap = num_inter / num_union if num_union > 0 else 0.0
    else:
        num_overlap = 0.0

    is_s3 = 1.0 if c_id.startswith("S3-") else 0.0
    country_eq = 1.0 if s1_country == c_country else 0.0
    comb_score = (0.65 * n_set) + (0.35 * a_set) if has_both_addr else n_set

    return [
        n_ratio,
        n_partial,
        n_sort,
        n_set,
        n_jw,
        n_char3,
        n_char4,
        n_contain,
        n_len_diff,
        n_exact,
        root_exact,
        a_ratio,
        a_sort,
        a_set,
        a_jaccard,
        addr_present,
        house_match,
        p_exact,
        p_prefix,
        num_overlap,
        is_s3,
        country_eq,
        comb_score,
    ]

"""
Multilingual Preprocessing & Normalization Engine.
Supports US, India, and France text characteristics with Unicode NFKD decomposition.
"""

import re
import unicodedata
from typing import Set, Tuple

# Missing value sentinels that represent empty/uninformative data
MISSING_SENTINELS = {
    "", "nan", "none", "null", "n/a", "na", "-", ".", "undefined", "unknown"
}

# Comprehensive corporate suffix patterns (applied after lowercasing).
# NOTE: 'fils' is intentionally excluded from the main pattern and handled
# separately via TRAILING_FILS_REGEX to avoid stripping from mid-name brand
# tokens like 'Le Fils Dupont' (a legitimate French family-business name).
LEGAL_SUFFIX_REGEX = re.compile(
    r"\b("
    # Multi-word Indian suffixes (must precede shorter overlaps)
    r"private limited|pvt ltd|pvt\.?\s*ltd\.?|"
    r"limited liability company|limited|ltd\.?|"
    r"incorporated|inc\.?|corporation|corp\.?|"
    r"company|co\.?|llc|llp|plc|gmbh|"
    # French corporate forms (fils excluded here)
    r"sarl|sas|sasu|sa|eurl|sci|snc|gie"
    r")\b",
    re.IGNORECASE,
)

# Strip 'fils' / 'et fils' / '& fils' only when trailing (end-of-name position)
# to avoid falsely stripping 'Le Fils Dupont'-style brand names.
TRAILING_FILS_REGEX = re.compile(
    r"(\s+(?:et\s+|and\s+|&\s+)?fils)\s*$",
    re.IGNORECASE,
)

# Address abbreviations mapping
ADDRESS_ABBR_MAP = {
    # US / General
    "st": "street", "st.": "street",
    "rd": "road", "rd.": "road",
    "ave": "avenue", "ave.": "avenue",
    "blvd": "boulevard", "blvd.": "boulevard",
    "dr": "drive", "dr.": "drive",
    "ln": "lane", "ln.": "lane",
    "ct": "court", "ct.": "court",
    "hwy": "highway", "hwy.": "highway",
    "pkwy": "parkway", "pkwy.": "parkway",
    "sq": "square", "sq.": "square",
    "apt": "apartment", "ste": "suite", "bldg": "building", "fl": "floor",
    # Indian noise tokens
    "nr": "near", "nr.": "near",
    "opp": "opposite", "opp.": "opposite",
    # French street abbreviations
    "r": "rue", "r.": "rue",
    "bd": "boulevard", "bvd": "boulevard", "bd.": "boulevard",
    "av": "avenue", "av.": "avenue",
    "pl": "place", "pl.": "place",
    "all": "allee", "all.": "allee",
    "imp": "impasse", "imp.": "impasse",
    "rte": "route", "rte.": "route",
    "ch": "chemin", "ch.": "chemin",
    "crs": "cours", "crs.": "cours",
    "bat": "batiment", "bat.": "batiment",
    "etg": "etage", "etg.": "etage",
}

PUNCT_REGEX = re.compile(r"[^\w\s\u0900-\u0D7F]")
WHITESPACE_REGEX = re.compile(r"\s+")
POSTAL_5_6_DIGIT_REGEX = re.compile(r"\b\d{5,6}\b")
POSTAL_4_DIGIT_REGEX = re.compile(r"\b\d{4}\b")
DIGIT_TOKEN_REGEX = re.compile(r"\b\d+[a-zA-Z]?\b")
SUITE_KEYWORD_REGEX = re.compile(
    r"\b(?:suite|ste|apt|apartment|unit|bldg|building|floor|fl|room|rm|box|po\s*box)\b\.?\s*(\d+[a-zA-Z]?)",
    re.IGNORECASE,
)


def _is_latin_accent(c: str) -> bool:
    """Check if character is a Latin/European combining diacritical mark (not an Indic vowel sign)."""
    cp = ord(c)
    return (
        (0x0300 <= cp <= 0x036F)
        or (0x1AB0 <= cp <= 0x1AFF)
        or (0x1DC0 <= cp <= 0x1DFF)
        or (0x20D0 <= cp <= 0x20FF)
        or (0xFE20 <= cp <= 0xFE2F)
    )


def strip_accents_and_normalize(text: str) -> str:
    """
    Normalize unicode, strip accents (NFKD), lowercase, and canonicalize symbols.
    Handles French ligatures (œ -> oe, æ -> ae), removes zero-width spaces,
    preserves non-Latin script characters (Devanagari matras), and returns empty
    string for missing value sentinels.
    """
    if not text or not isinstance(text, str):
        return ""
    if text.strip().lower() in MISSING_SENTINELS:
        return ""

    # Normalize ligatures and invisible characters before decomposition
    cleaned = text.replace("œ", "oe").replace("Œ", "oe")
    cleaned = cleaned.replace("æ", "ae").replace("Æ", "ae")
    cleaned = cleaned.replace("\u200b", "").replace("\u00ad", "")

    decomposed = unicodedata.normalize("NFKD", cleaned)
    stripped = "".join(c for c in decomposed if not (unicodedata.combining(c) and _is_latin_accent(c)))
    cleaned = stripped.lower()
    cleaned = cleaned.replace("&", " and ")
    cleaned = PUNCT_REGEX.sub(" ", cleaned)
    return WHITESPACE_REGEX.sub(" ", cleaned).strip()


def normalize_address(address: str) -> str:
    """Normalize address string and expand common directional / type abbreviations."""
    base_clean = strip_accents_and_normalize(address)
    if not base_clean:
        return ""
    tokens = base_clean.split()
    expanded = [ADDRESS_ABBR_MAP.get(tok, tok) for tok in tokens]
    return " ".join(expanded)


def clean_business_name(name: str) -> Tuple[str, str]:
    """
    Returns (cleaned_name, root_name_without_legal_suffixes).

    Applies two-pass suffix stripping:
      1. Main LEGAL_SUFFIX_REGEX removes common legal suffixes (LLC, Ltd, SARL, etc.)
      2. TRAILING_FILS_REGEX removes trailing 'fils'/'et fils' only at end-of-name
         (avoids falsely stripping 'Le Fils Dupont'-style brand names).

    Example: 'Maure Williams Colombier Inc' -> ('maure williams colombier inc', 'maure williams colombier')
    Example: 'Dupont et Fils SARL' -> ('dupont et fils sarl', 'dupont')
    Example: 'Le Fils Dupont' -> ('le fils dupont', 'le fils dupont')  # fils NOT stripped mid-name
    """
    cleaned = strip_accents_and_normalize(name)
    if not cleaned:
        return "", ""
    # Pass 1: strip common legal suffixes
    root = LEGAL_SUFFIX_REGEX.sub(" ", cleaned)
    # Pass 2: strip trailing 'fils'/'et fils' only at end
    root = TRAILING_FILS_REGEX.sub("", root)
    root = WHITESPACE_REGEX.sub(" ", root).strip()
    return cleaned, root if root else cleaned


def extract_postal_code(address: str) -> str:
    """
    Extract 5-6 digit numeric postal / PIN code from address.
    Prioritizes 5-6 digit codes (US ZIP 5-digit, France 5-digit, India PIN 6-digit),
    selecting the trailing postal code to avoid 4-digit street number collisions.
    """
    if not address or not isinstance(address, str):
        return ""
    # Look for 5-6 digit postal codes first (all 3 countries: US 5-digit, FR 5-digit, IN 6-digit)
    matches_5_6 = POSTAL_5_6_DIGIT_REGEX.findall(address)
    if matches_5_6:
        return matches_5_6[-1]
    # Fallback to 4-digit codes only if not at the start (building numbers appear at index 0)
    matches_4 = POSTAL_4_DIGIT_REGEX.findall(address)
    if matches_4:
        first_match_start = address.find(matches_4[-1])
        if first_match_start > 0:
            return matches_4[-1]
    return ""


def extract_numeric_tokens(address: str) -> Set[str]:
    """Extract set of street numbers, plot codes, or numeric tokens."""
    if not address or not isinstance(address, str):
        return set()
    return set(DIGIT_TOKEN_REGEX.findall(address.lower()))


def extract_building_number(address: str, postal_code: str = "") -> str:
    """
    Extract the primary building / house / street number, explicitly distinct from
    the postal code and internal unit/suite numbers.
    E.g. '85 Wayne Ave, NY 12883' -> '85' (not '12883').
    E.g. '1200 Main St, Suite 400, NY 10001' -> '1200' (not '400' or '10001').
    """
    if not address or not isinstance(address, str):
        return ""
    addr_lower = address.lower()
    suite_tokens = set(SUITE_KEYWORD_REGEX.findall(addr_lower))
    tokens = DIGIT_TOKEN_REGEX.findall(addr_lower)
    for tok in tokens:
        # Avoid picking the postal code as the building number
        if postal_code and tok == postal_code:
            continue
        # Avoid picking suite / unit numbers as the primary street building number
        if tok in suite_tokens:
            continue
        # Standard building numbers are typically 1 to 5 alphanumeric characters
        if len(tok) <= 5:
            return tok
    return ""

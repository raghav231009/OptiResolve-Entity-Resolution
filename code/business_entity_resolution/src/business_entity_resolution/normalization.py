"""
Multilingual Preprocessing & Normalization Engine.
Supports US, India, and France text characteristics with Unicode NFKD decomposition.
"""

import re
import unicodedata
from typing import Set, Tuple

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
}

PUNCT_REGEX = re.compile(r"[^\w\s]")
WHITESPACE_REGEX = re.compile(r"\s+")
POSTAL_CODE_REGEX = re.compile(r"\b\d{4,6}\b")
DIGIT_TOKEN_REGEX = re.compile(r"\b\d+[a-zA-Z]?\b")


def strip_accents_and_normalize(text: str) -> str:
    """Normalize unicode, strip accents (NFKD), lowercase, and canonicalize symbols."""
    if not text or not isinstance(text, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
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
      1. Main LEGAL_SUFFIX_REGEX removes most common suffixes (LLC, Ltd, SARL, etc.)
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
    """Extract 4-6 digit numeric postal / PIN code from address."""
    if not address or not isinstance(address, str):
        return ""
    match = POSTAL_CODE_REGEX.search(address)
    return match.group(0) if match else ""


def extract_numeric_tokens(address: str) -> Set[str]:
    """Extract set of street numbers, plot codes, or numeric tokens."""
    if not address or not isinstance(address, str):
        return set()
    return set(DIGIT_TOKEN_REGEX.findall(address.lower()))


def extract_building_number(address: str, postal_code: str = "") -> str:
    """
    Extract the primary building / house / street number, explicitly distinct from the postal code.
    E.g. '85 Wayne Ave, NY 12883' -> '85' (not '12883').
    """
    if not address or not isinstance(address, str):
        return ""
    tokens = DIGIT_TOKEN_REGEX.findall(address.lower())
    for tok in tokens:
        # Avoid picking the postal code as the building number
        if postal_code and tok == postal_code:
            continue
        # Standard building numbers are typically 1 to 4 digits
        if len(tok) <= 5:
            return tok
    return ""

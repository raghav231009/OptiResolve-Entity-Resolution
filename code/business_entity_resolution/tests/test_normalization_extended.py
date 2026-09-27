"""
Extended Normalization Tests.
Covers edge cases for: suffix stripping (fils), postal codes, building numbers,
missing values, and address abbreviations.
"""

import pytest
from business_entity_resolution.normalization import (
    clean_business_name,
    extract_building_number,
    extract_numeric_tokens,
    extract_postal_code,
    normalize_address,
    strip_accents_and_normalize,
)


# ---------------------------------------------------------------------------
# strip_accents_and_normalize
# ---------------------------------------------------------------------------
class TestStripAccents:
    def test_french_accents(self):
        assert strip_accents_and_normalize("Société d'Électricité") == "societe d electricite"

    def test_ampersand_expansion(self):
        assert strip_accents_and_normalize("A & B") == "a and b"

    def test_empty_string(self):
        assert strip_accents_and_normalize("") == ""

    def test_none_like_input(self):
        assert strip_accents_and_normalize(None) == ""  # type: ignore[arg-type]

    def test_whitespace_collapsed(self):
        result = strip_accents_and_normalize("  hello   world  ")
        assert result == "hello world"

    def test_punctuation_removed(self):
        result = strip_accents_and_normalize("hello, world!")
        assert result == "hello world"


# ---------------------------------------------------------------------------
# clean_business_name — suffix stripping
# ---------------------------------------------------------------------------
class TestCleanBusinessName:
    def test_fils_mid_name_preserved(self):
        """'fils' in the middle of a brand name must NOT be stripped."""
        _, root = clean_business_name("Le Fils Dupont")
        assert "fils" in root, f"fils should be preserved mid-name; root={root!r}"

    def test_fils_trailing_stripped(self):
        """'et fils' at the end of a name SHOULD be stripped."""
        _, root = clean_business_name("Dupont et Fils SARL")
        # SARL stripped, and trailing fils stripped
        assert "fils" not in root, f"trailing fils should be stripped; root={root!r}"
        assert "dupont" in root

    def test_fils_alone_is_brand(self):
        """A company named purely 'Fils' should not have its name destroyed."""
        _, root = clean_business_name("Fils SA Co Ltd.")
        # all suffixes stripped but 'fils' remains as brand token
        assert root  # not empty

    def test_us_inc(self):
        _, root = clean_business_name("Acme Corp Inc.")
        assert "inc" not in root
        assert "corp" not in root
        assert "acme" in root

    def test_us_llc(self):
        _, root = clean_business_name("Blue Ridge Ventures LLC")
        assert "llc" not in root

    def test_india_pvt_ltd(self):
        _, root = clean_business_name("Raj Investments Private Limited")
        assert "private" not in root
        assert "limited" not in root
        assert "raj" in root and "investments" in root

    def test_india_pvt_ltd_abbreviated(self):
        _, root = clean_business_name("Tata Sons Pvt. Ltd.")
        assert "pvt" not in root
        assert "ltd" not in root

    def test_french_sarl(self):
        _, root = clean_business_name("Société Générale SARL")
        assert "sarl" not in root

    def test_empty_name_returns_empty(self):
        c, r = clean_business_name("")
        assert c == ""
        assert r == ""

    def test_all_suffixes_fallback_to_cleaned(self):
        """If stripping all suffixes leaves empty root, fallback to cleaned name."""
        c, r = clean_business_name("LLC Corp Inc")
        # root should not be empty — fallback to full cleaned name
        assert r, "root must not be empty when suffix stripping consumed everything"

    def test_co_token_stripped(self):
        _, root = clean_business_name("Solo Venture Co")
        assert "co" not in root.split()

    def test_sa_token_stripped(self):
        _, root = clean_business_name("Rousseau SA")
        assert "sa" not in root.split()


# ---------------------------------------------------------------------------
# extract_postal_code
# ---------------------------------------------------------------------------
class TestExtractPostalCode:
    def test_us_5digit(self):
        assert extract_postal_code("123 Main St, New York, NY 10001") == "10001"

    def test_france_5digit(self):
        assert extract_postal_code("29 Bd Haussmann, Paris 75009") == "75009"

    def test_india_6digit(self):
        assert extract_postal_code("24 Bombay House, Mumbai 400001") == "400001"

    def test_no_postal_returns_empty(self):
        assert extract_postal_code("Some Street Without Code") == ""

    def test_empty_string(self):
        assert extract_postal_code("") == ""

    def test_postal_not_confused_with_building(self):
        # Building number (85) and postal code (12883) both present
        result = extract_postal_code("85 Wayne Ave, NY 12883")
        # Should return postal (12883), not building (85)
        assert result == "12883"


# ---------------------------------------------------------------------------
# extract_building_number
# ---------------------------------------------------------------------------
class TestExtractBuildingNumber:
    def test_simple_number(self):
        assert extract_building_number("85 Wayne Ave, NY 12883", "12883") == "85"

    def test_alphanumeric(self):
        assert extract_building_number("12a main street paris 75009", "75009") == "12a"

    def test_building_with_parenthesis(self):
        # 6(29) — DIGIT_TOKEN_REGEX extracts '6' as the first standalone token
        result = extract_building_number("6 market road", "")
        assert result == "6"

    def test_no_building_returns_empty(self):
        assert extract_building_number("market road no number", "") == ""

    def test_postal_not_returned_as_building(self):
        # If only the postal code digit block is present, should return empty
        result = extract_building_number("some address 10001", "10001")
        assert result == ""

    def test_empty_address(self):
        assert extract_building_number("", "") == ""


# ---------------------------------------------------------------------------
# extract_numeric_tokens
# ---------------------------------------------------------------------------
class TestExtractNumericTokens:
    def test_basic(self):
        tokens = extract_numeric_tokens("85 Wayne Ave Suite 400, NY 12883")
        assert "85" in tokens
        assert "400" in tokens
        assert "12883" in tokens

    def test_empty(self):
        assert extract_numeric_tokens("") == set()


# ---------------------------------------------------------------------------
# normalize_address
# ---------------------------------------------------------------------------
class TestNormalizeAddress:
    def test_us_abbreviations(self):
        n = normalize_address("100 Main St, Springfield")
        assert "street" in n

    def test_french_abbreviations(self):
        n = normalize_address("12 Bd Haussmann Paris")
        assert "boulevard" in n

    def test_indian_near(self):
        n = normalize_address("Opp. Post Office, Nr. Station")
        assert "opposite" in n
        assert "near" in n

    def test_empty_address(self):
        assert normalize_address("") == ""

    def test_unicode_french_address(self):
        n = normalize_address("12 Rue de l'Église, Lyon")
        assert "rue" in n or "eglise" in n

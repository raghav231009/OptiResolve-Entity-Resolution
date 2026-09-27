"""
Comprehensive Normalization Audit & Regression Test Suite for OptiResolve.
Covers:
  1. Unicode normalization (NFKD, ligatures œ/æ, invisible chars)
  2. Accent stripping (diacritics across French, Spanish, German)
  3. Punctuation handling (hyphens, apostrophes, slashes)
  4. Whitespace canonicalization (collapse, trim, non-breaking spaces)
  5. Ampersand canonicalization (& -> and)
  6. General corporate legal suffixes
  7. French legal forms (SARL, SAS, SASU, SA, EURL, SCI, SNC, GIE, et Fils)
  8. Indian corporate suffixes (Private Limited, Pvt Ltd, Ltd, LLP)
  9. US corporate suffixes (Inc, LLC, Corp, Co)
  10. Address abbreviations (US, French, Indian)
  11. Postal codes (US 5-digit, FR 5-digit, IN 6-digit)
  12. Building numbers (simple, alphanumeric)
  13. Apartment/Suite/Unit disambiguation from building numbers
  14. Multiple numeric tokens in address
  15. Missing value sentinel handling (NaN, None, N/A, -)
  16. Transliterated names
  17. Non-Latin scripts (Devanagari, multilingual word characters)
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


class TestUnicodeNormalization:
    """Requirement 1: Unicode normalization & ligature handling."""

    def test_french_ligatures_oe_and_ae(self):
        """French œ and æ must normalize to oe and ae."""
        assert strip_accents_and_normalize("Société du Cœur") == "societe du coeur"
        assert strip_accents_and_normalize("L’Œuvre Française") == "l oeuvre francaise"
        assert strip_accents_and_normalize("Ex Æquo SAS") == "ex aequo sas"

    def test_invisible_and_non_breaking_characters(self):
        """Zero-width spaces, soft hyphens, and non-breaking spaces must be cleaned."""
        # \u200b is zero-width space, \u00a0 is non-breaking space, \u202f is narrow non-breaking space
        raw = "Acme\u200b\u00a0Logistics\u202fInc"
        assert strip_accents_and_normalize(raw) == "acme logistics inc"


class TestAccentStripping:
    """Requirement 2: Diacritics and combining accents."""

    def test_french_accents_comprehensive(self):
        raw = "Société d'Électricité & Électronique Générale de l'Hôtel-Dieu"
        cleaned = strip_accents_and_normalize(raw)
        assert "societe" in cleaned
        assert "electricite" in cleaned
        assert "electronique" in cleaned
        assert "generale" in cleaned
        assert "hotel dieu" in cleaned

    def test_cedilla_and_circumflex(self):
        assert strip_accents_and_normalize("Façade & Châteaux") == "facade and chateaux"


class TestPunctuationAndWhitespace:
    """Requirement 3, 4: Punctuation and whitespace collapse."""

    def test_hyphenated_and_apostrophe_names(self):
        assert strip_accents_and_normalize("Saint-Gobain S.A.") == "saint gobain s a"
        assert strip_accents_and_normalize("L'Oréal") == "l oreal"
        assert strip_accents_and_normalize("O'Reilly Media") == "o reilly media"

    def test_whitespace_and_tabs_collapse(self):
        assert strip_accents_and_normalize("   Tata   \t\n  Motors   ") == "tata motors"


class TestAmpersandCanonicalization:
    """Requirement 5: & vs and."""

    def test_ampersand_expansion(self):
        assert strip_accents_and_normalize("Barnes & Noble") == "barnes and noble"
        assert strip_accents_and_normalize("A & B & C") == "a and b and c"


class TestFrenchCorporateForms:
    """Requirement 6, 7: French legal forms and brand protection."""

    def test_french_legal_forms_stripped_from_root(self):
        forms = [
            ("Société Générale SARL", "societe generale"),
            ("Carrefour SAS", "carrefour"),
            ("Alstom SA", "alstom"),
            ("Rousseau EURL", "rousseau"),
            ("Immobilière Saint-Honoré SCI", "immobiliere saint honore"),
            ("Transports Martin SNC", "transports martin"),
            ("Alliance Santé GIE", "alliance sante"),
            ("Danone SASU", "danone"),
        ]
        for raw, expected_root in forms:
            _, root = clean_business_name(raw)
            assert root == expected_root, f"Failed for {raw}: got {root!r}, expected {expected_root!r}"

    def test_french_trailing_fils_stripped(self):
        _, root = clean_business_name("Dupont et Fils SARL")
        assert root == "dupont"

        _, root2 = clean_business_name("Martin & Fils")
        assert root2 == "martin"

    def test_french_brand_token_fils_preserved(self):
        """Brand name 'Le Fils Dupont' must retain 'fils'."""
        _, root = clean_business_name("Le Fils Dupont")
        assert "fils" in root
        assert root == "le fils dupont"


class TestIndianCorporateSuffixes:
    """Requirement 8: Indian corporate legal suffixes."""

    def test_indian_suffixes_stripped(self):
        cases = [
            ("Raj Investments Private Limited", "raj investments"),
            ("Tata Sons Pvt. Ltd.", "tata sons"),
            ("Reliance Retail Limited", "reliance retail"),
            ("Infosys Technologies Ltd", "infosys technologies"),
            ("Adani Logistics LLP", "adani logistics"),
            ("Godrej Consumer Products Pvt Ltd", "godrej consumer products"),
        ]
        for raw, expected_root in cases:
            _, root = clean_business_name(raw)
            assert root == expected_root, f"Failed for {raw}: got {root!r}, expected {expected_root!r}"


class TestUSCorporateSuffixes:
    """Requirement 9: US corporate suffixes."""

    def test_us_suffixes_stripped(self):
        cases = [
            ("Acme Logistics Inc.", "acme logistics"),
            ("Blue Ridge Ventures LLC", "blue ridge ventures"),
            ("General Electric Corporation", "general electric"),
            ("Boeing Corp", "boeing"),
            ("Solo Venture Co", "solo venture"),
            ("Delta Air Lines Inc", "delta air lines"),
        ]
        for raw, expected_root in cases:
            _, root = clean_business_name(raw)
            assert root == expected_root, f"Failed for {raw}: got {root!r}, expected {expected_root!r}"


class TestAddressAbbreviations:
    """Requirement 10: Address abbreviations across US, France, India."""

    def test_us_abbreviations(self):
        raw = "123 Main St, Suite 400, Grand Blvd, Ticonderoga Rd"
        norm = normalize_address(raw)
        assert "street" in norm
        assert "suite" in norm
        assert "boulevard" in norm
        assert "road" in norm

    def test_french_abbreviations(self):
        raw = "29 Bd Haussmann, 12 R. de la Paix, Bât. C, 2ème Étg."
        norm = normalize_address(raw)
        assert "boulevard" in norm
        assert "rue" in norm
        assert "batiment" in norm
        assert "etage" in norm

    def test_indian_abbreviations(self):
        raw = "Opp. Post Office, Nr. Railway Station, M.G. Rd"
        norm = normalize_address(raw)
        assert "opposite" in norm
        assert "near" in norm
        assert "road" in norm


class TestPostalCodesAndBuildingNumbers:
    """Requirement 11, 12, 13, 14: Disambiguation of postal, building, and suite numbers."""

    def test_4digit_building_number_not_confused_with_5digit_zip(self):
        """CRITICAL FIX: 1200 Main St with ZIP 10001 must not swap building and postal."""
        addr = "1200 Main St, New York, NY 10001"
        postal = extract_postal_code(addr)
        building = extract_building_number(addr, postal)
        assert postal == "10001"
        assert building == "1200"

    def test_indian_6digit_pin_with_4digit_plot(self):
        addr = "Plot 1500, M.G. Road, Bangalore 560001"
        postal = extract_postal_code(addr)
        building = extract_building_number(addr, postal)
        assert postal == "560001"
        assert building == "1500"

    def test_french_5digit_code_and_building(self):
        addr = "29 Bd Haussmann, Paris 75009"
        postal = extract_postal_code(addr)
        building = extract_building_number(addr, postal)
        assert postal == "75009"
        assert building == "29"

    def test_alphanumeric_building_number(self):
        addr = "14A Rue de la Paix, 75002 Paris"
        postal = extract_postal_code(addr)
        building = extract_building_number(addr, postal)
        assert postal == "75002"
        assert building == "14a"

    def test_suite_disambiguation_from_building(self):
        """Suite numbers must not masquerade as the street building number."""
        addr1 = "85 Wayne Ave, Suite 400, NY 12883"
        postal1 = extract_postal_code(addr1)
        bldg1 = extract_building_number(addr1, postal1)
        assert postal1 == "12883"
        assert bldg1 == "85"

        addr2 = "Suite 200, 85 Wayne Ave, NY 12883"
        postal2 = extract_postal_code(addr2)
        bldg2 = extract_building_number(addr2, postal2)
        assert postal2 == "12883"
        assert bldg2 == "85"

    def test_multiple_numeric_tokens_extracted(self):
        addr = "12 Main St Suite 400 Apt 5B, NY 10001"
        tokens = extract_numeric_tokens(addr)
        assert "12" in tokens
        assert "400" in tokens
        assert "5b" in tokens
        assert "10001" in tokens


class TestMissingValues:
    """Requirement 15: Missing value sentinel handling."""

    def test_sentinels_return_empty(self):
        sentinels = ["nan", "NaN", "none", "None", "NULL", "n/a", "N/A", "NA", "-", ".", "  "]
        for s in sentinels:
            clean, root = clean_business_name(s)
            assert clean == "", f"Expected empty clean for sentinel {s!r}, got {clean!r}"
            assert root == "", f"Expected empty root for sentinel {s!r}, got {root!r}"
            assert normalize_address(s) == ""
            assert extract_postal_code(s) == ""
            assert extract_building_number(s) == ""

    def test_none_and_empty_types(self):
        assert strip_accents_and_normalize(None) == ""  # type: ignore[arg-type]
        assert strip_accents_and_normalize("") == ""


class TestNonLatinAndTransliteratedNames:
    """Requirement 16, 17: Non-Latin scripts and transliterated variations."""

    def test_devanagari_script_preserved(self):
        """Devanagari script should not be corrupted or stripped."""
        clean, root = clean_business_name("टाटा मोटर्स Pvt Ltd")
        assert "टाटा मोटर्स" in clean
        assert root == "टाटा मोटर्स"

    def test_devanagari_and_latin_mixed(self):
        clean, root = clean_business_name("State Bank of India - स्टेट बैंक")
        assert "state bank of india" in clean
        assert "स्टेट बैंक" in clean

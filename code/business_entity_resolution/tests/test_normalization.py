import pytest
from business_entity_resolution.normalization import (
    strip_accents_and_normalize,
    clean_business_name,
    normalize_address,
    extract_postal_code,
    extract_numeric_tokens,
)


def test_french_unicode_accent_stripping():
    # French accented words
    raw = "Société d'Électricité & Fils SARL"
    clean, root = clean_business_name(raw)
    assert "societe" in clean
    assert "electricite" in clean
    assert "sarl" not in root  # legal suffix stripped
    assert "fils" not in root


def test_indian_legal_suffix_stripping():
    clean, root = clean_business_name("Raj Investments Private Limited")
    assert clean == "raj investments private limited"
    assert root == "raj investments"

    clean2, root2 = clean_business_name("Tata Sons Pvt. Ltd.")
    assert "pvt" not in root2
    assert "ltd" not in root2


def test_us_legal_suffix_stripping():
    clean, root = clean_business_name("Maure Williams Colombier Inc.")
    assert root == "maure williams colombier"


def test_address_normalization():
    addr = "85 Wayne Ave, Suite 400, Ticonderoga, NY 12883"
    norm = normalize_address(addr)
    assert "avenue" in norm
    assert "suite" in norm
    assert extract_postal_code(addr) == "12883"
    assert "85" in extract_numeric_tokens(addr)
    assert "400" in extract_numeric_tokens(addr)

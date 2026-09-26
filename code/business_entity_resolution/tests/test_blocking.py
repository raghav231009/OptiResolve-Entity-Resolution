import pytest
from business_entity_resolution.blocking import MultiIndexBlocker


def test_blocking_retrieval_and_cap():
    blocker = MultiIndexBlocker(max_candidates=5)

    targets = [
        {
            "entity_id": "S2-001",
            "country": "US",
            "clean_name": "alpha logistics inc",
            "root_name": "alpha logistics",
            "clean_address": "100 broadway new york ny 10001",
            "postal_code": "10001",
            "numeric_tokens": {"100"},
        },
        {
            "entity_id": "S3-002",
            "country": "US",
            "clean_name": "alpha global logistics",
            "root_name": "alpha global logistics",
            "clean_address": "100 broadway ny",
            "postal_code": "",
            "numeric_tokens": {"100"},
        },
        {
            "entity_id": "S2-003",
            "country": "France",
            "clean_name": "alpha transport sarl",
            "root_name": "alpha transport",
            "clean_address": "15 rue de paris bordeaux",
            "postal_code": "33000",
            "numeric_tokens": {"15"},
        }
    ]

    blocker.index_targets(targets)

    # Query with S1 entity from US
    s1 = {
        "entity_id": "S1-999",
        "country": "US",
        "clean_name": "alpha logistics llc",
        "root_name": "alpha logistics",
        "clean_address": "100 broadway new york",
        "postal_code": "10001",
        "numeric_tokens": {"100"},
    }

    cands = blocker.retrieve_candidates(s1)
    # Should retrieve US candidates and NEVER France
    assert "S2-001" in cands
    assert "S3-002" in cands
    assert "S2-003" not in cands

import pytest
from business_entity_resolution.metrics import compute_entity_f05, compute_macro_f05


def test_singleton_scoring():
    # True singleton predicted empty -> 1.0
    assert compute_entity_f05(set(), set()) == 1.0

    # True singleton predicted non-empty -> 0.0
    assert compute_entity_f05(set(), {"S2-001"}) == 0.0


def test_non_singleton_scoring():
    # Non-singleton predicted empty -> 0.0
    assert compute_entity_f05({"S2-001"}, set()) == 0.0

    # Perfect match -> 1.0
    assert compute_entity_f05({"S2-001", "S3-002"}, {"S2-001", "S3-002"}) == 1.0

    # Official PDF Example:
    # Model predicts: S2-00047, S2-00193, S3-00812
    # Ground truth: S2-00047, S3-00812
    # Precision = 2/3 (0.6667), Recall = 2/2 = 1.0
    # Expected F_0.5 = (1.25 * 0.666667 * 1.0) / (0.25 * 0.666667 + 1.0) = 0.7142857
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    gt = {"S2-00047", "S3-00812"}
    score = compute_entity_f05(gt, pred)
    assert abs(score - 0.7142857) < 1e-4


def test_macro_f05():
    gt = {
        "S1-1": {"S2-1"},       # perfect match -> 1.0
        "S1-2": set(),          # singleton correct -> 1.0
        "S1-3": set(),          # singleton wrong -> 0.0
        "S1-4": {"S2-2", "S3-2"} # partial match
    }
    preds = {
        "S1-1": {"S2-1"},
        "S1-2": set(),
        "S1-3": {"S2-99"},
        "S1-4": {"S2-2"} # P=1.0, R=0.5 -> F0.5 = 1.25*1*0.5 / (0.25*1 + 0.5) = 0.625 / 0.75 = 0.83333
    }
    macro = compute_macro_f05(gt, preds)
    expected = (1.0 + 1.0 + 0.0 + (0.625 / 0.75)) / 4.0
    assert abs(macro - expected) < 1e-4

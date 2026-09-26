"""
Evaluation Metrics Module.
Implements the exact challenge Macro F_0.5 score with singleton penalties.
"""

from typing import Dict, Iterable, List, Set, Union


def compute_entity_f05(true_set: Set[str], pred_set: Set[str]) -> float:
    """
    Computes F_0.5 for a single Source 1 entity.
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Singleton rule:
      - true empty and pred empty => 1.0
      - true empty and pred non-empty => 0.0
      - true non-empty and pred empty => 0.0
    """
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0

    if len(pred_set) == 0:
        return 0.0

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    denom = (0.25 * precision) + recall
    if denom == 0.0:
        return 0.0

    return (1.25 * precision * recall) / denom


def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
) -> float:
    """
    Macro-averaged F_0.5 across all Source 1 entities in ground_truth.
    """
    if not ground_truth:
        return 0.0

    scores: List[float] = []
    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, set())
        scores.append(compute_entity_f05(true_matches, pred_matches))

    return sum(scores) / len(scores)

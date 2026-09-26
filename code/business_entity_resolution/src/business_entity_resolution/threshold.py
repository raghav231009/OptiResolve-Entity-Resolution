"""
Threshold Optimization Module.
Sweeps classification threshold to maximize challenge Macro F_0.5 on validation split.
"""

from typing import Dict, List, Set, Tuple
import numpy as np

from .metrics import compute_macro_f05


def optimize_threshold(
    val_ground_truth: Dict[str, Set[str]],
    val_scored_pairs: Dict[str, List[Tuple[str, float]]],
    search_start: float = 0.65,
    search_end: float = 0.96,
    step: float = 0.02,
) -> Tuple[float, float, Dict[float, float]]:
    """
    Search over candidate thresholds tau to find the one maximizing Macro F_0.5.
    Returns:
      (best_threshold, best_macro_f05, all_scores_dict)
    """
    best_tau = 0.88
    best_score = -1.0
    history: Dict[float, float] = {}

    tau = search_start
    while tau <= search_end + 1e-6:
        tau_rounded = round(tau, 3)

        # Form predictions at this threshold
        predictions: Dict[str, Set[str]] = {}
        for s1_id in val_ground_truth.keys():
            candidates = val_scored_pairs.get(s1_id, [])
            matched = {cid for cid, prob in candidates if prob >= tau_rounded}
            predictions[s1_id] = matched

        score = compute_macro_f05(val_ground_truth, predictions)
        history[tau_rounded] = score

        if score > best_score:
            best_score = score
            best_tau = tau_rounded

        tau += step

    return best_tau, best_score, history

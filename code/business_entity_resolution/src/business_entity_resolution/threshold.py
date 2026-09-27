"""
Threshold Optimization Module.
Sweeps classification threshold to maximize challenge Macro F_0.5 on validation split.
Operates strictly on validation predictions with two-phase coarse and fine grid search.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .metrics import compute_macro_f05


class ThresholdMetric(float):
    """
    Float subclass representing Macro F_0.5 score at a specific threshold,
    providing backward compatibility as a float while exposing rich audit metrics
    and dictionary-like access.
    """

    def __new__(cls, record: Dict[str, Any]):
        obj = super().__new__(cls, float(record["macro_f05"]))
        obj.record = dict(record)
        obj.threshold = float(record["threshold"])
        obj.macro_f05 = float(record["macro_f05"])
        obj.predicted_links = int(record["predicted_links"])
        obj.empty_predictions = int(record["empty_predictions"])
        obj.singleton_false_positives = int(record["singleton_false_positives"])
        return obj

    def __getitem__(self, key: str) -> Any:
        return self.record[key]

    def __contains__(self, key: str) -> bool:
        return key in self.record

    def __iter__(self):
        return iter(self.record)

    def keys(self):
        return self.record.keys()

    def values(self):
        return self.record.values()

    def items(self):
        return self.record.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self.record.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.record)

    def __repr__(self) -> str:
        return (
            f"ThresholdMetric(tau={self.threshold:.4f}, macro_f05={self.macro_f05:.4f}, "
            f"predicted_links={self.predicted_links}, empty_predictions={self.empty_predictions}, "
            f"singleton_fp={self.singleton_false_positives})"
        )


def evaluate_threshold(
    val_ground_truth: Dict[str, Set[str]],
    val_scored_pairs: Dict[str, List[Tuple[str, float]]],
    threshold: float,
) -> Dict[str, Union[float, int]]:
    """
    Evaluate validation predictions at a specific threshold tau.

    Calculates:
      - threshold: float
      - macro_f05: float (challenge Macro F0.5 across all validation S1 entities)
      - predicted_links: int (total links predicted across all S1 entities)
      - empty_predictions: int (entities where 0 links were predicted)
      - singleton_false_positives: int (true singleton entities where >= 1 link was predicted)
    """
    predictions: Dict[str, Set[str]] = {}
    predicted_links = 0
    empty_predictions = 0
    singleton_false_positives = 0

    for s1_id, gt_matches in val_ground_truth.items():
        candidates = val_scored_pairs.get(s1_id, [])
        matched = {cid for cid, prob in candidates if prob >= threshold}
        predictions[s1_id] = matched

        n_matched = len(matched)
        predicted_links += n_matched
        if n_matched == 0:
            empty_predictions += 1
        if len(gt_matches) == 0 and n_matched > 0:
            singleton_false_positives += 1

    macro_f05 = compute_macro_f05(val_ground_truth, predictions)

    return {
        "threshold": round(float(threshold), 4),
        "macro_f05": float(macro_f05),
        "predicted_links": int(predicted_links),
        "empty_predictions": int(empty_predictions),
        "singleton_false_positives": int(singleton_false_positives),
    }


def optimize_threshold(
    val_ground_truth: Dict[str, Set[str]],
    val_scored_pairs: Dict[str, List[Tuple[str, float]]],
    search_start: float = 0.50,
    search_end: float = 0.99,
    step: float = 0.01,
    fine_step: float = 0.002,
    fine_window: float = 0.03,
) -> Tuple[float, float, Dict[float, ThresholdMetric]]:
    """
    Search over candidate thresholds tau to find the one maximizing Macro F_0.5.

    Uses a two-phase optimization process:
      Phase 1: Coarse grid sweep from search_start (default 0.50) to search_end (default 0.99) with step (0.01).
      Phase 2: Fine grid sweep centered around best coarse optimum within +/- fine_window (default 0.03)
               with fine_step resolution (default 0.002).

    Returns:
      (best_threshold, best_macro_f05, all_scores_dict)
      where all_scores_dict maps tau -> ThresholdMetric.
    """
    history: Dict[float, ThresholdMetric] = {}

    # Phase 1: Coarse Grid
    tau = search_start
    best_coarse_tau = round(search_start, 4)
    best_coarse_score = -1.0

    while tau <= search_end + 1e-6:
        tau_rounded = round(float(tau), 4)
        record = evaluate_threshold(val_ground_truth, val_scored_pairs, tau_rounded)
        metric = ThresholdMetric(record)
        history[tau_rounded] = metric

        if metric.macro_f05 > best_coarse_score:
            best_coarse_score = metric.macro_f05
            best_coarse_tau = tau_rounded
        elif abs(metric.macro_f05 - best_coarse_score) <= 1e-9:
            best_coarse_tau = max(best_coarse_tau, tau_rounded)

        tau += step

    # Phase 2: Fine Grid around optimum
    if fine_step is not None and fine_step > 0 and fine_window > 0:
        fine_start = max(search_start, round(best_coarse_tau - fine_window, 4))
        fine_end = min(search_end, round(best_coarse_tau + fine_window, 4))

        fine_tau = fine_start
        while fine_tau <= fine_end + 1e-6:
            fine_rounded = round(float(fine_tau), 4)
            if fine_rounded not in history:
                record = evaluate_threshold(val_ground_truth, val_scored_pairs, fine_rounded)
                metric = ThresholdMetric(record)
                history[fine_rounded] = metric
            fine_tau += fine_step

    # Global Best Selection
    best_tau = round(search_start, 4)
    best_score = -1.0
    for tau_val, metric in sorted(history.items(), key=lambda x: x[0]):
        if metric.macro_f05 > best_score + 1e-9:
            best_score = metric.macro_f05
            best_tau = tau_val
        elif abs(metric.macro_f05 - best_score) <= 1e-9:
            best_tau = max(best_tau, tau_val)

    return best_tau, best_score, history

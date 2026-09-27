"""
Calibration and Probability Stability Audit Module.
Analyzes probability distributions, reliability diagrams, Expected Calibration Error (ECE),
threshold sensitivity around optimum, and honest validation calibration benchmarks.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np


def compute_distribution_statistics(scores: np.ndarray, bins: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Computes summary statistics and histogram for a 1D array of probability scores.
    """
    if len(scores) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50_median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "histogram": {},
        }

    if bins is None:
        bins = np.linspace(0.0, 1.0, 11)  # 10 bins: [0, 0.1), [0.1, 0.2), ...

    counts, bin_edges = np.histogram(scores, bins=bins)
    hist_dict = {}
    for i in range(len(counts)):
        bin_label = f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}{']' if i == len(counts) - 1 else ')'}"
        hist_dict[bin_label] = int(counts[i])

    return {
        "count": int(len(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "p10": float(np.percentile(scores, 10)),
        "p25": float(np.percentile(scores, 25)),
        "p50_median": float(np.median(scores)),
        "p75": float(np.percentile(scores, 75)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
        "max": float(np.max(scores)),
        "histogram": hist_dict,
    }


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Computes Expected Calibration Error (ECE) and reliability diagram bins.
    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    if len(y_true) == 0:
        return 0.0, []

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    reliability_bins = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)

        bin_count = int(np.sum(mask))
        if bin_count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_count / total_samples) * bin_error
        else:
            bin_acc = 0.0
            bin_conf = (low + high) / 2.0
            bin_error = 0.0

        reliability_bins.append({
            "bin_idx": i,
            "range": [round(float(low), 2), round(float(high), 2)],
            "count": bin_count,
            "confidence": round(float(bin_conf), 4),
            "accuracy": round(float(bin_acc), 4),
            "calibration_error": round(float(bin_error), 4),
        })

    return float(ece), reliability_bins


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Computes the Brier score: mean squared error of predicted probabilities.
    Brier = (1/N) * sum((y_prob_i - y_true_i)^2)
    """
    if len(y_true) == 0:
        return 0.0
    return float(np.mean((y_prob - y_true) ** 2))


def segment_probability_scores(
    val_ground_truth: Dict[str, Set[str]],
    val_scored_pairs: Dict[str, List[Tuple[str, float]]],
) -> Dict[str, np.ndarray]:
    """
    Segments candidate prediction probabilities into 5 key categories:
      1. true_positives: candidates where cid in val_ground_truth[s1_id]
      2. hard_negatives: candidates for non-singleton S1 where cid not in val_ground_truth[s1_id]
      3. singleton_negatives: candidates for true singleton S1 (ground truth is empty)
      4. s2_candidates: candidates from Source 2 (cid starts with "S2-")
      5. s3_candidates: candidates from Source 3 (cid starts with "S3-")
    """
    tp_scores: List[float] = []
    hard_neg_scores: List[float] = []
    sing_neg_scores: List[float] = []
    s2_scores: List[float] = []
    s3_scores: List[float] = []
    all_scores: List[float] = []

    for s1_id, candidates in val_scored_pairs.items():
        true_matches = val_ground_truth.get(s1_id, set())
        is_singleton = len(true_matches) == 0

        for cid, prob in candidates:
            all_scores.append(prob)
            if cid.startswith("S2-"):
                s2_scores.append(prob)
            elif cid.startswith("S3-"):
                s3_scores.append(prob)

            if cid in true_matches:
                tp_scores.append(prob)
            else:
                if is_singleton:
                    sing_neg_scores.append(prob)
                else:
                    hard_neg_scores.append(prob)

    return {
        "true_positives": np.array(tp_scores, dtype=np.float32),
        "hard_negatives": np.array(hard_neg_scores, dtype=np.float32),
        "singleton_negatives": np.array(sing_neg_scores, dtype=np.float32),
        "s2_candidates": np.array(s2_scores, dtype=np.float32),
        "s3_candidates": np.array(s3_scores, dtype=np.float32),
        "all_candidates": np.array(all_scores, dtype=np.float32),
    }


def analyze_threshold_sensitivity(
    val_ground_truth: Dict[str, Set[str]],
    val_scored_pairs: Dict[str, List[Tuple[str, float]]],
    threshold_grid: List[float],
) -> List[Dict[str, Any]]:
    """
    Analyzes sensitivity and rate of change of Macro F0.5, Precision, and Recall
    across a sorted grid of thresholds.
    """
    from .threshold import evaluate_threshold

    results: List[Dict[str, Any]] = []
    sorted_grid = sorted(threshold_grid)

    total_true_links = sum(len(m) for m in val_ground_truth.values())

    for i, tau in enumerate(sorted_grid):
        rec = evaluate_threshold(val_ground_truth, val_scored_pairs, tau)
        tau_val = rec["threshold"]
        macro_f05 = rec["macro_f05"]
        pred_links = rec["predicted_links"]
        empty_preds = rec["empty_predictions"]
        sing_fps = rec["singleton_false_positives"]

        # Link-level TP count
        tp_count = 0
        for s1_id, gt_set in val_ground_truth.items():
            if not gt_set:
                continue
            candidates = val_scored_pairs.get(s1_id, [])
            matched = {cid for cid, prob in candidates if prob >= tau}
            tp_count += len(matched & gt_set)

        precision = (tp_count / pred_links) if pred_links > 0 else 0.0
        recall = (tp_count / total_true_links) if total_true_links > 0 else 0.0

        # Sensitivity: derivative d(Macro F0.5) / d(tau) using finite difference
        if i > 0:
            prev = results[i - 1]
            d_tau = tau_val - prev["threshold"]
            d_f05 = macro_f05 - prev["macro_f05"]
            sensitivity = (d_f05 / d_tau) if abs(d_tau) > 1e-9 else 0.0
        else:
            sensitivity = 0.0

        # Stability flag: derivative magnitude <= 0.15 indicates a smooth, stable plateau
        stable = abs(sensitivity) <= 0.15 if i > 0 else True

        results.append({
            "threshold": round(float(tau_val), 4),
            "macro_f05": round(float(macro_f05), 5),
            "precision": round(float(precision), 5),
            "recall": round(float(recall), 5),
            "predicted_links": int(pred_links),
            "true_positives": int(tp_count),
            "empty_predictions": int(empty_preds),
            "singleton_false_positives": int(sing_fps),
            "sensitivity_d_score": round(float(sensitivity), 4),
            "is_stable": bool(stable),
        })

    return results


def evaluate_calibration_benefit(
    val_ground_truth: Dict[str, Set[str]],
    val_scored_pairs: Dict[str, List[Tuple[str, float]]],
    random_seed: int = 42,
    test_methods: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluates whether probability calibration (Sigmoid/Platt scaling or Isotonic regression)
    improves honest validation performance without data leakage.

    Uses a strict 2-fold cross-split of validation S1 entities:
      Fold 1: Calibrate on Split A, optimize threshold on Split B
      Fold 2: Calibrate on Split B, optimize threshold on Split A
    Averages honest evaluation metrics across both folds.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from .threshold import optimize_threshold

    if test_methods is None:
        test_methods = ["sigmoid", "isotonic"]

    all_s1_ids = list(val_ground_truth.keys())
    np.random.seed(random_seed)
    shuffled_ids = np.random.permutation(all_s1_ids).tolist()

    half = len(shuffled_ids) // 2
    split_a = set(shuffled_ids[:half])
    split_b = set(shuffled_ids[half:])

    def extract_flat_pairs(s1_subset: Set[str]):
        probs, labels = [], []
        for s1_id in s1_subset:
            true_matches = val_ground_truth.get(s1_id, set())
            for cid, prob in val_scored_pairs.get(s1_id, []):
                probs.append(prob)
                labels.append(1 if cid in true_matches else 0)
        return np.array(probs, dtype=np.float32), np.array(labels, dtype=np.int32)

    def calibrate_scores(calibrator, method: str, raw_pairs: Dict[str, List[Tuple[str, float]]], s1_subset: Set[str]):
        calibrated = {}
        for s1_id in s1_subset:
            cands = raw_pairs.get(s1_id, [])
            if not cands:
                calibrated[s1_id] = []
                continue
            ids = [c[0] for c in cands]
            raw_p = np.array([c[1] for c in cands], dtype=np.float32)
            if method == "sigmoid":
                # LogisticRegression on raw probabilities
                cal_p = calibrator.predict_proba(raw_p.reshape(-1, 1))[:, 1]
            elif method == "isotonic":
                cal_p = calibrator.transform(raw_p)
                cal_p = np.clip(cal_p, 0.0, 1.0)
            else:
                cal_p = raw_p
            calibrated[s1_id] = list(zip(ids, cal_p.tolist()))
        return calibrated

    # Evaluate Baseline (Uncalibrated) on Fold 1 (B) and Fold 2 (A)
    gt_a = {s: val_ground_truth[s] for s in split_a}
    gt_b = {s: val_ground_truth[s] for s in split_b}
    scored_a = {s: val_scored_pairs.get(s, []) for s in split_a}
    scored_b = {s: val_scored_pairs.get(s, []) for s in split_b}

    tau_raw_b, score_raw_b, _ = optimize_threshold(gt_b, scored_b, search_start=0.50, search_end=0.99, step=0.01)
    tau_raw_a, score_raw_a, _ = optimize_threshold(gt_a, scored_a, search_start=0.50, search_end=0.99, step=0.01)
    mean_raw_score = (score_raw_b + score_raw_a) / 2.0

    p_val_all, y_val_all = extract_flat_pairs(set(all_s1_ids))
    raw_ece, _ = compute_ece(y_val_all, p_val_all)
    raw_brier = compute_brier_score(y_val_all, p_val_all)

    results = {
        "uncalibrated": {
            "mean_macro_f05": round(float(mean_raw_score), 5),
            "fold_b_optimal_threshold": round(float(tau_raw_b), 4),
            "fold_b_macro_f05": round(float(score_raw_b), 5),
            "fold_a_optimal_threshold": round(float(tau_raw_a), 4),
            "fold_a_macro_f05": round(float(score_raw_a), 5),
            "ece": round(float(raw_ece), 5),
            "brier_score": round(float(raw_brier), 5),
        },
        "calibrated_methods": {},
    }

    best_method = "uncalibrated"
    best_score = mean_raw_score

    for method in test_methods:
        # Fold 1: Fit on Split A, evaluate on Split B
        p_train_a, y_train_a = extract_flat_pairs(split_a)
        if method == "sigmoid":
            calibrator_1 = LogisticRegression(C=1.0, max_iter=1000)
            calibrator_1.fit(p_train_a.reshape(-1, 1), y_train_a)
        else:
            calibrator_1 = IsotonicRegression(out_of_bounds="clip")
            calibrator_1.fit(p_train_a, y_train_a)

        cal_scored_b = calibrate_scores(calibrator_1, method, val_scored_pairs, split_b)
        tau_cal_b, score_cal_b, _ = optimize_threshold(gt_b, cal_scored_b, search_start=0.50, search_end=0.99, step=0.01)

        # Fold 2: Fit on Split B, evaluate on Split A
        p_train_b, y_train_b = extract_flat_pairs(split_b)
        if method == "sigmoid":
            calibrator_2 = LogisticRegression(C=1.0, max_iter=1000)
            calibrator_2.fit(p_train_b.reshape(-1, 1), y_train_b)
        else:
            calibrator_2 = IsotonicRegression(out_of_bounds="clip")
            calibrator_2.fit(p_train_b, y_train_b)

        cal_scored_a = calibrate_scores(calibrator_2, method, val_scored_pairs, split_a)
        tau_cal_a, score_cal_a, _ = optimize_threshold(gt_a, cal_scored_a, search_start=0.50, search_end=0.99, step=0.01)

        mean_cal_score = (score_cal_b + score_cal_a) / 2.0

        # Calibrated ECE on holdouts
        p_cal_eval = np.concatenate([
            np.array([c[1] for s in split_b for c in cal_scored_b.get(s, [])], dtype=np.float32),
            np.array([c[1] for s in split_a for c in cal_scored_a.get(s, [])], dtype=np.float32),
        ])
        y_cal_eval = np.concatenate([
            np.array([1 if c[0] in val_ground_truth.get(s, set()) else 0 for s in split_b for c in cal_scored_b.get(s, [])], dtype=np.int32),
            np.array([1 if c[0] in val_ground_truth.get(s, set()) else 0 for s in split_a for c in cal_scored_a.get(s, [])], dtype=np.int32),
        ])
        cal_ece, _ = compute_ece(y_cal_eval, p_cal_eval)
        cal_brier = compute_brier_score(y_cal_eval, p_cal_eval)

        diff = mean_cal_score - mean_raw_score
        results["calibrated_methods"][method] = {
            "mean_macro_f05": round(float(mean_cal_score), 5),
            "score_difference_vs_raw": round(float(diff), 5),
            "fold_b_optimal_threshold": round(float(tau_cal_b), 4),
            "fold_b_macro_f05": round(float(score_cal_b), 5),
            "fold_a_optimal_threshold": round(float(tau_cal_a), 4),
            "fold_a_macro_f05": round(float(score_cal_a), 5),
            "ece": round(float(cal_ece), 5),
            "brier_score": round(float(cal_brier), 5),
            "improves_validation_performance": bool(diff > 0.0005),
        }

        if mean_cal_score > best_score + 0.0005:
            best_score = mean_cal_score
            best_method = method

    results["recommendation"] = {
        "best_method": best_method,
        "recommend_calibration": bool(best_method != "uncalibrated"),
        "rationale": (
            f"Retain uncalibrated LightGBM probabilities: calibration does not improve honest "
            f"validation Macro F0.5 (raw={mean_raw_score:.4f} vs "
            f"sigmoid={results['calibrated_methods'].get('sigmoid', {}).get('mean_macro_f05', 0):.4f}, "
            f"isotonic={results['calibrated_methods'].get('isotonic', {}).get('mean_macro_f05', 0):.4f}). "
            f"Tree-based ranking already establishes optimal entity candidate ordering."
            if best_method == "uncalibrated"
            else f"Calibration with {best_method} provides an honest validation gain (+{best_score - mean_raw_score:.4f})."
        ),
    }

    return results

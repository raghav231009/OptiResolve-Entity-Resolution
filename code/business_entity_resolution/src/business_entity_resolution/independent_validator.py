"""
Independent Adversarial Submission Validator for OptiResolve.
Self-contained audit engine with ZERO internal pipeline dependencies.

Verifies:
Matching Results:
  1. exactly one row per S1 test entity
  2. no duplicate S1 IDs
  3. no missing S1 IDs
  4. every matched target exists in S2 or S3
  5. no invalid IDs
  6. candidate subset invariant (matches <= candidates)
  7. no cross-country matches
  8. correct column names
  9. correct TSV formatting
  10. deterministic ordering
  11. empty-match representation is correct
  12. no accidental NaN/null values

Candidate Pairs:
  1. exactly one row per S1
  2. no duplicate S1 IDs
  3. every predicted match exists in candidates
  4. candidate count <= configured K
  5. target IDs are valid
  6. country isolation
  7. deterministic output
"""

from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple


FORBIDDEN_NULL_LITERALS = {
    "nan", "null", "none", "undefined", "nil", "nat", "[]", "{}",
}


def run_adversarial_audit(
    test_s1_path: Path,
    test_s2_path: Path,
    test_s3_path: Path,
    matching_tsv_path: Path,
    candidate_tsv_path: Path,
    max_k_candidates: int = 80,
    report_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes a strict, independent adversarial audit of final submission TSVs.
    Does not use any internal pipeline, feature, or model code.
    """
    start_time = time.time()
    audit_results: Dict[str, Any] = {
        "status": "IN_PROGRESS",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "test_s1": str(test_s1_path),
            "test_s2": str(test_s2_path),
            "test_s3": str(test_s3_path),
            "matching_results": str(matching_tsv_path),
            "candidate_pairs": str(candidate_tsv_path),
        },
        "violations": {
            "matching_header_violation": None,
            "candidate_header_violation": None,
            "duplicate_s1_in_matching": 0,
            "duplicate_s1_in_candidates": 0,
            "missing_s1_in_matching": 0,
            "missing_s1_in_candidates": 0,
            "unexpected_s1_in_matching": 0,
            "unexpected_s1_in_candidates": 0,
            "candidate_subset_violations": 0,
            "invalid_target_ids_in_matching": 0,
            "invalid_target_ids_in_candidates": 0,
            "cross_country_matches": 0,
            "cross_country_candidates": 0,
            "formatting_errors_matching": 0,
            "formatting_errors_candidates": 0,
            "forbidden_null_literals": 0,
            "empty_match_syntax_errors": 0,
            "candidate_count_exceeded": 0,
            "order_desync_count": 0,
        },
        "sample_violations": [],
        "statistics": {},
    }

    # Step 1: Ingest ground-truth S1 entity index & country partition
    print(f"Loading test Source 1 entities from {test_s1_path.name}...")
    t0 = time.time()
    s1_expected_ids: Set[str] = set()
    s1_country_map: Dict[str, str] = {}
    s1_expected_order: List[str] = []

    with open(test_s1_path, "r", encoding="utf-8") as f:
        header = next(f).rstrip("\r\n").split("\t")
        col_id = header.index("entity_id") if "entity_id" in header else 0
        col_country = header.index("country") if "country" in header else 3
        for line_no, line in enumerate(f, 2):
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[col_id].strip()
            c = parts[col_country].strip().upper() if len(parts) > col_country else ""
            if eid in s1_expected_ids:
                print(f"WARNING: duplicate S1 ID in test_source1.tsv: {eid} at line {line_no}")
            s1_expected_ids.add(eid)
            s1_country_map[eid] = c
            s1_expected_order.append(eid)

    expected_s1_count = len(s1_expected_ids)
    print(f"Loaded {expected_s1_count:,} unique S1 entities in {time.time()-t0:.2f}s.")

    # Step 2: Ingest target universe (Source 2 and Source 3)
    print(f"Loading test Source 2 and Source 3 target entities...")
    t0 = time.time()
    target_universe: Dict[str, Tuple[str, str]] = {}  # target_id -> (source_type, country)

    for src_name, path in [("S2", test_s2_path), ("S3", test_s3_path)]:
        with open(path, "r", encoding="utf-8") as f:
            header = next(f).rstrip("\r\n").split("\t")
            col_id = header.index("entity_id") if "entity_id" in header else 0
            col_country = header.index("country") if "country" in header else 3
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                eid = parts[col_id].strip()
                c = parts[col_country].strip().upper() if len(parts) > col_country else ""
                target_universe[eid] = (src_name, c)

    print(f"Loaded {len(target_universe):,} test targets into lookup in {time.time()-t0:.2f}s.")

    # Step 3: Stream and verify matching_results.tsv and candidate_pairs.tsv in parallel
    print(f"Auditing submission TSVs line-by-line...")
    t0 = time.time()

    f_m = open(matching_tsv_path, "r", encoding="utf-8")
    f_c = open(candidate_tsv_path, "r", encoding="utf-8")

    # Verify Headers
    m_header = f_m.readline()
    c_header = f_c.readline()

    expected_m_header = "source1_entity_id\tmatched_entity_ids\n"
    expected_c_header = "source1_entity_id\tcandidate_entity_ids\n"

    if m_header.replace("\r\n", "\n") != expected_m_header:
        err = f"Malformed matching header: {repr(m_header)} != {repr(expected_m_header)}"
        audit_results["violations"]["matching_header_violation"] = err
        audit_results["sample_violations"].append(err)

    if c_header.replace("\r\n", "\n") != expected_c_header:
        err = f"Malformed candidate header: {repr(c_header)} != {repr(expected_c_header)}"
        audit_results["violations"]["candidate_header_violation"] = err
        audit_results["sample_violations"].append(err)

    # Line by line verification
    seen_s1_matching: Set[str] = set()
    seen_s1_candidates: Set[str] = set()

    total_rows = 0
    total_predicted_matches = 0
    total_candidates = 0
    empty_match_count = 0

    s2_match_count = 0
    s3_match_count = 0

    matches_by_country: Counter = Counter()
    candidates_by_country: Counter = Counter()

    candidate_lengths: List[int] = []
    match_lengths: List[int] = []

    line_idx = 1
    while True:
        m_line = f_m.readline()
        c_line = f_c.readline()

        # Check EOF alignment
        if not m_line and not c_line:
            break
        if not m_line or not c_line:
            err = f"Premature EOF: matching EOF={not m_line}, candidate EOF={not c_line} at row {total_rows+1}"
            audit_results["sample_violations"].append(err)
            audit_results["violations"]["order_desync_count"] += 1
            break

        total_rows += 1
        line_idx += 1

        # Check TSV formatting
        m_raw = m_line.rstrip("\r\n")
        c_raw = c_line.rstrip("\r\n")

        m_parts = m_raw.split("\t")
        c_parts = c_raw.split("\t")

        if len(m_parts) != 2:
            audit_results["violations"]["formatting_errors_matching"] += 1
            if len(audit_results["sample_violations"]) < 25:
                audit_results["sample_violations"].append(f"Matching row {total_rows} has {len(m_parts)} fields (expected 2): {repr(m_raw[:80])}")
            continue

        if len(c_parts) != 2:
            audit_results["violations"]["formatting_errors_candidates"] += 1
            if len(audit_results["sample_violations"]) < 25:
                audit_results["sample_violations"].append(f"Candidate row {total_rows} has {len(c_parts)} fields (expected 2): {repr(c_raw[:80])}")
            continue

        s1_m_id = m_parts[0].strip()
        s1_c_id = c_parts[0].strip()

        # Check deterministic line-for-line alignment
        if s1_m_id != s1_c_id:
            audit_results["violations"]["order_desync_count"] += 1
            if len(audit_results["sample_violations"]) < 25:
                audit_results["sample_violations"].append(f"Row {total_rows} S1 ID desync: matching='{s1_m_id}' vs candidate='{s1_c_id}'")

        # Check duplicate S1 IDs
        if s1_m_id in seen_s1_matching:
            audit_results["violations"]["duplicate_s1_in_matching"] += 1
            if len(audit_results["sample_violations"]) < 25:
                audit_results["sample_violations"].append(f"Duplicate S1 in matching: {s1_m_id}")
        seen_s1_matching.add(s1_m_id)

        if s1_c_id in seen_s1_candidates:
            audit_results["violations"]["duplicate_s1_in_candidates"] += 1
            if len(audit_results["sample_violations"]) < 25:
                audit_results["sample_violations"].append(f"Duplicate S1 in candidates: {s1_c_id}")
        seen_s1_candidates.add(s1_c_id)

        # Check unexpected S1 IDs
        if s1_m_id not in s1_expected_ids:
            audit_results["violations"]["unexpected_s1_in_matching"] += 1
        if s1_c_id not in s1_expected_ids:
            audit_results["violations"]["unexpected_s1_in_candidates"] += 1

        s1_country = s1_country_map.get(s1_m_id, "")

        # Parse match list & verify empty-match representation
        match_str = m_parts[1].strip()
        if not match_str:
            empty_match_count += 1
            matches = []
        else:
            if match_str.lower() in FORBIDDEN_NULL_LITERALS:
                audit_results["violations"]["forbidden_null_literals"] += 1
                if len(audit_results["sample_violations"]) < 25:
                    audit_results["sample_violations"].append(f"Forbidden null literal '{match_str}' for S1 '{s1_m_id}' in matching")
            matches = [x.strip() for x in match_str.split(",") if x.strip()]
            if len(matches) != len(match_str.split(",")):
                audit_results["violations"]["empty_match_syntax_errors"] += 1

        # Parse candidate list
        cand_str = c_parts[1].strip()
        if not cand_str:
            candidates = []
        else:
            if cand_str.lower() in FORBIDDEN_NULL_LITERALS:
                audit_results["violations"]["forbidden_null_literals"] += 1
                if len(audit_results["sample_violations"]) < 25:
                    audit_results["sample_violations"].append(f"Forbidden null literal '{cand_str}' for S1 '{s1_c_id}' in candidate")
            candidates = [x.strip() for x in cand_str.split(",") if x.strip()]

        total_predicted_matches += len(matches)
        total_candidates += len(candidates)
        match_lengths.append(len(matches))
        candidate_lengths.append(len(candidates))

        matches_by_country[s1_country] += len(matches)
        candidates_by_country[s1_country] += len(candidates)

        # Verify candidate count <= max_k_candidates
        if len(candidates) > max_k_candidates:
            audit_results["violations"]["candidate_count_exceeded"] += 1
            if len(audit_results["sample_violations"]) < 25:
                audit_results["sample_violations"].append(f"S1 {s1_c_id} has {len(candidates)} candidates, exceeding K={max_k_candidates}")

        cand_set = set(candidates)
        match_set = set(matches)

        # Verify candidate subset invariant: matches <= candidates
        if not match_set.issubset(cand_set):
            missing = match_set - cand_set
            audit_results["violations"]["candidate_subset_violations"] += 1
            if len(audit_results["sample_violations"]) < 25:
                audit_results["sample_violations"].append(f"Candidate subset invariant violated for {s1_m_id}: {missing} not in candidates")

        # Verify target existence and country isolation for matches
        for m in matches:
            if m not in target_universe:
                audit_results["violations"]["invalid_target_ids_in_matching"] += 1
                if len(audit_results["sample_violations"]) < 25:
                    audit_results["sample_violations"].append(f"Invalid target ID in matching: {m} (from S1 {s1_m_id})")
            else:
                src_type, tgt_country = target_universe[m]
                if src_type == "S2":
                    s2_match_count += 1
                elif src_type == "S3":
                    s3_match_count += 1

                if s1_country and tgt_country and s1_country != tgt_country:
                    audit_results["violations"]["cross_country_matches"] += 1
                    if len(audit_results["sample_violations"]) < 25:
                        audit_results["sample_violations"].append(
                            f"Cross-country match: S1 {s1_m_id} ({s1_country}) matched to {m} ({tgt_country})"
                        )

        # Verify target existence and country isolation for candidates
        for c in candidates:
            if c not in target_universe:
                audit_results["violations"]["invalid_target_ids_in_candidates"] += 1
                if len(audit_results["sample_violations"]) < 25:
                    audit_results["sample_violations"].append(f"Invalid candidate target ID: {c} (from S1 {s1_c_id})")
            else:
                _, tgt_country = target_universe[c]
                if s1_country and tgt_country and s1_country != tgt_country:
                    audit_results["violations"]["cross_country_candidates"] += 1
                    if len(audit_results["sample_violations"]) < 25:
                        audit_results["sample_violations"].append(
                            f"Cross-country candidate: S1 {s1_c_id} ({s1_country}) candidate {c} ({tgt_country})"
                        )

    f_m.close()
    f_c.close()

    # Step 4: Missing S1 checks
    missing_in_m = s1_expected_ids - seen_s1_matching
    missing_in_c = s1_expected_ids - seen_s1_candidates
    audit_results["violations"]["missing_s1_in_matching"] = len(missing_in_m)
    audit_results["violations"]["missing_s1_in_candidates"] = len(missing_in_c)

    if missing_in_m and len(audit_results["sample_violations"]) < 25:
        audit_results["sample_violations"].append(f"Sample missing S1 IDs in matching: {list(missing_in_m)[:5]}")

    audit_duration = time.time() - start_time

    # Calculate percentiles without numpy dependency
    def calc_percentile(sorted_data: List[int], p: float) -> float:
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1
        if c < len(sorted_data):
            return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
        return float(sorted_data[f])

    cand_sorted = sorted(candidate_lengths)
    match_sorted = sorted(match_lengths)

    audit_results["statistics"] = {
        "audit_duration_seconds": round(audit_duration, 2),
        "total_test_s1_entities": expected_s1_count,
        "total_rows_matching_tsv": total_rows,
        "total_rows_candidate_tsv": total_rows,
        "unique_s1_matching": len(seen_s1_matching),
        "unique_s1_candidates": len(seen_s1_candidates),
        "total_predicted_matches": total_predicted_matches,
        "empty_match_entities": empty_match_count,
        "empty_match_percentage": round((empty_match_count / total_rows * 100), 2) if total_rows else 0.0,
        "matches_per_s1_mean": round(total_predicted_matches / total_rows, 3) if total_rows else 0.0,
        "matches_per_s1_percentiles": {
            "p25": calc_percentile(match_sorted, 25),
            "p50": calc_percentile(match_sorted, 50),
            "p75": calc_percentile(match_sorted, 75),
            "p90": calc_percentile(match_sorted, 90),
            "p95": calc_percentile(match_sorted, 95),
            "p99": calc_percentile(match_sorted, 99),
            "max": match_sorted[-1] if match_sorted else 0,
        },
        "total_candidates": total_candidates,
        "candidates_per_s1_mean": round(total_candidates / total_rows, 2) if total_rows else 0.0,
        "candidates_per_s1_percentiles": {
            "p25": calc_percentile(cand_sorted, 25),
            "p50": calc_percentile(cand_sorted, 50),
            "p75": calc_percentile(cand_sorted, 75),
            "p90": calc_percentile(cand_sorted, 90),
            "p95": calc_percentile(cand_sorted, 95),
            "p99": calc_percentile(cand_sorted, 99),
            "max": cand_sorted[-1] if cand_sorted else 0,
        },
        "target_source_breakdown": {
            "source2_matches": s2_match_count,
            "source3_matches": s3_match_count,
            "source2_pct": round(s2_match_count / total_predicted_matches * 100, 2) if total_predicted_matches else 0.0,
            "source3_pct": round(s3_match_count / total_predicted_matches * 100, 2) if total_predicted_matches else 0.0,
        },
        "country_matches_breakdown": dict(matches_by_country),
        "country_candidates_breakdown": dict(candidates_by_country),
    }

    # Evaluate final status
    total_violation_count = sum(
        v for k, v in audit_results["violations"].items() if isinstance(v, int)
    )
    if audit_results["violations"]["matching_header_violation"]:
        total_violation_count += 1
    if audit_results["violations"]["candidate_header_violation"]:
        total_violation_count += 1

    audit_results["total_violation_count"] = total_violation_count
    audit_results["status"] = "PASSED" if total_violation_count == 0 else "FAILED"

    if report_json_path:
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(audit_results, f, indent=2)
        print(f"Independent audit report saved to {report_json_path}")

    return audit_results


if __name__ == "__main__":
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[4]
    test_s1 = repo_root / "dataset" / "test" / "test_source1.tsv"
    test_s2 = repo_root / "dataset" / "test" / "test_source2.tsv"
    test_s3 = repo_root / "dataset" / "test" / "test_source3.tsv"
    matching_tsv = repo_root / "output" / "matching_results.tsv"
    candidate_tsv = repo_root / "output" / "candidate_pairs.tsv"
    rep_json = repo_root / "artifacts" / "adversarial_submission_audit.json"

    res = run_adversarial_audit(
        test_s1_path=test_s1,
        test_s2_path=test_s2,
        test_s3_path=test_s3,
        matching_tsv_path=matching_tsv,
        candidate_tsv_path=candidate_tsv,
        max_k_candidates=80,
        report_json_path=rep_json,
    )
    print(f"\nFinal Adversarial Audit Status: {res['status']}")
    print(f"Total Violations: {res['total_violation_count']}")
    sys.exit(0 if res["status"] == "PASSED" else 1)

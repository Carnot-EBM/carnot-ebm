"""Exp 3117 sidecar score correlation diagnostics over exact fixtures.

Spec refs: REQ-VERIFY-3117, SCENARIO-VERIFY-3117.

This module keeps the EBT/ARM sidecar boundary diagnostic-only. It converts
checked-in exact fixture rows into replay-compatible sidecar records, computes
the existing cached replay score, and also reports a label-blind feature score.
The split matters because replay rows contain exact-label references for audit
purposes; those label-aware fields must never be described as a live pre-label
model score.
"""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

from carnot.inference.ebt_arm_sidecar_adapter import (
    SCHEMA_REL_PATH as SIDECAR_SCHEMA_REL_PATH,
    SidecarReplayScorer,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3117_ebt_arm_sidecar_score_correlation_boundary_v3"
SCHEMA = "carnot.ebt_arm_sidecar_score_correlation_boundary.v3"
OUTPUT_REL_PATH = Path("results/experiment_3117_ebt_arm_sidecar_score_correlation_boundary_v3.json")
EXP3097_REL_PATH = Path("results/experiment_3097_exact_fixture_eval_protocol_audit_v1.json")
EXP3104_REL_PATH = Path("results/experiment_3104_ebt_arm_sidecar_pipeline_boundary_v2.json")
MANIFEST_REL_PATH = Path("results/exact_fixture_eval_protocol_3097/stratified_eval_manifest.jsonl")
SIDECAR_SCORER_REL_PATH = Path("python/carnot/inference/ebt_arm_sidecar_adapter.py")
MIN_EXACT_FIXTURE_COUNT = 48
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
BLOCKED_PREFIXES = (
    "blocked_missing_inputs",
    "blocked_insufficient_exact_fixtures",
    "blocked_missing_outcome_classes",
    "blocked_incomplete_diagnostics",
)
REQUIRED_FIELDS = (
    "sidecar_score_correlation_boundary_v3_ready",
    "exact_fixture_count",
    "score_correlation_summary",
    "calibration_summary",
    "failure_cases",
    "no_live_model_integration_claim",
    "no_weight_update_claim",
    "no_speedup_claim",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3117_ebt_arm_sidecar_score_correlation_boundary_v3.py -q --no-cov",
    ".venv/bin/coverage run --source=python/carnot/eval/ebt_arm_sidecar_score_correlation_boundary_v3.py -m pytest -o addopts='' tests/python/test_experiment_3117_ebt_arm_sidecar_score_correlation_boundary_v3.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/ebt_arm_sidecar_score_correlation_boundary_v3.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_SPECS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("research_references", Path("research-references.md"), False),
    ("exp3097_exact_protocol", EXP3097_REL_PATH, True),
    ("exp3097_stratified_manifest", MANIFEST_REL_PATH, True),
    ("exp3104_sidecar_boundary", EXP3104_REL_PATH, False),
    ("sidecar_json_schema", SIDECAR_SCHEMA_REL_PATH, False),
    ("sidecar_replay_scorer", SIDECAR_SCORER_REL_PATH, False),
)

_SIMPLE_CMP_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(>=|<=|==)\s*(-?\d+)\s*$")
_SUM_EQ_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\+\s*([A-Za-z_]\w*)\s*==\s*(-?\d+)\s*$")


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object, returning empty evidence when the file is absent."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_jsonl_rows_from_text(text: str) -> list[JsonDict]:
    """Read JSONL object rows while skipping malformed and non-object lines."""

    rows: list[JsonDict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def read_jsonl_rows(path: Path) -> list[JsonDict]:
    """Read JSONL rows from disk, failing closed to an empty list."""

    try:
        return read_jsonl_rows_from_text(path.read_text(encoding="utf-8"))
    except OSError:
        return []


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    min_exact_count: int = MIN_EXACT_FIXTURE_COUNT,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3117: build the sidecar correlation artifact offline."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    exp3097 = read_json_object(root_path / EXP3097_REL_PATH)
    manifest_rel_path = Path(str(exp3097.get("stratified_eval_manifest_path") or MANIFEST_REL_PATH))
    manifest_rows = read_jsonl_rows(root_path / manifest_rel_path)
    scored_rows = score_fixture_rows(manifest_rows, root=root_path)
    outcome_counts = count_outcomes(scored_rows)
    sources = source_artifacts(root_path, manifest_rel_path)
    required_sources_present = all(
        source["exists"] is True for source in sources if source["required"] is True
    )
    correlation = score_correlation_summary(scored_rows)
    calibration = calibration_summary(scored_rows)
    failures = failure_cases(scored_rows, "label_blind_feature_energy")
    readiness_checks = {
        "exp3097_protocol_ready": exp3097.get("eval_protocol_ready") is True,
        "required_sources_present": required_sources_present,
        "minimum_exact_fixture_count_met": len(scored_rows) >= int(min_exact_count),
        "accepted_cases_present": outcome_counts.get("accepted", 0) > 0,
        "rejected_cases_present": outcome_counts.get("rejected", 0) > 0,
        "repairable_cases_present": outcome_counts.get("repairable", 0) > 0,
        "finite_correlation_metrics": finite_correlation_summary(correlation),
        "calibration_accounts_for_all_rows": calibration.get("accounts_for_all_rows") is True,
        "claim_boundary_recorded": True,
    }
    ready = all(readiness_checks.values())
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3117", "SCENARIO-VERIFY-3117"],
        "sidecar_score_correlation_boundary_v3_ready": ready,
        "exact_fixture_count": len(scored_rows),
        "minimum_exact_fixture_count": int(min_exact_count),
        "outcome_counts": outcome_counts,
        "score_correlation_summary": correlation,
        "calibration_summary": calibration,
        "failure_cases": failures,
        "diagnostic_rows": scored_rows,
        "readiness_checks": readiness_checks,
        "blocked_reasons": [name for name, ok in readiness_checks.items() if ok is not True],
        "score_claim_boundary": (
            "label_blind_feature_energy is computed from candidate payload features before exact "
            "label comparison; replay_total_energy_label_aware may include exact-label-reference "
            "mismatch terms and is audit evidence only, not a live pre-label score."
        ),
        "no_live_model_integration_claim": True,
        "no_weight_update_claim": True,
        "no_speedup_claim": True,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": sources,
        "source_checksums": {
            source["path"]: source["sha256"] for source in sources if source["sha256"] is not None
        },
        "inference_substrate": inference_substrate(),
        "duration_s": duration(started, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    min_exact_count: int = MIN_EXACT_FIXTURE_COUNT,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3117 terminal JSON artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(
        root_path,
        min_exact_count=min_exact_count,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    write_json(out_path, artifact)
    return out_path


def score_fixture_rows(
    rows: Sequence[Mapping[str, Any]], *, root: Path | str = REPO_ROOT
) -> list[JsonDict]:
    """Convert exact fixture rows to cached sidecar records and score them."""

    root_path = Path(root)
    scorer = SidecarReplayScorer()
    scored: list[JsonDict] = []
    for row in rows:
        manifest = dict(row)
        record, features = synthetic_sidecar_record(manifest, root=root_path)
        score = scorer.score(record)
        label_blind_energy = round(
            sum(
                float(term["weighted_value"])
                for term in score.energy_terms
                if term["name"] != "exact_label_mismatch_energy"
            ),
            10,
        )
        exact_action = expected_action(manifest)
        sidecar_action = str(features["sidecar_action"])
        exact_outcome = exact_outcome_class(manifest)
        scored.append(
            {
                "fixture_id": str(manifest.get("source_fixture_id") or ""),
                "task_family": str(manifest.get("task_family") or ""),
                "perturbation_type": str(manifest.get("perturbation_type") or ""),
                "expected_answer": str(manifest.get("expected_answer") or ""),
                "expected_action": exact_action,
                "exact_outcome": exact_outcome,
                "reject_or_repair_label": 0 if exact_outcome == "accepted" else 1,
                "outcome_ordinal_label": {"accepted": 0, "rejected": 1, "repairable": 2}[
                    exact_outcome
                ],
                "sidecar_action": sidecar_action,
                "sidecar_action_matches_exact": sidecar_action == exact_action,
                "label_blind_feature_energy": label_blind_energy,
                "replay_total_energy": score.total_energy,
                "feature_summary": features,
                "synthetic_sidecar_record": record,
                "replay_score": score.to_json(),
            }
        )
    scored.sort(key=lambda item: item["fixture_id"])
    return scored


def synthetic_sidecar_record(
    row: Mapping[str, Any], *, root: Path | str = REPO_ROOT
) -> tuple[JsonDict, JsonDict]:
    """Build a schema-compatible cached row from one exact fixture manifest row."""

    del root
    fixture_id = str(row.get("source_fixture_id") or "unknown-fixture")
    payload = _payload(row)
    violation, reason = label_blind_violation(row)
    complexity = surface_complexity(payload)
    sidecar_action = "reject" if violation > 0.0 else "accept"
    confidence = round(max(0.05, min(0.99, 1.0 / (1.0 + violation + complexity))), 6)
    label_ref = expected_action(row)
    label_id = f"exp3117-label-{fixture_id}"
    token_logprobs = [-round(complexity, 6)]
    record = {
        "record_id": f"exp3117-{fixture_id}",
        "candidate": {
            "candidate_id": f"exp3117-candidate-{fixture_id}",
            "prompt_id": fixture_id,
            "candidate_text": candidate_text(payload),
            "candidate_label": sidecar_action,
            "model_id": "synthetic-replay-compatible/no-live-model",
            "token_logprobs": token_logprobs,
        },
        "constraints": [
            {
                "constraint_id": f"exp3117-feature-{fixture_id}",
                "description": "Label-blind fixture feature check before exact-label comparison.",
                "satisfied": violation == 0.0,
                "weight": 1.0,
                "violation_energy": violation,
                "label_ref": label_id,
            }
        ],
        "energy_terms": [
            {
                "name": "label_blind_feature_energy",
                "source": reason,
                "value": violation,
                "weight": 1.0,
            },
            {
                "name": "surface_complexity_energy",
                "source": "fixture_payload_shape",
                "value": complexity,
                "weight": 0.1,
            },
        ],
        "verifier_feedback": [
            {
                "verifier_id": "exp3117-label-blind-fixture-feature-summary",
                "status": "fail" if violation > 0.0 else "pass",
                "energy": violation,
                "message": reason,
                "violations": [] if violation == 0.0 else [reason],
            }
        ],
        "confidence": {
            "confidence": confidence,
            "abstain": False,
            "abstention_reason": "",
            "calibration_ref": "exp3117-sidecar-correlation-boundary-v3",
        },
        "exact_label_reference": {
            "label_id": label_id,
            "label": label_ref,
            "authority": "deterministic_tests",
            "source_artifact": str(MANIFEST_REL_PATH),
            "checksum": sha256_text(f"{label_id}:{label_ref}"),
        },
        "source_artifacts": [str(MANIFEST_REL_PATH), str(SIDECAR_SCORER_REL_PATH)],
    }
    features = {
        "label_blind_violation": violation,
        "feature_reason": reason,
        "surface_complexity": complexity,
        "sidecar_action": sidecar_action,
        "uses_exact_label_reference_for_score": False,
    }
    return record, features


def label_blind_violation(row: Mapping[str, Any]) -> tuple[float, str]:
    """Return a deterministic violation summary without consulting exact labels."""

    payload = _payload(row)
    family = str(row.get("task_family") or "")
    perturbation = str(row.get("perturbation_type") or "")
    if family == "arithmetic_code_assertions" or perturbation == "python_assertion_repair":
        return arithmetic_violation(payload)
    if family == "smt_constraints" and "constraints" in payload:
        return smt_satisfiability_violation(payload)
    if perturbation == "json_syntax_repair":
        return json_repair_violation(payload)
    if perturbation == "numeric_bound_repair":
        return numeric_assignment_violation(payload)
    if "candidate_assertion" in payload:
        return arithmetic_violation(payload)
    if "candidate" in payload:
        return json_repair_violation(payload)
    return 0.0, "no_label_blind_violation_detected"


def arithmetic_violation(payload: Mapping[str, Any]) -> tuple[float, str]:
    """Evaluate simple assertion fixtures with a safe arithmetic AST walker."""

    assertion = str(payload.get("candidate_assertion") or "")
    try:
        parsed = ast.parse(assertion, mode="exec")
        statement = parsed.body[0]
        if not isinstance(statement, ast.Assert) or not isinstance(statement.test, ast.Compare):
            return 1.0, "unsupported_arithmetic_assertion_shape"
        compare = statement.test
        if len(compare.ops) != 1 or not isinstance(compare.ops[0], ast.Eq):
            return 1.0, "unsupported_arithmetic_comparison"
        left = _eval_arithmetic_ast(compare.left)
        right = _eval_arithmetic_ast(compare.comparators[0])
    except (SyntaxError, ValueError, IndexError):
        return 1.0, "arithmetic_assertion_parse_failed"
    gap = abs(left - right)
    if gap == 0.0:
        return 0.0, "arithmetic_assertion_passed"
    return round(1.0 + min(gap, 100.0) / 100.0, 10), "arithmetic_assertion_failed"


def smt_satisfiability_violation(payload: Mapping[str, Any]) -> tuple[float, str]:
    """Detect simple impossible integer bounds from constraint strings."""

    constraints = [str(item) for item in payload.get("constraints", [])]
    lowers: dict[str, float] = {}
    uppers: dict[str, float] = {}
    for text in constraints:
        match = _SIMPLE_CMP_RE.match(text)
        if match is None:
            continue
        var, op, raw_value = match.groups()
        value = float(raw_value)
        if op == ">=":
            lowers[var] = max(value, lowers.get(var, value))
        elif op == "<=":
            uppers[var] = min(value, uppers.get(var, value))
    gaps = [
        lowers[var] - uppers[var] for var in set(lowers) & set(uppers) if lowers[var] > uppers[var]
    ]
    if gaps:
        return round(1.0 + max(gaps), 10), "smt_bounds_contradiction_detected"
    return 0.0, "smt_no_obvious_bounds_contradiction"


def json_repair_violation(payload: Mapping[str, Any]) -> tuple[float, str]:
    """Check whether a JSON repair fixture candidate parses and has required keys."""

    raw_candidate = payload.get("candidate")
    if not isinstance(raw_candidate, str):
        return 1.0, "json_candidate_missing"
    try:
        parsed = json.loads(raw_candidate)
    except json.JSONDecodeError:
        return 1.0, "json_decode_error"
    required = [str(field) for field in payload.get("required_fields", [])]
    if isinstance(parsed, Mapping) and all(field in parsed for field in required):
        return 0.0, "json_candidate_valid"
    return 1.0, "json_required_fields_missing"


def numeric_assignment_violation(payload: Mapping[str, Any]) -> tuple[float, str]:
    """Evaluate simple integer assignment constraints for repair fixtures."""

    assignment = payload.get("candidate_assignment")
    if not isinstance(assignment, Mapping):
        return 1.0, "numeric_assignment_missing"
    values = {
        str(key): float(value)
        for key, value in assignment.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    total_gap = 0.0
    for text in [str(item) for item in payload.get("constraints", [])]:
        simple = _SIMPLE_CMP_RE.match(text)
        if simple is not None:
            var, op, raw_value = simple.groups()
            value = float(raw_value)
            actual = values.get(var)
            if actual is None:
                total_gap += 1.0
            elif op == ">=" and actual < value:
                total_gap += value - actual
            elif op == "<=" and actual > value:
                total_gap += actual - value
            elif op == "==" and actual != value:
                total_gap += abs(actual - value)
            continue
        summed = _SUM_EQ_RE.match(text)
        if summed is not None:
            left, right, raw_value = summed.groups()
            expected = float(raw_value)
            actual_sum = values.get(left, math.nan) + values.get(right, math.nan)
            if not math.isfinite(actual_sum):
                total_gap += 1.0
            elif actual_sum != expected:
                total_gap += abs(actual_sum - expected)
    if total_gap == 0.0:
        return 0.0, "numeric_assignment_satisfies_constraints"
    return round(1.0 + min(total_gap, 100.0) / 100.0, 10), "numeric_assignment_violates_constraints"


def score_correlation_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return rank-correlation and separability summaries for sidecar scores."""

    return {
        "label_blind_feature_energy": score_summary(
            rows,
            "label_blind_feature_energy",
            uses_exact_label_reference=False,
        ),
        "replay_total_energy_label_aware": score_summary(
            rows,
            "replay_total_energy",
            uses_exact_label_reference=True,
        ),
    }


def score_summary(
    rows: Sequence[Mapping[str, Any]], score_name: str, *, uses_exact_label_reference: bool
) -> JsonDict:
    """Summarize one score column against exact outcome labels."""

    scores = [float(row.get(score_name) or 0.0) for row in rows]
    binary = [float(row.get("reject_or_repair_label") or 0.0) for row in rows]
    ordinal = [float(row.get("outcome_ordinal_label") or 0.0) for row in rows]
    return {
        "score_name": score_name,
        "uses_exact_label_reference": uses_exact_label_reference,
        "count": len(rows),
        "spearman_reject_or_repair": spearman(scores, binary),
        "spearman_outcome_ordinal": spearman(scores, ordinal),
        "pearson_reject_or_repair": pearson(scores, binary),
        "separability_vs_accept": separability(rows, score_name),
        "mean_by_outcome": mean_by_outcome(rows, score_name),
        "finite": all(math.isfinite(value) for value in scores),
    }


def calibration_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return coarse score bins so calibration weaknesses remain inspectable."""

    bins = calibration_bins(rows, "label_blind_feature_energy", bin_count=4)
    return {
        "score_name": "label_blind_feature_energy",
        "bins": bins,
        "bin_count": len(bins),
        "accounts_for_all_rows": sum(row["count"] for row in bins) == len(rows),
        "calibration_boundary": (
            "Bins compare label-blind feature energy to exact outcomes; they are not "
            "probability calibration for live inference."
        ),
    }


def calibration_bins(
    rows: Sequence[Mapping[str, Any]], score_name: str, *, bin_count: int = 4
) -> list[JsonDict]:
    """Split scored rows into equal-count bins and report exact outcome rates."""

    if not rows:
        return []
    ordered = sorted(
        rows, key=lambda row: (float(row.get(score_name) or 0.0), str(row["fixture_id"]))
    )
    active_bins = max(1, min(int(bin_count), len(ordered)))
    bins: list[JsonDict] = []
    for index in range(active_bins):
        start = index * len(ordered) // active_bins
        end = (index + 1) * len(ordered) // active_bins
        chunk = ordered[start:end]
        scores = [float(row.get(score_name) or 0.0) for row in chunk]
        outcomes = Counter(str(row.get("exact_outcome") or "") for row in chunk)
        bins.append(
            {
                "bin_index": index,
                "count": len(chunk),
                "score_min": round(min(scores), 10),
                "score_max": round(max(scores), 10),
                "mean_score": round(sum(scores) / len(scores), 10),
                "accepted_count": outcomes.get("accepted", 0),
                "rejected_count": outcomes.get("rejected", 0),
                "repairable_count": outcomes.get("repairable", 0),
                "reject_or_repair_rate": rate(
                    outcomes.get("rejected", 0) + outcomes.get("repairable", 0),
                    len(chunk),
                ),
            }
        )
    return bins


def separability(rows: Sequence[Mapping[str, Any]], score_name: str) -> float:
    """Return the pairwise probability that bad rows score above accepted rows."""

    accepted = [
        float(row.get(score_name) or 0.0) for row in rows if row.get("exact_outcome") == "accepted"
    ]
    bad = [
        float(row.get(score_name) or 0.0) for row in rows if row.get("exact_outcome") != "accepted"
    ]
    if not accepted or not bad:
        return 0.0
    wins = 0.0
    total = 0
    for bad_score in bad:
        for accept_score in accepted:
            total += 1
            if bad_score > accept_score:
                wins += 1.0
            elif bad_score == accept_score:
                wins += 0.5
    return rate(wins, total)


def failure_cases(
    rows: Sequence[Mapping[str, Any]], score_name: str, *, limit: int = 10
) -> list[JsonDict]:
    """Return sidecar action mismatches and score-overlap cases for inspection."""

    accepted_scores = [
        float(row.get(score_name) or 0.0) for row in rows if row.get("exact_outcome") == "accepted"
    ]
    max_accepted = max(accepted_scores) if accepted_scores else math.inf
    failures: list[JsonDict] = []
    for row in rows:
        score = float(row.get(score_name) or 0.0)
        reason = ""
        if row.get("sidecar_action") != row.get("expected_action"):
            reason = "sidecar_action_mismatch"
        elif row.get("exact_outcome") != "accepted" and score <= max_accepted:
            reason = "nonaccepted_score_not_above_accepted_boundary"
        if reason:
            failures.append(
                {
                    "fixture_id": row.get("fixture_id"),
                    "reason": reason,
                    "exact_outcome": row.get("exact_outcome"),
                    "expected_action": row.get("expected_action"),
                    "sidecar_action": row.get("sidecar_action"),
                    "score_name": score_name,
                    "score": round(score, 10),
                    "perturbation_type": row.get("perturbation_type"),
                }
            )
    failures.sort(
        key=lambda item: (item["reason"] != "sidecar_action_mismatch", -float(item["score"]))
    )
    return failures[:limit]


def count_outcomes(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count accepted, rejected, and repairable exact fixture classes."""

    counts = Counter(str(row.get("exact_outcome") or "") for row in rows)
    return {
        "accepted": counts.get("accepted", 0),
        "rejected": counts.get("rejected", 0),
        "repairable": counts.get("repairable", 0),
    }


def expected_action(row: Mapping[str, Any]) -> str:
    """Return exact post-generation action for an exact fixture row."""

    target = row.get("verifier_target")
    if isinstance(target, Mapping) and target.get("expected_action"):
        return str(target["expected_action"])
    answer = str(row.get("expected_answer") or "").upper()
    if answer in {"VALID", "SAT"}:
        return "accept"
    if answer in {"INVALID", "UNSAT", "REPAIRABLE"}:
        return "reject"
    return "abstain"


def exact_outcome_class(row: Mapping[str, Any]) -> str:
    """Map exact fixture metadata into accepted, rejected, or repairable."""

    repair = row.get("repair_target")
    if isinstance(repair, Mapping) and repair.get("applicable") is True:
        return "repairable"
    return "accepted" if expected_action(row) == "accept" else "rejected"


def mean_by_outcome(rows: Sequence[Mapping[str, Any]], score_name: str) -> JsonDict:
    """Average one score by exact outcome class."""

    output: JsonDict = {}
    for outcome in ("accepted", "rejected", "repairable"):
        values = [
            float(row.get(score_name) or 0.0) for row in rows if row.get("exact_outcome") == outcome
        ]
        output[outcome] = round(sum(values) / len(values), 10) if values else 0.0
    return output


def finite_correlation_summary(summary: Mapping[str, Any]) -> bool:
    """Check that every numeric top-level score summary value is finite."""

    for score_payload in summary.values():
        if not isinstance(score_payload, Mapping):
            return False
        for key in (
            "spearman_reject_or_repair",
            "spearman_outcome_ordinal",
            "pearson_reject_or_repair",
            "separability_vs_accept",
        ):
            value = score_payload.get(key)
            if not isinstance(value, int | float) or not math.isfinite(float(value)):
                return False
    return True


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Compute Spearman rank correlation with average ranks for ties."""

    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    return pearson(_ranks(xs), _ranks(ys))


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Compute Pearson correlation, returning zero for constant inputs."""

    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [float(value) - mean_x for value in xs]
    dy = [float(value) - mean_y for value in ys]
    denom_x = math.sqrt(sum(value * value for value in dx))
    denom_y = math.sqrt(sum(value * value for value in dy))
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    return round(sum(x * y for x, y in zip(dx, dy, strict=True)) / (denom_x * denom_y), 10)


def rate(numerator: float, denominator: float) -> float:
    """Return a rounded rate, using zero when a metric has no denominator."""

    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 10)


def relative_path(root: Path, path: Path) -> str:
    """Return a repository-relative path when possible."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def source_artifacts(root: Path, manifest_rel_path: Path) -> list[JsonDict]:
    """Return concrete source files used by the diagnostic."""

    rows = []
    for source_id, rel_path, required in SOURCE_SPECS:
        path = manifest_rel_path if source_id == "exp3097_stratified_manifest" else rel_path
        absolute = root / path
        rows.append(
            {
                "id": source_id,
                "path": path.as_posix(),
                "required": required,
                "exists": absolute.is_file(),
                "sha256": sha256_file(absolute) if absolute.is_file() else None,
            }
        )
    return rows


def inference_substrate() -> JsonDict:
    """Return the diagnostic-only runtime boundary for the artifact."""

    return {
        "kind": "offline_checked_in_fixture_sidecar_diagnostic",
        "uses_checked_in_artifacts_only": True,
        "sidecar_only": True,
        "live_model_inference": False,
        "live_llm_inference": False,
        "model_weights_loaded": False,
        "generation_performed": False,
        "training_performed": False,
        "weight_update_performed": False,
        "gpu_required": False,
        "speedup_claimed": False,
    }


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict that does not hide blocked preconditions."""

    if artifact.get("sidecar_score_correlation_boundary_v3_ready") is True:
        return (
            "complete: sidecar_score_correlation_boundary_v3_ready=true; "
            f"exact_fixture_count={artifact.get('exact_fixture_count')}; "
            "diagnostic_only_no_live_integration_no_weight_update_no_speedup"
        )
    checks = artifact.get("readiness_checks")
    if not isinstance(checks, Mapping):
        return "blocked_incomplete_diagnostics: readiness checks missing"
    if (
        checks.get("exp3097_protocol_ready") is not True
        or checks.get("required_sources_present") is not True
    ):
        return "blocked_missing_inputs: exp3097 protocol or stratified manifest unavailable"
    if checks.get("minimum_exact_fixture_count_met") is not True:
        return (
            "blocked_insufficient_exact_fixtures: "
            f"exact_fixture_count={artifact.get('exact_fixture_count')} "
            f"minimum={artifact.get('minimum_exact_fixture_count')}"
        )
    if (
        checks.get("accepted_cases_present") is not True
        or checks.get("rejected_cases_present") is not True
        or checks.get("repairable_cases_present") is not True
    ):
        return "blocked_missing_outcome_classes: accepted_rejected_repairable_required"
    return "blocked_incomplete_diagnostics: finite metrics or calibration accounting missing"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject artifacts that omit metrics or overclaim sidecar integration."""

    missing = sorted(set(REQUIRED_FIELDS) - set(artifact))
    _require(not missing, f"missing required fields: {missing}")
    _require(
        artifact.get("no_live_model_integration_claim") is True,
        "no_live_model_integration_claim must be true",
    )
    _require(artifact.get("no_weight_update_claim") is True, "no_weight_update_claim must be true")
    _require(artifact.get("no_speedup_claim") is True, "no_speedup_claim must be true")
    _require(isinstance(artifact.get("failure_cases"), list), "failure_cases must be a list")
    _require(
        isinstance(artifact.get("source_artifacts"), list) and artifact.get("source_artifacts"),
        "source_artifacts must be non-empty",
    )
    substrate = artifact.get("inference_substrate")
    _require(isinstance(substrate, Mapping), "inference_substrate must be an object")
    _require(substrate.get("live_model_inference") is False, "live_model_inference must be false")
    _require(substrate.get("live_llm_inference") is False, "live_llm_inference must be false")
    _require(substrate.get("model_weights_loaded") is False, "model_weights_loaded must be false")
    _require(substrate.get("generation_performed") is False, "generation_performed must be false")
    _require(
        substrate.get("weight_update_performed") is False, "weight_update_performed must be false"
    )
    _require(substrate.get("speedup_claimed") is False, "speedup_claimed must be false")
    _require(
        finite_correlation_summary(artifact["score_correlation_summary"]), "finite metrics required"
    )
    calibration = artifact.get("calibration_summary")
    _require(isinstance(calibration, Mapping), "calibration_summary must be an object")
    _require(
        sum(row.get("count", 0) for row in calibration.get("bins", []))
        == artifact.get("exact_fixture_count"),
        "calibration bins must account for exact_fixture_count",
    )
    verdict = str(artifact.get("honest_verdict", ""))
    if artifact.get("sidecar_score_correlation_boundary_v3_ready") is True:
        _require(verdict.startswith(SUCCESS_PREFIXES), "honest_verdict must start success prefix")
        _require(
            int(artifact.get("exact_fixture_count") or 0)
            >= int(artifact.get("minimum_exact_fixture_count") or 0),
            "ready artifact must meet minimum exact fixture count",
        )
    else:
        _require(
            verdict.startswith(BLOCKED_PREFIXES),
            "honest_verdict must start success prefix or blocked prefix",
        )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON output for experiment artifacts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def duration(started_s: float, now_s: float | None) -> float:
    """Return a rounded wall-clock duration."""

    current = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, current - float(started_s)), 6)


def sha256_file(path: Path) -> str:
    """Hash a source artifact for traceability."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    """Hash a short deterministic label payload."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def surface_complexity(payload: Mapping[str, Any]) -> float:
    """Return a small deterministic complexity proxy from candidate shape."""

    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return round(min(1.0, len(text) / 500.0), 10)


def candidate_text(payload: Mapping[str, Any]) -> str:
    """Pick a compact candidate string for the synthetic sidecar row."""

    for key in ("candidate_assertion", "candidate", "candidate_assignment", "constraints"):
        value = payload.get(key)
        if value is not None:
            return json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
    return json.dumps(payload, sort_keys=True)


def _payload(row: Mapping[str, Any]) -> JsonDict:
    payload = row.get("leakage_safe_prompt_payload")
    return dict(payload) if isinstance(payload, Mapping) else {}


def _eval_arithmetic_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_arithmetic_ast(node.operand)
    if isinstance(node, ast.BinOp):
        left = _eval_arithmetic_ast(node.left)
        right = _eval_arithmetic_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div) and right != 0.0:
            return left / right
    msg = f"unsupported arithmetic node: {type(node).__name__}"
    raise ValueError(msg)


def _ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(float(value) for value in values), key=lambda item: item[1])
    ranks = [0.0] * len(indexed)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for index in range(cursor, end):
            ranks[indexed[index][0]] = average_rank
        cursor = end
    return ranks


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

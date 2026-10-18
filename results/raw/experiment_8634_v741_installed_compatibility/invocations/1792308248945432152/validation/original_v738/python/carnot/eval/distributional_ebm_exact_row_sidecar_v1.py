"""Exp 3182 distributional EBM exact-row sidecar v1.

Spec refs: REQ-VERIFY-3182, SCENARIO-VERIFY-3182.

This module is an offline diagnostic sidecar. It does not train a
distributional EBM, call a model, or install a verifier. The implementation
uses exact verifier rows, cached candidate evidence, EBCN/KAN sidecar records,
and controlled-invariance controls to build a deterministic proxy energy and
uncertainty ranking. That proxy is useful for routing and calibration
discussion, but exact verifier authority remains load-bearing for any accept or
reject decision.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
ARTIFACT = "experiment_3182_distributional_ebm_exact_row_sidecar_v1"
SCHEMA = "carnot.distributional_ebm_exact_row_sidecar.v1"
OUTPUT_REL_PATH = Path("results/experiment_3182_distributional_ebm_exact_row_sidecar_v1.json")

EXP3173_REL_PATH = Path("results/experiment_3173_ebcn_kan_bounded_diagnostic_expansion_v2.json")
EXP3180_REL_PATH = Path("results/experiment_3180_controlled_invariance_executor_v2.json")
RESEARCH_REFERENCES_REL_PATH = Path("research-references.md")
SPEC_REL_PATH = Path("openspec/capabilities/verification/spec.md")

REQUIRED_ARTIFACT_FIELDS = (
    "distributional_ebm_exact_row_sidecar_v1_ready",
    "source_reference_ids",
    "exact_labeled_row_count",
    "known_false_accept_rows_scored",
    "sidecar_method",
    "false_accept_separation_auc",
    "uncertainty_calibration",
    "abstention_policy",
    "comparison_to_ebcn_kan",
    "deployed_verifier_claim_allowed",
    "inference_substrate",
    "honest_verdict",
)
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
BLOCKED_PREFIXES = ("blocked_", "blocked:")
ACCEPT_LABELS = {"VALID", "SAT", "TRUE", "CORRECT", "PASS", "ACCEPT"}
REJECT_LABELS = {"INVALID", "UNSAT", "FALSE", "INCORRECT", "FAIL", "REJECT"}
SOURCE_REFERENCE_IDS = (
    "arXiv:2605.18871",
    "arXiv:2603.23398",
    "exp3173_ebcn_kan_bounded_diagnostic_expansion_v2",
    "exp3180_controlled_invariance_executor_v2",
    "research-references:2026-05-27-post-294-planning-sweep",
)
SOURCE_SPECS = (
    ("research_references", RESEARCH_REFERENCES_REL_PATH, True, "md"),
    ("verification_openspec", SPEC_REL_PATH, False, "md"),
    ("exp3173_ebcn_kan", EXP3173_REL_PATH, True, "json"),
    ("exp3180_controlled_invariance", EXP3180_REL_PATH, True, "json"),
    (
        "exp3182_module",
        Path("python/carnot/eval/distributional_ebm_exact_row_sidecar_v1.py"),
        False,
        "python",
    ),
    (
        "exp3182_tests",
        Path("tests/python/test_experiment_3182_distributional_ebm_exact_row_sidecar_v1.py"),
        False,
        "python",
    ),
)
FEATURE_WEIGHTS = {
    "candidate_conflict": 0.42,
    "shortcut_exposure": 0.28,
    "ebcn_scalar_energy": 0.20,
    "exact_replay_disagreement": 0.10,
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3182_distributional_ebm_exact_row_sidecar_v1.py -q --no-cov",
    ".venv/bin/coverage run --source=python/carnot/eval/distributional_ebm_exact_row_sidecar_v1.py -m pytest -o addopts='' tests/python/test_experiment_3182_distributional_ebm_exact_row_sidecar_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/distributional_ebm_exact_row_sidecar_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_3182_distributional_ebm_exact_row_sidecar_v1.py",
    ".venv/bin/pytest tests/python -q",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3182: build the offline exact-row uncertainty sidecar."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    exp3173 = read_json_object(root_path / EXP3173_REL_PATH)
    exp3180 = read_json_object(root_path / EXP3180_REL_PATH)
    source_rows = source_artifacts(root_path)
    source_errors = required_source_errors(source_rows)
    expected_false_ids = known_false_accept_ids(exp3173, exp3180)
    exact_rows = collect_exact_rows(exp3173, exp3180, expected_false_ids)
    scored_rows = score_rows(exact_rows, shortcut_exposure_ids(exp3180))
    positive_scores = [
        row["proxy_energy"] for row in scored_rows if row["known_false_accept"] is True
    ]
    negative_scores = [
        row["proxy_energy"] for row in scored_rows if row["known_false_accept"] is not True
    ]
    auc = false_accept_auc(positive_scores, negative_scores)
    calibration = expected_calibration_error(scored_rows, bin_count=5)
    policy = abstention_policy(scored_rows)
    coverage = coverage_denominator(scored_rows)
    substrate = inference_substrate(exp3173, exp3180)
    ready_checks = readiness_checks(
        source_errors=source_errors,
        scored_rows=scored_rows,
        expected_false_ids=expected_false_ids,
        auc=auc,
        calibration=calibration,
        policy=policy,
        substrate=substrate,
    )
    ready = all(ready_checks.values())
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3182", "SCENARIO-VERIFY-3182"],
        "distributional_ebm_exact_row_sidecar_v1_ready": ready,
        "source_reference_ids": list(SOURCE_REFERENCE_IDS),
        "exact_labeled_row_count": len(scored_rows),
        "known_false_accept_rows_scored": len(positive_scores),
        "sidecar_method": sidecar_method(),
        "false_accept_separation_auc": auc,
        "uncertainty_calibration": calibration,
        "abstention_policy": policy,
        "comparison_to_ebcn_kan": comparison_to_ebcn_kan(exp3173, exp3180, auc, scored_rows),
        "deployed_verifier_claim_allowed": False,
        "inference_substrate": substrate,
        "coverage_denominator": coverage,
        "row_scores": scored_rows,
        "uncertainty_ranking": uncertainty_ranking(scored_rows),
        "source_artifacts": source_rows,
        "source_checksums": {
            row["path"]: row["sha256"] for row in source_rows if row["sha256"] is not None
        },
        "source_errors": source_errors,
        "readiness_checks": ready_checks,
        "blocked_reasons": [name for name, ok in ready_checks.items() if ok is not True],
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
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
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3182 terminal JSON artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(
        root_path,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    write_json(out_path, artifact)
    return out_path


def collect_exact_rows(
    exp3173: Mapping[str, Any],
    exp3180: Mapping[str, Any],
    expected_false_ids: set[str],
) -> list[JsonDict]:
    """Join Exp 3173 exact rows to Exp 3180 cached candidate/control rows."""

    rows_by_id: dict[str, JsonDict] = {}
    exp3180_by_id = first_rows_by_id(mapping_rows(exp3180.get("exact_rows_evaluated")))
    for row in mapping_rows(exp3173.get("exact_rows")):
        row_id = str(row.get("row_id") or "")
        if not row_id:
            continue
        control_row = exp3180_by_id.get(row_id, {})
        joined = normalized_row(row_id, row, control_row, expected_false_ids)
        rows_by_id[row_id] = joined
    for row_id, row in exp3180_by_id.items():
        if row_id not in rows_by_id:
            rows_by_id[row_id] = normalized_row(row_id, {}, row, expected_false_ids)
    return sorted(rows_by_id.values(), key=lambda item: item["row_id"])


def normalized_row(
    row_id: str,
    exp3173_row: Mapping[str, Any],
    exp3180_row: Mapping[str, Any],
    expected_false_ids: set[str],
) -> JsonDict:
    """Normalize one row while preserving exact and cached-candidate provenance."""

    ebcn_score = exp3173_row.get("ebcn_score")
    ebcn_map = ebcn_score if isinstance(ebcn_score, Mapping) else {}
    kan_record = exp3173_row.get("kan_monitor_record")
    kan_map = kan_record if isinstance(kan_record, Mapping) else {}
    candidate_answers = string_list(exp3180_row.get("candidate_answers"))
    exact_label = str(exp3173_row.get("exact_label") or exp3180_row.get("exact_label") or "")
    return {
        "row_id": row_id,
        "exact_label": exact_label,
        "expected_action": str(exp3173_row.get("expected_action") or ""),
        "known_false_accept": bool(
            exp3173_row.get("known_false_accept") is True
            or exp3180_row.get("known_false_accept_regression") is True
            or row_id in expected_false_ids
        ),
        "fixture_family": str(exp3173_row.get("fixture_family") or ""),
        "contract_decision": str(exp3173_row.get("contract_decision") or ""),
        "live_decision": str(exp3173_row.get("live_decision") or ""),
        "exact_authority_decision": str(exp3180_row.get("exact_authority_decision") or ""),
        "candidate_answers": candidate_answers,
        "monitor_event_count": int(as_float(exp3173_row.get("monitor_event_count"), 0.0)),
        "ebcn_scalar_energy": nullable_float(ebcn_map.get("scalar_energy")),
        "ebcn_branch_count": len(mapping_rows(ebcn_map.get("energy_branches"))),
        "kan_monitor_present": bool(kan_map),
        "kan_solver_status": str(kan_map.get("solver_status") or ""),
    }


def score_rows(rows: Sequence[Mapping[str, Any]], shortcut_ids: Sequence[str]) -> list[JsonDict]:
    """Attach deterministic proxy-energy, probability, and graph features."""

    shortcut_set = set(shortcut_ids)
    energy_values = [
        as_float(row.get("ebcn_scalar_energy"))
        for row in rows
        if row.get("ebcn_scalar_energy") is not None
    ]
    min_energy = min(energy_values) if energy_values else 0.0
    max_energy = max(energy_values) if energy_values else 0.0
    scored: list[JsonDict] = []
    for row in rows:
        row_id = str(row.get("row_id") or "")
        branches = feature_branches(row, row_id in shortcut_set, min_energy, max_energy)
        proxy_energy = round(
            sum(branches[name] * FEATURE_WEIGHTS[name] for name in FEATURE_WEIGHTS),
            6,
        )
        risk_probability = round(logistic_probability(proxy_energy), 6)
        uncertainty = uncertainty_score(row, branches, risk_probability)
        scored.append(
            {
                **dict(row),
                "graph_features": graph_features(row),
                "feature_branches": branches,
                "proxy_energy": proxy_energy,
                "risk_probability": risk_probability,
                "uncertainty_score": uncertainty,
                "abstention_priority": round(
                    max(proxy_energy, risk_probability, uncertainty),
                    6,
                ),
                "score_explanation": score_explanation(branches),
            }
        )
    return scored


def feature_branches(
    row: Mapping[str, Any],
    shortcut_exposed: bool,
    min_energy: float,
    max_energy: float,
) -> JsonDict:
    """Return the deterministic feature branches used by the proxy score."""

    ebcn_scalar = row.get("ebcn_scalar_energy")
    return {
        "candidate_conflict": 1.0 if candidate_conflict(row.get("candidate_answers")) else 0.0,
        "shortcut_exposure": 1.0 if shortcut_exposed else 0.0,
        "ebcn_scalar_energy": normalized_energy(ebcn_scalar, min_energy, max_energy),
        "exact_replay_disagreement": 1.0 if exact_replay_disagreement(row) else 0.0,
    }


def graph_features(row: Mapping[str, Any]) -> JsonDict:
    """Represent a row as a small constraint graph summary."""

    candidate_count = len(string_list(row.get("candidate_answers")))
    evidence_nodes = int(row.get("ebcn_scalar_energy") is not None) + int(
        row.get("kan_monitor_present") is True
    )
    return {
        "constraint_family": str(row.get("fixture_family") or "unknown"),
        "node_count": 3 + candidate_count + evidence_nodes,
        "candidate_edge_count": candidate_count,
        "has_ebcn_energy_node": row.get("ebcn_scalar_energy") is not None,
        "has_kan_monitor_node": row.get("kan_monitor_present") is True,
        "controlled_invariance_edge": bool(row.get("exact_authority_decision")),
    }


def candidate_conflict(value: Any) -> bool:
    """Return true when cached answers contain both accept and reject tokens."""

    polarities = {answer_polarity(answer) for answer in string_list(value)}
    return "accept" in polarities and "reject" in polarities


def answer_polarity(answer: str) -> str:
    """Map cached answer tokens to coarse accept/reject/other polarity."""

    token = answer.strip().upper()
    if token in ACCEPT_LABELS:
        return "accept"
    if token in REJECT_LABELS:
        return "reject"
    return "other"


def exact_replay_disagreement(row: Mapping[str, Any]) -> bool:
    """Detect cached live accepts that exact authority routes to rejection."""

    expected = str(row.get("expected_action") or "").lower()
    live = str(row.get("live_decision") or "").lower()
    authority = str(row.get("exact_authority_decision") or "").lower()
    return live == "accept" and (expected == "reject" or authority == "reject")


def normalized_energy(value: Any, min_energy: float, max_energy: float) -> float:
    """Normalize available EBCN scalar energy without inventing missing evidence."""

    if value is None:
        return 0.0
    energy = as_float(value)
    spread = max_energy - min_energy
    if spread <= 0.0:
        return round(max(0.0, min(1.0, energy)), 6)
    return round(max(0.0, min(1.0, (energy - min_energy) / spread)), 6)


def logistic_probability(energy: float) -> float:
    """Map proxy energy to a smooth false-accept risk probability."""

    return 1.0 / (1.0 + math.exp(-6.0 * (energy - 0.5)))


def uncertainty_score(
    row: Mapping[str, Any],
    branches: Mapping[str, float],
    risk_probability: float,
) -> float:
    """Compute uncertainty from boundary proximity, missing evidence, and disagreement."""

    values = [float(value) for value in branches.values()]
    disagreement = max(values) - min(values) if values else 0.0
    boundary = 1.0 - abs(risk_probability - 0.5) * 2.0
    missing = 0.0 if row.get("ebcn_scalar_energy") is not None else 0.35
    return round(0.35 * boundary + 0.35 * missing + 0.30 * disagreement, 6)


def score_explanation(branches: Mapping[str, float]) -> str:
    """Explain whether the score is mainly candidate, shortcut, or energy driven."""

    if not branches:
        return "no_feature_evidence"
    name, value = max(branches.items(), key=lambda item: item[1])
    if value <= 0.0:
        return "low_proxy_energy_no_branch_triggered"
    return f"highest_branch={name}"


def false_accept_auc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float | None:
    """Compute false-accept-vs-clean AUROC with tie handling."""

    if not positive_scores or not negative_scores:
        return None
    wins = 0.0
    total = 0
    for positive in positive_scores:
        for negative in negative_scores:
            total += 1
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return round(wins / total, 6)


def expected_calibration_error(
    rows: Sequence[Mapping[str, Any]],
    *,
    bin_count: int,
) -> JsonDict:
    """Compute ECE for proxy false-accept probabilities when both classes exist."""

    labels = [1 if row.get("known_false_accept") is True else 0 for row in rows]
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if not rows or positive_count == 0 or negative_count == 0:
        return {
            "ece_meaningful": False,
            "reason": "requires both positive and negative false-accept classes",
            "sample_count": len(rows),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "bin_count": bin_count,
            "bins": [],
            "expected_calibration_error": None,
            "brier_score": None,
            "calibration_method": "deterministic proxy; no training or posthoc fit",
        }

    bins: list[JsonDict] = []
    for index in range(bin_count):
        bins.append(
            {
                "bin_index": index,
                "lower": round(index / bin_count, 6),
                "upper": round((index + 1) / bin_count, 6),
                "count": 0,
                "avg_probability": 0.0,
                "empirical_false_accept_rate": 0.0,
                "abs_gap": 0.0,
            }
        )

    assignments: list[list[Mapping[str, Any]]] = [[] for _ in range(bin_count)]
    for row in rows:
        probability = max(0.0, min(1.0, as_float(row.get("risk_probability"))))
        index = min(bin_count - 1, int(probability * bin_count))
        assignments[index].append(row)

    ece = 0.0
    brier_total = 0.0
    sample_count = len(rows)
    for index, assigned in enumerate(assignments):
        if assigned:
            probabilities = [as_float(row.get("risk_probability")) for row in assigned]
            assigned_labels = [1 if row.get("known_false_accept") is True else 0 for row in assigned]
            avg_probability = sum(probabilities) / len(probabilities)
            empirical_rate = sum(assigned_labels) / len(assigned_labels)
            gap = abs(avg_probability - empirical_rate)
            bins[index]["count"] = len(assigned)
            bins[index]["avg_probability"] = round(avg_probability, 6)
            bins[index]["empirical_false_accept_rate"] = round(empirical_rate, 6)
            bins[index]["abs_gap"] = round(gap, 6)
            ece += (len(assigned) / sample_count) * gap
            brier_total += sum(
                (probability - label) ** 2
                for probability, label in zip(probabilities, assigned_labels, strict=True)
            )
    return {
        "ece_meaningful": True,
        "reason": "computed over deterministic proxy risk probabilities",
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "bin_count": bin_count,
        "bins": bins,
        "expected_calibration_error": round(ece, 6),
        "brier_score": round(brier_total / sample_count, 6),
        "calibration_method": "deterministic proxy; no training or posthoc fit",
    }


def abstention_policy(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Propose an offline abstention threshold from the scored exact rows."""

    positives = [
        as_float(row.get("abstention_priority"))
        for row in rows
        if row.get("known_false_accept") is True
    ]
    negatives = [
        as_float(row.get("abstention_priority"))
        for row in rows
        if row.get("known_false_accept") is not True
    ]
    if positives and negatives:
        threshold = round((min(positives) + max(negatives)) / 2.0, 6)
        reason = "midpoint between lowest known false-accept priority and highest clean priority"
    else:
        threshold = None
        reason = "threshold unavailable without both clean and known false-accept rows"
    abstained = [
        row for row in rows if threshold is not None and as_float(row.get("abstention_priority")) >= threshold
    ]
    false_abstained = [row for row in abstained if row.get("known_false_accept") is True]
    return {
        "deployed_policy": False,
        "threshold_metric": "abstention_priority",
        "threshold": threshold,
        "threshold_selection": reason,
        "coverage_denominator": len(rows),
        "rows_abstained": len(abstained),
        "coverage_after_abstention": rate(len(rows) - len(abstained), len(rows)),
        "known_false_accepts_abstained": len(false_abstained),
        "known_false_accept_recall": rate(len(false_abstained), len(positives)),
        "policy_note": "diagnostic routing proposal only; exact verifier authority still decides",
    }


def comparison_to_ebcn_kan(
    exp3173: Mapping[str, Any],
    exp3180: Mapping[str, Any],
    sidecar_auc: float | None,
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Reconcile Exp 3182 with prior EBCN/KAN and shortcut-control artifacts."""

    ebcn_metrics = exp3173.get("ebcn_localization_metrics")
    ebcn_map = ebcn_metrics if isinstance(ebcn_metrics, Mapping) else {}
    kan_metrics = exp3173.get("kan_monitor_coverage_metrics")
    kan_map = kan_metrics if isinstance(kan_metrics, Mapping) else {}
    control_rows = shortcut_exposure_ids(exp3180)
    return {
        "exp3173_exact_labeled_row_count": int(
            as_float(exp3173.get("exact_labeled_row_count"), 0.0)
        ),
        "exp3173_known_false_accept_rows_scored": int(
            as_float(exp3173.get("known_false_accept_rows_scored"), 0.0)
        ),
        "exp3173_false_accept_vs_clean_auc": nullable_float(
            ebcn_map.get("false_accept_vs_clean_auc")
        ),
        "exp3173_kan_monitor_record_count": int(
            as_float(kan_map.get("monitor_record_count"), 0.0)
        ),
        "sidecar_scored_row_count": len(rows),
        "sidecar_known_false_accept_rows_scored": sum(
            1 for row in rows if row.get("known_false_accept") is True
        ),
        "sidecar_false_accept_separation_auc": sidecar_auc,
        "reconciliation": (
            "Exp 3182 extends coverage with deterministic cached-row features, "
            "but it inherits Exp 3173's tiny known-false-accept denominator and "
            "does not promote EBCN/KAN scores into verifier authority."
        ),
        "exp3180_controlled_invariance": {
            "controlled_invariance_passed": exp3180.get("controlled_invariance_passed") is True,
            "known_false_accept_regression_count": int(
                as_float(exp3180.get("known_false_accept_regression_count"), 0.0)
            ),
            "shortcut_exposure_row_ids": control_rows,
            "shortcut_failure_count": int(as_float(exp3180.get("shortcut_failure_count"), 0.0)),
            "semantic_false_accept_count": int(
                as_float(exp3180.get("semantic_false_accept_count"), 0.0)
            ),
            "finding": (
                "Shortcut-exposed rows align with the sidecar's top abstention "
                "priority rows; Exp 3180 still routes those rows to exact checks."
            ),
        },
    }


def sidecar_method() -> JsonDict:
    """Describe the reproducible deterministic proxy method."""

    return {
        "method_type": "deterministic_proxy_distributional_ebm",
        "training_performed": False,
        "random_seed": None,
        "feature_weights": dict(FEATURE_WEIGHTS),
        "feature_inputs": [
            "cached candidate answer polarity conflict",
            "controlled-invariance shortcut exposure",
            "Exp 3173 EBCN scalar energy when present",
            "exact replay disagreement between live decision and exact authority",
        ],
        "graph_feature_summary": (
            "Rows are represented as small constraint graphs with row, exact "
            "authority, candidate-answer, EBCN, KAN, and control-edge counts."
        ),
        "calibration_method": "ECE over deterministic proxy risk probabilities",
        "inspiration_sources": ["arXiv:2605.18871", "arXiv:2603.23398"],
        "nondeployment_boundary": (
            "The score may route rows to abstention/exact checks but cannot accept outputs."
        ),
    }


def inference_substrate(exp3173: Mapping[str, Any], exp3180: Mapping[str, Any]) -> JsonDict:
    """State that this sidecar uses cached exact artifacts, not live inference."""

    exp3180_substrate = exp3180.get("inference_substrate")
    substrate_map = exp3180_substrate if isinstance(exp3180_substrate, Mapping) else {}
    return {
        "mode": "offline_exact_artifact_replay",
        "executes_models": False,
        "generation_performed": False,
        "training_performed": False,
        "new_live_model_calls": 0,
        "offline_exact_artifact_replay": True,
        "source_exp3173_ready": exp3173.get("ebcn_kan_bounded_diagnostic_expansion_v2_ready")
        is True,
        "source_exp3180_ready": exp3180.get("controlled_invariance_executor_v2_ready") is True,
        "source_exp3180_declared_live_calls": int(
            as_float(substrate_map.get("new_live_model_calls"), 0.0)
        ),
    }


def readiness_checks(
    *,
    source_errors: Sequence[str],
    scored_rows: Sequence[Mapping[str, Any]],
    expected_false_ids: set[str],
    auc: float | None,
    calibration: Mapping[str, Any],
    policy: Mapping[str, Any],
    substrate: Mapping[str, Any],
) -> dict[str, bool]:
    """Evaluate whether the diagnostic artifact was fully materialized."""

    scored_false_ids = {
        str(row.get("row_id"))
        for row in scored_rows
        if row.get("known_false_accept") is True and row.get("row_id")
    }
    return {
        "required_sources_present": not source_errors,
        "exact_rows_scored": bool(scored_rows),
        "known_false_accept_rows_scored": bool(scored_false_ids),
        "all_known_false_accepts_included": bool(expected_false_ids)
        and expected_false_ids <= scored_false_ids,
        "false_accept_auc_computable": auc is not None,
        "calibration_meaningful": calibration.get("ece_meaningful") is True,
        "abstention_policy_defined": policy.get("threshold") is not None
        and int(as_float(policy.get("coverage_denominator"), 0.0)) == len(scored_rows),
        "no_live_model_calls": substrate.get("new_live_model_calls") == 0
        and substrate.get("executes_models") is False,
        "nondeployment_boundary": True,
    }


def coverage_denominator(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report denominator details so the sidecar cannot overstate coverage."""

    false_count = sum(1 for row in rows if row.get("known_false_accept") is True)
    return {
        "exact_labeled_row_count": len(rows),
        "scored_row_count": len(rows),
        "clean_comparator_count": len(rows) - false_count,
        "known_false_accept_scored_count": false_count,
        "rows_with_ebcn_scalar_energy": sum(
            1 for row in rows if row.get("ebcn_scalar_energy") is not None
        ),
        "rows_with_candidate_evidence": sum(1 for row in rows if row.get("candidate_answers")),
        "denominator_note": (
            "The sidecar scores every joined exact row, but EBCN scalar evidence "
            "covers only the rows where Exp 3173 recorded it."
        ),
    }


def uncertainty_ranking(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return the highest-priority rows for abstention review."""

    ranked = sorted(
        rows,
        key=lambda row: (
            as_float(row.get("abstention_priority")),
            as_float(row.get("proxy_energy")),
            str(row.get("row_id")),
        ),
        reverse=True,
    )
    return [
        {
            "rank": index + 1,
            "row_id": str(row.get("row_id") or ""),
            "known_false_accept": row.get("known_false_accept") is True,
            "proxy_energy": row.get("proxy_energy"),
            "risk_probability": row.get("risk_probability"),
            "uncertainty_score": row.get("uncertainty_score"),
            "abstention_priority": row.get("abstention_priority"),
            "score_explanation": row.get("score_explanation"),
        }
        for index, row in enumerate(ranked[:10])
    ]


def known_false_accept_ids(exp3173: Mapping[str, Any], exp3180: Mapping[str, Any]) -> set[str]:
    """Collect known false-accept IDs from both upstream artifacts."""

    ids = set(string_list(exp3173.get("known_false_accept_row_ids")))
    for row in mapping_rows(exp3173.get("exact_rows")):
        row_id = str(row.get("row_id") or "")
        if row_id and row.get("known_false_accept") is True:
            ids.add(row_id)
    for row in mapping_rows(exp3180.get("exact_rows_evaluated")):
        row_id = str(row.get("row_id") or "")
        if row_id and row.get("known_false_accept_regression") is True:
            ids.add(row_id)
    return ids


def shortcut_exposure_ids(exp3180: Mapping[str, Any]) -> list[str]:
    """Collect shortcut-exposed row IDs from Exp 3180 control details."""

    controls = exp3180.get("control_results")
    control_map = controls if isinstance(controls, Mapping) else {}
    seen: set[str] = set()
    ordered: list[str] = []
    for control in control_map.values():
        if not isinstance(control, Mapping):
            continue
        details = control.get("details")
        detail_map = details if isinstance(details, Mapping) else {}
        for row_id in string_list(detail_map.get("shortcut_exposure_row_ids")):
            if row_id not in seen:
                seen.add(row_id)
                ordered.append(row_id)
    return ordered


def first_rows_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Index rows by their first non-empty row_id."""

    indexed: dict[str, JsonDict] = {}
    for row in rows:
        row_id = str(row.get("row_id") or "")
        if row_id and row_id not in indexed:
            indexed[row_id] = dict(row)
    return indexed


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return the source files consumed or cited by the sidecar."""

    rows: list[JsonDict] = []
    for source_id, rel_path, required, source_type in SOURCE_SPECS:
        path = root / rel_path
        rows.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "required": required,
                "source_type": source_type,
                "exists": path.is_file(),
                "sha256": file_sha256(path),
            }
        )
    return rows


def required_source_errors(source_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return missing required source IDs."""

    return [
        str(row.get("id") or "")
        for row in source_rows
        if row.get("required") is True and row.get("exists") is not True
    ]


def read_json_object(path: Path) -> JsonDict:
    """Read one checked-in JSON object, returning empty evidence on failure."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def mapping_rows(value: Any) -> list[JsonDict]:
    """Keep only JSON object rows from untrusted artifact lists."""

    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def string_list(value: Any) -> list[str]:
    """Return string members from a JSON list."""

    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def nullable_float(value: Any) -> float | None:
    """Convert finite numeric values while preserving null for missing metrics."""

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def as_float(value: Any, default: float = 0.0) -> float:
    """Convert artifact values to finite floats."""

    numeric = nullable_float(value)
    return float(default) if numeric is None else numeric


def rate(numerator: int, denominator: int) -> float:
    """Return a rounded rate with a zero-denominator fallback."""

    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def file_sha256(path: Path) -> str | None:
    """Hash a source file when it exists."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist deterministic JSON for reviewable result diffs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def duration(started_s: float, now_s: float | None) -> float:
    """Return rounded wall-clock duration."""

    now = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, now - started_s), 6)


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict string required by the task."""

    if artifact.get("distributional_ebm_exact_row_sidecar_v1_ready") is True:
        return (
            "complete: distributional EBM exact-row sidecar materialized as "
            "offline diagnostics; no deployed verifier claim"
        )
    reasons = artifact.get("blocked_reasons")
    reason_text = ",".join(string_list(reasons)) or "missing_or_insufficient_exact_rows"
    return f"blocked_distributional_ebm_sidecar_precondition:{reason_text}"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3182 artifact contract before writing."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing required fields: {missing}")
    verdict = str(artifact.get("honest_verdict") or "")
    _require(
        verdict.startswith(SUCCESS_PREFIXES) or verdict.startswith(BLOCKED_PREFIXES),
        "honest_verdict must be complete/success/passed/shipped or blocked",
    )
    _require(
        artifact.get("deployed_verifier_claim_allowed") is False,
        "deployed verifier claims are forbidden",
    )
    substrate = artifact.get("inference_substrate")
    _require(isinstance(substrate, Mapping), "inference_substrate must be a dict")
    _require(substrate.get("new_live_model_calls") == 0, "new live model calls are forbidden")
    _require(substrate.get("executes_models") is False, "model execution is forbidden")
    method = artifact.get("sidecar_method")
    _require(isinstance(method, Mapping), "sidecar_method must be a dict")
    _require(method.get("training_performed") is False, "training is forbidden")
    auc = artifact.get("false_accept_separation_auc")
    _require(auc is None or metric_is_unit_interval(auc), "false_accept_separation_auc")
    calibration = artifact.get("uncertainty_calibration")
    _require(isinstance(calibration, Mapping), "uncertainty_calibration must be a dict")
    ece = calibration.get("expected_calibration_error")
    _require(ece is None or metric_is_unit_interval(ece), "expected_calibration_error")
    policy = artifact.get("abstention_policy")
    _require(isinstance(policy, Mapping), "abstention_policy must be a dict")
    _require(policy.get("deployed_policy") is False, "abstention policy cannot be deployed")
    ready = artifact.get("distributional_ebm_exact_row_sidecar_v1_ready") is True
    if ready:
        row_scores = artifact.get("row_scores")
        _require(isinstance(row_scores, list) and bool(row_scores), "row_scores required")
        _require(int(as_float(policy.get("coverage_denominator"), 0.0)) > 0, "coverage_denominator")
        _require(bool(artifact.get("source_reference_ids")), "source_reference_ids required")


def metric_is_unit_interval(value: Any) -> bool:
    """Return true for finite metrics in [0, 1]."""

    numeric = nullable_float(value)
    return numeric is not None and 0.0 <= numeric <= 1.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:  # pragma: no cover
    """CLI entrypoint for writing the requested result artifact."""

    output = write_artifact()
    artifact = read_json_object(output)
    print(
        json.dumps(
            {
                "artifact": output.as_posix(),
                "ready": artifact["distributional_ebm_exact_row_sidecar_v1_ready"],
                "exact_labeled_row_count": artifact["exact_labeled_row_count"],
                "known_false_accept_rows_scored": artifact["known_false_accept_rows_scored"],
                "false_accept_separation_auc": artifact["false_accept_separation_auc"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()

"""Select typed source-support actions with isolated tuning and certification.

The corpus is reused RAGTruth evidence. Exact bounds in this module test the
registered protocol assumptions, but they cannot create a fresh deployment
guarantee from previously exposed public data.

Spec refs: REQ-AUTO-7436 and SCENARIO-AUTO-7436-01 through -06.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7382_v648_decision_protocol import _clopper_pearson_upper
from carnot.experiment_7426_v651_static_decisions import (
    _apply_affine,
    _predict_state,
    feature_matrix,
    fit_ensemble_calibration,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    sidecar_reference,
    validate_current_work_receipt,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260919"
MILESTONE = "2026.09.652"
EXPERIMENT_ID = "exp7436-v652-selection-protocol"
SCHEMA = "carnot.exp7436.v652.selection_protocol.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7436_v652_selection_protocol.json")
RAW_DIR = Path("results/raw/experiment_7436_v652_selection_protocol")
PROTOCOL_PATH = RAW_DIR / "protocol.json"
ROW_PATH = RAW_DIR / "selection_rows.jsonl"
MODULE_PATH = Path("python/carnot/experiment_7436_v652_selection_protocol.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7436_v652_selection_protocol.py")
TEST_PATH = Path("tests/python/test_experiment_7436_v652_selection_protocol.py")
SPEC_PATH = REPO_ROOT / "openspec/capabilities/autoresearch/spec.md"
CORPUS_DIR = Path("results/raw/experiment_7423_v651_annotated_protocol")
CORPUS_MANIFEST_PATH = CORPUS_DIR / "corpus_manifest.json"
UPSTREAM_PATHS = {
    "annotated_protocol": Path("results/experiment_7423_v651_annotated_protocol.json"),
    "static_decisions": Path("results/experiment_7426_v651_static_decisions.json"),
    "decision_audit": Path("results/experiment_7428_v651_decision_audit.json"),
}
EXPECTED_UPSTREAM_HASHES = {
    "annotated_protocol": "sha256:4d91bd186aa7e0fd320694cc0fbc066a2c84ed85c823beccba47c357f903ba8e",
    "static_decisions": "sha256:82405ac1229cd49a3cb4bc700b4cd978d423c166b51da23f4e4af6b1e8ec7c36",
    "decision_audit": "sha256:f909711ee2fe4198197372fb37b220781c5aee41d81a97b00e407501f0d86081",
}
EXPECTED_CORPUS_HASH = "sha256:a065b2a64f2926c5bade1fd3495b1cbff23bd3b3c967bd104b973a6b118e30c0"

TRAINING_SEEDS = (65_101, 65_102, 65_103, 65_104, 65_105)
DEPLOYED_HEADS = ("gibbs_6_4_1", "sparse_spline_49", "raw_l2_logistic")
EQUIVALENCE_CONTROL = "dense_spline_logistic"
ALL_HEADS = (*DEPLOYED_HEADS, EQUIVALENCE_CONTROL)
ALLOWED_PARTITIONS = ("fit", "probability_calibration", "policy_calibration")
OLD_ACCEPT_THRESHOLDS = (0.95, 0.975, 0.99)
OLD_REJECT_THRESHOLDS = (0.01, 0.05, 0.10)
NO_ACCEPT_SENTINEL = 1.1
NO_REJECT_SENTINEL = -0.1
FAMILYWISE_DELTA = 0.05
FAMILYWISE_CHECK_COUNT = 9
ALPHA_PER_CHECK = FAMILYWISE_DELTA / FAMILYWISE_CHECK_COUNT
OLD_ALPHA_PER_CHECK = FAMILYWISE_DELTA / 90
ACCEPT_RISK_BUDGET = 0.05
REJECT_RISK_BUDGET = 0.10
COVERAGE_FLOOR = 0.25
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 6_520_436
INFERENCE_SUBSTRATE = "frozen_numeric_head_aggregation_and_exact_binomial_reduction"
INFERENCE_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"
SELECTION_DIAGNOSES = {
    "disabled_not_applicable",
    "empty_selection",
    "insufficient_certification_support",
    "excessive_observed_risk",
    "certified",
}

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7382_v648_decision_protocol.py"),
    Path("python/carnot/experiment_7423_v651_annotated_protocol.py"),
    Path("python/carnot/experiment_7426_v651_static_decisions.py"),
    Path("python/carnot/experiment_7428_v651_decision_audit.py"),
    Path("openspec/capabilities/autoresearch/spec.md"),
    CORPUS_MANIFEST_PATH,
    *UPSTREAM_PATHS.values(),
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "cold_artifact_replay",
    "independent_row_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

REQUIRED_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_details",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "phase_spans",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "verifier_is_oracle",
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "validation_receipts",
    "field_principles",
    "promotion_score",
    "selection_protocol_ready_score",
    "selection_diagnosis_rows",
    "protocol_manifest",
    "evaluation_reuse_disclosure",
    "deployment_certificate_valid",
    "certificate_scope",
    "small_ebm_training",
)


def utc_now() -> str:  # pragma: no cover - measured wall-clock boundary.
    """Return one aware UTC boundary for the current process."""

    return datetime.now(UTC).isoformat()


def progress(
    started: float, phase: str, event: str, **details: Any
) -> None:  # pragma: no cover - console boundary.
    """Emit a flushed phase boundary or a truthful pending heartbeat."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7436] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _load_object(path: Path) -> JsonDict:
    """Load one external JSON object without changing absence into an empty value."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def precondition_row(
    check: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    operator: str = "==",
    passed: bool | None = None,
) -> JsonDict:
    """Keep the exact prerequisite operand, including missing, None, and zero."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected if passed is None else bool(passed),
    }


def _stable_rank(group_id: str) -> str:
    """Rank a source identity without labels or corpus outcome statistics."""

    return hashlib.sha256(f"carnot-v652-selection-half:{group_id}".encode()).hexdigest()


def label_blind_representatives(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Choose the lowest opaque row identity once per group before labels join."""

    forbidden = {"label", "primary_label", "sensitivity_label", "annotations"}
    seen_rows: set[str] = set()
    selected: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if forbidden.intersection(row):
            raise ValueError("hidden label present in predictor selection")
        row_key = str(row.get("row_key") or "")
        group_id = str(row.get("group_id") or "")
        if not row_key or not group_id:
            raise ValueError("representatives require row and group identity")
        if row_key in seen_rows:
            raise ValueError(f"duplicate row identity:{row_key}")
        seen_rows.add(row_key)
        if group_id not in selected or row_key < str(selected[group_id]["row_key"]):
            selected[group_id] = row
    return [deepcopy(dict(selected[group])) for group in sorted(selected)]


def split_probability_groups(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[JsonDict]]:
    """Split source groups into exact deterministic halves before labels are visible."""

    representatives = label_blind_representatives(rows)
    ranked = sorted(
        (str(row["group_id"]) for row in representatives),
        key=lambda group: (_stable_rank(group), group),
    )
    midpoint = len(ranked) // 2
    calibration = set(ranked[:midpoint])
    tuning = set(ranked[midpoint:])
    return {
        "probability_calibration": [
            deepcopy(dict(row)) for row in rows if str(row.get("group_id")) in calibration
        ],
        "policy_tuning": [
            deepcopy(dict(row)) for row in rows if str(row.get("group_id")) in tuning
        ],
    }


def validate_role_groups(roles: Mapping[str, Sequence[str]]) -> JsonDict:
    """Reject duplicate membership inside one role or overlap between any roles."""

    normalized: dict[str, list[str]] = {}
    for role, groups in roles.items():
        values = [str(group) for group in groups]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate group in role:{role}")
        normalized[str(role)] = values
    names = sorted(normalized)
    overlaps: list[JsonDict] = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            shared = sorted(set(normalized[left]).intersection(normalized[right]))
            if shared:
                overlaps.append({"left": left, "right": right, "group_ids": shared})
    if overlaps:
        raise ValueError(f"role overlap:{overlaps[0]['left']}:{overlaps[0]['right']}")
    return {
        "passed": True,
        "role_count": len(normalized),
        "group_count": sum(len(groups) for groups in normalized.values()),
        "overlaps": [],
    }


def average_seed_probabilities(seed_scores: Any) -> np.ndarray:
    """Average exactly five finite seed rows before later scalar fitting."""

    values = np.asarray(seed_scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(TRAINING_SEEDS):
        raise ValueError("five seed probability rows are required")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("seed probabilities must be finite and bounded")
    return np.mean(values, axis=0)


def dense_equivalence_errors(
    sparse: Mapping[str, Sequence[float]],
    dense: Mapping[str, Sequence[float]],
    *,
    tolerance: float = 1e-10,
) -> list[str]:
    """Keep the dense spline as a parity control, never another deployed policy."""

    errors: list[str] = []
    for role in sorted(set(sparse).union(dense)):
        left = np.asarray(sparse.get(role, []), dtype=np.float64)
        right = np.asarray(dense.get(role, []), dtype=np.float64)
        if (
            left.shape != right.shape
            or not np.all(np.isfinite(left))
            or not np.all(np.isfinite(right))
        ):
            errors.append(f"dense_sparse_shape_invalid:{role}")
        elif left.size and float(np.max(np.abs(left - right))) > tolerance:
            errors.append(f"dense_sparse_probability_mismatch:{role}")
    return errors


def score_quantiles(scores: Sequence[float]) -> dict[str, float]:
    """Return the eleven registered empirical score deciles."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("score quantiles require non-empty finite values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("score quantiles require bounded probabilities")
    return {f"q{index * 10:02d}": float(np.quantile(values, index / 10.0)) for index in range(11)}


def candidate_thresholds(scores: Sequence[float]) -> dict[str, list[float]]:
    """Build tuning deciles and one explicit no-action sentinel per action."""

    quantiles = score_quantiles(scores)
    deciles = list(dict.fromkeys(quantiles.values()))
    return {
        "accept": [*deciles, NO_ACCEPT_SENTINEL],
        "reject": [NO_REJECT_SENTINEL, *deciles],
    }


def _actions(
    probabilities: Sequence[float],
    *,
    accept_threshold: float,
    reject_threshold: float,
    accept_enabled: bool,
    reject_enabled: bool,
) -> list[str]:
    """Apply one frozen pair without allowing its two actions to overlap."""

    if reject_threshold >= accept_threshold:
        raise ValueError("reject threshold must be lower than accept threshold")
    actions: list[str] = []
    for probability in probabilities:
        value = float(probability)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("policy probabilities must be finite and bounded")
        if accept_enabled and value >= accept_threshold:
            actions.append("accept")
        elif reject_enabled and value <= reject_threshold:
            actions.append("reject")
        else:
            actions.append("escalate")
    return actions


def policy_utility(labels: Sequence[int], actions: Sequence[str]) -> JsonDict:
    """Reduce the frozen asymmetric action utility without tuning its weights."""

    if not labels or len(labels) != len(actions):
        raise ValueError("labels and actions must have the same non-empty length")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("labels must be binary")
    if any(action not in {"accept", "reject", "escalate"} for action in actions):
        raise ValueError("actions must be typed")
    correct = sum(
        (label == 1 and action == "accept") or (label == 0 and action == "reject")
        for label, action in zip(labels, actions, strict=True)
    )
    harmful_accept = sum(
        label == 0 and action == "accept" for label, action in zip(labels, actions, strict=True)
    )
    harmful_reject = sum(
        label == 1 and action == "reject" for label, action in zip(labels, actions, strict=True)
    )
    escalations = actions.count("escalate")
    total = correct - 20 * harmful_accept - 10 * harmful_reject
    return {
        "correct_actions": correct,
        "harmful_accepts": harmful_accept,
        "harmful_rejects": harmful_reject,
        "escalations": escalations,
        "non_escalation_coverage": 1.0 - escalations / len(actions),
        "total_utility": total,
        "mean_utility": total / len(actions),
    }


def select_tuned_policy(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, list[JsonDict]]:
    """Select exactly one pair on tuning labels, with an immutable tie rule."""

    if not rows:
        raise ValueError("policy tuning rows are required")
    probabilities = [float(row["probability"]) for row in rows]
    labels = [int(row["label"]) for row in rows]
    thresholds = candidate_thresholds(probabilities)
    candidates: list[JsonDict] = []
    for accept_threshold in thresholds["accept"]:
        for reject_threshold in thresholds["reject"]:
            if reject_threshold >= accept_threshold:
                continue
            accept_enabled = accept_threshold != NO_ACCEPT_SENTINEL
            reject_enabled = reject_threshold != NO_REJECT_SENTINEL
            actions = _actions(
                probabilities,
                accept_threshold=accept_threshold,
                reject_threshold=reject_threshold,
                accept_enabled=accept_enabled,
                reject_enabled=reject_enabled,
            )
            utility = policy_utility(labels, actions)
            candidates.append(
                {
                    "candidate_index": len(candidates),
                    "accept_threshold": accept_threshold,
                    "reject_threshold": reject_threshold,
                    "accept_enabled": accept_enabled,
                    "reject_enabled": reject_enabled,
                    **utility,
                }
            )
    winner = max(
        candidates,
        key=lambda row: (
            float(row["total_utility"]),
            float(row["non_escalation_coverage"]),
            -int(row["candidate_index"]),
        ),
    )
    selected = {
        **deepcopy(winner),
        "selection_rule": "utility_then_coverage_then_registered_order",
        "utility_weights": {
            "correct_action": 1,
            "harmful_accept": -20,
            "harmful_reject": -10,
            "escalation": 0,
        },
        "candidate_count": len(candidates),
        "tuning_groups": len(rows),
        "score_quantiles": score_quantiles(probabilities),
    }
    for row in candidates:
        row["selected"] = row["candidate_index"] == selected["candidate_index"]
    return selected, candidates


def zero_error_sample_requirement(risk_budget: float, delta: float) -> int:
    """Compute the exact zero-error group count needed for one risk budget."""

    if not 0.0 < risk_budget < 1.0:
        raise ValueError("risk budget must be between zero and one")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be between zero and one")
    return math.ceil(math.log(delta) / math.log1p(-risk_budget))


def exact_action_check(
    harmful: Sequence[int],
    *,
    risk_budget: float,
    enabled: bool,
    alpha: float = ALPHA_PER_CHECK,
) -> JsonDict:
    """Certify one enabled action and preserve why a check cannot pass."""

    if any(value not in {0, 1} for value in harmful):
        raise ValueError("harmful outcomes must be binary")
    requirement = zero_error_sample_requirement(risk_budget, alpha)
    if not enabled:
        return {
            "applicable": False,
            "enabled_during_tuning": False,
            "selected_groups": 0,
            "harmful_outcomes": 0,
            "empirical_risk": None,
            "upper_risk_bound": None,
            "risk_budget": risk_budget,
            "alpha_allocated": alpha,
            "alpha_spent": 0.0,
            "zero_error_sample_requirement": requirement,
            "passed": True,
            "diagnosis": "disabled_not_applicable",
        }
    selected = len(harmful)
    failures = sum(harmful)
    if selected == 0:
        return {
            "applicable": True,
            "enabled_during_tuning": True,
            "selected_groups": 0,
            "harmful_outcomes": 0,
            "empirical_risk": None,
            "upper_risk_bound": None,
            "risk_budget": risk_budget,
            "alpha_allocated": alpha,
            "alpha_spent": alpha,
            "zero_error_sample_requirement": requirement,
            "passed": False,
            "diagnosis": "empty_selection",
        }
    upper = _clopper_pearson_upper(failures, selected, alpha)
    passed = upper <= risk_budget
    diagnosis = (
        "certified"
        if passed
        else "insufficient_certification_support"
        if failures == 0
        else "excessive_observed_risk"
    )
    return {
        "applicable": True,
        "enabled_during_tuning": True,
        "selected_groups": selected,
        "harmful_outcomes": failures,
        "empirical_risk": failures / selected,
        "upper_risk_bound": upper,
        "risk_budget": risk_budget,
        "alpha_allocated": alpha,
        "alpha_spent": alpha,
        "zero_error_sample_requirement": requirement,
        "passed": passed,
        "diagnosis": diagnosis,
    }


def _clopper_pearson_lower(successes: int, trials: int, alpha: float) -> float:
    """Derive the exact one-sided lower limit by binomial symmetry."""

    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("coverage counts must be ordered and non-empty")
    if successes == 0:
        return 0.0
    return 1.0 - _clopper_pearson_upper(trials - successes, trials, alpha)


def exact_coverage_check(actions: Sequence[str], *, alpha: float = ALPHA_PER_CHECK) -> JsonDict:
    """Certify non-escalation coverage on all independent certification groups."""

    if not actions or any(action not in {"accept", "reject", "escalate"} for action in actions):
        raise ValueError("coverage requires non-empty typed actions")
    selected = sum(action != "escalate" for action in actions)
    lower = _clopper_pearson_lower(selected, len(actions), alpha)
    return {
        "applicable": True,
        "selected_groups": selected,
        "total_groups": len(actions),
        "empirical_coverage": selected / len(actions),
        "lower_coverage_bound": lower,
        "coverage_floor": COVERAGE_FLOOR,
        "alpha_allocated": alpha,
        "alpha_spent": alpha,
        "passed": lower >= COVERAGE_FLOOR,
        "diagnosis": "certified"
        if lower >= COVERAGE_FLOOR
        else "insufficient_certification_support",
    }


def certify_policy(rows: Sequence[Mapping[str, Any]], tuned: Mapping[str, Any]) -> JsonDict:
    """Apply one frozen pair to certification data and disable it on any failed check."""

    probabilities = [float(row["probability"]) for row in rows]
    labels = [int(row["label"]) for row in rows]
    actions = _actions(
        probabilities,
        accept_threshold=float(tuned["accept_threshold"]),
        reject_threshold=float(tuned["reject_threshold"]),
        accept_enabled=tuned.get("accept_enabled") is True,
        reject_enabled=tuned.get("reject_enabled") is True,
    )
    accept_harm = [
        int(label == 0) for label, action in zip(labels, actions, strict=True) if action == "accept"
    ]
    reject_harm = [
        int(label == 1) for label, action in zip(labels, actions, strict=True) if action == "reject"
    ]
    accept = exact_action_check(
        accept_harm,
        risk_budget=ACCEPT_RISK_BUDGET,
        enabled=tuned.get("accept_enabled") is True,
    )
    reject = exact_action_check(
        reject_harm,
        risk_budget=REJECT_RISK_BUDGET,
        enabled=tuned.get("reject_enabled") is True,
    )
    coverage = exact_coverage_check(actions)
    checks = [accept, reject, coverage]
    all_passed = all(check["passed"] is True for check in checks)
    return {
        "frozen_candidate_index": int(tuned["candidate_index"]),
        "accept_threshold": float(tuned["accept_threshold"]),
        "reject_threshold": float(tuned["reject_threshold"]),
        "tuned_accept_enabled": tuned.get("accept_enabled") is True,
        "tuned_reject_enabled": tuned.get("reject_enabled") is True,
        "accept_check": accept,
        "reject_check": reject,
        "coverage_check": coverage,
        "checks": checks,
        "all_checks_passed": all_passed,
        "deployed_policy": "frozen_typed_policy" if all_passed else "all_escalate",
        "deployed_accept_enabled": all_passed and tuned.get("accept_enabled") is True,
        "deployed_reject_enabled": all_passed and tuned.get("reject_enabled") is True,
        "thresholds_reselected": False,
        "certification_groups": len(rows),
        "score_quantiles": score_quantiles(probabilities),
    }


def diagnose_old_thresholds(head: str, rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Diagnose each old action threshold without blaming correction for emptiness."""

    probabilities = [float(row["probability"]) for row in rows]
    labels = [int(row["label"]) for row in rows]
    quantiles = score_quantiles(probabilities)
    output: list[JsonDict] = []
    for action, thresholds, risk_budget in (
        ("accept", OLD_ACCEPT_THRESHOLDS, ACCEPT_RISK_BUDGET),
        ("reject", OLD_REJECT_THRESHOLDS, REJECT_RISK_BUDGET),
    ):
        for threshold in thresholds:
            selected = [
                int(label == (0 if action == "accept" else 1))
                for label, probability in zip(labels, probabilities, strict=True)
                if (probability >= threshold if action == "accept" else probability <= threshold)
            ]
            check = exact_action_check(
                selected,
                risk_budget=risk_budget,
                enabled=True,
                alpha=OLD_ALPHA_PER_CHECK,
            )
            output.append(
                {
                    "head": head,
                    "action": action,
                    "old_threshold": threshold,
                    "score_quantiles": quantiles,
                    "multiplicity_test_count": 90,
                    **check,
                }
            )
    return output


def _manifest_hash(value: Mapping[str, Any]) -> str:
    """Hash a protocol seal without its self-referential hash field."""

    payload = deepcopy(dict(value))
    payload["manifest_hash"] = ""
    return canonical_hash(payload)


def build_protocol_manifest(roles: Mapping[str, Sequence[str]]) -> JsonDict:
    """Build the immutable role and finite-sample plan before labels are used."""

    disjointness = validate_role_groups(roles)
    role_rows = {
        str(role): {
            "group_ids": sorted(str(group) for group in groups),
            "group_count": len(groups),
            "group_identity_sha256": canonical_hash(sorted(str(group) for group in groups)),
        }
        for role, groups in sorted(roles.items())
    }
    manifest: JsonDict = {
        "schema": "carnot.exp7436.selection_protocol_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "sealed_before_trial": True,
        "allowed_design_partitions": list(ALLOWED_PARTITIONS),
        "forbidden_design_partitions": ["final_test", "prospective_stream"],
        "roles": role_rows,
        "role_disjointness": disjointness,
        "group_sampling_assumptions": (
            "One label-blind representative per source group is iid across groups; "
            "siblings inside a group are not independent."
        ),
        "training_seeds": list(TRAINING_SEEDS),
        "deployed_heads": list(DEPLOYED_HEADS),
        "equivalence_control": EQUIVALENCE_CONTROL,
        "candidate_rule": "tuning score deciles plus accept and reject no-action sentinels",
        "tie_rule": "utility_then_coverage_then_registered_order",
        "old_accept_thresholds": list(OLD_ACCEPT_THRESHOLDS),
        "old_reject_thresholds": list(OLD_REJECT_THRESHOLDS),
        "no_action_sentinels": {
            "accept": NO_ACCEPT_SENTINEL,
            "reject": NO_REJECT_SENTINEL,
        },
        "utility": {
            "correct_action": 1,
            "harmful_accept": -20,
            "harmful_reject": -10,
            "escalation": 0,
        },
        "familywise_delta": FAMILYWISE_DELTA,
        "familywise_check_count": FAMILYWISE_CHECK_COUNT,
        "alpha_per_check": ALPHA_PER_CHECK,
        "risk_budgets": {
            "accept": ACCEPT_RISK_BUDGET,
            "reject": REJECT_RISK_BUDGET,
            "coverage_floor": COVERAGE_FLOOR,
        },
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "joint_certificate_paper_role": (
            "motivates role separation only; no claim that its bound beats binary exact bounds"
        ),
        "evaluation_reuse": "previously exposed public RAGTruth corpus",
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = _manifest_hash(manifest)
    return manifest


def validate_protocol_manifest(value: Mapping[str, Any]) -> list[str]:
    """Independently check role membership, constants, and the protocol seal hash."""

    errors: list[str] = []
    expected = {
        "schema": "carnot.exp7436.selection_protocol_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "sealed_before_trial": True,
        "allowed_design_partitions": list(ALLOWED_PARTITIONS),
        "forbidden_design_partitions": ["final_test", "prospective_stream"],
        "training_seeds": list(TRAINING_SEEDS),
        "deployed_heads": list(DEPLOYED_HEADS),
        "equivalence_control": EQUIVALENCE_CONTROL,
        "familywise_delta": FAMILYWISE_DELTA,
        "familywise_check_count": FAMILYWISE_CHECK_COUNT,
        "alpha_per_check": ALPHA_PER_CHECK,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            errors.append(f"protocol_field_mismatch:{field}")
    roles = value.get("roles")
    if not isinstance(roles, Mapping):
        errors.append("roles_invalid")
    else:
        groups: dict[str, list[str]] = {}
        for role, row in roles.items():
            if not isinstance(row, Mapping) or not isinstance(row.get("group_ids"), list):
                errors.append(f"role_invalid:{role}")
                continue
            group_ids = [str(group) for group in row["group_ids"]]
            groups[str(role)] = group_ids
            if row.get("group_count") != len(group_ids):
                errors.append(f"role_count_mismatch:{role}")
            if row.get("group_identity_sha256") != canonical_hash(sorted(group_ids)):
                errors.append(f"role_hash_mismatch:{role}")
        try:
            validate_role_groups(groups)
        except ValueError:
            errors.append("role_overlap")
        if set(groups) != {"fit", "probability_calibration", "policy_tuning", "certification"}:
            errors.append("role_names_mismatch")
    if value.get("manifest_hash") != _manifest_hash(value):
        errors.append("protocol_manifest_hash_mismatch")
    return list(dict.fromkeys(errors))


def seal_protocol(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write a protocol once and reject a rerun that would change sealed bytes."""

    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"protocol seal conflict:{path}")
        return
    errors = validate_protocol_manifest(manifest)
    if errors:
        raise ValueError(f"protocol manifest invalid:{errors}")
    atomic_json(path, manifest)


def run_mutation_controls() -> list[JsonDict]:
    """Exercise the four required leakage and empty-selection mutations."""

    rows = [
        {"row_key": "a", "group_id": "g-a", "partition": "probability_calibration"},
        {"row_key": "b", "group_id": "g-b", "partition": "probability_calibration"},
    ]
    controls: list[JsonDict] = []
    cases = (
        (
            "duplicated_groups",
            lambda: validate_role_groups({"fit": ["g-a", "g-a"]}),
            "duplicate group",
        ),
        (
            "tune_cert_overlap",
            lambda: validate_role_groups({"policy_tuning": ["g-a"], "certification": ["g-a"]}),
            "role overlap",
        ),
        (
            "hidden_labels",
            lambda: label_blind_representatives([{**rows[0], "label": 1}]),
            "hidden label",
        ),
    )
    for name, operation, expected in cases:
        observed = None
        try:
            operation()
        except ValueError as error:
            observed = str(error)
        controls.append(
            {
                "mutation": name,
                "expected": expected,
                "observed": observed,
                "passed": isinstance(observed, str) and expected in observed,
            }
        )
    empty = exact_action_check([], risk_budget=ACCEPT_RISK_BUDGET, enabled=True)
    controls.append(
        {
            "mutation": "empty_selection",
            "expected": "empty_selection_with_undefined_risk",
            "observed": (
                "empty_selection_with_undefined_risk"
                if empty["diagnosis"] == "empty_selection" and empty["upper_risk_bound"] is None
                else "incorrect_empty_semantics"
            ),
            "passed": empty["diagnosis"] == "empty_selection" and empty["upper_risk_bound"] is None,
        }
    )
    return controls


def _jsonl_rows(path: Path) -> list[JsonDict]:
    """Read one authenticated JSONL shard after its byte hash has passed."""

    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"jsonl_invalid:{path.name}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"jsonl_row_not_object:{path.name}:{line_number}")
            rows.append(value)
    return rows


def load_allowed_protocol_rows(
    root: Path,
) -> tuple[dict[str, list[JsonDict]], dict[str, list[JsonDict]], list[JsonDict]]:
    """Open only fit and calibration shards; never open final-test or stream rows."""

    manifest_path = root / CORPUS_MANIFEST_PATH
    manifest = _load_object(manifest_path)
    if not manifest or sha256_file(manifest_path) != EXPECTED_CORPUS_HASH:
        raise ValueError("corpus_manifest_identity_mismatch")
    predictors = {partition: [] for partition in ALLOWED_PARTITIONS}
    evaluators = {partition: [] for partition in ALLOWED_PARTITIONS}
    receipts: list[JsonDict] = []
    for row in manifest.get("shards") or []:
        if not isinstance(row, Mapping):
            raise ValueError("corpus_shard_manifest_invalid")
        name = str(row.get("path") or "")
        partition = next(
            (role for role in ALLOWED_PARTITIONS if f"-{role}-" in name),
            None,
        )
        if partition is None:
            continue
        kind = str(row.get("kind") or "")
        if kind not in {"predictor", "evaluator"}:
            continue
        path = root / CORPUS_DIR / name
        observed = sha256_file(path) if path.is_file() else None
        if observed != row.get("sha256"):
            raise ValueError(f"corpus_shard_hash_mismatch:{name}")
        loaded = _jsonl_rows(path)
        if len(loaded) != row.get("rows"):
            raise ValueError(f"corpus_shard_count_mismatch:{name}")
        target = predictors if kind == "predictor" else evaluators
        target[partition].extend(loaded)
        receipts.append(
            {
                "path": (CORPUS_DIR / name).as_posix(),
                "kind": kind,
                "partition": partition,
                "rows": len(loaded),
                "sha256": observed,
            }
        )
    for partition in ALLOWED_PARTITIONS:
        if not predictors[partition] or not evaluators[partition]:
            if partition == "fit" and predictors[partition]:
                continue
            raise ValueError(f"allowed_partition_incomplete:{partition}")
    return predictors, evaluators, receipts


def _attach_labels(
    predictors: Sequence[Mapping[str, Any]], evaluators: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Join labels only after label-blind representatives have been fixed."""

    by_key = {str(row.get("row_key") or ""): row for row in evaluators}
    if len(by_key) != len(evaluators):
        raise ValueError("evaluator row identities must be unique")
    output: list[JsonDict] = []
    for predictor in predictors:
        row_key = str(predictor.get("row_key") or "")
        evaluator = by_key.get(row_key)
        if evaluator is None:
            raise ValueError(f"evaluator row missing:{row_key}")
        label = evaluator.get("primary_label")
        if label not in {0, 1}:
            raise ValueError(f"evaluator label invalid:{row_key}")
        if evaluator.get("group_id") != predictor.get("group_id") or evaluator.get(
            "partition"
        ) != predictor.get("partition"):
            raise ValueError(f"predictor evaluator identity mismatch:{row_key}")
        output.append({**deepcopy(dict(predictor)), "label": int(label)})
    return output


def _load_frozen_states(
    root: Path, static_artifact: Mapping[str, Any]
) -> tuple[dict[str, list[JsonDict]], list[JsonDict]]:
    """Load only full-source states for the three heads and dense control."""

    states = {head: [] for head in ALL_HEADS}
    receipts: list[JsonDict] = []
    for row in static_artifact.get("checkpoint_manifest") or []:
        if (
            not isinstance(row, Mapping)
            or row.get("condition") != "full_source"
            or row.get("arm") not in ALL_HEADS
        ):
            continue
        path = root / str(row["path"])
        observed = sha256_file(path) if path.is_file() else None
        if observed != row.get("sha256"):
            raise ValueError(f"checkpoint_hash_mismatch:{row.get('path')}")
        value = _load_object(path)
        if value.get("seed") not in TRAINING_SEEDS or value.get("arm") != row.get("arm"):
            raise ValueError(f"checkpoint_identity_mismatch:{row.get('path')}")
        states[str(row["arm"])].append(
            {
                "arm": value["arm"],
                "seed": value["seed"],
                "checkpoint": value["checkpoint"],
            }
        )
        receipts.append(
            {
                "path": str(row["path"]),
                "sha256": observed,
                "arm": row["arm"],
                "seed": row["seed"],
                "source_receipt_class": "archived_small_ebm_training",
            }
        )
    for head, head_states in states.items():
        head_states.sort(key=lambda item: int(item["seed"]))
        if [row["seed"] for row in head_states] != list(TRAINING_SEEDS):
            raise ValueError(f"five_frozen_seed_states_required:{head}")
    return states, receipts


def _raw_seed_scores(
    states: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    """Score one common feature matrix with each frozen numeric seed state."""

    features = feature_matrix(rows)
    return np.asarray([_predict_state(state, features) for state in states], dtype=np.float64)


def _bootstrap_utility(
    rows: Sequence[Mapping[str, Any]], tuned: Mapping[str, Any], *, seed: int
) -> JsonDict:
    """Resample independent source groups for a descriptive utility interval."""

    probabilities = [float(row["probability"]) for row in rows]
    labels = [int(row["label"]) for row in rows]
    actions = _actions(
        probabilities,
        accept_threshold=float(tuned["accept_threshold"]),
        reject_threshold=float(tuned["reject_threshold"]),
        accept_enabled=tuned.get("accept_enabled") is True,
        reject_enabled=tuned.get("reject_enabled") is True,
    )
    values = np.asarray(
        [
            1.0
            if (label == 1 and action == "accept") or (label == 0 and action == "reject")
            else -20.0
            if label == 0 and action == "accept"
            else -10.0
            if label == 1 and action == "reject"
            else 0.0
            for label, action in zip(labels, actions, strict=True)
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for start in range(0, BOOTSTRAP_DRAWS, 1_000):
        draws = min(1_000, BOOTSTRAP_DRAWS - start)
        indices = rng.integers(0, len(values), size=(draws, len(values)))
        means[start : start + draws] = np.mean(values[indices], axis=1)
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": seed,
        "source_groups": len(values),
        "observed_mean_utility": float(np.mean(values)),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "role": "descriptive_only_not_a_replacement_for_exact_binary_bounds",
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Write one compact, deterministic, hash-bound evidence shard."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "byte_size": path.stat().st_size,
        "rows": len(rows),
    }


def _load_bound_rows(root: Path, manifest: Mapping[str, Any]) -> list[JsonDict]:
    """Reload detailed rows only when their exact bytes and counts match."""

    path = Path(str(manifest.get("path") or ""))
    resolved = path if path.is_absolute() else root / path
    if (
        not resolved.is_file()
        or sha256_file(resolved) != manifest.get("sha256")
        or resolved.stat().st_size != manifest.get("byte_size")
    ):
        raise ValueError("selection_row_shard_bytes_mismatch")
    rows = _jsonl_rows(resolved)
    if len(rows) != manifest.get("rows"):
        raise ValueError("selection_row_shard_count_mismatch")
    return rows


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], JsonDict, dict[str, JsonDict]]:
    """Authenticate every local authority and original upstream flag."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in INPUT_PATHS:
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            precondition_row(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                observed,
            )
        )
        if observed:
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "original_flagged_adversarial": None,
            }
    loaded = {name: _load_object(root / path) for name, path in UPSTREAM_PATHS.items()}
    expected_fields = {
        "annotated_protocol": {
            "experiment_id": "exp7423-v651-annotated-protocol",
            "milestone": "2026.09.651",
            "annotated_protocol_ready_score": 1,
            "flagged_adversarial": False,
        },
        "static_decisions": {
            "experiment_id": "exp7426-v651-static-decisions",
            "milestone": "2026.09.651",
            "decision_capture_complete_score": 1,
            "flagged_adversarial": False,
        },
        "decision_audit": {
            "experiment_id": "exp7428-v651-decision-audit",
            "milestone": "2026.09.651",
            "static_audit_complete_score": 1,
            "flagged_adversarial": False,
        },
    }
    for name, relative in UPSTREAM_PATHS.items():
        artifact = loaded[name]
        observed_hash = sha256_file(root / relative) if artifact else None
        checks.append(
            precondition_row(
                f"exact_hash:{name}",
                str(expected_fields[name]["experiment_id"]),
                relative.as_posix(),
                "sha256",
                EXPECTED_UPSTREAM_HASHES[name],
                observed_hash,
            )
        )
        verdict = artifact.get("verdict_class")
        checks.append(
            precondition_row(
                f"allowed_verdict:{name}",
                str(expected_fields[name]["experiment_id"]),
                relative.as_posix(),
                "verdict_class",
                ["positive", "circular_positive", "null"],
                verdict,
                operator="in",
                passed=verdict in {"positive", "circular_positive", "null"},
            )
        )
        for field, expected in expected_fields[name].items():
            checks.append(
                precondition_row(
                    f"upstream_field:{name}:{field}",
                    str(expected_fields[name]["experiment_id"]),
                    relative.as_posix(),
                    field,
                    expected,
                    artifact.get(field),
                )
            )
        if relative.as_posix() in hashes:
            hashes[relative.as_posix()]["original_flagged_adversarial"] = artifact.get(
                "flagged_adversarial"
            )
            hashes[relative.as_posix()]["original_verdict_class"] = verdict
    corpus_path = root / CORPUS_MANIFEST_PATH
    checks.append(
        precondition_row(
            "exact_hash:corpus_manifest",
            "exp7423-v651-annotated-protocol",
            CORPUS_MANIFEST_PATH.as_posix(),
            "sha256",
            EXPECTED_CORPUS_HASH,
            sha256_file(corpus_path) if corpus_path.is_file() else None,
        )
    )
    spec_text = SPEC_PATH.read_text(encoding="utf-8") if SPEC_PATH.is_file() else ""
    checks.append(
        precondition_row(
            "driving_requirement",
            "openspec/capabilities/autoresearch/spec.md",
            "openspec/capabilities/autoresearch/spec.md",
            "REQ-*",
            "REQ-AUTO-7436",
            "REQ-AUTO-7436" if "REQ-AUTO-7436" in spec_text else None,
        )
    )
    exclusion_path = root / "ops/exclusion_manifest.yaml"
    exclusion = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    checks.append(
        precondition_row(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            "experiment_id: 7436" in exclusion,
        )
    )
    return checks, hashes, loaded


def _field_principles() -> dict[str, str]:
    """Explain required fields while keeping their values machine-readable."""

    principles = {
        field: "This field records measured Exp7436 protocol evidence." for field in REQUIRED_FIELDS
    }
    principles.update(
        {
            "schema": "Use a versioned plain top-level schema with experiment identity and terminal status.",
            "run_date": "Use 20260919 and retain actual UTC plus monotonic boundaries.",
            "preconditions_checked": "Name every observed resource, identity, field, expectation, and value.",
            "MODEL_SPECS": "List current LLMs; this aggregation invokes none.",
            "model_invoked": "Separate current attempted model use from archived model-shaped evidence.",
            "invocation_counts": "Reconcile current attempted, completed, failed, cancelled, and in-flight calls.",
            "inference_substrate": "Describe current aggregation as a truthful plain string.",
            "inference_substrate_class": "Classify current compute as aggregation.",
            "execution_venue": "Use host and keep device detail separate.",
            "duration_s": "Measure current work and separate model, computation, cold-start, and validation time.",
            "phase_spans": "Bind phase times, progress boundaries, and completed units.",
            "random_seed": "Freeze all training, sampling, and resampling seeds.",
            "reproducibility_checksum": "Bind code, protocol, inputs, rows, and exact validation scope.",
            "source_artifact_hashes": "Preserve source identity, original verdict classes, and adversarial flags.",
            "rows": "Keep one completed unit for each deployed policy, including failed feasibility.",
            "sample_size_budget": "Separate planned, attempted, completed, failed, censored, and unstarted units.",
            "acceptance_gate_results": "Keep validity, completion, and scientific feasibility gates distinct.",
            "gate_check_summary": "Name exact blocked operands and preserve missing, None, and zero.",
            "verifier_is_oracle": "Human source-support annotations are fallible, not a deployed oracle.",
            "honest_verdict": "Use complete_ for measured findings and blocked_ for unchanged missing prerequisites.",
            "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
            "flagged_adversarial": "Critical validation findings prevent readiness.",
            "validation_receipts": "Retain actual scoped commands, environments, exits, times, and log hashes.",
            "field_principles": "Explain field intent separately from plain gate scalars.",
            "promotion_score": "Always zero; this milestone cannot authorize rollout or weight updates.",
            "selection_protocol_ready_score": "Require sealed roles, exact checks, label isolation, and validation independent of benefit.",
            "selection_diagnosis_rows": "Separate empty selection from insufficient support and excessive observed risk.",
            "protocol_manifest": "Bind roles, candidate rules, alpha, utility, seeds, and bootstrap design.",
            "evaluation_reuse_disclosure": "Renaming a public corpus cannot make it pristine confirmation evidence.",
            "deployment_certificate_valid": "Always false because historical label exposure prevents a fresh new-user guarantee.",
            "certificate_scope": "Use exploratory_reused_corpus for nominal bounds on reused evidence.",
            "small_ebm_training": "Keep compact scalar calibration separate from current LLM invocation receipts.",
        }
    )
    return principles


def _gate(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Record one explicit gate without hiding its operands in prose."""

    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def _gate_summary(
    gates: Sequence[Mapping[str, Any]], blocked: Mapping[str, Any] | None = None
) -> JsonDict:
    """Keep required protocol validity separate from nominal feasibility."""

    failed = [
        deepcopy(dict(row))
        for row in gates
        if row.get("category") != "scientific_benefit" and row.get("passed") is not True
    ]
    return {
        "required_checks_passed": not failed and blocked is None,
        "failed_required_checks": failed,
        "scientific_benefit_passed": all(
            row.get("passed") is True
            for row in gates
            if row.get("category") == "scientific_benefit"
        ),
        "blocked_upstream": blocked.get("upstream") if blocked else None,
        "blocked_path": blocked.get("path") if blocked else None,
        "blocked_check": blocked.get("check") if blocked else None,
        "blocked_field": blocked.get("field") if blocked else None,
        "blocked_expected": blocked.get("expected") if blocked else None,
        "blocked_observed": blocked.get("observed") if blocked else None,
    }


def _validation_passed(receipts: Sequence[Mapping[str, Any]], *, candidate: bool = False) -> bool:
    """Require the frozen affected set and terminal readers for the final record."""

    required = set(AFFECTED_CHECK_NAMES)
    if not candidate:
        required.update(TERMINAL_CHECK_NAMES)
    passed = {
        str(row.get("name"))
        for row in receipts
        if row.get("required") is True and row.get("passed") is True and row.get("exit_code") == 0
    }
    return required.issubset(passed)


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding clocks, logs, and this hash slot."""

    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = ""
    for field in (
        "started_at_utc",
        "completed_at_utc",
        "duration_s",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "phase_spans",
        "validation_receipts",
        "validation_duration_s",
        "cold_start_duration_s",
        "computation_duration_s",
    ):
        payload.pop(field, None)
    return canonical_hash(payload)


def _receipt(
    root: Path,
    *,
    started_ns: int,
    ended_ns: int,
    phase_spans: Sequence[Mapping[str, Any]],
    small_training: Mapping[str, Any],
) -> JsonDict:
    """Build zero-current-model provenance with typed upstream sidecars."""

    sidecars = [
        sidecar_reference(root / path, root=root, scope="historical_model_receipts")
        for path in UPSTREAM_PATHS.values()
        if (root / path).is_file()
    ]
    receipt = build_current_work_receipt(
        run_id=f"{EXPERIMENT_ID}-{os.getpid()}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={
            "device": "host_cpu",
            "cuda_used": False,
            "external_device": None,
            "current_llm": False,
            "frozen_head_aggregation": True,
        },
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
        sidecar_references=sidecars,
        phase_spans=phase_spans,
        small_ebm_training=small_training,
    )
    receipt["started_monotonic_ns"] = 0
    receipt["ended_monotonic_ns"] = ended_ns - started_ns
    return receipt


def _span(phase: str, phase_started: float, run_started: float, units: int) -> JsonDict:
    """Close one measured phase with a resumable completed-unit count."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint_at_utc": utc_now(),
    }


def _protocol_reference(root: Path, path: Path) -> JsonDict:
    """Bind the immutable protocol bytes and their internal manifest identity."""

    value = _load_object(path)
    try:
        label = path.relative_to(root).as_posix()
    except ValueError:
        label = str(path.resolve())
    return {
        "path": label,
        "sha256": sha256_file(path),
        "manifest_hash": value.get("manifest_hash"),
    }


def build_artifact(
    *,
    root: Path,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    policy_rows: Sequence[Mapping[str, Any]],
    diagnosis_rows: Sequence[Mapping[str, Any]],
    protocol_path: Path,
    row_manifest: Mapping[str, Any],
    mutation_controls: Sequence[Mapping[str, Any]],
    equivalence_errors: Sequence[str],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    started_ns: int,
    ended_ns: int,
    candidate: bool = False,
    fixture: bool = False,
    flagged_adversarial: bool = False,
) -> JsonDict:
    """Build one independently reducible protocol record from measured rows."""

    protocol = _load_object(protocol_path)
    protocol_ok = not validate_protocol_manifest(protocol)
    controls_ok = bool(mutation_controls) and all(
        row.get("passed") is True for row in mutation_controls
    )
    validation_ok = _validation_passed(validation_receipts, candidate=candidate)
    policies_complete = len(policy_rows) == len(DEPLOYED_HEADS) and all(
        row.get("completed") is True and row.get("failed") is False for row in policy_rows
    )
    ready = int(
        bool(preconditions)
        and all(row.get("passed") is True for row in preconditions)
        and protocol_ok
        and controls_ok
        and not equivalence_errors
        and policies_complete
        and validation_ok
        and not flagged_adversarial
    )
    nominal = ready == 1 and any(
        (row.get("certificate") or {}).get("all_checks_passed") is True for row in policy_rows
    )
    verdict_class = "positive" if nominal else "null"
    honest = (
        "complete_positive_nominal_selection_feasibility_on_reused_corpus"
        if nominal
        else "complete_null_selection_protocol_no_nominally_feasible_policy"
    )
    small_training = {
        "performed": True,
        "receipt_class": "small_ebm_training",
        "current_llm_calls": 0,
        "generator_weights_fitted": False,
        "components": "three_two_parameter_scalar_probability_calibrators",
        "parameter_count": 6,
        "maximum_steps": 500,
        "training_seeds_reused_before_calibration": list(TRAINING_SEEDS),
    }
    receipt = _receipt(
        root,
        started_ns=started_ns,
        ended_ns=ended_ns,
        phase_spans=phase_spans,
        small_training=small_training,
    )
    gates = [
        _gate(
            "authenticated_preconditions",
            "validity",
            "all",
            True,
            bool(preconditions) and all(row.get("passed") is True for row in preconditions),
            bool(preconditions) and all(row.get("passed") is True for row in preconditions),
            "Changed upstream evidence cannot silently enter the study.",
        ),
        _gate(
            "sealed_role_protocol",
            "validity",
            "==",
            True,
            protocol_ok,
            protocol_ok,
            "Policy fitting and certification must keep disjoint source groups.",
        ),
        _gate(
            "label_isolation_mutations",
            "safety",
            "all",
            True,
            controls_ok,
            controls_ok,
            "Required mutations must show the isolation checks fail closed.",
        ),
        _gate(
            "dense_sparse_equivalence_control",
            "validity",
            "==",
            [],
            list(equivalence_errors),
            not equivalence_errors,
            "The dense control tests representation only and cannot add a policy.",
        ),
        _gate(
            "affected_and_terminal_validation",
            "validation",
            "==",
            True,
            validation_ok,
            validation_ok,
            "Only the frozen validation scope and terminal readers authorize readiness.",
        ),
        _gate(
            "nominal_policy_feasibility",
            "scientific_benefit",
            "any",
            True,
            nominal,
            nominal,
            "A nominal pass justifies only a new independently sealed corpus.",
        ),
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": honest,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": utc_now(),
        **receipt,
        "model_duration_s": 0.0,
        "computation_duration_s": sum(
            float(span.get("duration_s") or 0.0)
            for span in phase_spans
            if span.get("phase") in {"aggregation", "certification", "bootstrap"}
        ),
        "validation_duration_s": sum(
            float(row.get("duration_s") or 0.0) for row in validation_receipts
        ),
        "cold_start_duration_s": sum(
            float(row.get("duration_s") or 0.0)
            for row in validation_receipts
            if row.get("name") in TERMINAL_CHECK_NAMES
        ),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "random_seed": {
            "frozen_training_seeds": list(TRAINING_SEEDS),
            "probability_group_split_salt": "carnot-v652-selection-half",
            "bootstrap": BOOTSTRAP_SEED,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [deepcopy(dict(row)) for row in policy_rows],
        "selection_row_shard": deepcopy(dict(row_manifest)),
        "sample_size_budget": {
            "planned_units": len(DEPLOYED_HEADS),
            "attempted_units": len(policy_rows),
            "completed_units": sum(row.get("completed") is True for row in policy_rows),
            "failed_units": sum(row.get("failed") is True for row in policy_rows),
            "censored_units": 0,
            "unstarted_units": len(DEPLOYED_HEADS) - len(policy_rows),
            "bootstrap_draws_per_policy": BOOTSTRAP_DRAWS,
            "stop_rule": "one frozen tuning choice and one certification pass per deployed head",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": False,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "flagged_adversarial": bool(flagged_adversarial),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "validation_manifest": {
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "affected_checks": list(AFFECTED_CHECK_NAMES),
            "terminal_checks": list(TERMINAL_CHECK_NAMES),
        },
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "selection_protocol_ready_score": ready,
        "selection_diagnosis_rows": [deepcopy(dict(row)) for row in diagnosis_rows],
        "protocol_manifest": _protocol_reference(root, protocol_path),
        "mutation_control_rows": [deepcopy(dict(row)) for row in mutation_controls],
        "dense_equivalence_errors": list(equivalence_errors),
        "bootstrap_rows": [deepcopy(dict(row)) for row in bootstrap_rows],
        "evaluation_reuse_disclosure": (
            "RAGTruth labels and evaluation roles were previously exposed. Renaming or "
            "resplitting this public corpus does not make it fresh external confirmation."
        ),
        "deployment_certificate_valid": False,
        "certificate_scope": "exploratory_reused_corpus",
        "fresh_new_user_safety_claimed": False,
        "candidate_artifact": candidate,
        "fixture_artifact": fixture,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def independent_reduce(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    """Recompute readiness and nominal feasibility from sealed bytes and raw rows."""

    protocol_ref = value.get("protocol_manifest") or {}
    protocol_path = Path(str(protocol_ref.get("path") or ""))
    resolved_protocol = protocol_path if protocol_path.is_absolute() else root / protocol_path
    protocol = _load_object(resolved_protocol)
    protocol_ok = (
        resolved_protocol.is_file()
        and sha256_file(resolved_protocol) == protocol_ref.get("sha256")
        and protocol.get("manifest_hash") == protocol_ref.get("manifest_hash")
        and not validate_protocol_manifest(protocol)
    )
    raw_rows = _load_bound_rows(root, value.get("selection_row_shard") or {})
    policies = value.get("rows") or []
    raw_by_head = {
        str(row.get("head")): row for row in raw_rows if row.get("row_type") == "policy_summary"
    }
    score_rows: dict[str, dict[str, list[JsonDict]]] = {
        head: {"policy_tuning": [], "certification": []} for head in DEPLOYED_HEADS
    }
    for raw_row in raw_rows:
        head = str(raw_row.get("head") or "")
        role = str(raw_row.get("role") or "")
        if (
            raw_row.get("row_type") == "source_group_score"
            and head in score_rows
            and role in score_rows[head]
        ):
            score_rows[head][role].append(deepcopy(dict(raw_row)))
    recomputed: dict[str, JsonDict] = {}
    for head in DEPLOYED_HEADS:
        tuning = score_rows[head]["policy_tuning"]
        certification = score_rows[head]["certification"]
        if tuning and certification:
            tuned, _candidate_rows = select_tuned_policy(tuning)
            recomputed[head] = {
                "tuned_policy": tuned,
                "certificate": certify_policy(certification, tuned),
                "diagnosis": diagnose_old_thresholds(head, certification),
            }
    manifest_roles = protocol.get("roles") or {}
    raw_roles_match = all(
        {str(row.get("source_group")) for row in score_rows[head][role]}
        == set((manifest_roles.get(role) or {}).get("group_ids") or [])
        for head in DEPLOYED_HEADS
        for role in ("policy_tuning", "certification")
    )
    policies_complete = len(policies) == len(DEPLOYED_HEADS) and all(
        isinstance(row, Mapping)
        and row.get("completed") is True
        and row.get("failed") is False
        and str(row.get("head")) in recomputed
        and canonical_hash(row.get("tuned_policy"))
        == canonical_hash(recomputed[str(row.get("head"))]["tuned_policy"])
        and canonical_hash(row.get("certificate"))
        == canonical_hash(recomputed[str(row.get("head"))]["certificate"])
        and canonical_hash(row.get("tuned_policy"))
        == canonical_hash((raw_by_head.get(str(row.get("head"))) or {}).get("tuned_policy"))
        and canonical_hash(row.get("certificate"))
        == canonical_hash((raw_by_head.get(str(row.get("head"))) or {}).get("certificate"))
        for row in policies
    )
    diagnosis_recomputed = [
        diagnosis
        for head in DEPLOYED_HEADS
        for diagnosis in recomputed.get(head, {}).get("diagnosis", [])
    ]
    diagnosis_match = canonical_hash(value.get("selection_diagnosis_rows") or []) == canonical_hash(
        diagnosis_recomputed
    )
    controls = value.get("mutation_control_rows") or []
    controls_ok = bool(controls) and all(row.get("passed") is True for row in controls)
    validation_ok = _validation_passed(
        value.get("validation_receipts") or [], candidate=value.get("candidate_artifact") is True
    )
    preconditions = value.get("preconditions_checked") or []
    ready = int(
        bool(preconditions)
        and all(row.get("passed") is True for row in preconditions)
        and protocol_ok
        and controls_ok
        and not value.get("dense_equivalence_errors")
        and policies_complete
        and raw_roles_match
        and diagnosis_match
        and validation_ok
        and value.get("flagged_adversarial") is False
    )
    nominal = ready == 1 and any(
        (row.get("certificate") or {}).get("all_checks_passed") is True for row in policies
    )
    return {
        "selection_protocol_ready_score": ready,
        "nominal_policy_feasibility": nominal,
        "deployment_certificate_valid": False,
        "promotion_score": 0,
        "protocol_valid": protocol_ok,
        "raw_row_count": len(raw_rows),
        "raw_roles_match": raw_roles_match,
        "diagnosis_match": diagnosis_match,
    }


def validate_artifact(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> list[str]:
    """Cold-check identity, sealed bytes, reductions, receipts, and safety scope."""

    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in value]
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": ZERO_INVOCATION_COUNTS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "verifier_is_oracle": False,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            errors.append(f"declaration_mismatch:{field}")
    if value.get("deployment_certificate_valid") is not False:
        errors.append("deployment_certificate_valid_mismatch")
    if value.get("certificate_scope") != "exploratory_reused_corpus":
        errors.append("certificate_scope_mismatch")
    if value.get("promotion_score") != 0:
        errors.append("promotion_score_mismatch")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if set(value.get("field_principles") or {}) != set(REQUIRED_FIELDS):
        errors.append("field_principles_mismatch")
    if value.get("verdict_class") == "blocked":
        if not str(value.get("honest_verdict") or "").startswith("blocked_"):
            errors.append("blocked_verdict_prefix_invalid")
    else:
        try:
            reduced = independent_reduce(value, root=root)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"independent_reduction_failed:{error}")
        else:
            for field in (
                "selection_protocol_ready_score",
                "deployment_certificate_valid",
                "promotion_score",
            ):
                if value.get(field) != reduced[field]:
                    errors.append(f"{field}_mismatch")
        if not value.get("fixture_artifact"):
            for row in (value.get("source_artifact_hashes") or {}).values():
                if not isinstance(row, Mapping):
                    errors.append("source_artifact_hash_row_invalid")
                    continue
                path = Path(str(row.get("path") or ""))
                resolved = path if path.is_absolute() else root / path
                if not resolved.is_file() or sha256_file(resolved) != row.get("sha256"):
                    errors.append(f"source_artifact_hash_mismatch:{row.get('path')}")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    errors.extend(validate_current_work_receipt(value, root=root))
    return list(dict.fromkeys(errors))


def build_blocked_artifact(failed: Mapping[str, Any]) -> JsonDict:
    """Publish an external prerequisite block without inventing dependent rows."""

    started = time.monotonic_ns()
    ended = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=f"{EXPERIMENT_ID}-blocked-{os.getpid()}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={
            "device": "host_cpu",
            "cuda_used": False,
            "external_device": None,
            "dependent_work_started": False,
        },
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        small_ebm_training={"performed": False, "reason": "blocked_prerequisite"},
    )
    receipt["started_monotonic_ns"] = 0
    receipt["ended_monotonic_ns"] = ended - started
    gate = _gate(
        str(failed.get("check")),
        "prerequisite",
        str(failed.get("operator") or "=="),
        failed.get("expected"),
        failed.get("observed"),
        False,
        "Dependent policy work cannot replace absent or changed authority evidence.",
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked_upstream_prerequisite",
        "run_date": RUN_DATE,
        "started_at_utc": utc_now(),
        "completed_at_utc": utc_now(),
        **receipt,
        "model_duration_s": 0.0,
        "computation_duration_s": 0.0,
        "validation_duration_s": 0.0,
        "cold_start_duration_s": 0.0,
        "preconditions_checked": [deepcopy(dict(failed))],
        "random_seed": {
            "frozen_training_seeds": list(TRAINING_SEEDS),
            "bootstrap": BOOTSTRAP_SEED,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "selection_row_shard": {},
        "sample_size_budget": {
            "planned_units": len(DEPLOYED_HEADS),
            "attempted_units": 0,
            "completed_units": 0,
            "failed_units": 0,
            "censored_units": 0,
            "unstarted_units": len(DEPLOYED_HEADS),
            "stop_rule": "stop before dependent work on failed prerequisite",
        },
        "acceptance_gate_results": [gate],
        "gate_check_summary": _gate_summary([gate], failed),
        "verifier_is_oracle": False,
        "honest_verdict": "blocked_upstream_prerequisite",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "validation_manifest": {
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
        },
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "selection_protocol_ready_score": 0,
        "selection_diagnosis_rows": [],
        "protocol_manifest": {},
        "evaluation_reuse_disclosure": (
            "RAGTruth is previously exposed public evidence, not fresh confirmation."
        ),
        "deployment_certificate_valid": False,
        "certificate_scope": "exploratory_reused_corpus",
        "mutation_control_rows": [],
        "dense_equivalence_errors": [],
        "bootstrap_rows": [],
        "candidate_artifact": False,
        "fixture_artifact": False,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_fixture_artifact(
    root: Path, *, validation_receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Build a small complete artifact without reading repository evidence."""

    root.mkdir(parents=True, exist_ok=True)
    roles = {
        "fit": ["fit-a", "fit-b"],
        "probability_calibration": ["cal-a", "cal-b"],
        "policy_tuning": [f"policy_tuning-group-{index:03d}" for index in range(20)],
        "certification": [f"certification-group-{index:03d}" for index in range(40)],
    }
    protocol = build_protocol_manifest(roles)
    protocol_path = root / "fixture-protocol.json"
    seal_protocol(protocol_path, protocol)
    policy_rows: list[JsonDict] = []
    raw_rows: list[JsonDict] = []
    for index, head in enumerate(DEPLOYED_HEADS):
        tuning = _rows_for_fixture(head, "policy_tuning", 20)
        certification = _rows_for_fixture(head, "certification", 40)
        tuned, _ = select_tuned_policy(tuning)
        certificate = certify_policy(certification, tuned)
        row = {
            "row_type": "policy_unit",
            "head": head,
            "attempted": True,
            "completed": True,
            "failed": False,
            "censored": False,
            "unstarted": False,
            "status": "completed",
            "tuned_policy": tuned,
            "certificate": certificate,
        }
        policy_rows.append(row)
        raw_rows.append(
            {
                "row_type": "policy_summary",
                "head": head,
                "tuned_policy": tuned,
                "certificate": certificate,
            }
        )
        for role, role_rows in (("policy_tuning", tuning), ("certification", certification)):
            raw_rows.extend(
                {
                    "row_type": "source_group_score",
                    "head": head,
                    "role": role,
                    "row_key": source_row["row_key"],
                    "source_group": source_row["group_id"],
                    "label": source_row["label"],
                    "probability": source_row["probability"],
                }
                for source_row in role_rows
            )
    row_manifest = _write_rows(root / "fixture-selection-rows.jsonl", raw_rows)
    preconditions = [precondition_row("fixture", "fixture", "fixture", "ready", True, True)]
    started_ns = 1_000_000_000
    ended_ns = 2_000_000_000
    return build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes={},
        policy_rows=policy_rows,
        diagnosis_rows=[
            row
            for head in DEPLOYED_HEADS
            for row in diagnose_old_thresholds(head, _rows_for_fixture(head, "certification", 40))
        ],
        protocol_path=protocol_path,
        row_manifest=row_manifest,
        mutation_controls=run_mutation_controls(),
        equivalence_errors=[],
        bootstrap_rows=[
            {"head": head, "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED + index}
            for index, head in enumerate(DEPLOYED_HEADS)
        ],
        validation_receipts=validation_receipts,
        phase_spans=[],
        started_at_utc="2026-09-19T00:00:00+00:00",
        started_ns=started_ns,
        ended_ns=ended_ns,
        fixture=True,
    )


def _rows_for_fixture(head: str, role: str, count: int) -> list[JsonDict]:
    """Create deterministic independent groups for local reducer tests."""

    del head
    return [
        {
            "row_key": f"{role}-{index:03d}",
            "group_id": f"{role}-group-{index:03d}",
            "source_group": f"{role}-group-{index:03d}",
            "role": role,
            "label": int(index >= count // 2),
            "probability": (index + 1) / (count + 1),
        }
        for index in range(count)
    ]


def cold_replay(path: Path, *, root: Path = REPO_ROOT) -> list[str]:
    """Reload a candidate from fresh bytes and validate all bound evidence."""

    value = _load_object(path)
    return validate_artifact(value, root=root) if value else ["artifact_unreadable_or_not_object"]


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover - E2E only.
    """Build fresh replay, independent reduction, adversarial, and strict checks."""

    python = ".venv/bin/python"
    common = (python, "-u", WRAPPER_PATH.as_posix(), "--date", RUN_DATE)
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "cold_artifact_replay", (*common, "--cold-replay", str(candidate)), "candidate"
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_row_reduction",
                (*common, "--independent-reduce", str(candidate)),
                "candidate",
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "candidate",
            ),
            "safety",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "candidate",
            ),
            "completion",
            True,
        ),
    ]


def _score_protocol(
    root: Path,
    predictors: Mapping[str, Sequence[Mapping[str, Any]]],
    evaluators: Mapping[str, Sequence[Mapping[str, Any]]],
    states: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    run_started: float,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[str], list[JsonDict]]:
    """Average frozen seeds, tune three policies, and certify each once."""

    del root
    probability_halves = split_probability_groups(predictors["probability_calibration"])
    calibration_predictors = label_blind_representatives(
        probability_halves["probability_calibration"]
    )
    tuning_predictors = label_blind_representatives(probability_halves["policy_tuning"])
    certification_predictors = label_blind_representatives(predictors["policy_calibration"])
    calibration_rows = _attach_labels(calibration_predictors, evaluators["probability_calibration"])
    tuning_rows = _attach_labels(tuning_predictors, evaluators["probability_calibration"])
    certification_rows = _attach_labels(certification_predictors, evaluators["policy_calibration"])
    policy_rows: list[JsonDict] = []
    raw_rows: list[JsonDict] = []
    diagnosis_rows: list[JsonDict] = []
    bootstrap_rows: list[JsonDict] = []
    calibrated_by_head: dict[str, dict[str, list[float]]] = {}
    for head_index, head in enumerate(ALL_HEADS):
        progress(run_started, "aggregation", "before_benchmark", head=head)
        calibration_seed = _raw_seed_scores(states[head], calibration_rows)
        calibration = fit_ensemble_calibration(
            calibration_seed,
            [int(row["label"]) for row in calibration_rows],
        )
        role_scores: dict[str, list[float]] = {}
        for role, role_rows in (
            ("calibration", calibration_rows),
            ("tuning", tuning_rows),
            ("certification", certification_rows),
        ):
            raw = average_seed_probabilities(_raw_seed_scores(states[head], role_rows))
            calibrated = _apply_affine(raw, calibration["affine"])
            role_scores[role] = calibrated.tolist()
        calibrated_by_head[head] = role_scores
        if head == EQUIVALENCE_CONTROL:
            progress(run_started, "aggregation", "after_benchmark", head=head)
            continue
        tuning_scored = [
            {**deepcopy(dict(row)), "role": "policy_tuning", "probability": float(score)}
            for row, score in zip(tuning_rows, role_scores["tuning"], strict=True)
        ]
        certification_scored = [
            {**deepcopy(dict(row)), "role": "certification", "probability": float(score)}
            for row, score in zip(certification_rows, role_scores["certification"], strict=True)
        ]
        tuned, candidates = select_tuned_policy(tuning_scored)
        certificate = certify_policy(certification_scored, tuned)
        diagnostics = diagnose_old_thresholds(head, certification_scored)
        bootstrap = {
            "head": head,
            **_bootstrap_utility(
                certification_scored,
                tuned,
                seed=BOOTSTRAP_SEED + head_index,
            ),
        }
        diagnosis_rows.extend(diagnostics)
        bootstrap_rows.append(bootstrap)
        calibration_summary = {
            "head": head,
            "seed_count": len(TRAINING_SEEDS),
            "fit_order": calibration["fit_order"],
            "affine": calibration["affine"],
            "calibration_group_count": len(calibration_rows),
            "calibration_input_sha256": canonical_hash(
                {
                    "row_keys": [row["row_key"] for row in calibration_rows],
                    "mean_probabilities": average_seed_probabilities(calibration_seed).tolist(),
                    "labels": [row["label"] for row in calibration_rows],
                }
            ),
        }
        policy = {
            "row_type": "policy_unit",
            "head": head,
            "attempted": True,
            "completed": True,
            "failed": False,
            "censored": False,
            "unstarted": False,
            "status": "completed",
            "seed_count": len(TRAINING_SEEDS),
            "calibration": calibration_summary,
            "tuned_policy": tuned,
            "certificate": certificate,
        }
        policy_rows.append(policy)
        raw_rows.append(
            {
                "row_type": "policy_summary",
                "head": head,
                "calibration": calibration_summary,
                "tuned_policy": tuned,
                "candidate_rows": candidates,
                "certificate": certificate,
                "bootstrap": bootstrap,
            }
        )
        for role, scored in (
            ("policy_tuning", tuning_scored),
            ("certification", certification_scored),
        ):
            for row in scored:
                raw_rows.append(
                    {
                        "row_type": "source_group_score",
                        "head": head,
                        "role": role,
                        "row_key": row["row_key"],
                        "source_group": row["group_id"],
                        "label": row["label"],
                        "probability": row["probability"],
                    }
                )
        progress(run_started, "aggregation", "after_benchmark", head=head)
    equivalence = dense_equivalence_errors(
        calibrated_by_head["sparse_spline_49"],
        calibrated_by_head[EQUIVALENCE_CONTROL],
    )
    return policy_rows, raw_rows, diagnosis_rows, equivalence, bootstrap_rows


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - declared entrypoint E2E.
    """Authenticate, seal, aggregate, validate, replay, and publish atomically."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []

    progress(run_started, "preconditions", "start")
    phase_started = time.monotonic()
    preconditions, source_hashes, upstreams = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, run_started, len(preconditions)))
    progress(run_started, "preconditions", "end", completed=len(preconditions))
    failed = next((row for row in preconditions if row.get("passed") is not True), None)
    if failed is not None:
        blocked = build_blocked_artifact(failed)
        progress(run_started, "write", "before_atomic_terminal", status="blocked")
        atomic_json(root / output_path, blocked)
        progress(run_started, "write", "after_atomic_terminal", status="blocked")
        return blocked

    progress(run_started, "allowed_rows", "before_benchmark")
    phase_started = time.monotonic()
    predictors, evaluators, shard_receipts = load_allowed_protocol_rows(root)
    probability_halves = split_probability_groups(predictors["probability_calibration"])
    roles = {
        "fit": sorted({str(row["group_id"]) for row in predictors["fit"]}),
        "probability_calibration": sorted(
            {str(row["group_id"]) for row in probability_halves["probability_calibration"]}
        ),
        "policy_tuning": sorted(
            {str(row["group_id"]) for row in probability_halves["policy_tuning"]}
        ),
        "certification": sorted({str(row["group_id"]) for row in predictors["policy_calibration"]}),
    }
    protocol = build_protocol_manifest(roles)
    protocol_path = root / PROTOCOL_PATH
    seal_protocol(protocol_path, protocol)
    spans.append(_span("allowed_rows", phase_started, run_started, sum(map(len, roles.values()))))
    progress(
        run_started, "allowed_rows", "after_benchmark", completed=sum(map(len, roles.values()))
    )

    progress(run_started, "frozen_states", "before_model_load")
    phase_started = time.monotonic()
    states, checkpoint_receipts = _load_frozen_states(root, upstreams["static_decisions"])
    spans.append(_span("frozen_states", phase_started, run_started, len(checkpoint_receipts)))
    progress(run_started, "frozen_states", "after_model_load", completed=len(checkpoint_receipts))

    progress(run_started, "aggregation", "before_benchmark", planned=len(ALL_HEADS))
    phase_started = time.monotonic()
    policy_rows, raw_rows, diagnosis_rows, equivalence, bootstrap_rows = _score_protocol(
        root,
        predictors,
        evaluators,
        states,
        run_started=run_started,
    )
    spans.append(_span("aggregation", phase_started, run_started, len(policy_rows)))
    progress(run_started, "aggregation", "after_benchmark", completed=len(policy_rows))

    mutation_controls = run_mutation_controls()
    row_manifest = _write_rows(root / ROW_PATH, raw_rows)
    for row in [*shard_receipts, *checkpoint_receipts]:
        source_hashes[row["path"]] = {
            "path": row["path"],
            "sha256": row["sha256"],
            "original_flagged_adversarial": None,
            **(
                {"source_receipt_class": row["source_receipt_class"]}
                if "source_receipt_class" in row
                else {}
            ),
        }
    for relative in (
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        Path("openspec/capabilities/autoresearch/spec.md"),
    ):
        path = root / relative
        source_hashes[relative.as_posix()] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(path),
            "original_flagged_adversarial": None,
        }
    source_hashes[PROTOCOL_PATH.as_posix()] = {
        "path": PROTOCOL_PATH.as_posix(),
        "sha256": sha256_file(protocol_path),
        "original_flagged_adversarial": None,
    }
    source_hashes[ROW_PATH.as_posix()] = {
        "path": ROW_PATH.as_posix(),
        "sha256": row_manifest["sha256"],
        "original_flagged_adversarial": None,
    }

    private_root = Path(tempfile.mkdtemp(prefix="exp7436-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    progress(run_started, "affected_validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic()
    affected = (
        []
        if plan_errors
        else run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=root / RAW_DIR / "validation/affected",
        )
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(_span("affected_validation", phase_started, run_started, len(affected)))
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=affected_reduction["passed"],
    )
    candidate = build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        policy_rows=policy_rows,
        diagnosis_rows=diagnosis_rows,
        protocol_path=protocol_path,
        row_manifest=row_manifest,
        mutation_controls=mutation_controls,
        equivalence_errors=equivalence,
        bootstrap_rows=bootstrap_rows,
        validation_receipts=affected,
        phase_spans=spans,
        started_at_utc=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        candidate=True,
        flagged_adversarial=not affected_reduction["passed"] or bool(plan_errors),
    )
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    terminal_commands = _terminal_commands(candidate_path)
    progress(
        run_started, "terminal_validation", "before_subprocesses", planned=len(terminal_commands)
    )
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        root,
        terminal_commands,
        log_dir=root / RAW_DIR / "validation/terminal",
    )
    spans.append(_span("terminal_validation", phase_started, run_started, len(terminal)))
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )
    final = build_artifact(
        root=root,
        preconditions=preconditions,
        source_hashes=source_hashes,
        policy_rows=policy_rows,
        diagnosis_rows=diagnosis_rows,
        protocol_path=protocol_path,
        row_manifest=row_manifest,
        mutation_controls=mutation_controls,
        equivalence_errors=equivalence,
        bootstrap_rows=bootstrap_rows,
        validation_receipts=[*affected, *terminal],
        phase_spans=spans,
        started_at_utc=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        flagged_adversarial=(
            not affected_reduction["passed"] or not terminal_passed or critical or bool(plan_errors)
        ),
    )
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(run_started, "write", "before_atomic_terminal", path=output_path)
    atomic_json(root / output_path, final)
    progress(run_started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed date and strict fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the protocol or one strict fresh-process artifact reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay, root=args.root)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        errors = (
            validate_artifact(value, root=args.root)
            if value
            else ["artifact_unreadable_or_not_object"]
        )
        reduced = independent_reduce(value, root=args.root) if value and not errors else {}
        print(json.dumps({"errors": errors, "reduction": reduced}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(args.root, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

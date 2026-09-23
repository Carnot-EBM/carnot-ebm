"""Audit V661 source decisions and delayed-learning evidence.

The audit reads sealed producer bytes. It does not load a model, refit a head,
or treat a missing measurement as a zero effect.

Spec refs: REQ-REPORT-7569 and SCENARIO-REPORT-7569-*.
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
import platform
import sys
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
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260923"
MILESTONE = "2026.09.661"
EXPERIMENT_ID = "exp7569-v661-decision-learning-audit"
SCHEMA = "carnot.exp7569.v661.decision_learning_audit.v1"
RESULT_PATH = Path("results/experiment_7569_v661_decision_learning_audit.json")
RAW_DIR = Path("results/raw/experiment_7569_v661_decision_learning_audit")
MODULE_PATH = Path("python/carnot/experiment_7569_v661_decision_learning_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7569_v661_decision_learning_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7569_v661_decision_learning_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
TERMINAL_CANDIDATE_PATH = RAW_DIR / "measured_terminal_candidate.json"
VALIDATION_MANIFEST_PATH = RAW_DIR / "affected_validation_manifest.json"

PRODUCER_IDS = tuple(f"exp{number}" for number in range(7563, 7569))
PRODUCER_PATHS = {
    "exp7563": Path("results/experiment_7563_v661_native_pilot.json"),
    "exp7564": Path("results/experiment_7564_v661_fit_capture.json"),
    "exp7565": Path("results/experiment_7565_v661_test_online_capture.json"),
    "exp7566": Path("results/experiment_7566_v661_energy_fit.json"),
    "exp7567": Path("results/experiment_7567_v661_source_evaluation.json"),
    "exp7568": Path("results/experiment_7568_v661_continuous_recalibration.json"),
}
DIAGNOSTIC_PATHS = {
    "exp7568": Path("results/experiment_7568_continuous_recalibration.json"),
}
STATIC_FEATURE_PATH = Path(
    "results/raw/experiment_7565_v661_test_online_capture/capture/predictions.jsonl"
)
STATIC_LABEL_PATH = Path(
    "results/raw/experiment_7565_v661_test_online_capture/capture/test_labels.jsonl"
)
STATIC_HEAD_PATH = Path("results/raw/experiment_7566_v661_energy_fit/frozen_heads.json")
STATIC_ROW_PATH = Path("results/raw/experiment_7567_v661_source_evaluation/source_group_rows.jsonl")
STATIC_REPORT_PATH = Path(
    "results/raw/experiment_7567_v661_source_evaluation/static_evaluation_report.json"
)

PROBABILITY_TOLERANCE = 1e-8
METRIC_TOLERANCE = 1e-10
BOOTSTRAP_SEED = 7_567_001
BOOTSTRAP_DRAWS = 2_000
ORDER_NAMES = ("supported_first", "unsupported_first")
CANDIDATE_ARM = "source_contrast_energy"
ARMS = (
    CANDIDATE_ARM,
    "temperature_original",
    "original_only_local_basis_energy",
    "unconstrained_equal_capacity_energy",
    "same_information_logistic",
    "raw_original",
)
TRAINED_ARMS = (
    CANDIDATE_ARM,
    "original_only_local_basis_energy",
    "unconstrained_equal_capacity_energy",
    "same_information_logistic",
)
BRIER_COMPARATORS = (
    "temperature_original",
    "original_only_local_basis_energy",
    "unconstrained_equal_capacity_energy",
)
MUTATION_NAMES = (
    "deleted_source_row",
    "duplicate_component",
    "wrong_option_orientation",
    "future_origin_feedback",
    "unchanged_shuffle",
    "altered_checkpoint",
    "missing_tail",
    "favorable_aggregate_contradictory_rows",
)
ZERO_INVOCATION_COUNTS = {
    operation: {state: 0 for state in ("attempted", "completed", "failed", "cancelled")}
    for operation in ("model_loads", "forward_calls", "generation_calls")
}
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_source_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
AFFECTED_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def canonical_hash(value: Any) -> str:
    """Hash stable JSON so changed evidence cannot keep the same identity."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact bytes before the audit trusts a producer or sidecar."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_json(path: Path) -> JsonDict:
    """Return one JSON object, or an empty object for malformed external bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def load_jsonl(path: Path) -> list[JsonDict]:
    """Read object rows without repairing malformed external evidence."""

    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"jsonl_row_not_object:{path}:{line_number}")
            rows.append(value)
    return rows


def sidecar_reference(path: Path, root: Path, *, rows: int | None = None) -> JsonDict:
    """Bind one evidence file by relative path, byte hash, size, and row count."""

    resolved = path.resolve()
    try:
        label = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:  # pragma: no cover - private tests can use absolute evidence.
        label = str(resolved)
    reference: JsonDict = {
        "path": label,
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    if rows is not None:
        reference["rows"] = rows
    return reference


def _resolve(root: Path, label: str) -> Path:
    """Resolve a stored repository label without changing its original spelling."""

    path = Path(label)
    return path if path.is_absolute() else root / path


def read_sidecar(root: Path, reference: Mapping[str, Any], name: str) -> list[JsonDict]:
    """Authenticate one JSONL sidecar before returning any scientific rows."""

    path = _resolve(root, str(reference.get("path") or ""))
    if not path.is_file() or sha256_file(path) != reference.get("sha256"):
        raise ValueError(f"sidecar_hash_mismatch:{name}")
    if path.stat().st_size != reference.get("bytes"):
        raise ValueError(f"sidecar_size_mismatch:{name}")
    rows = load_jsonl(path)
    if "rows" in reference and len(rows) != reference.get("rows"):
        raise ValueError(f"sidecar_row_count_mismatch:{name}")
    return rows


def _diagnostic_failure(value: Mapping[str, Any]) -> JsonDict:
    """Project the exact first conductor failure without interpreting it."""

    contract = value.get("blocked_diagnostic_contract")
    if not isinstance(contract, Mapping):
        return {}
    return {
        "upstream": contract.get("failed_upstream"),
        "path": contract.get("failed_evidence_path"),
        "field": contract.get("failed_field"),
        "op": contract.get("failed_operator"),
        "expected": contract.get("failed_expected"),
        "observed": contract.get("failed_observed"),
    }


def inventory_producers(root: Path) -> list[JsonDict]:
    """Inventory six roadmap producers and preserve a real pre-gate diagnostic."""

    output: list[JsonDict] = []
    for producer_id in PRODUCER_IDS:
        declared = PRODUCER_PATHS[producer_id]
        path = root / declared
        if path.is_file():
            value = load_json(path)
            state = "present" if value else "invalid"
            output.append(
                {
                    "producer_id": producer_id,
                    "declared_path": declared.as_posix(),
                    "evidence_path": declared.as_posix(),
                    "state": state,
                    "sha256": sha256_file(path),
                    "honest_verdict": value.get("honest_verdict"),
                    "verdict_class": value.get("verdict_class"),
                    "flagged_adversarial": value.get("flagged_adversarial"),
                    "gate_failure": None,
                }
            )
            continue
        diagnostic_relative = DIAGNOSTIC_PATHS.get(producer_id)
        diagnostic_path = root / diagnostic_relative if diagnostic_relative else None
        diagnostic = load_json(diagnostic_path) if diagnostic_path is not None else {}
        valid_diagnostic = bool(
            diagnostic
            and diagnostic.get("schema") == "blocked_gate_check_v1"
            and diagnostic.get("blocked_at_layer") == "conductor_pre_gate"
            and _diagnostic_failure(diagnostic)
        )
        output.append(
            {
                "producer_id": producer_id,
                "declared_path": declared.as_posix(),
                "evidence_path": (
                    diagnostic_relative.as_posix()
                    if valid_diagnostic and diagnostic_relative
                    else None
                ),
                "state": "blocked_pre_gate" if valid_diagnostic else "absent",
                "sha256": (
                    sha256_file(diagnostic_path)
                    if valid_diagnostic and diagnostic_path is not None
                    else None
                ),
                "honest_verdict": diagnostic.get("honest_verdict") if valid_diagnostic else None,
                "verdict_class": "blocked" if valid_diagnostic else None,
                "flagged_adversarial": None,
                "gate_failure": _diagnostic_failure(diagnostic) if valid_diagnostic else None,
            }
        )
    return output


REQUIRED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7550_v660_count_audit.py"),
    Path("python/carnot/experiment_7510_v657_causal_audit.py"),
    Path("python/carnot/experiment_7508_v657_static_audit.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    SPEC_PATH,
    STATIC_FEATURE_PATH,
    STATIC_LABEL_PATH,
    STATIC_HEAD_PATH,
    STATIC_ROW_PATH,
    STATIC_REPORT_PATH,
)


def _check(
    check: str,
    upstream: str,
    path_or_field: str,
    expected: Any,
    observed: Any,
    *,
    required: bool = True,
    op: str = "==",
) -> JsonDict:
    """Retain one exact operand so a block never becomes an ambiguous absence."""

    if op == "in":
        passed = observed in expected
    else:
        passed = observed == expected
    return {
        "check": check,
        "upstream": upstream,
        "path_or_field": path_or_field,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": op,
        "passed": passed,
        "required": required,
    }


def collect_preconditions(root: Path) -> JsonDict:
    """Check instructions, requirements, source bytes, and producer dispositions."""

    rows: list[JsonDict] = []
    source_hashes: list[JsonDict] = []
    for relative in REQUIRED_INPUTS:
        path = root / relative
        readable = path.is_file() and path.stat().st_size > 0
        rows.append(
            _check(
                "resource_readable",
                "worktree_instruction_or_evidence",
                relative.as_posix(),
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if readable else "missing_or_empty",
            )
        )
        if readable:
            source_hashes.append({**sidecar_reference(path, root), "role": "repository_input"})
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    rows.append(
        _check(
            "relevant_requirement_present",
            "OpenSpec",
            "REQ-REPORT-7569",
            True,
            "REQ-REPORT-7569" in spec_text,
        )
    )
    roadmap_text = (root / "research-roadmap.yaml").read_text(encoding="utf-8")
    for producer_id in PRODUCER_IDS:
        declared = PRODUCER_PATHS[producer_id].as_posix()
        rows.append(
            _check(
                "roadmap_deliverable_declared",
                producer_id,
                declared,
                True,
                declared in roadmap_text,
            )
        )
    inventory = inventory_producers(root)
    for item in inventory:
        rows.append(
            _check(
                "producer_or_authentic_diagnostic",
                str(item["producer_id"]),
                str(item["declared_path"]),
                ("present", "blocked_pre_gate"),
                item["state"],
                op="in",
            )
        )
        evidence_path = item.get("evidence_path")
        if isinstance(evidence_path, str):
            source_hashes.append(
                {
                    **sidecar_reference(_resolve(root, evidence_path), root),
                    "role": "producer_or_gate_diagnostic",
                    "producer_id": item["producer_id"],
                }
            )
    return {
        "rows": rows,
        "inventory": inventory,
        "source_artifact_hashes": source_hashes,
        "missing_external": [
            str(item["declared_path"])
            for item in inventory
            if item["state"] in {"blocked_pre_gate", "absent"}
        ],
        "invalid_present": [
            item["declared_path"] for item in inventory if item["state"] == "invalid"
        ],
    }


def sigmoid(value: float) -> float:
    """Compute one finite logistic value without overflow."""

    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def losses(probability: float, label: int) -> JsonDict:
    """Compute Brier and metric-only clipped log loss."""

    numeric = float(probability)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0 or label not in (0, 1):
        raise ValueError("probability_or_label_invalid")
    clipped = min(max(numeric, 1e-9), 1.0 - 1e-9)
    return {
        "brier": (numeric - label) ** 2,
        "log_loss": -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped)),
    }


def typed_decision(probability: float, label: int) -> JsonDict:
    """Apply the fixed accept, reject, or escalate policy with tie priority."""

    losses(probability, label)
    expected = {
        "accept": 5.0 * float(probability),
        "reject": 1.0 - float(probability),
        "escalate": 0.2,
    }
    order = ("escalate", "accept", "reject")
    action = min(order, key=lambda name: (expected[name], order.index(name)))
    realized = {
        "accept": 5.0 if label == 1 else 0.0,
        "reject": 1.0 if label == 0 else 0.0,
        "escalate": 0.2,
    }[action]
    return {"action": action, "realized_cost": float(realized), "expected_costs": expected}


def _basis_values(value: float, knots: Sequence[float]) -> list[float]:
    """Evaluate one open cubic B-spline without using producer code."""

    vector = [float(item) for item in knots]
    if len(vector) != 12 or any(not math.isfinite(item) for item in vector):
        raise ValueError("knot_vector_invalid")  # pragma: no cover - defensive evidence check.
    lower, upper = vector[3], vector[-4]
    evaluated = min(max(float(value), lower), upper)
    if evaluated == lower:
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if evaluated == upper:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    current = [
        1.0 if vector[index] <= evaluated < vector[index + 1] else 0.0
        for index in range(len(vector) - 1)
    ]
    for degree in range(1, 4):
        next_values: list[float] = []
        for index in range(len(vector) - degree - 1):
            left_width = vector[index + degree] - vector[index]
            right_width = vector[index + degree + 1] - vector[index + 1]
            left = (
                0.0
                if left_width == 0.0
                else (evaluated - vector[index]) * current[index] / left_width
            )
            right = (
                0.0
                if right_width == 0.0
                else (vector[index + degree + 1] - evaluated) * current[index + 1] / right_width
            )
            next_values.append(left + right)
        current = next_values
    return current


def _design(arm: str, normalized: Sequence[float], knots: Sequence[Sequence[float]]) -> list[float]:
    """Build one frozen design vector from three normalized features."""

    values = [float(item) for item in normalized]
    if arm == CANDIDATE_ARM:
        return [
            entry
            for index, value in enumerate(values)
            for entry in _basis_values(value, knots[index])
        ]
    if arm == "original_only_local_basis_energy":
        return _basis_values(values[0], knots[0])
    if arm == "same_information_logistic":
        return values
    if arm == "unconstrained_equal_capacity_energy":
        clipped = [min(4.0, max(-4.0, value)) for value in values]
        return [value**power for power in range(1, 9) for value in clipped]
    raise ValueError(f"trained_arm_invalid:{arm}")  # pragma: no cover - closed arm roster.


def _trained_probability(arm: str, features: Sequence[float], heads: Mapping[str, Any]) -> float:
    """Reconstruct one learned probability from frozen coefficients."""

    normalization = heads.get("normalization") or {}
    mean = [float(item) for item in normalization.get("mean") or []]
    scale = [float(item) for item in normalization.get("safe_scale") or []]
    if len(mean) != 3 or len(scale) != 3 or any(item <= 0.0 for item in scale):
        raise ValueError("normalization_invalid")  # pragma: no cover - authenticated real input.
    normalized = [(float(features[index]) - mean[index]) / scale[index] for index in range(3)]
    design = _design(arm, normalized, (heads.get("local_basis") or {}).get("knots") or [])
    head = (heads.get("selected_heads") or {}).get(arm) or {}
    checkpoint = head.get("checkpoint") or {}
    if head.get("checkpoint_sha256") != canonical_hash(checkpoint):
        raise ValueError(f"checkpoint_hash_mismatch:{arm}")
    weights = [float(item) for item in checkpoint.get("weights") or []]
    if len(weights) != len(design):
        raise ValueError(f"checkpoint_shape_mismatch:{arm}")  # pragma: no cover
    logit = float(checkpoint.get("bias")) + math.fsum(
        weight * entry for weight, entry in zip(weights, design, strict=True)
    )
    return sigmoid(logit)


def _temperature_probability(probability: float, temperature: float) -> float:
    """Apply the frozen scalar temperature to a raw option probability."""

    clipped = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return sigmoid(math.log(clipped / (1.0 - clipped)) / temperature)


def _probability(arm: str, features: Sequence[float], heads: Mapping[str, Any]) -> float:
    """Reconstruct one option-order probability for any registered arm."""

    if arm in TRAINED_ARMS:
        return _trained_probability(arm, features, heads)
    raw = sigmoid(float(features[0]))
    if arm == "raw_original":
        return raw
    if arm == "temperature_original":
        temperature = float((heads.get("temperature_baseline") or {}).get("selected_temperature"))
        return _temperature_probability(raw, temperature)
    raise ValueError(f"arm_invalid:{arm}")  # pragma: no cover - closed arm roster.


def _walk_references(value: Any) -> list[Mapping[str, Any]]:
    """Find path-and-hash references in a producer without trusting field names."""

    output: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            output.append(value)
        for item in value.values():
            output.extend(_walk_references(item))
    elif isinstance(value, list):
        for item in value:
            output.extend(_walk_references(item))
    return output


def _authenticate_declared(
    root: Path, relative: Path, producers: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Require one producer to bind the exact current bytes of a sidecar."""

    path = root / relative
    observed = sha256_file(path) if path.is_file() else None
    references = [
        row
        for producer in producers
        for row in _walk_references(producer)
        if row.get("path") == relative.as_posix()
    ]
    if observed is None or not any(row.get("sha256") == observed for row in references):
        raise ValueError(f"undeclared_or_changed_sidecar:{relative}")
    row_count = None
    if relative.suffix == ".jsonl":
        row_count = sum(1 for _line in path.open(encoding="utf-8"))
    return sidecar_reference(path, root, rows=row_count)


def load_static_inputs(root: Path) -> JsonDict:
    """Authenticate and load only operands needed for independent static replay."""

    capture = load_json(root / PRODUCER_PATHS["exp7565"])
    fit = load_json(root / PRODUCER_PATHS["exp7566"])
    evaluation = load_json(root / PRODUCER_PATHS["exp7567"])
    if not all((capture, fit, evaluation)):
        raise ValueError("static_producer_missing")  # pragma: no cover - real precondition.
    producers = (capture, fit, evaluation)
    references = {
        name: _authenticate_declared(root, path, producers)
        for name, path in {
            "features": STATIC_FEATURE_PATH,
            "labels": STATIC_LABEL_PATH,
            "heads": STATIC_HEAD_PATH,
            "published_rows": STATIC_ROW_PATH,
            "report": STATIC_REPORT_PATH,
        }.items()
    }
    feature_rows = [
        row
        for row in read_sidecar(root, references["features"], "features")
        if row.get("role") == "test"
    ]
    return {
        "features": feature_rows,
        "labels": read_sidecar(root, references["labels"], "labels"),
        "heads": load_json(root / STATIC_HEAD_PATH),
        "published_rows": read_sidecar(root, references["published_rows"], "published_rows"),
        "report": load_json(root / STATIC_REPORT_PATH),
        "references": references,
        "producer": evaluation,
    }


def _reconstruct_rows(
    features: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    heads: Mapping[str, Any],
) -> tuple[list[JsonDict], list[str]]:
    """Rebuild every probability, loss, and action from sealed operands."""

    errors: list[str] = []
    label_by_component: dict[str, int] = {}
    for row in labels:
        component = str(row.get("component_hash") or "")
        if component in label_by_component:
            errors.append(f"duplicate_label_component:{component}")
        if row.get("role") != "test" or row.get("label") not in (0, 1):
            errors.append(f"label_row_invalid:{component}")
        label_by_component[component] = int(row.get("label", -1))
    seen: set[str] = set()
    output: list[JsonDict] = []
    for row in features:
        component = str(row.get("component_hash") or "")
        if component in seen:
            errors.append(f"duplicate_component:{component}")
            continue
        seen.add(component)
        if component not in label_by_component:
            errors.append(f"source_label_missing:{component}")
            continue
        views = row.get("feature_views")
        if not isinstance(views, Mapping) or tuple(views) != ORDER_NAMES:
            errors.append(f"option_orientation_invalid:{component}")
            continue
        vectors = {name: [float(item) for item in views[name]] for name in ORDER_NAMES}
        if any(len(vector) != 3 for vector in vectors.values()):
            errors.append(f"feature_shape_invalid:{component}")  # pragma: no cover
            continue  # pragma: no cover
        raw_mean = math.fsum(sigmoid(vectors[name][0]) for name in ORDER_NAMES) / 2.0
        if not math.isclose(
            raw_mean,
            float(row.get("original_unsupported_probability")),
            abs_tol=PROBABILITY_TOLERANCE,
        ):
            errors.append(f"option_orientation_probability_mismatch:{component}")
        label = label_by_component[component]
        for arm in ARMS:
            option = {name: _probability(arm, vectors[name], heads) for name in ORDER_NAMES}
            probability = math.fsum(option.values()) / 2.0
            score = losses(probability, label)
            decision = typed_decision(probability, label)
            output.append(
                {
                    "source_component_hash": component,
                    "group_id": row.get("group_hash"),
                    "arm": arm,
                    "option_probabilities": option,
                    "probability": probability,
                    "label": label,
                    **score,
                    "action": decision["action"],
                    "realized_cost": decision["realized_cost"],
                }
            )
    if seen != set(label_by_component):
        errors.append("source_label_roster_mismatch")
    return sorted(output, key=lambda item: (item["source_component_hash"], item["arm"])), errors


def _expected_head_identity(arm: str, heads: Mapping[str, Any]) -> str:
    """Rebuild the immutable identity of a learned or analytic arm."""

    if arm in TRAINED_ARMS:
        return str((heads.get("selected_heads") or {}).get(arm, {}).get("checkpoint_sha256"))
    temperature = (
        float((heads.get("temperature_baseline") or {}).get("selected_temperature"))
        if arm == "temperature_original"
        else None
    )
    return canonical_hash({"arm": arm, "temperature": temperature})


def _compare_rows(
    rebuilt: Sequence[Mapping[str, Any]],
    published: Sequence[Mapping[str, Any]],
    heads: Mapping[str, Any],
    report: Mapping[str, Any],
) -> tuple[list[str], float, float]:
    """Compare every stable row field with separate probability and metric tolerances."""

    errors: list[str] = []
    published_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in published:
        key = (str(row.get("source_component_hash") or ""), str(row.get("arm") or ""))
        if key in published_by_key:
            errors.append(f"duplicate_published_component_arm:{key[0]}:{key[1]}")
        published_by_key[key] = row
    rebuilt_by_key = {(str(row["source_component_hash"]), str(row["arm"])): row for row in rebuilt}
    if set(rebuilt_by_key) != set(published_by_key):
        errors.append("published_row_roster_mismatch")
    max_probability_error = 0.0
    max_metric_error = 0.0
    freeze = (report.get("prediction_sidecar") or {}).get("sha256")
    for key, expected in rebuilt_by_key.items():
        observed = published_by_key.get(key)
        if observed is None:
            continue
        for order in ORDER_NAMES:
            difference = abs(
                float(expected["option_probabilities"][order])
                - float((observed.get("option_probabilities") or {}).get(order, math.nan))
            )
            max_probability_error = max(max_probability_error, difference)
        difference = abs(float(expected["probability"]) - float(observed.get("probability")))
        max_probability_error = max(max_probability_error, difference)
        for name in ("brier", "log_loss", "realized_cost"):
            difference = abs(float(expected[name]) - float(observed.get(name)))
            max_metric_error = max(max_metric_error, difference)
        if observed.get("action") != expected["action"]:
            errors.append(f"action_mismatch:{key[0]}:{key[1]}")
        if observed.get("head_identity") != _expected_head_identity(key[1], heads):
            errors.append(f"head_hash_mismatch:{key[0]}:{key[1]}")
        if freeze is not None and observed.get("prediction_freeze_sha256") != freeze:
            errors.append(f"prediction_hash_mismatch:{key[0]}:{key[1]}")
    if max_probability_error > PROBABILITY_TOLERANCE:
        errors.append("probability_tolerance_exceeded")
    if max_metric_error > METRIC_TOLERANCE:
        errors.append("metric_tolerance_exceeded")
    return errors, max_probability_error, max_metric_error


def _paired_interval(deltas: Sequence[float], indices: np.ndarray) -> JsonDict:
    """Reduce one source-paired contrast with the frozen bootstrap indices."""

    values = np.asarray(deltas, dtype=np.float64)
    means = values[indices].mean(axis=1)
    return {
        "delta": float(values.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "upper95": float(np.quantile(means, 0.95)),
        "one_sided_p": float((1 + np.sum(means >= 0.0)) / (len(means) + 1)),
        "direction": "candidate_minus_comparator_lower_is_better",
        "group_count": len(values),
        "draws": len(means),
        "seed": BOOTSTRAP_SEED,
        "bootstrap_means": means,
    }


def _static_reduction(rows: Sequence[Mapping[str, Any]], draws: int) -> JsonDict:
    """Reduce independent source components without treating arms as units."""

    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        component = str(row["source_component_hash"])
        arm = str(row["arm"])
        grouped.setdefault(component, {})[arm] = row
    components = sorted(grouped)
    if any(set(value) != set(ARMS) for value in grouped.values()):
        raise ValueError("arm_roster_invalid")  # pragma: no cover - reconstruction owns roster.
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(components), size=(draws, len(components))
    )
    raw_intervals = {
        comparator: _paired_interval(
            [
                float(grouped[component][CANDIDATE_ARM]["brier"])
                - float(grouped[component][comparator]["brier"])
                for component in components
            ],
            indices,
        )
        for comparator in BRIER_COMPARATORS
    }
    ordered = sorted(raw_intervals, key=lambda name: (raw_intervals[name]["one_sided_p"], name))
    adjusted = 0.0
    brier: dict[str, JsonDict] = {}
    for rank, comparator in enumerate(ordered, 1):
        row = raw_intervals[comparator]
        remaining = len(ordered) - rank + 1
        adjusted = max(adjusted, min(1.0, remaining * float(row["one_sided_p"])))
        means = row["bootstrap_means"]
        brier[comparator] = {
            key: deepcopy(value) for key, value in row.items() if key != "bootstrap_means"
        }
        brier[comparator].update(
            {
                "comparator": comparator,
                "holm_rank": rank,
                "holm_threshold": 0.05 / remaining,
                "holm_adjusted_p": adjusted,
                "simultaneous_upper95": float(np.quantile(means, 1.0 - 0.05 / remaining)),
            }
        )
    selected = "temperature_original"
    log_loss = _paired_interval(
        [
            float(grouped[item][CANDIDATE_ARM]["log_loss"])
            - float(grouped[item][selected]["log_loss"])
            for item in components
        ],
        indices,
    )
    log_loss.pop("bootstrap_means")
    log_loss["comparator"] = selected
    cost = _paired_interval(
        [
            float(grouped[item][CANDIDATE_ARM]["realized_cost"])
            - float(grouped[item][selected]["realized_cost"])
            for item in components
        ],
        indices,
    )
    cost.pop("bootstrap_means")
    cost["comparator"] = selected
    return {
        "source_count": len(components),
        "row_count": len(rows),
        "probability_metrics": {
            arm: {
                "brier": float(np.mean([grouped[item][arm]["brier"] for item in components])),
                "log_loss": float(np.mean([grouped[item][arm]["log_loss"] for item in components])),
                "primary_cost": float(
                    np.mean([grouped[item][arm]["realized_cost"] for item in components])
                ),
                "non_escalation_fraction": float(
                    np.mean([grouped[item][arm]["action"] != "escalate" for item in components])
                ),
                "n_source_components": len(components),
            }
            for arm in ARMS
        },
        "paired_intervals": {"brier": brier, "log_loss": log_loss},
        "primary_cost_contrast": cost,
    }


def _nested_error(left: Any, right: Any) -> float:
    """Return the largest numeric difference, or infinity for structural drift."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return math.inf
        return max((_nested_error(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return math.inf
        return max((_nested_error(a, b) for a, b in zip(left, right, strict=True)), default=0.0)
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return abs(float(left) - float(right))
    return 0.0 if left == right else math.inf


def _strongest_comparator(heads: Mapping[str, Any]) -> str:
    """Select the smallest frozen tuning Brier without reading test outcomes."""

    candidates = {
        "temperature_original": float((heads.get("temperature_baseline") or {})["tune_brier"])
    }
    for arm, row in (heads.get("selected_heads") or {}).items():
        if arm != CANDIDATE_ARM:
            candidates[str(arm)] = float(row["tune_brier"])
    return min(candidates, key=lambda name: (candidates[name], name))


def reconstruct_static(
    *,
    features: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    heads: Mapping[str, Any],
    published_rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    references: Mapping[str, Any],
    producer: Mapping[str, Any],
    draws: int,
) -> JsonDict:
    """Reconstruct source decisions and registered intervals from raw operands."""

    del references
    rebuilt, errors = _reconstruct_rows(features, labels, heads)
    row_errors, max_probability_error, max_metric_error = _compare_rows(
        rebuilt, published_rows, heads, report
    )
    errors.extend(row_errors)
    reduction = _static_reduction(rebuilt, draws)
    producer_reduction = report.get("reduction") or {}
    comparable = {
        "probability_metrics": reduction["probability_metrics"],
        "paired_intervals": reduction["paired_intervals"],
        "primary_cost_contrast": reduction["primary_cost_contrast"],
    }
    reported = {key: producer_reduction.get(key) for key in comparable}
    aggregate_error = _nested_error(comparable, reported)
    if aggregate_error > METRIC_TOLERANCE:
        errors.append("producer_aggregate_mismatch")
    strongest = _strongest_comparator(heads)
    if strongest != (heads.get("strongest_comparator") or {}).get("family"):
        errors.append("strongest_comparator_mismatch")
    if reduction["source_count"] != 80 or reduction["row_count"] != 480:
        errors.append("static_count_mismatch")
    if producer.get("static_measurement_complete_score") != 1:
        errors.append("producer_measurement_incomplete")
    if producer.get("flagged_adversarial") is not False:
        errors.append("producer_critical_flag")
    validation = producer.get("validation_summary") or {}
    if validation.get("required_checks_passed") is not True:
        errors.append("producer_validation_failed")
    source_qualified = not errors
    probability_score = int(producer_reduction.get("probability_benefit_score", 0))
    decision_score = int(producer_reduction.get("decision_benefit_score", 0))
    return {
        "errors": list(dict.fromkeys(errors)),
        "source_count": reduction["source_count"],
        "row_count": reduction["row_count"],
        "strongest_comparator": strongest,
        "max_probability_error": max_probability_error,
        "max_metric_error": max(max_metric_error, aggregate_error),
        "probability_tolerance": PROBABILITY_TOLERANCE,
        "metric_tolerance": METRIC_TOLERANCE,
        "floating_point_uncertainty": {
            "probability": "float64 scalar reconstruction; absolute tolerance fixed at 1e-8",
            "metric": "float64 paired reduction; absolute tolerance fixed at 1e-10",
        },
        "producer_agreement": aggregate_error <= METRIC_TOLERANCE and not row_errors,
        "source_claims_qualified": source_qualified,
        "producer_probability_benefit_score": probability_score,
        "producer_decision_benefit_score": decision_score,
        "probability_metrics": reduction["probability_metrics"],
        "paired_intervals": reduction["paired_intervals"],
        "primary_cost_contrast": reduction["primary_cost_contrast"],
        "rows": rebuilt,
    }


def causal_fixture() -> JsonDict:
    """Create compact private learning evidence for fail-closed reader tests."""

    labels = [0, 1, 0, 1]
    predictions = [
        {
            "event_id": f"event-{index}",
            "prediction_time": index,
            "label_available": False,
            "probability": 0.2 + 0.15 * index,
            "state_hash": "sha256:initial",
        }
        for index in range(4)
    ]
    release_payload = {
        "release_id": "release-0",
        "release_time": 8,
        "event_ids": [row["event_id"] for row in predictions],
        "labels": labels,
        "shuffle_origin_event_ids": ["event-1", "event-2", "event-3", "event-0"],
        "shuffled_labels": [1, 0, 1, 0],
        "origin_prediction_times": [1, 2, 3, 0],
        "changed_label_binding_count": 4,
    }
    states: list[JsonDict] = []
    updates: list[JsonDict] = []
    previous_hash = "sha256:initial"
    for index, event in enumerate(predictions):
        payload = {
            "event_id": event["event_id"],
            "previous_state_hash": previous_hash,
            "parameters": [0.30 + index * 0.01, 0.70 - index * 0.01],
            "primal_violation": 0.0,
            "optimizer_residual": 1e-12,
        }
        state_hash = canonical_hash(payload)
        states.append({**payload, "state_hash": state_hash})
        updates.append(
            {
                "event_id": event["event_id"],
                "release_id": "release-0",
                "update_time": 8 + index,
                "before_hash": previous_hash,
                "after_hash": state_hash,
            }
        )
        previous_hash = state_hash
    return {
        "settings": {"delay": 8, "release_block": 8, "minimum_shuffle_changes": 1},
        "predictions": predictions,
        "releases": [release_payload],
        "updates": updates,
        "states": states,
        "checkpoints": [
            {
                "arrival": 4,
                "persisted_state_hash": previous_hash,
                "reloaded_state_hash": previous_hash,
                "uninterrupted_state_hash": previous_hash,
            }
        ],
        "tail": {
            "event_ids": ["tail-0", "tail-1"],
            "expected_count": 2,
            "explicitly_unreleased": True,
        },
        "retention_rows": [
            {"arrival": 0, "evaluator_only": True, "label_returned": False},
            {"arrival": 4, "evaluator_only": True, "label_returned": False},
        ],
    }


def audit_causal_fixture(value: Mapping[str, Any]) -> JsonDict:
    """Independently verify chronology and durable constrained state evidence."""

    errors: list[str] = []
    predictions = value.get("predictions") or []
    releases = value.get("releases") or []
    updates = value.get("updates") or []
    states = value.get("states") or []
    by_event = {str(row.get("event_id")): row for row in predictions}
    if len(by_event) != len(predictions):
        errors.append("duplicate_prediction")
    released: dict[str, int] = {}
    for release in releases:
        release_time = int(release.get("release_time", -1))
        event_ids = [str(item) for item in release.get("event_ids") or []]
        labels = list(release.get("labels") or [])
        shuffled = list(release.get("shuffled_labels") or [])
        origins = list(release.get("origin_prediction_times") or [])
        if sorted(labels) != sorted(shuffled) or labels == shuffled:
            errors.append("unchanged_or_invalid_shuffle")
        changed = sum(a != b for a, b in zip(labels, shuffled, strict=True))
        if changed != release.get("changed_label_binding_count") or changed < int(
            (value.get("settings") or {}).get("minimum_shuffle_changes", 1)
        ):
            errors.append("shuffle_change_count_invalid")
        for event_id, origin in zip(event_ids, origins, strict=True):
            prediction = by_event.get(event_id)
            if prediction is None or int(prediction.get("prediction_time", -1)) >= release_time:
                errors.append(f"prediction_not_before_release:{event_id}")
            if int(origin) >= release_time:
                errors.append(f"future_origin_feedback:{event_id}")
            if prediction is not None and prediction.get("label_available") is not False:
                errors.append(f"label_visible_at_prediction:{event_id}")
            released[event_id] = release_time
    update_counts: dict[str, int] = {}
    for row in updates:
        event_id = str(row.get("event_id") or "")
        update_counts[event_id] = update_counts.get(event_id, 0) + 1
        if event_id not in released or int(row.get("update_time", -1)) < released.get(event_id, 0):
            errors.append(f"update_before_release:{event_id}")
    if set(update_counts) != set(released) or any(count != 1 for count in update_counts.values()):
        errors.append("exactly_once_update_invalid")
    previous = "sha256:initial"
    for row in states:
        payload = {
            key: row.get(key)
            for key in (
                "event_id",
                "previous_state_hash",
                "parameters",
                "primal_violation",
                "optimizer_residual",
            )
        }
        if row.get("previous_state_hash") != previous or row.get("state_hash") != canonical_hash(
            payload
        ):
            errors.append("state_hash_chain_invalid")
        if float(row.get("primal_violation", math.inf)) > METRIC_TOLERANCE:
            errors.append("primal_constraint_failed")
        if float(row.get("optimizer_residual", math.inf)) > PROBABILITY_TOLERANCE:
            errors.append("optimizer_residual_failed")
        previous = str(row.get("state_hash"))
    for row in value.get("checkpoints") or []:
        hashes = {
            row.get("persisted_state_hash"),
            row.get("reloaded_state_hash"),
            row.get("uninterrupted_state_hash"),
        }
        if len(hashes) != 1 or previous not in hashes:
            errors.append("checkpoint_restart_mismatch")
    tail = value.get("tail") or {}
    tail_ids = list(tail.get("event_ids") or [])
    if tail.get("explicitly_unreleased") is not True or len(tail_ids) != tail.get("expected_count"):
        errors.append("missing_tail")
    if set(tail_ids) & set(update_counts):
        errors.append("tail_updated")  # pragma: no cover - extra defensive condition.
    retention = value.get("retention_rows") or []
    if not retention or any(
        row.get("evaluator_only") is not True or row.get("label_returned") is not False
        for row in retention
    ):
        errors.append("retention_isolation_failed")
    unique = list(dict.fromkeys(errors))
    return {
        "errors": unique,
        "chronology_passed": not any("release" in error or "visible" in error for error in unique),
        "shuffle_passed": not any("shuffle" in error for error in unique),
        "restart_passed": "checkpoint_restart_mismatch" not in unique,
        "tail_passed": "missing_tail" not in unique and "tail_updated" not in unique,
        "retention_passed": "retention_isolation_failed" not in unique,
        "primal_constraints_passed": "primal_constraint_failed" not in unique,
        "optimizer_residuals_passed": "optimizer_residual_failed" not in unique,
    }


def _fixture_heads() -> JsonDict:
    """Create small frozen heads for private source-mutation checks."""

    knots = [[-3.0] * 4 + [-1.0, 0.0, 1.0, 2.0] + [3.0] * 4 for _ in range(3)]
    specifications = {
        CANDIDATE_ARM: ([0.04] * 24, 0.10),
        "original_only_local_basis_energy": ([0.03] * 8, 0.16),
        "unconstrained_equal_capacity_energy": ([0.01] * 24, 0.14),
        "same_information_logistic": ([0.2, -0.1, 0.1], 0.15),
    }
    selected: JsonDict = {}
    for arm, (weights, tune_brier) in specifications.items():
        checkpoint = {"bias": 0.0, "weights": weights}
        selected[arm] = {
            "checkpoint": checkpoint,
            "checkpoint_sha256": canonical_hash(checkpoint),
            "tune_brier": tune_brier,
        }
    return {
        "normalization": {"mean": [0.0, 0.0, 0.0], "safe_scale": [1.0, 1.0, 1.0]},
        "local_basis": {"knots": knots},
        "selected_heads": selected,
        "temperature_baseline": {"selected_temperature": 0.5, "tune_brier": 0.12},
        "strongest_comparator": {"family": "temperature_original"},
    }


def static_fixture() -> JsonDict:
    """Build compact private source rows without using producer implementations."""

    heads = _fixture_heads()
    features: list[JsonDict] = []
    labels: list[JsonDict] = []
    for index in range(4):
        first = [-1.0 + index * 0.6, 0.3, -0.2]
        second = [-0.8 + index * 0.5, 0.2, -0.1]
        component = f"fixture-component-{index}"
        features.append(
            {
                "component_hash": component,
                "group_hash": f"fixture-group-{index}",
                "role": "test",
                "tool_type": "fixture",
                "feature_views": {
                    "supported_first": first,
                    "unsupported_first": second,
                },
                "original_unsupported_probability": (sigmoid(first[0]) + sigmoid(second[0])) / 2.0,
            }
        )
        labels.append({"component_hash": component, "role": "test", "label": index % 2})
    rebuilt, errors = _reconstruct_rows(features, labels, heads)
    if errors:  # pragma: no cover - fixture construction invariant.
        raise AssertionError(errors)
    freeze = "sha256:fixture-predictions"
    published = [
        {
            **row,
            "head_identity": _expected_head_identity(str(row["arm"]), heads),
            "prediction_freeze_sha256": freeze,
        }
        for row in rebuilt
    ]
    reduction = _static_reduction(rebuilt, 16)
    report = {
        "prediction_sidecar": {"sha256": freeze},
        "reduction": {
            **reduction,
            "probability_benefit_score": 0,
            "decision_benefit_score": 0,
        },
    }
    return {
        "features": features,
        "labels": labels,
        "heads": heads,
        "published_rows": published,
        "report": report,
    }


def run_private_mutations() -> list[JsonDict]:
    """Prove all eight private corruptions fail without publishing their bytes."""

    output: list[JsonDict] = []
    for name in MUTATION_NAMES:
        errors: list[str] = []
        if name in {
            "deleted_source_row",
            "duplicate_component",
            "wrong_option_orientation",
            "favorable_aggregate_contradictory_rows",
        }:
            fixture = static_fixture()
            if name == "deleted_source_row":
                fixture["features"].pop()
                _rows, errors = _reconstruct_rows(
                    fixture["features"], fixture["labels"], fixture["heads"]
                )
            elif name == "duplicate_component":
                fixture["features"].append(deepcopy(fixture["features"][0]))
                _rows, errors = _reconstruct_rows(
                    fixture["features"], fixture["labels"], fixture["heads"]
                )
            elif name == "wrong_option_orientation":
                views = fixture["features"][0]["feature_views"]
                fixture["features"][0]["feature_views"] = {
                    "unsupported_first": views["unsupported_first"],
                    "supported_first": views["supported_first"],
                }
                _rows, errors = _reconstruct_rows(
                    fixture["features"], fixture["labels"], fixture["heads"]
                )
            else:
                expected = _static_reduction(
                    [
                        {
                            key: row[key]
                            for key in (
                                "source_component_hash",
                                "arm",
                                "brier",
                                "log_loss",
                                "realized_cost",
                                "action",
                            )
                        }
                        for row in fixture["published_rows"]
                    ],
                    16,
                )
                changed = deepcopy(expected)
                changed["probability_metrics"][CANDIDATE_ARM]["brier"] = 0.0
                if _nested_error(expected, changed) > METRIC_TOLERANCE:
                    errors = ["producer_aggregate_mismatch"]
        else:
            fixture = causal_fixture()
            if name == "future_origin_feedback":
                fixture["releases"][0]["origin_prediction_times"][0] = 99
            elif name == "unchanged_shuffle":
                fixture["releases"][0]["shuffled_labels"] = list(fixture["releases"][0]["labels"])
                fixture["releases"][0]["changed_label_binding_count"] = 0
            elif name == "altered_checkpoint":
                fixture["checkpoints"][0]["reloaded_state_hash"] = "sha256:altered"
            elif name == "missing_tail":
                fixture["tail"]["event_ids"].pop()
            errors = audit_causal_fixture(fixture)["errors"]
        output.append(
            {
                "mutation": name,
                "passed": bool(errors),
                "observed_errors": errors,
                "corrupted_fixture_published": False,
            }
        )
    return output


REQUIRED_FIELDS = (
    "experiment_id",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_specs",
    "model_invoked",
    "invocation_counts",
    "inference_substrate_class",
    "inference_substrate",
    "execution_venue",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "honest_verdict",
    "verdict_class",
    "verifier_is_oracle",
    "flagged_adversarial",
    "validation_receipts",
    "field_principles",
    "source_claims_qualified_score",
    "learning_claims_qualified_score",
    "qualified_source_benefit_score",
    "qualified_learning_benefit_score",
    "branch_dispositions",
    "mutation_rows",
)


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    op: str,
    passed: bool,
) -> JsonDict:
    """Attach one exact operand and its claim-boundary principle."""

    principles = {
        "validity": "Invalid evidence cannot support science.",
        "readiness": "A valid null remains reusable.",
        "benefit": "Completion cannot substitute for empirical value.",
    }
    return {
        "check": check,
        "category": category,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": op,
        "passed": passed,
        "principle": principles[category],
    }


def _gate_summary(
    gates: Sequence[Mapping[str, Any]], inventory: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Name the exact upstream operand behind the first failed gate."""

    failed = [row for row in gates if row.get("passed") is not True]
    learning = next((row for row in inventory if row.get("producer_id") == "exp7568"), {})
    exact = learning.get("gate_failure") if learning.get("state") == "blocked_pre_gate" else None
    first = failed[0] if failed else None
    return {
        "passed": not failed,
        "failed_count": len(failed),
        "failed_checks": [row.get("check") for row in failed],
        "first_failure": (
            {
                "check": first.get("check") if first else None,
                "upstream": exact.get("upstream") if isinstance(exact, Mapping) else None,
                "path": exact.get("path") if isinstance(exact, Mapping) else None,
                "field": exact.get("field") if isinstance(exact, Mapping) else None,
                "expected": exact.get("expected")
                if isinstance(exact, Mapping)
                else first.get("expected"),
                "observed": exact.get("observed")
                if isinstance(exact, Mapping)
                else first.get("observed"),
                "op": exact.get("op") if isinstance(exact, Mapping) else first.get("op"),
                "category": first.get("category") if first else None,
            }
            if first
            else None
        ),
    }


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain the silent failure that each emitted field prevents."""

    specific = {
        "experiment_id": "Exact identity prevents cross-task substitution.",
        "preconditions_checked": "Observed resources prevent fabricated fallback data.",
        "MODEL_SPECS": "An empty plan prevents historical model work becoming current work.",
        "model_specs": "No resolved model identity is valid because this audit loads no model.",
        "model_invoked": "The false flag prevents aggregation from being called inference.",
        "invocation_counts": "Typed zero counters expose hidden model calls.",
        "inference_substrate_class": "The aggregation class applies the correct runtime floor.",
        "inference_substrate": "The substrate separates byte reduction from generation.",
        "execution_venue": "A legal venue keeps CPU identity in its separate field.",
        "duration_s": "Monotonic duration prevents invented runtime claims.",
        "random_seed": "Frozen seeds prevent outcome-dependent resampling.",
        "reproducibility_checksum": "The checksum binds settings, evidence, code, and rows.",
        "rows": "Every producer disposition remains visible; missing is not zero.",
        "sample_size_budget": "Disposition counts prevent silent attrition.",
        "acceptance_gate_results": "Typed gates keep validity, readiness, and benefit separate.",
        "gate_check_summary": "Exact failed operands make external blocks reproducible.",
        "honest_verdict": "A terminal prefix prevents an external block from retrying as partial.",
        "verdict_class": "The closed class prevents blocked evidence from becoming null.",
        "verifier_is_oracle": "Probabilistic energy cannot certify source truth.",
        "flagged_adversarial": "Recorded safety determinations cannot be erased to open a gate.",
        "validation_receipts": "Exact command outcomes prevent unrun checks appearing green.",
        "source_claims_qualified_score": "Raw reconstruction can qualify an honest source null.",
        "learning_claims_qualified_score": "Causal claims require complete online replay.",
        "qualified_source_benefit_score": "Source benefit cannot exceed reconstructed producer gates.",
        "qualified_learning_benefit_score": "Learning benefit requires fresh effect and retention gates.",
        "branch_dispositions": "One blocked branch cannot erase independent completed work.",
        "mutation_rows": "Private corruptions must fail without changing public records.",
        "field_principles": "Every field names the omission or drift it prevents.",
    }
    return {
        key: specific.get(key, f"Retaining {key} prevents silent omission or scope drift.")
        for key in keys
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable content while excluding clocks and this self-reference."""

    excluded = {
        "reproducibility_checksum",
        "duration_s",
        "phase_spans",
        "process_identity",
        "started_utc",
        "finished_utc",
    }
    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key not in excluded}
    )


def _receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one successful receipt for every affected and terminal command."""

    expected = {*validation_scope.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES}
    by_name: dict[str, list[Mapping[str, Any]]] = {}
    for row in receipts:
        by_name.setdefault(str(row.get("name") or ""), []).append(row)
    return all(
        len(by_name.get(name, [])) == 1
        and by_name[name][0].get("passed") is True
        and by_name[name][0].get("timed_out") is not True
        for name in expected
    )


def _artifact_gates(
    inventory: Sequence[Mapping[str, Any]],
    static: Mapping[str, Any],
    mutations: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Keep branch readiness and benefit as separate exact operands."""

    learning = next(row for row in inventory if row.get("producer_id") == "exp7568")
    source_qualified = static.get("source_claims_qualified") is True
    probability_benefit = static.get("producer_probability_benefit_score")
    decision_benefit = static.get("producer_decision_benefit_score")
    return [
        _gate(
            "learning_producer_available",
            "readiness",
            "present",
            learning.get("state"),
            "==",
            learning.get("state") == "present",
        ),
        _gate(
            "source_raw_reconstruction",
            "validity",
            True,
            source_qualified,
            "==",
            source_qualified,
        ),
        _gate(
            "private_corruptions_fail_closed",
            "validity",
            len(MUTATION_NAMES),
            sum(row.get("passed") is True for row in mutations),
            "==",
            all(row.get("passed") is True for row in mutations),
        ),
        _gate(
            "required_validation",
            "validity",
            True,
            _receipts_pass(receipts),
            "==",
            _receipts_pass(receipts),
        ),
        _gate(
            "source_probability_benefit",
            "benefit",
            1,
            probability_benefit,
            "==",
            probability_benefit == 1,
        ),
        _gate(
            "source_decision_benefit",
            "benefit",
            1,
            decision_benefit,
            "==",
            decision_benefit == 1,
        ),
        _gate("learning_effect_and_retention", "benefit", 1, None, "==", False),
    ]


def _sample_budget(static: Mapping[str, Any], learning_present: bool) -> JsonDict:
    """Record source and learning unit disposition without mapping missing to zero."""

    completed = int(static.get("source_count", 0))
    return {
        "unit": "independent_source_component_or_learning_producer",
        "planned": completed + 1,
        "attempted": completed + int(learning_present),
        "completed": completed,
        "excluded": 0,
        "failed": 0,
        "censored": 0,
        "unstarted": 0 if learning_present else 1,
        "source": {
            "planned": completed,
            "attempted": completed,
            "completed": completed,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 0,
        },
        "learning": {
            "planned": 1,
            "attempted": int(learning_present),
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 0 if learning_present else 1,
        },
    }


def _build_artifact(
    *,
    root: Path,
    preconditions: Mapping[str, Any],
    static: Mapping[str, Any],
    mutations: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    evidence_mode: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]] = (),
    started_utc: str | None = None,
    finished_utc: str | None = None,
) -> JsonDict:
    """Assemble one terminal record while retaining both branch dispositions."""

    inventory = list(preconditions.get("inventory") or [])
    source_rows = list(static.get("rows") or [])
    rows = [{"row_kind": "producer_inventory", **deepcopy(row)} for row in inventory] + [
        {"row_kind": "static_source_arm", **deepcopy(row)} for row in source_rows
    ]
    static_summary = {key: deepcopy(value) for key, value in static.items() if key != "rows"}
    gates = _artifact_gates(inventory, static, mutations, validation_receipts)
    source_qualified = static.get("source_claims_qualified") is True
    source_benefit = bool(
        source_qualified
        and static.get("producer_probability_benefit_score") == 1
        and static.get("producer_decision_benefit_score") == 1
    )
    learning = next(row for row in inventory if row.get("producer_id") == "exp7568")
    value: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "worktree_root": str(root.resolve()),
        "evidence_mode": evidence_mode,
        "preconditions_checked": deepcopy(preconditions.get("rows") or []),
        "source_artifact_hashes": deepcopy(preconditions.get("source_artifact_hashes") or []),
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_work": {
            "counted_as_current": False,
            "custody": "retained_in_referenced_producer_artifacts",
        },
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "execution_venue": "host",
        "cpu_device": f"{platform.system()} {platform.machine()}".strip(),
        "duration_s": duration_s,
        "phase_spans": deepcopy(list(phase_spans)),
        "process_identity": {
            "pid": os.getpid(),
            "python": sys.executable,
            "worktree": str(root.resolve()),
        },
        "random_seed": {
            "model": None,
            "fitting": 7_566_001,
            "ordering": 659_033,
            "bootstrap": BOOTSTRAP_SEED,
        },
        "rows": rows,
        "producer_inventory": deepcopy(inventory),
        "static_reconstruction": static_summary,
        "sample_size_budget": _sample_budget(static, learning.get("state") == "present"),
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates, inventory),
        "honest_verdict": "complete_blocked_learning_external_source_null",
        "verdict_class": "blocked",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "adversarial_determinations": [
            {
                "producer_id": row.get("producer_id"),
                "flagged_adversarial": row.get("flagged_adversarial"),
            }
            for row in inventory
        ],
        "validation_receipts": deepcopy(list(validation_receipts)),
        "validation_summary": {
            "required_checks_passed": _receipts_pass(validation_receipts),
            "scoped_only": True,
            "unscoped_suite_run": False,
        },
        "source_claims_qualified_score": int(source_qualified),
        "learning_claims_qualified_score": 0,
        "qualified_source_benefit_score": int(source_benefit),
        "qualified_learning_benefit_score": 0,
        "branch_dispositions": {
            "source": {
                "verdict_class": (
                    "positive" if source_benefit else "null" if source_qualified else "disqualified"
                ),
                "claims_qualified_score": int(source_qualified),
                "benefit_score": int(source_benefit),
            },
            "learning": {
                "verdict_class": "blocked",
                "claims_qualified_score": 0,
                "benefit_score": 0,
                "producer_state": learning.get("state"),
                "gate_failure": deepcopy(learning.get("gate_failure")),
            },
        },
        "mutation_rows": deepcopy(list(mutations)),
    }
    value["field_principles"] = _field_principles(
        [*value, "field_principles", "reproducibility_checksum"]
    )
    value["reproducibility_checksum"] = reproducibility_checksum(value)
    return value


def build_test_artifact(
    root: Path, *, validation_receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Build one compact blocked artifact for pure schema and reader tests."""

    fixture = static_fixture()
    reduction = _static_reduction(fixture["published_rows"], 16)
    static: JsonDict = {
        "errors": [],
        "source_count": reduction["source_count"],
        "row_count": reduction["row_count"],
        "strongest_comparator": "temperature_original",
        "max_probability_error": 0.0,
        "max_metric_error": 0.0,
        "probability_tolerance": PROBABILITY_TOLERANCE,
        "metric_tolerance": METRIC_TOLERANCE,
        "floating_point_uncertainty": {
            "probability": "private deterministic fixture",
            "metric": "private deterministic fixture",
        },
        "producer_agreement": True,
        "source_claims_qualified": True,
        "producer_probability_benefit_score": 0,
        "producer_decision_benefit_score": 0,
        **reduction,
        "rows": fixture["published_rows"],
    }
    inventory = [
        {
            "producer_id": producer_id,
            "declared_path": PRODUCER_PATHS[producer_id].as_posix(),
            "evidence_path": PRODUCER_PATHS[producer_id].as_posix(),
            "state": "present",
            "sha256": "sha256:" + str(index + 1) * 64,
            "honest_verdict": "complete_null_private_fixture",
            "verdict_class": "null",
            "flagged_adversarial": False,
            "gate_failure": None,
        }
        for index, producer_id in enumerate(PRODUCER_IDS[:-1])
    ]
    inventory.append(
        {
            "producer_id": "exp7568",
            "declared_path": PRODUCER_PATHS["exp7568"].as_posix(),
            "evidence_path": DIAGNOSTIC_PATHS["exp7568"].as_posix(),
            "state": "blocked_pre_gate",
            "sha256": "sha256:" + "6" * 64,
            "honest_verdict": "blocked_gate_check_failed",
            "verdict_class": "blocked",
            "flagged_adversarial": None,
            "gate_failure": {
                "upstream": "exp7561-recalibration-prototype",
                "path": str(root / "results/experiment_7561_v661_recalibration_prototype.json"),
                "field": "recalibration_ready_score",
                "op": "==",
                "expected": 1,
                "observed": 0,
            },
        }
    )
    preconditions = {
        "rows": [
            _check(
                "private_fixture",
                "test",
                "in_memory",
                "valid",
                "valid",
            )
        ],
        "inventory": inventory,
        "source_artifact_hashes": [],
    }
    return _build_artifact(
        root=root,
        preconditions=preconditions,
        static=static,
        mutations=run_private_mutations(),
        validation_receipts=validation_receipts,
        evidence_mode="private_fixture",
        duration_s=0.1,
    )


def _verify_source_hashes(value: Mapping[str, Any], root: Path) -> list[str]:
    """Re-authenticate every declared input byte without repairing a mismatch."""

    errors: list[str] = []
    for row in value.get("source_artifact_hashes") or []:
        path = _resolve(root, str(row.get("path") or ""))
        if not path.is_file():
            errors.append(f"source_missing:{row.get('path')}")
        elif sha256_file(path) != row.get("sha256"):
            errors.append(f"source_hash_mismatch:{row.get('path')}")
        elif path.stat().st_size != row.get("bytes"):
            errors.append(f"source_size_mismatch:{row.get('path')}")
    return errors


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, verify_sources: bool = True
) -> list[str]:
    """Reject identity, custody, branch, score, receipt, row, or checksum drift."""

    errors: list[str] = []
    if not isinstance(value, Mapping):  # pragma: no cover - typed callers pass mappings.
        return ["artifact_not_object"]
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("artifact_identity_mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("task_binding_mismatch")
    missing = [field for field in REQUIRED_FIELDS if field not in value]
    if missing:
        errors.append("required_fields_missing:" + ",".join(missing))
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        errors.append("model_specs_not_empty")
    if value.get("model_invoked") is not False:
        errors.append("model_invocation_mismatch")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("invocation_counts_mismatch")
    if (
        value.get("inference_substrate_class") != "aggregation"
        or value.get("inference_substrate") != "aggregation_from_upstream_artifacts"
    ):
        errors.append("inference_substrate_mismatch")
    if value.get("execution_venue") != "host":
        errors.append("execution_venue_mismatch")
    inventory = value.get("producer_inventory") or []
    expected_rows = [{"row_kind": "producer_inventory", **deepcopy(row)} for row in inventory] + [
        deepcopy(row)
        for row in value.get("rows") or []
        if row.get("row_kind") == "static_source_arm"
    ]
    if list(value.get("rows") or []) != expected_rows:
        errors.append("producer_rows_mismatch")
    source = value.get("static_reconstruction") or {}
    expected_source_qualified = int(
        source.get("source_claims_qualified") is True and not source.get("errors")
    )
    if value.get("source_claims_qualified_score") != expected_source_qualified:
        errors.append("source_claims_score_mismatch")
    source_benefit = int(
        expected_source_qualified == 1
        and source.get("producer_probability_benefit_score") == 1
        and source.get("producer_decision_benefit_score") == 1
    )
    if value.get("qualified_source_benefit_score") != source_benefit:
        errors.append("source_benefit_score_mismatch")
    if (
        value.get("learning_claims_qualified_score") != 0
        or value.get("qualified_learning_benefit_score") != 0
    ):
        errors.append("learning_score_mismatch")
    if value.get("verdict_class") != "blocked" or not str(
        value.get("honest_verdict") or ""
    ).startswith("complete_blocked_"):
        errors.append("terminal_verdict_mismatch")
    if not _receipts_pass(value.get("validation_receipts") or []):
        errors.append("required_validation_failed")
    mutations = value.get("mutation_rows") or []
    if [row.get("mutation") for row in mutations] != list(MUTATION_NAMES) or not all(
        row.get("passed") is True and row.get("corrupted_fixture_published") is False
        for row in mutations
    ):
        errors.append("mutation_contract_failed")
    if set(value) - set(value.get("field_principles") or {}):
        errors.append("field_principles_incomplete")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    if verify_sources:
        errors.extend(_verify_source_hashes(value, root))
    return list(dict.fromkeys(errors))


def cold_replay(path: Path, *, root: Path = REPO_ROOT, verify_sources: bool = True) -> list[str]:
    """Read serialized bytes through the public defensive artifact validator."""

    value = load_json(path)
    if not value:
        return ["artifact_not_object"]
    return validate_artifact(value, root=root, verify_sources=verify_sources)


def _embedded_static_rows(value: Mapping[str, Any]) -> list[JsonDict]:
    """Recover source-arm rows without importing a producer summary function."""

    return [
        {key: deepcopy(item) for key, item in row.items() if key != "row_kind"}
        for row in value.get("rows") or []
        if row.get("row_kind") == "static_source_arm"
    ]


def independent_replay(path: Path) -> list[str]:
    """Re-reduce source rows and, for real evidence, sealed raw features."""

    value = load_json(path)
    if not value:
        return ["artifact_not_object"]
    root = Path(str(value.get("worktree_root") or REPO_ROOT)).resolve()
    verify_sources = value.get("evidence_mode") == "sealed_real"
    errors = validate_artifact(value, root=root, verify_sources=verify_sources)
    rows = _embedded_static_rows(value)
    summary = value.get("static_reconstruction") or {}
    intervals = (summary.get("paired_intervals") or {}).get("brier") or {}
    first_interval = next(iter(intervals.values()), {})
    draws = int(first_interval.get("draws", BOOTSTRAP_DRAWS))
    try:
        reduced = _static_reduction(rows, draws)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"embedded_source_reduction_failed:{type(exc).__name__}")
    else:
        for key in ("probability_metrics", "paired_intervals", "primary_cost_contrast"):
            if _nested_error(reduced[key], summary.get(key)) > METRIC_TOLERANCE:
                errors.append(f"embedded_source_reduction_mismatch:{key}")
    if verify_sources:
        try:
            reconstructed = reconstruct_static(**load_static_inputs(root), draws=BOOTSTRAP_DRAWS)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            errors.append(f"sealed_source_replay_failed:{type(exc).__name__}")
        else:
            if reconstructed.get("errors"):
                errors.append("sealed_source_replay_errors")  # pragma: no cover
            if _nested_error(reconstructed.get("rows"), rows) > METRIC_TOLERANCE:
                errors.append("sealed_source_rows_mismatch")
            for key in ("probability_metrics", "paired_intervals", "primary_cost_contrast"):
                if _nested_error(reconstructed.get(key), summary.get(key)) > METRIC_TOLERANCE:
                    errors.append(f"sealed_source_reduction_mismatch:{key}")
    return list(dict.fromkeys(errors))


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze serial tests, changed-module coverage, lint, type, and spec checks."""

    commands = build_command_plan(root, AFFECTED_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, AFFECTED_MANIFEST, commands)
    if plan_errors:  # pragma: no cover - shared builder produces a frozen valid plan.
        raise ValueError("validation_plan_invalid:" + ",".join(plan_errors))
    return commands


def terminal_commands(candidate: Path, root: Path = REPO_ROOT) -> list[PlannedCommand]:
    """Build four bounded checks over the exact terminal candidate bytes."""

    python = ".venv/bin/python"
    common = ("--date", RUN_DATE, "--root", str(root.resolve()))
    specifications = (
        validation_scope.CommandSpec(
            TERMINAL_CHECK_NAMES[0],
            (python, "-u", WRAPPER_PATH.as_posix(), *common, "--cold-replay", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            TERMINAL_CHECK_NAMES[1],
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--independent-reduce",
                str(candidate),
            ),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            TERMINAL_CHECK_NAMES[2],
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            TERMINAL_CHECK_NAMES[3],
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact_terminal_candidate",
            300.0,
        ),
    )
    return [PlannedCommand(spec, "validity", True) for spec in specifications]


def progress(  # pragma: no cover - visible only through the declared entrypoint.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Flush every boundary and slow-operation edge with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7569] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(  # pragma: no cover - monotonic clock belongs to the entrypoint.
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:
    """Close one disjoint current-work interval with its completed unit count."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_offset_s": phase_started - run_started,
        "end_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
    }


def _write_validation_manifest(path: Path) -> None:  # pragma: no cover - entrypoint helper.
    """Freeze exact affected paths before expanding any subprocess command."""

    atomic_json(
        path,
        {
            "experiment_id": EXPERIMENT_ID,
            "test_paths": list(AFFECTED_MANIFEST.test_paths),
            "changed_modules": list(AFFECTED_MANIFEST.changed_modules),
            "static_paths": list(AFFECTED_MANIFEST.static_paths),
        },
    )


def _provisional_terminal_receipts() -> list[JsonDict]:  # pragma: no cover - entrypoint helper.
    """Give a private candidate its final reader shape before real commands run."""

    return [
        {
            "name": name,
            "command": "pending exact terminal candidate check",
            "command_argv": ["pending", name],
            "scope": "private_provisional_candidate",
            "exit_code": 0,
            "duration_s": 0.0,
            "log_path": "pending",
            "log_sha256": "sha256:" + "0" * 64,
            "passed": True,
            "timed_out": False,
            "output_tail": "private provisional receipt; never published",
            "command_category": "validity",
            "required": True,
        }
        for name in TERMINAL_CHECK_NAMES
    ]


def _append_owned_hashes(
    root: Path, preconditions: JsonDict, manifest_path: Path
) -> None:  # pragma: no cover - entrypoint helper.
    """Bind implementation, tests, wrapper, and frozen manifest after they exist."""

    for relative, role in (
        (MODULE_PATH, "implementation"),
        (WRAPPER_PATH, "entrypoint"),
        (TEST_PATH, "tests"),
        (manifest_path.relative_to(root), "affected_validation_manifest"),
    ):
        path = root / relative
        preconditions["source_artifact_hashes"].append(
            {**sidecar_reference(path, root), "role": role}
        )


def _empty_static(reason: str) -> JsonDict:  # pragma: no cover - entrypoint helper.
    """Represent an external block without inventing a source measurement."""

    return {
        "errors": [reason],
        "source_count": 0,
        "row_count": 0,
        "strongest_comparator": None,
        "max_probability_error": None,
        "max_metric_error": None,
        "probability_tolerance": PROBABILITY_TOLERANCE,
        "metric_tolerance": METRIC_TOLERANCE,
        "producer_agreement": False,
        "source_claims_qualified": False,
        "producer_probability_benefit_score": 0,
        "producer_decision_benefit_score": 0,
        "probability_metrics": {},
        "paired_intervals": {"brier": {}, "log_loss": {}},
        "primary_cost_contrast": {},
        "rows": [],
    }


def run_experiment(  # pragma: no cover - exercised by the declared capability E2E.
    root: Path, run_date: str, *, output_path: Path | None = None
) -> JsonDict:
    """Authenticate, reconstruct, validate, and atomically publish exact bytes."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    root = root.resolve()
    destination = output_path or root / RESULT_PATH
    raw_root = root / RAW_DIR
    candidate_path = root / TERMINAL_CANDIDATE_PATH
    started = time.monotonic()
    started_utc = datetime.now(UTC).isoformat()
    spans: list[JsonDict] = []

    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    preconditions = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, started, len(preconditions["rows"])))
    failed = next(
        (
            row
            for row in preconditions["rows"]
            if row.get("required") is True and row.get("passed") is not True
        ),
        None,
    )
    progress(
        started,
        "preconditions",
        "complete",
        completed_units=len(preconditions["rows"]),
        failed=int(failed is not None),
    )
    if failed is not None:
        blocked = _build_artifact(
            root=root,
            preconditions=preconditions,
            static=_empty_static(str(failed.get("check"))),
            mutations=[],
            validation_receipts=[],
            evidence_mode="external_precondition_block",
            duration_s=time.monotonic() - started,
            phase_spans=spans,
            started_utc=started_utc,
            finished_utc=datetime.now(UTC).isoformat(),
        )
        progress(started, "publish", "before_atomic_blocked")
        atomic_json(destination, blocked)
        progress(started, "publish", "complete_blocked")
        return blocked

    progress(started, "manifest", "start")
    phase_started = time.monotonic()
    manifest_path = root / VALIDATION_MANIFEST_PATH
    _write_validation_manifest(manifest_path)
    _append_owned_hashes(root, preconditions, manifest_path)
    spans.append(_span("manifest", phase_started, started, 1))
    progress(started, "manifest", "complete", completed_units=1)

    progress(started, "source_reconstruction", "before_benchmark", draws=BOOTSTRAP_DRAWS)
    phase_started = time.monotonic()
    static = reconstruct_static(**load_static_inputs(root), draws=BOOTSTRAP_DRAWS)
    spans.append(_span("source_reconstruction", phase_started, started, static["source_count"]))
    progress(
        started,
        "source_reconstruction",
        "after_benchmark",
        completed_units=static["source_count"],
        errors=len(static["errors"]),
    )

    progress(started, "private_mutations", "start", planned=len(MUTATION_NAMES))
    phase_started = time.monotonic()
    mutations = run_private_mutations()
    spans.append(_span("private_mutations", phase_started, started, len(mutations)))
    progress(
        started,
        "private_mutations",
        "complete",
        completed_units=len(mutations),
        passed=sum(row["passed"] is True for row in mutations),
    )

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7569-", dir="/tmp"))
    commands = build_validation_commands(root, private_root)
    planned = [PlannedCommand(command, "validity", True) for command in commands]
    progress(started, "scoped_validation", "before_subprocesses", planned=len(planned))
    phase_started = time.monotonic()
    affected_receipts = run_categorized_commands(
        root, planned, log_dir=raw_root / "validation_logs", heartbeat_s=60.0
    )
    affected_summary = reduce_affected_receipts(root, AFFECTED_MANIFEST, affected_receipts)
    spans.append(_span("scoped_validation", phase_started, started, len(affected_receipts)))
    progress(
        started,
        "scoped_validation",
        "after_subprocesses",
        completed_units=len(affected_receipts),
        passed=affected_summary["passed"],
    )
    if affected_summary["passed"] is not True:
        raise RuntimeError("required_scoped_validation_failed")

    provisional = _build_artifact(
        root=root,
        preconditions=preconditions,
        static=static,
        mutations=mutations,
        validation_receipts=[*affected_receipts, *_provisional_terminal_receipts()],
        evidence_mode="sealed_real",
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        started_utc=started_utc,
        finished_utc=datetime.now(UTC).isoformat(),
    )
    atomic_json(candidate_path, provisional)

    first_terminal_plan = terminal_commands(candidate_path, root)
    progress(
        started,
        "terminal_validation",
        "before_subprocesses",
        planned=len(first_terminal_plan),
    )
    phase_started = time.monotonic()
    terminal_receipts = run_categorized_commands(
        root,
        first_terminal_plan,
        log_dir=raw_root / "terminal_logs_provisional",
        heartbeat_s=60.0,
    )
    spans.append(_span("terminal_validation", phase_started, started, len(terminal_receipts)))
    if not all(row.get("passed") is True for row in terminal_receipts):
        raise RuntimeError("terminal_candidate_validation_failed")
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal_receipts),
        passed=True,
    )

    final = _build_artifact(
        root=root,
        preconditions=preconditions,
        static=static,
        mutations=mutations,
        validation_receipts=[*affected_receipts, *terminal_receipts],
        evidence_mode="sealed_real",
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        started_utc=started_utc,
        finished_utc=datetime.now(UTC).isoformat(),
    )
    final_errors = validate_artifact(final, root=root)
    if final_errors:
        raise RuntimeError("terminal_artifact_invalid:" + ",".join(final_errors))
    atomic_json(candidate_path, final)

    exact_plan = terminal_commands(candidate_path, root)
    progress(
        started,
        "exact_candidate_validation",
        "before_subprocesses",
        planned=len(exact_plan),
    )
    exact_receipts = run_categorized_commands(
        root,
        exact_plan,
        log_dir=raw_root / "terminal_logs_exact",
        heartbeat_s=60.0,
    )
    if not all(row.get("passed") is True for row in exact_receipts):
        raise RuntimeError("exact_terminal_candidate_validation_failed")
    progress(
        started,
        "exact_candidate_validation",
        "after_subprocesses",
        completed_units=len(exact_receipts),
        passed=True,
    )

    progress(started, "publish", "before_atomic_terminal")
    atomic_json(destination, final)
    if sha256_file(destination) != sha256_file(candidate_path):
        raise RuntimeError("published_bytes_differ_from_exact_candidate")
    progress(
        started,
        "publish",
        "complete_terminal",
        bytes=destination.stat().st_size,
        verdict=final["verdict_class"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed run date and two fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    arguments.root = arguments.root.resolve()
    return arguments


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary.
    """Run the audit or one exact read-only terminal validation mode."""

    arguments = parse_args(argv)
    if arguments.cold_replay is not None:
        errors = cold_replay(arguments.cold_replay, root=arguments.root)
        print(json.dumps({"mode": "cold_replay", "errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if arguments.independent_reduce is not None:
        errors = independent_replay(arguments.independent_reduce)
        print(
            json.dumps({"mode": "independent_reduction", "errors": errors}, sort_keys=True),
            flush=True,
        )
        return int(bool(errors))
    run_experiment(arguments.root, arguments.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

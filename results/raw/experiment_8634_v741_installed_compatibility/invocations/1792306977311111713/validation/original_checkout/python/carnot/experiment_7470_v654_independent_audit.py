"""Independently audit the three V654 evidence branches.

The reducer reads immutable producer bytes. It does not import a producer's
verdict reducer, fit a numeric head, or invoke an LLM. Missing producers stay
branch-local so an available extraction branch can still finish.

Spec refs: REQ-REPORT-7470 and SCENARIO-REPORT-7470-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260921"
MILESTONE = "2026.09.654"
EXPERIMENT_ID = "exp7470-v654-independent-audit"
SCHEMA = "carnot.exp7470.v654.independent_audit.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7470_v654_independent_audit.json")
RAW_DIR = Path("results/raw/experiment_7470_v654_independent_audit")
MODULE_PATH = Path("python/carnot/experiment_7470_v654_independent_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7470_v654_independent_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7470_v654_independent_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEEDS = {"audit": 6_547_470, "ordering": 6_547_471, "bootstrap": 6_547_472}
OPTIONS = ("accept", "reject", "escalate")
REQUIRED_MUTATIONS = (
    "option_mapping",
    "leaked_labels",
    "missing_failure_rows",
    "delayed_label_timestamp",
    "changed_checkpoint",
    "fabricated_improvement",
)
TERMINAL_CHECKS = (
    "declared_entrypoint_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


@dataclass(frozen=True)
class ProducerSpec:
    """Describe one producer without making its presence a global gate."""

    branch: str
    experiment_id: str
    path: Path
    conductor_path: Path


PRODUCERS = {
    "typed_decision": ProducerSpec(
        "typed_decision",
        "exp7466-v654-typed-energy-calibration",
        Path("results/experiment_7466_v654_typed_energy_calibration.json"),
        Path("results/experiment_7466_typed_energy_calibration.json"),
    ),
    "residual_learning": ProducerSpec(
        "residual_learning",
        "exp7469-v654-continuous-residual-learning",
        Path("results/experiment_7469_v654_continuous_residual_learning.json"),
        Path("results/experiment_7469_continuous_residual_learning.json"),
    ),
    "extraction": ProducerSpec(
        "extraction",
        "exp7467-v654-factual-span-canary",
        Path("results/experiment_7467_v654_factual_span_canary.json"),
        Path("results/experiment_7467_factual_span_canary.json"),
    ),
}

SOURCE_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7455_v653_decision_audit.py"),
    Path("python/carnot/experiment_7456_v653_extraction_audit.py"),
    Path("results/experiment_7455_v653_decision_audit.json"),
    Path("results/experiment_7456_v653_extraction_audit.json"),
    SPEC_PATH,
    Path("research-roadmap.yaml"),
)

MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

REQUIRED_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "started_at_utc",
    "completed_at_utc",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_specs",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
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
    "honest_verdict",
    "verdict_class",
    "verifier_is_oracle",
    "flagged_adversarial",
    "validation_receipts",
    "field_principles",
    "independent_audit_complete_score",
    "branch_rows",
    "independent_update_replay",
    "mutation_rows",
)

FIELD_PRINCIPLES = {
    "schema": "Use a versioned schema with exact experiment identity, milestone, and terminal status.",
    "run_date": "Use 20260921 with measured UTC and monotonic boundaries and a named clock.",
    "preconditions_checked": "Record exact paths, evidence ownership, device identity, and observed prerequisites.",
    "MODEL_SPECS": "Use an empty list because this audit performs no current LLM work.",
    "model_specs": "Repeat the empty model list for lowercase readers.",
    "model_invoked": "Keep current attempted calls separate from archived producer calls.",
    "invocation_counts": "Balance every current load and generation count at zero.",
    "inference_substrate": "Name aggregation from upstream artifacts for this pure reducer.",
    "inference_substrate_class": "Use aggregation because no model is loaded or fitted.",
    "execution_venue": "Use host and keep historical CUDA identity inside source receipts.",
    "duration_s": "Measure current work and separate computation from validation without padding.",
    "phase_spans": "Bind progress phases to monotonic times and completed-unit counts.",
    "random_seed": "Freeze audit, ordering, and bootstrap seeds even when a branch is absent.",
    "reproducibility_checksum": "Bind code, protocol, source bytes, rows, mutations, and validation scope.",
    "source_artifact_hashes": "Retain exact upstream bytes, original classes, and original flags.",
    "rows": "Keep every available development and evaluation unit, including unstarted calls.",
    "sample_size_budget": "Separate planned, attempted, completed, failed, censored, and unstarted units.",
    "acceptance_gate_results": "Keep validity, audit completion, and benefit gates in distinct categories.",
    "gate_check_summary": "Name each failed check and its exact upstream field or path.",
    "honest_verdict": "Complete the audit without turning branch absence or a scientific null into benefit.",
    "verdict_class": "Use a closed terminal class; external absence is never retryable partial work.",
    "verifier_is_oracle": "False because this audit checks evidence but supplies no natural semantic oracle.",
    "flagged_adversarial": "Retain source and current structural flags without clearing them for a gate.",
    "validation_receipts": "Capture exact scoped commands, exits, durations, environments, and log hashes.",
    "field_principles": "Explain why each field exists without wrapping its ordinary value.",
    "independent_audit_complete_score": "One means every branch and mutation has a disposition, not benefit.",
    "branch_rows": "Keep producer class, original flag, availability, validity, and benefit separate.",
    "independent_update_replay": "Use scalar event reduction so aggregate metrics cannot establish causality.",
    "mutation_rows": "Show that each registered corruption is caught by the independent reader.",
}


def principle_value(value: Any) -> Any:
    """Unwrap only the project's explicit principle wrapper shape.

    Ordinary evidence often uses dictionaries. Treating every dictionary as a
    wrapper discards fields and can silently rehabilitate an invalid producer.
    """

    if isinstance(value, Mapping) and "principle" in value and "value" in value:
        return value["value"]
    return value


def _load_object(path: Path) -> JsonDict:
    """Read one JSON object while keeping malformed external bytes explicit."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _text_hash(value: str) -> str:
    """Match the producer's typed hash of one JSON string value."""

    return canonical_hash(value)


def _reference(root: Path, path: Path, evidence_class: str, **metadata: Any) -> JsonDict:
    """Bind one readable file and preserve its evidence role."""

    resolved = path if path.is_absolute() else root / path
    try:
        label = resolved.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        label = str(resolved.resolve())
    return {
        "path": label,
        "sha256": sha256_file(resolved),
        "evidence_class": evidence_class,
        **deepcopy(metadata),
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    upstream: str,
    path: str,
    field: str,
    op: str = "==",
    passed: bool | None = None,
) -> JsonDict:
    """Build one plain gate row with a precise failure location."""

    if passed is None:
        passed = observed == expected if op == "==" else observed in expected
    return {
        "check": check,
        "category": category,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": op,
        "operator": op,
        "passed": bool(passed),
        "principle": "Keep branch validity separate from scientific benefit.",
        "upstream": upstream,
        "path": path,
        "field": field,
    }


def locate_producer(root: Path, spec: ProducerSpec) -> JsonDict:
    """Locate an exact producer or exact-identity pre-gate under any basename."""

    base = root.resolve()
    declared = base / spec.path
    candidates = [declared, base / spec.conductor_path]
    results = base / "results"
    if results.is_dir():
        candidates.extend(sorted(results.glob("*.json")))
    seen: set[Path] = set()
    invalid_declared: JsonDict | None = None
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        value = _load_object(resolved)
        identity = principle_value(value.get("experiment_id"))
        milestone = principle_value(value.get("milestone"))
        exact = identity == spec.experiment_id and milestone == MILESTONE
        if not exact:
            if resolved == declared.resolve():
                invalid_declared = {
                    "branch": spec.branch,
                    "availability": "invalid",
                    "path": spec.path.as_posix(),
                    "sha256": sha256_file(resolved),
                    "artifact": value,
                    "identity_errors": ["producer_identity_or_milestone_mismatch"],
                }
            continue
        availability = (
            "pre_gate" if value.get("blocked_at_layer") == "conductor_pre_gate" else "available"
        )
        return {
            "branch": spec.branch,
            "availability": availability,
            "path": resolved.relative_to(base).as_posix(),
            "sha256": sha256_file(resolved),
            "artifact": value,
            "identity_errors": [],
        }
    if invalid_declared is not None:
        return invalid_declared
    return {
        "branch": spec.branch,
        "availability": "missing",
        "path": spec.path.as_posix(),
        "sha256": None,
        "artifact": {},
        "identity_errors": ["producer_evidence_missing"],
    }


def softmax_by_option(logits: Sequence[float], option_order: Sequence[str]) -> dict[str, float]:
    """Restore option names and compute stable probabilities from raw logits."""

    if len(logits) != len(OPTIONS) or set(option_order) != set(OPTIONS):
        raise ValueError("option_mapping_invalid")
    numbers = [float(value) for value in logits]
    if any(not math.isfinite(value) for value in numbers):
        raise ValueError("raw_logit_invalid")
    maximum = max(numbers)
    exponentials = [math.exp(value - maximum) for value in numbers]
    denominator = math.fsum(exponentials)
    return {name: exponentials[index] / denominator for index, name in enumerate(option_order)}


def _checkpoint_hash(checkpoint: Mapping[str, Any]) -> str:
    """Hash checkpoint content without its self-reference."""

    return canonical_hash(
        {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
    )


def _fit_free_logits(
    features: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> dict[str, float]:
    """Score a saved linear checkpoint without fitting or importing producer code."""

    names = checkpoint.get("feature_names")
    weights = checkpoint.get("weights")
    bias = checkpoint.get("bias")
    if (
        not isinstance(names, list)
        or not isinstance(weights, Mapping)
        or not isinstance(bias, Mapping)
    ):
        raise ValueError("checkpoint_shape_invalid")
    vector = [float(features[name]) for name in names]
    return {
        option: float(bias[option])
        + math.fsum(
            float(weight) * value for weight, value in zip(weights[option], vector, strict=True)
        )
        for option in OPTIONS
    }


def audit_decision_rows(
    rows: Sequence[Mapping[str, Any]], checkpoint: Mapping[str, Any]
) -> JsonDict:
    """Independently remap logits, score a checkpoint, and reduce group Brier."""

    errors: list[str] = []
    if checkpoint.get("checkpoint_sha256") != _checkpoint_hash(checkpoint):
        errors.append("checkpoint_hash_mismatch")
    groups: set[str] = set()
    briers: list[float] = []
    full_source = True
    for row in rows:
        unit = str(row.get("unit_id") or "missing")
        group = str(row.get("group_id") or "")
        if not group or group in groups:
            errors.append(f"group_separation:{unit}")
        groups.add(group)
        features = row.get("features")
        if not isinstance(features, Mapping):
            errors.append(f"features_invalid:{unit}")
            continue
        for name in features:
            lowered = str(name).lower()
            if any(token in lowered for token in ("label", "gold", "outcome", "source_id")):
                errors.append(f"label_leakage:{unit}:{name}")
        try:
            observed = softmax_by_option(row.get("raw_logits") or [], row.get("option_order") or [])
            expected_logits = _fit_free_logits(features, checkpoint)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"decision_shape:{unit}:{exc}")
            continue
        mapped_logits = {
            option: float(
                (row.get("raw_logits") or [])[list(row.get("option_order") or []).index(option)]
            )
            for option in OPTIONS
        }
        if any(abs(mapped_logits[name] - expected_logits[name]) > 1e-12 for name in OPTIONS):
            errors.append(f"option_mapping_mismatch:{unit}")
        stored = row.get("stored_probabilities")
        if not isinstance(stored, Mapping) or any(
            abs(float(stored.get(name, -1.0)) - observed[name]) > 1e-12 for name in OPTIONS
        ):
            errors.append(f"probability_mismatch:{unit}")
        label = row.get("label")
        if label not in OPTIONS:
            errors.append(f"label_invalid:{unit}")
            continue
        briers.append(math.fsum((observed[name] - float(name == label)) ** 2 for name in OPTIONS))
        full_source = full_source and row.get("eligible") is True
        if row.get("source_swap_control") is not True:
            errors.append(f"source_swap_control_missing:{unit}")
        if not isinstance(row.get("all_escalate_cost"), (int, float)):
            errors.append(f"all_escalate_baseline_missing:{unit}")
    return {
        "errors": list(dict.fromkeys(errors)),
        "groups": len(groups),
        "rows": len(rows),
        "mean_brier": math.fsum(briers) / len(briers) if briers else None,
        "full_source_eligible": full_source and bool(rows),
        "multiplicity_checked": all("adjusted_p" in row for row in rows) if rows else False,
    }


def verify_declared_improvement(rows: Sequence[Mapping[str, Any]], *, declared: float) -> list[str]:
    """Reject an improvement that does not equal the raw all-escalate contrast."""

    checkpoint = {
        "feature_names": ["support", "missing"],
        "weights": {"accept": [1.0, 0.0], "reject": [-1.0, 0.0], "escalate": [0.0, 0.0]},
        "bias": {name: 0.0 for name in OPTIONS},
    }
    briers: list[float] = []
    baselines: list[float] = []
    for row in rows:
        probabilities = softmax_by_option(row["raw_logits"], row["option_order"])
        label = row["label"]
        briers.append(
            math.fsum((probabilities[name] - float(name == label)) ** 2 for name in OPTIONS)
        )
        baselines.append(
            math.fsum((float(name == "escalate") - float(name == label)) ** 2 for name in OPTIONS)
        )
    del checkpoint
    observed = math.fsum(briers) / len(briers) - math.fsum(baselines) / len(baselines)
    return [] if math.isclose(observed, declared, abs_tol=1e-12) else ["fabricated_improvement"]


def residual_state_hash(state: Mapping[str, Any]) -> str:
    """Hash only prediction-time scalar state and its seed boundary."""

    return canonical_hash(
        {
            "seed": int(state["seed"]),
            "weights": [float(value) for value in state["weights"]],
            "updates": int(state["updates"]),
        }
    )


def _sigmoid(value: float) -> float:
    """Compute a stable scalar logistic prediction."""

    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def replay_residual_updates(
    events: Sequence[Mapping[str, Any]], checkpoint: Mapping[str, Any]
) -> JsonDict:
    """Replay delayed scalar residual updates from prediction-time values."""

    errors: list[str] = []
    if checkpoint.get("checkpoint_sha256") != _checkpoint_hash(checkpoint):
        errors.append("checkpoint_hash_mismatch")
    state = {
        "seed": int(checkpoint["seed"]),
        "weights": [float(value) for value in checkpoint["weights"]],
        "updates": int(checkpoint["updates"]),
    }
    learning_rate = float(checkpoint["learning_rate"])
    predictions: dict[str, Mapping[str, Any]] = {}
    seen_feedback: set[str] = set()
    updates = 0
    retained = 0
    for event in events:
        kind = event.get("type")
        event_id = str(event.get("event_id") or "missing")
        if kind == "prediction":
            predictions[event_id] = event
            if event.get("seed") != state["seed"]:
                errors.append(f"seed_pooling:{event_id}")
            if event.get("state_hash_before") != residual_state_hash(state):
                errors.append(f"stale_state:{event_id}")
            vector = [float(value) for value in event.get("features") or []]
            probability = _sigmoid(
                math.fsum(a * b for a, b in zip(state["weights"], vector, strict=True))
            )
            if not math.isclose(probability, float(event.get("probability")), abs_tol=1e-12):
                errors.append(f"prediction_probability_mismatch:{event_id}")
        elif kind == "feedback":
            prediction_id = str(event.get("prediction_event_id") or "")
            if prediction_id in seen_feedback:
                errors.append(f"duplicate_feedback:{prediction_id}")
            seen_feedback.add(prediction_id)
            prediction = predictions.get(prediction_id)
            if prediction is None:
                errors.append(f"feedback_prediction_missing:{event_id}")
                continue
            if int(event.get("timestamp_ns", -1)) <= int(prediction.get("timestamp_ns", -1)):
                errors.append(f"label_before_prediction:{event_id}")
            if event.get("seed") != state["seed"]:
                errors.append(f"seed_pooling:{event_id}")
            if event.get("audit_selected") is not True or event.get("audit_probability") != 0.5:
                errors.append(f"unaudited_update:{event_id}")
            if checkpoint.get("frozen_control") is True:
                errors.append(f"nonfrozen_control:{event_id}")
            before = residual_state_hash(state)
            if event.get("state_hash_before") != before:
                errors.append(f"stale_state:{event_id}")
            probability = float(prediction["probability"])
            label = event.get("label")
            if label not in {0, 1} or isinstance(label, bool):
                errors.append(f"label_invalid:{event_id}")
                continue
            vector = [float(value) for value in prediction.get("features") or []]
            state["weights"] = [
                weight + learning_rate * (int(label) - probability) * feature
                for weight, feature in zip(state["weights"], vector, strict=True)
            ]
            state["updates"] += 1
            updates += 1
            if event.get("state_hash_after") != residual_state_hash(state):
                errors.append(f"state_hash_mismatch:{event_id}")
        elif kind == "retention":
            vector = [float(value) for value in event.get("features") or []]
            probability = _sigmoid(
                math.fsum(a * b for a, b in zip(state["weights"], vector, strict=True))
            )
            if event.get("state_hash") != residual_state_hash(state):
                errors.append(f"retention_state_mismatch:{event_id}")
            if not math.isclose(probability, float(event.get("probability")), abs_tol=1e-12):
                errors.append(f"retention_prediction_mismatch:{event_id}")
            retained += 1
        else:
            errors.append(f"event_type_invalid:{event_id}")
    return {
        "errors": list(dict.fromkeys(errors)),
        "updates_replayed": updates,
        "retention_predictions_checked": retained,
        "final_state_hash": residual_state_hash(state),
    }


def parse_extraction_row(metadata: Mapping[str, Any], raw: Mapping[str, Any]) -> JsonDict:
    """Parse one raw reply and derive literal status without producer summaries."""

    errors: list[str] = []
    call_id = str(metadata.get("call_id") or raw.get("call_id") or "missing")
    for field in ("call_id", "attempted", "terminal_state"):
        if field in metadata and field in raw and metadata.get(field) != raw.get(field):
            errors.append(f"raw_{field}_mismatch")
    raw_reply = str(raw.get("raw_reply") or "")
    for field in ("raw_request", "raw_response"):
        expected = metadata.get(f"{field}_sha256")
        if expected is not None and expected != canonical_hash(raw.get(field) or {}):
            errors.append(f"{field}_hash_mismatch")
    expected_reply = metadata.get("raw_reply_sha256")
    if expected_reply is not None and expected_reply != _text_hash(raw_reply):
        errors.append("raw_reply_hash_mismatch")
    attempted = raw.get("attempted") is True
    if not attempted:
        if raw_reply or raw.get("terminal_state") != "unstarted":
            errors.append("unstarted_payload_invalid")
        return {
            "call_id": call_id,
            "disposition": "unstarted",
            "errors": list(dict.fromkeys(errors)),
            "claims": [],
            "claim_spans": [],
            "literal": False,
            "factual_content_count": 0,
        }
    try:
        decoded = json.loads(raw_reply)
    except json.JSONDecodeError:
        errors.append("malformed_json")
        decoded = {}
    claims_value = decoded.get("claims") if isinstance(decoded, Mapping) else None
    if not isinstance(claims_value, list):
        errors.append("claims_shape")
        claims_value = []
    paragraph = str(metadata.get("paragraph") or "")
    arm = metadata.get("arm")
    claims: list[str] = []
    spans: list[list[int]] = []
    if arm == "span":
        for item in claims_value:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or any(not isinstance(value, int) or isinstance(value, bool) for value in item)
            ):
                errors.append("span_shape")
                continue
            start, end = item
            if start < 0 or end <= start or end > len(paragraph):
                errors.append("span_bounds")
                continue
            spans.append([start, end])
            claims.append(paragraph[start:end])
    elif arm == "verbatim":
        for item in claims_value:
            if not isinstance(item, str):
                errors.append("verbatim_shape")
                continue
            occurrences = paragraph.count(item)
            if occurrences == 0:
                errors.append("nonliteral_claim")
                continue
            if occurrences > 1 or item in claims:
                errors.append("ambiguous_claim")
                continue
            start = paragraph.index(item)
            spans.append([start, start + len(item)])
            claims.append(item)
    else:
        errors.append("arm_invalid")
    if errors:
        disposition = "malformed"
    elif claims:
        disposition = "usable_nonempty"
    elif metadata.get("canary_kind") == "nonfactual_control":
        disposition = "correct_empty"
    else:
        disposition = "empty"
    return {
        "call_id": call_id,
        "disposition": disposition,
        "errors": list(dict.fromkeys(errors)),
        "claims": claims,
        "claim_spans": spans,
        "literal": bool(claims) and not errors,
        "factual_content_count": len(claims),
    }


def validate_extraction_completeness(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Require all twelve development and 96 evaluation dispositions."""

    call_ids = [str(row.get("call_id") or "") for row in rows]
    phases = Counter(str(row.get("capture_phase")) for row in rows)
    if (
        len(rows) != 108
        or len(set(call_ids)) != 108
        or phases
        != {
            "development": 12,
            "evaluation": 96,
        }
    ):
        return ["planned_extraction_row_count_mismatch"]
    return []


def _load_hash_named(
    directory: Path, root: Path
) -> tuple[dict[str, JsonDict], list[JsonDict], list[str]]:
    """Authenticate hash-named JSON sidecars and index them by call ID."""

    values: dict[str, JsonDict] = {}
    references: list[JsonDict] = []
    errors: list[str] = []
    for path in sorted(directory.glob("sha256-*.json")):
        observed = sha256_file(path)
        expected = "sha256:" + path.stem.removeprefix("sha256-")
        if observed != expected:
            errors.append(f"sidecar_filename_hash_mismatch:{path.name}")
        value = _load_object(path)
        call_id = str(value.get("call_id") or "")
        if not call_id or call_id in values:
            errors.append(f"sidecar_call_id_invalid:{path.name}")
            continue
        values[call_id] = value
        references.append(_reference(root, path, "historical_model_sidecar", call_id=call_id))
    return values, references, errors


def audit_extraction_branch(root: Path, producer: Mapping[str, Any]) -> JsonDict:
    """Reparse twelve development replies and 96 evaluation dispositions."""

    base = root / "results/raw/experiment_7467_v654_factual_span_canary/owned_runtime"
    development_raw, development_refs, errors = _load_hash_named(base / "responses", root)
    evaluation_raw, evaluation_refs, evaluation_errors = _load_hash_named(base / "evaluation", root)
    errors.extend(evaluation_errors)
    development = producer.get("development_rows")
    evaluation = producer.get("extraction_rows")
    if not isinstance(development, list) or not isinstance(evaluation, list):
        return {
            "errors": [*errors, "producer_extraction_rows_invalid"],
            "rows": [],
            "sidecars": [*development_refs, *evaluation_refs],
            "counts": {},
        }
    errors.extend(validate_extraction_completeness([*development, *evaluation]))
    reduced_rows: list[JsonDict] = []
    factual_counts = {"span": 0, "verbatim": 0}
    correct_empty = 0
    for metadata in development:
        call_id = str(metadata.get("call_id") or "")
        raw = development_raw.get(call_id)
        if raw is None:
            errors.append(f"development_sidecar_missing:{call_id}")
            continue
        parsed = parse_extraction_row(metadata, raw)
        expected = metadata.get("development_disposition")
        if parsed["disposition"] != expected:
            errors.append(f"development_disposition_mismatch:{call_id}")
        arm = str(metadata.get("arm"))
        if metadata.get("canary_kind") == "factual":
            factual_counts[arm] += int(parsed["disposition"] == "usable_nonempty")
        correct_empty += int(parsed["disposition"] == "correct_empty")
        reduced_rows.append(_extraction_output_row(metadata, parsed))
    for metadata in evaluation:
        call_id = str(metadata.get("call_id") or "")
        raw = evaluation_raw.get(call_id)
        if raw is None:
            errors.append(f"evaluation_sidecar_missing:{call_id}")
            continue
        if canonical_hash(raw) != canonical_hash(metadata):
            errors.append(f"evaluation_sidecar_row_mismatch:{call_id}")
        parsed = parse_extraction_row(metadata, raw)
        if parsed["disposition"] != metadata.get("disposition"):
            errors.append(f"evaluation_disposition_mismatch:{call_id}")
        reduced_rows.append(_extraction_output_row(metadata, parsed))
    semantic_rows = producer.get("semantic_pair_rows")
    constructed = semantic_rows if isinstance(semantic_rows, list) else []
    if len(constructed) != 12:
        errors.append("constructed_pair_count_mismatch")
    qualifier_losses = 0
    for row in constructed:
        if row.get("authority") != "constructed_exact_string":
            errors.append(f"constructed_authority_invalid:{row.get('pair_id')}")
        expected_delta = int(row.get("span_qualifier_retained") is True) - int(
            row.get("verbatim_qualifier_retained") is True
        )
        if row.get("paired_delta") != expected_delta:
            errors.append(f"constructed_delta_mismatch:{row.get('pair_id')}")
        qualifier_losses += int(row.get("span_qualifier_retained") is not True)
        qualifier_losses += int(row.get("verbatim_qualifier_retained") is not True)
    counts = {
        "planned": len(reduced_rows),
        "attempted": sum(row["attempted"] for row in reduced_rows),
        "completed": sum(row["completed"] for row in reduced_rows),
        "failed": sum(row["failed"] for row in reduced_rows),
        "censored": sum(row["censored"] for row in reduced_rows),
        "unstarted": sum(row["unstarted"] for row in reduced_rows),
    }
    return {
        "errors": list(dict.fromkeys(errors)),
        "rows": reduced_rows,
        "sidecars": [*development_refs, *evaluation_refs],
        "counts": counts,
        "factual_content_counts": factual_counts,
        "correct_empty_controls": correct_empty,
        "natural_annotation_uncertainty": True,
        "constructed_exact_pairs": len(constructed),
        "constructed_qualifier_losses": qualifier_losses,
    }


def _extraction_output_row(metadata: Mapping[str, Any], parsed: Mapping[str, Any]) -> JsonDict:
    """Keep one compact audit row for every planned extraction unit."""

    attempted = metadata.get("attempted") is True
    failed = attempted and parsed["disposition"] in {"malformed", "failed"}
    completed = attempted and not failed and metadata.get("censored") is not True
    return {
        "unit_id": str(metadata.get("unit_id") or metadata.get("call_id")),
        "call_id": str(metadata.get("call_id")),
        "branch": "extraction",
        "phase": str(metadata.get("capture_phase")),
        "arm": str(metadata.get("arm")),
        "attempted": attempted,
        "completed": completed,
        "failed": failed,
        "censored": metadata.get("censored") is True,
        "unstarted": not attempted,
        "disposition": parsed["disposition"],
        "factual_content_count": parsed["factual_content_count"],
        "literal_span_reconstruction": parsed["literal"],
        "qualifier_loss": None,
        "natural_annotation_authority": "uncertain_not_exact",
        "costs": {"current_model_calls": 0},
    }


def _decision_fixture() -> tuple[list[JsonDict], JsonDict]:
    """Build a private fit-free fixture for mutation controls."""

    checkpoint: JsonDict = {
        "feature_names": ["support", "missing"],
        "weights": {"accept": [1.0, 0.0], "reject": [-1.0, 0.0], "escalate": [0.0, 0.0]},
        "bias": {name: 0.0 for name in OPTIONS},
    }
    checkpoint["checkpoint_sha256"] = _checkpoint_hash(checkpoint)
    rows: list[JsonDict] = []
    for index, (support, label) in enumerate(((1.0, "accept"), (-1.0, "reject"))):
        order = ["reject", "escalate", "accept"]
        logits = {"accept": support, "reject": -support, "escalate": 0.0}
        raw = [logits[name] for name in order]
        rows.append(
            {
                "unit_id": f"decision-{index}",
                "group_id": f"group-{index}",
                "option_order": order,
                "raw_logits": raw,
                "stored_probabilities": softmax_by_option(raw, order),
                "features": {"support": support, "missing": 0.0},
                "label": label,
                "eligible": True,
                "source_swap_control": True,
                "all_escalate_cost": 1.0,
            }
        )
    return rows, checkpoint


def _update_fixture() -> tuple[list[JsonDict], JsonDict]:
    """Build one private update and retention fixture for mutation controls."""

    checkpoint: JsonDict = {
        "seed": 17,
        "weights": [0.0, 0.0],
        "updates": 0,
        "learning_rate": 0.25,
        "frozen_control": False,
    }
    checkpoint["checkpoint_sha256"] = _checkpoint_hash(checkpoint)
    before = residual_state_hash(checkpoint)
    after_state = {"seed": 17, "weights": [0.125, 0.0], "updates": 1}
    after = residual_state_hash(after_state)
    return [
        {
            "type": "prediction",
            "event_id": "p1",
            "timestamp_ns": 10,
            "seed": 17,
            "features": [1.0, 0.0],
            "probability": 0.5,
            "state_hash_before": before,
        },
        {
            "type": "feedback",
            "event_id": "f1",
            "prediction_event_id": "p1",
            "timestamp_ns": 20,
            "seed": 17,
            "label": 1,
            "audit_selected": True,
            "audit_probability": 0.5,
            "state_hash_before": before,
            "state_hash_after": after,
        },
    ], checkpoint


def run_mutation_controls() -> list[JsonDict]:
    """Plant six corruptions and require distinct independent rejections."""

    decisions, decision_checkpoint = _decision_fixture()
    updates, update_checkpoint = _update_fixture()
    option_rows = deepcopy(decisions)
    option_rows[0]["option_order"] = ["accept", "escalate", "reject"]
    leaked_rows = deepcopy(decisions)
    leaked_rows[0]["features"]["gold_label"] = 1.0
    delayed = deepcopy(updates)
    delayed[1]["timestamp_ns"] = 9
    changed = deepcopy(update_checkpoint)
    changed["weights"] = [0.1, 0.0]
    producer = _load_object(REPO_ROOT / PRODUCERS["extraction"].path)
    extraction_rows = [
        *(producer.get("development_rows") or []),
        *(producer.get("extraction_rows") or []),
    ]
    observed = {
        "option_mapping": audit_decision_rows(option_rows, decision_checkpoint)["errors"],
        "leaked_labels": audit_decision_rows(leaked_rows, decision_checkpoint)["errors"],
        "missing_failure_rows": validate_extraction_completeness(extraction_rows[:-1]),
        "delayed_label_timestamp": replay_residual_updates(delayed, update_checkpoint)["errors"],
        "changed_checkpoint": replay_residual_updates(updates, changed)["errors"],
        "fabricated_improvement": verify_declared_improvement(decisions, declared=-0.9),
    }
    expected_tokens = {
        "option_mapping": "option_mapping_mismatch",
        "leaked_labels": "label_leakage",
        "missing_failure_rows": "planned_extraction_row_count_mismatch",
        "delayed_label_timestamp": "label_before_prediction",
        "changed_checkpoint": "checkpoint_hash_mismatch",
        "fabricated_improvement": "fabricated_improvement",
    }
    return [
        {
            "mutation": name,
            "expected_rejection": expected_tokens[name],
            "observed_errors": errors,
            "caught": any(expected_tokens[name] in error for error in errors),
            "disposition": "complete",
            "attempted": True,
            "completed": True,
            "failed": False,
            "censored": False,
            "unstarted": False,
            "costs": {"current_model_calls": 0},
        }
        for name, errors in observed.items()
    ]


def collect_preconditions(root: Path) -> tuple[list[JsonDict], dict[str, JsonDict]]:
    """Authenticate required sources and the two historical audit dispositions."""

    checks: list[JsonDict] = []
    sources: dict[str, JsonDict] = {}
    for relative in SOURCE_INPUTS:
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            _gate(
                f"source_bytes:{relative.as_posix()}",
                "precondition",
                "readable_nonempty_bytes",
                observed,
                upstream=relative.as_posix(),
                path=relative.as_posix(),
                field="bytes",
            )
        )
        if observed is not None:
            sources[relative.as_posix()] = _reference(root, relative, "current_audit_input")
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        _gate(
            "driving_requirement",
            "precondition",
            "REQ-REPORT-7470",
            "REQ-REPORT-7470" if "REQ-REPORT-7470" in spec_text else None,
            upstream=SPEC_PATH.as_posix(),
            path=SPEC_PATH.as_posix(),
            field="REQ-*",
        )
    )
    historical_expectations = {
        "results/experiment_7455_v653_decision_audit.json": (
            "disqualified",
            False,
            "complete_disqualified_required_validation",
        ),
        "results/experiment_7456_v653_extraction_audit.json": (
            "null",
            False,
            "complete_null_extraction_audit_preserved_development_gate_closed",
        ),
    }
    for label, expected in historical_expectations.items():
        value = _load_object(root / label)
        for field, wanted in zip(
            ("verdict_class", "flagged_adversarial", "honest_verdict"), expected, strict=True
        ):
            checks.append(
                _gate(
                    f"historical:{Path(label).stem}:{field}",
                    "historical_authentication",
                    wanted,
                    principle_value(value.get(field)),
                    upstream=Path(label).stem,
                    path=label,
                    field=field,
                )
            )
    return checks, sources


def _blocked_branch(slot: Mapping[str, Any]) -> JsonDict:
    """Create one terminal blocked row without hiding an available peer."""

    return {
        "branch": slot["branch"],
        "availability": slot["availability"],
        "validity": "not_auditable",
        "producer_path": slot["path"],
        "producer_sha256": slot.get("sha256"),
        "original_verdict_class": None,
        "original_flagged_adversarial": None,
        "verdict_class": "blocked",
        "honest_verdict": f"blocked_{slot['branch']}_producer_evidence_missing",
        "measured_benefit": None,
        "promotion_score": 0,
        "gate_check_summary": {
            "check": "producer_evidence",
            "upstream": PRODUCERS[str(slot["branch"])].experiment_id,
            "path": slot["path"],
            "field": "bytes/experiment_id/milestone",
            "expected": "exact_current_or_pre_gate_evidence",
            "observed": slot["availability"],
            "passed": False,
        },
    }


def audit_sources(root: Path) -> JsonDict:
    """Audit every available branch and preserve missing peers independently."""

    branches: list[JsonDict] = []
    rows: list[JsonDict] = []
    sources: dict[str, JsonDict] = {}
    update_replay: JsonDict = {
        "availability": "missing",
        "errors": [],
        "updates_replayed": 0,
        "retention_predictions_checked": 0,
        "disposition": "blocked_residual_learning_producer_missing",
    }
    extraction_audit: JsonDict = {}
    for branch_name in PRODUCERS:
        spec = PRODUCERS[branch_name]
        slot = locate_producer(root, spec)
        if slot["availability"] == "missing":
            branches.append(_blocked_branch(slot))
            continue
        artifact = slot["artifact"]
        reference = {
            "path": slot["path"],
            "sha256": slot["sha256"],
            "evidence_class": (
                "conductor_pre_gate" if slot["availability"] == "pre_gate" else "producer_artifact"
            ),
            "original_verdict_class": principle_value(artifact.get("verdict_class")),
            "original_flagged_adversarial": principle_value(artifact.get("flagged_adversarial")),
            "original_honest_verdict": principle_value(artifact.get("honest_verdict")),
        }
        sources[slot["path"]] = reference
        if slot["availability"] == "pre_gate":
            blocked = _blocked_branch(slot)
            blocked.update(
                {
                    "availability": "pre_gate",
                    "original_verdict_class": reference["original_verdict_class"],
                    "original_flagged_adversarial": reference["original_flagged_adversarial"],
                    "honest_verdict": f"blocked_{branch_name}_by_authenticated_pre_gate",
                }
            )
            branches.append(blocked)
            continue
        if slot["availability"] == "invalid":
            invalid = _blocked_branch(slot)
            invalid.update(
                {
                    "validity": "invalid",
                    "verdict_class": "disqualified",
                    "honest_verdict": f"complete_disqualified_{branch_name}_identity",
                }
            )
            branches.append(invalid)
            continue
        if not isinstance(artifact.get("flagged_adversarial"), bool):
            errors = ["producer_flag_type_invalid"]
            branch_result: JsonDict = {}
        elif branch_name == "extraction":
            extraction_audit = audit_extraction_branch(root, artifact)
            errors = extraction_audit["errors"]
            rows.extend(extraction_audit["rows"])
            for row in extraction_audit["sidecars"]:
                sources[row["path"]] = row
            branch_result = extraction_audit
        elif branch_name == "typed_decision":
            checkpoints = artifact.get("frozen_checkpoints")
            checkpoint = checkpoints[0] if isinstance(checkpoints, list) and checkpoints else {}
            branch_result = audit_decision_rows(artifact.get("rows") or [], checkpoint)
            errors = branch_result["errors"]
        else:
            event_rows = artifact.get("causal_event_rows")
            checkpoint = artifact.get("initial_checkpoint")
            if not isinstance(event_rows, list) or not isinstance(checkpoint, Mapping):
                branch_result = {"errors": ["residual_raw_evidence_missing"]}
            else:
                branch_result = replay_residual_updates(event_rows, checkpoint)
            errors = branch_result["errors"]
            update_replay = {
                "availability": "available",
                **branch_result,
                "disposition": "complete" if not errors else "disqualified",
            }
        validity = "valid" if not errors else "invalid"
        original_class = principle_value(artifact.get("verdict_class"))
        branch_class = (
            "disqualified" if errors or artifact.get("flagged_adversarial") is True else "null"
        )
        branches.append(
            {
                "branch": branch_name,
                "availability": "available",
                "validity": validity,
                "producer_path": slot["path"],
                "producer_sha256": slot["sha256"],
                "original_verdict_class": original_class,
                "original_flagged_adversarial": artifact.get("flagged_adversarial"),
                "verdict_class": branch_class,
                "honest_verdict": (
                    f"complete_disqualified_{branch_name}_audit"
                    if branch_class == "disqualified"
                    else f"complete_null_{branch_name}_audit_preserved"
                ),
                "measured_benefit": (
                    None if branch_name == "extraction" else branch_result.get("benefit")
                ),
                "promotion_score": 0,
                "audit_errors": errors,
                "audit_summary": {
                    key: deepcopy(value)
                    for key, value in branch_result.items()
                    if key not in {"rows", "sidecars", "errors"}
                },
                "gate_check_summary": None,
            }
        )
    return {
        "branch_rows": branches,
        "rows": rows,
        "source_artifact_hashes": sources,
        "independent_update_replay": update_replay,
        "extraction_audit": extraction_audit,
    }


def _validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require the exact affected set and any terminal readers already present."""

    names = Counter(str(row.get("name")) for row in receipts)
    affected = all(names[name] == 1 for name in validation_scope.REQUIRED_CHECK_NAMES)
    terminals_present = any(name in names for name in TERMINAL_CHECKS)
    terminals = not terminals_present or all(names[name] == 1 for name in TERMINAL_CHECKS)
    return (
        affected
        and terminals
        and all(
            row.get("passed") is True and row.get("exit_code") == 0
            for row in receipts
            if row.get("required") is not False
        )
    )


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain all failed checks and expose the first exact location."""

    failures = [row for row in gates if row.get("passed") is not True]
    if not failures:
        return {"all_passed": True, "failed_count": 0, "failed_checks": []}
    first = failures[0]
    return {
        "all_passed": False,
        "failed_count": len(failures),
        "failed_checks": [row.get("check") for row in failures],
        **{
            key: deepcopy(first.get(key))
            for key in ("check", "upstream", "path", "field", "expected", "observed", "op")
        },
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable evidence while excluding clocks and command durations."""

    receipt_scope = [
        {
            key: deepcopy(row.get(key))
            for key in (
                "name",
                "command_argv",
                "command_environment",
                "exit_code",
                "passed",
                "log_sha256",
            )
        }
        for row in value.get("validation_receipts") or []
    ]
    return canonical_hash(
        {
            "schema": value.get("schema"),
            "experiment_id": value.get("experiment_id"),
            "milestone": value.get("milestone"),
            "preconditions_checked": value.get("preconditions_checked"),
            "source_artifact_hashes": value.get("source_artifact_hashes"),
            "rows": value.get("rows"),
            "sample_size_budget": value.get("sample_size_budget"),
            "branch_rows": value.get("branch_rows"),
            "independent_update_replay": value.get("independent_update_replay"),
            "mutation_rows": value.get("mutation_rows"),
            "acceptance_gate_results": value.get("acceptance_gate_results"),
            "field_principles": value.get("field_principles"),
            "validation_scope": receipt_scope,
        }
    )


def _build_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Any],
    audit: Mapping[str, Any],
    mutations: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one schema-complete terminal audit from independent rows."""

    branches = list(audit["branch_rows"])
    rows = list(audit["rows"])
    mutation_rows = [deepcopy(dict(row)) for row in mutations]
    required_validation = _validation_passed(receipts)
    audit_complete = len(branches) == 3 and all(row.get("caught") is True for row in mutation_rows)
    any_invalid = any(row.get("verdict_class") == "disqualified" for row in branches)
    terminal_validation_present = all(
        sum(row.get("name") == name for row in receipts) == 1 for name in TERMINAL_CHECKS
    )
    terminal_validation_failed = terminal_validation_present and not required_validation
    verdict_class = "disqualified" if any_invalid or terminal_validation_failed else "null"
    status = (
        "complete_disqualified_independent_audit"
        if verdict_class == "disqualified"
        else "complete_independent_audit"
    )
    honest = (
        "complete_disqualified_independent_audit_evidence_or_validation"
        if verdict_class == "disqualified"
        else "complete_null_independent_audit_two_missing_branches_extraction_null"
    )
    extraction = audit.get("extraction_audit") or {}
    sample_budget = extraction.get("counts") or {
        "planned": 0,
        "attempted": 0,
        "completed": 0,
        "failed": 0,
        "censored": 0,
        "unstarted": 0,
    }
    gates = [
        _gate(
            "required_sources",
            "validity",
            True,
            all(row.get("passed") is True for row in preconditions),
            upstream="current_audit_inputs",
            path="preconditions_checked",
            field="passed",
        ),
        _gate(
            "branch_dispositions_complete",
            "audit_completion",
            3,
            len(branches),
            upstream=EXPERIMENT_ID,
            path="branch_rows",
            field="count",
        ),
        _gate(
            "mutations_rejected",
            "audit_completion",
            len(REQUIRED_MUTATIONS),
            sum(row.get("caught") is True for row in mutation_rows),
            upstream=EXPERIMENT_ID,
            path="mutation_rows",
            field="caught",
        ),
        _gate(
            "available_branch_validity",
            "validity",
            True,
            not any_invalid,
            upstream=EXPERIMENT_ID,
            path="branch_rows",
            field="validity",
        ),
        _gate(
            "typed_decision_benefit",
            "benefit",
            "measured_improvement",
            "not_evaluated_missing_branch",
            upstream=PRODUCERS["typed_decision"].experiment_id,
            path=PRODUCERS["typed_decision"].path.as_posix(),
            field="measured_benefit",
            passed=False,
        ),
        _gate(
            "residual_learning_benefit",
            "benefit",
            "measured_improvement",
            "not_evaluated_missing_branch",
            upstream=PRODUCERS["residual_learning"].experiment_id,
            path=PRODUCERS["residual_learning"].path.as_posix(),
            field="measured_benefit",
            passed=False,
        ),
        _gate(
            "required_validation",
            "required_validation",
            True,
            required_validation,
            upstream=EXPERIMENT_ID,
            path="validation_receipts",
            field="passed",
        ),
    ]
    receipt = build_current_work_receipt(
        run_id="exp7470-host-aggregation",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={
            "operation": "independent_json_and_scalar_reduction",
            "numeric_fitting": False,
            "historical_model_events_current": False,
        },
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=started_monotonic_ns,
        ended_monotonic_ns=ended_monotonic_ns,
        phase_spans=phase_spans,
    )
    duration = (ended_monotonic_ns - started_monotonic_ns) / 1_000_000_000
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "clock_identity": {
            "utc": "datetime.now(datetime.UTC)",
            "monotonic": "time.monotonic_ns",
        },
        "device_identity": {
            "execution_venue": "host",
            "hostname": platform.node(),
            "cpu": platform.processor() or platform.machine(),
            "cuda_used": False,
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        **receipt,
        "model_specs": [],
        "duration_s": duration,
        "model_duration_s": 0.0,
        "numeric_fitting_duration_s": 0.0,
        "computation_duration_s": max(
            0.0, duration - sum(float(row.get("duration_s", 0.0)) for row in receipts)
        ),
        "validation_duration_s": sum(float(row.get("duration_s", 0.0)) for row in receipts),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": deepcopy(RANDOM_SEEDS),
        "source_artifact_hashes": deepcopy(dict(sources)),
        "rows": rows,
        "sample_size_budget": deepcopy(sample_budget),
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "verifier_is_oracle": False,
        "flagged_adversarial": any(
            row.get("original_flagged_adversarial") is True for row in branches
        )
        or terminal_validation_failed,
        "validation_receipts": [deepcopy(dict(row)) for row in receipts],
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "independent_audit_complete_score": int(audit_complete and required_validation),
        "branch_rows": branches,
        "independent_update_replay": deepcopy(audit["independent_update_replay"]),
        "mutation_rows": mutation_rows,
        "extraction_audit": {
            key: deepcopy(value)
            for key, value in extraction.items()
            if key not in {"rows", "sidecars"}
        },
        "small_ebm_training": {"performed": False, "duration_s": 0.0},
        "promotion_score": 0,
        "external_publication_authorized": False,
        "numbered_e2e_applicable": False,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
    """Cold-check identity, branches, rows, mutations, receipts, and checksum."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in value:
            errors.append(f"required_field_missing:{field}")
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("artifact_identity_mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("artifact_execution_contract_mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        errors.append("model_specs_not_empty")
    if (
        value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("current_invocation_not_zero")
    if value.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_invalid")
    if (
        value.get("inference_substrate_class") != "aggregation"
        or value.get("execution_venue") != "host"
    ):
        errors.append("execution_declaration_invalid")
    errors.extend(validate_current_work_receipt(value, root=REPO_ROOT))
    branches = value.get("branch_rows")
    if not isinstance(branches, list) or [row.get("branch") for row in branches] != list(PRODUCERS):
        errors.append("branch_rows_invalid")
    mutations = value.get("mutation_rows")
    if not isinstance(mutations, list) or [row.get("mutation") for row in mutations] != list(
        REQUIRED_MUTATIONS
    ):
        errors.append("mutation_rows_invalid")
    elif not all(row.get("caught") is True for row in mutations):
        errors.append("mutation_not_caught")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != int(
        (value.get("sample_size_budget") or {}).get("planned", -1)
    ):
        errors.append("sample_budget_row_mismatch")
    receipts = value.get("validation_receipts")
    if not isinstance(receipts, list) or not _validation_passed(receipts):
        errors.append("required_validation_failed")
    if value.get("independent_audit_complete_score") not in {0, 1}:
        errors.append("independent_audit_complete_score_invalid")
    if value.get("promotion_score") != 0:
        errors.append("promotion_forbidden")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _passing_receipts(*, terminal: bool = True) -> list[JsonDict]:
    """Build deterministic receipt fixtures for artifact unit tests."""

    names = [*validation_scope.REQUIRED_CHECK_NAMES]
    if terminal:
        names.extend(TERMINAL_CHECKS)
    return [
        {
            "name": name,
            "command_argv": ["fixture", name],
            "command_environment": {},
            "exit_code": 0,
            "passed": True,
            "required": True,
            "duration_s": 0.0,
            "log_sha256": canonical_hash(name),
        }
        for name in names
    ]


def build_artifact_for_test() -> JsonDict:
    """Build a complete fixture from the real immutable extraction sidecars."""

    preconditions, sources = collect_preconditions(REPO_ROOT)
    audited = audit_sources(REPO_ROOT)
    sources.update(audited["source_artifact_hashes"])
    return _build_artifact(
        preconditions=preconditions,
        sources=sources,
        audit=audited,
        mutations=run_mutation_controls(),
        receipts=_passing_receipts(),
        started_at_utc="2026-09-21T00:00:00+00:00",
        completed_at_utc="2026-09-21T00:00:01+00:00",
        started_monotonic_ns=1_000_000_000,
        ended_monotonic_ns=2_000_000_000,
        phase_spans=[
            {
                "phase": "audit",
                "start_s": 0.0,
                "end_s": 1.0,
                "duration_s": 1.0,
                "completed_units": 108,
            }
        ],
    )


def cold_replay(path: Path) -> list[str]:
    """Reload a candidate in a fresh process and apply strict validation."""

    value = _load_object(path)
    return validate_artifact(value) if value else ["candidate_json_invalid"]


def independent_replay(path: Path, *, root: Path = REPO_ROOT) -> list[str]:
    """Re-read producer bytes and compare all branch and per-unit reductions."""

    value = _load_object(path)
    if not value:
        return ["candidate_json_invalid"]
    observed = audit_sources(root)
    errors: list[str] = []
    if observed["branch_rows"] != value.get("branch_rows"):
        errors.append("branch_reduction_mismatch")
    if observed["rows"] != value.get("rows"):
        errors.append("row_reduction_mismatch")
    if observed["independent_update_replay"] != value.get("independent_update_replay"):
        errors.append("update_replay_mismatch")
    return errors


def _utc_now() -> str:  # pragma: no cover - real execution clock boundary.
    """Return one measured UTC boundary."""

    return datetime.now(UTC).isoformat()


def _progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Emit a flushed phase boundary with completed work and elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7470] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(
    phase: str, phase_started_ns: int, run_started_ns: int, completed_units: int
) -> JsonDict:  # pragma: no cover - real execution clock boundary.
    """Record one monotonic phase interval and its completed-unit count."""

    ended = time.monotonic_ns()
    return {
        "phase": phase,
        "start_s": (phase_started_ns - run_started_ns) / 1_000_000_000,
        "end_s": (ended - run_started_ns) / 1_000_000_000,
        "duration_s": (ended - phase_started_ns) / 1_000_000_000,
        "completed_units": completed_units,
    }


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build capability replay and unchanged strict readers for one candidate."""

    python = str(REPO_ROOT / ".venv/bin/python")
    wrapper = str(REPO_ROOT / WRAPPER_PATH)
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                "declared_entrypoint_cold_replay",
                (python, "-u", wrapper, "--cold-replay", str(candidate)),
                "measured_candidate",
            ),
            "capability_e2e",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "independent_raw_reduction",
                (python, "-u", wrapper, "--independent-replay", str(candidate)),
                "measured_candidate",
            ),
            "completion",
            True,
        ),
        PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "measured_candidate",
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
                "measured_candidate",
            ),
            "completion",
            True,
        ),
    ]


def run_experiment(  # pragma: no cover - exercised through the declared entrypoint.
    repo_root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
) -> JsonDict:
    """Authenticate, reduce, validate, replay, and atomically publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = _utc_now()
    spans: list[JsonDict] = []

    phase_ns = time.monotonic_ns()
    _progress(started, "preconditions", "begin", completed_units=0)
    preconditions, sources = collect_preconditions(root)
    audited = audit_sources(root)
    sources.update(audited["source_artifact_hashes"])
    mutations = run_mutation_controls()
    spans.append(_span("audit", phase_ns, started_ns, len(audited["rows"])))
    _progress(
        started,
        "preconditions",
        "complete",
        completed_units=len(audited["rows"]),
        branches=len(audited["branch_rows"]),
    )

    private = Path(tempfile.mkdtemp(prefix="exp7470-validation-", dir="/tmp"))
    commands = build_command_plan(root, MANIFEST, private)
    plan_errors = validate_command_plan(root, MANIFEST, commands)
    if plan_errors:
        raise RuntimeError(f"validation_plan_invalid:{plan_errors}")
    raw = root / RAW_DIR
    phase_ns = time.monotonic_ns()
    _progress(started, "validation", "before_affected_subprocesses", completed_units=0)
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=raw / "validation/affected",
    )
    reduction = reduce_affected_receipts(root, MANIFEST, affected)
    spans.append(_span("validation", phase_ns, started_ns, len(affected)))
    _progress(
        started,
        "validation",
        "after_affected_subprocesses",
        completed_units=len(affected),
        passed=reduction["passed"],
    )
    candidate = _build_artifact(
        preconditions=preconditions,
        sources=sources,
        audit=audited,
        mutations=mutations,
        receipts=affected,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=spans,
    )
    candidate_path = raw / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    phase_ns = time.monotonic_ns()
    _progress(started, "terminal_validation", "before_subprocesses", completed_units=0)
    terminal = run_categorized_commands(
        root,
        _terminal_commands(candidate_path),
        log_dir=raw / "validation/terminal",
    )
    spans.append(_span("terminal_validation", phase_ns, started_ns, len(terminal)))
    _progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal),
        passed=all(row["passed"] for row in terminal),
    )
    final = _build_artifact(
        preconditions=preconditions,
        sources=sources,
        audit=audited,
        mutations=mutations,
        receipts=[*affected, *terminal],
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=spans,
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    _progress(started, "publish", "before_atomic_write", completed_units=1)
    atomic_json(candidate_path, final)
    atomic_json(root / output_path, final)
    _progress(started, "publish", "after_atomic_write", completed_units=1, path=output_path)
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse normal execution and the two fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-replay", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the reducer or one fresh-process terminal reader."""

    args = parse_args(argv)
    if args.cold_replay:
        errors = cold_replay(args.cold_replay)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_replay:
        errors = independent_replay(args.independent_replay)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if not args.date:
        raise SystemExit("--date is required")
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

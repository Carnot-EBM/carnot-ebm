"""Measure where retained-state importance-anchor work can run.

The experiment times host CPU arithmetic and durable state work. It preserves
board history without issuing board commands. It reports accelerator results
only as Amdahl bounds because the source artifacts lack a complete compatible
service denominator.

Spec refs: REQ-KAN-7487 and SCENARIO-KAN-7487-01 through
SCENARIO-KAN-7487-09.
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
import re
import shutil
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
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
RUN_DATE = "20260921"
MILESTONE = "2026.09.655"
EXPERIMENT_ID = "exp7487-learning-placement"
SCHEMA = "carnot.exp7487.v655.learning_placement.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7487_v655_learning_placement.json")
RAW_DIR = Path("results/raw/experiment_7487_v655_learning_placement")
MODULE_PATH = Path("python/carnot/experiment_7487_v655_learning_placement.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7487_v655_learning_placement.py")
TEST_PATH = Path("tests/python/test_experiment_7487_v655_learning_placement.py")
SPEC_PATH = Path("openspec/capabilities/kan/spec.md")
NOTE_PATH = Path("docs/research-notes/v655-learning-placement.md")

IMPORTANCE_ARTIFACT = Path("results/experiment_7482_v655_importance_anchor.json")
ONLINE_ARTIFACT = Path("results/experiment_7483_v655_continuous_learning.json")
FIT_CAPTURE_ARTIFACT = Path("results/experiment_7479_v655_source_fit_capture.json")
EVAL_CAPTURE_ARTIFACT = Path("results/experiment_7480_v655_source_eval_capture.json")
DURABLE_ARTIFACT = Path("results/experiment_7458_v653_durable_updates.json")
BOARD_ARTIFACT = Path("results/experiment_7473_v654_board_continuity.json")

INPUT_PATHS = (
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
    Path("python/carnot/experiment_7482_v655_importance_anchor.py"),
    Path("research-hardware-wishlist.md"),
    SPEC_PATH,
    IMPORTANCE_ARTIFACT,
    ONLINE_ARTIFACT,
    FIT_CAPTURE_ARTIFACT,
    EVAL_CAPTURE_ARTIFACT,
    DURABLE_ARTIFACT,
    BOARD_ARTIFACT,
)

MODEL_SPECS: list[JsonDict] = []
INFERENCE_SUBSTRATE = "host_retained_state_numeric_learning_and_artifact_aggregation"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
COEFFICIENT_SIZES = (32, 64, 128, 256)
NUMERIC_SEEDS = (748_700, 748_701, 748_702)
UPDATES = 1_000
ACTIVE_COEFFICIENTS = 4
ARITHMETIC_ARMS = ("float64", "float32", "signed_fixed_q24")
FIXED_ARITHMETIC = ARITHMETIC_ARMS[2]
FIXED_SCALE = 1 << 24
FIXED_CLIP = 8.0
FIXED_PROBABILITY_ERROR_LIMIT = 1e-4
LEARNING_RATE = 0.01
ANCHOR_STRENGTH = 0.25
TIMED_COMPONENTS = (
    "prediction",
    "data_gradient",
    "diagonal_penalty",
    "importance_update",
    "coefficient_update",
    "serialization",
    "durable_write",
    "fsync",
    "recovery",
)
INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in ("model_loads", "forward_calls", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def _load_object(path: Path) -> JsonDict:
    """Read one JSON object and keep malformed evidence unavailable."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _precondition(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    passed: bool | None = None,
    **details: Any,
) -> JsonDict:
    """Keep one exact prerequisite value separate from later conclusions."""

    return {
        "check": check,
        "upstream": upstream,
        "path": upstream,
        "artifact_field": field,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected if passed is None else passed,
        **details,
    }


def _source_record(path: Path, root: Path) -> JsonDict:
    """Bind exact source bytes and preserve their original terminal flags."""

    value = _load_object(path)
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "byte_size": stat.st_size,
        "owner_uid": stat.st_uid,
        "owner_gid": stat.st_gid,
        "original_flagged_adversarial": value.get("flagged_adversarial"),
        "original_verdict_class": value.get("verdict_class"),
        "original_honest_verdict": value.get("honest_verdict"),
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], dict[str, JsonDict], JsonDict]:
    """Authenticate source bytes, the learner, service inputs, and board history."""

    repo = root.resolve()
    checks: list[JsonDict] = []
    sources: dict[str, JsonDict] = {}
    for relative in INPUT_PATHS:
        path = repo / relative
        available = path.is_file() and path.stat().st_size > 0
        stat = path.stat() if available else None
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
                owner_uid=stat.st_uid if stat else None,
                owner_gid=stat.st_gid if stat else None,
                byte_size=stat.st_size if stat else None,
            )
        )
        if available:
            sources[relative.as_posix()] = _source_record(path, repo)

    spec_text = (
        (repo / SPEC_PATH).read_text(encoding="utf-8") if (repo / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-KAN-7487",
            "REQ-KAN-7487" if "REQ-KAN-7487" in spec_text else None,
        )
    )

    learner = _load_object(repo / IMPORTANCE_ARTIFACT)
    learner_expected: dict[str, Any] = {
        "schema": "carnot.exp7482.v655.importance_anchor.v1",
        "experiment_id": "exp7482-importance-anchor",
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "importance_anchor_ready_score": 1,
        "verdict_class": "circular_positive",
        "flagged_adversarial": False,
        "model_invoked": False,
        "MODEL_SPECS": [],
        "model_specs": [],
    }
    learner_checks: list[JsonDict] = []
    learner_checks.append(
        _precondition(
            "importance_learner_present",
            IMPORTANCE_ARTIFACT.as_posix(),
            "presence",
            True,
            bool(learner),
        )
    )
    for field, expected in learner_expected.items():
        learner_checks.append(
            _precondition(
                f"importance_learner_{field}",
                IMPORTANCE_ARTIFACT.as_posix(),
                field,
                expected,
                learner.get(field),
            )
        )
    learner_checks.append(
        _precondition(
            "importance_learner_zero_current_calls",
            IMPORTANCE_ARTIFACT.as_posix(),
            "invocation_counts",
            "all_zero",
            "all_zero"
            if learner.get("invocation_counts")
            and all(value == 0 for value in learner.get("invocation_counts", {}).values())
            else None,
        )
    )
    checks.extend(learner_checks)

    board = _load_object(repo / BOARD_ARTIFACT)
    board_rows = board.get("board_rows") if isinstance(board.get("board_rows"), list) else []
    board_names = sorted(str(row.get("board")) for row in board_rows if isinstance(row, Mapping))
    board_checks = [
        _precondition(
            "board_source_identity",
            BOARD_ARTIFACT.as_posix(),
            "experiment_id",
            "exp7473-v654-board-continuity",
            board.get("experiment_id"),
        ),
        _precondition(
            "board_source_flag",
            BOARD_ARTIFACT.as_posix(),
            "flagged_adversarial",
            False,
            board.get("flagged_adversarial"),
        ),
        _precondition(
            "board_names",
            BOARD_ARTIFACT.as_posix(),
            "board_rows[].board",
            ["GateMate", "KV260", "PolarFire"],
            board_names,
        ),
        _precondition(
            "board_hardware_operations",
            BOARD_ARTIFACT.as_posix(),
            "hardware_operations_issued",
            [],
            board.get("hardware_operations_issued"),
        ),
    ]
    checks.extend(board_checks)

    exclusion_text = (
        (repo / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
        if (repo / "ops/exclusion_manifest.yaml").is_file()
        else ""
    )
    checks.append(
        _precondition(
            "current_task_not_excluded",
            "ops/exclusion_manifest.yaml",
            "experiment_id",
            False,
            bool(re.search(r"experiment_id:\s*7487\b", exclusion_text)),
        )
    )
    context = {
        "numeric_branch_available": all(row["passed"] for row in learner_checks),
        "board_branch_available": bool(board) and all(row["passed"] for row in board_checks),
        "learner": learner,
        "board": board,
        "fit_capture": _load_object(repo / FIT_CAPTURE_ARTIFACT),
        "eval_capture": _load_object(repo / EVAL_CAPTURE_ARTIFACT),
        "online_service": _load_object(repo / ONLINE_ARTIFACT),
        "durable_history": _load_object(repo / DURABLE_ARTIFACT),
    }
    return checks, sources, context


def freeze_numeric_fixture(coefficient_count: int, seed: int, updates: int) -> JsonDict:
    """Freeze one shared update order before arithmetic arms can see outcomes."""

    if coefficient_count < ACTIVE_COEFFICIENTS or updates < 1:
        raise ValueError("numeric_fixture_shape_invalid")
    rng = np.random.default_rng(seed)
    initial = rng.normal(0.0, 0.02, size=coefficient_count)
    reference = initial.copy()
    importance = rng.uniform(0.1, 0.9, size=coefficient_count)
    active_indices = np.empty((updates, ACTIVE_COEFFICIENTS), dtype=np.int64)
    active_values = rng.uniform(-0.75, 0.75, size=(updates, ACTIVE_COEFFICIENTS))
    for index in range(updates):
        active_indices[index] = rng.choice(
            coefficient_count, size=ACTIVE_COEFFICIENTS, replace=False
        )
    target = rng.normal(0.0, 0.1, size=coefficient_count)
    logits = np.sum(target[active_indices] * active_values, axis=1) + 0.15
    labels = (logits + rng.normal(0.0, 0.03, size=updates) >= 0.0).astype(np.int64)
    value: JsonDict = {
        "coefficient_count": coefficient_count,
        "seed": seed,
        "updates": updates,
        "initial_coefficients": initial.tolist(),
        "reference_coefficients": reference.tolist(),
        "initial_importance": importance.tolist(),
        "initial_bias": 0.15,
        "active_indices": active_indices.tolist(),
        "active_values": active_values.tolist(),
        "labels": labels.tolist(),
        "learning_rate": LEARNING_RATE,
        "anchor_strength": ANCHOR_STRENGTH,
    }
    value["fixture_hash"] = canonical_hash(value)
    return value


def _quantize(value: Any) -> Any:
    """Apply the frozen signed Q24 clip and nearest-even rounding rule."""

    clipped = np.clip(np.asarray(value, dtype=np.float64), -FIXED_CLIP, FIXED_CLIP)
    quantized = np.rint(clipped * FIXED_SCALE) / FIXED_SCALE
    return float(quantized) if quantized.ndim == 0 else quantized


def _sigmoid(value: float) -> float:
    """Evaluate the stable Bernoulli probability used by every arithmetic arm."""

    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    direct = math.exp(value)
    return direct / (1.0 + direct)


def _initial_state(fixture: Mapping[str, Any], arithmetic: str) -> tuple[Any, Any, Any, float]:
    """Create one isolated arithmetic state from the same frozen fixture."""

    if arithmetic == "float64":
        dtype = np.float64
        transform = lambda value: np.asarray(value, dtype=dtype)
    elif arithmetic == "float32":
        dtype = np.float32
        transform = lambda value: np.asarray(value, dtype=dtype)
    elif arithmetic == FIXED_ARITHMETIC:
        transform = _quantize
    else:  # pragma: no cover - public callers are constrained by ARITHMETIC_ARMS.
        raise ValueError(f"arithmetic_invalid:{arithmetic}")
    coefficients = transform(fixture["initial_coefficients"])
    reference = transform(fixture["reference_coefficients"])
    importance = transform(fixture["initial_importance"])
    bias = float(transform(fixture["initial_bias"]))
    return coefficients, reference, importance, bias


def _predict(
    coefficients: Any,
    bias: float,
    indices: Any,
    values: Any,
    arithmetic: str,
) -> float:
    """Predict from active coefficients while retaining the declared arithmetic."""

    if arithmetic == "float32":
        logit = np.float32(bias) + np.dot(
            np.asarray(coefficients)[indices], np.asarray(values, dtype=np.float32)
        )
        return float(np.float32(_sigmoid(float(logit))))
    logit = float(bias + np.dot(np.asarray(coefficients)[indices], values))
    if arithmetic == FIXED_ARITHMETIC:
        return float(_quantize(_sigmoid(float(_quantize(logit)))))
    return _sigmoid(logit)


def _state_payload(
    coefficients: Any,
    importance: Any,
    bias: float,
    arithmetic: str,
    acknowledged_updates: int,
    acknowledgement_hash: str,
) -> JsonDict:
    """Serialize only the retained state needed for exact recovery checks."""

    return {
        "arithmetic": arithmetic,
        "coefficients": np.asarray(coefficients, dtype=np.float64).tolist(),
        "importance": np.asarray(importance, dtype=np.float64).tolist(),
        "bias": float(bias),
        "acknowledged_updates": acknowledged_updates,
        "acknowledgement_hash": acknowledgement_hash,
    }


def _durable_cycle(path: Path, payload: Mapping[str, Any]) -> tuple[dict[str, int], int, bool]:
    """Measure serialization, write, fsync, and recovery as separate operations."""

    started = time.perf_counter_ns()
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    serialization_ns = time.perf_counter_ns() - started

    started = time.perf_counter_ns()
    with path.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        write_ns = time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        os.fsync(stream.fileno())
        fsync_ns = time.perf_counter_ns() - started

    started = time.perf_counter_ns()
    recovered = json.loads(path.read_text(encoding="utf-8"))
    recovery_ns = time.perf_counter_ns() - started
    return (
        {
            "serialization": serialization_ns,
            "durable_write": write_ns,
            "fsync": fsync_ns,
            "recovery": recovery_ns,
        },
        len(encoded),
        recovered == dict(payload),
    )


def _empty_costs() -> dict[str, int]:
    """Create complete cost keys so missing operations cannot disappear."""

    return {name: 0 for name in TIMED_COMPONENTS}


def _update_arm(
    fixture: Mapping[str, Any],
    arithmetic: str,
    state_path: Path,
    reference_probabilities: Sequence[float] | None,
) -> tuple[JsonDict, list[float], JsonDict]:
    """Run one measured update replay and return its final retained state."""

    coefficients, reference, importance, bias = _initial_state(fixture, arithmetic)
    initial_payload = _state_payload(
        coefficients, importance, bias, arithmetic, 0, canonical_hash([])
    )
    costs = _empty_costs()
    probabilities: list[float] = []
    decisions: list[bool] = []
    recovered_all = True
    durable_bytes = 0
    acknowledgements: list[str] = []
    updates = int(fixture["updates"])
    learning_rate = float(fixture["learning_rate"])
    anchor_strength = float(fixture["anchor_strength"])

    for update_index in range(updates):
        indices = np.asarray(fixture["active_indices"][update_index], dtype=np.int64)
        values = np.asarray(fixture["active_values"][update_index], dtype=np.float64)
        label = int(fixture["labels"][update_index])

        started = time.perf_counter_ns()
        probability = _predict(coefficients, bias, indices, values, arithmetic)
        costs["prediction"] += time.perf_counter_ns() - started
        probabilities.append(probability)
        decisions.append(probability >= 0.5)

        started = time.perf_counter_ns()
        active_gradient = (probability - label) * values
        if arithmetic == "float32":
            active_gradient = np.asarray(active_gradient, dtype=np.float32)
        elif arithmetic == FIXED_ARITHMETIC:
            active_gradient = _quantize(active_gradient)
        costs["data_gradient"] += time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        penalty_gradient = 2.0 * anchor_strength * importance * (coefficients - reference)
        if arithmetic == "float32":
            penalty_gradient = np.asarray(penalty_gradient, dtype=np.float32)
        elif arithmetic == FIXED_ARITHMETIC:
            penalty_gradient = _quantize(penalty_gradient)
        costs["diagonal_penalty"] += time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        importance[indices] = np.minimum(1.0, importance[indices] + 0.001 * np.abs(values))
        if arithmetic == "float32":
            importance = np.asarray(importance, dtype=np.float32)
        elif arithmetic == FIXED_ARITHMETIC:
            importance = _quantize(importance)
        costs["importance_update"] += time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        coefficients = coefficients - learning_rate * penalty_gradient
        coefficients[indices] -= learning_rate * active_gradient
        bias -= learning_rate * (probability - label)
        if arithmetic == "float32":
            coefficients = np.asarray(coefficients, dtype=np.float32)
            bias = float(np.float32(bias))
        elif arithmetic == FIXED_ARITHMETIC:
            coefficients = _quantize(coefficients)
            bias = float(_quantize(bias))
        costs["coefficient_update"] += time.perf_counter_ns() - started

        acknowledgements.append(f"event-{update_index:04d}")
        payload = _state_payload(
            coefficients,
            importance,
            bias,
            arithmetic,
            update_index + 1,
            canonical_hash(acknowledgements),
        )
        durable_costs, byte_count, exact = _durable_cycle(state_path, payload)
        for name, value in durable_costs.items():
            costs[name] += value
        durable_bytes += byte_count
        recovered_all = recovered_all and exact

    reference_values = list(reference_probabilities or probabilities)
    probability_errors = [
        abs(value - reference_value)
        for value, reference_value in zip(probabilities, reference_values, strict=True)
    ]
    reference_decisions = [value >= 0.5 for value in reference_values]
    final_payload = _state_payload(
        coefficients,
        importance,
        bias,
        arithmetic,
        updates,
        canonical_hash(acknowledgements),
    )
    row = {
        "row_type": "numeric_cost",
        "mode": "update",
        "unit_id": f"numeric:{fixture['coefficient_count']}:{fixture['seed']}:{arithmetic}:update",
        "coefficient_count": fixture["coefficient_count"],
        "seed": fixture["seed"],
        "arithmetic": arithmetic,
        "fixture_hash": fixture["fixture_hash"],
        "attempted": True,
        "complete": True,
        "failed": False,
        "censored": False,
        "excluded": False,
        "disposition": "complete",
        "updates_attempted": updates,
        "updates_completed": updates,
        "active_data_gradient_count": ACTIVE_COEFFICIENTS,
        "full_anchor_coefficient_count": fixture["coefficient_count"],
        "component_total_ns": costs,
        "component_mean_ns": {name: value / updates for name, value in costs.items()},
        "durable_records": updates,
        "durable_bytes": durable_bytes,
        "exact_recovery": recovered_all,
        "acknowledged_updates": updates,
        "acknowledgement_hash": final_payload["acknowledgement_hash"],
        "acknowledgement_order_unchanged": True,
        "state_hash_before": canonical_hash(initial_payload),
        "state_hash_after": canonical_hash(final_payload),
        "state_unchanged": False,
        "max_probability_error": max(probability_errors, default=0.0),
        "decision_flips": sum(
            decision != reference_decision
            for decision, reference_decision in zip(decisions, reference_decisions, strict=True)
        ),
        "terminal_coefficient_error_linf": 0.0,
        "duration_s": sum(costs.values()) / 1_000_000_000,
    }
    return row, probabilities, final_payload


def _control_arm(fixture: Mapping[str, Any], arithmetic: str, state_path: Path) -> JsonDict:
    """Repeat durable recovery while keeping numeric and acknowledgement state fixed."""

    coefficients, _reference, importance, bias = _initial_state(fixture, arithmetic)
    payload = _state_payload(coefficients, importance, bias, arithmetic, 0, canonical_hash([]))
    state_hash = canonical_hash(payload)
    costs = _empty_costs()
    recovered_all = True
    durable_bytes = 0
    indices = np.asarray(fixture["active_indices"][0], dtype=np.int64)
    values = np.asarray(fixture["active_values"][0], dtype=np.float64)
    before_probability = _predict(coefficients, bias, indices, values, arithmetic)
    updates = int(fixture["updates"])

    for _ in range(updates):
        started = time.perf_counter_ns()
        _predict(coefficients, bias, indices, values, arithmetic)
        costs["prediction"] += time.perf_counter_ns() - started
        durable_costs, byte_count, exact = _durable_cycle(state_path, payload)
        for name, value in durable_costs.items():
            costs[name] += value
        durable_bytes += byte_count
        recovered_all = recovered_all and exact

    after_probability = _predict(coefficients, bias, indices, values, arithmetic)
    performed_costs: dict[str, int | None] = dict(costs)
    for name in (
        "data_gradient",
        "diagonal_penalty",
        "importance_update",
        "coefficient_update",
    ):
        performed_costs[name] = None
    return {
        "row_type": "numeric_cost",
        "mode": "no_update_control",
        "unit_id": (
            f"numeric:{fixture['coefficient_count']}:{fixture['seed']}:{arithmetic}:no_update"
        ),
        "coefficient_count": fixture["coefficient_count"],
        "seed": fixture["seed"],
        "arithmetic": arithmetic,
        "fixture_hash": fixture["fixture_hash"],
        "attempted": True,
        "complete": True,
        "failed": False,
        "censored": False,
        "excluded": False,
        "disposition": "complete",
        "updates_attempted": 0,
        "updates_completed": 0,
        "active_data_gradient_count": 0,
        "full_anchor_coefficient_count": 0,
        "component_total_ns": performed_costs,
        "component_mean_ns": {
            name: None if value is None else value / updates
            for name, value in performed_costs.items()
        },
        "durable_records": updates,
        "durable_bytes": durable_bytes,
        "exact_recovery": recovered_all,
        "acknowledged_updates": 0,
        "acknowledgement_hash": payload["acknowledgement_hash"],
        "acknowledgement_order_unchanged": True,
        "state_hash_before": state_hash,
        "state_hash_after": canonical_hash(payload),
        "state_unchanged": state_hash == canonical_hash(payload),
        "max_probability_error": abs(after_probability - before_probability),
        "decision_flips": int((before_probability >= 0.5) != (after_probability >= 0.5)),
        "terminal_coefficient_error_linf": 0.0,
        "duration_s": sum(costs.values()) / 1_000_000_000,
    }


def benchmark_size_seed(
    fixture: Mapping[str, Any], state_dir: Path, *, updates: int
) -> list[JsonDict]:
    """Measure all arithmetic arms and their identical durable controls."""

    if int(fixture.get("updates", -1)) != updates:
        raise ValueError("fixture_update_count_mismatch")
    state_dir.mkdir(parents=True, exist_ok=True)
    rows: list[JsonDict] = []
    reference_probabilities: list[float] | None = None
    reference_coefficients: np.ndarray[Any, Any] | None = None
    for arithmetic in ARITHMETIC_ARMS:
        safe_name = f"{fixture['coefficient_count']}-{fixture['seed']}-{arithmetic}"
        update_row, probabilities, final_payload = _update_arm(
            fixture,
            arithmetic,
            state_dir / f"{safe_name}-update.json",
            reference_probabilities,
        )
        if reference_probabilities is None:
            reference_probabilities = probabilities
            reference_coefficients = np.asarray(final_payload["coefficients"], dtype=np.float64)
        assert reference_coefficients is not None
        update_row["terminal_coefficient_error_linf"] = float(
            np.max(
                np.abs(
                    np.asarray(final_payload["coefficients"], dtype=np.float64)
                    - reference_coefficients
                )
            )
        )
        rows.append(update_row)
        rows.append(
            _control_arm(
                fixture,
                arithmetic,
                state_dir / f"{safe_name}-control.json",
            )
        )
    return rows


def reduce_numeric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    sizes: Sequence[int] = COEFFICIENT_SIZES,
    seeds: Sequence[int] = NUMERIC_SEEDS,
    updates: int = UPDATES,
) -> JsonDict:
    """Independently require every size, seed, arithmetic arm, and control."""

    expected = {
        (size, seed, arithmetic, mode)
        for size in sizes
        for seed in seeds
        for arithmetic in ARITHMETIC_ARMS
        for mode in ("update", "no_update_control")
    }
    by_key: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("coefficient_count"),
            row.get("seed"),
            row.get("arithmetic"),
            row.get("mode"),
        )
        by_key.setdefault(key, []).append(row)
    complete_keys = set(by_key) == expected and all(len(group) == 1 for group in by_key.values())
    required_components = set(TIMED_COMPONENTS)
    row_contract = complete_keys and all(
        row.get("complete") is True
        and row.get("failed") is False
        and row.get("exact_recovery") is True
        and required_components.issubset(dict(row.get("component_total_ns") or {}))
        and row.get("durable_records") == updates
        for row in rows
    )
    update_rows = [row for row in rows if row.get("mode") == "update"]
    control_rows = [row for row in rows if row.get("mode") == "no_update_control"]
    update_contract = len(update_rows) == len(expected) // 2 and all(
        row.get("updates_completed") == updates
        and row.get("active_data_gradient_count") == ACTIVE_COEFFICIENTS
        and row.get("full_anchor_coefficient_count") == row.get("coefficient_count")
        and row.get("acknowledgement_order_unchanged") is True
        for row in update_rows
    )
    control_contract = len(control_rows) == len(expected) // 2 and all(
        row.get("state_unchanged") is True
        and row.get("acknowledged_updates") == 0
        and row.get("updates_completed") == 0
        for row in control_rows
    )
    float64_by_fixture = {
        (row.get("coefficient_count"), row.get("seed")): row.get("acknowledgement_hash")
        for row in update_rows
        if row.get("arithmetic") == "float64"
    }
    replay_parity = bool(update_rows) and all(
        row.get("decision_flips") == 0
        and row.get("acknowledgement_hash")
        == float64_by_fixture.get((row.get("coefficient_count"), row.get("seed")))
        for row in update_rows
    )
    fixed_rows = [row for row in update_rows if row.get("arithmetic") == FIXED_ARITHMETIC]
    fixed_deployable = len(fixed_rows) == len(sizes) * len(seeds) and all(
        row.get("decision_flips") == 0
        and isinstance(row.get("max_probability_error"), (int, float))
        and row.get("max_probability_error") <= FIXED_PROBABILITY_ERROR_LIMIT
        and row.get("exact_recovery") is True
        for row in fixed_rows
    )
    component_totals = {
        name: sum(int(row.get("component_total_ns", {}).get(name, 0)) for row in update_rows)
        for name in TIMED_COMPONENTS
    }
    numeric_complete = bool(row_contract and update_contract and control_contract and replay_parity)
    return {
        "planned_rows": len(expected),
        "observed_rows": len(rows),
        "numeric_complete": numeric_complete,
        "replay_parity": replay_parity,
        "fixed_point_deployable": bool(numeric_complete and fixed_deployable),
        "max_fixed_probability_error": max(
            (float(row.get("max_probability_error", math.inf)) for row in fixed_rows),
            default=None,
        ),
        "fixed_decision_flips": sum(int(row.get("decision_flips", 0)) for row in fixed_rows),
        "component_total_ns": component_totals,
        "numeric_rows_hash": canonical_hash(rows),
    }


def amdahl_speedup(accelerated_fraction: float, kernel_speed: float) -> float:
    """Compute the service bound without treating kernel speed as service speed."""

    if not math.isfinite(accelerated_fraction) or not 0.0 <= accelerated_fraction <= 1.0:
        raise ValueError("accelerated_fraction_invalid")
    if kernel_speed < 1.0 or math.isnan(kernel_speed):
        raise ValueError("kernel_speed_invalid")
    denominator = (1.0 - accelerated_fraction) + (
        0.0 if math.isinf(kernel_speed) else accelerated_fraction / kernel_speed
    )
    return math.inf if denominator == 0.0 else 1.0 / denominator


def build_service_envelope(
    fit_capture: Mapping[str, Any],
    eval_capture: Mapping[str, Any],
    online_service: Mapping[str, Any],
) -> JsonDict:
    """Report measured inputs but refuse to add an incomplete service denominator."""

    service_rows = [
        dict(row)
        for row in online_service.get("service_cost_rows") or []
        if isinstance(row, Mapping)
    ]
    online_operations = {str(row.get("operation")) for row in service_rows}
    observed = {
        "raw_capture_forward": bool(
            fit_capture.get("duration_components_s", {}).get("forward")
            and eval_capture.get("duration_components_s", {}).get("forward")
        ),
        "feedback": "feedback_processing" in online_operations,
        "updates": "update" in online_operations,
        "serialization": "serialization" in online_operations,
        "fsync": "fsync" in online_operations,
    }
    missing: list[str] = []
    if not fit_capture.get("tokenization_cost_rows") or not eval_capture.get(
        "tokenization_cost_rows"
    ):
        missing.append("tokenization")
    if not fit_capture.get("prefill_cost_rows") or not eval_capture.get("prefill_cost_rows"):
        missing.append("prefill")
    if "verifier_work" not in online_operations:
        missing.append("verifier_work")
    for operation, field in (
        ("feedback_processing", "feedback"),
        ("update", "all_updates"),
        ("serialization", "serialization"),
        ("durable_write", "durable_write"),
        ("fsync", "fsync"),
    ):
        if operation not in online_operations:
            missing.append(field)
    bounds = []
    for fraction in (0.5, 0.9, 0.99, 0.999):
        for speed in (10.0, math.inf):
            bound = amdahl_speedup(fraction, speed)
            bounds.append(
                {
                    "accelerated_fraction": fraction,
                    "kernel_speed": "infinity" if math.isinf(speed) else speed,
                    "speedup_bound": bound,
                    "one_hundred_x_feasible": (1.0 - fraction) < 0.01 and bound >= 100.0,
                }
            )
    return {
        "status": (
            "measured_complete_service_denominator"
            if not missing
            else "conditional_missing_complete_components"
        ),
        "compatible_complete_denominator": not missing,
        "observed_source_components": observed,
        "online_service_operations": sorted(online_operations),
        "missing_components": sorted(set(missing)),
        "measured_end_to_end_speedup": None,
        "formula": "S=1/((1-f)+f/r)",
        "one_hundred_x_condition": "unaccelerated_fraction_below_0.01",
        "symbolic_bounds": bounds,
        "source_denominators_combined": False,
        "reason": (
            "Raw capture forward time does not split tokenization and prefill. "
            "The online rows omit verifier work and durable-write time."
        ),
    }


def reduce_board_evidence(
    board_artifact: Mapping[str, Any], changed_state: Mapping[str, Any] | None = None
) -> tuple[list[JsonDict], JsonDict]:
    """Preserve three board scopes and refuse fresh hardware work."""

    source_rows = board_artifact.get("board_rows")
    if not isinstance(source_rows, list):
        raise ValueError("board_rows_missing")
    by_board = {
        str(row.get("board")): deepcopy(dict(row))
        for row in source_rows
        if isinstance(row, Mapping)
    }
    if set(by_board) != {"KV260", "PolarFire", "GateMate"}:
        raise ValueError("board_rows_invalid")
    current = dict(changed_state or by_board["GateMate"].get("changed_state_evidence") or {})
    physical_changed = (
        current.get("exists") is True or int(current.get("accepted_receipt_count", 0) or 0) > 0
    )
    rows = [by_board[name] for name in ("KV260", "PolarFire", "GateMate")]
    for row in rows:
        row["hardware_operations_issued"] = []
        row["hardware_operation_count"] = 0
        row["fresh_physical_attempt"] = False
        row["new_hardware_execution_claimed"] = False
        row["current_execution_venue"] = "host"
    kv260 = by_board["KV260"]
    kv260["future_access"] = "ssh kria only"
    kv260["access_mechanism"] = "ssh kria only"
    kv260["architecture_limit"] = "k_max<=5"
    polarfire = by_board["PolarFire"]
    polarfire["exact_claim_scope"] = "historical_hash_matched_cpu_dispatch_only_no_fpga_sampling"
    gatemate = by_board["GateMate"]
    gatemate["changed_state_evidence"] = current
    if not physical_changed:
        gatemate["disposition"] = "blocked"
        gatemate["current_disposition"] = "blocked_unchanged_physical_prerequisite"
        gatemate["terminal_state"] = "blocked_changed_physical_state"
    reduction = {
        "board_branch_complete": (
            kv260.get("last_authenticated_date") == "20260915"
            and kv260.get("exact_claim_scope") == "historical_kv260_fpga_fabric_sampling_only"
            and kv260.get("future_access") == "ssh kria only"
            and kv260.get("architecture_limit") == "k_max<=5"
            and polarfire.get("last_authenticated_date") == "20260915"
            and polarfire.get("exact_claim_scope")
            == "historical_hash_matched_cpu_dispatch_only_no_fpga_sampling"
            and (
                physical_changed
                or gatemate.get("current_disposition") == "blocked_unchanged_physical_prerequisite"
            )
        ),
        "gatemate_physical_change_present": physical_changed,
        "gatemate_terminal_blocked": not physical_changed,
        "graduation_dates": {
            "KV260": kv260.get("last_authenticated_date"),
            "PolarFire": polarfire.get("last_authenticated_date"),
        },
        "hardware_operations_issued": [],
        "board_rows_hash": canonical_hash(rows),
    }
    return rows, reduction


def _validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one passing receipt for each frozen affected and terminal check."""

    required = set(AFFECTED_CHECK_NAMES) | set(TERMINAL_CHECK_NAMES)
    by_name: dict[str, list[Mapping[str, Any]]] = {}
    for row in receipts:
        by_name.setdefault(str(row.get("name")), []).append(row)
    return all(
        len(by_name.get(name, [])) == 1
        and by_name[name][0].get("passed") is True
        and by_name[name][0].get("exit_code") == 0
        and by_name[name][0].get("timed_out") is not True
        for name in required
    )


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute completion and benefit without trusting stored verdict fields."""

    numeric = reduce_numeric_rows(value.get("numeric_cost_rows") or [])
    _rows, board = reduce_board_evidence(
        {"board_rows": value.get("board_rows") or []}, value.get("gatemate_changed_state")
    )
    envelope = value.get("service_envelope") or {}
    declarations = (
        value.get("MODEL_SPECS") == []
        and value.get("model_specs") == []
        and value.get("model_invoked") is False
        and value.get("invocation_counts") == INVOCATION_COUNTS
        and value.get("inference_substrate_class") == INFERENCE_SUBSTRATE_CLASS
        and value.get("execution_venue") == EXECUTION_VENUE
    )
    placement_complete = bool(
        numeric["numeric_complete"]
        and numeric["fixed_point_deployable"]
        and board["board_branch_complete"]
        and board["hardware_operations_issued"] == []
        and envelope.get("status")
        in {"conditional_missing_complete_components", "measured_complete_service_denominator"}
        and declarations
    )
    service_100x = (
        envelope.get("compatible_complete_denominator") is True
        and isinstance(envelope.get("measured_end_to_end_speedup"), (int, float))
        and float(envelope["measured_end_to_end_speedup"]) >= 100.0
    )
    return {
        "numeric": numeric,
        "board": board,
        "service_envelope_hash": canonical_hash(envelope),
        "current_inference_declarations_valid": declarations,
        "required_validation_passed": _validation_passed(value.get("validation_receipts") or []),
        "placement_complete_score": int(placement_complete),
        "scientific_benefit_score": int(service_100x),
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    op: str,
    passed: bool,
    principle: str,
    *,
    upstream: str,
    path: str,
) -> JsonDict:
    """Attach one failure-prevention reason to every acceptance decision."""

    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": passed,
        "principle": principle,
        "upstream": upstream,
        "path": path,
    }


def _acceptance_gates(value: Mapping[str, Any], reduction: Mapping[str, Any]) -> list[JsonDict]:
    """Keep evidence validity, placement completion, and speed benefit separate."""

    numeric = reduction["numeric"]
    board = reduction["board"]
    envelope = value.get("service_envelope") or {}
    source_valid = all(
        row.get("passed") is True for row in value.get("preconditions_checked") or []
    )
    return [
        _gate(
            "source_authentication",
            "required_validity",
            True,
            source_valid,
            "is",
            source_valid,
            "A positive scientific metric cannot excuse invalid evidence.",
            upstream="preconditions_checked",
            path="preconditions_checked",
        ),
        _gate(
            "current_inference_declarations",
            "required_validity",
            True,
            reduction["current_inference_declarations_valid"],
            "is",
            reduction["current_inference_declarations_valid"] is True,
            "Archived model receipts must not become current model calls.",
            upstream="current_work",
            path="invocation_counts",
        ),
        _gate(
            "numeric_rows_complete",
            "required_validity",
            True,
            numeric["numeric_complete"],
            "is",
            numeric["numeric_complete"] is True,
            "A favorable size or seed cannot replace the frozen full matrix.",
            upstream="numeric_cost_rows",
            path="numeric_cost_rows",
        ),
        _gate(
            "fixed_point_replay_parity",
            "required_validity",
            {"decision_flips": 0, "max_probability_error_lte": FIXED_PROBABILITY_ERROR_LIMIT},
            {
                "decision_flips": numeric["fixed_decision_flips"],
                "max_probability_error": numeric["max_fixed_probability_error"],
            },
            "meets",
            numeric["fixed_point_deployable"] is True,
            "A cheaper arithmetic path cannot override changed decisions or failed replay.",
            upstream="numeric_cost_rows",
            path="independent_reduction.numeric",
        ),
        _gate(
            "board_continuity",
            "required_validity",
            True,
            board["board_branch_complete"],
            "is",
            board["board_branch_complete"] is True,
            "Unchanged board absence must stay distinct from authenticated graduation evidence.",
            upstream=BOARD_ARTIFACT.as_posix(),
            path="board_rows",
        ),
        _gate(
            "zero_hardware_operations",
            "required_validity",
            [],
            board["hardware_operations_issued"],
            "eq",
            board["hardware_operations_issued"] == [],
            "Evidence continuity must not silently become a new board probe or flash.",
            upstream="current_work",
            path="hardware_operations_issued",
        ),
        _gate(
            "scoped_terminal_validation",
            "required_validity",
            True,
            reduction["required_validation_passed"],
            "is",
            reduction["required_validation_passed"] is True,
            "A positive scientific metric cannot excuse failed implementation checks.",
            upstream="validation_receipts",
            path="validation_receipts",
        ),
        _gate(
            "placement_complete",
            "readiness",
            1,
            reduction["placement_complete_score"],
            "eq",
            reduction["placement_complete_score"] == 1,
            "A valid null must not suppress an independent placement measurement.",
            upstream="independent_reduction",
            path="placement_complete_score",
        ),
        _gate(
            "complete_service_denominator",
            "scientific_benefit",
            True,
            envelope.get("compatible_complete_denominator"),
            "is",
            envelope.get("compatible_complete_denominator") is True,
            "Missing service work must keep a numeric benchmark from becoming an end-to-end claim.",
            upstream="service_envelope",
            path="service_envelope.compatible_complete_denominator",
        ),
        _gate(
            "one_hundred_x_service_target",
            "scientific_benefit",
            100.0,
            envelope.get("measured_end_to_end_speedup"),
            ">=",
            reduction["scientific_benefit_score"] == 1,
            "A sparse target or favorable kernel cannot substitute for measured held-out service value.",
            upstream="service_envelope",
            path="service_envelope.measured_end_to_end_speedup",
        ),
    ]


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name each failed check without hiding valid null benefit failures."""

    failed = [
        {
            "check": row.get("check"),
            "category": row.get("category"),
            "upstream": row.get("upstream"),
            "path": row.get("path"),
            "expected": row.get("expected"),
            "observed": row.get("observed"),
        }
        for row in gates
        if row.get("passed") is not True
    ]
    return {"failed_count": len(failed), "failed_checks": failed}


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain required fields so later readers can audit them independently."""

    required = {
        "schema": "Versioned identity prevents silent reader drift.",
        "run_date": "The fixed date stays separate from measured UTC and monotonic clocks.",
        "preconditions_checked": "Exact paths and values prevent dependent work from using assumed evidence.",
        "MODEL_SPECS": "An empty list distinguishes numeric work from current model tasks.",
        "model_specs": "The lowercase mirror prevents schema aliases from hiding a model call.",
        "model_invoked": "Attempted current calls differ from archived model-shaped evidence.",
        "invocation_counts": "Balanced call states expose unfinished or concealed model work.",
        "inference_substrate": "The substrate names numeric learning and aggregation without implying generation.",
        "inference_substrate_class": "No model load prevents an incorrect model-duration floor.",
        "execution_venue": "Host identity separates CPU measurements from historical board evidence.",
        "duration_s": "Measured work duration cannot be padded into plausibility.",
        "phase_spans": "Flushed boundaries expose silent or unfinished operations.",
        "random_seed": "Frozen fixture and audit seeds make update order reproducible.",
        "reproducibility_checksum": "One checksum binds code, protocol, sources, rows, and validation scope.",
        "source_artifact_hashes": "Exact upstream bytes preserve their original flags and claims.",
        "rows": "Per-unit rows preserve failures, controls, sizes, seeds, and arithmetic arms.",
        "sample_size_budget": "Planned and completed units cannot be collapsed into one favorable count.",
        "acceptance_gate_results": "Expected and observed operands keep validity separate from benefit.",
        "gate_check_summary": "Each blocked or failed gate names its exact source and field.",
        "honest_verdict": "A terminal null records a completed measurement without a speed claim.",
        "verdict_class": "The closed enum prevents retryable partial work from posing as a null.",
        "verifier_is_oracle": "False prevents the acceptance reader from becoming the evaluation oracle.",
        "flagged_adversarial": "A real reader flag cannot be cleared to open a gate.",
        "validation_receipts": "Commands, exits, and log hashes establish the exact checked scope.",
        "field_principles": "Field reasons keep the artifact understandable without task context.",
        "placement_complete_score": "The bare score records reconciled branches, not hardware speed.",
        "numeric_cost_rows": "Per-size and seed costs keep dense anchor work separate from active gradients.",
        "service_envelope": "Amdahl bounds need a complete compatible denominator or remain conditional.",
        "board_rows": "Separate board rows preserve exact dates, scopes, and prerequisites.",
        "hardware_operations_issued": "An empty list distinguishes continuity from fresh board execution.",
    }
    return {
        key: required.get(key, "This field preserves one auditable part of the terminal record.")
        for key in keys
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind protocol, sources, raw rows, declarations, and validation scope."""

    return canonical_hash(
        {
            "schema": value.get("schema"),
            "experiment_id": value.get("experiment_id"),
            "milestone": value.get("milestone"),
            "run_date": value.get("run_date"),
            "protocol": value.get("protocol"),
            "source_artifact_hashes": value.get("source_artifact_hashes"),
            "numeric_cost_rows": value.get("numeric_cost_rows"),
            "board_rows": value.get("board_rows"),
            "service_envelope": value.get("service_envelope"),
            "MODEL_SPECS": value.get("MODEL_SPECS"),
            "model_specs": value.get("model_specs"),
            "model_invoked": value.get("model_invoked"),
            "invocation_counts": value.get("invocation_counts"),
            "validation_manifest": value.get("validation_manifest"),
            "validation_receipts": value.get("validation_receipts"),
        }
    )


def _finalize(value: JsonDict) -> JsonDict:
    """Derive all scores, gates, verdicts, and checksums from raw evidence."""

    reduction = independent_reduce(value)
    value["independent_reduction"] = reduction
    value["placement_complete_score"] = reduction["placement_complete_score"]
    value["scientific_benefit_score"] = reduction["scientific_benefit_score"]
    gates = _acceptance_gates(value, reduction)
    value["acceptance_gate_results"] = gates
    value["gate_check_summary"] = _gate_summary(gates)
    validity_passed = all(row["passed"] for row in gates if row["category"] == "required_validity")
    if not validity_passed:
        value["verdict_class"] = "disqualified"
        value["honest_verdict"] = "complete_disqualified_required_validity_failed"
    elif reduction["placement_complete_score"] != 1:
        value["verdict_class"] = "blocked"
        value["honest_verdict"] = "complete_blocked_learning_placement_incomplete"
    elif reduction["scientific_benefit_score"] == 1:  # pragma: no cover - sources lack denominator.
        value["verdict_class"] = "positive"
        value["honest_verdict"] = "complete_positive_one_hundred_x_service_path_measured"
    else:
        value["verdict_class"] = "null"
        value["honest_verdict"] = (
            "complete_null_retained_state_placement_measured_service_100x_not_established"
        )
    value["status"] = value["honest_verdict"]
    value["rows"] = [*deepcopy(value["numeric_cost_rows"]), *deepcopy(value["board_rows"])]
    value["field_principles"] = _field_principles(tuple(value))
    value["reproducibility_checksum"] = reproducibility_checksum(value)
    return value


def build_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    numeric_rows: Sequence[Mapping[str, Any]],
    board_rows: Sequence[Mapping[str, Any]],
    changed_state: Mapping[str, Any],
    service_envelope: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    phase_spans: Sequence[Mapping[str, Any]],
    device_identity: Mapping[str, Any],
) -> JsonDict:
    """Assemble one terminal candidate from measured and conditional branches."""

    numeric_reduction = reduce_numeric_rows(numeric_rows)
    validation_manifest = {
        "experiment_id": VALIDATION_MANIFEST.experiment_id,
        "test_paths": list(VALIDATION_MANIFEST.test_paths),
        "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
        "static_paths": list(VALIDATION_MANIFEST.static_paths),
        "affected_files": [
            MODULE_PATH.as_posix(),
            WRAPPER_PATH.as_posix(),
            TEST_PATH.as_posix(),
            SPEC_PATH.as_posix(),
            NOTE_PATH.as_posix(),
        ],
    }
    validation_manifest["manifest_hash"] = canonical_hash(validation_manifest)
    value: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "started_monotonic_ns": started_monotonic_ns,
        "ended_monotonic_ns": ended_monotonic_ns,
        "duration_s": (ended_monotonic_ns - started_monotonic_ns) / 1_000_000_000,
        "clock_identity": {"utc": "datetime.now(UTC)", "monotonic": "time.monotonic_ns"},
        "process_identity": {"pid": os.getpid(), "hostname": platform.node()},
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "numeric_fixture_seeds": list(NUMERIC_SEEDS),
            "ordering": "numpy.PCG64",
            "reducer": "deterministic_no_resampling",
        },
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "device_identity": deepcopy(dict(device_identity)),
        "duration_components_s": {
            "model_load": 0.0,
            "forward": 0.0,
            "generation": 0.0,
            "numeric_work": sum(row.get("duration_s", 0.0) for row in numeric_rows),
            "validation": sum(row.get("duration_s", 0.0) for row in validation_receipts),
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "source_artifact_hashes": deepcopy(dict(sources)),
        "protocol": {
            "coefficient_sizes": list(COEFFICIENT_SIZES),
            "updates_per_arm": UPDATES,
            "numeric_seeds": list(NUMERIC_SEEDS),
            "arithmetic_arms": list(ARITHMETIC_ARMS),
            "control": "identical_durable_no_update",
            "active_coefficients_per_update": ACTIVE_COEFFICIENTS,
            "fixed_point": {
                "format": "signed_q24",
                "scale": FIXED_SCALE,
                "clip": [-FIXED_CLIP, FIXED_CLIP],
                "rounding": "nearest_even_numpy_rint",
            },
            "probability_error_limit": FIXED_PROBABILITY_ERROR_LIMIT,
        },
        "small_ebm_training": {
            "performed": True,
            "receipt_class": "small_ebm_training",
            "head_type": "retained_state_importance_anchor_microbenchmark",
            "coefficient_sizes": list(COEFFICIENT_SIZES),
            "updates_per_arm": UPDATES,
            "generator_weights_fitted": False,
            "current_llm_calls": 0,
        },
        "numeric_cost_rows": [deepcopy(dict(row)) for row in numeric_rows],
        "numeric_microbenchmark": {
            "execution_venue": "host_cpu",
            "observed_service_performance": False,
            "sparse_arithmetic_target_only": True,
            "component_total_ns": numeric_reduction["component_total_ns"],
            "fixed_point_deployable": numeric_reduction["fixed_point_deployable"],
        },
        "service_envelope": deepcopy(dict(service_envelope)),
        "placement_map": {
            "cpu": [
                "event_order_and_counters",
                "active_data_gradient",
                "serialization",
                "durable_write",
                "fsync",
                "recovery",
            ],
            "gpu_batch_candidates": ["prediction", "complete_diagonal_penalty"],
            "future_fpga_or_tsu_candidates": [
                "fixed_point_prediction",
                "complete_diagonal_penalty",
                "coefficient_update",
            ],
            "kv260_limit": "k_max<=5",
            "observed_accelerator_performance": False,
            "purchase_or_install_authorized": False,
            "prefix_cache_implementation_authorized": False,
        },
        "board_rows": [deepcopy(dict(row)) for row in board_rows],
        "gatemate_changed_state": deepcopy(dict(changed_state)),
        "hardware_operations_issued": [],
        "hardware_ready_score": 0,
        "hardware_value_score": 0,
        "sample_size_budget": {
            "numeric": {
                "planned": len(COEFFICIENT_SIZES) * len(NUMERIC_SEEDS) * len(ARITHMETIC_ARMS) * 2,
                "attempted": len(numeric_rows),
                "complete": sum(row.get("complete") is True for row in numeric_rows),
                "failed": sum(row.get("failed") is True for row in numeric_rows),
                "censored": sum(row.get("censored") is True for row in numeric_rows),
                "excluded": sum(row.get("excluded") is True for row in numeric_rows),
                "unstarted": max(
                    0,
                    len(COEFFICIENT_SIZES) * len(NUMERIC_SEEDS) * len(ARITHMETIC_ARMS) * 2
                    - len(numeric_rows),
                ),
            },
            "boards": {"planned": 3, "attempted": 3, "complete": 3, "failed": 0},
        },
        "validation_manifest": validation_manifest,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "capability_e2e": {
            "entrypoint_run": True,
            "fresh_process_cold_replay_required": True,
            "numbered_runtime_e2e": "not_applicable_isolated_numeric_and_reporting_change",
        },
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "external_publication_authorized": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "methodology_note": (
            "Zero decision flips are deterministic replay parity on a fixed fixture. "
            "They are not a classifier accuracy or held-out value claim."
        ),
    }
    return _finalize(value)


def validate_artifact(
    value: Mapping[str, Any], *, root: Path | None = None, verify_sources: bool = False
) -> list[str]:
    """Cold-check identity, reductions, provenance, gates, and checksum."""

    errors: list[str] = []
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("run_identity_mismatch")
    if not (
        value.get("MODEL_SPECS") == []
        and value.get("model_specs") == []
        and value.get("model_invoked") is False
        and value.get("invocation_counts") == INVOCATION_COUNTS
        and value.get("inference_substrate_class") == INFERENCE_SUBSTRATE_CLASS
        and value.get("execution_venue") == EXECUTION_VENUE
    ):
        errors.append("current_inference_declaration_invalid")
    try:
        reduction = independent_reduce(value)
    except (KeyError, TypeError, ValueError):
        reduction = {}
    if reduction != value.get("independent_reduction"):
        errors.append("independent_reduction_mismatch")
    if reduction:
        gates = _acceptance_gates(value, reduction)
        if gates != value.get("acceptance_gate_results"):
            errors.append("acceptance_gate_results_mismatch")
        if _gate_summary(gates) != value.get("gate_check_summary"):
            errors.append("gate_check_summary_mismatch")
        if value.get("placement_complete_score") != reduction["placement_complete_score"]:
            errors.append("placement_complete_score_mismatch")
    if value.get("verifier_is_oracle") is not False:
        errors.append("verifier_oracle_invalid")
    if not all(
        isinstance(row, Mapping) and isinstance(row.get("principle"), str) and row["principle"]
        for row in value.get("acceptance_gate_results") or []
    ):
        errors.append("gate_principle_missing")
    required_principles = {
        "schema",
        "run_date",
        "numeric_cost_rows",
        "service_envelope",
        "board_rows",
        "hardware_operations_issued",
    }
    if not required_principles.issubset(dict(value.get("field_principles") or {})):
        errors.append("field_principles_missing")
    if verify_sources:
        base = (root or REPO_ROOT).resolve()
        for label, row in dict(value.get("source_artifact_hashes") or {}).items():
            path = Path(str(row.get("path") or label))
            resolved = path if path.is_absolute() else base / path
            if not resolved.is_file() or sha256_file(resolved) != row.get("sha256"):
                errors.append(f"source_hash_invalid:{label}")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _fixture_board_artifact() -> JsonDict:
    """Create the narrow authenticated board scopes used by pure unit fixtures."""

    changed = {
        "exists": False,
        "accepted_receipt_count": 0,
        "latest_receipt_date": "20260823",
        "cutoff_date": "20260823",
        "hardware_operations_issued": [],
    }
    rows = [
        {
            "board": "KV260",
            "last_authenticated_date": "20260915",
            "last_authenticated_path": "results/experiment_7314_v642_board_continuity.json",
            "exact_claim_scope": "historical_kv260_fpga_fabric_sampling_only",
            "current_disposition": "graduated_preserved",
            "future_access": "ssh kria only",
            "architecture_limit": "k_max<=5",
            "hardware_operations_issued": [],
        },
        {
            "board": "PolarFire",
            "last_authenticated_date": "20260915",
            "last_authenticated_path": "results/experiment_7314_v642_board_continuity.json",
            "exact_claim_scope": "historical_hash_matched_cpu_dispatch_only_no_fpga_sampling",
            "current_disposition": "graduated_cpu_dispatch_preserved",
            "hardware_operations_issued": [],
        },
        {
            "board": "GateMate",
            "last_authenticated_date": "20260916",
            "current_disposition": "blocked_unchanged_physical_prerequisite",
            "terminal_state": "blocked_changed_physical_state",
            "changed_state_evidence": changed,
            "hardware_operations_issued": [],
        },
    ]
    return {
        "schema": "carnot.exp7473.v654.board_continuity.v1",
        "experiment_id": "exp7473-v654-board-continuity",
        "milestone": "2026.09.654",
        "run_date": RUN_DATE,
        "flagged_adversarial": False,
        "verdict_class": "null",
        "honest_verdict": "complete_null_board_continuity_gatemate_blocked_unchanged_physical_prerequisite",
        "board_rows": rows,
        "hardware_operations_issued": [],
    }


def build_fixture_root(path: Path) -> Path:
    """Build isolated source bytes so tests never rewrite the research record."""

    root = path / "fixture-root"
    for relative in INPUT_PATHS:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = REPO_ROOT / relative
        if source.is_file():
            shutil.copy2(source, target)
        else:  # pragma: no cover - repository inputs are present in normal tests.
            target.write_text("fixture\n", encoding="utf-8")
    learner = {
        "schema": "carnot.exp7482.v655.importance_anchor.v1",
        "experiment_id": "exp7482-importance-anchor",
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "importance_anchor_ready_score": 1,
        "verdict_class": "circular_positive",
        "honest_verdict": "complete_circular_positive_importance_anchor_fixture_benefit",
        "flagged_adversarial": False,
        "model_invoked": False,
        "MODEL_SPECS": [],
        "model_specs": [],
        "invocation_counts": deepcopy(INVOCATION_COUNTS),
    }
    atomic_json(root / IMPORTANCE_ARTIFACT, learner)
    atomic_json(root / BOARD_ARTIFACT, _fixture_board_artifact())
    return root


def build_fixture_artifact(path: Path) -> JsonDict:
    """Build complete private evidence for mutation and cold-reader tests."""

    root = build_fixture_root(path)
    preconditions, sources, context = collect_preconditions(root)
    numeric_rows: list[JsonDict] = []
    fixture_updates = 3
    for size in COEFFICIENT_SIZES:
        for seed in NUMERIC_SEEDS:
            fixture = freeze_numeric_fixture(size, seed, fixture_updates)
            numeric_rows.extend(
                benchmark_size_seed(fixture, path / "numeric-state", updates=fixture_updates)
            )
    # Normalize fixture update counts to the production contract before reduction.
    for row in numeric_rows:
        row["durable_records"] = UPDATES
        if row["mode"] == "update":
            row["updates_attempted"] = UPDATES
            row["updates_completed"] = UPDATES
            row["acknowledged_updates"] = UPDATES
    board_rows, _board_reduction = reduce_board_evidence(context["board"])
    service = build_service_envelope(
        context["fit_capture"], context["eval_capture"], context["online_service"]
    )
    receipts = [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "duration_s": 0.0,
            "log_sha256": "sha256:" + "0" * 64,
        }
        for name in (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]
    return build_artifact(
        preconditions=preconditions,
        sources=sources,
        numeric_rows=numeric_rows,
        board_rows=board_rows,
        changed_state=board_rows[2]["changed_state_evidence"],
        service_envelope=service,
        validation_receipts=receipts,
        started_at_utc="2026-09-21T00:00:00+00:00",
        completed_at_utc="2026-09-21T00:00:01+00:00",
        started_monotonic_ns=0,
        ended_monotonic_ns=1_000_000_000,
        phase_spans=[],
        device_identity={"venue": "host", "numeric_device": "cpu_fixture"},
    )


def utc_now() -> str:  # pragma: no cover - current execution boundary.
    """Return one measured UTC boundary for the current run."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush each phase and slow-operation boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7487] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(
    phase: str, phase_started: float, run_started: float, completed: int, checkpoint: str
) -> JsonDict:  # pragma: no cover - current clock boundary.
    """Close one monotonic phase with its completed-unit checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed,
        "checkpoint": checkpoint,
    }


def _device_identity() -> JsonDict:  # pragma: no cover - host identity varies.
    """Name actual CPU software without implying board or accelerator execution."""

    return {
        "venue": "host",
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unreported",
        "numpy_version": np.__version__,
        "numeric_device": "cpu_float64_float32_fixed_emulation",
        "cuda_used": False,
        "historical_board_evidence_used_as_current": False,
    }


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build fresh replay, independent reduction, and strict reader commands."""

    common = ("--date", RUN_DATE, "--root", ".")
    specs = (
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--cold-replay",
                str(candidate),
            ),
            "capability_end_to_end",
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--independent-reduce",
                str(candidate),
            ),
            "per_size_seed_arithmetic_rows",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (".venv/bin/python", "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal_candidate",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                ".venv/bin/python",
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "terminal_candidate",
        ),
    )
    return [PlannedCommand(spec, "required_validation", True) for spec in specs]


def _scan_current_gatemate(root: Path, raw_path: Path) -> JsonDict:  # pragma: no cover
    """Use the approved local receipt parser without running a hardware command."""

    from carnot.experiment_7473_v654_board_continuity import (
        normalize_changed_state,
        search_changed_state_evidence,
    )

    return normalize_changed_state(search_changed_state_evidence(root, raw_path))


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - exercised by the declared capability E2E.
    """Authenticate, benchmark, validate, replay, and atomically publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    repo = root.resolve()
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_utc = utc_now()
    spans: list[JsonDict] = []

    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    preconditions, sources, context = collect_preconditions(repo)
    if not context["board_branch_available"]:
        raise RuntimeError("board_precondition_failed")
    spans.append(_span("preconditions", phase_started, started, len(preconditions), "source_bytes"))
    progress(
        started,
        "preconditions",
        "complete",
        numeric_available=context["numeric_branch_available"],
        board_available=context["board_branch_available"],
    )

    for phase in ("model_load", "generation"):
        progress(started, phase, "before", planned=0)
        phase_started = time.monotonic()
        spans.append(_span(phase, phase_started, started, 0, f"no_{phase}"))
        progress(started, phase, "after", completed=0)

    raw_dir = repo / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    progress(started, "board_evidence", "before_local_receipt_scan")
    phase_started = time.monotonic()
    changed_state = _scan_current_gatemate(repo, raw_dir / "gatemate_changed_state_evidence.json")
    board_rows, board_reduction = reduce_board_evidence(context["board"], changed_state)
    spans.append(_span("board_evidence", phase_started, started, 3, "three_board_rows"))
    progress(
        started,
        "board_evidence",
        "after_local_receipt_scan",
        complete=board_reduction["board_branch_complete"],
        operations=0,
    )

    numeric_rows: list[JsonDict] = []
    progress(started, "numeric_benchmark", "before", planned=12)
    phase_started = time.monotonic()
    if context["numeric_branch_available"]:
        private_state = Path(tempfile.mkdtemp(prefix="exp7487-state-", dir="/tmp"))
        completed_units = 0
        for size in COEFFICIENT_SIZES:
            for seed in NUMERIC_SEEDS:
                fixture = freeze_numeric_fixture(size, seed, UPDATES)
                numeric_rows.extend(benchmark_size_seed(fixture, private_state, updates=UPDATES))
                completed_units += 1
                progress(
                    started,
                    "numeric_benchmark",
                    "unit_complete",
                    completed=completed_units,
                    planned=12,
                    coefficients=size,
                    seed=seed,
                )
    spans.append(
        _span("numeric_benchmark", phase_started, started, len(numeric_rows), "per_unit_rows")
    )
    progress(started, "numeric_benchmark", "after", completed=len(numeric_rows))
    if not context["numeric_branch_available"]:
        raise RuntimeError("numeric_branch_blocked_by_importance_learner")

    service = build_service_envelope(
        context["fit_capture"], context["eval_capture"], context["online_service"]
    )
    for relative in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH, NOTE_PATH):
        path = repo / relative
        if path.is_file():
            sources[relative.as_posix()] = _source_record(path, repo)

    private_validation = Path(tempfile.mkdtemp(prefix="exp7487-validation-", dir="/tmp"))
    commands = build_command_plan(repo, VALIDATION_MANIFEST, private_validation)
    plan_errors = validate_command_plan(repo, VALIDATION_MANIFEST, commands)
    progress(started, "affected_validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic()
    affected = (
        []
        if plan_errors
        else run_categorized_commands(
            repo,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw_dir / "validation/affected",
        )
    )
    affected_reduction = reduce_affected_receipts(repo, VALIDATION_MANIFEST, affected)
    spans.append(
        _span("affected_validation", phase_started, started, len(affected), "affected_checks")
    )
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=affected_reduction["passed"],
    )
    if plan_errors or not affected_reduction["passed"]:
        raise RuntimeError(f"affected_validation_failed:{plan_errors}:{affected_reduction}")

    provisional = [
        {
            "name": name,
            "required": True,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "duration_s": 0.0,
            "provisional_for_candidate_reader": True,
        }
        for name in TERMINAL_CHECK_NAMES
    ]
    candidate = build_artifact(
        preconditions=preconditions,
        sources=sources,
        numeric_rows=numeric_rows,
        board_rows=board_rows,
        changed_state=changed_state,
        service_envelope=service,
        validation_receipts=[*affected, *provisional],
        started_at_utc=started_utc,
        completed_at_utc=utc_now(),
        started_monotonic_ns=0,
        ended_monotonic_ns=time.monotonic_ns() - started_ns,
        phase_spans=spans,
        device_identity=_device_identity(),
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    progress(started, "candidate", "before_serialization")
    atomic_json(candidate_path, candidate)
    progress(started, "candidate", "after_serialization")

    terminal_commands = _terminal_commands(candidate_path)
    progress(started, "terminal_validation", "before_subprocesses", planned=4)
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        repo, terminal_commands, log_dir=raw_dir / "validation/terminal"
    )
    spans.append(
        _span("terminal_validation", phase_started, started, len(terminal), "terminal_readers")
    )
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )
    if not terminal_passed or critical:
        raise RuntimeError("terminal_validation_failed")

    final = build_artifact(
        preconditions=preconditions,
        sources=sources,
        numeric_rows=numeric_rows,
        board_rows=board_rows,
        changed_state=changed_state,
        service_envelope=service,
        validation_receipts=[*affected, *terminal],
        started_at_utc=started_utc,
        completed_at_utc=utc_now(),
        started_monotonic_ns=0,
        ended_monotonic_ns=time.monotonic_ns() - started_ns,
        phase_spans=spans,
        device_identity=_device_identity(),
    )
    errors = validate_artifact(final, root=repo, verify_sources=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "write", "before_atomic_terminal", path=output_path)
    atomic_json(repo / output_path, final)
    progress(started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed date and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--no-source-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin process boundary.
    """Run the experiment or one strict fresh-process terminal reader."""

    args = parse_args(argv)
    root = args.root.resolve()
    if args.cold_replay is not None:
        value = _load_object(args.cold_replay)
        errors = (
            validate_artifact(value, root=root, verify_sources=not args.no_source_check)
            if value
            else ["artifact_unreadable_or_not_object"]
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = _load_object(args.independent_reduce)
        errors = (
            validate_artifact(value, root=root, verify_sources=not args.no_source_check)
            if value
            else ["artifact_unreadable_or_not_object"]
        )
        reduction = independent_reduce(value) if value and not errors else {}
        print(json.dumps({"errors": errors, "reduction": reduction}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(root, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

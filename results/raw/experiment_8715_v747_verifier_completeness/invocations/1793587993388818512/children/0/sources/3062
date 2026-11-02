"""Measure a Rust process port of the qualified recalibration service.

Both arms use the same durable state and event order. The measurement includes
process IPC and storage work, so an isolated solver gain cannot become a
whole-service claim.

Spec refs: REQ-CL-7585, SCENARIO-CL-7585-*, REQ-REPORT-7585, and
SCENARIO-REPORT-7585-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, TextIO

import numpy as np

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    build_command_plan,
    validate_command_plan,
)
from carnot.experiment_7561_v661_recalibration_prototype import (
    KNOTS,
    MOVEMENT_BOUND,
    RIDGE_MASS,
    RELEASE_BLOCK_SIZE,
    SOLVER_MAX_ITERATIONS,
    SOLVER_TOLERANCE,
    SufficientStatisticMap,
    constraint_errors,
    piecewise_design,
    quadratic_objective,
    typed_decision,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
RUN_DATE = "20260924"
MILESTONE = "2026.09.662"
EXPERIMENT_ID = "exp7585-v662-portable-service"
SCHEMA = "carnot.exp7585.v662.portable_service.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7585_v662_portable_service.json")
RAW_DIR = Path("results/raw/experiment_7585_v662_portable_service")
MODULE_PATH = Path("python/carnot/experiment_7585_v662_portable_service.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7585_v662_portable_service.py")
TEST_PATH = Path("tests/python/test_experiment_7585_v662_portable_service.py")
RUST_SOURCE_PATH = Path("crates/carnot-core/src/bin/portable-recalibration-service.rs")
CL_SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
REPORT_SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
REQUALIFICATION_PATH = Path("results/experiment_7574_v662_measurement_requalification.json")
PORTABLE_PATH = Path("results/experiment_7571_v661_portable_calibration.json")
STATE_SCHEMA_PATH = Path(
    "results/raw/experiment_7561_v661_recalibration_prototype/numerical_state_schema.json"
)
RUST_BINARY = Path("target/release/portable-recalibration-service")
MODEL_SPECS: list[JsonDict] = []
MODEL_SPECS_RESOLVED: list[JsonDict] = []
PARITY_STREAM_COUNT = 1_000
SERVICE_REPEATS = 30
SERVICE_EVENT_COUNT = 160
PARITY_TOLERANCE = 1e-8
DURABILITY_POLICY = "atomic_file_fsync_rename_directory_fsync_reload_ack"
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
ZERO_INVOCATION_COUNTS = {
    "model_loads": 0,
    "forward_calls": 0,
    "generation_calls": 0,
    "current_llm_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
}
INPUT_PATHS = (
    REQUALIFICATION_PATH,
    PORTABLE_PATH,
    STATE_SCHEMA_PATH,
    CL_SPEC_PATH,
    REPORT_SPEC_PATH,
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("research-hardware-wishlist.md"),
    Path("ops/hardware-bringup-prep.md"),
    Path("ops/north-star.md"),
)
AFFECTED_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def load_object(path: Path) -> JsonDict:
    """Return one JSON object, or no evidence when bytes are unusable."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def source_row(path: Path, root: Path) -> JsonDict:
    """Bind exact bytes and retain any literal source disposition."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        label = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        label = str(resolved)
    value = load_object(resolved)
    return {
        "path": label,
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        "original_honest_verdict": value.get("honest_verdict"),
        "original_verdict_class": value.get("verdict_class"),
        "original_flagged_adversarial": value.get("flagged_adversarial"),
    }


def _check(
    check: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    op: str = "eq",
) -> JsonDict:
    """Retain both operands so missing evidence cannot become a zero."""

    passed = observed == expected if op == "eq" else bool(observed)
    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "op": op,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": passed,
    }


def collect_preconditions(root: Path) -> JsonDict:
    """Authenticate inputs and numerical readiness before measurement."""

    repo = root.resolve()
    rows: list[JsonDict] = []
    sources: dict[str, JsonDict] = {}
    for relative in INPUT_PATHS:
        path = repo / relative
        present = path.is_file() and path.stat().st_size > 0
        rows.append(
            _check(
                f"source_readable:{relative.as_posix()}",
                relative.as_posix(),
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if present else None,
            )
        )
        if present:
            sources[relative.as_posix()] = source_row(path, repo)

    requalification = load_object(repo / REQUALIFICATION_PATH)
    portable = load_object(repo / PORTABLE_PATH)
    rows.extend(
        [
            _check(
                "exp7574_recalibration_ready_score",
                "Exp7574",
                REQUALIFICATION_PATH.as_posix(),
                "recalibration_ready_score",
                1,
                requalification.get("recalibration_ready_score"),
            ),
            _check(
                "exp7574_allowed_verdict_class",
                "Exp7574",
                REQUALIFICATION_PATH.as_posix(),
                "verdict_class",
                True,
                requalification.get("verdict_class") in {"positive", "null", "circular_positive"},
            ),
            _check(
                "exp7574_not_flagged",
                "Exp7574",
                REQUALIFICATION_PATH.as_posix(),
                "flagged_adversarial",
                False,
                requalification.get("flagged_adversarial"),
            ),
        ]
    )
    for spec_path, requirement in (
        (CL_SPEC_PATH, "REQ-CL-7585"),
        (REPORT_SPEC_PATH, "REQ-REPORT-7585"),
    ):
        text = (
            (repo / spec_path).read_text(encoding="utf-8") if (repo / spec_path).is_file() else ""
        )
        rows.append(
            _check(
                f"driving_requirement:{requirement}",
                spec_path.as_posix(),
                spec_path.as_posix(),
                "REQ-*",
                requirement,
                requirement if requirement in text else None,
            )
        )
    rows.append(
        _check(
            "rust_toolchain_available",
            "local toolchain",
            "PATH",
            "cargo",
            True,
            shutil.which("cargo") is not None,
        )
    )
    failed = next((row for row in rows if row["passed"] is not True), None)
    blocker = (
        {
            key: deepcopy(failed[key])
            for key in ("check", "upstream", "path", "field", "op", "expected", "observed")
        }
        if failed
        else None
    )
    return {
        "rows": rows,
        "blocker": blocker,
        "kernel_branch_ready": failed is None,
        "requalification": requalification,
        "portable_artifact": portable,
        "source_artifact_hashes": sources,
        "resource_observations": {
            "cargo_path": shutil.which("cargo"),
            "logical_cpu_count": os.cpu_count(),
            "tmp_free_bytes": os.statvfs("/tmp").f_bavail * os.statvfs("/tmp").f_frsize,
        },
    }


def build_board_rows(portable_artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Copy the three dated board rows without issuing a new probe."""

    rows = deepcopy(list(portable_artifact.get("board_rows") or []))
    if [row.get("board") for row in rows] != ["KV260", "PolarFire", "GateMate"]:
        raise ValueError("exp7571_board_rows_incomplete")
    for row in rows:
        row["historical_source"] = PORTABLE_PATH.as_posix()
        row["current_reachability"] = "unknown_not_probed"
        row["current_reachability_probe_issued"] = False
        row["present_reachability_asserted"] = False
        row["new_hardware_execution_claimed"] = False
        row["hardware_operations_issued"] = []
        row["hardware_operation_count"] = 0
    return rows


def seeded_streams(count: int = PARITY_STREAM_COUNT) -> list[JsonDict]:
    """Create deterministic streams with registered probability boundaries."""

    if count < 1:
        raise ValueError("stream_count_must_be_positive")
    streams: list[JsonDict] = []
    boundaries = [0.0, 1.0, 0.04, 0.16, 0.2, 0.8]
    for index in range(count):
        seed = 7_585_100 + index
        generator = random.Random(seed)
        probabilities = boundaries + [generator.random() for _ in range(10)]
        labels = [int(generator.random() < probability) for probability in probabilities]
        streams.append(
            {"stream_id": index, "seed": seed, "probabilities": probabilities, "labels": labels}
        )
    return streams


def service_stream(seed: int = 7_585_001, count: int = SERVICE_EVENT_COUNT) -> JsonDict:
    """Create the fixed service trace without reading outcome-dependent state."""

    generator = random.Random(seed)
    probabilities = [generator.random() for _ in range(count)]
    labels = [int(generator.random() < probability) for probability in probabilities]
    return {"stream_id": seed, "seed": seed, "probabilities": probabilities, "labels": labels}


def stream_events(stream: Mapping[str, Any], *, prefix: str | None = None) -> list[JsonDict]:
    """Pair probabilities and labels in the one accepted event order."""

    probabilities = list(stream["probabilities"])
    labels = list(stream["labels"])
    if len(probabilities) != len(labels):
        raise ValueError("probability_label_length_mismatch")
    stem = prefix or f"stream-{stream['stream_id']}"
    return [
        {"event_id": f"{stem}-{index:04d}", "probability": float(probability), "label": int(label)}
        for index, (probability, label) in enumerate(zip(probabilities, labels))
    ]


def _project_monotone(values: np.ndarray) -> np.ndarray:
    levels: list[float] = []
    weights: list[int] = []
    for value in values:
        levels.append(float(value))
        weights.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            weight = weights[-2] + weights[-1]
            merged = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / weight
            levels[-2:] = [merged]
            weights[-2:] = [weight]
    return np.asarray(
        [level for level, weight in zip(levels, weights) for _ in range(weight)], dtype=float
    )


def _project_constraints(values: np.ndarray) -> np.ndarray:
    lower = np.maximum(0.0, KNOTS - MOVEMENT_BOUND)
    upper = np.minimum(1.0, KNOTS + MOVEMENT_BOUND)
    current = np.asarray(values, dtype=float).copy()
    box_residual = np.zeros(9)
    order_residual = np.zeros(9)
    for _iteration in range(2_000):
        box_input = current + box_residual
        boxed = np.minimum(upper, np.maximum(lower, box_input))
        box_residual = box_input - boxed
        order_input = boxed + order_residual
        ordered = _project_monotone(order_input)
        order_residual = order_input - ordered
        delta = float(np.max(np.abs(current - ordered)))
        current = ordered
        if delta <= 1e-14:
            break
    return current


def solve_portable_map(
    gram: np.ndarray, target: np.ndarray, theta_start: Sequence[float] | None = None
) -> tuple[np.ndarray, JsonDict]:
    """Solve the qualified convex program with the process-port algorithm."""

    matrix = np.asarray(gram, dtype=float)
    vector = np.asarray(target, dtype=float)
    if matrix.shape != (9, 9) or vector.shape != (9,):
        raise ValueError("sufficient_statistic_shape_invalid")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)):
        raise ValueError("sufficient_statistic_not_finite")
    theta = np.asarray(theta_start if theta_start is not None else KNOTS, dtype=float).copy()
    lipschitz = max(
        2.0 * (sum(abs(float(matrix[row, column])) for column in range(9)) + RIDGE_MASS)
        for row in range(9)
    )
    converged = False
    iterations = 0
    for iteration in range(1, SOLVER_MAX_ITERATIONS + 1):
        gradient = np.asarray(
            [
                2.0
                * (
                    sum(float(matrix[row, column]) * float(theta[column]) for column in range(9))
                    - float(vector[row])
                    + RIDGE_MASS * (float(theta[row]) - float(KNOTS[row]))
                )
                for row in range(9)
            ]
        )
        following = _project_constraints(theta - gradient / lipschitz)
        delta = float(np.max(np.abs(theta - following)))
        theta = following
        iterations = iteration
        if delta <= SOLVER_TOLERANCE * 0.01:
            converged = True
            break
    errors = constraint_errors(theta)
    receipt = {
        "method": "projected_gradient_port_of_slsqp_contract",
        "tolerance": SOLVER_TOLERANCE,
        "iteration_cap": SOLVER_MAX_ITERATIONS,
        "converged": converged and not errors,
        "status": 0 if converged and not errors else 1,
        "iterations": iterations,
        "objective": quadratic_objective(theta, matrix, vector),
        "constraint_errors": errors,
        "message": "converged" if converged and not errors else "iteration limit reached",
    }
    return theta, receipt


def _portable_update_batch(
    machine: SufficientStatisticMap, rows: Sequence[tuple[str, float, int]]
) -> JsonDict:
    """Apply one release transaction with the shared portable solver."""

    candidate = SufficientStatisticMap.from_payload(machine.to_payload())
    seen: set[str] = set()
    event_ids: list[str] = []
    for event_id, probability, label in rows:
        name = str(event_id)
        if name in seen or name in candidate.processed_event_ids:
            raise ValueError(f"duplicate_feedback:{name}")
        if label not in (0, 1):
            raise ValueError("binary_label_required")
        seen.add(name)
        event_ids.append(name)
        design = piecewise_design(probability)
        candidate.gram += np.outer(design, design)
        candidate.target += int(label) * design
    theta, receipt = solve_portable_map(candidate.gram, candidate.target, candidate.theta)
    if receipt["converged"] is not True:
        raise RuntimeError("recalibration_solver_did_not_converge")
    machine.gram = candidate.gram
    machine.target = candidate.target
    machine.theta = theta
    machine.sample_count += len(rows)
    machine.processed_event_ids.update(event_ids)
    machine.last_solver_receipt = deepcopy(receipt)
    return {**receipt, "sample_count": machine.sample_count, "update_count": len(rows)}


def _directory_fsync(path: Path) -> None:
    """Make the rename durable under the shared service policy."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_state_write(path: Path, payload: Mapping[str, Any]) -> tuple[int, JsonDict]:
    """Write, fsync, rename, fsync the directory, and return stage times."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    stages: JsonDict = {"write": 0, "fsync": 0, "rename": 0}
    started = time.perf_counter_ns()
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        wrote = time.perf_counter_ns()
        stages["write"] += wrote - started
        os.fsync(stream.fileno())
        synced = time.perf_counter_ns()
        stages["fsync"] += synced - wrote
    temporary.replace(path)
    renamed = time.perf_counter_ns()
    stages["rename"] += renamed - synced
    _directory_fsync(path.parent)
    stages["fsync"] += time.perf_counter_ns() - renamed
    return len(encoded), stages


def _load_state(path: Path) -> SufficientStatisticMap:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("state_payload_not_object")
    return SufficientStatisticMap.from_payload(value)


def initialize_state(path: Path) -> int:
    """Create the common identity state with the full durability policy."""

    size, _stages = _durable_state_write(path, SufficientStatisticMap.create().to_payload())
    reloaded = _load_state(path)
    if reloaded.sample_count != 0:
        raise ValueError("identity_state_reload_failed")
    return size


def run_python_service_request(request: Mapping[str, Any]) -> JsonDict:
    """Run one trace through read, predict, solve, persist, reload, and ack."""

    if request.get("operation") == "solver_failure":
        return {"ok": False, "error": "recalibration_solver_did_not_converge"}
    if request.get("operation") != "trace":
        return {"ok": False, "error": "unknown_operation"}
    state_path = Path(str(request["state_path"]))
    events = [dict(row) for row in request.get("events") or []]
    if not state_path.exists():
        initialize_state(state_path)
    stage_ns = {
        name: 0 for name in ("read", "predict", "solve", "write", "fsync", "rename", "reload")
    }
    predictions: list[JsonDict] = []
    acknowledgments: list[int] = []
    try:
        started = time.perf_counter_ns()
        machine = _load_state(state_path)
        stage_ns["read"] += time.perf_counter_ns() - started
        for offset in range(0, len(events), RELEASE_BLOCK_SIZE):
            block = events[offset : offset + RELEASE_BLOCK_SIZE]
            update_rows: list[tuple[str, float, int]] = []
            for row in block:
                started = time.perf_counter_ns()
                probability = machine.predict(float(row["probability"]))
                prediction = typed_decision(probability)
                stage_ns["predict"] += time.perf_counter_ns() - started
                predictions.append(
                    {
                        "event_id": row["event_id"],
                        "probability": probability,
                        "action": prediction["action"],
                    }
                )
                update_rows.append(
                    (str(row["event_id"]), float(row["probability"]), int(row["label"]))
                )
            started = time.perf_counter_ns()
            _portable_update_batch(machine, update_rows)
            stage_ns["solve"] += time.perf_counter_ns() - started
            _size, write_stages = _durable_state_write(state_path, machine.to_payload())
            for name, duration in write_stages.items():
                stage_ns[name] += duration
            started = time.perf_counter_ns()
            reloaded = _load_state(state_path)
            stage_ns["reload"] += time.perf_counter_ns() - started
            if reloaded.state_hash() != machine.state_hash():
                raise ValueError("reloaded_state_mismatch")
            machine = reloaded
            acknowledgments.append(offset // RELEASE_BLOCK_SIZE)
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        return {"ok": False, "error": str(error), "stage_ns": stage_ns}
    return {
        "ok": True,
        "predictions": predictions,
        "acknowledgments": acknowledgments,
        "acknowledged_release_count": len(acknowledgments),
        "processed_event_count": len(events),
        "reloaded_state_matches": True,
        "state_bytes": state_path.stat().st_size,
        "state": machine.to_payload(),
        "stage_ns": stage_ns,
        "kernel_ns": stage_ns["predict"] + stage_ns["solve"],
        "durability_policy": DURABILITY_POLICY,
    }


def service_worker_loop(stream_in: TextIO, stream_out: TextIO) -> int:  # pragma: no cover
    """Serve newline-delimited requests for a fresh process measurement."""

    for line in stream_in:
        try:
            request = json.loads(line)
            if not isinstance(request, Mapping):
                raise ValueError("request_not_object")
            response = run_python_service_request(request)
        except (json.JSONDecodeError, ValueError) as error:
            response = {"ok": False, "error": str(error)}
        stream_out.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n")
        stream_out.flush()
    return 0


def _worker_command(root: Path, arm: str) -> list[str]:  # pragma: no cover
    if arm == "python":
        return [
            str(root / ".venv/bin/python"),
            "-u",
            str(root / WRAPPER_PATH),
            "--date",
            RUN_DATE,
            "--root",
            str(root),
            "--service-worker",
        ]
    if arm == "rust":
        return [str(root / RUST_BINARY)]
    raise ValueError(f"unknown_worker_arm:{arm}")


def _start_worker(root: Path, arm: str) -> subprocess.Popen[str]:  # pragma: no cover
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = f"{root / 'python'}:{root}"
    return subprocess.Popen(  # noqa: S603 - fixed worktree executables and no shell.
        _worker_command(root, arm),
        cwd=root,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def _exchange(  # pragma: no cover
    worker: subprocess.Popen[str], request: Mapping[str, Any]
) -> tuple[JsonDict, int]:
    """Measure one complete IPC exchange with a bounded live child."""

    if worker.stdin is None or worker.stdout is None:
        raise RuntimeError("worker_pipe_missing")
    started = time.perf_counter_ns()
    worker.stdin.write(json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n")
    worker.stdin.flush()
    line = worker.stdout.readline()
    elapsed = time.perf_counter_ns() - started
    if not line:
        error = worker.stderr.read() if worker.stderr is not None else ""
        raise RuntimeError(f"worker_ended_without_response:{error[-1000:]}")
    value = json.loads(line)
    if not isinstance(value, Mapping):
        raise RuntimeError("worker_response_not_object")
    return dict(value), elapsed


def _stop_worker(worker: subprocess.Popen[str]) -> None:  # pragma: no cover
    """Stop only the process created by this measurement."""

    if worker.stdin is not None:
        worker.stdin.close()
    try:
        exit_code = worker.wait(timeout=10)
    except subprocess.TimeoutExpired:
        worker.terminate()
        exit_code = worker.wait(timeout=10)
    if exit_code != 0:
        error = worker.stderr.read() if worker.stderr is not None else ""
        raise RuntimeError(f"worker_exit_failed:{exit_code}:{error[-1000:]}")


def _state_core(value: Mapping[str, Any]) -> JsonDict:
    """Compare numerical state while allowing solver receipt text to differ."""

    return {
        "gram": value.get("gram"),
        "target": value.get("target"),
        "theta": value.get("theta"),
        "sample_count": value.get("sample_count"),
        "processed_event_ids": value.get("processed_event_ids"),
        "solver_config": value.get("solver_config"),
        "constrained": value.get("constrained"),
    }


def _numeric_nested_error(left: Any, right: Any) -> float:
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if not isinstance(right, Sequence) or len(left) != len(right):
            return math.inf
        return max((_numeric_nested_error(a, b) for a, b in zip(left, right)), default=0.0)
    return abs(float(left) - float(right))


def compare_worker_responses(
    python_response: Mapping[str, Any], rust_response: Mapping[str, Any]
) -> JsonDict:
    """Reduce probability, decision, state, and acknowledgment agreement."""

    if python_response.get("ok") is not True or rust_response.get("ok") is not True:
        return {
            "passed": False,
            "max_probability_absolute_error": None,
            "typed_decision_mismatches": None,
            "acknowledgment_match": False,
            "state_match": False,
            "error": {
                "python": python_response.get("error"),
                "rust": rust_response.get("error"),
            },
        }
    python_predictions = list(python_response.get("predictions") or [])
    rust_predictions = list(rust_response.get("predictions") or [])
    if len(python_predictions) != len(rust_predictions):
        raise ValueError("prediction_count_mismatch")
    errors = [
        abs(float(left["probability"]) - float(right["probability"]))
        for left, right in zip(python_predictions, rust_predictions)
    ]
    decision_mismatches = sum(
        left.get("action") != right.get("action")
        for left, right in zip(python_predictions, rust_predictions)
    )
    acknowledgment_match = python_response.get("acknowledgments") == rust_response.get(
        "acknowledgments"
    )
    python_state = _state_core(dict(python_response.get("state") or {}))
    rust_state = _state_core(dict(rust_response.get("state") or {}))
    theta_error = max(
        abs(float(left) - float(right))
        for left, right in zip(python_state.get("theta") or [], rust_state.get("theta") or [])
    )
    numeric_state_error = max(
        _numeric_nested_error(python_state.get(field), rust_state.get(field))
        for field in ("gram", "target")
    )
    exact_state_fields = all(
        python_state.get(field) == rust_state.get(field)
        for field in ("sample_count", "processed_event_ids", "solver_config", "constrained")
    )
    maximum = max(errors + [theta_error, numeric_state_error], default=0.0)
    return {
        "passed": maximum <= PARITY_TOLERANCE
        and decision_mismatches == 0
        and acknowledgment_match
        and exact_state_fields,
        "max_probability_absolute_error": maximum,
        "typed_decision_mismatches": decision_mismatches,
        "exact_tie_disclosures": 0,
        "acknowledgment_match": acknowledgment_match,
        "state_match": exact_state_fields and theta_error <= PARITY_TOLERANCE,
    }


def _failure_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare typed failure meaning without treating timing metadata as state."""

    return bool(
        left.get("ok") is False
        and right.get("ok") is False
        and left.get("error") == right.get("error")
    )


def _parity_row(  # pragma: no cover
    stream: Mapping[str, Any], arm: str, comparison: Mapping[str, Any]
) -> JsonDict:
    metric = 0.0 if arm == "python" else comparison.get("max_probability_absolute_error")
    return {
        "row_type": "parity_stream",
        "unit_id": f"stream:{stream['stream_id']}",
        "pair_id": f"stream:{stream['stream_id']}",
        "arm": arm,
        "seed": stream["seed"],
        "numerator": metric,
        "denominator": len(stream["probabilities"]),
        "metric": metric,
        "metric_name": "probability_absolute_error",
        "metric_direction": "lower_is_better",
        "censored": False,
        "provenance": "fresh_python_rust_processes_same_serialized_state",
        "typed_decision_mismatches": comparison.get("typed_decision_mismatches"),
        "acknowledgment_match": comparison.get("acknowledgment_match"),
        "state_match": comparison.get("state_match"),
    }


def run_parity_suite(  # pragma: no cover
    root: Path, scratch: Path, *, started: float
) -> tuple[list[JsonDict], JsonDict]:
    """Run all registered streams in two fresh process epochs."""

    streams = seeded_streams()
    rows: list[JsonDict] = []
    comparisons: list[JsonDict] = []
    lifecycle = {
        "duplicate_feedback_match": False,
        "restart_match": False,
        "solver_failure_match": False,
    }
    scratch.mkdir(parents=True, exist_ok=True)
    for epoch, selected in enumerate((streams[:500], streams[500:])):
        workers = {arm: _start_worker(root, arm) for arm in ("python", "rust")}
        try:
            for offset, stream in enumerate(selected):
                absolute = epoch * 500 + offset
                events = stream_events(stream)
                stream_root = scratch / f"stream-{absolute:04d}"
                responses: dict[str, JsonDict] = {}
                for arm in ("python", "rust"):
                    state = stream_root / f"{arm}.json"
                    initialize_state(state)
                    responses[arm], _elapsed = _exchange(
                        workers[arm],
                        {"operation": "trace", "state_path": str(state), "events": events},
                    )
                comparison = compare_worker_responses(responses["python"], responses["rust"])
                comparisons.append(comparison)
                rows.extend(_parity_row(stream, arm, comparison) for arm in ("python", "rust"))
                if absolute == 0:
                    duplicate = {
                        arm: _exchange(
                            workers[arm],
                            {
                                "operation": "trace",
                                "state_path": str(scratch / "stream-0000" / f"{arm}.json"),
                                "events": events,
                            },
                        )[0]
                        for arm in ("python", "rust")
                    }
                    lifecycle["duplicate_feedback_match"] = bool(
                        duplicate["python"].get("ok") is False
                        and duplicate["rust"].get("ok") is False
                        and str(duplicate["python"].get("error", "")).startswith(
                            "duplicate_feedback:"
                        )
                        and str(duplicate["rust"].get("error", "")).startswith(
                            "duplicate_feedback:"
                        )
                    )
                    failures = {
                        arm: _exchange(workers[arm], {"operation": "solver_failure"})[0]
                        for arm in ("python", "rust")
                    }
                    lifecycle["solver_failure_match"] = _failure_match(
                        failures["python"], failures["rust"]
                    )
                if absolute and absolute % 10 == 0:
                    progress(started, "parity", "unit_complete", completed=absolute, planned=1_000)
        finally:
            for worker in workers.values():
                _stop_worker(worker)
    lifecycle["restart_match"] = bool(
        all(comparison.get("passed") is True for comparison in comparisons[499:501])
    )
    maximum = max(float(row.get("max_probability_absolute_error") or 0.0) for row in comparisons)
    summary = {
        "stream_count": len(comparisons),
        "event_count": sum(int(row["denominator"]) for row in rows if row["arm"] == "python"),
        "max_probability_absolute_error": maximum,
        "typed_decision_mismatch_count": sum(
            int(row.get("typed_decision_mismatches") or 0) for row in comparisons
        ),
        "all_acknowledgments_match": all(
            row.get("acknowledgment_match") is True for row in comparisons
        ),
        "all_states_match": all(row.get("state_match") is True for row in comparisons),
        "lifecycle_controls": lifecycle,
        "passed": bool(
            len(comparisons) == PARITY_STREAM_COUNT
            and maximum <= PARITY_TOLERANCE
            and all(row.get("passed") is True for row in comparisons)
            and all(lifecycle.values())
        ),
    }
    return rows, summary


def _service_row(
    arm: str,
    mode: str,
    repeat: int,
    elapsed_ns: int,
    response: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover
    return {
        "row_type": "service_trace",
        "unit_id": f"{mode}:{repeat}",
        "pair_id": f"{mode}:{repeat}",
        "arm": arm,
        "mode": mode,
        "seed": 7_585_000 + repeat,
        "numerator": elapsed_ns,
        "denominator": 1,
        "metric": elapsed_ns,
        "metric_name": "whole_service_ns",
        "metric_direction": "lower_is_better",
        "kernel_ns": int(response["kernel_ns"]),
        "state_bytes": int(response["state_bytes"]),
        "stage_ns": deepcopy(dict(response["stage_ns"])),
        "processed_event_count": int(response["processed_event_count"]),
        "acknowledged_release_count": int(response["acknowledged_release_count"]),
        "reloaded_state_matches": response["reloaded_state_matches"],
        "censored": False,
        "provenance": "fresh_process_paired_trace",
        "durability_policy": response["durability_policy"],
    }


def measure_service_rows(  # pragma: no cover
    root: Path, scratch: Path, *, started: float
) -> list[JsonDict]:
    """Measure 30 fresh processes per arm with cold and warm requests."""

    rows: list[JsonDict] = []
    stream = service_stream()
    scratch.mkdir(parents=True, exist_ok=True)
    for repeat in range(SERVICE_REPEATS):
        for arm in ("python", "rust"):
            repeat_root = scratch / f"repeat-{repeat:02d}-{arm}"
            cold_state = repeat_root / "cold.json"
            warm_state = repeat_root / "warm.json"
            initialize_state(cold_state)
            initialize_state(warm_state)
            cold_events = stream_events(stream, prefix=f"cold-{repeat}-{arm}")
            warm_events = stream_events(stream, prefix=f"warm-{repeat}-{arm}")
            spawned = time.perf_counter_ns()
            worker = _start_worker(root, arm)
            try:
                cold_response, _cold_exchange_ns = _exchange(
                    worker,
                    {"operation": "trace", "state_path": str(cold_state), "events": cold_events},
                )
                cold_ns = time.perf_counter_ns() - spawned
                if cold_response.get("ok") is not True:
                    raise RuntimeError(f"cold_service_failed:{arm}:{cold_response.get('error')}")
                rows.append(_service_row(arm, "cold", repeat, cold_ns, cold_response))
                warm_response, warm_ns = _exchange(
                    worker,
                    {"operation": "trace", "state_path": str(warm_state), "events": warm_events},
                )
                if warm_response.get("ok") is not True:
                    raise RuntimeError(f"warm_service_failed:{arm}:{warm_response.get('error')}")
                rows.append(_service_row(arm, "warm", repeat, warm_ns, warm_response))
            finally:
                _stop_worker(worker)
        progress(started, "service", "unit_complete", completed=repeat + 1, planned=SERVICE_REPEATS)
    return rows


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile_requires_values")
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    durations = [float(row["metric"]) for row in rows]
    kernels = [float(row["kernel_ns"]) for row in rows]
    state_bytes = [int(row["state_bytes"]) for row in rows]
    return {
        "p50_ns": _percentile(durations, 0.50),
        "p95_ns": _percentile(durations, 0.95),
        "kernel_p50_ns": _percentile(kernels, 0.50),
        "kernel_p95_ns": _percentile(kernels, 0.95),
        "state_bytes_p50": _percentile(state_bytes, 0.50),
    }


def _paired_ratio(rows: Sequence[Mapping[str, Any]], field: str) -> JsonDict:
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        grouped.setdefault(str(row["pair_id"]), {})[str(row["arm"])] = float(row[field])
    ratios = [arms["python"] / arms["rust"] for arms in grouped.values()]
    return {
        "estimate": statistics.median(ratios),
        "lower95": _percentile(ratios, 0.025),
        "upper95": _percentile(ratios, 0.975),
        "pair_count": len(ratios),
        "direction": "python_over_rust_higher_favors_rust",
    }


def summarize_service_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce only complete, paired, equal-durability service rows."""

    policies = {row.get("durability_policy") for row in rows}
    if policies != {DURABILITY_POLICY}:
        raise ValueError("durability_policy_mismatch")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("mode")), str(row.get("arm"))), []).append(row)
    if set(grouped) != {(mode, arm) for mode in ("cold", "warm") for arm in ("python", "rust")}:
        raise ValueError("service_arm_or_mode_missing")
    pair_counts: dict[str, int] = {}
    result: JsonDict = {}
    for mode in ("cold", "warm"):
        mode_rows = [row for row in rows if row.get("mode") == mode]
        pairs: dict[str, set[str]] = {}
        for row in mode_rows:
            pairs.setdefault(str(row.get("pair_id")), set()).add(str(row.get("arm")))
        if len(pairs) != SERVICE_REPEATS or any(
            arms != {"python", "rust"} for arms in pairs.values()
        ):
            raise ValueError("service_pair_incomplete")
        pair_counts[mode] = len(pairs)
        result[mode] = {
            "python": _latency_summary(grouped[(mode, "python")]),
            "rust": _latency_summary(grouped[(mode, "rust")]),
            "whole_service_speedup": _paired_ratio(mode_rows, "metric"),
            "kernel_speedup": _paired_ratio(mode_rows, "kernel_ns"),
        }
    warm_python = grouped[("warm", "python")]
    replaceable = statistics.median(
        float(row["kernel_ns"]) / float(row["metric"]) for row in warm_python
    )
    ideal = 1.0 / (1.0 - min(replaceable, 1.0 - 1e-12))
    result["paired_repeat_count"] = pair_counts
    result["amdahl_ceiling"] = {
        "measured_replaceable_fraction": replaceable,
        "ideal_update_only_speedup": ideal,
        "denominator": "measured_whole_service_ns",
    }
    return result


def _receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    expected = {*validation_scope.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in receipts:
        grouped.setdefault(str(row.get("name") or ""), []).append(row)
    return all(
        len(grouped.get(name, [])) == 1
        and grouped[name][0].get("passed") is True
        and grouped[name][0].get("exit_code") == 0
        and grouped[name][0].get("timed_out") is not True
        for name in expected
    )


def _board_complete(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(
        [row.get("board") for row in rows] == ["KV260", "PolarFire", "GateMate"]
        and rows[0].get("future_access") == "ssh kria only"
        and rows[0].get("architecture_limit") == "k_max<=5"
        and rows[0].get("historical_execution_venue") == "kv260_fpga_fabric"
        and rows[1].get("fpga_sampling_claimed") is False
        and rows[1].get("historical_execution_venue") == "polarfire_linux_cpu"
        and rows[2].get("current_disposition") == "blocked_unchanged_physical_prerequisite"
        and all(row.get("hardware_operations_issued") == [] for row in rows)
        and all(row.get("present_reachability_asserted") is False for row in rows)
    )


def _parity_complete(rows: Sequence[Mapping[str, Any]]) -> bool:
    if not rows:
        return False
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("pair_id")), {})[str(row.get("arm"))] = row
    return bool(
        grouped
        and all(set(arms) == {"python", "rust"} for arms in grouped.values())
        and all(
            float(arms["rust"].get("metric") or 0.0) <= PARITY_TOLERANCE
            and int(arms["rust"].get("typed_decision_mismatches") or 0) == 0
            and arms["rust"].get("acknowledgment_match") is True
            and arms["rust"].get("state_match", True) is True
            for arms in grouped.values()
        )
    )


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute parity, service, board, validation, and benefit gates."""

    rows = list(value.get("rows") or [])
    parity_rows = [row for row in rows if row.get("row_type") == "parity_stream"]
    service_rows = [row for row in rows if row.get("row_type") == "service_trace"]
    parity = _parity_complete(parity_rows)
    service = False
    summary: JsonDict = {}
    if service_rows:
        try:
            summary = summarize_service_rows(service_rows)
            service = True
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            service = False
    boards = _board_complete(list(value.get("board_rows") or []))
    validation = _receipts_pass(list(value.get("validation_receipts") or []))
    benefit = bool(
        parity
        and service
        and float(summary.get("warm", {}).get("whole_service_speedup", {}).get("lower95", 0.0))
        > 1.0
    )
    return {
        "portable_parity_score": int(parity),
        "service_measurement_complete_score": int(service),
        "board_continuity_complete_score": int(boards),
        "required_validation_passed": validation,
        "whole_service_benefit": benefit,
        "service_summary": summary,
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
    op: str = "eq",
) -> JsonDict:
    passed = observed == expected if op == "eq" else bool(float(observed) > float(expected))
    return {
        "check": check,
        "category": category,
        "upstream": upstream,
        "path": path,
        "field": field,
        "op": op,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": passed,
        "principle": "Validity, readiness, and benefit are independent terminal checks.",
    }


def acceptance_gates(value: Mapping[str, Any], reduction: Mapping[str, Any]) -> list[JsonDict]:
    blocker = value.get("external_blocker")
    readiness_observed = 0 if blocker else reduction["portable_parity_score"]
    lower = (
        reduction.get("service_summary", {})
        .get("warm", {})
        .get("whole_service_speedup", {})
        .get("lower95")
    )
    return [
        _gate(
            "kernel_numerical_readiness",
            "readiness",
            1,
            readiness_observed,
            upstream="Exp7574 and current parity",
            path="portable_parity_score",
            field="portable_parity_score",
        ),
        _gate(
            "equal_durability_service_complete",
            "validity",
            1,
            reduction["service_measurement_complete_score"],
            upstream="current work",
            path="rows",
            field="service_measurement_complete_score",
        ),
        _gate(
            "board_continuity_complete",
            "readiness",
            1,
            reduction["board_continuity_complete_score"],
            upstream="Exp7571",
            path="board_rows",
            field="board_continuity_complete_score",
        ),
        _gate(
            "required_validation",
            "validity",
            True,
            reduction["required_validation_passed"],
            upstream="current work",
            path="validation_receipts",
            field="required_validation_passed",
        ),
        _gate(
            "whole_service_improvement_lower95",
            "benefit",
            1.0,
            lower,
            upstream="current service rows",
            path="whole_service_speedup.warm",
            field="lower95",
            op="gt" if lower is not None else "eq",
        ),
    ]


def field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Carry the one-line reporting principles next to emitted fields."""

    specific = {
        "honest_verdict": "Use a complete_ terminal prefix; completion does not establish benefit.",
        "verdict_class": "Use exactly one closed class; partial is only unfinished owned work.",
        "flagged_adversarial": "Persist the exact terminal verifier outcome; flagged evidence opens no gate.",
        "gate_check_summary": "Every block names upstream, path, field, operator, expected, and observed values.",
        "acceptance_gate_results": "Separate validity, readiness, and benefit so a valid null stays usable.",
        "rows": "Retain one arm row per unit with raw operands, direction, seed, censoring, and provenance.",
        "inference_substrate_class": "Record actual and planned substrate classes separately.",
        "MODEL_SPECS": "No current LLM task means an empty mandated-model list and zero current calls.",
        "invocation_counts": "Count current loads, forwards, generations, calls, and tokens independently.",
        "duration_s": "Use monotonic current work and exclude inherited timing and artificial sleeps.",
        "source_artifact_hashes": "Bind each conclusion to exact bytes and distinguish absent producer evidence.",
        "validation_receipts": "Bind each check to command, worktree, exit code, and log hash.",
        "field_principles": "Keep the reporting rule beside every emitted field.",
        "verifier_is_oracle": "Oracle controls cannot support an oracle-distinct positive claim.",
        "board_continuity_complete_score": "One requires all three dated board states and next changes.",
        "portable_parity_score": "One requires measured numerical, decision, and persistence agreement.",
        "service_measurement_complete_score": "One requires equal-durability complete-service traces.",
        "whole_service_speedup": "Use complete service time and never isolated update time.",
        "board_rows": "Keep historical graduation, current observation, and physical prerequisites distinct.",
        "kernel_branch_disposition": "Record numerical readiness independently from board continuity.",
    }
    return {
        key: specific.get(key, f"Retain {key} so scope or evidence cannot disappear silently.")
        for key in keys
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable evidence while excluding clocks and this self-reference."""

    excluded = {
        "reproducibility_checksum",
        "duration_s",
        "phase_spans",
        "started_at_utc",
        "completed_at_utc",
        "process_identity",
    }
    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key not in excluded}
    )


def _base_artifact(
    *,
    root: Path,
    board_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    duration_s: float,
    evidence_mode: str,
    blocker: Mapping[str, Any] | None = None,
) -> JsonDict:
    value: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "worktree_root": str(root.resolve()),
        "evidence_mode": evidence_mode,
        "MODEL_SPECS": [],
        "model_specs": [],
        "no_model_load": True,
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate_class": "no_model_load",
        "inference_substrate": "host_rust_process_and_python_process",
        "planned_inference_substrate_class": "prospective_fpga_local_basis_arithmetic",
        "execution_venue": "host",
        "host_rust_kernel_is_fpga_execution": False,
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "fresh_confirmatory_claim_allowed": False,
        "external_blocker": deepcopy(dict(blocker)) if blocker else None,
        "kernel_branch_disposition": (
            "blocked_numerical_prerequisite" if blocker else "measured_host_process_boundary"
        ),
        "rows": deepcopy(list(rows)),
        "board_rows": deepcopy(list(board_rows)),
        "validation_receipts": deepcopy(list(validation_receipts)),
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "duration_s": float(duration_s),
        "hardware_operations_issued": [],
        "hardware_smoke": False,
        "e2e_applicability": {
            "capability_e2e": "predict_release_update_persist_reload",
            "E2E-003": "not_applicable_no_shared_binding_change",
            "E2E-004": "not_applicable_process_cli_uses_existing_json_state",
            "E2E-001_002": "not_applicable_no_sampler_change",
            "E2E-009_013": "not_applicable_no_arc_change",
        },
    }
    reduction = independent_reduce(value)
    value["independent_reduction"] = reduction
    value["portable_parity_score"] = reduction["portable_parity_score"]
    value["service_measurement_complete_score"] = reduction["service_measurement_complete_score"]
    value["board_continuity_complete_score"] = reduction["board_continuity_complete_score"]
    service_summary = reduction["service_summary"]
    value["whole_service_speedup"] = {
        mode: deepcopy(service_summary.get(mode, {}).get("whole_service_speedup"))
        for mode in ("cold", "warm")
    }
    value["kernel_speedup"] = {
        mode: deepcopy(service_summary.get(mode, {}).get("kernel_speedup"))
        for mode in ("cold", "warm")
    }
    value["service_latency_summary"] = {
        mode: {arm: deepcopy(service_summary.get(mode, {}).get(arm)) for arm in ("python", "rust")}
        for mode in ("cold", "warm")
    }
    value["amdahl_ceiling"] = deepcopy(service_summary.get("amdahl_ceiling"))
    value["acceptance_gate_results"] = acceptance_gates(value, reduction)
    value["gate_check_summary"] = deepcopy(dict(blocker)) if blocker else None
    if blocker:
        value["verdict_class"] = "blocked"
        value["honest_verdict"] = f"complete_blocked_{blocker['check']}"
    elif reduction["whole_service_benefit"] and reduction["required_validation_passed"]:
        value["verdict_class"] = "positive"
        value["honest_verdict"] = "complete_positive_portable_service_improvement"
    else:
        value["verdict_class"] = "null"
        value["honest_verdict"] = "complete_null_portable_service_benefit_not_established"
    value["positive_portability"] = value["verdict_class"] == "positive"
    value["hardware_expansion_recommendation"] = (
        "defer_no_measured_end_to_end_headroom"
        if not reduction["whole_service_benefit"]
        or float((value.get("amdahl_ceiling") or {}).get("ideal_update_only_speedup", 1.0)) < 2.0
        else "consider_only_after_separate_board_authorization"
    )
    value["field_principles"] = field_principles(
        [*value, "field_principles", "reproducibility_checksum"]
    )
    value["reproducibility_checksum"] = reproducibility_checksum(value)
    return value


def _fixture_boards() -> list[JsonDict]:
    portable = load_object(REPO_ROOT / PORTABLE_PATH)
    return build_board_rows(portable)


def build_test_artifact(
    root: Path,
    *,
    service_rows: Sequence[Mapping[str, Any]],
    parity_rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build compact valid evidence for pure reader and mutation tests."""

    return _base_artifact(
        root=root,
        board_rows=_fixture_boards(),
        rows=[*parity_rows, *service_rows],
        validation_receipts=validation_receipts,
        source_hashes={},
        duration_s=0.1,
        evidence_mode="private_fixture",
    )


def build_blocked_artifact(
    blocker: Mapping[str, Any], root: Path, validation_receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Close only the numerical branch while retaining board continuity."""

    return _base_artifact(
        root=root,
        board_rows=_fixture_boards(),
        rows=[],
        validation_receipts=validation_receipts,
        source_hashes={},
        duration_s=0.1,
        evidence_mode="private_blocked_fixture",
        blocker=blocker,
    )


def _verify_sources(value: Mapping[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    for label, row in dict(value.get("source_artifact_hashes") or {}).items():
        path = Path(str(row.get("path") or label))
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file() or sha256_file(resolved) != row.get("sha256"):
            errors.append(f"source_hash_invalid:{label}")
        elif resolved.stat().st_size != row.get("bytes"):
            errors.append(f"source_size_invalid:{label}")
    return errors


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, verify_sources: bool = True
) -> list[str]:
    """Reject identity, raw reduction, claim, custody, or receipt drift."""

    errors: list[str] = []
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("artifact_identity_mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("task_binding_mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        errors.append("model_specs_not_empty")
    if (
        value.get("no_model_load") is not True
        or value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("current_invocation_claim_invalid")
    if (
        value.get("inference_substrate_class") != "no_model_load"
        or value.get("execution_venue") != "host"
        or value.get("host_rust_kernel_is_fpga_execution") is not False
    ):
        errors.append("inference_declaration_invalid")
    for field in (
        "portable_parity_score",
        "service_measurement_complete_score",
        "board_continuity_complete_score",
    ):
        if type(value.get(field)) is not int or value.get(field) not in (0, 1):
            errors.append(f"score_not_bare_numeric:{field}")
    try:
        reduction = independent_reduce(value)
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError):
        reduction = {}
        errors.append("independent_reduction_failed")
    if reduction:
        if value.get("independent_reduction") != reduction:
            errors.append("independent_reduction_mismatch")
        for field in (
            "portable_parity_score",
            "service_measurement_complete_score",
            "board_continuity_complete_score",
        ):
            if value.get(field) != reduction[field]:
                errors.append(f"{field}_mismatch")
        expected_gates = acceptance_gates(value, reduction)
        if value.get("acceptance_gate_results") != expected_gates:
            errors.append("acceptance_gate_results_mismatch")
        expected_speedup = {
            mode: deepcopy(reduction["service_summary"].get(mode, {}).get("whole_service_speedup"))
            for mode in ("cold", "warm")
        }
        if value.get("whole_service_speedup") != expected_speedup:
            errors.append("whole_service_speedup_mismatch")
    blocker = value.get("external_blocker")
    if blocker is not None:
        required = {"check", "upstream", "path", "field", "op", "expected", "observed"}
        if not isinstance(blocker, Mapping) or not required <= set(blocker):
            errors.append("external_blocker_incomplete")
        else:
            if value.get("gate_check_summary") != blocker:
                errors.append("gate_check_summary_mismatch")
            if value.get("verdict_class") != "blocked" or value.get("honest_verdict") != (
                f"complete_blocked_{blocker['check']}"
            ):
                errors.append("terminal_verdict_mismatch")
            if value.get("kernel_branch_disposition") != "blocked_numerical_prerequisite":
                errors.append("kernel_branch_disposition_mismatch")
            if (
                value.get("portable_parity_score") != 0
                or value.get("service_measurement_complete_score") != 0
            ):
                errors.append("blocked_measurement_must_be_unstarted")
    elif reduction:
        expected_class = (
            "positive"
            if reduction["whole_service_benefit"] and reduction["required_validation_passed"]
            else "null"
        )
        if value.get("verdict_class") != expected_class:
            errors.append("terminal_verdict_mismatch")
        if value.get("gate_check_summary") is not None:
            errors.append("unblocked_gate_summary_must_be_null")
    if not _receipts_pass(list(value.get("validation_receipts") or [])):
        errors.append("required_validation_failed")
    if value.get("hardware_operations_issued") != [] or value.get("hardware_smoke") is not False:
        errors.append("unauthorized_hardware_claim")
    principles = value.get("field_principles") or {}
    if set(value) - set(principles):
        errors.append("field_principles_incomplete")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    if verify_sources:
        errors.extend(_verify_sources(value, root))
    return list(dict.fromkeys(errors))


def cold_replay(path: Path, *, root: Path, verify_sources: bool = True) -> list[str]:
    value = load_object(path)
    return (
        ["artifact_not_object"]
        if not value
        else validate_artifact(value, root=root, verify_sources=verify_sources)
    )


def independent_replay(path: Path, *, root: Path, verify_sources: bool = True) -> list[str]:
    value = load_object(path)
    if not value:
        return ["artifact_not_object"]
    reduced = independent_reduce(value)
    errors = validate_artifact(value, root=root, verify_sources=verify_sources)
    return (
        errors
        if reduced == value.get("independent_reduction")
        else [*errors, "independent_reduction_mismatch"]
    )


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze focused tests, coverage, formatting, typing, and spec checks."""

    commands = build_command_plan(root, AFFECTED_MANIFEST, private_root)
    errors = validate_command_plan(root, AFFECTED_MANIFEST, commands)
    if errors:  # pragma: no cover - the shared builder is frozen and tested separately.
        raise ValueError("validation_plan_invalid:" + ",".join(errors))
    for path in private_basetemps(commands):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return list(commands)


def private_basetemps(commands: Sequence[validation_scope.CommandSpec]) -> list[str]:
    """Return private pytest targets whose parents must exist before launch."""

    return [
        argument.split("=", 1)[1]
        for command in commands
        for argument in command.argv
        if argument.startswith("--basetemp=")
    ]


def terminal_commands(candidate: Path, root: Path) -> list[validation_scope.CommandSpec]:
    """Build four exact readers for one immutable candidate path."""

    common = ("--date", RUN_DATE, "--root", str(root.resolve()))
    return [
        validation_scope.CommandSpec(
            TERMINAL_CHECK_NAMES[0],
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--cold-replay",
                str(candidate),
            ),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            TERMINAL_CHECK_NAMES[1],
            (
                ".venv/bin/python",
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
            (".venv/bin/python", "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            TERMINAL_CHECK_NAMES[3],
            (
                ".venv/bin/python",
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact_terminal_candidate",
            300.0,
        ),
    ]


def _provisional_terminal_receipts() -> list[JsonDict]:  # pragma: no cover
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
        }
        for name in TERMINAL_CHECK_NAMES
    ]


def _run_commands(
    root: Path,
    commands: Sequence[validation_scope.CommandSpec],
    log_dir: Path,
) -> list[JsonDict]:  # pragma: no cover
    """Run bounded children with each command's private environment."""

    receipts: list[JsonDict] = []
    for index, command in enumerate(commands):
        for target in private_basetemps([command]):
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        command_env = dict(getattr(command, "command_environment", ()))
        batch = validation_scope.run_commands(
            root,
            [command],
            log_dir=log_dir / f"{index:02d}_{command.name}",
            extra_env=command_env,
            heartbeat_s=60.0,
        )
        receipts.extend(batch)
    return receipts


def _extra_sources(
    root: Path, manifest_path: Path, raw_paths: Sequence[Path]
) -> dict[str, JsonDict]:  # pragma: no cover
    sources: dict[str, JsonDict] = {}
    for relative in (
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        RUST_SOURCE_PATH,
        CL_SPEC_PATH,
        REPORT_SPEC_PATH,
    ):
        sources[relative.as_posix()] = source_row(root / relative, root)
    for path in (manifest_path, *raw_paths):
        sources[path.relative_to(root).as_posix()] = source_row(path, root)
    return sources


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every phase and slow-operation boundary with monotonic time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7585] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(  # pragma: no cover
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_offset_s": phase_started - run_started,
        "end_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
    }


def _add_artifact_extras(  # pragma: no cover
    value: JsonDict, extras: Mapping[str, Any]
) -> JsonDict:
    value.update(deepcopy(dict(extras)))
    value["field_principles"] = field_principles(
        [*value, "field_principles", "reproducibility_checksum"]
    )
    value["reproducibility_checksum"] = reproducibility_checksum(value)
    return value


def run_experiment(  # pragma: no cover - declared entrypoint is the capability E2E.
    root: Path, run_date: str, *, output_path: Path | None = None
) -> JsonDict:
    """Build, measure, validate, and atomically publish exact evidence."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    repo = root.resolve()
    destination = output_path or repo / RESULT_PATH
    raw_root = repo / RAW_DIR
    raw_root.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(prefix="carnot-exp7585-work-", dir="/tmp"))
    candidate_path = raw_root / "measured_terminal_candidate.json"
    exact_path = raw_root / "exact_terminal_candidate.json"
    parity_path = raw_root / "parity_rows.json"
    service_path = raw_root / "service_rows.json"
    manifest_path = raw_root / "affected_validation_manifest.json"
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: list[JsonDict] = []

    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    context = collect_preconditions(repo)
    boards = build_board_rows(context["portable_artifact"])
    atomic_json(
        raw_root / "preconditions_checkpoint.json",
        {"rows": context["rows"], "blocker": context["blocker"], "board_rows": boards},
    )
    spans.append(_span("preconditions", phase_started, started, len(context["rows"])))
    progress(
        started,
        "preconditions",
        "complete",
        completed=len(context["rows"]),
        kernel_branch_ready=context["kernel_branch_ready"],
    )

    for phase in ("model_load", "generation"):
        progress(started, phase, "before", planned=0)
        phase_started = time.monotonic()
        spans.append(_span(phase, phase_started, started, 0))
        progress(started, phase, "after", completed=0)

    parity_rows: list[JsonDict] = []
    service_rows: list[JsonDict] = []
    parity_summary: JsonDict = {
        "stream_count": 0,
        "passed": False,
        "disposition": "blocked_numerical_prerequisite",
    }
    rust_receipts: list[JsonDict] = []
    if context["kernel_branch_ready"]:
        rust_commands = [
            validation_scope.CommandSpec(
                "rust_source_format",
                ("rustfmt", "--edition", "2021", "--check", RUST_SOURCE_PATH.as_posix()),
                "changed_rust_source",
                300.0,
            ),
            validation_scope.CommandSpec(
                "rust_release_build",
                (
                    "cargo",
                    "build",
                    "--release",
                    "-p",
                    "carnot-core",
                    "--bin",
                    "portable-recalibration-service",
                ),
                "changed_rust_binary",
                900.0,
            ),
        ]
        progress(started, "rust_build", "before_subprocesses", planned=len(rust_commands))
        phase_started = time.monotonic()
        rust_receipts = _run_commands(repo, rust_commands, raw_root / "validation/rust")
        spans.append(_span("rust_build", phase_started, started, len(rust_receipts)))
        progress(
            started,
            "rust_build",
            "after_subprocesses",
            completed=len(rust_receipts),
            passed=all(row.get("passed") is True for row in rust_receipts),
        )
        if not all(row.get("passed") is True for row in rust_receipts):
            raise RuntimeError("rust_build_or_format_failed")

        progress(started, "parity", "before", planned=PARITY_STREAM_COUNT)
        phase_started = time.monotonic()
        parity_rows, parity_summary = run_parity_suite(repo, work_root / "parity", started=started)
        spans.append(_span("parity", phase_started, started, PARITY_STREAM_COUNT))
        progress(
            started,
            "parity",
            "after",
            completed=PARITY_STREAM_COUNT,
            passed=parity_summary["passed"],
        )
        atomic_json(parity_path, {"summary": parity_summary, "rows": parity_rows})
        if parity_summary["passed"] is not True:
            raise RuntimeError("portable_parity_failed")

        progress(started, "service", "before", planned=SERVICE_REPEATS)
        phase_started = time.monotonic()
        service_rows = measure_service_rows(repo, work_root / "service", started=started)
        service_summary = summarize_service_rows(service_rows)
        spans.append(_span("service", phase_started, started, SERVICE_REPEATS))
        progress(
            started,
            "service",
            "after",
            completed=SERVICE_REPEATS,
            warm_lower95=service_summary["warm"]["whole_service_speedup"]["lower95"],
        )
        atomic_json(service_path, {"summary": service_summary, "rows": service_rows})
    else:
        progress(started, "parity", "before", planned=PARITY_STREAM_COUNT)
        progress(started, "parity", "after", completed=0, unstarted=PARITY_STREAM_COUNT)
        progress(started, "service", "before", planned=SERVICE_REPEATS)
        progress(started, "service", "after", completed=0, unstarted=SERVICE_REPEATS)

    shutil.rmtree(work_root)

    progress(started, "manifest", "start")
    atomic_json(
        manifest_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "test_paths": list(AFFECTED_MANIFEST.test_paths),
            "changed_modules": list(AFFECTED_MANIFEST.changed_modules),
            "python_static_paths": list(AFFECTED_MANIFEST.static_paths),
            "rust_paths": [RUST_SOURCE_PATH.as_posix()],
            "capability_specs": [CL_SPEC_PATH.as_posix(), REPORT_SPEC_PATH.as_posix()],
        },
    )
    raw_sources = [path for path in (parity_path, service_path) if path.is_file()]
    sources = deepcopy(context["source_artifact_hashes"])
    sources.update(_extra_sources(repo, manifest_path, raw_sources))
    progress(started, "manifest", "complete", files=len(sources))

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7585-validation-", dir="/tmp"))
    commands = build_validation_commands(repo, private_root)
    progress(started, "scoped_validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic()
    affected = _run_commands(repo, commands, raw_root / "validation/affected")
    spans.append(_span("scoped_validation", phase_started, started, len(affected)))
    affected_summary = validation_scope.reduce_required_checks(affected)
    progress(
        started,
        "scoped_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=affected_summary["required_checks_passed"],
    )
    if affected_summary["required_checks_passed"] is not True:
        raise RuntimeError("required_scoped_validation_failed")

    rows = [*parity_rows, *service_rows]
    common_extras: JsonDict = {
        "preconditions_checked": deepcopy(context["rows"]),
        "parity_summary": deepcopy(parity_summary),
        "rust_validation_receipts": deepcopy(rust_receipts),
        "raw_measurement_receipts": {
            path.relative_to(repo).as_posix(): source_row(path, repo) for path in raw_sources
        },
        "affected_file_validation_manifest": source_row(manifest_path, repo),
        "phase_spans": spans,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "process_identity": {"pid": os.getpid(), "python": sys.executable},
        "sample_size_budget": {
            "parity_streams": {
                "planned": PARITY_STREAM_COUNT,
                "completed": len(parity_rows) // 2,
                "unstarted": PARITY_STREAM_COUNT - len(parity_rows) // 2,
            },
            "service_pairs": {
                "planned": SERVICE_REPEATS * 2,
                "completed": len(service_rows) // 2,
                "unstarted": SERVICE_REPEATS * 2 - len(service_rows) // 2,
            },
        },
        "prospective_fpga_mapping": {
            "mapped_operations": [
                "two-knot local basis evaluation",
                "nine-by-nine sufficient-statistic accumulation",
                "bounded monotone map update",
            ],
            "execution_measured": False,
            "claim_scope": "prospective_only",
        },
        "prior_verdict_retirement": {
            "literal_verdict": "complete_blocked_exp7561_recalibration_ready_score",
            "retired": context["kernel_branch_ready"],
            "narrow_scope": "Exp7571 numerical precondition before Exp7574 requalification",
            "scientific_hypothesis_retired": False,
        },
    }

    def build_current(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
        artifact = _base_artifact(
            root=repo,
            board_rows=boards,
            rows=rows,
            validation_receipts=receipts,
            source_hashes=sources,
            duration_s=time.monotonic() - started,
            evidence_mode="measured_host_processes"
            if not context["blocker"]
            else "blocked_pre_gate",
            blocker=context["blocker"],
        )
        return _add_artifact_extras(artifact, common_extras)

    provisional = build_current([*rust_receipts, *affected, *_provisional_terminal_receipts()])
    progress(started, "candidate", "before_serialization", path=candidate_path)
    atomic_json(candidate_path, provisional)
    progress(started, "candidate", "after_serialization", bytes=candidate_path.stat().st_size)

    first_plan = terminal_commands(candidate_path, repo)
    progress(started, "terminal_validation", "before_subprocesses", planned=len(first_plan))
    phase_started = time.monotonic()
    terminal = _run_commands(repo, first_plan, raw_root / "validation/terminal_provisional")
    spans.append(_span("terminal_validation", phase_started, started, len(terminal)))
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=all(row.get("passed") is True for row in terminal),
    )
    if not all(row.get("passed") is True for row in terminal):
        raise RuntimeError("terminal_candidate_validation_failed")

    common_extras["phase_spans"] = spans
    common_extras["completed_at_utc"] = datetime.now(UTC).isoformat()
    final = build_current([*rust_receipts, *affected, *terminal])
    errors = validate_artifact(final, root=repo)
    if errors:
        raise RuntimeError("terminal_artifact_invalid:" + ",".join(errors))
    atomic_json(exact_path, final)

    exact_plan = terminal_commands(exact_path, repo)
    progress(started, "exact_candidate_validation", "before_subprocesses", planned=len(exact_plan))
    exact = _run_commands(repo, exact_plan, raw_root / "validation/terminal_exact")
    progress(
        started,
        "exact_candidate_validation",
        "after_subprocesses",
        completed=len(exact),
        passed=all(row.get("passed") is True for row in exact),
    )
    if not all(row.get("passed") is True for row in exact):
        raise RuntimeError("exact_terminal_candidate_validation_failed")
    atomic_json(raw_root / "exact_terminal_validation_receipts.json", {"receipts": exact})

    progress(started, "publish", "before_atomic_terminal", path=destination)
    atomic_json(destination, final)
    if sha256_file(destination) != sha256_file(exact_path):
        raise RuntimeError("published_bytes_differ_from_exact_candidate")
    progress(
        started,
        "publish",
        "after_atomic_terminal",
        bytes=destination.stat().st_size,
        verdict=final["honest_verdict"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed date, producer path, worker, and read-only modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--service-worker", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--no-source-check", action="store_true")
    args = parser.parse_args(argv)
    if args.date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    args.root = args.root.resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the producer, durable worker, or one strict read-only reader."""

    args = parse_args(argv)
    if args.service_worker:
        return service_worker_loop(sys.stdin, sys.stdout)  # pragma: no cover
    if args.cold_replay:
        errors = cold_replay(
            args.cold_replay,
            root=args.root,
            verify_sources=not args.no_source_check,
        )
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce:
        errors = independent_replay(
            args.independent_reduce,
            root=args.root,
            verify_sources=not args.no_source_check,
        )
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    output = (  # pragma: no cover - producer path is the declared capability E2E.
        args.output if args.output.is_absolute() else args.root / args.output
    )
    run_experiment(args.root, args.date, output_path=output)  # pragma: no cover
    return 0  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

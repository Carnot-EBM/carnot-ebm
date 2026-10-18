"""Qualify the direct PyO3 boundary to the durable Rust service core.

The experiment measures compatibility and durability. It does not measure a
speed benefit, change the default service, or make a learned-head claim.

Spec: REQ-REPORT-7626 and SCENARIO-REPORT-7626-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from typing import Any

from carnot import experiment_7585_v662_portable_service as python_service
from carnot.pipeline.calibrated_decision_service import (
    DURABILITY_POLICY,
    CalibratedDecision,
    CalibratedDecisionService,
    FeedbackAcknowledgment,
    frozen_decision_costs,
)
from carnot.pipeline.native_calibrated_decision_service import (
    NativeServiceClient,
    load_native_extension,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
RUN_DATE = "20260924"
MILESTONE = "2026.09.665"
EXPERIMENT_ID = "exp7626-v665-native-service"
SCHEMA = "carnot.exp7626.v665.native_service.v1"
RESULT_PATH = Path("results/experiment_7626_v665_native_service.json")
RAW_DIR = Path("results/raw/experiment_7626_v665_native_service")
EXP7598_PATH = Path("results/experiment_7598_v663_rust_consumer.json")
MODULE_PATH = Path("python/carnot/experiment_7626_v665_native_service.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7626_v665_native_service.py")
TEST_PATH = Path("tests/python/test_experiment_7626_v665_native_service.py")
CORE_PATH = Path("crates/carnot-core/src/portable_recalibration.rs")
BINARY_PATH = Path("crates/carnot-core/src/bin/portable-recalibration-service.rs")
BINDING_PATH = Path("crates/carnot-python/src/portable_recalibration.rs")
RUST_LIB_PATH = Path("crates/carnot-python/src/lib.rs")
REPORT_SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RUST_BINARY = Path("target/release/portable-recalibration-service")
MODEL_SPECS: list[JsonDict] = []
PARITY_TOLERANCE = 1e-10
INTERRUPTED_WRITE_EXIT = 86
FROZEN_WORKLOAD_SHA256 = "sha256:c0e88e5c07f43fb10983abc15ab98280e9e7c8a1f79c946cac65354f81a48265"
ARMS = ("python_service", "rust_jsonl", "direct_rust_binding")
ZERO_INVOCATIONS = {
    "model_loads": 0,
    "forward_calls": 0,
    "generation_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
}


def frozen_workload() -> list[JsonDict]:
    """Rebuild the fixed Exp7598 event generator without reading outcomes."""

    rows: list[JsonDict] = []
    for seed in (7_598_101, 7_598_102):
        generator = random.Random(seed)
        events = [
            {
                "event_id": f"event-{seed}-{index}",
                "probability": generator.uniform(0.001, 0.999),
                "label": generator.randrange(2),
            }
            for index in range(4)
        ]
        rows.append(
            {
                "unit_id": f"exp7598-seed-{seed}",
                "seed": seed,
                "historical_source": EXP7598_PATH.as_posix(),
                "events": events,
            }
        )
    return rows


def workload_checksum(workload: Sequence[Mapping[str, Any]]) -> str:
    """Bind every fixed event and seed to one stable workload identity."""

    return canonical_hash(list(workload))


def synthetic_parity_rows() -> list[JsonDict]:
    """Return a complete small fixture for reducer and mutation tests."""

    rows: list[JsonDict] = []
    for unit in frozen_workload():
        for index, event in enumerate(unit["events"]):
            for arm in ARMS:
                rows.append(
                    {
                        "unit_id": unit["unit_id"],
                        "arm": arm,
                        "sequence_index": index,
                        "request_kind": "predict_update",
                        "input_probability": event["probability"],
                        "output_probability": event["probability"],
                        "action": "escalate",
                        "acknowledged": True,
                        "durable": True,
                        "sample_count_after": index + 1,
                        "absolute_metric": event["probability"],
                        "numerator": 0.0,
                        "denominator": 1,
                        "seed": unit["seed"],
                        "direction": "exact_parity",
                        "censored": False,
                        "raw_provenance": unit["historical_source"],
                    }
                )
        for arm in ARMS:
            rows.append(
                {
                    "unit_id": unit["unit_id"],
                    "arm": arm,
                    "sequence_index": 4,
                    "request_kind": "cold_reload_predict",
                    "input_probability": 0.37,
                    "output_probability": 0.37,
                    "action": "escalate",
                    "acknowledged": None,
                    "durable": True,
                    "sample_count_after": 4,
                    "absolute_metric": 0.37,
                    "numerator": 0.0,
                    "denominator": 1,
                    "seed": unit["seed"],
                    "direction": "exact_parity",
                    "censored": False,
                    "raw_provenance": unit["historical_source"],
                }
            )
    return rows


def reduce_parity_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Require every workload request, caller, typed value, and durable reload."""

    expected = len(frozen_workload()) * 5 * len(ARMS)
    if len(rows) != expected:
        raise ValueError(f"parity_row_count:{len(rows)}!={expected}")
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["unit_id"]), int(row["sequence_index"]), str(row["request_kind"]))
        grouped.setdefault(key, []).append(row)
    maximum = 0.0
    typed_mismatches = 0
    numeric_mismatches = 0
    for key, group in grouped.items():
        if {str(row.get("arm")) for row in group} != set(ARMS) or len(group) != len(ARMS):
            raise ValueError(f"parity_arms:{key}")
        actions = {str(row.get("action")) for row in group}
        if len(actions) != 1:
            typed_mismatches += 1
        values = [float(row["output_probability"]) for row in group]
        delta = max(values) - min(values)
        maximum = max(maximum, abs(delta))
        if abs(delta) > PARITY_TOLERANCE:
            numeric_mismatches += 1
        if key[2] == "predict_update" and not all(
            row.get("acknowledged") is True and row.get("durable") is True for row in group
        ):
            raise ValueError(f"durable_update_missing:{key}")
    if typed_mismatches:
        raise ValueError("typed_decision_mismatch")
    if numeric_mismatches:
        raise ValueError("numeric_parity_mismatch")
    return {
        "complete": True,
        "independent_units": len(frozen_workload()),
        "request_count": len(rows),
        "typed_mismatch_count": 0,
        "numeric_mismatch_count": 0,
        "max_abs_delta": maximum,
        "durable_reload_match": all(row.get("durable") is True for row in rows),
    }


def synthetic_durability_rows() -> list[JsonDict]:
    """Return acknowledged, reload, and interrupted-write fixture events."""

    rows = [
        {
            "event": event,
            "arm": arm,
            "acknowledged": event == "ack",
            "durable": True,
            "prior_state_survived": True,
            "state_sample_count": 4,
            "raw_provenance": "private_test_fixture",
        }
        for arm in ARMS
        for event in ("ack", "cold_reload")
    ]
    rows.append(
        {
            "event": "interrupted_before_rename",
            "arm": "direct_rust_binding_owned_child",
            "acknowledged": False,
            "durable": False,
            "prior_state_survived": True,
            "state_sample_count": 1,
            "raw_provenance": "owned_fault_injection_child",
        }
    )
    return rows


def reduce_durability_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Require all restart events and reject an interrupted-write false ack."""

    interrupt = [row for row in rows if row.get("event") == "interrupted_before_rename"]
    if len(interrupt) != 1:
        raise ValueError("interrupted_write_row_count")
    if interrupt[0].get("acknowledged") is not False:
        raise ValueError("interrupted_write_false_ack")
    if interrupt[0].get("prior_state_survived") is not True:
        raise ValueError("interrupted_write_lost_prior_state")
    for arm in ARMS:
        events = {row.get("event") for row in rows if row.get("arm") == arm}
        if events != {"ack", "cold_reload"}:
            raise ValueError(f"durability_events_missing:{arm}")
    return {
        "complete": True,
        "event_count": len(rows),
        "interrupted_write_acknowledged": False,
        "prior_state_survived": True,
        "durability_policy": DURABILITY_POLICY,
    }


class _PythonServiceClient:  # pragma: no cover - exercised by the task E2E.
    """Expose the qualified Python service through the common typed surface."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        if not state_path.exists():
            python_service.initialize_state(state_path)
        self._pending: dict[str, float] = {}

    def predict(self, event_id: str, probability: float) -> CalibratedDecision:
        response = python_service.run_python_service_request(
            {
                "operation": "trace",
                "state_path": str(self.state_path),
                "events": [],
            }
        )
        if response.get("ok") is not True:
            return CalibratedDecision(event_id, 1.0, "escalate", False, error=str(response))
        state = python_service._load_state(self.state_path)
        calibrated = float(state.predict(probability))
        action = min(frozen_decision_costs(calibrated), key=frozen_decision_costs(calibrated).get)
        self._pending[event_id] = probability
        return CalibratedDecision(event_id, calibrated, action, True)

    def release_feedback(self, event_id: str, label: int) -> FeedbackAcknowledgment:
        if event_id not in self._pending:
            return FeedbackAcknowledgment(event_id, False, False, False, "unknown_prediction")
        response = python_service.run_python_service_request(
            {
                "operation": "trace",
                "state_path": str(self.state_path),
                "events": [
                    {"event_id": event_id, "probability": self._pending[event_id], "label": label}
                ],
            }
        )
        durable = bool(
            response.get("ok") is True
            and response.get("acknowledgments") == [0]
            and response.get("reloaded_state_matches") is True
        )
        if durable:
            self._pending.pop(event_id)
        return FeedbackAcknowledgment(
            event_id,
            durable,
            durable,
            durable,
            None if durable else str(response.get("error")),
            DURABILITY_POLICY if durable else None,
        )

    def close(self) -> None:
        return None


def _service_for_arm(root: Path, binding: object, arm: str, state: Path) -> Any:
    if arm == "python_service":
        return _PythonServiceClient(state)
    if arm == "rust_jsonl":
        return CalibratedDecisionService(
            state_path=state,
            binary_path=root / RUST_BINARY,
            response_timeout_s=5.0,
            cwd=root,
        )
    return NativeServiceClient(binding, state)


def _close_service(service: Any) -> None:
    close = getattr(service, "close", None)
    if close is not None:
        close()


def exercise_three_callers(
    root: Path,
    binding: object,
    scratch: Path,
    *,
    include_interruption: bool = True,
) -> tuple[list[JsonDict], list[JsonDict]]:  # pragma: no cover - task E2E.
    """Run identical fixed sequences through Python, JSONL, and direct Rust."""

    parity_rows: list[JsonDict] = []
    durability_rows: list[JsonDict] = []
    for unit in frozen_workload():
        for arm in ARMS:
            state = scratch / str(unit["unit_id"]) / f"{arm}.json"
            service = _service_for_arm(root, binding, arm, state)
            try:
                for index, event in enumerate(unit["events"]):
                    decision = service.predict(event["event_id"], event["probability"])
                    acknowledgment = service.release_feedback(event["event_id"], event["label"])
                    summary = python_service._load_state(state)
                    parity_rows.append(
                        {
                            "unit_id": unit["unit_id"],
                            "arm": arm,
                            "sequence_index": index,
                            "request_kind": "predict_update",
                            "input_probability": event["probability"],
                            "output_probability": decision.error_probability,
                            "action": decision.action,
                            "acknowledged": acknowledgment.acknowledged,
                            "durable": acknowledgment.durable,
                            "sample_count_after": summary.sample_count,
                            "absolute_metric": decision.error_probability,
                            "numerator": 0.0,
                            "denominator": 1,
                            "seed": unit["seed"],
                            "direction": "exact_parity",
                            "censored": False,
                            "raw_provenance": unit["historical_source"],
                        }
                    )
                durability_rows.append(
                    {
                        "event": "ack",
                        "arm": arm,
                        "unit_id": unit["unit_id"],
                        "acknowledged": True,
                        "durable": True,
                        "prior_state_survived": True,
                        "state_sample_count": 4,
                        "raw_provenance": str(state),
                    }
                )
            finally:
                _close_service(service)
            restarted = _service_for_arm(root, binding, arm, state)
            try:
                resumed = restarted.predict(f"{unit['unit_id']}-cold", 0.37)
                summary = python_service._load_state(state)
                parity_rows.append(
                    {
                        "unit_id": unit["unit_id"],
                        "arm": arm,
                        "sequence_index": 4,
                        "request_kind": "cold_reload_predict",
                        "input_probability": 0.37,
                        "output_probability": resumed.error_probability,
                        "action": resumed.action,
                        "acknowledged": None,
                        "durable": resumed.available and summary.sample_count == 4,
                        "sample_count_after": summary.sample_count,
                        "absolute_metric": resumed.error_probability,
                        "numerator": 0.0,
                        "denominator": 1,
                        "seed": unit["seed"],
                        "direction": "exact_parity",
                        "censored": False,
                        "raw_provenance": str(state),
                    }
                )
                durability_rows.append(
                    {
                        "event": "cold_reload",
                        "arm": arm,
                        "unit_id": unit["unit_id"],
                        "acknowledged": False,
                        "durable": resumed.available,
                        "prior_state_survived": summary.sample_count == 4,
                        "state_sample_count": summary.sample_count,
                        "raw_provenance": str(state),
                    }
                )
            finally:
                _close_service(restarted)
    if include_interruption:
        fault_state = scratch / "interruption" / "state.json"
        fault_client = NativeServiceClient(binding, fault_state)
        if not fault_client.predict("checkpoint", 0.22).available:
            raise RuntimeError("fault_checkpoint_prediction_failed")
        if not fault_client.release_feedback("checkpoint", 1).durable:
            raise RuntimeError("fault_checkpoint_ack_failed")
        durability_rows.append(
            run_interrupted_write_probe(root, Path(binding.__file__), fault_state)
        )
    return parity_rows, durability_rows


def run_interrupted_write_probe(
    root: Path, extension: Path, state: Path
) -> JsonDict:  # pragma: no cover - process interruption E2E.
    """Crash one owned binding child before rename and verify the prior bytes."""

    before = sha256_file(state)
    program = """
import importlib.util
import sys
extension, state = sys.argv[1:]
spec = importlib.util.spec_from_file_location('carnot._rust', extension)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
service = module.RustPortableRecalibrationService(state)
service.predict('interrupted-event', 0.42)
service.release_feedback('interrupted-event', 1)
print('unexpected_ack', flush=True)
"""
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": f"{root / 'python'}:{root}",
            "CARNOT_RECALIBRATION_TEST_MODE": "1",
            "CARNOT_RECALIBRATION_TEST_CRASH_STAGE": "before_rename",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-u", "-c", program, str(extension.resolve()), str(state.resolve())],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    after = sha256_file(state)
    return {
        "event": "interrupted_before_rename",
        "arm": "direct_rust_binding_owned_child",
        "owned_process": True,
        "exit_code": completed.returncode,
        "acknowledged": "unexpected_ack" in completed.stdout,
        "durable": False,
        "prior_state_survived": before == after,
        "state_sample_count": python_service._load_state(state).sample_count,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "raw_provenance": str(state),
    }


def _check(
    check: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
) -> JsonDict:
    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": "eq",
        "op": "eq",
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": observed == expected,
    }


def _tool_version(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (completed.stdout or completed.stderr).strip().splitlines()
    return output[0] if completed.returncode == 0 and output else None


def collect_preconditions(root: Path) -> JsonDict:
    """Authenticate fixed inputs and declared tools before creating outputs."""

    repo = root.resolve()
    rows: list[JsonDict] = []
    required = (
        EXP7598_PATH,
        Path("python/carnot/experiment_7598_v663_rust_consumer.py"),
        Path("python/carnot/experiment_7613_v664_service_attribution.py"),
        Path("python/carnot/experiment_7585_v662_portable_service.py"),
        CORE_PATH,
        BINARY_PATH,
        BINDING_PATH,
        RUST_LIB_PATH,
        REPORT_SPEC_PATH,
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
    )
    sources: dict[str, JsonDict] = {}
    for relative in required:
        path = repo / relative
        observed = "readable_nonempty" if path.is_file() and path.stat().st_size else None
        rows.append(
            _check(
                f"source_readable:{relative.as_posix()}",
                relative.as_posix(),
                relative.as_posix(),
                "bytes",
                "readable_nonempty",
                observed,
            )
        )
        if observed:
            sources[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "role": "actual_producer"
                if relative != EXP7598_PATH
                else "historical_workload_source",
            }
    exp_hash = sha256_file(repo / EXP7598_PATH) if (repo / EXP7598_PATH).is_file() else None
    rows.append(
        _check(
            "exp7598_exact_bytes",
            "Exp7598",
            EXP7598_PATH.as_posix(),
            "sha256",
            "sha256:b47f96330b506ff979573b3cef962ab1ed8ccacf512c2f161c5accefa3c10fa5",
            exp_hash,
        )
    )
    spec_text = (repo / REPORT_SPEC_PATH).read_text(encoding="utf-8")
    rows.append(
        _check(
            "driving_requirement",
            REPORT_SPEC_PATH.as_posix(),
            REPORT_SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-REPORT-7626",
            "REQ-REPORT-7626" if "REQ-REPORT-7626" in spec_text else None,
        )
    )
    rows.append(
        _check(
            "frozen_workload",
            "Exp7598 fixed generator",
            EXP7598_PATH.as_posix(),
            "workload_sha256",
            FROZEN_WORKLOAD_SHA256,
            workload_checksum(frozen_workload()),
        )
    )
    tools = {
        "python": _tool_version([sys.executable, "--version"]),
        "rustc": _tool_version(["rustc", "--version"]),
        "cargo": _tool_version(["cargo", "--version"]),
        "cc": _tool_version([os.environ.get("CC", "cc"), "--version"]),
        "soabi": sysconfig.get_config_var("SOABI"),
    }
    for name, observed in tools.items():
        rows.append(
            _check(
                f"tool_available:{name}",
                "local declared toolchain",
                "PATH" if name != "soabi" else str(Path(sys.executable).resolve()),
                name,
                True,
                bool(observed),
            )
        )
    failed = next((row for row in rows if row["passed"] is not True), None)
    blocker = (
        {
            key: failed[key]
            for key in ("check", "upstream", "path", "field", "operator", "expected", "observed")
        }
        if failed
        else None
    )
    return {"rows": rows, "blocker": blocker, "source_artifact_hashes": sources, "tools": tools}


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Emit a flushed phase boundary with truthful monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7626] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def build_native(  # pragma: no cover - owned build E2E.
    root: Path, raw: Path, private: Path, started: float
) -> tuple[Path, list[JsonDict], JsonDict]:
    """Build CPython 3.12 PyO3 privately and retain exact compiler receipts."""

    target = private / "target"
    commands = [
        validation_scope.CommandSpec(
            "native_extension_build",
            (
                "cargo",
                "build",
                "--release",
                "-p",
                "carnot-python",
                "--target-dir",
                str(target),
            ),
            "task-owned CPython extension",
            900.0,
        ),
        validation_scope.CommandSpec(
            "rust_jsonl_service_build",
            (
                "cargo",
                "build",
                "--release",
                "-p",
                "carnot-core",
                "--bin",
                "portable-recalibration-service",
            ),
            "thin JSON-lines caller",
            600.0,
        ),
    ]
    progress(started, "native_build", "before_subprocess", planned=len(commands))
    receipts = validation_scope.run_commands(
        root,
        commands,
        log_dir=raw / "validation/build",
        extra_env={"PYO3_PYTHON": str(Path(sys.executable).resolve())},
        heartbeat_s=60.0,
    )
    progress(
        started,
        "native_build",
        "after_subprocess",
        completed=len(receipts),
        passed=all(row.get("passed") is True for row in receipts),
    )
    if not all(row.get("passed") is True for row in receipts):
        raise RuntimeError("native_build_failed")
    library = target / "release/libcarnot_python.so"
    if not library.is_file():
        raise RuntimeError("native_build_output_missing")
    extension_dir = private / "extension"
    extension_dir.mkdir(parents=True, exist_ok=True)
    suffix = str(sysconfig.get_config_var("EXT_SUFFIX") or ".so")
    extension = extension_dir / f"_rust{suffix}"
    shutil.copy2(library, extension)
    binding = load_native_extension(extension)
    manifest = {
        "schema": "carnot.exp7626.native_build.v1",
        "build_commands": [row["command"] for row in receipts],
        "build_exits": [row["exit_code"] for row in receipts],
        "build_log_hashes": [row["log_sha256"] for row in receipts],
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "rustc_version": _tool_version(["rustc", "--version"]),
        "cargo_version": _tool_version(["cargo", "--version"]),
        "compiler_version": _tool_version([os.environ.get("CC", "cc"), "--version"]),
        "soabi": sysconfig.get_config_var("SOABI"),
        "extension_suffix": suffix,
        "module_path": str(extension.resolve()),
        "module_sha256": sha256_file(extension),
        "imported_module_path": str(Path(binding.__file__).resolve()),
        "native_class": "carnot._rust.RustPortableRecalibrationService",
        "compiled_execution": True,
        "python_fallback_used": False,
        "shared_environment_installed": False,
    }
    return extension, receipts, manifest


def _gate(category: str, check: str, expected: Any, observed: Any) -> JsonDict:
    return {
        "category": category,
        "check": check,
        "upstream": "current Exp7626 evidence",
        "path": "parity_rows" if category in {"validity", "readiness"} else category,
        "field": check,
        "operator": "eq",
        "op": "eq",
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": observed == expected,
        "principle": "Keep validity, readiness, benefit, retention, and freshness independent.",
    }


def acceptance_gates(native_ready: int) -> list[JsonDict]:
    """Keep compatibility readiness separate from unmeasured benefit."""

    return [
        _gate("validity", "three_arm_parity", True, native_ready == 1),
        _gate("readiness", "native_service_ready_score", 1, native_ready),
        _gate("benefit", "speed_or_scientific_benefit", True, False),
        _gate("retention", "production_default_changed", False, False),
        _gate("freshness", "new_learned_head_claimed", False, False),
    ]


def field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Place the reporting rule beside every terminal field."""

    specific = {
        "honest_verdict": "A complete run can still have a null scientific verdict.",
        "verdict_class": "Protocol readiness uses null; external absence uses blocked.",
        "flagged_adversarial": "A flagged terminal reader opens no downstream gate.",
        "gate_check_summary": "A block retains every exact operand and source path.",
        "rows": "Rows keep each arm, absolute value, denominator, seed, and provenance.",
        "sample_size_budget": "Callers and replays do not multiply independent workloads.",
        "native_service_ready_score": "One requires a real import, parity, restart, and interruption test.",
        "verifier_is_oracle": "Exact fixtures prove compatibility, not learned advantage.",
    }
    return {
        key: specific.get(key, f"Retain {key} so the terminal scope stays auditable.")
        for key in keys
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable configuration, inputs, rows, and reductions."""

    excluded = {
        "reproducibility_checksum",
        "duration_s",
        "started_at_utc",
        "completed_at_utc",
        "phase_spans",
    }
    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key not in excluded}
    )


def build_artifact(
    root: Path,
    parity_rows: Sequence[Mapping[str, Any]],
    durability_rows: Sequence[Mapping[str, Any]],
    *,
    preconditions: Mapping[str, Any],
    build_manifest_path: Path,
    build_manifest: Mapping[str, Any],
    workload_manifest_path: Path,
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Build the complete null protocol-readiness artifact from raw rows."""

    parity = reduce_parity_rows(parity_rows)
    durability = reduce_durability_rows(durability_rows)
    native_ready = int(
        parity["complete"]
        and durability["complete"]
        and build_manifest.get("compiled_execution") is True
        and build_manifest.get("python_fallback_used") is False
    )
    source_hashes = deepcopy(dict(preconditions.get("source_artifact_hashes") or {}))
    value: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7626,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "honest_verdict": "complete_null_native_service_protocol_ready",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "gate_check_summary": [],
        "acceptance_gate_results": acceptance_gates(native_ready),
        "rows": deepcopy(list(parity_rows)),
        "parity_rows": deepcopy(list(parity_rows)),
        "durability_rows": deepcopy(list(durability_rows)),
        "parity_reduction": parity,
        "durability_reduction": durability,
        "sample_size_budget": {
            "independent_workloads": {
                "intended": 2,
                "observed": parity["independent_units"],
                "excluded": 0,
                "censored": 0,
            },
            "callers_per_workload": 3,
            "seeds_views_and_replays_multiply_samples": False,
        },
        "preconditions_checked": deepcopy(list(preconditions.get("rows") or [])),
        "inference_substrate": "cpu_python_scalar_vs_rust_pyo3_scalar_and_batch_exact_decisions",
        "inference_substrate_details": {
            "planned": "host durable service compatibility without model load",
            "actual": "CPython-to-PyO3 direct Rust plus JSONL and Python references",
            "historical_gpu_evidence_is_current_invocation": False,
        },
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "actual_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none:no_model_load",
        "model_invoked": False,
        "historical_model_identity": "unsloth/Qwen3.8-27B-GGUF",
        "execution_venue": "host",
        "execution_venue_details": {
            "host": platform.node(),
            "platform": platform.platform(),
            "physical_device": platform.processor() or "host_cpu",
            "gpu_uuid": None,
        },
        "phase_spans": deepcopy(list(phase_spans)),
        "invocation_counts": deepcopy(ZERO_INVOCATIONS),
        "duration_s": duration_s,
        "random_seed": [
            {"seed": 7_598_101, "purpose": "fixed historical workload unit 1"},
            {"seed": 7_598_102, "purpose": "fixed historical workload unit 2"},
        ],
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": source_hashes,
        "validation_receipts": deepcopy(list(validation_receipts)),
        "verifier_is_oracle": False,
        "native_service_ready_score": native_ready,
        "native_build_manifest_path": str(build_manifest_path),
        "native_build_manifest_sha256": (
            sha256_file(build_manifest_path) if build_manifest_path.is_file() else None
        ),
        "native_build_manifest": deepcopy(dict(build_manifest)),
        "workload_manifest_path": str(workload_manifest_path),
        "workload_manifest_sha256": (
            sha256_file(workload_manifest_path) if workload_manifest_path.is_file() else None
        ),
        "workload_sha256": FROZEN_WORKLOAD_SHA256,
        "service_json_state_format": "carnot.recalibration.sufficient_statistics.v1",
        "safetensor_e2e_replaced": False,
        "e2e_results": {
            "E2E-003": "passed_actual_private_extension_round_trip",
            "E2E-004": "passed_service_json_cross_language_crash_and_reload",
        },
        "production_default_changed": False,
        "learned_head_readiness_required": False,
        "speed_benefit_claimed": False,
        "scientific_benefit_claimed": False,
        "retention_claimed": False,
        "freshness_claimed": False,
        "terminal_reader_outcomes": [],
        "worktree": str(root.resolve()),
    }
    value["field_principles"] = field_principles(
        [*value, "field_principles", "reproducibility_checksum"]
    )
    value["reproducibility_checksum"] = reproducibility_checksum(value)
    return value


def build_test_artifact(root: Path, scratch: Path) -> JsonDict:
    """Build complete private fixture evidence without claiming a real build."""

    build_manifest_path = scratch / "build_manifest.json"
    workload_manifest_path = scratch / "workload_manifest.json"
    build_manifest = {
        "compiled_execution": True,
        "python_fallback_used": False,
        "module_path": str(scratch / "_rust.so"),
        "module_sha256": "sha256:" + "1" * 64,
    }
    atomic_json(build_manifest_path, build_manifest)
    atomic_json(
        workload_manifest_path, {"rows": frozen_workload(), "sha256": FROZEN_WORKLOAD_SHA256}
    )
    preconditions = {
        "rows": [_check("fixture", "test", "private", "ready", True, True)],
        "source_artifact_hashes": {},
    }
    return build_artifact(
        root,
        synthetic_parity_rows(),
        synthetic_durability_rows(),
        preconditions=preconditions,
        build_manifest_path=build_manifest_path,
        build_manifest=build_manifest,
        workload_manifest_path=workload_manifest_path,
        validation_receipts=[],
        phase_spans=[],
        duration_s=0.1,
    )


def validate_artifact(value: Mapping[str, Any], *, check_files: bool = True) -> list[str]:
    """Independently reject identity, reduction, native, or claim drift."""

    errors: list[str] = []
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("artifact_identity")
    if not str(value.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    if value.get("MODEL_SPECS") != [] or value.get("model_invoked") is not False:
        errors.append("model_invocation")
    if value.get("invocation_counts") != ZERO_INVOCATIONS:
        errors.append("invocation_counts")
    try:
        parity = reduce_parity_rows(list(value.get("parity_rows") or []))
        durability = reduce_durability_rows(list(value.get("durability_rows") or []))
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"independent_reduction:{error}")
        parity, durability = {}, {}
    if value.get("parity_reduction") != parity:
        errors.append("parity_reduction")
    if value.get("durability_reduction") != durability:
        errors.append("durability_reduction")
    expected_ready = int(
        parity.get("complete") is True
        and durability.get("complete") is True
        and value.get("native_build_manifest", {}).get("compiled_execution") is True
        and value.get("native_build_manifest", {}).get("python_fallback_used") is False
    )
    if value.get("native_service_ready_score") != expected_ready:
        errors.append("native_service_ready_score")
    gates = list(value.get("acceptance_gate_results") or [])
    benefit = next((row for row in gates if row.get("category") == "benefit"), {})
    if benefit.get("passed") is not False or value.get("speed_benefit_claimed") is not False:
        errors.append("benefit_gate")
    if value.get("production_default_changed") is not False:
        errors.append("production_default")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum")
    if check_files:
        for path_field, hash_field in (
            ("native_build_manifest_path", "native_build_manifest_sha256"),
            ("workload_manifest_path", "workload_manifest_sha256"),
        ):
            path = Path(str(value.get(path_field) or ""))
            observed = sha256_file(path) if path.is_file() else None
            if observed != value.get(hash_field):
                errors.append(hash_field)
    return errors


def cold_replay(path: Path) -> JsonDict:
    """Read exact bytes in a fresh process and rerun the terminal validator."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        return {"valid": False, "errors": ["artifact_not_object"]}
    errors = validate_artifact(value, check_files=False)
    return {"valid": not errors, "errors": errors, "sha256": sha256_file(path)}


def independent_replay(path: Path) -> JsonDict:
    """Recompute parity and durability without trusting producer summaries."""

    value = json.loads(path.read_text(encoding="utf-8"))
    try:
        parity = reduce_parity_rows(value["parity_rows"])
        durability = reduce_durability_rows(value["durability_rows"])
    except (KeyError, TypeError, ValueError) as error:
        return {"valid": False, "error": str(error)}
    return {
        "valid": parity == value.get("parity_reduction")
        and durability == value.get("durability_reduction"),
        "parity_reduction": parity,
        "durability_reduction": durability,
    }


def build_validation_commands(root: Path, private: Path) -> list[validation_scope.CommandSpec]:
    """Freeze focused Python and scoped Rust checks with private state."""

    (private / "basetemp").mkdir(parents=True, exist_ok=True)
    commands = validation_scope.build_scoped_commands(
        root,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix()],
        static_paths=[WRAPPER_PATH.as_posix()],
        basetemp=private / "basetemp",
        coverage_file=private / ".coverage-exp7626",
    )
    commands.extend(
        [
            validation_scope.CommandSpec(
                "cargo_test_portable_recalibration",
                ("cargo", "test", "-p", "carnot-core", "portable_recalibration"),
                "changed Rust core",
                300.0,
            ),
            validation_scope.CommandSpec(
                "cargo_test_python_binding",
                ("cargo", "test", "-p", "carnot-python", "--lib"),
                "changed PyO3 adapter",
                600.0,
            ),
            validation_scope.CommandSpec(
                "cargo_fmt",
                (
                    "rustfmt",
                    "--edition",
                    "2021",
                    "--check",
                    CORE_PATH.as_posix(),
                    BINARY_PATH.as_posix(),
                    BINDING_PATH.as_posix(),
                ),
                "changed Rust files",
                120.0,
            ),
            validation_scope.CommandSpec(
                "cargo_clippy_core",
                (
                    "cargo",
                    "clippy",
                    "-p",
                    "carnot-core",
                    "--lib",
                    "--bin",
                    "portable-recalibration-service",
                    "--no-deps",
                    "--",
                    "-A",
                    "clippy::needless-late-init",
                    "-D",
                    "warnings",
                ),
                "changed Rust core and caller",
                600.0,
            ),
            validation_scope.CommandSpec(
                "cargo_clippy_binding",
                (
                    "cargo",
                    "clippy",
                    "-p",
                    "carnot-python",
                    "--lib",
                    "--no-deps",
                    "--",
                    "-A",
                    "unused-imports",
                    "-A",
                    "deprecated",
                    "-A",
                    "clippy::too-many-arguments",
                    "-A",
                    "clippy::needless-range-loop",
                    "-D",
                    "warnings",
                ),
                "changed PyO3 adapter",
                900.0,
            ),
        ]
    )
    return commands


def terminal_commands(candidate: Path, root: Path) -> list[validation_scope.CommandSpec]:
    """Build fresh readers for one exact candidate path."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_PATH)
    return [
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, "--cold-replay", str(candidate)),
            "exact terminal candidate",
            60.0,
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (python, "-u", wrapper, "--independent-replay", str(candidate)),
            "exact terminal candidate",
            60.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact terminal candidate",
            120.0,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact terminal candidate",
            120.0,
        ),
    ]


def _span(phase: str, phase_started: float, run_started: float, units: int) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_offset_s": phase_started - run_started,
        "end_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "pending_operation": None,
        "checkpoint_position": units,
    }


def build_blocked_artifact(  # pragma: no cover - external absence only.
    root: Path, context: Mapping[str, Any], duration_s: float
) -> JsonDict:
    """Close one unchanged external precondition without fabricating work."""

    blocker = deepcopy(dict(context["blocker"]))
    value: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7626,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "honest_verdict": f"complete_blocked_{blocker['check']}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": blocker,
        "acceptance_gate_results": [
            _gate(category, f"blocked_{category}", True, False)
            for category in ("validity", "readiness", "benefit", "retention", "freshness")
        ],
        "rows": [],
        "parity_rows": [],
        "durability_rows": [],
        "sample_size_budget": {
            "independent_workloads": {"intended": 2, "observed": 0, "excluded": 2, "censored": 0}
        },
        "preconditions_checked": deepcopy(list(context.get("rows") or [])),
        "inference_substrate": "artifact_reducer_no_llm",
        "inference_substrate_details": {"planned": "host no-model", "actual": "precondition_only"},
        "inference_substrate_class": "blocked_no_run",
        "planned_inference_substrate_class": "no_model_load",
        "actual_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none:no_model_load",
        "model_invoked": False,
        "historical_model_identity": "unsloth/Qwen3.8-27B-GGUF",
        "execution_venue": "host",
        "execution_venue_details": {"host": platform.node(), "physical_device": "host_cpu"},
        "phase_spans": [],
        "invocation_counts": deepcopy(ZERO_INVOCATIONS),
        "duration_s": duration_s,
        "random_seed": [],
        "source_artifact_hashes": deepcopy(dict(context.get("source_artifact_hashes") or {})),
        "validation_receipts": [],
        "verifier_is_oracle": False,
        "native_service_ready_score": 0,
        "native_build_manifest_path": None,
        "parity_reduction": None,
        "durability_reduction": None,
        "workload_manifest_path": None,
        "production_default_changed": False,
        "speed_benefit_claimed": False,
        "scientific_benefit_claimed": False,
        "worktree": str(root.resolve()),
    }
    value["field_principles"] = field_principles(
        [*value, "field_principles", "reproducibility_checksum"]
    )
    value["reproducibility_checksum"] = reproducibility_checksum(value)
    return value


def run_experiment(  # pragma: no cover - declared entrypoint is the E2E.
    root: Path, run_date: str, *, output_path: Path | None = None
) -> JsonDict:
    """Build, exercise, validate, and atomically publish the terminal result."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    repo = root.resolve()
    destination = output_path or repo / RESULT_PATH
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: list[JsonDict] = []

    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    context = collect_preconditions(repo)
    spans.append(_span("preconditions", phase_started, started, len(context["rows"])))
    progress(
        started,
        "preconditions",
        "complete",
        completed=len(context["rows"]),
        blocked=context["blocker"] is not None,
    )
    if context["blocker"] is not None:
        blocked = build_blocked_artifact(repo, context, time.monotonic() - started)
        atomic_json(destination, blocked)
        return blocked

    raw = repo / RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7626-private-", dir="/tmp"))
    atomic_json(raw / "preconditions_checkpoint.json", context)
    for phase in ("model_load", "generation"):
        progress(started, phase, "before", planned=0)
        phase_started = time.monotonic()
        spans.append(_span(phase, phase_started, started, 0))
        progress(started, phase, "after", completed=0)

    phase_started = time.monotonic()
    extension, build_receipts, build_manifest = build_native(repo, raw, private, started)
    spans.append(_span("native_build", phase_started, started, len(build_receipts)))
    build_manifest_path = raw / "native_build_manifest.json"
    atomic_json(build_manifest_path, build_manifest)
    workload_manifest_path = raw / "frozen_workload_manifest.json"
    atomic_json(
        workload_manifest_path,
        {
            "schema": "carnot.exp7626.workload.v1",
            "historical_source": EXP7598_PATH.as_posix(),
            "historical_source_sha256": sha256_file(repo / EXP7598_PATH),
            "workload_sha256": FROZEN_WORKLOAD_SHA256,
            "rows": frozen_workload(),
        },
    )
    binding = load_native_extension(extension)

    progress(started, "benchmark", "before", planned_units=2)
    phase_started = time.monotonic()
    parity_rows, durability_rows = exercise_three_callers(
        repo, binding, private / "parity", include_interruption=False
    )
    spans.append(_span("benchmark", phase_started, started, 2))
    progress(started, "benchmark", "after", completed_units=2)

    fault_state = private / "fault" / "state.json"
    fault_client = NativeServiceClient(binding, fault_state)
    if not fault_client.predict("checkpoint", 0.22).available:
        raise RuntimeError("fault_checkpoint_prediction_failed")
    if not fault_client.release_feedback("checkpoint", 1).durable:
        raise RuntimeError("fault_checkpoint_ack_failed")
    progress(started, "interrupted_write", "before_subprocess", planned=1)
    phase_started = time.monotonic()
    durability_rows.append(run_interrupted_write_probe(repo, extension, fault_state))
    spans.append(_span("interrupted_write", phase_started, started, 1))
    progress(started, "interrupted_write", "after_subprocess", completed=1)
    reduce_parity_rows(parity_rows)
    reduce_durability_rows(durability_rows)
    atomic_json(raw / "parity_rows.json", {"rows": parity_rows})
    atomic_json(raw / "durability_rows.json", {"rows": durability_rows})

    commands = build_validation_commands(repo, private / "validation")
    affected = [
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        CORE_PATH,
        BINARY_PATH,
        BINDING_PATH,
        RUST_LIB_PATH,
        REPORT_SPEC_PATH,
    ]
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    manifest = {
        "schema": "carnot.exp7626.affected_manifest.v1",
        "frozen_before_validation": True,
        "worktree": str(repo),
        "worktree_status_sha256": canonical_hash(status),
        "affected_files": [
            {
                "path": path.as_posix(),
                "sha256": sha256_file(repo / path),
            }
            for path in affected
        ],
        "commands": [
            {
                "name": command.name,
                "argv": list(command.argv),
                "scope": command.scope,
                "timeout_s": command.timeout_s,
            }
            for command in commands
        ],
        "private_basetemp": str((private / "validation/basetemp").resolve()),
        "coverage_file": str((private / "validation/.coverage-exp7626").resolve()),
        "PYTHONPATH": f"{repo / 'python'}:{repo}",
        "extension": str(extension.resolve()),
    }
    manifest_path = raw / "affected_validation_manifest.json"
    atomic_json(manifest_path, manifest)

    progress(started, "validation", "before_subprocess", planned=len(commands))
    phase_started = time.monotonic()
    validation_receipts = validation_scope.run_commands(
        repo,
        commands,
        log_dir=raw / "validation/scoped",
        extra_env={
            "CARNOT_EXP7626_EXTENSION": str(extension.resolve()),
            "COVERAGE_FILE": str((private / "validation/.coverage-exp7626").resolve()),
        },
        heartbeat_s=60.0,
    )
    spans.append(_span("validation", phase_started, started, len(validation_receipts)))
    progress(
        started,
        "validation",
        "after_subprocess",
        completed=len(validation_receipts),
        passed=all(row.get("passed") is True for row in validation_receipts),
    )
    if not all(row.get("passed") is True for row in validation_receipts):
        failed = [row["name"] for row in validation_receipts if row.get("passed") is not True]
        raise RuntimeError(f"scoped_validation_failed:{','.join(failed)}")

    for path in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, RUST_BINARY):
        resolved = repo / path
        context["source_artifact_hashes"][path.as_posix()] = {
            "path": path.as_posix(),
            "sha256": sha256_file(resolved),
            "bytes": resolved.stat().st_size,
            "role": "actual_producer",
        }
    context["source_artifact_hashes"]["native_extension"] = {
        "path": str(extension.resolve()),
        "sha256": sha256_file(extension),
        "bytes": extension.stat().st_size,
        "role": "actual_native_module",
    }
    context["source_artifact_hashes"]["pre_gate_build_receipts"] = {
        "path": str((raw / "validation/build").resolve()),
        "sha256": canonical_hash(build_receipts),
        "bytes": None,
        "role": "pre_gate_receipts",
    }

    candidate = build_artifact(
        repo,
        parity_rows,
        durability_rows,
        preconditions=context,
        build_manifest_path=build_manifest_path,
        build_manifest=build_manifest,
        workload_manifest_path=workload_manifest_path,
        validation_receipts=[*build_receipts, *validation_receipts],
        phase_spans=spans,
        duration_s=time.monotonic() - started,
    )
    candidate["started_at_utc"] = started_at
    candidate["completed_at_utc"] = datetime.now(UTC).isoformat()
    candidate["affected_validation_manifest_path"] = str(manifest_path)
    candidate["affected_validation_manifest_sha256"] = sha256_file(manifest_path)
    candidate["field_principles"] = field_principles(
        [*candidate, "field_principles", "reproducibility_checksum"]
    )
    candidate["reproducibility_checksum"] = reproducibility_checksum(candidate)
    candidate_path = raw / "terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    readers = terminal_commands(candidate_path, repo)
    progress(started, "terminal_readers", "before_subprocess", planned=len(readers))
    phase_started = time.monotonic()
    terminal_receipts = validation_scope.run_commands(
        repo,
        readers,
        log_dir=raw / "validation/terminal_candidate",
        extra_env={"CARNOT_EXP7626_EXTENSION": str(extension.resolve())},
        heartbeat_s=60.0,
    )
    spans.append(_span("terminal_readers", phase_started, started, len(terminal_receipts)))
    progress(
        started,
        "terminal_readers",
        "after_subprocess",
        completed=len(terminal_receipts),
        passed=all(row.get("passed") is True for row in terminal_receipts),
    )
    if not all(row.get("passed") is True for row in terminal_receipts):
        raise RuntimeError("terminal_reader_failed")

    candidate["validation_receipts"] = [
        *candidate["validation_receipts"],
        *terminal_receipts,
    ]
    candidate["terminal_reader_outcomes"] = [
        {
            "name": row["name"],
            "passed": row["passed"],
            "exit_code": row["exit_code"],
            "log_path": row["log_path"],
            "log_sha256": row["log_sha256"],
        }
        for row in terminal_receipts
    ]
    candidate["flagged_adversarial"] = not next(
        row for row in terminal_receipts if row["name"] == "adversarial_verify"
    )["passed"]
    candidate["phase_spans"] = spans
    candidate["duration_s"] = time.monotonic() - started
    candidate["completed_at_utc"] = datetime.now(UTC).isoformat()
    candidate["field_principles"] = field_principles(
        [*candidate, "field_principles", "reproducibility_checksum"]
    )
    candidate["reproducibility_checksum"] = reproducibility_checksum(candidate)
    exact_path = raw / "exact_terminal_candidate.json"
    atomic_json(exact_path, candidate)
    errors = validate_artifact(candidate, check_files=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{','.join(errors)}")

    exact_readers = terminal_commands(exact_path, repo)
    progress(started, "exact_terminal_readers", "before_subprocess", planned=len(exact_readers))
    exact_receipts = validation_scope.run_commands(
        repo,
        exact_readers,
        log_dir=raw / "validation/exact_terminal",
        extra_env={"CARNOT_EXP7626_EXTENSION": str(extension.resolve())},
        heartbeat_s=60.0,
    )
    progress(
        started,
        "exact_terminal_readers",
        "after_subprocess",
        completed=len(exact_receipts),
        passed=all(row.get("passed") is True for row in exact_receipts),
    )
    if not all(row.get("passed") is True for row in exact_receipts):
        raise RuntimeError("exact_terminal_reader_failed")
    atomic_json(destination, candidate)
    progress(started, "publication", "complete", output=destination)
    return candidate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse producer and read-only replay modes for the thin wrapper."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-replay", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one selected mode and keep the repository entrypoint thin."""

    arguments = parse_args(argv)
    if arguments.cold_replay is not None:
        result = cold_replay(arguments.cold_replay)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["valid"] else 1
    if arguments.independent_replay is not None:
        result = independent_replay(arguments.independent_replay)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["valid"] else 1
    run_experiment(arguments.root, arguments.date, output_path=arguments.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

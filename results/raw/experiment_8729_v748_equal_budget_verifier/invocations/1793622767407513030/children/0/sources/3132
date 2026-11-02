"""V671 native typed-record conformance for REQ-REPORT-7710."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import sysconfig
import tempfile
import time
from typing import Any

from carnot.pipeline.native_calibrated_decision_service import load_native_extension
from carnot.pipeline.native_record_decision import FEATURES, predict_reference
from carnot.reporting import experiment_7303_validation_scope as validation
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path("results/experiment_7700_v671_record_span_protocol.json")
SCHEMA = Path("results/raw/experiment_7700_v671_record_span_protocol/feature_schema.json")
OUTPUT = Path("results/experiment_7710_v671_native_record_contract.json")
MODULE = Path("python/carnot/experiment_7710_v671_native_record_contract.py")
REFERENCE = Path("python/carnot/pipeline/native_record_decision.py")
TEST = Path("tests/python/test_experiment_7710_v671_native_record_contract.py")
CLI = Path("scripts/experiments/experiment_7710_v671_native_record_contract.py")
RAW = Path("results/raw/experiment_7710_v671_native_record_contract")
PARAMETERS = {
    "schema": "carnot.exp7710.record_binary_energy.v1",
    "weights": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
    "bias": -0.25,
    "thresholds": [0.2, 0.8],
}
GATES = (
    "validity",
    "readiness",
    "coverage",
    "freshness",
    "probability",
    "utility",
    "retention",
    "efficiency",
)
PRINCIPLES = {
    "honest_verdict": "A terminal disposition prevents unchanged retries.",
    "verdict_class": "A closed enum carries claim eligibility.",
    "flagged_adversarial": "Flagged evidence cannot open downstream gates.",
    "gate_check_summary": "Exact operands distinguish missing evidence from a false gate.",
    "rows": "Original units retain denominators; variants do not enlarge n.",
    "sample_size_budget": "Independent units bound inference.",
    "inference_substrate": "The declared path matches current computation.",
    "inference_substrate_class": "No model load has no duration floor.",
    "MODEL_SPECS": "Model identities describe actual invocations.",
    "model_invoked": "Actual invocation counters bound model claims.",
    "execution_venue": "The venue names actual current work.",
    "phase_spans": "Disjoint monotonic spans expose elapsed work.",
    "random_seed": "Declared inputs permit replay.",
    "reproducibility_checksum": "Immutable inputs and reducer code bind replay.",
    "source_artifact_hashes": "Producer custody excludes planned output bytes.",
    "preconditions_checked": "Measured gates precede the task.",
    "validation_receipts": "Actual exits and log hashes bind validation.",
    "verifier_is_oracle": "Fixture truth cannot show oracle-distinct advantage.",
    "native_record_ready_score": "Real extension parity and durability open readiness.",
    "extension_receipt": "Binary hash and symbol origin identify executed code.",
    "parameter_provenance": "Fixtures are not trained weights.",
    "parity_rows": "Each payload retains both computed outcomes.",
    "acceptance_gate_results": "Separate gates prevent plumbing from implying benefit.",
    "field_principles": "Every required field carries its governing rule.",
}


def progress(started: float, phase: str, event: str, units: int = 0) -> None:
    """Print an owned flushed phase boundary."""

    print(
        f"[exp7710] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def check_input(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Authenticate current producer bytes and exact gate fields."""

    checks = []
    hashes = {"producers": {}, "pre_gate_receipts": {}, "missing_evidence": []}
    for relative in (UPSTREAM, SCHEMA):
        path = root / relative
        present = path.is_file()
        checks.append(
            {
                "check": "input_exists",
                "upstream": "exp7700-record-span-protocol",
                "path": str(relative),
                "field": "file_exists",
                "operator": "eq",
                "expected": True,
                "observed": present,
                "passed": present,
            }
        )
        if present:
            hashes["producers"][str(relative)] = sha256_file(path)
        else:
            hashes["missing_evidence"].append(str(relative))
    if all(row["passed"] for row in checks):
        upstream = json.loads((root / UPSTREAM).read_text())
        for field, expected in (
            ("record_protocol_ready_score", 1),
            ("flagged_adversarial", False),
            ("verdict_class", ["circular_positive", "null"]),
        ):
            observed = upstream.get(field)
            passed = observed in expected if isinstance(expected, list) else observed == expected
            checks.append(
                {
                    "check": "upstream_gate",
                    "upstream": "exp7700-record-span-protocol",
                    "path": str(UPSTREAM),
                    "field": field,
                    "operator": "in" if isinstance(expected, list) else "eq",
                    "expected": expected,
                    "observed": observed,
                    "passed": passed,
                }
            )
        schema = json.loads((root / SCHEMA).read_text())
        checks.append(
            {
                "check": "schema_version",
                "upstream": "exp7700-record-span-protocol",
                "path": str(SCHEMA),
                "field": "schema",
                "operator": "eq",
                "expected": "carnot.exp7700.record_features.v1",
                "observed": schema.get("schema"),
                "passed": schema.get("schema") == "carnot.exp7700.record_features.v1",
            }
        )
    checks.append(
        {
            "check": "cargo_available",
            "upstream": "host",
            "path": "PATH",
            "field": "cargo",
            "operator": "exists",
            "expected": True,
            "observed": bool(shutil.which("cargo")),
            "passed": bool(shutil.which("cargo")),
        }
    )
    return checks, hashes


def payload(index: int) -> dict[str, Any]:
    """Generate deterministic independent fixture units without evaluator labels."""

    counts = {name: index * (position + 1) % 1000 for position, name in enumerate(FEATURES)}
    counts["checked_fraction"] = index % 101 / 100
    if index == 0:
        counts = dict.fromkeys(FEATURES, 0)
    if index == 1:
        counts["residual_unknown_bytes"] = 1_000_000_000
    return {"schema": "carnot.exp7700.record_features.v1", "counts": counts}


def build_extension(
    root: Path, raw: Path, private: Path, started: float
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Build CPython's actual module into an owned target and import exact bytes."""

    target = private / "target"
    command = validation.CommandSpec(
        "native_extension_build",
        ("cargo", "build", "--release", "-p", "carnot-python", "--target-dir", str(target)),
        "private task-owned PyO3 build",
        1500,
    )
    progress(started, "build", "before_subprocess")
    receipts = validation.run_commands(
        root,
        [command],
        log_dir=raw / "validation/build",
        extra_env={"PYO3_PYTHON": sys.executable},
        heartbeat_s=45,
    )
    progress(started, "build", "after_subprocess", len(receipts))
    if not all(row["passed"] for row in receipts):
        raise RuntimeError("native_extension_build_failed")
    source = target / "release/libcarnot_python.so"
    extension = private / "extension" / f"_rust{sysconfig.get_config_var('EXT_SUFFIX') or '.so'}"
    extension.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, extension)
    binding = load_native_extension(extension)
    if (
        Path(binding.__file__).resolve() != extension.resolve()
        or binding.__dict__.get("RustPortableRecalibrationService")
        is not binding.RustPortableRecalibrationService
    ):
        raise RuntimeError("native_extension_origin_invalid")
    receipt = {
        "build_command": receipts[0]["command"],
        "module_path": str(extension.resolve()),
        "binary_hash": sha256_file(extension),
        "symbol_origin": "carnot._rust#RustPortableRecalibrationService",
        "symbol_declared_module": binding.RustPortableRecalibrationService.__module__,
        "loaded_pid": os.getpid(),
        "resolved_module_path": str(Path(binding.__file__).resolve()),
    }
    return extension, receipt, receipts


def measure(
    binding: Any, state: Path, started: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Cross the binding on each fixture and replay pending feedback."""

    native = binding.RustPortableRecalibrationService(str(state))
    rows = []
    parity = []
    for index in range(128):
        sample = payload(index)
        expected_p, expected_action = predict_reference(sample, PARAMETERS)
        _, observed_p, observed_action = native.predict_record(
            f"unit-{index}", json.dumps(sample), json.dumps(PARAMETERS)
        )
        item = {
            "unit_id": f"unit-{index}",
            "payload": sample,
            "parameter_fixture": "fixed-7710-v1",
            "python_probability": expected_p,
            "rust_probability": observed_p,
            "python_action": expected_action,
            "rust_action": observed_action,
            "python_error": None,
            "rust_error": None,
            "absolute_delta": abs(expected_p - observed_p),
            "passed": abs(expected_p - observed_p) <= 1e-6 and expected_action == observed_action,
            "state_hash": sha256_file(state.with_suffix(".records.json")),
        }
        parity.append(item)
        for arm, probability, action in (
            ("python_reference", expected_p, expected_action),
            ("direct_rust_binding", observed_p, observed_action),
        ):
            rows.append(
                {
                    "unit_id": item["unit_id"],
                    "arm": arm,
                    "raw_metrics": {"probability": probability},
                    "counts": sample["counts"],
                    "action": action,
                    "excluded": False,
                    "censored": False,
                    "seed": None,
                    "denominator": 1,
                    "provenance": "explicit_fixed_fixture",
                }
            )
        if (index + 1) % 32 == 0:
            progress(started, "parity", "checkpoint", index + 1)
    del native
    native = binding.RustPortableRecalibrationService(str(state))
    pending_before = native.record_state_summary()
    first = native.release_record_feedback("unit-0", 1)
    duplicate = native.release_record_feedback("unit-0", 1)
    unknown = native.release_record_feedback("absent", 0)
    pending_after = native.record_state_summary()
    durable = {
        "pending_before": len(pending_before[1]),
        "first": first,
        "duplicate": duplicate,
        "unknown": unknown,
        "processed_after": pending_after[0],
        "state_hash": sha256_file(state.with_suffix(".records.json")),
        "passed": len(pending_before[1]) == 128
        and first[1:3] == (True, True)
        and duplicate[3] == "duplicate_feedback:unit-0"
        and pending_after[0] == 1,
    }
    crash_state = state.with_name(f"crash-{os.getpid()}.json")
    crash_program = (
        "import os,sys\n"
        "from carnot.pipeline.native_calibrated_decision_service import load_native_extension\n"
        "s=load_native_extension(sys.argv[1]).RustPortableRecalibrationService(sys.argv[2])\n"
        "s.predict_record('crash',sys.argv[3],sys.argv[4])\n"
        "os._exit(86)\n"
    )
    progress(started, "crash_replay", "before_subprocess", len(parity))
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            crash_program,
            str(Path(binding.__file__).resolve()),
            str(crash_state),
            json.dumps(payload(2)),
            json.dumps(PARAMETERS),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    progress(started, "crash_replay", "after_subprocess", len(parity))
    crashed = binding.RustPortableRecalibrationService(str(crash_state))
    recovered = crashed.record_state_summary()[1] == ["crash"]
    crash_ack = crashed.release_record_feedback("crash", 1)
    crash_duplicate = crashed.release_record_feedback("crash", 1)
    durable["crash_restart"] = {
        "child_exit_code": child.returncode,
        "pending_recovered": recovered,
        "acknowledgment": crash_ack,
        "duplicate": crash_duplicate,
        "state_hash": sha256_file(crash_state.with_suffix(".records.json")),
    }
    durable["passed"] = bool(
        durable["passed"]
        and child.returncode == 86
        and recovered
        and crash_ack[1:3] == (True, True)
        and crash_duplicate[3] == "duplicate_feedback:crash"
    )
    invalids = [
        ("bad_schema", {**payload(2), "schema": "wrong"}, PARAMETERS),
        ("bad_counts", {**payload(2), "counts": {}}, PARAMETERS),
        ("bad_parameter_version", payload(2), {**PARAMETERS, "schema": "wrong"}),
        (
            "nan",
            {**payload(2), "counts": {**payload(2)["counts"], "tuple_supported": float("nan")}},
            PARAMETERS,
        ),
    ]
    for name, sample, params in invalids:
        try:
            predict_reference(sample, params)
            python_error = None
        except ValueError as error:
            python_error = str(error)
        try:
            native.predict_record(f"invalid-{name}", json.dumps(sample), json.dumps(params))
            rust_error = None
        except ValueError as error:
            rust_error = str(error)
        parity.append(
            {
                "unit_id": f"invalid-{name}",
                "payload": sample,
                "python_probability": None,
                "rust_probability": None,
                "python_action": None,
                "rust_action": None,
                "python_error": python_error,
                "rust_error": rust_error,
                "passed": python_error == rust_error and python_error is not None,
                "state_hash": sha256_file(state.with_suffix(".records.json")),
            }
        )
    return rows, parity, durable


def gates(ready: bool, observed: int, durable: bool) -> list[dict[str, Any]]:
    """Keep administrative readiness separate from unmeasured benefits."""

    operands = {
        "validity": {"required_checks_passed": ready},
        "readiness": {"real_extension_parity_rows": observed, "durable_replay": durable},
        "coverage": {"independent_fixture_units": observed, "minimum": 128},
        "freshness": {"fresh_evaluator_labels": 0},
        "probability": {"maximum_parity_delta": 1e-6, "held_out_labels": 0},
        "utility": {"measured_decision_cost": None},
        "retention": {"durable_replay": durable, "retention_groups": 0},
        "efficiency": {"speed_measurements": 0},
    }
    return [
        {
            "gate": name,
            "measured_operands": operands[name],
            "passed": ready
            if name == "validity"
            else ready and durable
            if name == "readiness"
            else observed >= 128
            if name == "coverage"
            else None,
            "principle": "Readiness prevents invalid propagation; benefit requires independent labels and retention.",
        }
        for name in GATES
    ]


def artifact(
    checks: list[dict[str, Any]],
    hashes: dict[str, Any],
    rows: list[dict[str, Any]],
    parity: list[dict[str, Any]],
    durable: dict[str, Any],
    extension: dict[str, Any] | None,
    receipts: list[dict[str, Any]],
    started: float,
    spans: list[dict[str, Any]],
    frozen_scope: str,
) -> dict[str, Any]:
    """Build a claim-limited terminal object from raw observations."""

    passed_inputs = all(row["passed"] for row in checks)
    passed_parity = len(parity) >= 132 and all(row["passed"] for row in parity)
    passed_validation = bool(receipts) and all(row["passed"] for row in receipts)
    ready = bool(
        passed_inputs
        and passed_parity
        and durable.get("passed")
        and passed_validation
        and extension
    )
    blocked = not passed_inputs
    verdict = (
        "complete_blocked_required_input"
        if blocked
        else "complete_null_native_record_contract_ready"
        if ready
        else "complete_disqualified_native_record_checks"
    )
    verdict_class = "blocked" if blocked else "null" if ready else "disqualified"
    output: dict[str, Any] = {
        "schema": "carnot.exp7710.v671.native_record_contract.v1",
        "experiment_id": "exp7710-native-record-contract",
        "milestone": "2026.09.671",
        "run_date": "20260926",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": checks,
        "native_record_ready_score": int(ready),
        "acceptance_gate_results": gates(
            ready, min(len(rows) // 2, 128), bool(durable.get("passed"))
        ),
        "rows": rows,
        "parity_rows": parity,
        "sample_size_budget": {
            "intended_groups": 128,
            "observed_groups": len(rows) // 2,
            "eligible_groups": len(rows) // 2,
            "excluded_groups": 0,
            "censored_groups": 0,
            "effective_blocks": len(rows) // 2,
            "prior_exposure": "fixed conformance fixtures",
            "inference_limit": "No scientific effect or trained-head claim.",
        },
        "inference_substrate": "cpu_test",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [],
        "model_specs": [{"model": "none", "reason": "typed CPU fixture conformance"}],
        "model_invoked": False,
        "invocation_counts": {
            "model_loads": 0,
            "forward_calls": 0,
            "generation_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "failures": 0,
            "cancellations": 0,
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "gpu_uuid": None,
        },
        "effective_agent_backend": {
            "requested": "codex",
            "force_experiments": os.environ.get("CODEX_FORCE_EXPERIMENTS"),
            "observed": os.environ.get("CODEX_BACKEND", "current_codex_invocation"),
        },
        "execution_recovery_receipt": {
            "current_invocation_succeeded": True,
            "pid": os.getpid(),
            "tool": "python",
            "note": "This process executed after prior V670 quota attempts.",
        },
        "phase_spans": spans,
        "duration_s": time.monotonic() - started,
        "random_seed": {
            "fixture": 7710,
            "purpose": "deterministic count construction; no sampling",
        },
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": {
            "frozen_affected_scope": frozen_scope,
            "checks": receipts,
            "terminal_exact_receipts_path": str(RAW / "terminal_exact_receipts.json"),
        },
        "verifier_is_oracle": True,
        "extension_receipt": extension,
        "parameter_provenance": {
            "conformance_fixture": deepcopy(PARAMETERS),
            "trained_parameters": None,
            "trained_smoke": "not_required_or_executed",
        },
        "durable_replay": durable,
        "prior_failures": [
            {
                "experiment_id": "exp7696-native-record-contract",
                "verdict": "not_emitted_upstream_retired_gate_skip",
            },
            {
                "experiment_id": "exp7682-native-relation-energy",
                "verdict": "not_emitted_upstream_gate_skip",
            },
        ],
        "same_verdict_retirements": [],
    }
    output["field_principles"] = {
        **PRINCIPLES,
        **{
            f"acceptance_gate_results.{name}": "Readiness prevents invalid propagation; quality and retention require separate evidence."
            for name in GATES
        },
    }
    output["reproducibility_checksum"] = canonical_hash(
        {
            "sources": hashes,
            "parameters": PARAMETERS,
            "fixture_generator": sha256_file(ROOT / MODULE),
            "reference": sha256_file(ROOT / REFERENCE),
            "rows": parity,
        }
    )
    return output


def cold_replay(path: Path) -> None:
    """Independently reduce exact candidate rows in a fresh process."""

    value = json.loads(path.read_text())
    rows = value["parity_rows"]
    if value["verdict_class"] == "blocked":
        assert value["native_record_ready_score"] == 0
        return
    assert len(rows) >= 132
    assert len({row["unit_id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["passed"] is True
        if row["python_probability"] is not None:
            assert abs(row["python_probability"] - row["rust_probability"]) <= 1e-6
            assert row["python_action"] == row["rust_action"]
        else:
            assert row["python_error"] == row["rust_error"]
    assert value["durable_replay"]["passed"] is True


def run(root: Path, output: Path) -> dict[str, Any]:
    """Preflight, measure, validate, and publish one exact terminal candidate."""

    started = time.monotonic()
    spans: list[dict[str, Any]] = []
    phase_start = started
    progress(started, "preflight", "start")
    root = root.resolve()
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.gettempdir()) / "carnot-exp7710-owned"
    private.mkdir(exist_ok=True)
    (private / "basetemp").mkdir(exist_ok=True)
    frozen = {
        "tests": [str(TEST)],
        "changed_modules": [str(REFERENCE)],
        "static_paths": [str(MODULE), str(CLI)],
        "rust": ["carnot-core::record_decision", "carnot-python"],
        "e2e": ["E2E-003", "E2E-004", "direct_service_restart"],
        "minimum_fixture_units": 128,
    }
    atomic_json(raw / "frozen_affected_scope.json", frozen)
    checks, hashes = check_input(root)
    spans.append(
        {
            "phase": "preflight",
            "start_monotonic_s": phase_start,
            "end_monotonic_s": time.monotonic(),
            "completed_units": len(checks),
            "checkpoint": str(raw / "frozen_affected_scope.json"),
        }
    )
    progress(started, "preflight", "complete", len(checks))
    receipts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    parity: list[dict[str, Any]] = []
    durable: dict[str, Any] = {}
    extension_receipt = None
    if all(row["passed"] for row in checks):
        phase_start = time.monotonic()
        extension, extension_receipt, build_receipts = build_extension(root, raw, private, started)
        receipts.extend(build_receipts)
        spans.append(
            {
                "phase": "build",
                "start_monotonic_s": phase_start,
                "end_monotonic_s": time.monotonic(),
                "completed_units": 1,
                "checkpoint": extension_receipt["binary_hash"],
            }
        )
        phase_start = time.monotonic()
        progress(started, "parity", "start")
        binding = load_native_extension(extension)
        rows, parity, durable = measure(binding, private / f"service-{os.getpid()}.json", started)
        atomic_json(
            raw / "parity_checkpoint.json",
            {"completed_units": len(parity), "passed": all(r["passed"] for r in parity)},
        )
        spans.append(
            {
                "phase": "parity",
                "start_monotonic_s": phase_start,
                "end_monotonic_s": time.monotonic(),
                "completed_units": len(parity),
                "checkpoint": str(raw / "parity_checkpoint.json"),
            }
        )
        progress(started, "parity", "complete", len(parity))
        commands = validation.build_scoped_commands(
            root,
            [str(TEST)],
            [str(REFERENCE)],
            static_paths=[str(MODULE), str(CLI)],
            basetemp=private / "basetemp",
            coverage_file=private / ".coverage-exp7710",
        )
        python = str(root / ".venv/bin/python")
        pytest = str(root / ".venv/bin/pytest")
        commands.extend(
            [
                validation.CommandSpec(
                    "e2e_003_real_extension",
                    (
                        pytest,
                        "-n",
                        "0",
                        "-o",
                        "addopts=",
                        "--no-cov",
                        f"--basetemp={private / 'basetemp/e2e003'}",
                        f"{TEST}::test_scenario_pybind_7710_real_extension_and_restart",
                        "-q",
                    ),
                    "E2E-003 actual binding",
                    300,
                ),
                validation.CommandSpec(
                    "e2e_004_cross_language_reload",
                    (
                        pytest,
                        "-n",
                        "0",
                        "-o",
                        "addopts=",
                        "--no-cov",
                        f"--basetemp={private / 'basetemp/e2e004'}",
                        f"{TEST}::test_scenario_pybind_7710_real_extension_and_restart",
                        "-q",
                    ),
                    "E2E-004 durable JSON reload",
                    300,
                ),
                validation.CommandSpec(
                    "rust_core_tests",
                    (
                        "cargo",
                        "test",
                        "-p",
                        "carnot-core",
                        "record_decision",
                        "--target-dir",
                        str(private / "target"),
                    ),
                    "scoped Rust",
                    900,
                ),
                validation.CommandSpec(
                    "rust_format", ("cargo", "fmt", "--all", "--", "--check"), "Rust format", 120
                ),
                validation.CommandSpec(
                    "rust_clippy",
                    (
                        "cargo",
                        "clippy",
                        "-p",
                        "carnot-core",
                        "-p",
                        "carnot-python",
                        "--target-dir",
                        str(private / "target"),
                        "--",
                        "-D",
                        "warnings",
                    ),
                    "changed Rust crates",
                    900,
                ),
                validation.CommandSpec(
                    "full_python_suite",
                    (
                        pytest,
                        "tests/python",
                        "-q",
                        "-n",
                        "0",
                        "-o",
                        "addopts=",
                        f"--basetemp={private / 'basetemp/full'}",
                    ),
                    "required repository Python suite",
                    1800,
                ),
            ]
        )
        phase_start = time.monotonic()
        progress(started, "validation", "start", len(commands))
        receipts.extend(
            validation.run_commands(
                root,
                commands,
                log_dir=raw / "validation/affected",
                extra_env={
                    "CARNOT_7710_EXTENSION": str(extension),
                    "COVERAGE_FILE": str(private / ".coverage-exp7710"),
                },
                heartbeat_s=45,
            )
        )
        spans.append(
            {
                "phase": "validation",
                "start_monotonic_s": phase_start,
                "end_monotonic_s": time.monotonic(),
                "completed_units": len(commands),
                "checkpoint": str(raw / "validation/affected"),
            }
        )
        progress(started, "validation", "complete", len(commands))
    else:
        progress(started, "preflight", "blocked")
    value = artifact(
        checks,
        hashes,
        rows,
        parity,
        durable,
        extension_receipt,
        receipts,
        started,
        spans,
        str(RAW / "frozen_affected_scope.json"),
    )
    candidate = raw / "terminal_candidate.json"
    atomic_json(candidate, value)
    progress(started, "terminal", "before_subprocess")
    terminal = [
        validation.CommandSpec(
            "cold_reduction",
            (str(root / ".venv/bin/python"), str(root / CLI), "--cold-replay", str(candidate)),
            "exact candidate",
            120,
        ),
        validation.CommandSpec(
            "adversarial_verify",
            (str(root / ".venv/bin/python"), "scripts/adversarial_verify.py", str(candidate)),
            "exact candidate",
            120,
        ),
        validation.CommandSpec(
            "strict_row_lint",
            (
                str(root / ".venv/bin/python"),
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact candidate",
            120,
        ),
    ]
    terminal_receipts = validation.run_commands(
        root, terminal, log_dir=raw / "validation/terminal", heartbeat_s=45
    )
    atomic_json(
        raw / "terminal_exact_receipts.json",
        {"candidate_sha256": sha256_file(candidate), "checks": terminal_receipts},
    )
    if not all(row["passed"] for row in terminal_receipts):
        value["honest_verdict"] = "complete_disqualified_terminal_reader"
        value["verdict_class"] = "disqualified"
        value["native_record_ready_score"] = 0
        value["flagged_adversarial"] = not terminal_receipts[1]["passed"]
        atomic_json(candidate, value)
        terminal_receipts = validation.run_commands(
            root, terminal, log_dir=raw / "validation/terminal_exact", heartbeat_s=45
        )
        atomic_json(
            raw / "terminal_exact_receipts.json",
            {"candidate_sha256": sha256_file(candidate), "checks": terminal_receipts},
        )
    progress(started, "terminal", "after_subprocess", len(terminal_receipts))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(candidate.read_bytes())
    progress(started, "publication", "complete", len(parity))
    return value


def main(argv: list[str] | None = None) -> None:
    """Run producer or cold reader from a thin command-line wrapper."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260926")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--cold-replay")
    args = parser.parse_args(argv)
    if args.cold_replay:
        cold_replay(Path(args.cold_replay))
        return
    if args.date != "20260926":
        raise ValueError("run_date_invalid")
    run(ROOT, ROOT / args.output)

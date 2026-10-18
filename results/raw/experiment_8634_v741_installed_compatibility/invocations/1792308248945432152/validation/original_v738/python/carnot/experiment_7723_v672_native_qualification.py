"""V672 qualification of the existing typed-record native service.

This experiment checks fixed fixtures and durable service behavior. It does not
train a policy or infer a decision benefit from fixture agreement.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import sysconfig
import tempfile
from time import monotonic
from typing import Any

from carnot import experiment_7710_v671_native_record_contract as prior
from carnot.pipeline.native_calibrated_decision_service import load_native_extension
from carnot.pipeline.native_record_decision import predict_reference
from carnot.reporting import experiment_7303_validation_scope as validation
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7723_v672_native_qualification")
OUTPUT = Path("results/experiment_7723_v672_native_qualification.json")
MODULE = Path("python/carnot/experiment_7723_v672_native_qualification.py")
CLI = Path("scripts/experiments/experiment_7723_v672_native_qualification.py")
TEST = Path("tests/python/test_experiment_7723_v672_native_qualification.py")
PRIOR_TEST = Path("tests/python/test_experiment_7710_v671_native_record_contract.py")
PRIOR_RESULT = Path("results/experiment_7710_v671_native_record_contract.json")
PRIOR_RAW = Path("results/raw/experiment_7710_v671_native_record_contract")
REQUIRED_OLD_FAILURES = {
    "changed_module_coverage_report",
    "ruff_format",
    "rust_format",
    "rust_clippy",
    "full_python_suite",
}
GATE_NAMES = (
    "measured_validity",
    "administrative_readiness",
    "probability",
    "utility",
    "coverage",
    "source_dependence",
    "retention",
    "efficiency",
)
PRINCIPLE = "Measured evidence bounds the claim and prevents invalid downstream use."


def progress(started: float, phase: str, event: str, units: int = 0) -> None:
    """Emit one flushed owned boundary with elapsed time and completed units."""

    print(
        f"[exp7723] phase={phase} event={event} "
        f"elapsed_s={monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def check_preconditions(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Authenticate producer bytes, exact V671 failure fields, and build resources."""

    checks: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {
        "valid_producers": {},
        "flagged_historical_evidence": {},
        "pre_gate_receipts": {},
        "missing_custody": [],
    }

    def check(
        name: str,
        upstream: str,
        path: str,
        field: str,
        expected: Any,
        observed: Any,
        category: str | None = None,
    ) -> None:
        passed = observed == expected
        checks.append(
            {
                "check": name,
                "upstream_id": upstream,
                "artifact_path": path,
                "field": field,
                "operator": "eq",
                "expected": expected,
                "observed": observed,
                "passed": passed,
            }
        )
        if category and passed and (root / path).is_file():
            hashes[category][path] = sha256_file(root / path)
        elif category and not (root / path).is_file():
            hashes["missing_custody"].append(path)

    for path, category, upstream in (
        (str(prior.UPSTREAM), "valid_producers", "exp7700"),
        (str(prior.SCHEMA), "valid_producers", "exp7700"),
        (str(PRIOR_RESULT), "flagged_historical_evidence", "exp7710"),
        (str(PRIOR_TEST), "pre_gate_receipts", "exp7710-standalone-repair"),
    ):
        check(
            "input_exists", upstream, path, "file_exists", True, (root / path).is_file(), category
        )
    if (root / prior.UPSTREAM).is_file():
        producer = json.loads((root / prior.UPSTREAM).read_bytes())
        for field, expected in (
            ("record_protocol_ready_score", 1),
            ("flagged_adversarial", False),
        ):
            check(
                "producer_gate",
                "exp7700",
                str(prior.UPSTREAM),
                field,
                expected,
                producer.get(field),
            )
    if (root / prior.SCHEMA).is_file():
        schema = json.loads((root / prior.SCHEMA).read_bytes())
        check(
            "feature_schema",
            "exp7700",
            str(prior.SCHEMA),
            "schema",
            "carnot.exp7700.record_features.v1",
            schema.get("schema"),
        )
    if (root / PRIOR_RESULT).is_file():
        old = json.loads((root / PRIOR_RESULT).read_bytes())
        check(
            "historical_verdict",
            "exp7710",
            str(PRIOR_RESULT),
            "verdict_class",
            "disqualified",
            old.get("verdict_class"),
        )
        check(
            "historical_readiness",
            "exp7710",
            str(PRIOR_RESULT),
            "native_record_ready_score",
            0,
            old.get("native_record_ready_score"),
        )
        receipts = old.get("validation_receipts", {}).get("checks", [])
        failed = {r.get("name") for r in receipts if r.get("passed") is False}
        check(
            "historical_failures",
            "exp7710",
            str(PRIOR_RESULT),
            "validation_receipts.failed_names",
            sorted(REQUIRED_OLD_FAILURES),
            sorted(failed),
        )
        for name in REQUIRED_OLD_FAILURES:
            for receipt in receipts:
                if receipt.get("name") == name:
                    path = str(receipt.get("log_path"))
                    observed = sha256_file(root / path) if (root / path).is_file() else None
                    check(
                        "historical_log_hash",
                        "exp7710",
                        path,
                        "log_sha256",
                        receipt.get("log_sha256"),
                        observed,
                        "flagged_historical_evidence",
                    )
    if (root / PRIOR_TEST).is_file():
        source = (root / PRIOR_TEST).read_text()
        check(
            "standalone_repair_source",
            "exp7710-standalone-repair",
            str(PRIOR_TEST),
            "fallback_build_and_load",
            True,
            "CARNOT_7710_EXTENSION" in source
            and "cargo" in source
            and "load_native_extension" in source,
        )
    check("cargo_available", "host", "PATH", "cargo", True, bool(shutil.which("cargo")))
    check(
        "python_abi",
        "host",
        sys.executable,
        "EXT_SUFFIX",
        True,
        bool(sysconfig.get_config_var("EXT_SUFFIX")),
    )
    return checks, hashes


def measure_qualification(
    binding: Any, state: Path, raw: Path, *, started: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Replay original fixtures and both hard-exit sides of feedback."""

    progress(started, "parity", "before_original_128")
    rows, parity, original = prior.measure(binding, state, started)
    for item in parity:
        item["restart_outcome"] = (
            "normal_reopen" if item["unit_id"].startswith("unit-") else "not_applicable"
        )
    progress(started, "parity", "after_original_128", 128)
    native = binding.RustPortableRecalibrationService(str(state.with_name("boundary.json")))
    valid = [
        (
            "fraction-one",
            {**prior.payload(2), "counts": {**prior.payload(2)["counts"], "checked_fraction": 1.0}},
            prior.PARAMETERS,
        ),
        (
            "large-count",
            {
                **prior.payload(3),
                "counts": {**prior.payload(3)["counts"], "source_records": 1_000_000_000},
            },
            prior.PARAMETERS,
        ),
    ]
    invalid = [
        (
            "negative-count",
            {**prior.payload(2), "counts": {**prior.payload(2)["counts"], "source_records": -1}},
            prior.PARAMETERS,
        ),
        ("bad-thresholds", prior.payload(2), {**prior.PARAMETERS, "thresholds": [0.9, 0.1]}),
    ]
    for name, sample, params in [*valid, *invalid]:
        try:
            py_probability, py_action = predict_reference(sample, params)
            py_error = None
        except ValueError as error:
            py_probability, py_action, py_error = None, None, str(error)
        try:
            _, rust_probability, rust_action = native.predict_record(
                name, json.dumps(sample), json.dumps(params)
            )
            rust_error = None
        except ValueError as error:
            rust_probability, rust_action, rust_error = None, None, str(error)
        passed = (
            abs(py_probability - rust_probability) <= 1e-6 and py_action == rust_action
            if py_probability is not None and rust_probability is not None
            else py_error is not None and py_error == rust_error
        )
        parity.append(
            {
                "unit_id": f"boundary-{name}",
                "payload": sample,
                "parameters": params,
                "python_probability": py_probability,
                "rust_probability": rust_probability,
                "python_action": py_action,
                "rust_action": rust_action,
                "python_error": py_error,
                "rust_error": rust_error,
                "passed": passed,
                "restart_outcome": "not_applicable",
                "provenance": "fixed_boundary_fixture",
            }
        )
    progress(started, "parity", "after_boundaries", len(parity))
    crash_after = state.with_name("crash-after-ack.json")
    script = (
        "import json,os,sys\n"
        "from carnot.pipeline.native_calibrated_decision_service import load_native_extension\n"
        "s=load_native_extension(sys.argv[1]).RustPortableRecalibrationService(sys.argv[2])\n"
        "s.predict_record('crash-after',sys.argv[3],sys.argv[4])\n"
        "assert s.release_record_feedback('crash-after',1)[1:3] == (True,True)\n"
        "print('acknowledged before hard exit',flush=True)\n"
        "os._exit(87)\n"
    )
    command = validation.CommandSpec(
        "hard_exit_after_ack",
        (
            sys.executable,
            "-u",
            "-c",
            script,
            str(Path(binding.__file__).resolve()),
            str(crash_after),
            json.dumps(prior.payload(4)),
            json.dumps(prior.PARAMETERS),
        ),
        "E2E-004 real PyO3 process",
        30,
    )
    progress(started, "restart", "before_subprocess", len(parity))
    child = validation.run_commands(
        ROOT, [command], log_dir=raw / "validation/restart", heartbeat_s=10
    )[0]
    progress(started, "restart", "after_subprocess", len(parity))
    reopened = binding.RustPortableRecalibrationService(str(crash_after))
    count, pending, processed, _schema = reopened.record_state_summary()
    duplicate = reopened.release_record_feedback("crash-after", 1)
    after = {
        "child_exit_code": child["exit_code"],
        "child_log_sha256": child["log_sha256"],
        "processed_count": count,
        "pending": pending,
        "processed_ids": processed,
        "duplicate": duplicate,
        "state_hash": sha256_file(crash_after.with_suffix(".records.json")),
        "passed": child["exit_code"] == 87
        and count == 1
        and pending == []
        and processed == ["crash-after"]
        and duplicate[3] == "duplicate_feedback:crash-after",
    }
    before = original["crash_restart"]
    durability = {
        "normal_restart": {"passed": original["passed"], "raw": original},
        "crash_before_ack": {
            "passed": before["child_exit_code"] == 86 and before["pending_recovered"],
            "raw": before,
        },
        "crash_after_ack": after,
        "fsync_policy": "atomic_file_fsync_rename_directory_fsync_reload_ack",
    }
    durability["passed"] = all(
        durability[key]["passed"]
        for key in ("normal_restart", "crash_before_ack", "crash_after_ack")
    )
    return rows, parity, durability


def build_artifact(
    *,
    checks: list[dict[str, Any]],
    hashes: dict[str, Any],
    rows: list[dict[str, Any]],
    parity: list[dict[str, Any]],
    durability: dict[str, Any],
    extension: dict[str, Any] | None,
    affected_receipts: list[dict[str, Any]],
    global_receipts: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    frozen_scope: str,
    started: float | None = None,
) -> dict[str, Any]:
    """Bind exact measurements to a terminal, claim-limited result."""

    origin = monotonic() if started is None else started
    base = prior.artifact(
        checks,
        hashes,
        rows,
        parity,
        durability,
        extension,
        affected_receipts,
        origin,
        spans,
        frozen_scope,
    )
    original = [r for r in parity if r["unit_id"].startswith("unit-")]
    parity_ok = len(original) == 128 and len(parity) >= 136 and all(r["passed"] for r in parity)
    required = validation.reduce_required_checks(affected_receipts)
    named = {r.get("name"): r for r in affected_receipts}
    e2e_ok = all(
        named.get(name, {}).get("passed") is True
        for name in (
            "e2e_003",
            "e2e_004",
            "rust_core_tests",
            "rust_python_tests",
            "rust_affected_format",
            "rust_affected_clippy",
        )
    )
    valid = all(r["passed"] for r in checks)
    ready = bool(
        valid
        and parity_ok
        and durability.get("passed")
        and extension
        and required["required_checks_passed"]
        and e2e_ok
    )
    blocked = any(not row["passed"] and row["upstream_id"] != "exp7723" for row in checks)
    verdict = (
        "complete_blocked_required_input"
        if blocked
        else "complete_circular_positive_native_service_qualified"
        if ready
        else "complete_disqualified_native_service_checks"
    )
    sample = {
        "intended_families": 128,
        "observed_families": len(original),
        "eligible_families": len(original),
        "excluded_families": 0,
        "censored_families": 0,
        "effective_blocks": len(original),
        "roles": ["python_reference", "direct_rust_binding"],
        "exposure": "fixed conformance fixture; two arms share each original payload",
        "boundary_error_cases": len(parity) - len(original),
    }
    gate_values = {
        "measured_validity": {
            "passed": valid and parity_ok,
            "input_checks": len(checks),
            "parity_rows": len(parity),
        },
        "administrative_readiness": {
            "passed": ready,
            "affected_checks": required,
            "e2e_passed": e2e_ok,
        },
        "probability": {
            "passed": None,
            "held_out_labels": 0,
            "max_parity_delta": max((r.get("absolute_delta", 0) for r in parity), default=None),
        },
        "utility": {"passed": None, "measured_decision_cost": None},
        "coverage": {
            "passed": len(original) == 128,
            "original_payloads": len(original),
            "minimum": 128,
        },
        "source_dependence": {"passed": None, "original_source_bytes_hashes": []},
        "retention": {
            "passed": None,
            "durable_service_replay": durability.get("passed"),
            "independent_retention_groups": 0,
        },
        "efficiency": {"passed": None, "speed_measurements": 0},
    }
    gates = [{"gate": name, **gate_values[name], "principle": PRINCIPLE} for name in GATE_NAMES]
    base.update(
        {
            "schema": "carnot.exp7723.v672.native_qualification.v1",
            "experiment_id": "exp7723-v672-native-qualification",
            "milestone": "2026.09.672",
            "run_date": "20260926",
            "honest_verdict": verdict,
            "verdict_class": "blocked"
            if blocked
            else "circular_positive"
            if ready
            else "disqualified",
            "flagged_adversarial": False,
            "gate_check_summary": checks,
            "native_service_ready_score": int(ready),
            "native_record_ready_score": int(ready),
            "acceptance_gate_results": gates,
            "sample_size_budget": sample,
            "model_specs": [],
            "MODEL_SPECS": [],
            "planned_MODEL_SPECS": [],
        "inference_substrate": "deterministic_verifier_plus_replay",
            "inference_substrate_class": "no_model_load",
            "planned_inference_substrate_class": "no_model_load",
            "source_artifact_hashes": hashes,
            "preconditions_checked": checks,
            "validation_receipts": {
                "frozen_affected_scope": frozen_scope,
                "checks": affected_receipts,
                "global_repository_health": global_receipts,
                "terminal_exact_receipts_path": str(RAW / "terminal_exact_receipts.json"),
                **required,
            },
            "verifier_is_oracle": True,
            "durable_replay": durability,
            "native_scope_manifest": {
                "api": ["predict_record", "release_record_feedback", "record_state_summary"],
                "parameter_fixture": "fixed-7710-v1",
                "source_alignment_model": "excluded",
                "entrypoints_frozen_for_exp7724": [
                    "python/carnot/pipeline/native_record_decision.py",
                    "crates/carnot-core/src/record_decision.rs",
                    "crates/carnot-python/src/portable_recalibration.rs",
                ],
            },
            "historical_v671_verdict": "complete_disqualified_native_record_checks",
            "repository_health": {
                "status": "degraded_open"
                if any(not r.get("passed") for r in global_receipts)
                else "healthy",
                "checks": global_receipts,
            },
            "same_verdict_retirements": [],
        }
    )
    base["field_principles"] = {
        **{
            field: PRINCIPLE
            for field in (
                "honest_verdict",
                "verdict_class",
                "flagged_adversarial",
                "gate_check_summary",
                "acceptance_gate_results",
                "rows",
                "sample_size_budget",
                "inference_substrate",
                "inference_substrate_class",
                "MODEL_SPECS",
                "model_invoked",
                "execution_venue",
                "phase_spans",
                "random_seed",
                "reproducibility_checksum",
                "source_artifact_hashes",
                "preconditions_checked",
                "validation_receipts",
                "verifier_is_oracle",
                "native_service_ready_score",
                "native_scope_manifest",
                "extension_receipt",
                "parity_rows",
            )
        },
        **{f"acceptance_gate_results.{name}": PRINCIPLE for name in GATE_NAMES},
    }
    base["reproducibility_checksum"] = canonical_hash(
        {
            "source_hashes": hashes,
            "fixture": prior.PARAMETERS,
            "reducer_code": sha256_file(ROOT / MODULE),
            "parity_rows": parity,
        }
    )
    return base


def cold_reduce(path: Path) -> None:
    """Recompute terminal claim constraints from exact candidate bytes."""

    result = json.loads(path.read_bytes())
    if result["verdict_class"] in {"blocked", "disqualified"} and not result["parity_rows"]:
        assert result["native_service_ready_score"] == 0
        assert any(not row["passed"] for row in result["gate_check_summary"])
        return
    parity = result["parity_rows"]
    assert len([r for r in parity if r["unit_id"].startswith("unit-")]) == 128
    assert len(parity) >= 136
    assert len({r["unit_id"] for r in parity}) == len(parity)
    for row in parity:
        if row["python_probability"] is not None:
            assert abs(row["python_probability"] - row["rust_probability"]) <= 1e-6
            assert row["python_action"] == row["rust_action"]
        else:
            assert row["python_error"] is not None
            assert row["python_error"] == row["rust_error"]
        assert row["passed"] is True
    assert result["durable_replay"]["passed"] is True
    assert result["native_service_ready_score"] == int(
        result["verdict_class"] == "circular_positive"
    )


def run(root: Path, output: Path) -> dict[str, Any]:
    """Run frozen preflight, measurement, validation, and exact publication."""

    started = monotonic()
    root = root.resolve()
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7723-"))
    (private / "basetemp").mkdir()
    spans: list[dict[str, Any]] = []

    def span(name: str, begin: float, units: int, checkpoint: str) -> None:
        end = monotonic()
        spans.append(
            {
                "phase": name,
                "start_monotonic_s": begin,
                "end_monotonic_s": end,
                "duration_s": end - begin,
                "heartbeat_timestamps": [begin, end],
                "completed_units": units,
                "checkpoint": checkpoint,
            }
        )

    phase_start = monotonic()
    progress(started, "preflight", "start")
    frozen = raw / "frozen_scope.json"
    if not frozen.is_file():
        raise FileNotFoundError(f"frozen_scope_missing:{frozen}")
    checks, hashes = check_preconditions(root)
    span("preflight", phase_start, len(checks), str(frozen))
    progress(started, "preflight", "complete", len(checks))
    rows: list[dict[str, Any]] = []
    parity: list[dict[str, Any]] = []
    durability: dict[str, Any] = {}
    extension_receipt: dict[str, Any] | None = None
    affected: list[dict[str, Any]] = []
    global_receipts: list[dict[str, Any]] = []
    if all(row["passed"] for row in checks):
        phase_start = monotonic()
        progress(started, "build", "start")
        try:
            extension, extension_receipt, build = prior.build_extension(root, raw, private, started)
            affected.extend(build)
            extension_receipt["abi"] = str(sysconfig.get_config_var("SOABI"))
            extension_receipt["build_hash"] = extension_receipt["binary_hash"]
            span("build", phase_start, 1, extension_receipt["binary_hash"])
            progress(started, "build", "complete", 1)
            phase_start = monotonic()
            binding = load_native_extension(extension)
            rows, parity, durability = measure_qualification(
                binding, private / "service.json", raw, started=started
            )
            checkpoint = raw / "parity_checkpoint.json"
            atomic_json(
                checkpoint,
                {
                    "completed_units": len(parity),
                    "passed": all(r["passed"] for r in parity),
                    "durable": durability.get("passed"),
                },
            )
            span("parity_and_restart", phase_start, len(parity), str(checkpoint))
            progress(started, "parity_and_restart", "complete", len(parity))
            phase_start = monotonic()
            progress(started, "validation", "before_subprocess", 0)
            commands = validation.build_scoped_commands(
                root,
                [str(TEST), str(PRIOR_TEST)],
                [str(MODULE)],
                static_paths=[str(CLI)],
                basetemp=private / "basetemp",
                coverage_file=private / ".coverage-exp7723",
            )
            pytest = str(root / ".venv/bin/pytest")
            commands.extend(
                [
                    validation.CommandSpec(
                        "e2e_003",
                        (
                            pytest,
                            "-n",
                            "0",
                            "-o",
                            "addopts=",
                            "--no-cov",
                            f"--basetemp={private / 'e2e003'}",
                            f"{TEST}::test_scenario_pybind_7723_restart_and_parity",
                            "-q",
                        ),
                        "real PyO3 parity",
                        300,
                    ),
                    validation.CommandSpec(
                        "e2e_004",
                        (
                            pytest,
                            "-n",
                            "0",
                            "-o",
                            "addopts=",
                            "--no-cov",
                            f"--basetemp={private / 'e2e004'}",
                            f"{TEST}::test_scenario_pybind_7723_restart_and_parity",
                            "-q",
                        ),
                        "normal and hard-exit replay",
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
                        "affected Rust module",
                        900,
                    ),
                    validation.CommandSpec(
                        "rust_python_tests",
                        (
                            "cargo",
                            "test",
                            "-p",
                            "carnot-python",
                            "--target-dir",
                            str(private / "target"),
                        ),
                        "affected native crate",
                        900,
                    ),
                    validation.CommandSpec(
                        "rust_affected_format",
                        (
                            "rustfmt",
                            "--edition",
                            "2021",
                            "--check",
                            "crates/carnot-core/src/record_decision.rs",
                            "crates/carnot-python/src/portable_recalibration.rs",
                        ),
                        "affected Rust files",
                        120,
                    ),
                    validation.CommandSpec(
                        "rust_affected_clippy",
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
                        "affected crates",
                        900,
                    ),
                ]
            )
            affected.extend(
                validation.run_commands(
                    root,
                    commands,
                    log_dir=raw / "validation/affected",
                    extra_env={
                        "CARNOT_7710_EXTENSION": str(extension),
                        "COVERAGE_FILE": str(private / ".coverage-exp7723"),
                    },
                    heartbeat_s=45,
                )
            )
            span("validation", phase_start, len(commands), str(raw / "validation/affected"))
            progress(started, "validation", "after_subprocess", len(commands))
        except (OSError, RuntimeError, ValueError) as error:
            checks.append(
                {
                    "check": "owned_execution",
                    "upstream_id": "exp7723",
                    "artifact_path": str(raw),
                    "field": "execution_error",
                    "operator": "eq",
                    "expected": None,
                    "observed": str(error),
                    "passed": False,
                }
            )
            progress(started, "execution", "failed", len(parity))
    else:
        progress(started, "preflight", "blocked", len(checks))
    if all(row["passed"] for row in checks):
        phase_start = monotonic()
        progress(started, "global_health", "before_subprocess")
        pytest = str(root / ".venv/bin/pytest")
        global_receipts = validation.run_commands(
            root,
            [
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
                        f"--basetemp={private / 'full'}",
                    ),
                    "repository health; separate from frozen affected scope",
                    1800,
                ),
                validation.CommandSpec(
                    "cargo_fmt_all",
                    ("cargo", "fmt", "--all", "--", "--check"),
                    "repository health",
                    120,
                ),
                validation.CommandSpec(
                    "cargo_clippy_workspace",
                    (
                        "cargo",
                        "clippy",
                        "--workspace",
                        "--exclude",
                        "carnot-python",
                        "--target-dir",
                        str(private / "target"),
                        "--",
                        "-D",
                        "warnings",
                    ),
                    "repository health",
                    900,
                ),
            ],
            log_dir=raw / "validation/global",
            heartbeat_s=45,
        )
        span("global_health", phase_start, len(global_receipts), str(raw / "validation/global"))
        progress(started, "global_health", "after_subprocess", len(global_receipts))
    value = build_artifact(
        checks=checks,
        hashes=hashes,
        rows=rows,
        parity=parity,
        durability=durability,
        extension=extension_receipt,
        affected_receipts=affected,
        global_receipts=global_receipts,
        spans=spans,
        frozen_scope=str(RAW / "frozen_scope.json"),
        started=started,
    )
    value["execution_venue"] = "host"
    value["execution_venue_details"] = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "gpu_uuid": None,
    }
    value["effective_execution_backend"] = {
        "actual": "local_python_process_with_cargo_subprocesses",
        "python_executable": sys.executable,
        "requested_routing": os.environ.get("CODEX_BACKEND"),
        "routing_verified": False,
    }
    candidate = raw / "terminal_candidate.json"
    atomic_json(candidate, value)
    phase_start = monotonic()
    progress(started, "terminal", "before_subprocess", len(parity))
    terminal = [
        validation.CommandSpec(
            "cold_reduction",
            (str(root / ".venv/bin/python"), str(root / CLI), "--cold-replay", str(candidate)),
            "exact candidate in fresh process",
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
        root, terminal, log_dir=raw / "validation/terminal", heartbeat_s=30
    )
    if not all(row["passed"] for row in terminal_receipts):
        value["honest_verdict"] = "complete_disqualified_terminal_reader"
        value["verdict_class"] = "disqualified"
        value["native_service_ready_score"] = 0
        value["native_record_ready_score"] = 0
        value["flagged_adversarial"] = not terminal_receipts[1]["passed"]
        atomic_json(candidate, value)
        terminal_receipts = validation.run_commands(
            root, terminal, log_dir=raw / "validation/terminal_exact", heartbeat_s=30
        )
    receipt_path = raw / "terminal_exact_receipts.json"
    atomic_json(
        receipt_path,
        {
            "candidate_sha256": sha256_file(candidate),
            "checks": terminal_receipts,
        },
    )
    span("terminal", phase_start, len(terminal_receipts), str(receipt_path))
    progress(started, "terminal", "after_subprocess", len(terminal_receipts))
    atomic_json(output, value)
    progress(started, "publication", "complete", len(parity))
    return value


def main(argv: list[str] | None = None) -> None:
    """Run the producer or the independent candidate reader."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260926")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--cold-replay")
    args = parser.parse_args(argv)
    if args.cold_replay:
        cold_reduce(Path(args.cold_replay))
    elif args.date != "20260926":
        raise ValueError("run_date_invalid")
    else:
        run(ROOT, ROOT / args.output)

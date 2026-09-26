"""Publish the CPU acquisition fixture qualification for REQ-REPORT-7719."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

from carnot.reporting.acquisition_qualification import (
    ARMS,
    fit_static_closure,
    fixture_groups,
    run_fixture,
)
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7719_v672_acquisition_qualification")
OUTPUT = Path("results/experiment_7719_v672_acquisition_qualification.json")
V671 = Path("results/experiment_7705_v671_constraint_bank_protocol.json")
CAPABILITY = "python/carnot/reporting/acquisition_qualification.py"
ORCHESTRATION = "python/carnot/experiment_7719_v672_acquisition_qualification.py"
WRAPPER = "scripts/experiments/experiment_7719_v672_acquisition_qualification.py"
TEST = "tests/python/test_experiment_7719_v672_acquisition_qualification.py"
SCOPE = {
    "tests": [TEST],
    "changed_modules": [CAPABILITY, ORCHESTRATION],
    "static_paths": [WRAPPER],
    "specs": ["REQ-REPORT-7719", "REQ-CL-7719-CAUSAL-ADMISSION"],
    "e2e": ["causal_commit", "harmful_rejection", "hard_exit_replay", "cold_reduce"],
}
MODEL_SPECS: list[str] = []
PRINCIPLE = "Measured evidence bounds the claim and prevents invalid downstream use."
FIELDS = (
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
    "field_principles",
    "acquisition_protocol_ready_score",
    "commit_reachability_rows",
    "static_closure_receipt",
)
GATES = (
    "validity",
    "readiness",
    "probability",
    "utility",
    "coverage",
    "source_dependence",
    "retention",
    "efficiency",
)


def progress(started: float, phase: str, event: str, units: int = 0) -> None:
    """Flush each phase boundary and checkpoint with elapsed CPU task time."""
    print(
        f"[exp7719] {phase} {event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def check(
    name: str, upstream: str, path: str, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Keep literal operands for every missing or failed input."""
    return {
        "check": name,
        "upstream_id": upstream,
        "artifact_path": path,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def preflight(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Authenticate V671 bytes and host resources without requiring its readiness."""
    root = root.resolve()
    checks = [
        check(
            "cpu_available",
            "host",
            "/proc/self",
            "cpu_count_positive",
            True,
            (os.cpu_count() or 0) > 0,
        ),
        check(
            "disk_available",
            "host",
            str(root),
            "free_bytes_at_least_10m",
            True,
            os.statvfs(root).f_bavail * os.statvfs(root).f_frsize >= 10_000_000,
        ),
    ]
    hashes: dict[str, Any] = {
        "valid_producers": {},
        "flagged_historical_evidence": {},
        "pre_gate_receipts": {},
        "missing_custody": [],
    }
    path = root / V671
    checks.append(
        check(
            "historical_exists",
            "exp7705-constraint-bank-protocol",
            str(V671),
            "exists",
            True,
            path.is_file(),
        )
    )
    if not path.is_file():
        hashes["missing_custody"].append(str(V671))
        return checks, hashes
    digest = sha256_file(path)
    try:
        previous = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        previous = {}
    for field, expected in (
        ("schema", "carnot.exp7705.v671.constraint_bank.v1"),
        ("experiment_id", 7705),
        ("constraint_bank_ready_score", 0),
        ("honest_verdict", "complete_circular_positive_fixture_mechanics_no_acquisition"),
    ):
        checks.append(
            check(
                "historical_schema",
                "exp7705-constraint-bank-protocol",
                str(V671),
                field,
                expected,
                previous.get(field),
            )
        )
    hashes[
        "flagged_historical_evidence" if previous.get("flagged_adversarial") else "valid_producers"
    ][str(V671)] = digest
    return checks, hashes


def cold_reduce(path: Path) -> dict[str, Any]:
    """In a fresh process, recompute row losses and reopen every bank ledger."""
    artifact = json.loads(path.read_text())
    rows = artifact.get("rows", [])
    arms = artifact.get("arms", [])
    keys = {(row["arm"], row["unit_id"]) for row in rows}
    complete = (
        len(rows) == 96 * len(arms)
        and len(keys) == len(rows)
        and len({row["unit_id"] for row in rows}) == 96
    )
    arithmetic = all(
        row["raw_metrics"]["label"] in (0, 1)
        and abs(
            row["raw_metrics"]["brier"]
            - (row["raw_metrics"]["probability"] - row["raw_metrics"]["label"]) ** 2
        )
        < 1e-12
        and row["raw_metrics"]["exact_status"] == row["raw_metrics"]["advisory_status"]
        for row in rows
    )
    from carnot.reporting.constraint_bank_protocol import Bank, grammar

    groups = fixture_groups()
    originals = {case["unit_id"]: case for group in groups.values() for case in group}
    source_valid = all(
        row["unit_id"] in originals
        and row["provenance"]["source_bytes_hex"] == originals[row["unit_id"]]["source_bytes_hex"]
        and row["raw_metrics"]["label"] == originals[row["unit_id"]]["label"]
        for row in rows
    )
    closure_valid = artifact.get("static_closure_receipt") == fit_static_closure(
        groups["development"]
    )
    bank_rows_valid = True
    replay = True
    for arm, state_path in artifact.get("bank_state_paths", {}).items():
        scheduler = "fixed" if arm == "fixed_2" else arm
        bank = Bank(Path(state_path), grammar(), scheduler, 0.1, 2)
        replay &= bank.replay_ledger()["state_hash"] == bank.state_hash
        for row in (item for item in rows if item["arm"] == arm):
            prediction = bank.state["predictions"].get(row["unit_id"])
            feedback = bank.state["feedback"].get(row["unit_id"])
            bank_rows_valid &= bool(
                prediction
                and feedback
                and prediction["probability"] == row["raw_metrics"]["probability"]
                and prediction["exact_status"] == row["raw_metrics"]["exact_status"]
                and feedback["label"] == row["raw_metrics"]["label"]
            )
    bank_rows_valid &= len(artifact.get("bank_state_paths", {})) == len(arms)
    return {
        "valid": complete
        and arithmetic
        and replay
        and source_valid
        and closure_valid
        and bank_rows_valid,
        "complete": complete,
        "arithmetic": arithmetic,
        "restart_replay": replay,
        "source_valid": source_valid,
        "closure_valid": closure_valid,
        "bank_rows_valid": bank_rows_valid,
        "rows": len(rows),
        "independent_families": len({row["unit_id"] for row in rows}),
    }


def build_artifact(
    date: str,
    started: float,
    checks: list[dict[str, Any]],
    hashes: dict[str, Any],
    measured: dict[str, Any],
    spans: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep fixture qualification distinct from empirical predictive benefit."""
    rows = measured.get("rows", [])
    reach = measured.get("commit_reachability_rows", [])
    committed = any(
        row["decision"] == "commit" and row["later_forecast_delta"] > 0 for row in reach
    )
    rejected = any(
        row["decision"] == "rollback" and row["later_forecast_delta"] == 0 for row in reach
    )
    closure = measured.get("static_closure_receipt", {})
    coverage = len({row["unit_id"] for row in rows}) == 96 and len(rows) == 96 * len(ARMS)
    mechanics = all(
        (
            committed,
            rejected,
            closure.get("different_from_empty_bank") is True,
            len(closure.get("features", [])) == 36,
            measured.get("budget_valid") is True,
            measured.get("restart_exact_parity") is True,
            measured.get("v671_rollback_reproduced") is True,
            bool(measured.get("feedback_diagnostics"))
            and all(measured["feedback_diagnostics"].values()),
            coverage,
        )
    )
    blocked = any(not row["passed"] for row in checks)
    score = int(not blocked and mechanics)
    verdict = (
        "complete_blocked_current_input"
        if blocked
        else "complete_circular_positive_acquisition_qualification"
        if score
        else "complete_null_acquisition_unqualified"
    )
    verdict_class = "blocked" if blocked else "circular_positive" if score else "null"
    gates = [
        {
            "gate": "validity",
            "passed": not blocked and mechanics,
            "operands": {
                "preconditions_passed": all(row["passed"] for row in checks),
                "causal_commit": committed,
                "harmful_rejection": rejected,
                "restart_parity": measured.get("restart_exact_parity"),
                "budget_valid": measured.get("budget_valid"),
            },
        },
        {
            "gate": "readiness",
            "passed": None,
            "operands": {
                "administrative_readiness": None,
                "acquisition_protocol_ready_score": score,
                "empirical_learning": None,
            },
        },
        {
            "gate": "probability",
            "passed": None,
            "operands": {"fixture_oracle": True, "natural_brier_delta": None},
        },
        {"gate": "utility", "passed": None, "operands": {"natural_decision_cost_delta": None}},
        {
            "gate": "coverage",
            "passed": coverage,
            "operands": {
                "observed_families": len({row["unit_id"] for row in rows}),
                "intended_families": 96,
                "arms": len(ARMS),
            },
        },
        {
            "gate": "source_dependence",
            "passed": None,
            "operands": {
                "original_source_bytes_retained": bool(rows),
                "natural_source_intervention": None,
            },
        },
        {
            "gate": "retention",
            "passed": measured.get("restart_exact_parity") if rows else None,
            "operands": {
                "retention_families": 32 if rows else 0,
                "restart_exact_parity": measured.get("restart_exact_parity"),
            },
        },
        {
            "gate": "efficiency",
            "passed": None,
            "operands": {"cpu_fixture_rows": len(rows), "native_speedup": None},
        },
    ]
    return {
        "schema": "carnot.exp7719.v672.acquisition_qualification.v1",
        "experiment_id": 7719,
        "milestone": "2026.09.672",
        "run_date": date,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": [row for row in checks if not row["passed"]],
        "acceptance_gate_results": gates,
        "rows": rows,
        "arms": measured.get("arms", []),
        "sample_size_budget": {
            "intended_families": {"development": 32, "admission": 32, "retention": 32},
            "observed_families": len({row["unit_id"] for row in rows}),
            "eligible_families": len({row["unit_id"] for row in rows}),
            "excluded_families": 0,
            "censored_families": 0,
            "effective_blocks": len({row["unit_id"] for row in rows}),
            "exposure": "deterministic_fixture_oracle",
            "arms_are_not_independent": True,
        },
        "inference_substrate": "cpu_deterministic_fixture_constraint_bank",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": MODEL_SPECS,
        "planned_MODEL_SPECS": [],
        "model_specs": [{"declaration": "no_model_invoked", "model_id": None}],
        "model_invoked": False,
        "invocation_counts": {
            "model_loads": 0,
            "forwards": 0,
            "generations": 0,
            "tokens": 0,
            "failures": 0,
            "cancellations": 0,
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "host": platform.node(),
            "pid": os.getpid(),
            "gpu_uuid": None,
            "effective_backend": "local_python_cpu",
            "requested_agent_backend": os.getenv("CODEX_FORCE_EXPERIMENTS"),
            "agent_runtime_verified": False,
        },
        "phase_spans": spans,
        "duration_s": time.monotonic() - started,
        "random_seed": {
            "fixture": 7719,
            "purpose": "deterministic family construction; no stochastic model fit",
        },
        "reproducibility_checksum": canonical_hash(
            {
                "source_artifact_hashes": hashes,
                "fixture_checksum": measured.get("fixture_checksum"),
                "capability_code": sha256_file(ROOT / CAPABILITY),
                "reducer_code": sha256_file(ROOT / ORCHESTRATION),
            }
        ),
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": {
            "frozen_affected_scope": SCOPE,
            "affected": [],
            "full_suite": [],
            "e2e": [],
            "terminal_readers": [],
            "global_debt": [],
        },
        "verifier_is_oracle": True,
        "field_principles": {
            **{name: PRINCIPLE for name in FIELDS},
            **{f"acceptance_gate_{name}": PRINCIPLE for name in GATES},
        },
        "acquisition_protocol_ready_score": score,
        "commit_reachability_rows": reach,
        "static_closure_receipt": closure,
        "fixture_checksum": measured.get("fixture_checksum"),
        "budget_accounting": measured.get("budget_accounting", {}),
        "bank_state_paths": measured.get("bank_state_paths", {}),
        "restart_exact_parity": measured.get("restart_exact_parity"),
        "v671_rollback_reproduced": measured.get("v671_rollback_reproduced"),
        "feedback_diagnostics": measured.get("feedback_diagnostics", {}),
        "prior_failures": [
            {
                "experiment_id": "exp7705-constraint-bank-protocol",
                "honest_verdict": "complete_circular_positive_fixture_mechanics_no_acquisition",
                "readiness_score": 0,
                "scope": "V671 bank fixture",
            },
            {
                "experiment_id": "exp7662-delayed-update-protocol",
                "honest_verdict": "complete_null_delayed_update_no_fresh_benefit",
                "scope": "natural delayed learning",
            },
        ],
        "same_verdict_retirements": [
            {
                "mechanism": "V671 fixture acquisition schedule",
                "status": "replaced by new five-case qualification; no empirical retirement inferred",
            }
        ],
    }


def span(
    started: float, phase: str, begin: float, units: int, checkpoint: str | None = None
) -> dict[str, Any]:
    """Record nonoverlapping monotonic phase bounds and a real heartbeat."""
    end = time.monotonic() - started
    return {
        "phase": phase,
        "start_s": begin,
        "end_s": end,
        "duration_s": end - begin,
        "completed_units": units,
        "heartbeat_timestamps": [time.time()],
        "checkpoint": checkpoint,
    }


def run_experiment(root: Path, date: str, output: Path, *, validate: bool = True) -> dict[str, Any]:
    """Freeze inputs, qualify mechanics, validate and atomically publish."""
    root = root.resolve()
    output = output.resolve()
    started = time.monotonic()
    spans: list[dict[str, Any]] = []
    progress(started, "preflight", "start")
    begin = time.monotonic() - started
    checks, hashes = preflight(root)
    spans.append(span(started, "preflight", begin, len(checks)))
    progress(started, "preflight", "complete", len(checks))
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    affected_files = [
        CAPABILITY,
        ORCHESTRATION,
        WRAPPER,
        TEST,
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/continuous-learning/spec.md",
    ]
    frozen_scope = {
        **SCOPE,
        "frozen_at_epoch_s": time.time(),
        "source_hashes": hashes,
        "affected_file_sha256": {name: sha256_file(ROOT / name) for name in affected_files},
    }
    atomic_json(raw / "frozen_affected_scope.json", frozen_scope)
    measured: dict[str, Any] = {}
    if all(row["passed"] for row in checks):
        progress(started, "measurement", "start")
        begin = time.monotonic() - started
        measured = run_fixture(
            raw, lambda phase, event, units: progress(started, phase, event, units)
        )
        atomic_json(raw / "raw_reduction_input.json", measured)
        spans.append(
            span(
                started,
                "measurement",
                begin,
                len(measured["rows"]),
                str(RAW / "measurement_checkpoint.json"),
            )
        )
        progress(started, "measurement", "complete", len(measured["rows"]))
    else:
        atomic_json(
            raw / "measurement_checkpoint.json",
            {"completed_units": 0, "blocked_checks": [row for row in checks if not row["passed"]]},
        )
    artifact = build_artifact(date, started, checks, hashes, measured, spans)
    artifact["validation_receipts"]["frozen_affected_scope"] = frozen_scope
    if not validate:
        atomic_json(output, artifact)
        return artifact

    with tempfile.TemporaryDirectory(prefix="exp7719-validation-") as private_dir:
        private = Path(private_dir)
        (private / "basetemp").mkdir()
        commands = build_scoped_commands(
            root,
            [TEST],
            [CAPABILITY, ORCHESTRATION],
            static_paths=[WRAPPER],
            basetemp=private / "basetemp",
            coverage_file=private / ".coverage",
        )
        progress(started, "affected_validation", "before_subprocess")
        begin = time.monotonic() - started
        affected = run_commands(
            root,
            commands,
            log_dir=raw / "validation/affected",
            extra_env={"JAX_PLATFORMS": "cpu", "COVERAGE_FILE": str(private / ".coverage")},
            heartbeat_s=30,
        )
        artifact["validation_receipts"]["affected"] = affected
        spans.append(span(started, "affected_validation", begin, len(affected)))
        progress(started, "affected_validation", "after_subprocess", len(affected))

        python = str(root / ".venv/bin/python")
        suite = [
            CommandSpec(
                "full_python_suite",
                (
                    str(root / ".venv/bin/pytest"),
                    "tests/python",
                    "-q",
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    f"--basetemp={private / 'full'}",
                ),
                "repository_health",
                timeout_s=1800,
            )
        ]
        progress(started, "full_python_suite", "before_subprocess")
        begin = time.monotonic() - started
        full = run_commands(
            root,
            suite,
            log_dir=raw / "validation/full",
            extra_env={"JAX_PLATFORMS": "cpu", "COVERAGE_FILE": str(private / ".coverage")},
            heartbeat_s=30,
        )
        artifact["validation_receipts"]["full_suite"] = full
        if not all(item["passed"] for item in full):
            artifact["validation_receipts"]["global_debt"] = full
        spans.append(span(started, "full_python_suite", begin, len(full)))
        progress(started, "full_python_suite", "after_subprocess", len(full))

        task_e2e = [
            CommandSpec(
                "task_e2e",
                (
                    str(root / ".venv/bin/pytest"),
                    TEST,
                    "-q",
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    f"--basetemp={private / 'e2e'}",
                ),
                "task_e2e",
            ),
            CommandSpec(
                "hard_exit_replay",
                (
                    python,
                    "-c",
                    "import os,sys; from pathlib import Path; from carnot.reporting.constraint_bank_protocol import Bank,grammar; Bank(Path(sys.argv[1]),grammar(),'priority',.1,2); os._exit(0)",
                    measured.get("bank_state_paths", {}).get("priority", str(raw / "missing.json")),
                ),
                "task_e2e",
            ),
        ]
        progress(started, "task_e2e", "before_subprocess")
        begin = time.monotonic() - started
        e2e = run_commands(
            root,
            task_e2e,
            log_dir=raw / "validation/e2e",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        artifact["validation_receipts"]["e2e"] = e2e
        spans.append(span(started, "task_e2e", begin, len(e2e)))
        progress(started, "task_e2e", "after_subprocess", len(e2e))

        candidate = raw / "terminal_candidate.json"
        artifact["phase_spans"] = spans
        artifact["duration_s"] = time.monotonic() - started
        atomic_json(candidate, artifact)
        readers = [
            CommandSpec(
                "cold_reduce",
                (
                    python,
                    "-u",
                    "-m",
                    "carnot.experiment_7719_v672_acquisition_qualification",
                    "--cold-reduce",
                    str(candidate),
                ),
                "exact_candidate",
            ),
            CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "exact_candidate",
            ),
            CommandSpec(
                "verdict_row_consistency",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "exact_candidate",
            ),
        ]
        progress(started, "terminal_readers", "before_subprocess")
        begin = time.monotonic() - started
        terminal = run_commands(
            root,
            readers,
            log_dir=raw / "validation/terminal",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        artifact["validation_receipts"]["terminal_readers"] = terminal
        artifact["validation_receipts"]["exact_terminal_candidate_sha256"] = sha256_file(candidate)
        spans.append(span(started, "terminal_readers", begin, len(terminal)))
        progress(started, "terminal_readers", "after_subprocess", len(terminal))

    failed = [
        item
        for group in (
            artifact["validation_receipts"]["affected"],
            artifact["validation_receipts"]["full_suite"],
            artifact["validation_receipts"]["e2e"],
            artifact["validation_receipts"]["terminal_readers"],
        )
        for item in group
        if not item["passed"]
    ]
    if failed:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_required_checks"
        artifact["acquisition_protocol_ready_score"] = 0
        artifact["flagged_adversarial"] = any(
            item["name"] == "adversarial_verify" for item in failed
        )
        artifact["gate_check_summary"].extend(
            check(
                "required_validation",
                "current",
                item["log_path"],
                "exit_code",
                0,
                item["exit_code"],
            )
            for item in failed
        )
        for gate in artifact["acceptance_gate_results"]:
            if gate["gate"] in {"validity", "readiness"}:
                gate["passed"] = False
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - started
    progress(started, "publish", "before_atomic_write")
    atomic_json(output, artifact)
    progress(started, "publish", "complete")
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Run the declared experiment or cold-reduce an exact candidate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260926")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--cold-reduce", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce:
        result = cold_reduce(args.cold_reduce)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["valid"] else 1
    run_experiment(ROOT.resolve(), args.date, ROOT / args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI process boundary.
    raise SystemExit(main())

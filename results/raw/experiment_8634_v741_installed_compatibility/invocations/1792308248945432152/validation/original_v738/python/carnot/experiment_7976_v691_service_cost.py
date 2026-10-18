"""REQ-REPORT-7976: publish independently qualified complete CPU service cost.

Private fixtures and cold replay use the same callable service. External
absence remains blocked; measured null timing does not assert science benefit.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot import experiment_7972_v691_qwen_energy_calibration as prior
from carnot.reporting import service_cost_7976 as s
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt

Json = dict[str, Any]
ROOT = s.ROOT
NAME = "experiment_7976_v691_service_cost"
TASK = "exp7976-service-cost"
MODEL_SPECS: list[str] = []
OWNED = [
    f"python/carnot/{NAME}.py",
    f"scripts/experiments/{NAME}.py",
    "python/carnot/reporting/service_cost_7976.py",
]
TESTS = [
    "tests/python/test_service_cost_7976.py",
    f"tests/python/test_{NAME}.py",
    "tests/python/test_source_boundary_7852.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_qwen_energy_calibration_7972.py",
]
INCLUDE = ",".join(str(ROOT / p) for p in OWNED)


def base(plan: Json) -> Json:
    """Declare every boundary even when external evidence cannot qualify."""
    value: Json = dict(
        experiment_id=7976,
        task_id=TASK,
        milestone="2026.10.691",
        schema="carnot.exp7976.service_cost.v1",
        run_date="20261001",
        execution_date="20261001",
        started_at=None,
        finished_at=None,
        duration_s=0.0,
        phase_spans=[],
        random_seed=69176,
        honest_verdict="complete_blocked_no_qualified_service",
        verdict_class="blocked",
        flagged_adversarial=False,
        gate_check_summary=[],
        service_measurement_ready_score=0,
        branch_readiness=plan.get("branch_readiness", {}),
        branch_gate_check_summary=plan.get("branch_gate_check_summary", {}),
        acceptance_gate_results=dict(
            validity=True,
            readiness=False,
            calibration=None,
            decision_benefit=None,
            retention=None,
            efficiency="descriptive_only",
        ),
        rows=[],
        service_rows=[],
        exclusive_phase_spans=[],
        original_model_cost_rows=[],
        cached_incremental_cost=None,
        complete_service_cost=None,
        durable_costs={k: None for k in ("lookup", "update", "commit_fsync", "restart")},
        sample_size_budget=dict(
            unit="intended_evaluation_family",
            intended=0,
            eligible=0,
            started=0,
            completed=0,
            failed=0,
            censored=0,
            excluded=0,
            independent=0,
        ),
        hardware_compatible_operations=[],
        compatible_fraction=None,
        transfer_bytes=None,
        modeled_100x_bound=None,
        ideal_amdahl_bound=None,
        source_artifact_hashes=plan.get("source_artifact_hashes", []),
        cited_upstream_artifacts=[],
        preconditions_checked=plan.get("branch_gate_check_summary", {}),
        resolved_imports={},
        validation_receipts=[],
        validation_command_manifest_path=None,
        observed_child_commands=[],
        coverage_statement_counts={},
        historical_required_failures=[],
        repository_health={},
        primary_resolution_receipt=None,
        terminal_validation_sidecar_path=None,
        scratch_root_receipt={},
        verifier_is_oracle=False,
        claim_scope="exposed_development",
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        target_model=None,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        field_principles={},
        reproducibility_checksum=None,
        oracle_distinct_corrigendum="September 28 preserved: human annotations are model-independent but fallible; GAP-ORACLE-DISTINCT remains open.",
        methodology="One complete-request warmup, ten hash-alternated paired repetitions on identical serialized requests and Gibbs heads. Repeats estimate timing variability. Historical per-row model acquisition is separate from cached incremental CPU scoring. No fresh inference speedup.",
        generator_weights_changed=False,
        production_defaults_changed=False,
        host_identity=dict(
            node=platform.node(), platform=platform.platform(), processor=platform.processor()
        ),
        prior_verdicts_unchanged={
            k: v.get("honest_verdict") for k, v in plan.get("upstream", {}).items()
        },
    )
    value["gate_check_summary"] = [
        r
        for checks in value["branch_gate_check_summary"].values()
        for r in checks
        if not r["passed"]
    ]
    return value


def apply_checks(value: Json, receipts: list[Json]) -> None:
    """Keep full-suite health separate while every owned failure zeros readiness."""
    value["validation_receipts"] = receipts
    value["observed_child_commands"] = [r.get("command_argv", []) for r in receipts]
    health = [r for r in receipts if not r.get("required", True)]
    value["repository_health"] = dict(
        current=health,
        status="degraded_open" if any(not r["passed"] for r in health) else "healthy",
    )
    if any(not r["passed"] for r in receipts if r.get("required", True)):
        value.update(
            honest_verdict="complete_disqualified_required_validation",
            verdict_class="disqualified",
            service_measurement_ready_score=0,
        )
        value["acceptance_gate_results"].update(validity=False, readiness=False)


def terminal_check(candidate: Path) -> Json:
    """Inspect the exact candidate with cold accounting and both terminal tools."""
    s.replay(json.loads(candidate.read_text()))
    receipts = run_commands(
        ROOT,
        [
            CommandSpec(
                name,
                (str(ROOT / ".venv/bin/python"), "-u", script, flag, str(candidate)),
                "terminal_candidate",
                60,
            )
            for name, script, flag in [
                ("adversarial", "scripts/adversarial_verify.py", "--json"),
                ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
            ]
        ],
        log_dir=candidate.parent / "terminal_logs",
        heartbeat_s=10,
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def publish(output: Path, value: Json) -> None:
    """Bind the terminal sidecars and both readers to one checked primary."""
    raw = output.parent / "raw" / output.stem
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["primary_resolution_receipt"] = dict(path=str(raw / "primary_resolution.json"))
    value["field_principles"] = {
        k: "Bind identity, exact operands or measured boundaries; readiness is validity, not improvement."
        for k in value
    }
    value["field_principles"].update(
        compatible_fraction="Only concrete quadratic Ising operations qualify; nonlinear Gibbs tanh and sigmoid operations are incompatible, so f=0.",
        modeled_100x_bound="Modeled 1/((1-f)+f/100+transfer_fraction); no hardware execution. Missing operands stay null.",
        ideal_amdahl_bound="Modeled 1/(1-f); f=1 denotes an unbounded ideal gain and is represented by null.",
        complete_service_cost="Add authenticated per-row model acquisition to current CPU requests. Setup receipts and unmeasured download cost remain explicit.",
        durable_costs="Unknown without a qualified durable learner; static response fsync is not learning or retention.",
        sample_size_budget="Source clusters determine independent evidence; repeats are timing observations only.",
    )
    receipt = publish_primary(output, value, terminal_check)
    atomic_json(raw / "terminal_validation.json", receipt)
    atomic_json(
        raw / "newer_nested_sidecar.json", dict(task_id=TASK, service_measurement_ready_score=99)
    )
    readers = reader_receipt(
        TASK,
        output.parent,
        field="service_measurement_ready_score",
        expected=value["service_measurement_ready_score"],
    )
    if not readers["passed"]:
        raise ValueError("primary_resolution")
    atomic_json(raw / "primary_resolution.json", readers)


def freeze_commands(raw: Path, scratch: Path) -> Json:
    """Freeze scenario destinations and identical coverage includes before results."""
    scratch.mkdir(parents=True, exist_ok=True)
    inputs, heads = s.fixture()
    fixture = scratch / "input.json"
    atomic_json(fixture, dict(requests=inputs, heads=heads))
    atomic_json(scratch / "request.json", inputs[0])
    (scratch / "heads.json").write_text(json.dumps(heads))
    cov, py = str(ROOT / ".venv/bin/coverage"), str(ROOT / ".venv/bin/python")
    cli = str(ROOT / OWNED[1])
    covered = [
        cov,
        "run",
        "--parallel-mode",
        f"--data-file={scratch}/.coverage",
        f"--include={INCLUDE}",
    ]
    success = scratch / "success" / f"{NAME}.json"
    commands = []

    def add(
        name: str,
        argv: list[str],
        expected: int = 0,
        reason: str | None = None,
        *,
        required: bool = True,
    ) -> None:
        commands.append(
            dict(
                name=name,
                argv=argv,
                expected_exit=expected,
                reason=reason,
                required=required,
                deadline_s=300,
            )
        )

    add(
        "unit_consumers_e2e015",
        [
            *covered,
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "-q",
            *TESTS,
            f"--basetemp={scratch}/pytest",
        ],
    )
    add("fixture_cli", [*covered, cli, "--fixture-e2e", str(fixture), "--output", str(success)])
    add("cold_replay_cli", [*covered, cli, "--cold-replay", str(success)], reason="replay_passed")
    add("terminal_recheck_cli", [*covered, cli, "--terminal-recheck", str(success)])
    add(
        "private_request_cli",
        [
            *covered,
            cli,
            "--request",
            str(scratch / "request.json"),
            "--heads",
            str(scratch / "heads.json"),
            "--output",
            str(scratch / "request-reply.json"),
        ],
    )
    add(
        "blocked_cli",
        [
            *covered,
            cli,
            "--data-root",
            str(scratch / "absent"),
            "--skip-validation",
            "--output",
            str(scratch / "blocked" / f"{NAME}.json"),
        ],
    )
    add("invalid_date_cli", [*covered, cli, "--date", "20260930"], 1, "run_date")
    add(
        "invalid_input_cli",
        [*covered, cli, "--fixture-e2e", str(scratch / "missing")],
        1,
        "rejected=",
    )
    add(
        "private_live_cli",
        [*covered, cli, "--skip-validation", "--output", str(scratch / "live" / f"{NAME}.json")],
    )
    add("coverage_combine", [cov, "combine", f"--data-file={scratch}/.coverage", str(scratch)])
    add(
        "coverage_json",
        [
            cov,
            "json",
            f"--data-file={scratch}/.coverage",
            f"--include={INCLUDE}",
            "-o",
            str(scratch / "coverage.json"),
        ],
    )
    add(
        "coverage_report",
        [
            cov,
            "report",
            f"--data-file={scratch}/.coverage",
            f"--include={INCLUDE}",
            "--fail-under=100",
            "--show-missing",
        ],
    )
    add("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *OWNED, *TESTS[:2]])
    add("ruff_format", [str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, *TESTS[:2]])
    add(
        "strict_mypy",
        [str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED[:1], OWNED[2]],
    )
    add("spec_coverage", [py, "scripts/check_spec_coverage.py", *TESTS])
    add(
        "repository_health_full_suite",
        [
            str(ROOT / ".venv/bin/pytest"),
            "tests/python",
            "-q",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            f"--basetemp={scratch}/full-suite",
        ],
        required=False,
    )
    libraries = [
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/experiment_7303_validation_scope.py",
        "python/carnot/experiment_7972_v691_qwen_energy_calibration.py",
        "python/carnot/verify/qwen_energy_calibration_7972.py",
        "scripts/conductor_gates.py",
        "scripts/in_process_doc_reconcile.py",
    ]
    manifest = dict(
        commands=commands,
        coverage_includes=INCLUDE,
        affected_files=OWNED,
        transitive_consumers=libraries,
        test_paths=TESTS,
        scratch_root=str(scratch),
        code_config_hashes=[prior.reference(ROOT / p) for p in OWNED + libraries + TESTS],
        config=dict(
            repetitions=10, warmup="complete_request_once", family_limit=64, relative_tolerance=0.01
        ),
    )
    atomic_json(raw / "validation_commands.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Run current measurement or a private replay without changing generators."""
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    print("[exp7976] start flushed=true", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261001")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / f"{NAME}.json")
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--terminal-recheck", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--heads", type=Path)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.date != "20261001":
            raise ValueError("run_date")
        if args.cold_replay or args.terminal_recheck:
            path = args.cold_replay or args.terminal_recheck
            s.replay(json.loads(path.read_text()))
            if args.terminal_recheck and not terminal_check(path)["passed"]:
                raise ValueError("terminal_recheck")
            print("[exp7976] replay_passed", flush=True)
            return 0
        if args.request:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            print("[exp7976] before_benchmark private_request", flush=True)
            s.request(args.request, args.heads, args.output, "exclusive")
            print("[exp7976] after_benchmark private_request", flush=True)
            return 0
        output = args.output.absolute()
        raw = output.parent / "raw" / output.stem
        print("[exp7976] phase=authenticate", flush=True)
        if args.fixture_e2e:
            fixture = json.loads(args.fixture_e2e.read_text())
            plan = dict(
                requests=fixture["requests"],
                heads=fixture["heads"],
                branch_readiness=dict(qwen_calibration=1),
                branch_gate_check_summary={},
            )
        else:
            plan = s.authenticate(args.data_root)
        value = base(plan)
        value["started_at"] = started_at
        with TemporaryDirectory(prefix="carnot-7976-", dir="/tmp") as folder:
            scratch = Path(folder)
            value["scratch_root_receipt"] = dict(
                path=folder, outside_checkout=True, removed_after_exit=True
            )
            manifest = freeze_commands(raw, scratch)
            value["validation_command_manifest_path"] = str(raw / "validation_commands.json")
            value["code_config_hashes"] = manifest["code_config_hashes"]
            value["resolved_imports"] = {name: str(ROOT / name) for name in OWNED}
            atomic_json(
                raw / "input_checkpoint.json", dict(requests=plan["requests"], heads=plan["heads"])
            )
            value["input_checkpoint"] = prior.reference(raw / "input_checkpoint.json")
            value["reproducibility_checksum"] = s.canonical_hash(
                dict(
                    code=manifest["code_config_hashes"],
                    input=value["input_checkpoint"],
                    date=args.date,
                )
            )
            if plan["requests"]:
                print("[exp7976] phase=measure_private_cpu", flush=True)
                value.update(s.measure(plan["requests"], plan["heads"], scratch / "service"))
                value.update(
                    honest_verdict="complete_null_service_cost",
                    verdict_class="null",
                    service_measurement_ready_score=1,
                    trained_head_specs=plan["heads"],
                    original_model_cost_rows=plan.get("original_model_cost_rows", []),
                    acquisition_setup=plan.get("acquisition_setup", {}),
                )
                value["acceptance_gate_results"].update(readiness=True)
                if args.fixture_e2e:
                    value.update(
                        honest_verdict="complete_circular_positive_service_fixture",
                        verdict_class="circular_positive",
                    )
            value["cited_upstream_artifacts"] = [
                dict(
                    r,
                    imported_fields=[
                        "readiness",
                        "verdict_class",
                        "rows",
                        "calibrator_checkpoints",
                    ],
                )
                for r in value["source_artifact_hashes"]
            ]
            upstream = plan.get("upstream", {}).get("qwen_calibration", {})
            value["historical_required_failures"] = upstream.get("historical_required_failures", [])
            if not args.fixture_e2e and not args.skip_validation:
                print("[exp7976] phase=owned_validation", flush=True)
                receipts = prior.execute_commands(manifest, raw, scratch)
                apply_checks(value, receipts)
                coverage = json.loads((scratch / "coverage.json").read_text())
                value["coverage_statement_counts"] = {
                    k: v["summary"] for k, v in coverage["files"].items()
                }
                for path in scratch.glob(".coverage*"):
                    archive = raw / "coverage_data" / path.name
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, archive)
                shutil.copy2(scratch / "coverage.json", raw / "coverage.json")
            for reference in value["source_artifact_hashes"] + manifest["code_config_hashes"]:
                prior.checked_reference(reference)
            value["duration_s"] = time.monotonic() - started
            value["phase_spans"] = [
                dict(
                    phase="owned_cpu_measurement_and_validation",
                    start_s=0.0,
                    end_s=value["duration_s"],
                )
            ]
            value["finished_at"] = datetime.now(UTC).isoformat()
            print("[exp7976] phase=terminal_publication", flush=True)
            publish(output, value)
            print(
                f"[exp7976] published path={output} elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
            return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp7976] rejected={type(error).__name__}:{error}", flush=True)
        return 1

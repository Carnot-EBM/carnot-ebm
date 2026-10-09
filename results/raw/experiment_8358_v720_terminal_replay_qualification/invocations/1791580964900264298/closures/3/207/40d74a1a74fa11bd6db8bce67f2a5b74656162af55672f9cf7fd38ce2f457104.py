"""REQ-REPORT-8213: qualify an owned recorder without loading model weights.

Frozen validation commands and byte copies make the invocation auditable. The
prospective workload is designed research demand, with no deployment estimate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot.verify import request_recorder_8213 as e
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS
from carnot.reporting.request_trace_inventory_8200 import copy_bytes, operand
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, build_scoped_commands
from carnot.reporting.primary_publication import publish_primary
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
MODULES = [
    "python/carnot/verify/request_recorder_8213.py",
    "python/carnot/verify/recorder_fixtures_8213.py",
    "python/carnot/reporting/recorder_execution_8213.py",
]
TEST = "tests/python/test_prospective_request_recorder_8213.py"
UPSTREAM = "results/experiment_8182_v707_fit_sentence_capture.json"
PIN = "sha256:6f0f6f3dd5c1b067ab50040a3ce9bc3358a48dee2e71180cf91b3c335bdbcf19"


def checksum(value: Json) -> str:
    """Exclude the checksum itself so private candidates can be independently read."""
    return str(e.key({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def inputs(root: Path, raw: Path) -> Json:
    """Authenticate only roster and native join operands; missing siblings are optional."""
    raw.mkdir(parents=True, exist_ok=True)
    data: Json = dict(
        ready=False,
        checks=[],
        refs=[],
        roster=[],
        schedule={},
        identity={},
        precondition_receipts=[],
        optional_dispositions=[],
    )
    primary = root / UPSTREAM
    path = primary
    data["checks"].append(operand("fit_roster_exists", primary, True, primary.is_file()))
    try:
        if not primary.is_file():
            raise ValueError("missing_fit_roster")
        value = json.loads(primary.read_text())
        data["refs"].append(copy_bytes(primary, raw))
        data["checks"].append(operand("fit_primary_sha256", primary, PIN, e.sha256_file(primary)))
        for field, expected in [
            ("fit_capture_ready_score", 1),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            data["checks"].append(operand(field, primary, expected, value.get(field)))
        if not all(c["passed"] for c in data["checks"]):
            raise ValueError("failed_fit_gate")
        source = next(
            r for r in value["raw_shard_hashes"] if Path(r["path"]).name == "source_plan.json"
        )
        path = Path(source["path"])
        data["checks"].append(
            operand("source_plan_sha256", path, source["sha256"], e.sha256_file(path))
        )
        if not all(c["passed"] for c in data["checks"]):
            raise ValueError("failed_source_gate")
        data["refs"].append(copy_bytes(path, raw))
        roster = json.loads(path.read_text())["rows"]
        data["roster"] = roster
        data["identity"] = value["model_receipt"]["capture_identity"]
        service_path = root / "results/experiment_8174_v706_complete_request_cost.json"
        path = service_path
        service = json.loads(service_path.read_text())
        data["checks"].append(
            operand(
                "service_primary_sha256",
                service_path,
                "sha256:15a13284335442ff04a491c6683d825e2b4d96de3950e6aa602a7c0afa1a252f",
                e.sha256_file(service_path),
            )
        )
        if not all(c["passed"] for c in data["checks"]):
            raise ValueError("failed_service_gate")
        data["refs"].append(copy_bytes(service_path, raw))
        ref = next(
            r for r in service["raw_shard_hashes"] if Path(r["path"]).name == "input_data.json"
        )
        path = Path(ref["path"])
        data["checks"].append(
            operand("service_config_sha256", path, ref["sha256"], e.sha256_file(path))
        )
        if not all(c["passed"] for c in data["checks"]):
            raise ValueError("failed_config_gate")
        data["refs"].append(copy_bytes(path, raw))
        frozen = json.loads(path.read_text())
        data.update({k: frozen[k] for k in ["library", "head", "geometry"]})
        path = Path(data["library"]["path"])
        data["checks"].append(
            operand("native_library_sha256", path, data["library"]["sha256"], e.sha256_file(path))
        )
        if not all(c["passed"] for c in data["checks"]):
            raise ValueError("failed_library_gate")
        data["refs"].append(copy_bytes(path, raw))
        data["schedule"] = e.schedule(roster, data["identity"])
        data["ready"] = all(c["passed"] for c in data["checks"])
        if data["ready"]:
            data["schedule_ref"] = e.seal(raw / "schedule.json", data["schedule"])
    except (OSError, ValueError, KeyError, StopIteration, TypeError) as error:
        if all(c["passed"] for c in data["checks"]):
            data["checks"].append(
                operand(
                    "required_operand_structure",
                    path,
                    "authenticated roster/service",
                    str(error),
                )
            )
    optional = root / "results/experiment_8201_v708_replayed_request_service.json"
    data["optional_dispositions"].append(
        dict(
            path=str(optional),
            required=False,
            disposition="present_not_used" if optional.exists() else "absent_not_required",
        )
    )
    commands = [
        CommandSpec("operand_" + str(i), ("test", "-r", c["path"]), "preconditions", 5)
        for i, c in enumerate(data["checks"])
    ]
    private = Path(tempfile.mkdtemp(prefix="carnot-8213-preconditions-"))
    private.chmod(0o700)
    commands += [
        CommandSpec(
            "python_environment",
            (sys.executable, "-c", "import pytest,coverage,ruff,mypy,sys; print(sys.version)"),
            "preconditions",
            15,
        ),
        CommandSpec(
            "private_scratch",
            (
                sys.executable,
                "-c",
                'import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.write_bytes(b"writable"); print(p.read_bytes().decode())',
                str(private / "probe"),
            ),
            "preconditions",
            5,
        ),
    ]
    data["precondition_receipts"] = execute(commands, raw / "preflight")
    data["ready"] = data["ready"] and all(r["passed"] for r in data["precondition_receipts"])
    data["input_path"] = str((raw / "input.json").absolute())
    e.seal(raw / "input.json", data)
    return data


def execute(plan: list[CommandSpec], raw: Path) -> list[Json]:
    """Bound children and their descendants while preserving complete output hashes."""
    raw.mkdir(parents=True, exist_ok=True)
    receipts = []
    for i, spec in enumerate(plan):
        e.progress("8213_subprocess_before_" + spec.name, i, len(plan) - i)
        began = time.monotonic_ns()
        timed_out = False
        stdout, stderr = raw / (spec.name + ".stdout"), raw / (spec.name + ".stderr")
        with stdout.open("wb") as out, stderr.open("wb") as err:
            process = subprocess.Popen(
                spec.argv,
                cwd=e.ROOT,
                stdout=out,
                stderr=err,
                start_new_session=True,
                env=dict(os.environ, JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1"),
            )
            while process.poll() is None:
                try:
                    remaining = spec.timeout_s - (time.monotonic_ns() - began) / 1e9
                    process.wait(timeout=min(30, max(0.001, remaining)))
                except subprocess.TimeoutExpired:
                    e.progress("8213_child_heartbeat_" + spec.name, i, len(plan) - i)
                    if (time.monotonic_ns() - began) / 1e9 >= spec.timeout_s:
                        timed_out = True
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
        ended = time.monotonic_ns()
        receipts.append(
            dict(
                name=spec.name,
                command_argv=list(spec.argv),
                scope=spec.scope,
                expected_exit=0,
                actual_exit=process.returncode,
                exit_code=process.returncode,
                normal_exit=process.returncode >= 0 and not timed_out,
                timed_out=timed_out,
                passed=process.returncode == 0 and not timed_out,
                started_monotonic_ns=began,
                ended_monotonic_ns=ended,
                duration_s=(ended - began) / 1e9,
                stdout_path=str(stdout),
                stdout_sha256=e.sha256_file(stdout),
                stderr_path=str(stderr),
                stderr_sha256=e.sha256_file(stderr),
            )
        )
        stdout.chmod(0o444)
        stderr.chmod(0o444)
        e.progress("8213_subprocess_after_" + spec.name, i + 1, len(plan) - i - 1)
    return receipts


def validation_plan(private: Path) -> list[CommandSpec]:
    """Freeze owned statements including real CLI children; E2E uses private scratch."""
    plan = build_scoped_commands(
        e.ROOT,
        [TEST],
        MODULES,
        static_paths=[e.CLI],
        basetemp=private,
        coverage_file=private / ".coverage",
    )
    config = private / "coverage.ini"
    include = ",".join("*/" + p for p in MODULES + [e.CLI])
    config.write_text(
        "[run]\nparallel = true\npatch = subprocess\ninclude =\n"
        + "".join("    */" + p + "\n" for p in MODULES + [e.CLI])
    )
    result = []
    for spec in plan:
        argv = spec.argv
        if spec.name == "changed_module_coverage":
            argv = tuple(a for a in argv if not a.startswith("--include="))
            argv = (argv[0], argv[1], "--rcfile=" + str(config), *argv[2:])
        if spec.name == "changed_module_coverage_report":
            result.append(
                CommandSpec(
                    "coverage_combine",
                    (
                        str(e.ROOT / ".venv/bin/coverage"),
                        "combine",
                        "--data-file=" + str(private / ".coverage"),
                        str(private),
                    ),
                    "owned",
                    60,
                )
            )
            argv = tuple("--include=" + include if a.startswith("--include=") else a for a in argv)
        if spec.name == "changed_module_mypy":
            argv += ("--strict", "--follow-imports=silent")
        if spec.name == "scoped_spec_coverage":
            argv = (
                str(e.ROOT / ".venv/bin/python"),
                "scripts/check_spec_coverage.py",
                "--files",
                TEST,
            )
        result.append(CommandSpec(spec.name, argv, spec.scope, 180))
    for name, test in [
        ("e2e015", "tests/python/test_source_boundary_7852.py"),
        ("e2e019", "tests/python/test_experiment_7942_v689_sentence_labels.py"),
        ("affected_consumers", "tests/python/test_primary_publication_7928.py"),
    ]:
        result.append(
            CommandSpec(
                name,
                (
                    str(e.ROOT / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "--basetemp=" + str(private / name),
                    test,
                    "-q",
                ),
                "private_e2e",
                180,
            )
        )
    return result


def validators(path: Path) -> list[CommandSpec]:
    """Keep terminal auditors unchanged and run replay through the actual CLI."""
    py = str(e.ROOT / ".venv/bin/python")
    return [
        CommandSpec(
            "cold_replay",
            (py, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)),
            "terminal",
            60,
        ),
        CommandSpec(
            "adversarial",
            (py, str(e.ROOT / "scripts/adversarial_verify.py"), "--json", str(path)),
            "terminal",
            60,
        ),
        CommandSpec(
            "strict_rows",
            (py, str(e.ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)),
            "terminal",
            60,
        ),
    ]


def build(data: Json, work: Json, raw: Path, receipts: list[Json], duration: float) -> Json:
    """Readiness certifies recorder envelopes; synthetic truth remains circular."""
    passed = all(r["passed"] and r.get("normal_exit", True) for r in receipts)
    cases = work.get("fixture_rows", [])
    qualified = bool(cases) and all(c["passed"] for c in cases)
    ready = data["ready"] and qualified and passed
    cls = "circular_positive" if ready else "blocked"
    verdict = (
        "complete_circular_positive_prospective_recorder_qualified"
        if ready
        else "complete_blocked_required_operand"
    )
    if not passed or (data["ready"] and not qualified):
        cls, verdict = "disqualified", "complete_disqualified_owned_validation"
    schedule_rows = data["schedule"].get("rows", [])
    rows = [
        dict(
            source_cluster_id=r["source_cluster_id"],
            request_id=r["request_id"],
            seed=e.SEED,
            condition="original",
            status="completed",
            metric="scheduled_envelope_qualified",
            numerator=int(ready),
            denominator=1,
            model_call_performed=False,
        )
        for r in schedule_rows
    ]
    checks = [
        *data["checks"],
        dict(check="owned_validation", passed=passed),
        dict(check="scripted_boundary_and_join", passed=qualified if data["ready"] else False),
    ]
    refs = [e.reference(Path(data["input_path"]))]
    if data["ready"]:
        refs.append(data["schedule_ref"])
    if work:
        refs.extend(
            [
                e.reference(Path(work["work_path"])),
                work["journal"],
                *[r["store"] for r in work["service_join_rows"]],
            ]
        )
    code = [copy_bytes(e.ROOT / p, raw / "code") for p in MODULES + [e.CLI, TEST]]
    if work:
        configuration = raw / "exp8214_configuration.json"
        if not configuration.exists():
            e.seal(
                configuration,
                dict(
                    request_schema_version=e.SCHEMA,
                    code=code,
                    schedule=data["schedule_ref"],
                    service=work["service_configuration"],
                    prospective_identity=data["identity"],
                    max_tokens=128,
                    seed=e.SEED,
                    transport_attempts=1,
                    workload_scope="designed_research_requests",
                    deployment_demand_observed=False,
                ),
            )
        refs.append(e.reference(configuration))
    value = dict(
        experiment_id=8213,
        experiment=8213,
        task_id="exp8213-prospective-request-recorder",
        milestone="v709",
        run_date="20261006",
        schema="carnot.prospective_recorder_qualification.v1",
        honest_verdict=verdict,
        verdict_class=cls,
        status="completed",
        title="Prospective research request recorder qualification",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        call_ledger=[],
        preconditions_checked=True,
        precondition_receipts=data["precondition_receipts"],
        optional_dispositions=data["optional_dispositions"],
        duration_s=duration,
        random_seed=e.SEED,
        source_artifact_hashes=data["refs"],
        cited_upstream_artifacts=[
            dict(
                path=r["path"],
                sha256=r["sha256"],
                fields_imported=["original_source_roster_or_service_configuration"],
            )
            for r in data["refs"]
        ],
        rows=rows,
        intended_count=24,
        completed_count=len(rows),
        failed_count=0,
        censored_count=0,
        excluded_count=24 - len(rows),
        independent_count=len({r["source_cluster_id"] for r in rows}),
        verifier_is_oracle=True,
        exposure_scope="exposed fit development roster and synthetic HTTP fixtures",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        gate_check_summary=checks,
        acceptance_gates=dict(
            required_inputs=data["ready"],
            owned_validation=passed,
            fixture_boundary=qualified,
            immutable_schedule=len(rows) == 24,
        ),
        request_recorder_ready_score=int(ready),
        request_schema_version=e.SCHEMA,
        schedule_path=data.get("schedule_ref", {}).get("path"),
        schedule_sha256=data.get("schedule_ref", {}).get("sha256"),
        fixture_rows=cases,
        envelope_roundtrip_rows=work.get("envelope_roundtrip_rows", []),
        service_join_rows=work.get("service_join_rows", []),
        qualified_service_configuration=work.get("service_configuration", {}),
        workload_scope="designed_research_requests",
        deployment_demand_observed=False,
        frozen_exp8214_configuration=e.reference(raw / "exp8214_configuration.json")
        if work
        else None,
        scheduled_model_invocation_count=0,
        fixture_generation_counts=dict(intended=4, completed=2, failed=1, censored=1)
        if work
        else {},
        qualification_scope="schedule envelope preparation and scripted recorder boundary; no prospective Qwen call",
        validation_receipts=receipts,
        required_checks_passed=passed,
        raw_shard_hashes=refs,
        code_config_hashes=code,
        flagged_adversarial=False,
        input_path=data["input_path"],
        raw_path=str(raw),
        work_path=work.get("work_path"),
        work_sha256=e.sha256_file(Path(work["work_path"])) if work else None,
        benchmark_performed=False,
        sample_size_budget=dict(scheduled_sources=24, current_model_calls=0),
        methodology="Fixed source identity order; full original condition; one sentence per source; max_tokens128; one future transport attempt. Current qualification uses a scripted HTTP peer, durable issue and terminal events, real Python/Rust signatures and disjoint spans. Oracle fixtures certify plumbing only.",
        field_principles=dict(
            identity="Exact task, invocation, frozen source and code bytes.",
            request_recorder_ready_score="Complete envelope, scripted durable boundary, frozen schedule and owned validation.",
            rows="One qualified prospective envelope per original source; no model outcome implied.",
            inference_substrate="No model load; imported model identity is historical provenance.",
            workload_scope="Designed research demand; no deployment frequency estimate.",
            verdict="Owned failures disqualify; external missing operands block; oracle success is circular.",
            generalization="Exposed development and fixtures provide zero generalization credit.",
        ),
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def main(argv: list[str] | None = None) -> int:
    """Validate a private candidate before replacing the sole primary atomically."""
    began = time.monotonic()
    e.progress("8213_start", 0, 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261006"], default="20261006")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("8213_replay_after", int(passed), 0)
        return 0 if passed else 1
    output = (args.fixture_e2e or args.output).absolute()
    if args.fixture_e2e and output.is_relative_to(e.ROOT / "results"):
        parser.error("fixture output requires private scratch")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-8213-"))
    private.chmod(0o700)
    plan = validation_plan(private)
    candidate = private / (e.NAME + ".json")
    health = CommandSpec(
        "repository_health_once",
        (str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
        "unrelated_repository_health",
        1800,
    )
    e.seal(
        raw / "validation_commands.json",
        dict(
            commands=[asdict(c) for c in plan],
            terminal=[asdict(c) for c in validators(candidate)],
            repository_health=asdict(health),
            frozen_before_measurement=True,
            configuration=dict(seed=e.SEED, max_tokens=128, transport_attempts=1),
        ),
    )
    e.progress("8213_preconditions_before", 0, 1)
    data = inputs(args.root, raw)
    e.progress("8213_preconditions_after", int(data["ready"]), 0)
    receipts = (
        execute(plan[:1] if args.fixture_e2e else plan, raw / "validation") if data["ready"] else []
    )
    work: Json = {}
    if data["ready"] and all(r["passed"] for r in receipts):
        e.progress("8213_qualification_before", 0, 1)
        work = e.qualify(data, raw)
        e.progress("8213_qualification_after", 1, 0)
    value = build(data, work, raw, receipts, time.monotonic() - began)
    if not args.fixture_e2e:
        value["repository_health"] = execute([health], raw / "health")
    value["raw_shard_hashes"].append(e.reference(raw / "validation_commands.json"))
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = checksum(value)
    e.atomic_json(candidate, value)

    def validate(path: Path) -> Json:
        rows = execute(validators(path), raw / ("terminal-" + str(time.time_ns())))
        return dict(passed=all(r["passed"] for r in rows), receipts=rows)

    publication = publish_primary(output, value, validate)
    e.seal(raw / "terminal_validation.json", dict(publication=publication))
    e.progress("8213_published", 1, 0)
    return 0

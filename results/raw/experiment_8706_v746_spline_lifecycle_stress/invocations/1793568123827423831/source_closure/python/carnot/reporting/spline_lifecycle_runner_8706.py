"""REQ-VERIFY-8706: bounded real restart, immutable replay and atomic publication."""

from __future__ import annotations

import argparse
import configparser
from datetime import datetime, UTC
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, cast
from unittest.mock import patch

from carnot.reporting import spline_lifecycle_study_8706 as e
from carnot.reporting import v743_capstone as validation
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.gatemate_continuity_8400 import budget
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def measure(raw: Path, date: str, private: bool) -> Json:
    """Seal the same original vectors and labels before measuring any lifecycle arm."""
    start, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    raw.mkdir(parents=True, exist_ok=True)
    e.progress("preconditions_before")
    with tempfile.TemporaryDirectory(prefix="spline8706-", dir="/var/tmp") as directory:
        pre = resources(Path(directory))
        pre["software_versions"] = {
            name: version(name) for name in ["numpy", "scipy", "coverage", "pytest", "ruff", "mypy"]
        }
        pre["write_probe_passed"] = (
            Path(directory) / "write_probe"
        ).read_bytes() == b"private disk scratch"
    pre.update(
        optional_devices_required=False, language_model_calls=0, python=sys.version, workers=1
    )
    inputs = e.authenticate()
    failures = inputs["failures"].copy()
    contract: Json = {}
    try:
        contract = e.authority(e.ROOT)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        failures.append(
            dict(
                upstream="V746_contract",
                check="authority",
                path=str(e.ROOT / e.DESIGN),
                sha256=e.reference(e.ROOT / e.DESIGN)["sha256"],
                field="fourteen_full_objects",
                operator="==",
                expected=e.PIN,
                observed=str(error),
                operand_state="invalid_or_changed_authority",
                passed=False,
            )
        )
    if not (
        pre["disk_backed"]
        and pre["tools_available"]
        and pre["write_probe_passed"]
        and min(pre["free_disk_bytes"], pre["memory_available_bytes"]) >= 2**30
    ):
        failures.append(
            dict(
                upstream="host",
                check="cpu_resources",
                path="/var/tmp",
                sha256=None,
                field="disk_memory_tools",
                operator="==",
                expected=True,
                observed=pre,
                operand_state="resource_unavailable",
                passed=False,
            )
        )
    dependencies = [
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        e.DESIGN,
        "research-roadmap.yaml",
        "ops/exclusion_manifest.yaml",
        "ops/e2e-test-plan.md",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/verification/spec.md",
        "research-references.md",
        "research-hardware-wishlist.md",
        "pyproject.toml",
        "scripts/experiment_template.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "scripts/check_spec_coverage.py",
        "tests/python/conftest.py",
        "docs/research-notes/v746-spline-lifecycle-stress.md",
    ]
    dependencies += [
        str(Path(cast(str, m.__file__)).resolve().relative_to(e.ROOT))
        for m in list(sys.modules.values())
        if getattr(m, "__file__", None)
        and Path(cast(str, m.__file__)).suffix == ".py"
        and Path(cast(str, m.__file__)).is_file()
        and Path(cast(str, m.__file__)).resolve().is_relative_to(e.ROOT)
        and ".venv" not in Path(cast(str, m.__file__)).parts
    ]
    code_refs = [e.reference(e.ROOT / p) for p in e.OWNED + e.TESTS]
    source_refs = inputs["refs"].copy()
    for name in sorted(set(dependencies + e.OWNED + e.TESTS)):
        source = e.ROOT / name
        target = raw / "source_closure" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_refs.extend([e.reference(source), e.reference(target)])
    atomic_json(raw / "frozen.json", inputs["frozen"])
    source_refs.append(e.reference(raw / "frozen.json"))
    jobs: list[Json] = []
    if not failures:
        heads = inputs["heads"][:1] if private else inputs["heads"]
        for hi, head in enumerate(heads):
            for sequence, labels in sorted(inputs["frozen"]["sequences"].items())[
                : 1 if private else 4
            ]:
                jobs.append(
                    dict(
                        head=head,
                        head_index=hi,
                        sequence=sequence,
                        labels=labels,
                        vectors=inputs["frozen"]["vectors"],
                        boundary=False,
                        updates=8 if private else 4096,
                    )
                )
        for hi, head in enumerate(heads):
            jobs.append(
                dict(
                    head=head,
                    head_index=hi,
                    sequence="boundary",
                    labels=inputs["frozen"]["sequences"]["alternating"],
                    vectors=inputs["frozen"]["vectors"],
                    boundary=True,
                    boundary_offset=hi - 2,
                    updates=4 if private else 64,
                )
            )
    job_refs = []
    for index, job in enumerate(jobs):
        path = raw / "jobs" / f"{index:02}.json"
        atomic_json(path, job)
        job_refs.append(e.reference(path))
    source_refs += job_refs
    atomic_json(
        raw / "premeasurement_seal.json",
        dict(
            source_refs=source_refs,
            code_refs=code_refs,
            job_refs=job_refs,
            configuration=dict(
                format=[32, 24],
                rule=e.numerical.numeric.original.RULE,
                seed=e.numerical.numeric.SEED,
                checkpoint_every=256,
                restart_at=2048,
                private=private,
            ),
        ),
    )
    summaries, restarts, commands, restart_costs = [], [], [], []
    e.progress("sources_inputs_frozen_before_benchmark", 0, len(jobs))
    for index, (job, ref) in enumerate(zip(jobs, job_refs, strict=True)):
        remaining = 3600 - (time.monotonic_ns() - start) / 1e9
        if remaining <= 0:
            raise TimeoutError("measurement_deadline")
        folder = raw / "workers" / f"{index:02}-uninterrupted"
        base = [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--worker",
            ref["path"],
            "--work-dir",
            str(folder),
            "--stop",
            str(job["updates"]),
        ]
        command = child(
            f"full_{index}", base + ["--reference"], raw / "worker_logs", deadline=remaining
        )
        commands.append(command)
        if not command["passed"]:
            raise ValueError("failed_or_killed_measurement_child")
        summary = e.read(e.reference(folder / f"summary-0000-{job['updates']:04}.json"))
        summary.update(
            head_index=job["head_index"],
            sequence=job["sequence"],
            boundary=job["boundary"],
            job_reference=ref,
            rows_reference=e.reference(Path(summary["rows_path"])),
        )
        summaries.append(summary)
        if not job["boundary"]:
            split = job["updates"] // 2
            restarted = raw / "workers" / f"{index:02}-restarted"
            restart_argv = [
                sys.executable,
                "-u",
                str(e.ROOT / e.CLI),
                "--worker",
                ref["path"],
                "--work-dir",
                str(restarted),
            ]
            for phase, begin, end, expected in [
                ("exit", 0, split, 73),
                ("resume", split, job["updates"], 0),
            ]:
                receipt = child(
                    f"{phase}_{index}",
                    restart_argv
                    + ["--start", str(begin), "--stop", str(end)]
                    + (["--commit-exit"] if expected == 73 else []),
                    raw / "worker_logs",
                    expected=expected,
                    deadline=max(0.001, 3600 - (time.monotonic_ns() - start) / 1e9),
                )
                commands.append(receipt)
                if not receipt["passed"]:
                    raise ValueError("failed_or_killed_restart_child")
                resumed = e.read(e.reference(restarted / f"summary-{begin:04}-{end:04}.json"))
                restart_costs += [
                    dict(
                        head_index=job["head_index"],
                        sequence=job["sequence"],
                        arm="restart_replay",
                        operation=k,
                        device="cpu",
                        **v,
                    )
                    for k, v in resumed["costs"].items()
                ]
            serialized_same = (folder / "checkpoint.bin").read_bytes() == (
                restarted / "checkpoint.bin"
            ).read_bytes()
            same = (
                serialized_same
                and summary["checkpoint"]["state"] == resumed["checkpoint"]["state"]
                and summary["checkpoint"]["ack"] == resumed["checkpoint"]["ack"]
            )
            restarts.append(
                dict(
                    head_index=job["head_index"],
                    sequence=job["sequence"],
                    committed_exit_update=split,
                    real_exit=commands[-2]["actual_exit"],
                    resume_exit=commands[-1]["actual_exit"],
                    replay_updates=job["updates"],
                    state_identical=same,
                    serialized_state_identical=serialized_same,
                    acknowledgement_identical=summary["checkpoint"]["ack"]
                    == resumed["checkpoint"]["ack"],
                    passed=same,
                    uninterrupted=summary["checkpoint"],
                    restarted=resumed["checkpoint"],
                )
            )
        e.progress("lifecycle_units", index + 1, len(jobs) - index - 1)
    raw_refs = [
        e.reference(p)
        for folder in [raw / "workers", raw / "jobs", raw / "worker_logs"]
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    ]
    end = time.monotonic_ns()
    work = dict(
        raw=str(raw),
        run_date=date,
        private_control=private,
        summaries=summaries,
        restart_rows=restarts,
        restart_cost_rows=restart_costs,
        worker_receipts=commands,
        raw_refs=raw_refs,
        source_refs=source_refs,
        code_refs=code_refs,
        upstream_refs=inputs["refs"],
        failures=failures,
        authority=contract,
        historical_model_provenance=inputs["historical_model_provenance"],
        preconditions=pre,
        started_at_utc=wall,
        ended_at_utc=datetime.now(UTC).isoformat(),
        start_ns=start,
        end_ns=end,
        duration_s=(end - start) / 1e9,
        phase_spans=[
            dict(
                phase="authenticate_freeze_measure",
                start_ns=start,
                end_ns=end,
                duration_s=(end - start) / 1e9,
            )
        ],
    )
    atomic_json(raw / "measurement.json", work)
    atomic_json(
        raw.parents[1] / "seals" / (raw.name + ".json"),
        dict(measurement=e.reference(raw / "measurement.json"), code_refs=code_refs),
    )
    e.progress("measurement_complete", len(summaries), len(jobs) - len(summaries))
    return work


def validate(raw: Path) -> list[Json]:
    """Keep original disjoint test nodes and consumer coverage separate from owned code."""
    with tempfile.TemporaryDirectory(prefix="spline8706-cli-custody-", dir="/var/tmp") as directory:
        with (
            patch.object(validation, "e", e),
            patch.dict(os.environ, {"CARNOT_8706_CLI_RETENTION": directory}),
        ):
            receipts = list(validation.validate(raw))
        shutil.copytree(directory, raw / "private_cli_receipts")
    private_refs = [
        e.reference(p) for p in sorted((raw / "private_cli_receipts").rglob("*")) if p.is_file()
    ]
    receipts.append(
        dict(
            name="retained_private_cli_children", passed=bool(private_refs), references=private_refs
        )
    )
    for receipt in receipts:
        if receipt.get("name") == "coverage_report":
            config = Path(
                next(a.split("=", 1)[1] for a in receipt["argv"] if a.startswith("--rcfile="))
            )
            parser = configparser.ConfigParser()
            parser.read(config)
            final = Path(parser["run"]["data_file"])
            retained = raw / "retained_validation"
            retained.mkdir()
            shutil.copyfile(final, retained / "owned.sqlite")
            shutil.copyfile(config, retained / "coverage.ini")
            refs = [e.reference(p) for p in sorted(retained.iterdir())]
            refs += [e.reference(p) for p in sorted((raw / "validation").glob("*.xml"))]
            refs += [e.reference(raw / "coverage.json")]
            receipts.append(
                dict(
                    name="retained_final_validation",
                    passed=all(r["sha256"] for r in refs),
                    references=refs,
                )
            )
    return receipts


def replay(path: Path) -> None:
    """An independently located seal rejects even self-consistently rehashed candidates."""
    value = json.loads(path.read_bytes())
    anchor = Path(value["publication_output"]).parent / "raw" / e.NAME
    raw = anchor / "invocations" / value["invocation_id"]
    seal = e.read(e.reference(anchor / "seals" / (raw.name + ".json")))
    if seal["measurement"] != value["primitive_reference"]:
        raise ValueError("independent_measurement_seal")
    work = e.read(value["primitive_reference"])
    if work["raw"] != str(raw) or work["code_refs"] != seal["code_refs"]:
        raise ValueError("independent_source_seal")
    for ref in work["source_refs"] + work["code_refs"] + work["raw_refs"]:
        e.verify(ref)
    if work["duration_s"] != (work["end_ns"] - work["start_ns"]) / 1e9:
        raise ValueError("clock_drift")
    e.progress("cold_numerical_reconstruction_before", 0, len(work["summaries"]))
    with tempfile.TemporaryDirectory(prefix="spline8706-replay-", dir="/var/tmp") as directory:
        for index, summary in enumerate(work["summaries"]):
            job = e.read(summary["job_reference"])
            rebuilt = e.numerical.drive(job, Path(directory) / str(index), 0, job["updates"], True)
            if Path(rebuilt["rows_path"]).read_bytes() != Path(
                summary["rows_reference"]["path"]
            ).read_bytes() or any(
                rebuilt[k] != summary[k]
                for k in [
                    "checkpoint",
                    "counts",
                    "operations",
                    "accumulator_abs_max",
                    "maximum_probability_error",
                    "maximum_decay_error",
                    "maximum_coefficient_error",
                    "first_failure",
                    "passed",
                ]
            ):
                raise ValueError("numerical_replay_drift")
            e.progress(
                "cold_numerical_reconstruction", index + 1, len(work["summaries"]) - index - 1
            )
    validation_seal = e.read(e.reference(anchor / "validation_seals" / (raw.name + ".json")))
    if validation_seal["validation"] != value["validation_reference"]:
        raise ValueError("independent_validation_seal")
    receipts = e.read(value["validation_reference"])["receipts"]
    for receipt in receipts:
        for ref in receipt.get("references", []) + [
            s["reference"] for s in receipt.get("shards", [])
        ]:
            e.verify(ref)
        for stream in ("stdout", "stderr"):
            if stream + "_path" in receipt:
                e.verify(dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"]))
    if e.build(work, receipts, Path(value["publication_output"])) != value:
        raise ValueError("report_drift")
    e.progress("cold_replay_passed", len(work["summaries"]), 0)


def terminal(path: Path, raw: Path) -> Json:
    """Unchanged validators inspect the exact candidate bytes and retain actual severities."""
    receipts = [
        child(
            name, [sys.executable, "-u", *argv], raw, deadline=900 if name == "cold_replay" else 240
        )
        for name, argv in [
            ("cold_replay", [e.CLI, "--cold-replay", str(path)]),
            ("adversarial", ["scripts/adversarial_verify.py", "--json", str(path)]),
            ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)]),
        ]
    ]
    findings = json.loads(Path(receipts[1]["stdout_path"]).read_bytes())
    return dict(
        passed=all(r["passed"] for r in receipts) and findings["flagged_count"] == 0,
        receipts=receipts,
        actual_findings=findings,
    )


def run(output: Path, date: str, private: bool) -> int:
    """Private evidence stays outside lexical and resolved checkout paths before any writes."""
    if private and (
        output.absolute().is_relative_to(e.ROOT) or output.resolve().is_relative_to(e.ROOT)
    ):
        raise ValueError("private output must be outside checkout")
    output = output.resolve()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = measure(raw, date, private)
    receipts = (
        [dict(name="private_control_only", passed=True, scope="no_production_readiness")]
        if private
        else validate(raw)
    )
    atomic_json(raw / "validation.json", dict(receipts=receipts))
    atomic_json(
        raw.parents[1] / "validation_seals" / (raw.name + ".json"),
        dict(validation=e.reference(raw / "validation.json")),
    )
    value = e.build(work, receipts, output)
    publication = publish_primary(output, value, lambda path: terminal(path, raw / "terminal"))
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="lifecycle_rows_ready_score",
        expected=value["lifecycle_rows_ready_score"],
    )
    atomic_json(
        raw / "terminal_validation.json",
        dict(
            publication=publication,
            consumer=consumer,
            primary_sha256=publication["primary_sha256"],
            passed=consumer["passed"],
        ),
    )
    e.progress(
        "complete_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return 0 if value["required_checks_passed"] and consumer["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """One standalone entrypoint also owns genuine worker exits and fresh-process resumes."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_language_model_calls_zero")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261101")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=4096)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--commit-exit", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.worker:
            if (
                args.work_dir is None
                or not 0 <= args.start < args.stop <= 4096
                or (args.reference and args.start)
            ):
                raise ValueError("worker_contract")
            e.numerical.drive(
                e.read(e.reference(args.worker)),
                args.work_dir,
                args.start,
                args.stop,
                args.reference,
            )
            return 73 if args.commit_exit else 0
        if args.cold_replay:
            replay(args.cold_replay)
            return 0
        if len(args.date) != 8:
            raise ValueError("calendar_date")
        datetime.strptime(args.date, "%Y%m%d")
        with budget(4800):
            return run(args.output.absolute(), args.date, args.private_control)
    except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
        print(f"[exp8706] error={error}", file=sys.stderr, flush=True)
        return 1

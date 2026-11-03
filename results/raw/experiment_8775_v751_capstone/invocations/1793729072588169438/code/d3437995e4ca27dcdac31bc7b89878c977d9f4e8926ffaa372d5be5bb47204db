"""REQ-VERIFY-8719: bounded native execution and independently sealed publication."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from threading import Event, Thread
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import compiled_sparse_study_8719 as e
from carnot.reporting import v743_capstone as validation
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.gatemate_continuity_8400 import budget
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def heartbeat(stop: Event) -> None:
    """The parent remains visible during large evidence reads and validation."""
    while not stop.wait(30):
        e.progress("parent_heartbeat")


def native_build(raw: Path) -> list[Json]:
    """Build the actual opt-in crate; isolate lint and unit checks to the owned module."""
    project = raw / "native_checks"
    (project / "src").mkdir(parents=True)
    (project / "Cargo.toml").write_text(
        '[workspace]\n[package]\nname="spline-check-8719"\nversion="0.1.0"\nedition="2021"\n'
        '[lib]\ncrate-type=["cdylib","rlib"]\n[dependencies]\npyo3={version="=0.24.2",features=["auto-initialize"]}\n'
        'serde={version="1",features=["derive"]}\nserde_json="1"\n'
    )
    (project / "src/lib.rs").write_text(
        'use pyo3::prelude::*;\n#[path = "' + str(e.ROOT / e.RUST[0]) + '"]\nmod arithmetic;\n'
        "#[pymodule]\nfn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {arithmetic::register(m)}\n"
    )
    manifest = str(project / "Cargo.toml")
    child(
        "prepare_harness_format",
        ["rustfmt", "--config", "skip_children=true", str(project / "src/lib.rs")],
        raw / "build_logs",
    )
    plan = [
        ("rustc_version", ["rustc", "-vV"]),
        ("cargo_version", ["cargo", "--version"]),
        (
            "cargo_build",
            [
                "cargo",
                "build",
                "-p",
                "carnot-python",
                "--features",
                "spline-fixed-8719",
                "--offline",
            ],
        ),
        ("cargo_test", ["cargo", "test", "--manifest-path", manifest, "--offline"]),
        ("cargo_fmt", ["cargo", "fmt", "--manifest-path", manifest, "--", "--check"]),
        (
            "scoped_clippy",
            [
                "cargo",
                "clippy",
                "--manifest-path",
                manifest,
                "--all-targets",
                "--offline",
                "--",
                "-D",
                "warnings",
            ],
        ),
    ]
    with patch.dict(
        os.environ, {"PYO3_PYTHON": sys.executable, "RUSTFLAGS": "-C instrument-coverage"}
    ):
        return [child(name, argv, raw / "build_logs", deadline=300) for name, argv in plan]


def measure(raw: Path, date: str, private: bool, extension: Path) -> Json:
    """Freeze original jobs and source identities before any arithmetic measurement."""
    start, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    raw.mkdir(parents=True, exist_ok=True)
    e.progress("preconditions_before")
    with tempfile.TemporaryDirectory(prefix="compiled8719-", dir="/var/tmp") as directory:
        pre = resources(Path(directory))
        pre["write_probe_passed"] = (
            Path(directory) / "write_probe"
        ).read_bytes() == b"private disk scratch"
    admitted = e.inputs()
    failures = admitted["failures"].copy()
    if not (
        pre["disk_backed"]
        and pre["tools_available"]
        and pre["write_probe_passed"]
        and min(pre["free_disk_bytes"], pre["memory_available_bytes"]) >= 2**30
        and shutil.which("cargo")
        and shutil.which("rustc")
    ):
        failures.append(
            dict(
                upstream="host",
                path="/var/tmp",
                sha256=None,
                check="resources",
                field="private_resources",
                operator="==",
                expected=True,
                observed=pre,
                operand_state="resource_unavailable",
                passed=False,
            )
        )
    code_refs = [e.reference(e.ROOT / p) for p in e.OWNED + e.TESTS + e.RUST]
    source_refs = (
        admitted["refs"].copy()
        if not failures
        else [e.reference(Path(ref["path"])) for ref in admitted["refs"]]
    )
    dependencies = (
        e.OWNED
        + e.TESTS
        + e.RUST
        + [
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/v709_execution.py",
            "python/carnot/reporting/v743_capstone.py",
            "python/carnot/reporting/v747_contract_methods.py",
            "python/carnot/verify/spline_precision_8692.py",
            "python/carnot/verify/spline_lifecycle_8706.py",
            "scripts/experiment_template.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "scripts/check_spec_coverage.py",
            "pyproject.toml",
            "tests/python/conftest.py",
            "openspec/capabilities/research-reporting/spec.md",
            "openspec/capabilities/verification/spec.md",
            "ops/e2e-test-plan.md",
            "ops/exclusion_manifest.yaml",
            "openspec/change-proposals/research-roadmap-v746-preserved-20261102.md",
            "research-references.md",
        ]
    )
    for name in dependencies:
        target = raw / "source_closure" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(e.ROOT / name, target)
        source_refs += [e.reference(e.ROOT / name), e.reference(target)]
    build_receipts = [] if private or failures else native_build(raw)
    target = raw / "loaded" / "libcarnot_python.so"
    target.parent.mkdir()
    module = None
    loaded: Json = dict(
        e.reference(target), actual_loaded=False, module_file=None, feature="spline-fixed-8719"
    )
    if extension.is_file():
        shutil.copyfile(extension, target)
        module = e.numerical.load(target)
        loaded.update(e.reference(target), actual_loaded=True, module_file=module.__file__)
        source_refs.append(e.reference(target))
    else:
        failures.append(
            dict(
                upstream="native_extension",
                **e.reference(extension),
                check="loaded_extension_identity",
                field="actual_loaded",
                operator="==",
                expected=True,
                observed=None,
                operand_state="missing_file",
                passed=False,
            )
        )
    source_refs += [e.reference(p) for p in (raw / "native_checks").glob("*") if p.is_file()]
    jobs = [] if failures else admitted["jobs"]
    historical = admitted["historical_refs"]
    if private and jobs:
        jobs = [dict(jobs[0], updates=8), dict(jobs[20], updates=4)]
    job_refs = []
    for i, job in enumerate(jobs):
        path = raw / "jobs" / f"{i:02}.json"
        atomic_json(path, job)
        job_refs.append(e.reference(path))
    source_refs += job_refs
    atomic_json(
        raw / "premeasurement_seal.json",
        dict(
            sources=source_refs,
            code=code_refs,
            jobs=job_refs,
            configuration=dict(format=[32, 24], private=private, restart_at=2048),
        ),
    )
    summaries, restarts, commands = [], [], []
    e.progress("sources_inputs_frozen_before_benchmark", 0, len(jobs))
    for index, (job, ref) in enumerate(zip(jobs, job_refs, strict=True)):
        folder = raw / "workers" / f"{index:02}-full"
        base = [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--worker",
            ref["path"],
            "--extension",
            str(target),
        ]
        hist = [] if private else ["--historical", historical[index]["path"]]
        command = child(
            f"full_{index}",
            base + ["--work-dir", str(folder), "--stop", str(job["updates"])] + hist,
            raw / "worker_logs",
            deadline=max(0.001, 3600 - (time.monotonic_ns() - start) / 1e9),
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
            historical_reference=None if private else historical[index],
            rows_reference=e.reference(Path(summary["rows_path"])),
        )
        summaries.append(summary)
        if not job["boundary"]:
            split = job["updates"] // 2
            restarted = raw / "workers" / f"{index:02}-restart"
            for phase, begin, end, expected in [
                ("exit", 0, split, 73),
                ("resume", split, job["updates"], 0),
            ]:
                receipt = child(
                    f"{phase}_{index}",
                    base
                    + ["--work-dir", str(restarted), "--start", str(begin), "--stop", str(end)]
                    + hist
                    + (["--commit-exit"] if expected == 73 else []),
                    raw / "worker_logs",
                    expected=expected,
                    deadline=max(0.001, 3600 - (time.monotonic_ns() - start) / 1e9),
                )
                commands.append(receipt)
                if not receipt["passed"]:
                    raise ValueError("failed_or_killed_restart_child")
            resumed = e.read(
                e.reference(restarted / f"summary-{split:04}-{job['updates']:04}.json")
            )
            same = (
                summary["checkpoint"] == resumed["checkpoint"]
                and (folder / "checkpoint.bin").read_bytes()
                == (restarted / "checkpoint.bin").read_bytes()
            )
            restarts.append(
                dict(
                    head_index=job["head_index"],
                    sequence=job["sequence"],
                    committed_exit_update=split,
                    real_exit=73,
                    resume_exit=0,
                    replay_updates=job["updates"],
                    state_acknowledgement_and_bytes_identical=same,
                    passed=same,
                )
            )
        e.progress("completed_jobs", index + 1, len(jobs) - index - 1)
    raw_refs = [
        e.reference(p)
        for name in ["workers", "jobs", "worker_logs"]
        for p in sorted((raw / name).rglob("*"))
        if p.is_file()
    ]
    end = time.monotonic_ns()
    work = dict(
        raw=str(raw),
        run_date=date,
        private_control=private,
        summaries=summaries,
        restart_rows=restarts,
        worker_receipts=commands,
        rejection_rows=e.numerical.controls(module) if module else [],
        raw_refs=raw_refs,
        source_refs=source_refs,
        code_refs=code_refs,
        upstream_refs=admitted["refs"],
        failures=failures,
        contract=admitted["contract"],
        historical_model_provenance=admitted["historical_model_provenance"],
        preconditions=pre,
        extension=loaded,
        build_receipts=build_receipts,
        host=dict(
            python=sys.version,
            architecture=platform.machine(),
            platform=platform.platform(),
            cpu=platform.processor(),
            cpu_model=next(
                (
                    line
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                "unknown",
            ),
        ),
        start_ns=start,
        end_ns=end,
        duration_s=(end - start) / 1e9,
        started_at_utc=wall,
        ended_at_utc=datetime.now(UTC).isoformat(),
        phase_spans=[
            dict(
                phase="authenticate_freeze_build_and_measure",
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
    return work


def validate(raw: Path) -> list[Json]:
    """Reuse disjoint bounded children and preserve their genuine coverage databases."""

    def scoped_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
        """A committed hard exit must flush its Python coverage shard before leaving."""
        for arg in argv:
            if arg.startswith("--rcfile="):
                config = Path(arg.split("=", 1)[1])
                config.write_text(
                    config.read_text().replace("patch=subprocess", "patch=subprocess,_exit")
                )
        return dict(child(name, argv, logs, **kwargs))

    with patch.multiple(validation, e=e, child=scoped_child):
        return list(validation.validate(raw))


def replay(path: Path) -> None:
    """An independent seal anchors fresh native reconstruction of every primitive row."""
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
    extension = (
        e.numerical.load(Path(work["extension"]["path"]))
        if work["extension"]["actual_loaded"]
        else None
    )
    e.progress("cold_native_replay_before", 0, len(work["summaries"]))
    with tempfile.TemporaryDirectory(prefix="compiled8719-replay-", dir="/var/tmp") as directory:
        for index, summary in enumerate(work["summaries"]):
            job = e.read(summary["job_reference"])
            history = summary["historical_reference"]
            rebuilt = e.numerical.drive(
                job,
                Path(directory) / str(index),
                0,
                job["updates"],
                extension,
                Path(history["path"]) if history else None,
            )
            if (
                rebuilt["checkpoint"] != summary["checkpoint"]
                or Path(rebuilt["rows_path"]).read_bytes()
                != Path(summary["rows_reference"]["path"]).read_bytes()
            ):
                raise ValueError("native_replay_drift")
            e.progress("cold_native_replay_jobs", index + 1, len(work["summaries"]) - index - 1)
    validation_seal = e.read(e.reference(anchor / "validation_seals" / (raw.name + ".json")))
    if validation_seal["validation"] != value["validation_reference"]:
        raise ValueError("independent_validation_seal")
    receipts = e.read(value["validation_reference"])["receipts"]
    for receipt in receipts:
        for ref in receipt.get("references", []) + [
            s["reference"] for s in receipt.get("shards", [])
        ]:
            e.verify(ref)
        for stream in ["stdout", "stderr"]:
            if stream + "_path" in receipt:
                e.verify(dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"]))
    if e.build(work, receipts, Path(value["publication_output"])) != value:
        raise ValueError("report_drift")
    e.progress("cold_replay_passed")


def terminal(path: Path, raw: Path) -> Json:
    """Unchanged validators inspect exact final bytes before readers can select them."""
    receipts = [
        child(name, [sys.executable, "-u", *argv], raw, deadline=900)
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


def run(output: Path, date: str, private: bool, extension: Path) -> int:
    """Private CLI evidence cannot write through lexical or resolved checkout paths."""
    if private and (
        output.absolute().is_relative_to(e.ROOT) or output.resolve().is_relative_to(e.ROOT)
    ):
        raise ValueError("private output must be outside checkout")
    output = output.resolve()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = measure(raw, date, private, extension)
    receipts = work["build_receipts"] + (
        [dict(name="private_control_only", passed=True)] if private else validate(raw)
    )
    atomic_json(raw / "validation.json", dict(receipts=receipts))
    atomic_json(
        raw.parents[1] / "validation_seals" / (raw.name + ".json"),
        dict(validation=e.reference(raw / "validation.json")),
    )
    value = e.build(work, receipts, output)
    publication = publish_primary(
        output, value, lambda candidate: terminal(candidate, raw / "terminal")
    )
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="native_update_ready_score",
        expected=value["native_update_ready_score"],
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
    """One CLI owns actual process exits, worker resumes and fresh cold replays."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_language_model_calls_zero")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261102")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--extension", type=Path, default=Path(e.numerical.DEFAULT_EXTENSION))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=4096)
    parser.add_argument("--commit-exit", action="store_true")
    args = parser.parse_args(argv)
    stop = Event()
    thread = Thread(target=heartbeat, args=(stop,), daemon=True)
    thread.start()
    try:
        if args.worker:
            if args.work_dir is None or not 0 <= args.start < args.stop <= 4096:
                raise ValueError("worker_contract")
            e.numerical.drive(
                e.read(e.reference(args.worker)),
                args.work_dir,
                args.start,
                args.stop,
                e.numerical.load(args.extension),
                args.historical,
            )
            if args.commit_exit:
                os._exit(73)
            return 0
        if args.cold_replay:
            replay(args.cold_replay)
            return 0
        if len(args.date) != 8:
            raise ValueError("calendar_date")
        datetime.strptime(args.date, "%Y%m%d")
        with budget(4800):
            return run(args.output.absolute(), args.date, args.private_control, args.extension)
    except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
        print(f"[exp8719] error={error}", file=sys.stderr, flush=True)
        return 1
    finally:
        stop.set()
        thread.join(timeout=1)

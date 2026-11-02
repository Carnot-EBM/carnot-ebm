"""REQ-VERIFY-8732: retain bounded child evidence and validate exact final bytes."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import resident_update_cost_8732 as e
from carnot.reporting import v743_capstone as checks
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
E2E = [
    "tests/python/test_source_boundary_7852.py",
    "tests/python/test_experiment_7942_v689_sentence_labels.py",
]


def measure(raw: Path, date: str, private: bool) -> Json:
    """Freeze sources, methods and order before the first timed child starts."""
    start, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    raw.mkdir(parents=True)
    e.progress("preconditions_before")
    scratch = Path(tempfile.mkdtemp(prefix="resident8732-", dir="/var/tmp"))
    pre = resources(scratch)
    pre["write_probe_passed"] = (scratch / "write_probe").read_bytes() == b"private disk scratch"
    admitted = e.inputs()
    failures = admitted["failures"].copy()
    if not (
        pre["disk_backed"]
        and pre["tools_available"]
        and pre["write_probe_passed"]
        and min(pre["memory_available_bytes"], pre["free_disk_bytes"]) >= 2**30
    ):
        failures.append(
            dict(
                upstream="host",
                path=str(scratch),
                sha256=None,
                check="resources",
                field="disk_memory_tools",
                operator="==",
                expected=True,
                observed=pre,
                passed=False,
            )
        )
    dependencies = (
        e.OWNED
        + e.TESTS
        + e.upstream.e.OWNED
        + e.upstream.e.TESTS
        + e.upstream.e.RUST
        + [
            "python/carnot/verify/spline_precision_8692.py",
            "python/carnot/verify/block_feedback_state_8433.py",
            "python/carnot/verify/compiled_sparse_8719.py",
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/v709_execution.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/v743_capstone.py",
            "python/carnot/reporting/v748_collection_contract.py",
            "python/carnot/reporting/roadmap_contract.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "scripts/check_spec_coverage.py",
            "scripts/experiment_template.py",
            "pyproject.toml",
            "tests/python/conftest.py",
            "ops/e2e-test-plan.md",
            "ops/exclusion_manifest.yaml",
            "openspec/capabilities/research-reporting/spec.md",
            "openspec/capabilities/verification/spec.md",
            "research-references.md",
        ]
    )
    code = [e.reference(e.ROOT / p) for p in dict.fromkeys(dependencies)]
    snapshots = []
    for ref in code:
        relative = Path(ref["path"]).relative_to(e.ROOT)
        target = raw / "source_closure" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ref["path"], target)
        snapshots.append(dict(path=str(target), sha256=ref["sha256"]))
    roster = e.roster(private)
    jobs = [] if failures else [admitted["jobs"][0], admitted["jobs"][20]]
    for i, job in enumerate(jobs):
        atomic_json(raw / f"job-{i}.json", job)
    freeze = dict(
        code=code,
        inputs=admitted["refs"],
        roster=roster,
        jobs=[e.reference(raw / f"job-{i}.json") for i in range(len(jobs))],
    )
    atomic_json(raw / "premeasurement_seal.json", freeze)
    e.progress("inputs_sources_roster_frozen", 0, len(roster))
    processes: list[Json] = []
    refs = snapshots.copy()
    probe: Json = {}
    if not private and not failures:
        e.progress("durability_budget_probe_before", 0, 4)
        measured = e.numerical.drive(jobs[0], scratch / "budget-probe", "python", 0, 4, None)
        minimum_ns = min(measured["latencies_ns"])
        projection = minimum_ns * 122880 / 1e9
        probe = dict(
            observed_updates=4,
            minimum_complete_update_ns=minimum_ns,
            projected_minimum_measurement_s=projection,
            projection_excludes_launch_restart_and_dense_controls=True,
            latencies_ns=measured["latencies_ns"],
            filesystem=pre["filesystem"],
        )
        atomic_json(raw / "durability_probe.json", probe)
        probe_ref = e.reference(raw / "durability_probe.json")
        shutil.copyfile(measured["rows_path"], raw / "durability_probe_rows.json")
        refs += [probe_ref, e.reference(raw / "durability_probe_rows.json")]
        if projection > 3600:
            failures.append(
                dict(
                    upstream="host_disk_durability",
                    **probe_ref,
                    check="complete_roster_fits_measurement_cap",
                    field="projected_complete_measurement_s",
                    operator="<=",
                    expected=3600,
                    observed=projection,
                    passed=False,
                    operand_state="durability_resource_budget_exceeded",
                )
            )
        e.progress("durability_budget_probe_after", 4, 0)
    for cell in [] if failures else roster:
        folder = scratch / cell["unit_id"].replace(":", "-")
        commands, summaries = [], []
        begin = time.monotonic_ns()
        for segment, (low, high) in enumerate(
            [(0, cell["restart_at"]), (cell["restart_at"], cell["updates"])]
        ):
            command = child(
                cell["unit_id"].replace(":", "-") + f"-{segment}",
                [
                    sys.executable,
                    "-u",
                    str(e.ROOT / e.CLI),
                    "--worker",
                    str(raw / "job-0.json"),
                    "--control",
                    str(raw / "job-1.json"),
                    "--work-dir",
                    str(folder),
                    "--arm",
                    cell["arm"],
                    "--start",
                    str(low),
                    "--stop",
                    str(high),
                    "--boundary-updates",
                    str(cell["boundary_updates"]),
                    "--extension",
                    admitted["extension"]["path"],
                ],
                raw / "worker_logs",
                deadline=max(0.001, 3600 - (time.monotonic_ns() - start) / 1e9),
            )
            commands.append(command)
            if not command["passed"]:
                raise ValueError("failed_or_killed_measurement_child")
            summaries += [folder / f"summary-{low}-{high}.json"]
        complete = time.monotonic_ns() - begin
        summaries += [folder / "control" / f"summary-0-{cell['boundary_updates']}.json"]
        bound = []
        for i, source in enumerate(summaries):
            summary = json.loads(source.read_bytes())
            rows = raw / "workers" / cell["unit_id"].replace(":", "-") / f"rows-{i}.json"
            rows.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(summary["rows_path"], rows)
            summary.update(rows_reference=e.reference(rows), control=i == 2)
            dest = rows.with_name(f"summary-{i}.json")
            atomic_json(dest, summary)
            bound.append(e.reference(dest))
            refs += [e.reference(rows), e.reference(dest)]
        semantic = [
            dict(r, latency_ns=0)
            for ref in bound[:2]
            for r in e.read(e.read(ref)["rows_reference"])["rows"]
        ]
        processes.append(
            dict(
                cell,
                complete_ns=complete,
                summaries=bound,
                commands=commands,
                kernel_ns=sum(e.read(r)["kernel_ns"] for r in bound),
                semantic_hash=canonical_hash(semantic),
                final_state_hash=canonical_hash(e.read(bound[1])["checkpoint"]["head"]),
                restart_replay_updates=0,
                restart_resume_child=commands[1],
                passed=True,
            )
        )
        e.progress("paired_lifecycles_completed", len(processes), len(roster) - len(processes))
    for repeat in {p["repeat"] for p in processes}:
        group = [p for p in processes if p["repeat"] == repeat]
        matched = len({p["semantic_hash"] for p in group}) == 1
        for p in group:
            p["passed"] = matched
    end = time.monotonic_ns()
    work = dict(
        raw=str(raw),
        run_date=date,
        private_control=private,
        roster=roster,
        process_rows=processes,
        durability_probe=probe,
        failures=failures,
        contract=admitted["contract"],
        preconditions=pre,
        code_refs=code,
        source_refs=[e.reference(Path(r["path"])) for r in admitted["refs"]],
        raw_refs=refs + [e.reference(raw / "premeasurement_seal.json")],
        upstream_refs=admitted["refs"],
        durability_boundary=dict(
            implementation="block_feedback_state_8433.Store.commit",
            prediction_seal_before_label=True,
            feedback_snapshot_and_pointer_fsync=True,
            directory_fsync_before_ack=True,
            commits_per_update=2,
            buffered_only=False,
        ),
        start_ns=start,
        end_ns=end,
        duration_s=(end - start) / 1e9,
        started_at_utc=wall,
        ended_at_utc=datetime.now(UTC).isoformat(),
        phase_spans=[
            dict(
                phase="authenticate_freeze_complete_lifecycle",
                start_ns=start,
                end_ns=end,
                duration_s=(end - start) / 1e9,
            )
        ],
    )
    atomic_json(raw / "measurement.json", work)
    atomic_json(
        raw.parents[1] / "seals" / (raw.name + ".json"),
        dict(measurement=e.reference(raw / "measurement.json"), code=code),
    )
    shutil.rmtree(scratch)
    return work


def replay(path: Path) -> None:
    """Independent measurement and validation seals reject self-rehashed candidates."""
    value = json.loads(path.read_bytes())
    anchor = Path(value["publication_output"]).parent / "raw" / e.NAME
    raw = anchor / "invocations" / value["invocation_id"]
    seal = e.read(e.reference(anchor / "seals" / (raw.name + ".json")))
    if seal["measurement"] != value["primitive_reference"]:
        raise ValueError("independent_measurement_seal")
    work = e.read(value["primitive_reference"])
    if work["code_refs"] != seal["code"] or work["roster"] != e.roster(work["private_control"]):
        raise ValueError("independent_source_or_roster_seal")
    for ref in work["source_refs"] + work["code_refs"] + work["raw_refs"]:
        e.verify(ref)
    if work["duration_s"] != (work["end_ns"] - work["start_ns"]) / 1e9:
        raise ValueError("clock_drift")
    frozen = e.read(e.reference(raw / "premeasurement_seal.json"))
    if frozen["code"] != work["code_refs"] or frozen["roster"] != work["roster"]:
        raise ValueError("premeasurement_drift")
    for ref in frozen["jobs"]:
        e.verify(ref)
    validation = e.read(e.reference(anchor / "validation_seals" / (raw.name + ".json")))
    if validation["validation"] != value["validation_reference"]:
        raise ValueError("independent_validation_seal")
    receipts = e.read(value["validation_reference"])["receipts"]
    if e.build(work, receipts, Path(value["publication_output"])) != value:
        raise ValueError("report_drift")
    e.progress("cold_replay_passed")


def terminal(path: Path, raw: Path) -> Json:
    """Existing validators inspect final candidate bytes before atomic replacement."""
    receipts = [
        child(name, [sys.executable, "-u", *args], raw, deadline=300)
        for name, args in [
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


def validate(raw: Path) -> list[Json]:
    """Reuse scoped validation custody, retaining original node IDs and SQLite."""
    with patch.multiple(checks, e=e, E2E=E2E):
        return list(checks.validate(raw))


def run(output: Path, date: str, private: bool) -> int:
    """Private controls cannot overwrite public outputs or grant cost readiness."""
    if private and (
        output.absolute().is_relative_to(e.ROOT) or output.resolve().is_relative_to(e.ROOT)
    ):
        raise ValueError("private output must be outside checkout")
    output = output.resolve()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = measure(raw, date, private)
    receipts = [dict(name="private_control_only", passed=True)] if private else validate(raw)
    atomic_json(raw / "validation.json", dict(receipts=receipts))
    atomic_json(
        raw.parents[1] / "validation_seals" / (raw.name + ".json"),
        dict(validation=e.reference(raw / "validation.json")),
    )
    value = e.build(work, receipts, output)
    publication = publish_primary(
        output, value, lambda candidate: terminal(candidate, raw / "terminal")
    )
    atomic_json(raw / "terminal_validation.json", dict(publication=publication, passed=True))
    e.progress(
        "complete_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return 0 if value["required_checks_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """One bounded CLI handles worker segments, cold replay and final publication."""
    e.progress("start_no_model_load_MODEL_SPECS_empty")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261102")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--arm", choices=e.ARMS)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=4096)
    parser.add_argument("--boundary-updates", type=int, default=64)
    parser.add_argument("--extension", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            if not (
                args.control
                and args.work_dir
                and args.arm
                and args.extension
                and 0 <= args.start < args.stop <= 4096
            ):
                raise ValueError("worker_contract")
            extension = e.numerical.resident.load(args.extension) if args.arm != "python" else None
            if args.start == 0:
                e.numerical.drive(
                    e.read(e.reference(args.control)),
                    args.work_dir / "control",
                    args.arm,
                    0,
                    args.boundary_updates,
                    extension,
                )
            e.numerical.drive(
                e.read(e.reference(args.worker)),
                args.work_dir,
                args.arm,
                args.start,
                args.stop,
                extension,
            )
            return 0
        if args.cold_replay:
            replay(args.cold_replay)
            return 0
        if args.date != "20261102":
            raise ValueError("calendar_date")
        return run(args.output, args.date, args.private_control)
    except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
        print(f"[exp8732] error={error}", file=sys.stderr, flush=True)
        return 1

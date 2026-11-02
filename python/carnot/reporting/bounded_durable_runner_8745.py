"""REQ-VERIFY-8745: retain actual admission evidence and publish checked bytes."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import bounded_durable_cost_8745 as e
from carnot.reporting import v743_capstone as checks
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.resident_update_runner_8732 import terminal as old_terminal

Json = dict[str, Any]
E2E = [
    "tests/python/test_resident_sparse_8731.py",
    "tests/python/test_block_feedback_state_8433.py::test_real_crashes_and_future_decisions",
    "tests/python/test_source_boundary_7852.py",
    "tests/python/test_experiment_7942_v689_sentence_labels.py",
]


def measure(raw: Path, date: str) -> Json:
    """Freeze the complete plan before probing the actual destination filesystem."""
    start, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    raw.mkdir(parents=True)
    e.progress("authenticate_before")
    admitted = e.inputs()
    pre = resources(raw)
    pre.update(
        no_model_load=True,
        MODEL_SPECS=[],
        exclusion_manifest=e.reference(e.ROOT / "ops/exclusion_manifest.yaml"),
    )
    code = [
        e.reference(e.ROOT / p)
        for p in e.OWNED
        + e.TESTS
        + [
            "python/carnot/verify/resident_update_cost_8732.py",
            "python/carnot/verify/block_feedback_state_8433.py",
            "python/carnot/verify/spline_precision_8692.py",
            "scripts/experiment_template.py",
            "ops/exclusion_manifest.yaml",
            "ops/e2e-test-plan.md",
            "openspec/change-proposals/research-roadmap-vNEXT.md",
            "openspec/change-proposals/research-roadmap-v748-preserved-20261102.md",
        ]
    ]
    freeze = dict(method=e.METHOD, roster=e.roster(), inputs=admitted["refs"], code=code)
    atomic_json(raw / "premeasurement_seal.json", freeze)
    failures = list(admitted["failures"])
    starts: list[Json] = []
    costs: list[Json] = []
    probe: Json = {}
    projection: Json = {}
    loaded: Json = {}
    if not pre["disk_backed"]:
        failures.append(e.failure(pre["exclusion_manifest"], "disk_backed", True, False))
    e.progress("sources_plan_frozen", 0, 90)
    if not failures:
        e.progress("native_binary_load_before")
        module = e.previous.numerical.resident.load(Path(admitted["extension"]["path"]))
        loaded = dict(e.reference(Path(module.__file__)), actual_loaded=True)
        if loaded["sha256"] != admitted["extension"]["sha256"]:
            raise ValueError("actual_loaded_binary")
        e.progress("native_binary_load_after")
        starts, costs, gate = e.starting_states(admitted, raw)
        if not gate["passed"]:
            failures.append(gate)
        atomic_json(raw / "probe-job.json", admitted["job"])
        e.progress("four_update_durability_probe_before", 0, 4)
        command = child(
            "durability_probe",
            [
                sys.executable,
                "-u",
                e.CLI,
                "--probe-job",
                str(raw / "probe-job.json"),
                "--work-dir",
                str(raw / "probe"),
            ],
            raw / "probe_logs",
            deadline=120,
        )
        if not command["passed"]:
            raise ValueError("durability_probe_child_failed")
        summary = e.read(e.reference(raw / "probe/summary-0-4.json"))
        probe = dict(
            summary,
            process_receipt=command,
            filesystem=pre["filesystem"],
            measured_updates=4,
            scope="admission_only_excluded_from_2880_updates",
        )
        projection = e.projection(summary["latencies_ns"], round(command["duration_s"] * 1e9))
        atomic_json(raw / "durability_probe.json", probe)
        atomic_json(raw / "resource_projection.json", projection)
        if not projection["admitted"]:
            failures.append(
                e.failure(
                    e.reference(raw / "resource_projection.json"),
                    "projected_with_reserve_s",
                    3000,
                    projection["projected_with_reserve_s"],
                    operator="<=",
                )
            )
        e.progress("four_update_durability_probe_after", 4, 0)
    end = time.monotonic_ns()
    work = dict(
        raw=str(raw),
        run_date=date,
        roster=e.roster(),
        failures=failures,
        starting_state_rows=starts,
        excluded_prefix_cost_rows=costs,
        durability_probe=probe,
        resource_projection=projection,
        historical_probe=admitted["historical_probe"],
        loaded_extension_reference=loaded,
        preconditions=pre,
        source_refs=admitted["refs"],
        code_refs=code,
        raw_refs=[e.reference(p) for p in sorted(raw.rglob("*")) if p.is_file()],
        start_ns=start,
        end_ns=end,
        duration_s=(end - start) / 1e9,
        started_at_utc=wall,
        ended_at_utc=datetime.now(UTC).isoformat(),
        phase_spans=[
            dict(
                phase="authenticate_prefix_and_admission_probe",
                start_ns=start,
                end_ns=end,
                duration_s=(end - start) / 1e9,
            )
        ],
    )
    atomic_json(raw / "measurement.json", work)
    atomic_json(
        raw / "measurement_seal.json", dict(measurement=e.reference(raw / "measurement.json"))
    )
    return work


def replay(path: Path) -> None:
    """Rebuild terminal fields from independently sealed real measurement bytes."""
    value = json.loads(path.read_bytes())
    raw = (
        Path(value["publication_output"]).parent
        / "raw"
        / e.NAME
        / "invocations"
        / value["invocation_id"]
    )
    seal = e.read(e.reference(raw / "measurement_seal.json"))
    if seal["measurement"] != value["primitive_reference"]:
        raise ValueError("independent_measurement_seal")
    work = e.read(value["primitive_reference"])
    for ref in work["source_refs"] + work["code_refs"] + work["raw_refs"]:
        e.verify(ref)
    frozen = e.read(e.reference(raw / "premeasurement_seal.json"))
    if work["duration_s"] != (work["end_ns"] - work["start_ns"]) / 1e9 or frozen != dict(
        method=e.METHOD, roster=e.roster(), inputs=work["source_refs"], code=work["code_refs"]
    ):
        raise ValueError("clock_or_method_drift")
    checks_ref = e.read(e.reference(raw / "validation_seal.json"))["validation"]
    if checks_ref != value["validation_reference"]:
        raise ValueError("independent_validation_seal")
    receipts = e.read(checks_ref)["receipts"]
    if e.build(work, receipts, Path(value["publication_output"])) != value:
        raise ValueError("report_drift")
    e.progress("cold_replay_passed")


def terminal(path: Path, raw: Path) -> Json:
    """Use the unchanged adversarial and row validators on exact candidate bytes."""
    with patch("carnot.reporting.resident_update_runner_8732.e", e):
        return dict(old_terminal(path, raw))


def validate(raw: Path) -> list[Json]:
    """Reuse the original disjoint JUnit and real subprocess SQLite collector."""
    with patch.multiple(checks, e=e, E2E=E2E):
        return list(checks.validate(raw))


def run(output: Path, date: str, *, private: bool = False) -> int:
    """Private controls cannot overwrite the public primary or qualify missing work."""
    if private and output.resolve().is_relative_to(e.ROOT):
        raise ValueError("private_output")
    output = output.resolve()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = measure(raw, date)
    receipts = [dict(name="private_control_only", passed=True)] if private else validate(raw)
    if not private:
        receipts.append(
            child(
                "full_python_suite",
                [str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
                raw / "full_suite",
                deadline=180,
            )
        )
    atomic_json(raw / "validation.json", dict(receipts=receipts))
    atomic_json(raw / "validation_seal.json", dict(validation=e.reference(raw / "validation.json")))
    value = e.build(work, receipts, output)
    publication = publish_primary(
        output, value, lambda candidate: terminal(candidate, raw / "terminal")
    )
    atomic_json(raw / "terminal_validation.json", dict(publication=publication, passed=True))
    e.progress(value["honest_verdict"], 0, 90)
    return 0 if value["required_checks_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """The only measurement worker runs the original four-update durable probe."""
    e.progress("start_no_model_load_MODEL_SPECS_empty")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261102")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--probe-job", type=Path)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.probe_job:
            if not args.work_dir:
                raise ValueError("probe_work_dir")
            e.previous.numerical.drive(
                e.read(e.reference(args.probe_job)), args.work_dir, "python", 0, 4, None
            )
            return 0
        if args.cold_replay:
            replay(args.cold_replay)
            return 0
        if args.date != "20261102":
            raise ValueError("calendar_date")
        return run(args.output, args.date)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp8745] error={error}", file=sys.stderr, flush=True)
        return 1

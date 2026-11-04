"""REQ-VERIFY-8796: seal failed preconditions before publishing missing timing rows."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import observer_intervention_8796 as e
from carnot.reporting import v743_capstone as checks
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.resident_update_runner_8732 import terminal as prior_terminal
from carnot.reporting.lean_runner_8608 import E2E
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def seal(work: Json) -> None:
    """The primitive digest lives outside the report to reject rehashed summaries."""
    raw = Path(work["raw"])
    e.atomic_json(raw / "measurement.json", work)
    e.atomic_json(
        raw / "measurement_seal.json", dict(measurement=e.reference(raw / "measurement.json"))
    )


def measure(raw: Path, date: str, private_input: Path | None = None) -> Json:
    """Record preflight cost only; rejected authority prohibits even a timing probe."""
    begin, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    raw.mkdir(parents=True, mode=0o700)
    scratch = Path(tempfile.mkdtemp(prefix="carnot8796-", dir="/var/tmp"))
    e.progress("preconditions_before", 0, 3)
    pre = resources(scratch)
    admitted = e.read(e.reference(private_input)) if private_input else e.inputs()
    failures = list(admitted["failures"])
    if not pre["disk_backed"] or not pre["tools_available"]:
        failures.append(
            e.failure(
                e.reference(scratch / "write_probe")
                if (scratch / "write_probe").exists()
                else dict(path=str(scratch), sha256=None),
                "private_disk_and_dependencies",
                True,
                False,
                "missing_resource",
            )
        )
    if not failures:
        raise ValueError("qualified_observer_execution_required")
    refs = list(admitted["refs"])
    if private_input:
        refs.append(e.reference(private_input))
    for index, failed in enumerate(failures):
        source = Path(failed["path"])
        if source.is_file():
            frozen = raw / f"rejected-operand-{index}.bin"
            frozen.write_bytes(source.read_bytes())
            refs.append(dict(e.reference(frozen), original_path=str(source)))
    dependencies = [
        "python/carnot/reporting/durable_cost_components_8784.py",
        "python/carnot/reporting/durable_component_clocks_8784.py",
        "python/carnot/reporting/durable_component_runner_8784.py",
        "python/carnot/reporting/durable_producer_identity_8783.py",
        "python/carnot/reporting/primary_publication.py",
        "scripts/experiment_template.py",
        "ops/exclusion_manifest.yaml",
    ]
    code = [e.reference(e.ROOT / name) for name in [*e.OWNED, *e.TESTS, *dependencies]]
    end = time.monotonic_ns()
    projection = dict(
        admitted=False,
        measurement_executed=False,
        complete_triple_ns=None,
        projected_36_window_execution_s=None,
        maximum_execution_s=2400,
        validation_reserve_s=900,
        measured_preflight_duration_s=(end - begin) / 1e9,
        measured_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        reason="upstream_admission_failed_before_control_or_triple",
    )
    pre.update(
        dependency_imports=True,
        MODEL_SPECS=[],
        no_model_load=True,
        current_loaded_extension=None,
        native_extension_candidate=admitted.get("extension"),
        control_executed=False,
        measurement_executed=False,
    )
    e.atomic_json(
        raw / "premeasurement_seal.json",
        dict(method=e.METHOD, roster=e.roster(), code=code, sources=refs),
    )
    work = dict(
        raw=str(raw),
        run_date=date,
        private_control=private_input is not None,
        failures=failures,
        source_refs=refs,
        code_refs=code,
        raw_refs=[e.reference(raw / "premeasurement_seal.json")],
        preconditions_checked=pre,
        resource_projection=projection,
        duration_s=(end - begin) / 1e9,
        start_ns=begin,
        end_ns=end,
        started_at_utc=wall,
        ended_at_utc=datetime.now(UTC).isoformat(),
        phase_spans=[
            dict(
                phase="authenticate_and_preflight_only",
                start_ns=begin,
                end_ns=end,
                duration_s=(end - begin) / 1e9,
            )
        ],
        historical_observer_diagnostics=admitted.get("historical_observer_diagnostics", []),
    )
    seal(work)
    e.progress("admission_blocked_no_measurement", 3, 36)
    return work


def replay(path: Path) -> None:
    """Rebuild missing rows from sealed admission operands, never from claimed scores."""
    value = json.loads(path.read_bytes())
    if (value["experiment_id"], value["task_id"], value["milestone"]) != (
        8796,
        e.TASK,
        "2026.11.753",
    ):
        raise ValueError("producer_identity")
    raw = (
        Path(value["publication_output"]).parent
        / "raw"
        / e.NAME
        / "invocations"
        / value["invocation_id"]
    )
    if (
        e.read(e.reference(raw / "measurement_seal.json"))["measurement"]
        != value["primitive_reference"]
    ):
        raise ValueError("independent_measurement_seal")
    work = e.read(value["primitive_reference"])
    for ref in work["source_refs"] + work["code_refs"] + work["raw_refs"]:
        e.verify(ref)
    frozen = e.read(e.reference(raw / "premeasurement_seal.json"))
    if (
        frozen["method"] != e.METHOD
        or frozen["roster"] != e.roster()
        or frozen["code"] != work["code_refs"]
        or frozen["sources"] != work["source_refs"]
        or work["duration_s"] != (work["end_ns"] - work["start_ns"]) / 1e9
    ):
        raise ValueError("clock_or_method_drift")
    if (
        e.read(e.reference(raw / "validation_seal.json"))["validation"]
        != value["validation_reference"]
    ):
        raise ValueError("independent_validation_seal")
    receipts = e.read(value["validation_reference"])["receipts"]
    for receipt in receipts:
        for stream in ["stdout", "stderr"]:
            if stream + "_path" in receipt:
                e.verify(dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"]))
    if e.build(work, receipts, Path(value["publication_output"])) != value:
        raise ValueError("report_drift")
    e.progress("cold_replay_passed", 1, 0)


def validate(raw: Path) -> list[Json]:
    """Reuse original node, JUnit and child SQLite custody for this owned scope."""
    with patch.multiple(checks, e=e, E2E=E2E):
        return list(checks.validate(raw))


def terminal(path: Path, raw: Path) -> Json:
    """Unchanged scanners and a cold child inspect the exact publication bytes."""
    with patch("carnot.reporting.resident_update_runner_8732.e", e):
        return dict(prior_terminal(path, raw))


def run(output: Path, date: str, private_input: Path | None = None) -> int:
    """Private fixtures cannot publish into the checkout or authorize readiness."""
    if private_input and (
        output.absolute().is_relative_to(e.ROOT) or output.resolve().is_relative_to(e.ROOT)
    ):
        raise ValueError("private_output")
    output = output.resolve()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = measure(raw, date, private_input)
    receipts = [dict(name="private_control_only", passed=True)] if private_input else validate(raw)
    if not private_input:
        health = child(
            "full_python_suite",
            [str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
            raw / "repository_health",
            deadline=180,
            scope="repository_health",
        )
        work["repository_health_reference"] = dict(
            current_receipt=health, full_suite_pass_claimed=False
        )
        seal(work)
    e.atomic_json(raw / "validation.json", dict(receipts=receipts))
    e.atomic_json(
        raw / "validation_seal.json", dict(validation=e.reference(raw / "validation.json"))
    )
    value = e.build(work, receipts, output)
    publication = publish_primary(
        output, value, lambda candidate: terminal(candidate, raw / "terminal")
    )
    e.atomic_json(raw / "terminal_validation.json", dict(publication=publication, passed=True))
    e.progress(value["honest_verdict"], 0, 36)
    return 0 if value["required_checks_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """One thin CLI supports sealed blocked publication and outside-process replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261104")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            replay(args.cold_replay)
            return 0
        if args.date != "20261104":
            raise ValueError("calendar_date")
        return run(args.output, args.date, args.private_input)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp8796] error={error}", file=sys.stderr, flush=True)
        return 1

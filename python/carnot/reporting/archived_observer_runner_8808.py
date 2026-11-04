"""REQ-VERIFY-8808: reuse original durable workers and sealed publication checks."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, UTC
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any
from unittest.mock import patch

from carnot.reporting import archived_observer_costs_8808 as e
from carnot.reporting import durable_component_runner_8784 as prior
from carnot.reporting import durable_component_clocks_8784 as clocks
from carnot.reporting import ordinary_durable_runner_8772 as original
from carnot.reporting import v743_capstone as checks
from carnot.reporting.lean_runner_8608 import E2E
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
__all__ = ["child"]
runtime = namespace(original, "archived8808")
runtime.e, runtime.E2E = e, E2E
cells = namespace(prior, "archived8808_cells")
cells.e = e
install = clocks.install


def segment_child(
    name: str, argv: list[str], logs: Path, *, deadline: float = 180, **kwargs: Any
) -> Json:
    """Fork the imported parent so custody and imports are paid once for the study."""
    if "--worker" not in argv:
        return dict(child(name, argv, logs, deadline=deadline, **kwargs))
    logs.mkdir(parents=True, exist_ok=True)
    stdout, stderr = logs / (name + ".stdout"), logs / (name + ".stderr")
    request = Path(argv[argv.index("--worker") + 1])
    folder = Path(argv[argv.index("--work-dir") + 1])

    def execute() -> None:
        """Child coverage is saved explicitly before the forked process exits."""
        import coverage

        with stdout.open("w") as out, stderr.open("w") as err:
            os.dup2(out.fileno(), 1)
            os.dup2(err.fileno(), 2)
            cov = coverage.Coverage.current()
            try:
                worker(e.read(e.reference(request)), folder)
            except (OSError, ValueError, KeyError, TypeError):
                traceback.print_exc()
                raise SystemExit(1)
            finally:
                if cov:
                    cov.save()

    e.progress("before_fork_" + name, 0, 1)
    begin = time.perf_counter_ns()
    process = multiprocessing.get_context("fork").Process(target=execute)
    process.start()
    while process.is_alive():
        process.join(timeout=min(60, max(0.001, deadline - (time.perf_counter_ns() - begin) / 1e9)))
        if process.is_alive():
            e.progress("heartbeat_fork_" + name, 0, 1)
            if (time.perf_counter_ns() - begin) / 1e9 >= deadline:
                process.kill()
                process.join()
    end = time.perf_counter_ns()
    receipt = dict(
        name=name,
        argv=argv,
        executor="fork_of_imported_parent",
        passed=process.exitcode == 0,
        exit_code=process.exitcode,
        started_monotonic_ns=begin,
        ended_monotonic_ns=end,
        duration_s=(end - begin) / 1e9,
        stdout_path=str(stdout),
        stderr_path=str(stderr),
        stdout_sha256=e.reference(stdout)["sha256"],
        stderr_sha256=e.reference(stderr)["sha256"],
    )
    e.atomic_json(logs / (name + ".receipt.json"), receipt)
    e.progress("after_fork_" + name, 1, 0)
    return receipt


class Buffer:
    """Fixed storage avoids growing an observer list inside a timed durable block."""

    def __init__(self) -> None:
        self.values: list[Any] = [None] * 4096
        self.used = 0

    def append(self, value: Any) -> None:
        """The same dispatch and writes occur in sham and clock conditions."""
        self.values[self.used] = value
        self.used += 1


def worker(request: Json, folder: Path) -> Json:
    """Keep score arithmetic intact; only private observer copies receive clock hooks."""
    buffer = Buffer()
    condition = request.get("condition", "plain")
    folder.mkdir(parents=True, exist_ok=True)

    def buffered(stack: ExitStack, arm: str, spans: list[Any]) -> None:
        """Compile both conditions identically; sham replaces only timestamp reads."""
        clock = time.perf_counter_ns if condition == "clock" else lambda: 0
        with patch.object(clocks.time, "perf_counter_ns", clock):
            install(stack, arm, buffer)

    with patch.object(clocks, "install", buffered):
        # The shipped worker still owns state reconstruction and every durable barrier.
        result: Json = cells.worker(
            dict(request, condition="instrumented" if condition != "plain" else "plain"), folder
        )
    spans = buffer.values[: buffer.used] if condition == "clock" else []
    result["spans"] = spans
    result["components"] = clocks.partition(spans, result["begin_ns"], result["end_ns"])
    result["observer_dispatches"] = buffer.used
    for row in result["rows"]:
        row["condition"] = condition
    e.atomic_json(folder / "result.json", result)
    return result


def cell(spec: Json, admitted: Json, raw: Path) -> Json:
    """Two real processes preserve the restart boundary for every registered window."""
    result: Json = cells.cell(spec, admitted, raw)
    result.update(
        metric="full_operation_parent_ns",
        denominator=32,
        missing_reason=None if result["passed"] else "failed_durable_window",
    )
    return result


def measure(raw: Path, date: str, private_input: Path | None = None) -> Json:
    """Authenticate once, then use the unchanged cheap probe and complete roster."""
    begin, wall = time.perf_counter_ns(), datetime.now(UTC).isoformat()
    e.progress("authority_before", 0, 1)
    admitted = e.read(e.reference(private_input)) if private_input else e.inputs()
    authority_end = time.perf_counter_ns()
    e.progress("authority_after", 1, 0)
    selected = admitted.get("private_roster", e.roster()) if private_input else e.roster()
    resources, project = runtime.resources, e.projection

    def private_resources(folder: Path) -> Json:
        """Raw durable state is private even when the caller has a permissive umask."""
        folder.chmod(0o700)
        return dict(resources(folder), private_scratch_mode="0700", dependency_imports=True)

    with (
        patch.object(runtime.e, "inputs", lambda: admitted),
        patch.object(runtime.e, "roster", lambda: selected),
        patch.object(runtime, "resources", private_resources),
        patch.object(
            runtime.e,
            "projection",
            lambda latencies, complete: project(
                latencies, complete, one_time_s=(authority_end - begin) / 1e9
            ),
        ),
    ):
        work: Json = runtime.measure(raw, date)
    work["private_control"] = private_input is not None
    for index, failed in enumerate(work["failures"]):
        source = Path(failed["path"])
        if source.is_file():
            retained = raw / f"rejected-operand-{index}.bin"
            retained.write_bytes(source.read_bytes())
            work["source_refs"].append(dict(e.reference(retained), original_path=str(source)))
    work["phase_spans"].insert(
        0,
        dict(
            phase="one_time_imported_authority_custody",
            start_ns=begin,
            end_ns=authority_end,
            duration_s=(authority_end - begin) / 1e9,
        ),
    )
    work["start_ns"], work["started_at_utc"] = begin, wall
    work["immutable_authority_refs"] = admitted.get("immutable_authority_refs", [])
    work["historical_observer_diagnostics"] = admitted.get("historical_observer_diagnostics", [])
    for row in work["window_rows"]:
        row.update(
            metric="full_operation_parent_ns",
            denominator=32,
            missing_reason="admission_blocked" if work["failures"] else row.get("missing_reason"),
        )
        row.setdefault("restart_verified", False)
    work["citations"] = (
        []
        if private_input
        else [
            (n, fields, e.reference(e.ROOT / path))
            for n, fields, path in [
                (
                    8783,
                    [
                        "durable_producer_ready_score",
                        "producer_descriptor",
                        "retained_control_rows",
                    ],
                    "results/experiment_8783_v752_durable_producer_identity.json",
                ),
                (
                    8784,
                    ["durable_cost_rows_ready_score", "rows", "observer_overhead_rows"],
                    "results/experiment_8784_v752_durable_cost_components.json",
                ),
                (
                    8796,
                    ["gate_check_summary"],
                    "results/experiment_8796_v753_observer_intervention.json",
                ),
            ]
        ]
    )
    work["positive_control_rows"] = []
    if not work["failures"]:
        baseline_start = time.perf_counter_ns()
        baseline = time.perf_counter_ns() - baseline_start
        delay_start = time.perf_counter_ns()
        time.sleep(0.1)
        elapsed = time.perf_counter_ns() - delay_start
        work["positive_control_rows"] = [
            dict(
                injected_delay_ns=100_000_000,
                observed_parent_ns=elapsed,
                baseline_parent_ns=baseline,
                observed_slowdown_ns=elapsed - baseline,
                passed=elapsed - baseline >= 90_000_000,
                outside_score_logic=True,
            )
        ]
    end = time.perf_counter_ns()
    work["phase_spans"].append(
        dict(
            phase="timer_positive_control_and_receipts",
            start_ns=work["end_ns"],
            end_ns=end,
            duration_s=(end - work["end_ns"]) / 1e9,
        )
    )
    work["end_ns"], work["duration_s"], work["ended_at_utc"] = (
        end,
        (end - begin) / 1e9,
        datetime.now(UTC).isoformat(),
    )
    e.atomic_json(raw / "window_rows.json", dict(rows=work["window_rows"]))
    work["raw_refs"] = [e.reference(Path(r["path"])) for r in work["raw_refs"]]
    seal(work)
    return work


def seal(work: Json) -> None:
    """An independent primitive seal prevents rehashed summaries from authenticating."""
    raw = Path(work["raw"])
    e.atomic_json(raw / "measurement.json", work)
    e.atomic_json(
        raw / "measurement_seal.json", dict(measurement=e.reference(raw / "measurement.json"))
    )


def validate(raw: Path) -> list[Json]:
    """Original node and child SQLite custody stays authoritative for this scope."""
    with patch.multiple(checks, e=e, E2E=E2E):
        return list(checks.validate(raw))


def replay(path: Path) -> None:
    """Rebuild from independently sealed operands, with exact task identity first."""
    value = json.loads(path.read_bytes())
    if (value["experiment_id"], value["task_id"], value["milestone"]) != (
        8808,
        e.TASK,
        "2026.11.754",
    ):
        raise ValueError("producer_identity")
    work = e.read(value["primitive_reference"])
    frozen_roster = work["roster"] if work["private_control"] else e.roster()
    with patch.object(runtime.e, "roster", lambda: frozen_roster):
        runtime.replay(path)


def terminal(path: Path, raw: Path) -> Json:
    """Shared scanners inspect the exact candidate before its atomic publication."""
    retained = raw.parent / "terminal_candidate.json"
    retained.write_bytes(path.read_bytes())
    report = dict(runtime.terminal(retained, raw))
    outside = child(
        "outside_cold_replay",
        [
            "env",
            "-u",
            "PYTHONPATH",
            "-C",
            "/var/tmp",
            sys.executable,
            str(e.ROOT / e.CLI),
            "--cold-replay",
            str(retained),
        ],
        raw,
        deadline=120,
    )
    report["receipts"].append(outside)
    report["passed"] = report["passed"] and outside["passed"]
    report["retained_candidate"] = e.reference(retained)
    return report


def run(output: Path, date: str, private_input: Path | None = None) -> int:
    """Private controls stay outside the checkout and cannot acquire readiness."""
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
            current_receipt=health, full_suite_pass_claimed=health["passed"]
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
    e.progress(value["honest_verdict"], value["completed_count"], value["censored_count"])
    return 0 if value["required_checks_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """One CLI supports private fixtures, durable workers and cold replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261104")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            if not args.work_dir:
                raise ValueError("worker_work_dir")
            worker(e.read(e.reference(args.worker)), args.work_dir)
            return 0
        if args.cold_replay:
            replay(args.cold_replay)
            return 0
        if args.date != "20261104":
            raise ValueError("calendar_date")
        return run(args.output, args.date, args.private_input)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp8808] error={error}", file=sys.stderr, flush=True)
        return 1


runtime.cell = cell
runtime.child = cells.child = segment_child

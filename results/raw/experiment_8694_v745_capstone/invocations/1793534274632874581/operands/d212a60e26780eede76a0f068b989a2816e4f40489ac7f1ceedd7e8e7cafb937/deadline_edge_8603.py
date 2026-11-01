"""REQ-REPORT-8603: qualify exact deadline reads without changing the learner."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import signal
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import learning_clock_8592 as old
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import progress

Json = dict[str, Any]
ROOT, stream = old.ROOT, old.stream
NAME = "experiment_8603_v739_deadline_edge_qualification"
TASK = "exp8603-deadline-edge-qualification"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [*old.TESTS, "tests/python/test_deadline_edge_8603.py"]
OWNED = [
    "python/carnot/reporting/deadline_edge_8603.py",
    "python/carnot/reporting/deadline_edge_validation_8603.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
_BUILD = old.build
PIN_8592 = "sha256:fc73c667216efe5dd26a0662667ab39b1b7a54901365dd0229b45c26b895dc69"


class DeadlineClock:
    """Keep finite test reads local so numerical timing stays on the real clock."""

    def __init__(self, values: list[float]):
        self.values = iter(values)
        self.reads = 0

    def __call__(self) -> float:
        value = next(self.values)
        self.reads += 1
        return value


def deadline_case(fixture: Json, directory: Path, elapsed: float) -> Json:
    """Execute the original runner body and retain its separate diagnostic reading."""
    clock = DeadlineClock([50.0, 50.0 + elapsed, 50.25 + elapsed])
    paths = []
    temporary = old.runner.TemporaryDirectory

    def tracked(**kwargs: Any) -> Any:
        instance = temporary(**kwargs)
        paths.append(Path(instance.name))
        return instance

    output = directory / (stream.NAME + ".json")
    actual_clock = time.monotonic
    with (
        patch.object(old.runner, "deadline_clock", clock),
        patch.object(old.runner, "TemporaryDirectory", tracked),
    ):
        exit_code = old.runner.run(output, private=True, measurement=Path(fixture["path"]))
    value = json.loads(output.read_bytes())
    failures = [
        row for row in value["gate_check_summary"] if row["field"] == "measurement_deadline"
    ]
    observed = failures[0]["observed"] if failures else None
    cleanup = bool(paths) and all(not path.exists() for path in paths)
    expired = elapsed >= 3600
    row = dict(
        unit=f"deadline_{elapsed:g}",
        predicate_elapsed_s=elapsed,
        observed_elapsed_s=observed,
        deadline_failed=bool(failures),
        clock_reads=clock.reads,
        normal_cleanup=cleanup,
        exit_code=exit_code,
        verdict_class=value["verdict_class"],
        positive_class_absent=value["verdict_class"] not in {"positive", "circular_positive"},
        output_reference=old.ref(output),
        kernel_clock_unchanged=time.monotonic is actual_clock,
        missing_reason=None,
    )
    row["passed"] = bool(
        exit_code == 0
        and cleanup
        and row["positive_class_absent"]
        and row["kernel_clock_unchanged"]
        and bool(failures) == expired
        and observed == (elapsed + 0.25 if expired else None)
        and clock.reads == (3 if expired else 2)
        and stream.replay(output)
    )
    atomic_json(directory / "case.json", row)
    return row


def prepare(raw: Path) -> Json:
    """Reuse sealed original authority while authenticating this invocation's ownership."""
    work = old.prepare(raw)
    prior_path = ROOT / "results/experiment_8592_v738_learning_clock_qualification.json"
    prior = stream.old.read(dict(path=str(prior_path), sha256=PIN_8592))
    previous = stream.old.read(prior["primitive_reference"])
    coverage = stream.old.read(previous["coverage_reference"])
    runner_coverage = coverage["files"]["python/carnot/reporting/causal_brier_runner_8582.py"]
    work["failure_reproduction"] = dict(
        passed=runner_coverage["missing_lines"] == [94],
        missing_lines=runner_coverage["missing_lines"],
        historical_primary=old.ref(prior_path),
        coverage_reference=previous["coverage_reference"],
        scope="private replay of original owned-coverage failure; no global clock patch",
    )
    work["current_source_reference"] += [old.ref(ROOT / p) for p in [*OWNED, TESTS[-1]]]
    work.update(deadline_case_rows=[], revision=old.ref(ROOT / OWNED[0]))
    work["preconditions"]["input_path_aliases"] = {
        "python/carnot/reporting/learning_clock_qualification_8592.py": "python/carnot/reporting/learning_clock_8592.py",
        "tests/python/test_causal_brier_runner_8582.py": "tests/python/test_causal_brier_stream_8582.py",
    }
    work["preconditions"]["optional_gpu_and_boards_required"] = False
    atomic_json(raw / "work.json", work)
    return work


def qualified(work: Json) -> bool:
    """Readiness requires the exact original union and every newly owned statement."""
    coverage = stream.old.read(work["coverage_reference"]) if work["coverage_reference"] else {}
    union = work["node_union"]
    rows = work["deadline_case_rows"]
    return bool(
        len(rows) == 3
        and all(row["passed"] for row in rows)
        and work["failure_reproduction"]["passed"]
        and all(row["passed"] for row in work["clock_independence_rows"])
        and old.REQUIRED <= {r["name"] for r in work["receipts"] if r["passed"]}
        and all(r["passed"] and not r.get("timed_out") for r in work["receipts"])
        and len(work["original_nodes"]) == 27
        and stream.validation.exact_union(
            work["original_nodes"], union.get("original_children", [])
        )
        and stream.validation.exact_union(
            union.get("collected", []), [[r["node_id"] for r in union.get("outcomes", [])]]
        )
        and bool(union.get("outcomes"))
        and all(r["outcome"] == "passed" for r in union["outcomes"])
        and set([*OWNED, "python/carnot/reporting/causal_brier_runner_8582.py"])
        <= set(coverage.get("files", {}))
        and all(not f["summary"]["missing_lines"] for f in coverage.get("files", {}).values())
    )


def build(work: Json, output: Path) -> Json:
    """Keep the established schema and preserve every historical scientific limit."""
    with patch.object(old, "qualified", qualified):
        value = _BUILD(work, output)
    value.update(
        experiment_id=8603,
        task_id=TASK,
        milestone="2026.10.739",
        honest_verdict=f"complete_{value['verdict_class']}_v739_deadline_edge_qualification",
        target_model=value["numerical_inference"],
        rows=work["deadline_case_rows"],
        intended_count=3,
        completed_count=len(work["deadline_case_rows"]),
        failed_count=sum(not r["passed"] for r in work["deadline_case_rows"]),
        independent_count=0,
        sample_size_budget=dict(
            unit="deterministic deadline controls", intended=3, independent_questions=0
        ),
        deadline_case_rows=work["deadline_case_rows"],
        original_node_ids=work["original_nodes"],
        executed_node_ids=[r["node_id"] for r in work["node_union"].get("outcomes", [])],
        exact_union_receipt=work["node_union"],
        measured_statement_anchors=work.get("measured_statement_anchors", {}),
        kernel_clock_independence=work["clock_independence_rows"],
        current_revision_reference=work["revision"],
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8602, 8615)
            if n != 8603
        ],
        retirement_recommendation=None
        if qualified(work)
        else "retire_exact_deadline_qualification_scope_if_terminal_owned_checks_fail",
    )
    value["raw_shard_hashes"] += [
        old.ref(Path(row["output_reference"]["path"]).parent / "case.json")
        for row in work["deadline_case_rows"]
    ]
    value["historical_quarantines"].update(
        exp8592="unchanged owned-coverage failure; original diagnostic line94 missing"
    )
    value["field_principles"].update(
        {
            key: "Bind exact deadline execution to immutable original authority; execution is not learning benefit."
            for key in value
            if key not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = old.digest(value)
    return value


def replay(path: Path) -> bool:
    """Reject repaired summaries by rebuilding from authenticated original operands."""
    try:
        value = json.loads(path.read_bytes())
        work = stream.old.read(value["primitive_reference"])
        rows = work["deadline_case_rows"]
        if rows:
            if [r["predicate_elapsed_s"] for r in rows] != [3599.0, 3600.0, 3601.0]:
                return False
            for row in rows:
                directory = Path(row["output_reference"]["path"]).parent
                if json.loads((directory / "case.json").read_bytes()) != row:
                    return False
                case = stream.old.read(row["output_reference"])
                expired = row["predicate_elapsed_s"] >= 3600
                gates = [
                    r for r in case["gate_check_summary"] if r["field"] == "measurement_deadline"
                ]
                if (
                    not row["passed"]
                    or row["observed_elapsed_s"]
                    != (row["predicate_elapsed_s"] + 0.25 if expired else None)
                    or bool(gates) != expired
                    or not stream.replay(Path(row["output_reference"]["path"]))
                ):
                    return False
        historical = work["failure_reproduction"]
        if historical["historical_primary"] != dict(
            path=str(ROOT / "results/experiment_8592_v738_learning_clock_qualification.json"),
            sha256=PIN_8592,
        ):
            return False
        original_coverage = stream.old.read(historical["coverage_reference"])
        if (
            original_coverage["files"]["python/carnot/reporting/causal_brier_runner_8582.py"][
                "missing_lines"
            ]
            != historical["missing_lines"]
        ):
            return False
        if work["original_authority"] != stream.old.read(work["source_refs"][-1])["authority"]:
            return False
        with patch.object(old, "build", build):
            return bool(old.replay(path))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(path: Path, raw: Path) -> Json:
    """Use unchanged scanner and row consumers with this producer's cold CLI."""
    with patch.object(old, "CLI", CLI):
        return old.terminal(path, raw)


def run(output: Path, run_date: str, *, work_path: Path | None = None) -> int:
    """Publish exact checked bytes after bounded CPU qualification and fresh replay."""
    from carnot.reporting.deadline_edge_validation_8603 import execute
    from scripts.adversarial_verify import verify_artifact

    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    progress("preconditions_no_model_load_MODEL_SPECS=[]_current_llm_calls=0")
    work = json.loads(work_path.read_bytes()) if work_path else prepare(raw)
    if not work_path:
        execute(work)
    work["run_date"] = run_date
    work["duration_s"] = (time.monotonic_ns() - work["started_monotonic_ns"]) / 1e9
    work["phase_spans"] = [dict(phase="qualification", duration_s=work["duration_s"])]
    raw = Path(work["raw"])
    atomic_json(raw / "work.json", work)
    preliminary = raw / "preterminal.json"
    atomic_json(preliminary, build(work, output))
    work["adversarial_findings"] = verify_artifact(preliminary)["flags"]
    atomic_json(raw / "work.json", work)
    value = build(work, output)
    publication = publish_primary(output, value, lambda p: terminal(p, raw))
    selected = reader_receipt(
        TASK,
        output.parent,
        field="learning_execution_ready_score",
        expected=value["learning_execution_ready_score"],
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(publication=publication, selected_identity=selected),
    )
    progress("normal_exit_" + value["verdict_class"], 1, 0)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Keep wall supervision real while exposing calendar dates and private controls."""
    print("[exp8603] start no_model_load MODEL_SPECS=[] current_llm_calls=0", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261017")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    try:
        datetime.strptime(args.date, "%Y%m%d")
        if len(args.date) != 8 or (
            args.private_output and args.private_output.resolve().is_relative_to(ROOT)
        ):
            return 2
    except ValueError:
        return 2
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        return int(not replay(args.cold_replay))
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(
            args.private_output or ROOT / "results" / (NAME + ".json"),
            args.date,
            work_path=args.work,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)

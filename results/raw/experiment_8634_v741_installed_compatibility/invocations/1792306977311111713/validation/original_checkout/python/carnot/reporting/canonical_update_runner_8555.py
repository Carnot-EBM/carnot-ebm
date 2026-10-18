"""REQ-VERIFY-8555: supervise real children and publish truthful counter evidence.

Each child has a deadline and durable logs. Private mode exercises publication
without recursively launching historical validation from within a unit test.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import coverage as coverage

from carnot.reporting import canonical_update_8555 as e
from carnot.reporting import canonical_update_report_8555 as report
from carnot.reporting import historical_learning_runner_8541 as resources
from carnot.reporting import objective_publication_8516 as objective
from carnot.reporting import v717_contract_runner as validation
from carnot.reporting import v733_update_accounting as old
from carnot.reporting import v733_update_fixture as fixture
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v726_feature_panel_runner import scan
from carnot.reporting.v729_contract_runner import collect

Json = dict[str, Any]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush completed and pending counts because this CPU run has no model heartbeat."""
    print(f"[exp8555] phase={phase} completed={completed} pending={pending}", flush=True)


def snapshot(work: Json, raw: Path) -> None:
    """Seal enriched receipts as well as logs before a fresh child reads the work."""
    for row in work["receipts"]:
        atomic_json(Path(row["stdout_path"]).with_suffix(".receipt.json"), row)
    atomic_json(raw / "measurement.json", work)


def restart(raw: Path) -> tuple[list[Json], bool]:
    """Use the shipped fixture in three actual processes, including a durable exit73."""
    atomic_json(raw / "configuration.json", fixture.configuration())
    argv = [
        sys.executable,
        "-u",
        str(e.ROOT / e.CLI),
        "--configuration",
        str(raw / "configuration.json"),
    ]
    receipts: list[Json] = []
    for name, extra, expected in [
        ("baseline", [], 0),
        ("restart", ["--crash"], 73),
        ("resume", [], 0),
    ]:
        directory = raw / ("baseline" if name == "baseline" else "restart")
        receipts.append(
            child(
                name,
                [*argv, "--fixture-worker", str(directory), *extra],
                raw / "logs",
                expected=expected,
                deadline=90,
            )
        )
    passed = all(r["passed"] for r in receipts)
    passed = passed and all(
        json.loads((raw / "baseline" / name).read_bytes())
        == json.loads((raw / "restart" / name).read_bytes())
        for name in ("final.json", "accounting.json")
    )
    return receipts, passed


def negative_control(value: Json, raw: Path) -> Json:
    """The final metrics representation must expose deliberately coincident distinct counters."""
    changed = deepcopy(value)
    changed.setdefault("metrics", {}).update(
        physical_coordinate_writes=12345, changed_coordinate_count=12345
    )
    path = raw / "coincident_distinct_counters.json"
    atomic_json(path, changed)
    receipts, findings = scan(path, raw / "negative_validator")
    return dict(
        passed=any(r["kind"] == "TAUTOLOGY" for r in findings),
        findings=findings,
        receipts=receipts,
        candidate=e.reference(path),
        changed_canonical_names=["physical_coordinate_writes", "changed_coordinate_count"],
        scope="negative control only; truthful measurement unchanged",
    )


def historical_shards(work: Json, raw: Path, private: Path) -> None:
    """Collect the exact original union before disjoint children with original suite authority."""
    original = e.historical.original()[1]
    collected = []
    for suite in original["authorities"]:
        nodes, receipt = collect(raw, suite)
        collected.extend(nodes)
        work["receipts"].append(receipt)
    atomic_json(
        raw / "collection.json",
        dict(
            original_node_ids=original["nodes"],
            collected_node_ids=collected,
            partitions=e.historical.partition(original["nodes"]),
        ),
    )
    if set(collected) != set(original["nodes"]) or len(collected) != len(set(collected)):
        work["failures"].append(
            e.gate(
                raw / "collection.json", "exact_original_collection", original["nodes"], collected
            )
        )
    receipts: list[Json] = []
    for index, nodes in enumerate(e.historical.partition(original["nodes"])):
        remaining = 3600 - (time.monotonic_ns() - work["start_ns"]) / 1e9
        progress(
            "historical_nodes",
            sum(len(r["collected_node_ids"]) for r in receipts),
            len(original["nodes"]) - sum(len(r["collected_node_ids"]) for r in receipts),
        )
        if remaining <= 0:
            work["failures"].append(
                e.gate(raw, "measurement_deadline_s", "<=3600", 3600 - remaining)
            )
            break
        # These original children contain no newly owned code to cover.
        with patch.dict(os.environ):
            os.environ.pop("COVERAGE_PROCESS_CONFIG", None)
            receipts.append(
                e.historical.node_child(
                    "shard_" + str(index),
                    nodes,
                    private / str(index),
                    raw / "children",
                    min(480, remaining),
                )
            )
    work["receipts"].extend(receipts)
    work["union"] = e.historical.exact_union(original["nodes"], receipts)
    work["collected_node_ids"] = collected


def verification(work: Json, raw: Path, private: Path) -> None:
    """Run only owned coverage and scoped tools; original families run as bounded pytest children."""
    adapter = SimpleNamespace(ROOT=e.ROOT, OWNED=e.OWNED, TEST=e.TESTS[0])
    with patch.object(validation, "m", adapter):
        plan = validation.manifest(private)
    for row in plan:
        row["argv"] = [a for a in row["argv"] if a not in e.TESTS]
        if row["name"] in {"owned_tests", "ruff_check", "ruff_format", "spec_coverage"}:
            row["argv"] += e.TESTS
        if row["name"] == "owned_tests":
            row["deadline"] = 480
        if row["name"] == "coverage_combine":
            row["argv"].append("--keep")
    plan = [p for p in plan if p["name"] != "private_E2E018_consumers"]
    config = private / "coverage.ini"
    config.write_text(config.read_text().replace("patch=subprocess", "patch=subprocess,_exit"))
    work["execution_manifest"] = plan
    for index, spec in enumerate(plan):
        progress("owned_validation", index, len(plan) - index)
        work["receipts"].append(
            child(
                spec["name"],
                spec["argv"],
                raw / "validation",
                expected=spec["expected"],
                deadline=spec["deadline"],
            )
        )
    if (private / "coverage.json").exists():
        atomic_json(raw / "coverage.json", json.loads((private / "coverage.json").read_bytes()))


def restore_historical(work: Json, raw: Path) -> None:
    """Authenticate earlier completed children after a controller-only interruption.

    Original assertions and authority did not change. Logs, real terminal nodes,
    command arguments, measured times and the complete disjoint union must still
    agree; a summary alone cannot authorize reuse.
    """
    original = e.historical.original()[1]
    receipts = []
    for index, nodes in enumerate(e.historical.partition(original["nodes"])):
        path = raw / "children" / f"shard_{index}.receipt.json"
        row = json.loads(path.read_bytes())
        for stream in ("stdout", "stderr"):
            ref = dict(path=row[stream + "_path"], sha256=row[stream + "_sha256"])
            if e.reference(Path(ref["path"])) != ref:
                raise ValueError("recovery_child_log_hash")
            work["sources"].append(ref)
        xml = Path(row["stdout_path"]).with_suffix(".xml")
        row.update(
            collected_node_ids=nodes,
            authority=e.historical.old.authorities(nodes)[nodes[0].split("::")[0]],
            node_elapsed_s=e.historical.node_times(xml, nodes),
            junit_reference=e.reference(xml),
        )
        work["sources"].append(row["junit_reference"])
        receipts.append(row)
    union = e.historical.exact_union(original["nodes"], receipts)
    if not union["passed"]:
        raise ValueError("recovery_original_union")
    collection = json.loads((raw / "collection.json").read_bytes())
    if set(collection["collected_node_ids"]) != set(original["nodes"]):
        raise ValueError("recovery_original_collection")
    work.update(
        union=union,
        collected_node_ids=collection["collected_node_ids"],
        recovery_reference=e.reference(raw / "collection.json"),
    )
    work["receipts"].extend(receipts)
    work["sources"].append(work["recovery_reference"])


def run(output: Path, private: Path, *, control: bool = False, resume: Path | None = None) -> int:
    """Keep failures publishable while requiring clean unchanged checks for online readiness."""
    started = time.monotonic_ns()
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    preconditions = resources.resources(private)
    preconditions.update(minimum_memory_bytes=268435456, minimum_disk_bytes=104857600)
    failures = []
    if (
        preconditions["filesystem"] in {"tmpfs", "ramfs"}
        or not all(preconditions["tools"].values())
        or preconditions["available_memory_bytes"] < 268435456
        or preconditions["free_disk_bytes"] < 104857600
    ):
        failures.append(e.gate(private, "disk_memory_tools", True, preconditions))
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess,_exit\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in e.OWNED)
        + "[report]\nexclude_lines=\n"
    )
    progress("authenticate_original_inputs")
    historical, refs, missing = e.authenticate(e.ROOT)
    failures.extend(missing)
    events, original_refs = (e.controls(), []) if control else e.history()
    atomic_json(raw / "event_rows.json", events)
    primary = objective.ROOT / "results" / (objective.NAME + ".json")
    producer = e.read(dict(path=str(primary), sha256=old.old.UPSTREAM_PIN))
    terminal = e.read(e.reference(Path(producer["terminal_validation_sidecar_path"])))
    consumer = objective.consumer(
        primary, Path(terminal["publication"]["sidecar_path"]), replay=False
    )
    progress("independent_finite_differences")
    numerical = e.numerical(private / "numerical")
    receipts, restarted = restart(raw / "restart_controls")
    work: Json = dict(
        raw=str(raw),
        private_control=control,
        events=e.reference(raw / "event_rows.json"),
        sources=[*refs, *original_refs, e.reference(primary)],
        code=[e.reference(e.ROOT / p) for p in [*e.OWNED, *e.TESTS]],
        receipts=receipts,
        failures=failures,
        numerical=numerical,
        restart_passed=restarted,
        upstream_consumer=consumer,
        historical=historical,
        historical_validation_reference=refs[0] if refs else None,
        union=dict(passed=False, reason="private controls do not qualify original tests"),
        preconditions=preconditions,
        start_ns=started,
        end_ns=time.monotonic_ns(),
        negative_control=dict(passed=False),
        findings=[],
        phases=[],
    )
    if not control:
        if resume is None:
            historical_shards(work, raw, private / "historical")
        else:
            restore_historical(work, resume)
            for failure_log in sorted(resume.glob("*controller_failure.log")):
                work["sources"].append(e.reference(failure_log))
                failure = e.gate(failure_log, "prior_owned_controller_exit", 0, 1)
                failure["owned_required_check"] = True
                work["failures"].append(failure)
        verification(work, raw, private)
    work["end_ns"] = time.monotonic_ns()
    work["phases"] = [
        dict(phase="measurement_and_validation", start_ns=started, end_ns=work["end_ns"])
    ]
    snapshot(work, raw)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, report.build(work, output))
    work["negative_control"] = negative_control(json.loads(candidate.read_bytes()), raw)
    snapshot(work, raw)
    atomic_json(candidate, report.build(work, output))
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    changed = deepcopy(json.loads(candidate.read_bytes()))
    changed["metrics"]["physical_coordinate_writes"] += 1
    changed["reproducibility_checksum"] = e.checksum(changed)
    atomic_json(raw / "rehashed.json", changed)
    for name, args, expected in [
        ("cold", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent")], 1),
        ("owned_failure", ["--deliberate-error"], 7),
        ("rehashed_tamper", ["--cold-replay", str(raw / "rehashed.json")], 1),
    ]:
        work["receipts"].append(
            child(name, [*argv, *args], raw / "controls", expected=expected, deadline=120)
        )
    snapshot(work, raw)
    atomic_json(candidate, report.build(work, output))
    checks, findings = scan(candidate, raw / "scanners")
    work["receipts"].extend(checks)
    work["findings"] = findings
    work["end_ns"] = time.monotonic_ns()
    snapshot(work, raw)
    value = report.build(work, output)

    def validate(path: Path) -> Json:
        """Failure publication records validator rejection and grants no adapter qualification."""
        checks, flags = scan(path, raw / "terminal_scanners")
        integrity = e.replay(path)
        clean = all(r["passed"] for r in checks)
        failure = (
            value["verdict_class"] in {"disqualified", "blocked"}
            and value["online_adapter_ready_score"] == 0
        )
        return dict(
            passed=integrity and (clean or failure),
            publication_qualification_passed=clean,
            integrity_passed=integrity,
            findings=flags,
            receipts=checks,
            failure_evidence_only=failure,
        )

    progress("atomic_publication")
    publication = publish_primary(output, value, validate)
    with patch.multiple(old, TASK=e.TASK, replay=e.replay):
        selected = old.stream_consumer(output, Path(publication["sidecar_path"]))
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            stream_consumer=selected,
            downstream_rejection_expected=value["online_adapter_ready_score"] == 0,
            selected_identity=reader_receipt(
                e.TASK,
                output.parent,
                field="online_adapter_ready_score",
                expected=value["online_adapter_ready_score"],
            ),
            primary_sha256=e.reference(output)["sha256"],
        ),
    )
    progress("published_" + value["verdict_class"], 1, 0)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Accept only this invocation's date and keep private fixture writes outside the checkout."""
    progress("start_no_model_load_current_LLM_calls_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261015")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--fixture-worker", type=Path)
    parser.add_argument("--crash", action="store_true")
    parser.add_argument("--resume-raw", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20261015":
        return 2
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    if args.fixture_worker:
        try:
            fixture.worker(
                json.loads(args.configuration.read_bytes()), args.fixture_worker, args.crash
            )
            return 0
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return 1
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        return 2
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8555-owned-", dir="/var/tmp") as directory:
            return run(
                output, Path(directory), control=args.private_control, resume=args.resume_raw
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)

"""REQ-VERIFY-8554: bounded measured children retain exact historical assertions."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from types import SimpleNamespace
from unittest.mock import patch

import coverage

from carnot.reporting import historical_learning_8554 as e
from carnot.reporting import historical_learning_runner_8541 as shipped
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.historical_consumer_roots_8402 import assertion_sources
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v726_feature_panel_runner import scan
from carnot.reporting.v729_contract_runner import collect

Json = dict[str, Any]


def commands(work: Json, raw: Path) -> dict[str, tuple[list[str], int]]:
    """Freeze tool identities and controls without accepting a self-authored command plan."""
    adapter = SimpleNamespace(**dict(vars(e.old), OWNED=e.OWNED, TEST=e.TEST))
    with patch.object(shipped, "e", adapter):
        result = {
            n: (argv, 0) for n, argv in shipped.verification_commands(raw, raw / "coverage.ini")
        }
    config = str(raw / "coverage.ini")
    result["owned_tests"] = (
        [
            str(e.ROOT / ".venv/bin/coverage"),
            "run",
            "--rcfile=" + config,
            "-m",
            "pytest",
            *e.old.COMMON,
            e.TEST,
            "--basetemp=" + work["preconditions"]["scratch_path"] + "/owned",
        ],
        0,
    )
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for name, args, expected in [
        ("cold", ["--cold-replay", str(raw / "preterminal.json")], 0),
        ("missing", ["--cold-replay", str(raw / "absent")], 1),
        ("owned_failure", ["--deliberate-error"], 7),
        ("rehashed_tamper", ["--cold-replay", str(raw / "rehashed.json")], 1),
    ]:
        result[name] = (argv + args, expected)
    for name, script, args in [
        ("adversarial", "adversarial_verify.py", ["--json"]),
        ("strict_rows", "verdict_row_consistency_lint.py", ["--strict"]),
    ]:
        result[name] = (
            [sys.executable, "-u", "scripts/" + script, *args, str(raw / "preterminal.json")],
            0,
        )
    for suite in work["authorities"]:
        result["collect_" + Path(suite).stem] = (
            [
                str(e.ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--collect-only",
                "-q",
                suite,
            ],
            0,
        )
    if work["nodes"]:
        result["reproduce_node"] = (
            e.node_argv(
                "reproduce_node",
                [e.INTERRUPTED],
                Path(work["preconditions"]["scratch_path"]) / "reproduction",
                raw / "reproduction",
            ),
            0,
        )
    for name, nodes in fallback_suites(work):
        result[name] = (
            e.node_argv(
                name, nodes, Path(work["preconditions"]["scratch_path"]) / name, raw / "e2e"
            ),
            0,
        )
    return result


def fallback_suites(work: Json) -> list[tuple[str, list[str]]]:
    """Required private consumers still run when an earlier timeout stops the qualification union."""
    result = []
    for number, tag in [(7891, "E2E018"), (8210, "E2E021")]:
        nodes = [n for n in work["nodes"] if str(number) in n.split("::")[0]]
        if nodes and not any(
            row["passed"] and set(nodes) <= set(row["collected_node_ids"])
            for row in work["receipts"]
        ):
            result.append(("private_" + tag, nodes))
    return result


def validate_consumers(work: Json, raw: Path, private: Path) -> None:
    """Finish independent consumer checks even when the original union cannot qualify."""
    for name, nodes in fallback_suites(work):
        work["checks"].append(e.node_child(name, nodes, private / name, raw / "e2e"))


def measure(work: Json, raw: Path, private: Path) -> None:
    """A single-node timeout ends qualification with measured pending nodes and no invented failures."""
    old, original = e.original()
    work.update(
        nodes=original["nodes"],
        authorities=original["authorities"],
        historical_model_provenance=old["historical_model_provenance"],
        assertion_sources=assertion_sources(original["nodes"]),
        upstream_failed_family_diagnostics=old["failed_family_diagnostics"],
    )
    work["sources"] = [
        e.PIN,
        old["primitive_reference"],
        *original["sources"],
        *[e.old.reference(e.ROOT / s) for s in work["authorities"]],
        *[
            dict(path=r[s + "_path"], sha256=r[s + "_sha256"])
            for r in original["receipts"]
            for s in ("stdout", "stderr")
        ],
    ]
    collected = []
    for i, suite in enumerate(work["authorities"]):
        e.old.progress("8554_collect", i, len(work["authorities"]) - i)
        nodes, receipt = collect(raw, suite)
        collected.extend(nodes)
        work["checks"].append(receipt)
    work["failures"].extend(shipped.collection_gate(work["nodes"], collected, raw))
    atomic_json(
        raw / "collection.json",
        dict(
            collected_node_ids=collected,
            original_node_ids=work["nodes"],
            partitions=e.partition(work["nodes"]),
        ),
    )
    budget_start = time.monotonic()
    reproduction = e.node_child(
        "reproduce_node", [e.INTERRUPTED], private / "reproduction", raw / "reproduction"
    )
    work["checks"].append(reproduction)
    work["historical_failure_reproduction"] = [reproduction]
    if not reproduction["passed"]:
        work["diagnostics"].append(
            dict(
                phase="isolated_reproduction",
                node_id=e.INTERRUPTED,
                actual_exit=reproduction["actual_exit"],
                timed_out=reproduction["timed_out"],
                duration_s=reproduction["duration_s"],
                diagnosis="isolated_node_timeout"
                if reproduction["timed_out"]
                else "isolated_node_failed",
                assertion_failure_inferred=False,
            )
        )
        return
    for i, group in enumerate(e.partition(work["nodes"])):
        remaining = min(
            2400 - (time.monotonic() - budget_start),
            3600 - (time.monotonic_ns() - work["start_ns"]) / 1e9,
        )
        e.old.progress(
            "8554_original_union",
            sum(len(r["node_elapsed_s"]) for r in work["receipts"]),
            len(work["nodes"]) - sum(len(r["node_elapsed_s"]) for r in work["receipts"]),
        )
        if remaining <= 0:
            work["failures"].append(
                e.report.gate(str(raw), "union_budget_s", 2400, time.monotonic() - budget_start)
            )
            break
        row = e.node_child(
            "shard_" + str(i),
            group,
            private / ("shard_" + str(i)),
            raw / "children",
            min(480, remaining),
        )
        work["receipts"].append(row)
        if row["timed_out"]:
            work["diagnostics"].append(
                dict(
                    child=row["name"],
                    pending_node_ids=[n for n in group if n not in row["node_elapsed_s"]],
                    timed_out=True,
                    duration_s=row["duration_s"],
                    diagnosis="single_node_timeout" if len(group) == 1 else "bounded_shard_timeout",
                    assertion_failure_inferred=False,
                )
            )
            break


def run(output: Path, private: Path, *, control: bool = False) -> int:
    """Perform real work, record required checks, then expose only validated terminal bytes."""
    start = time.monotonic_ns()
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    preconditions = shipped.resources(private)
    preconditions.update(union_budget_s=2400, heartbeat_s=30)
    work: Json = dict(
        nodes=[],
        receipts=[],
        checks=[],
        failures=[],
        sources=[],
        authorities={},
        preconditions=preconditions,
        start_ns=start,
        end_ns=start,
        historical_failure_reproduction=[],
        coverage=None,
        diagnostics=[],
    )
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\ncore=sysmon\nparallel=true\npatch=subprocess,_exit\ndata_file="
        + str(raw / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in e.OWNED)
    )
    tracer = coverage.Coverage.current()
    if tracer is None:
        tracer = coverage.Coverage(config_file=str(config))
        tracer.start()
    if (
        preconditions["filesystem"] in ("tmpfs", "ramfs", "unknown")
        or not all(preconditions["tools"].values())
        or min(preconditions["available_memory_bytes"], preconditions["free_disk_bytes"])
        < 536870912
    ):
        work["failures"].append(
            e.report.gate(str(private), "disk_memory_tools", True, preconditions)
        )
    if not control and not work["failures"]:
        try:
            measure(work, raw, private)
        except (OSError, ValueError, KeyError) as error:
            work["failures"].append(
                e.report.gate(
                    e.PIN["path"], "original_inputs", "authenticated", str(error), e.PIN["sha256"]
                )
            )
    if not control:
        validate_consumers(work, raw, private)
        argv, expected = commands(work, raw)["owned_tests"]
        work["checks"].append(
            shipped.invoke("owned_tests", argv, raw / "validation", expected=expected)
        )
        tracer.save()
        for name, (argv, expected) in commands(work, raw).items():
            if name.startswith(("coverage_", "ruff_", "strict_mypy", "spec_coverage")):
                work["checks"].append(
                    shipped.invoke(name, argv, raw / "validation", deadline=120, expected=expected)
                )
        if (raw / "coverage.json").exists():
            work["coverage"] = e.old.reference(raw / "coverage.json")
    work["end_ns"] = time.monotonic_ns()
    work["phase_spans"] = [
        dict(phase="measured_nodes_and_validation", start_ns=start, end_ns=work["end_ns"])
    ]
    atomic_json(raw / "measurement.json", work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, e.build(work, raw, output))
    probe = raw / "consumer/results" / (e.NAME + ".json")
    atomic_json(probe, json.loads(candidate.read_bytes()))
    work["accounting_consumer"] = reader_receipt(
        e.TASK, probe.parent, field="historical_learning_validation_ready_score", expected=0
    )
    changed = deepcopy(json.loads(candidate.read_bytes()))
    changed["historical_learning_validation_ready_score"] = 999
    changed["reproducibility_checksum"] = e.report.checksum(changed)
    atomic_json(raw / "rehashed.json", changed)
    for name in ("cold", "missing", "owned_failure", "rehashed_tamper"):
        argv, expected = commands(work, raw)[name]
        work["checks"].append(
            shipped.invoke(name, argv, raw / "controls", deadline=60, expected=expected)
        )
    atomic_json(raw / "measurement.json", work)
    atomic_json(candidate, e.build(work, raw, output))
    checks, findings = scan(candidate, raw / "scanners")
    work["checks"].extend(checks)
    work["adversarial_findings"] = findings
    work["end_ns"] = time.monotonic_ns()
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, raw, output)

    def validate(path: Path) -> Json:
        """Final scanners retain every finding even when honest disqualification permits publication."""
        receipts, flags = scan(path, raw / "terminal_scanners")
        return dict(
            passed=e.replay(path)
            and (
                all(r["passed"] for r in receipts)
                or value["verdict_class"] in ("disqualified", "blocked")
            ),
            validation_receipts=receipts,
            adversarial_findings=flags,
        )

    publication = publish_primary(output, value, validate)
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="historical_learning_validation_ready_score",
        expected=value["historical_learning_validation_ready_score"],
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication, consumer=consumer, primary_sha256=publication["primary_sha256"]
        ),
    )
    e.old.progress(
        "8554_published_" + value["verdict_class"],
        value["completed_count"],
        value["censored_count"],
    )
    tracer.save()
    return int(not e.replay(output) or not consumer["passed"])


def main(argv: list[str] | None = None) -> int:
    """Accept the sealed date and keep private rejection controls away from primary results."""
    e.old.progress("8554_start_no_model_load_current_LLM_calls_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261015"], default="20261015")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        return int(not e.replay(args.cold_replay))
    if args.private_control and args.output.absolute().is_relative_to(e.ROOT):
        parser.error("private control output must be outside the repository")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8554-", dir="/var/tmp") as directory:
            return run(args.output.absolute(), Path(directory), control=args.private_control)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)

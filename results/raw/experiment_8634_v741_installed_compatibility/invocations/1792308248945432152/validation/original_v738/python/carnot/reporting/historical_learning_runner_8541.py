"""REQ-VERIFY-8541: bounded private children preserve the exact original union."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import historical_learning_8541 as e
from carnot.reporting import historical_learning_report_8541 as report
from carnot.reporting.block_checkpoint_qualification_8433 import filesystem_type
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v726_feature_panel_runner import scan
from carnot.reporting.v729_contract_runner import collect

Json = dict[str, Any]


def collection_gate(nodes: list[str], collected: list[str], raw: Path) -> list[Json]:
    """Unexpected or duplicate collected nodes must remain a failed gate."""
    if set(collected) != set(nodes) or len(collected) != len(set(collected)):
        return [report.gate(str(raw), "original_node_union", nodes, collected)]
    return []


def families(work: Json, argv: list[str], raw: Path, private: Path) -> None:
    """The measurement cap stops new children while preserving every pending node."""
    for i, suite in enumerate(work["authorities"]):
        group = [n for n in work["nodes"] if n.split("::")[0] == suite]
        remaining = 3600 - (time.monotonic_ns() - work["start_ns"]) / 1e9
        e.progress("original_family", len(work["receipts"]), len(work["authorities"]) - i)
        if remaining <= 0:
            work["failures"].append(
                report.gate(str(raw), "measurement_deadline", "<=3600", 3600 - remaining)
            )
            break
        receipt = invoke(
            "family_" + str(i),
            argv
            + [
                "--suite",
                suite,
                *e.COMMON,
                *group,
                "--basetemp=" + str(private / ("family-" + str(i))),
            ],
            raw / "families",
            deadline=min(480, remaining),
        )
        receipt["collected_node_ids"] = group
        work["receipts"].append(receipt)


def verification_commands(raw: Path, config: Path) -> list[tuple[str, list[str]]]:
    """One command plan lets replay check every required tool identity."""
    commands: list[tuple[str, list[str]]] = [
        ("coverage_combine", ["coverage", "combine", "--keep"]),
        ("coverage_report", ["coverage", "report", "--fail-under=100"]),
        ("coverage_json", ["coverage", "json", "-o", str(raw / "coverage.json")]),
        ("ruff_check", ["ruff", "check", *e.OWNED, e.TEST]),
        ("ruff_format", ["ruff", "format", "--check", *e.OWNED, e.TEST]),
        ("strict_mypy", ["mypy", "--strict", "--follow-imports=silent", *e.OWNED]),
        ("spec_coverage", ["python", "scripts/check_spec_coverage.py", "--files", e.TEST]),
    ]
    result = []
    for name, command in commands:
        tool, *arguments = command
        if tool == "coverage":
            arguments.insert(1, "--rcfile=" + str(config))
        result.append((name, [str(e.ROOT / ".venv/bin" / tool), *arguments]))
    return result


def verification(work: Json, raw: Path, config: Path, *, collect_current: bool = True) -> None:
    """Incomplete measured coverage stays a failed check beside lint and type receipts."""
    import coverage

    current = coverage.Coverage.current()
    if current is not None and collect_current:
        current.save()
        data = Path(current.config.data_file)
        for shard in data.parent.glob(data.name + ".*"):
            if shard.parent != raw:
                shutil.copyfile(shard, raw / shard.name)
    for name, command in verification_commands(raw, config):
        work["checks"].append(invoke(name, command, raw / "validation", deadline=120))
    if (raw / "coverage.json").exists():
        work["coverage"] = e.reference(raw / "coverage.json")


def resources(private: Path) -> Json:
    """Check real disk, memory and tools before any original fixture runs."""
    private.mkdir(parents=True, exist_ok=True)
    probe = private / "write_probe"
    probe.write_text("disk-backed scratch")
    with probe.open("rb") as stream:
        os.fsync(stream.fileno())
    probe.unlink()
    memory = (
        int(
            next(
                s.split()[1]
                for s in Path("/proc/meminfo").read_text().splitlines()
                if s.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    return dict(
        scratch_path=str(private),
        filesystem=filesystem_type(private),
        writable=True,
        available_memory_bytes=memory,
        free_disk_bytes=shutil.disk_usage(private).free,
        tools={
            n: os.access(e.ROOT / ".venv/bin" / n, os.X_OK)
            for n in ("python", "pytest", "coverage", "ruff", "mypy")
        },
        child_deadline_s=480,
        heartbeat_s=30,
        measurement_cap_s=3600,
        validation_reserve_s=900,
        hard_cap_s=4800,
        optional_cuda_and_board="not_required_for_independent_CPU_work",
    )


def invoke(
    name: str, args: list[str], raw: Path, *, deadline: float = 480, expected: int = 0
) -> Json:
    """Keep every command bounded with real exits and durable stream hashes."""
    return child(name, args, raw, deadline=deadline, expected=expected, heartbeat=30)


def run(output: Path, private: Path, *, control: bool = False) -> int:
    """Classify all owned failures before exposing one checked reader-visible artifact."""
    start = time.monotonic_ns()
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    preconditions = resources(private)
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
    )
    if (
        preconditions["filesystem"] in ("tmpfs", "ramfs", "unknown")
        or not all(preconditions["tools"].values())
        or min(preconditions["available_memory_bytes"], preconditions["free_disk_bytes"])
        < 536870912
    ):
        work["failures"].append(report.gate(str(private), "disk_memory_tools", True, preconditions))
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\ncore=sysmon\nparallel=true\npatch=subprocess,_exit\ndata_file="
        + str(raw / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in e.OWNED)
    )
    old: Json = {}
    nodes: list[str] = []
    if not control and not work["failures"]:
        try:
            old, nodes = e.original_union()
            work["authorities"] = e.authorities(nodes)
        except (OSError, ValueError, KeyError) as error:
            work["failures"].append(
                report.gate(str(e.ROOT / "results"), "original_inputs", "authenticated", str(error))
            )
    if not control and not work["failures"]:
        work.update(
            nodes=nodes,
            authorities=e.authorities(nodes),
            sources=[e.primary(8530), old["primitive_reference"]],
        )
        work["sources"] += [e.reference(e.ROOT / suite) for suite in work["authorities"]]
        for family in work["authorities"].values():
            if family["producer"]:
                work["sources"] += [family["producer"], family["work"]]
                work["sources"] += [
                    dict(path=r["snapshot_path"], sha256=r["sha256"]) for r in family["refs"]
                ]
        for r in old["validation_receipts"]:
            if r["name"].startswith("historical_shard_") and not r["passed"]:
                work["historical_failure_reproduction"].append(r)
                work["sources"] += [
                    dict(path=r[s + "_path"], sha256=r[s + "_sha256"]) for s in ("stdout", "stderr")
                ]
                shutil.copyfile(r["stdout_path"], private / (r["name"] + ".original.stdout"))
        e.progress("reproduce_original_reporting_failure")
        suite = "tests/python/test_objective_reporting_8461.py"
        bootstrap = "import sys;from carnot.reporting.objective_publication_runner_8516 import historical;raise SystemExit(historical(sys.argv[1:]))"
        reproduction = invoke(
            "original_failure_reproduction",
            [
                sys.executable,
                "-u",
                "-c",
                bootstrap,
                *e.COMMON,
                suite + "::test_qualified_primitive_replay",
                suite + "::test_rehashed_primitive_and_receipt_controls[authority_object]",
                "--basetemp=" + str(private / "reproduce"),
            ],
            raw / "reproduction",
            deadline=180,
            expected=1,
        )
        work["checks"].append(reproduction)
        e.progress("representative_original_fixture")
        representative = invoke(
            "representative",
            argv
            + [
                "--suite",
                suite,
                *e.COMMON,
                suite + "::test_qualified_primitive_replay",
                "--basetemp=" + str(private / "representative"),
            ],
            raw / "prototype",
        )
        work["checks"].append(representative)
        collected = []
        for i, suite in enumerate(work["authorities"]):
            e.progress("collect_original_nodes", i, len(work["authorities"]) - i)
            actual, receipt = collect(raw, suite)
            collected.extend(actual)
            work["checks"].append(receipt)
        work["failures"].extend(collection_gate(nodes, collected, raw))
        atomic_json(
            raw / "collection.json", dict(recorded_node_ids=nodes, collected_node_ids=collected)
        )
        families(work, argv, raw, private)
        work["historical_model_provenance"] = old["historical_model_provenance"]
        owned = invoke(
            "owned_tests",
            [
                str(e.ROOT / ".venv/bin/coverage"),
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                *e.COMMON,
                e.TEST,
                "--basetemp=" + str(private / "owned"),
            ],
            raw / "validation",
        )
        work["checks"].append(owned)
        verification(work, raw, config)
    work["end_ns"] = time.monotonic_ns()
    work["phase_spans"] = [
        dict(phase="original_families_and_validation", start_ns=start, end_ns=work["end_ns"])
    ]
    atomic_json(raw / "measurement.json", work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, report.build(work, raw, output))
    for name, args, expected in [
        ("cold", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent")], 1),
        ("owned_failure", ["--deliberate-error"], 7),
    ]:
        work["checks"].append(
            invoke(name, argv + args, raw / "controls", deadline=60, expected=expected)
        )
    changed = deepcopy(json.loads(candidate.read_bytes()))
    changed["historical_learning_validation_ready_score"] = 999
    changed["reproducibility_checksum"] = report.checksum(changed)
    atomic_json(raw / "rehashed.json", changed)
    work["checks"].append(
        invoke(
            "rehashed_tamper",
            argv + ["--cold-replay", str(raw / "rehashed.json")],
            raw / "controls",
            deadline=60,
            expected=1,
        )
    )
    atomic_json(raw / "measurement.json", work)
    atomic_json(candidate, report.build(work, raw, output))
    checks, findings = scan(candidate, raw / "scanners")
    work["checks"].extend(checks)
    work["adversarial_findings"] = findings
    work["end_ns"] = time.monotonic_ns()
    atomic_json(raw / "measurement.json", work)
    value = report.build(work, raw, output)

    def validate(path: Path) -> Json:
        """Scanners still run on final bytes; honest disqualification remains publishable."""
        terminal, flags = scan(path, raw / "terminal_scanners")
        return dict(
            passed=report.replay(path)
            and (
                all(r["passed"] for r in terminal)
                or value["verdict_class"] in ("disqualified", "blocked")
            ),
            validation_receipts=terminal,
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
    e.progress(
        "published_" + value["verdict_class"],
        value["completed_count"],
        value["intended_count"] - value["completed_count"],
    )
    return int(not report.replay(output) or not consumer["passed"])


def main(argv: list[str] | None = None) -> int:
    """Keep historical worker modes separate from the dated publication command."""
    e.progress("start_no_model_load_current_LLM_calls_0")
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] in ("--suite", "--exec"):
        return e.worker(arguments[1], arguments[2:], script=arguments[0] == "--exec")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261015"], default="20261015")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(arguments)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        return int(not report.replay(args.cold_replay))
    if args.private_control and args.output.absolute().is_relative_to(e.ROOT):
        parser.error("private control output must be outside the repository")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8541-owned-", dir="/var/tmp") as directory:
            return run(args.output.absolute(), Path(directory), control=args.private_control)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)


def required_commands(work: Json, raw: Path) -> dict[str, tuple[list[str], int]]:
    """Repaired receipts cannot replace a required check with an unrelated passing process."""
    private = Path(work["preconditions"].get("scratch_path", "/var/tmp/absent"))
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    suite = "tests/python/test_objective_reporting_8461.py"
    bootstrap = "import sys;from carnot.reporting.objective_publication_runner_8516 import historical;raise SystemExit(historical(sys.argv[1:]))"
    result = {
        name: (command, 0) for name, command in verification_commands(raw, raw / "coverage.ini")
    }
    result["original_failure_reproduction"] = (
        [
            sys.executable,
            "-u",
            "-c",
            bootstrap,
            *e.COMMON,
            suite + "::test_qualified_primitive_replay",
            suite + "::test_rehashed_primitive_and_receipt_controls[authority_object]",
            "--basetemp=" + str(private / "reproduce"),
        ],
        1,
    )
    result["representative"] = (
        argv
        + [
            "--suite",
            suite,
            *e.COMMON,
            suite + "::test_qualified_primitive_replay",
            "--basetemp=" + str(private / "representative"),
        ],
        0,
    )
    result["owned_tests"] = (
        [
            str(e.ROOT / ".venv/bin/coverage"),
            "run",
            "--rcfile=" + str(raw / "coverage.ini"),
            "-m",
            "pytest",
            *e.COMMON,
            e.TEST,
            "--basetemp=" + str(private / "owned"),
        ],
        0,
    )
    for suite in dict.fromkeys(n.split("::")[0] for n in work["nodes"]):
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
    for name, args, expected in [
        ("cold", ["--cold-replay", str(raw / "preterminal.json")], 0),
        ("missing", ["--cold-replay", str(raw / "absent")], 1),
        ("owned_failure", ["--deliberate-error"], 7),
        ("rehashed_tamper", ["--cold-replay", str(raw / "rehashed.json")], 1),
    ]:
        result[name] = (argv + args, expected)
    result["adversarial"] = (
        [
            sys.executable,
            "-u",
            "scripts/adversarial_verify.py",
            "--json",
            str(raw / "preterminal.json"),
        ],
        0,
    )
    result["strict_rows"] = (
        [
            sys.executable,
            "-u",
            "scripts/verdict_row_consistency_lint.py",
            "--strict",
            str(raw / "preterminal.json"),
        ],
        0,
    )
    return result

"""REQ-VERIFY-8695: bounded original producers and independent terminal publication."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
import os
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from coverage import Coverage

from carnot.reporting import v746_contract_authority as a
from carnot.reporting import v746_contract_report as report
from carnot.reporting import v746_retained_evidence as e
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
CONSUMERS = [
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]


def private_output(path: Path) -> None:
    """Reject both public lexical paths and aliases before creating private evidence."""
    if path.absolute().is_relative_to(a.ROOT) or path.resolve().is_relative_to(a.ROOT.resolve()):
        raise ValueError("private outputs must remain outside the checkout")


def collect(private: Path, raw: Path) -> tuple[list[Json], Json]:
    """Collect exactly once, then freeze disjoint producers, controls, owned tests and consumers."""
    suites = a.original.TESTS + a.TESTS + CONSUMERS
    receipt = child(
        "collect_once",
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "--collect-only",
            "-q",
            *suites,
        ],
        raw / "collection",
        deadline=60,
    )
    nodes = [
        line
        for line in Path(receipt["stdout_path"]).read_text().splitlines()
        if any(line.startswith(s + "::") for s in suites)
    ]
    if not receipt["passed"] or len(nodes) != len(set(nodes)):
        raise ValueError("single_exact_collection_failed")
    plan = []
    control = a.TESTS[0] + "::test_historical_failure_and_original_rejection"
    for index, suite in enumerate(suites + ["original_rejection_control"]):
        group = (
            [control]
            if suite == "original_rejection_control"
            else [n for n in nodes if n.startswith(suite + "::") and n != control]
        )
        if not group:
            raise ValueError("missing_required_nodes:" + suite)
        role = (
            "producer_control"
            if suite == "original_rejection_control"
            else "producer"
            if suite in a.original.TESTS
            else "owned"
            if suite in a.TESTS
            else "consumer"
        )
        scratch = private / str(index)
        scratch.mkdir()
        config = raw / "configs" / f"{index}.ini"
        config.parent.mkdir(exist_ok=True)
        config.write_text(
            "[run]\nparallel=true\npatch=subprocess\ndata_file="
            + str(scratch / ".coverage")
            + "\n[report]\nexclude_lines=\n"
        )
        argv = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(config),
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "-p",
            "carnot.reporting.v745_private_inputs",
            "-q",
            *group,
            "--basetemp=" + str(scratch / "tmp"),
            "--junitxml=" + str(scratch / "junit.xml"),
        ]
        plan.append(
            dict(
                name=f"{role}_{index}",
                argv=argv,
                nodes=group,
                role=role,
                scratch=str(scratch),
                config=e.reference(config),
                deadline=900,
            )
        )
    e.atomic_json(raw / "collection_inventory.json", dict(node_ids=nodes, receipt=receipt))
    return plan, receipt


def source_hashes() -> list[Json]:
    """Freeze transitive Python source and test configuration, including subprocess entry points."""
    paths = set((a.ROOT / "python/carnot").rglob("*.py"))
    paths.update((a.ROOT / "scripts").rglob("*.py"))
    paths.update(a.ROOT / p for p in a.TESTS + a.original.TESTS + CONSUMERS)
    paths.update((a.ROOT / "tests").glob("**/conftest.py"))
    paths.update(
        a.ROOT / p
        for p in [
            "conftest.py",
            "pyproject.toml",
            "ops/exclusion_manifest.yaml",
            a.DESIGN,
            a.ACTIVE,
            a.STAGED,
        ]
    )
    return [e.reference(p) for p in sorted(paths) if p.is_file()]


def execute(
    plan: list[Json], raw: Path, sources: list[Json], stop: float
) -> tuple[list[Json], list[Json]]:
    """Retain each child before another can run; bounded timeouts never count as completion."""
    receipts, retained = [], []
    for i, spec in enumerate(plan):
        a.progress("child", i, len(plan) - i)
        receipt = child(
            spec["name"],
            spec["argv"],
            raw / "checks",
            deadline=max(0.001, min(spec["deadline"], stop - time.monotonic())),
            heartbeat=30,
        )
        receipts.append(receipt)
        retained.append(
            e.retain(
                receipt,
                spec["nodes"],
                Path(spec["scratch"]),
                raw / "children" / spec["name"],
                sources,
                spec["role"],
            )
        )
    return receipts, retained


def coverage_summary(rows: list[Json], root: Path) -> Json:
    """Measure owned statements from actual SQLite while keeping consumer data separate."""
    combined = e.combine(rows, root)
    cov = Coverage(data_file=combined["combined_reference"]["path"], config_file=False)
    cov.load()
    files = []
    for path in a.OWNED:
        _, statements, _, missing, _ = cov.analysis2(str(a.ROOT / path))
        files.append(dict(path=path, statements=len(statements), missing=missing))
    total = sum(f["statements"] for f in files)
    missed = sum(len(f["missing"]) for f in files)
    return dict(
        complete=total > 0 and missed == 0,
        total_statements=total,
        missing_statements=missed,
        percent=100 * (total - missed) / total,
        files=files,
        combined=combined,
    )


def qualify_producer(plan: list[Json], retained: list[Json], history: Json) -> Json:
    """Keep original execution authority independent from owned and consumer outcomes."""
    pairs = [
        (s, r) for s, r in zip(plan, retained) if s["role"] in {"producer", "producer_control"}
    ]
    rows = [r for _, r in pairs]
    expected = [
        n for s in plan if s["role"] in {"producer", "producer_control"} for n in s["nodes"]
    ]
    aggregate = e.aggregate(rows, expected)
    nodes = [n for s in plan if s["role"] == "producer" for n in s["nodes"]]
    binding = bool(
        aggregate["passed"]
        and all(
            e.load(r)["node_ids"] == s["nodes"]
            and e.load(r)["role"] == s["role"]
            and e.load(r)["command_receipt"]["argv"] == s["argv"]
            for s, r in pairs
        )
    )
    rejection = bool(aggregate["passed"] and e.rejection_executed(rows))
    passed = bool(
        binding
        and len(nodes) == len(set(nodes)) == 77
        and nodes == history["original_node_ids"]
        and all(
            Path(r["path"]).is_file() and e.sha256_file(Path(r["path"])) == r["sha256"]
            for r in history["original_test_source_hashes"]
        )
        and rejection
    )
    return dict(
        passed=passed,
        original_node_count=len(nodes),
        original_node_ids=nodes,
        original_rejection_line=113,
        original_rejection_line_executed=rejection,
        aggregate=aggregate,
        command_binding=binding,
    )


def replay(path: Path) -> bool:
    """Reconstruct final bytes from independently sealed primitives in a fresh process."""
    try:
        value = json.loads(path.read_bytes())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        seal = json.loads((raw / "independent_seal.json").read_bytes())
        if (
            seal["primitive_reference"] != value["primitive_reference"]
            or seal["validation_reference"] != value["validation_reference"]
        ):
            return False
        work, validation = (
            e.load(value["primitive_reference"]),
            e.load(value["validation_reference"]),
        )
        for ref in (
            work["source_hashes"]
            + work["methods_manifest"]["references"]
            + validation["raw_references"]
        ):
            if e.sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        if (
            a.authority(Path(work["root"])) != work["contract"]
            or a.methods() != work["methods_manifest"]
        ):
            return False
        if validation["retained_child_rows"]:
            expected = [n for s in work["plan"] for n in s["nodes"]]
            aggregate = e.aggregate(validation["retained_child_rows"], expected)
            if aggregate != validation["all_child_aggregate"]:
                return False
            if (
                qualify_producer(
                    work["plan"], validation["retained_child_rows"], work["historical_failure"]
                )
                != validation["producer_qualification"]
            ):
                return False
        return bool(
            value
            == report.build(
                work, value["validation_reference"], raw, Path(value["publication_output"])
            )
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def publish(value: Json, output: Path, raw: Path) -> None:
    """Validate the exact locked candidate with unchanged scanners before atomic exposure."""
    checks: list[Json] = []

    def validate(candidate: Path) -> Json:
        commands = [
            ("terminal_cold", [a.CLI, "--cold-replay", str(candidate)]),
            ("adversarial", ["scripts/adversarial_verify.py", "--json", str(candidate)]),
            (
                "strict_rows",
                ["scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
            ),
        ]
        for name, args in commands:
            receipt = child(name, [sys.executable, "-u", *args], raw / "terminal", deadline=120)
            if name == "adversarial":
                receipt["adversarial_findings"] = json.loads(
                    Path(receipt["stdout_path"]).read_bytes()
                ).get("reports", [])
            checks.append(receipt)
        return dict(passed=all(r["passed"] for r in checks), checks=checks)

    publication = publish_primary(output, value, validate)
    e.atomic_json(
        raw / "terminal_validation.json",
        dict(
            publication=publication,
            checks=checks,
            primary_path=publication["primary_path"],
            primary_sha256=publication["primary_sha256"],
            normal_process_completion=True,
        ),
    )
    a.progress("published", 1, 0)


def preconditions(private: Path) -> Json:
    """Check real tools, available memory and writable disk-backed private scratch."""
    memory = {
        line.split(":")[0]: int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
    }
    mounts = [line.split() for line in Path("/proc/mounts").read_text().splitlines()]
    mount = max((r for r in mounts if private.is_relative_to(Path(r[1]))), key=lambda r: len(r[1]))
    probe = private / "write_probe"
    probe.write_bytes(b"disk-backed scratch")
    probe.unlink()
    tools = [
        e.reference(a.ROOT / ".venv/bin/python"),
        e.reference(a.ROOT / ".venv/bin/ruff"),
        e.reference(a.ROOT / ".venv/bin/mypy"),
    ]
    return dict(
        tools=tools,
        free_memory_kib=memory["MemAvailable"],
        free_disk_bytes=shutil.disk_usage(private).free,
        scratch=str(private),
        filesystem=mount[2],
        disk_backed=mount[2] not in {"tmpfs", "ramfs"},
        writable=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Own one bounded invocation; private controls use the same report and publication paths."""
    a.progress("start_no_model_load_calls_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261101"], default="20261101")
    parser.add_argument("--root", type=Path, default=a.ROOT)
    parser.add_argument("--output", type=Path, default=a.ROOT / "results" / (a.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--failed-child", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = replay(args.cold_replay)
        a.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    private_mode = args.private_control or args.failed_child
    if private_mode:
        try:
            private_output(output)
        except ValueError as error:
            parser.error(str(error))
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8695-", dir="/var/tmp") as directory:
            private = Path(directory)
            start, wall = time.monotonic_ns(), time.time_ns()
            raw = output.parent / "raw" / a.NAME / "invocations" / str(wall)
            raw.mkdir(parents=True)
            a.progress("preconditions")
            pre = preconditions(private)
            plan, collected = ([], None) if private_mode else collect(private, raw)
            sources = source_hashes() + [s["config"] for s in plan] + pre["tools"]
            work: Json = dict(
                root=str(args.root.absolute()),
                run_date=args.date,
                invocation_id=str(wall),
                started_monotonic_ns=start,
                started_at_utc=datetime.fromtimestamp(wall / 1e9, UTC).isoformat(),
                private_control=private_mode,
                plan=plan,
                source_hashes=sources,
                preconditions=pre,
                precondition_failures=[],
                contract=a.authority(args.root),
                methods_manifest=a.methods(),
                authority_mutation_rows=a.mutations(),
                historical_failure=e.historical_failure(),
                original_node_ids=[n for s in plan if s["role"] == "producer" for n in s["nodes"]],
            )
            if (
                not pre["disk_backed"]
                or pre["free_memory_kib"] < 131072
                or pre["free_disk_bytes"] < 67108864
            ):
                work["precondition_failures"].append(a.gate(private, "cpu_resources", True, pre))
            measured_end = time.monotonic_ns()
            work["phase_spans"] = [
                dict(
                    phase="preconditions_and_registration",
                    start_monotonic_ns=start,
                    end_monotonic_ns=measured_end,
                    duration_s=(measured_end - start) / 1e9,
                )
            ]
            e.atomic_json(raw / "measurement.json", work)
            a.progress("source_sealed", len(sources), 0)
            receipts, retained = execute(
                plan,
                raw,
                [e.reference(a.ROOT / p) for p in a.OWNED + a.TESTS + a.original.TESTS]
                + [e.reference(e.REJECTION_SOURCE)],
                start / 1e9 + 3600,
            )
            if collected is not None:
                receipts.insert(0, collected)
            if private_mode:
                receipts.append(
                    child(
                        "private_child",
                        [
                            sys.executable,
                            "-c",
                            "raise SystemExit(3)"
                            if args.failed_child
                            else "print('private CPU control')",
                        ],
                        raw / "checks",
                        deadline=10,
                    )
                )
            a.progress("qualify_retained_children", len(retained), 0)
            consumers = [r for r in retained if e.load(r)["role"] == "consumer"]
            expected = [n for s in plan for n in s["nodes"]]
            union = e.aggregate(retained, expected)
            qualification = qualify_producer(plan, retained, work["historical_failure"])
            exact = qualification["passed"]
            owned = [r for r in retained if e.load(r)["role"] != "consumer"]
            coverage = (
                coverage_summary(owned, raw / "producer_combine")
                if owned and union["passed"]
                else dict(complete=False, files=[], percent=0)
            )
            if consumers and union["passed"]:
                e.combine(consumers, raw / "consumer_combine")
            a.progress("owned_validation")
            if not private_mode:
                commands = [
                    ("ruff_check", [".venv/bin/ruff", "check", *a.OWNED, *a.TESTS]),
                    ("ruff_format", [".venv/bin/ruff", "format", "--check", *a.OWNED, *a.TESTS]),
                    (
                        "strict_mypy",
                        [".venv/bin/mypy", "--strict", "--follow-imports=silent", *a.OWNED],
                    ),
                    (
                        "spec_coverage",
                        [sys.executable, "scripts/check_spec_coverage.py", "--files", *a.TESTS],
                    ),
                ]
                for name, command in commands:
                    receipts.append(child(name, command, raw / "checks", deadline=120))
                for name, passed in [
                    ("retained_exact_union", exact),
                    ("owned_statement_coverage", coverage["complete"]),
                    (
                        "source_seal",
                        all(e.sha256_file(Path(r["path"])) == r["sha256"] for r in sources),
                    ),
                ]:
                    receipts.append(
                        dict(
                            name=name,
                            passed=passed,
                            expected_exit=0,
                            actual_exit=int(not passed),
                            stdout_path=str(raw / "measurement.json"),
                        )
                    )
            end = time.monotonic_ns()
            raw_refs = retained + [
                ref for row in retained for ref in e.load(row)["sqlite_references"]
            ]
            for r in receipts:
                for stream in ["stdout", "stderr"]:
                    if stream + "_sha256" in r:
                        raw_refs.append(
                            dict(path=r[stream + "_path"], sha256=r[stream + "_sha256"])
                        )
            validation = dict(
                receipts=receipts,
                retained_child_rows=retained,
                producer_qualification=qualification,
                all_child_aggregate=union,
                owned_coverage=coverage,
                raw_references=raw_refs,
                ended_monotonic_ns=end,
                ended_wall_ns=time.time_ns(),
                phase_spans=[
                    dict(
                        phase="bounded_execution_and_validation",
                        start_monotonic_ns=measured_end,
                        end_monotonic_ns=end,
                        duration_s=(end - measured_end) / 1e9,
                    )
                ],
            )
            e.atomic_json(raw / "validation.json", validation)
            validation_ref = e.reference(raw / "validation.json")
            e.atomic_json(
                raw / "independent_seal.json",
                dict(
                    primitive_reference=e.reference(raw / "measurement.json"),
                    validation_reference=validation_ref,
                ),
            )
            value = report.build(work, validation_ref, raw, output)
            publish(value, output, raw)
            return 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)

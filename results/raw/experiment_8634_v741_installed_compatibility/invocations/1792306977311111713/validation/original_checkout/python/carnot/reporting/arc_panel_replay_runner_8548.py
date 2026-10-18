"""REQ-VERIFY-8548: bound actual children and preserve their terminal evidence.

Private controls avoid recursive validation. The real invocation runs original
assertions and measures its own child statements before publishing any readiness.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

import coverage
import yaml

from carnot.reporting import arc_panel_replay_8548 as e
from carnot.reporting import arc_capability_fixtures_8536 as fixtures
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.historical_learning_runner_8541 import resources
from carnot.reporting.historical_consumer_roots_8402 import inject
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
HISTORICAL = [
    *e.panel.TESTS,
    *e.panel.qualification.TESTS,
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov"]


def invoke(
    name: str, argv: list[str], logs: Path, *, deadline: float = 240, expected: int = 0
) -> Json:
    """A60-second heartbeat and process-group deadline supervise every child."""
    measured = coverage.Coverage.current()
    if measured is not None and argv[:2] == [sys.executable, "-u"] and str(e.ROOT / e.CLI) in argv:
        argv = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(measured.config.config_file),
            *argv[2:],
        ]
    return child(name, argv, logs, deadline=deadline, expected=expected, heartbeat=60)


def empty_work(private: Path) -> Json:
    """Missing independent operands are explicit gates, never measured zeros."""
    return dict(
        checks=[],
        failures=[
            dict(
                check="original_inputs",
                upstream=e.panel.TASK,
                path=str(e.ROOT / e.ORIGINAL),
                hash=e.PIN,
                field="independently_authenticated_original_episodes",
                operator="==",
                expected=16,
                observed=None,
            )
        ],
        coverage={},
        union={},
        replay_rows=[],
        findings=[],
        preconditions=resources(private),
        duration_s=0.0,
        phase_spans=[],
        sources={},
        code_hashes={str(e.ROOT / p): sha256_file(e.ROOT / p) for p in e.OWNED},
        raw_hashes={},
        terminal_replays=[],
        episode_hashes={},
        history={},
        reproductions=[],
        fixtures={},
        family_receipts=[],
    )


def historical_suite(suite: str, argv: list[str]) -> int:
    """Preserve original assertions while supplying their own sealed activation bytes."""
    import pytest

    if suite != e.panel.qualification.TESTS[1]:
        return int(pytest.main(argv))
    primary = e.panel.checked(e.ROOT / e.panel.UPSTREAM, e.panel.UPSTREAM_PIN)
    work = e.panel.checked(
        Path(primary["work_reference"]["path"]), primary["work_reference"]["sha256"]
    )
    authority = work["activation_reference"]
    source = Path(authority["snapshot_path"])
    plan = yaml.safe_load(e.panel.checked_bytes(str(source), authority["sha256"]))
    task = next(t for t in plan["tasks"] if t["id"] == e.panel.qualification.TASK)
    if e.canonical_hash(task) != e.panel.qualification.TASK_PIN:
        raise ValueError("suite_authority")
    reference = dict(
        path=str(source),
        snapshot_path=str(source),
        source_path=str(e.ROOT / "research-roadmap.yaml"),
        sha256=authority["sha256"],
        exists=True,
    )
    print(
        json.dumps(
            dict(suite=suite, original_authority=reference, producer_sha256=e.panel.UPSTREAM_PIN)
        ),
        flush=True,
    )
    with TemporaryDirectory(prefix="exp8548-authority-", dir="/var/tmp") as directory:
        with inject(dict(refs=[reference]), Path(directory)):
            return int(pytest.main(argv))


def validation(work: Json, raw: Path, private: Path, config: Path) -> None:
    """Collect before sharding, retaining unchanged assertions and terminal node IDs."""
    all_nodes, collections = [], []
    for index, suite in enumerate(HISTORICAL):
        receipt = invoke(
            f"collect_{index}",
            [sys.executable, "-m", "pytest", *COMMON, "--collect-only", "-q", suite],
            raw / "validation",
            deadline=90,
        )
        work["checks"].append(receipt)
        nodes = [
            n
            for n in Path(receipt["stdout_path"]).read_text().splitlines()
            if n.startswith(suite + "::")
        ]
        collections.append((index, suite, nodes))
        all_nodes.extend(nodes)
    for index, suite, nodes in collections:
        size = 1 if "8210" in suite else 6
        for shard, offset in enumerate(range(0, len(nodes), size)):
            group = nodes[offset : offset + size]
            e.progress("original_test_shard", offset, len(nodes) - offset)
            receipt = invoke(
                f"historical_{index}_{shard}",
                [
                    sys.executable,
                    "-u",
                    str(e.ROOT / e.CLI),
                    "--suite",
                    suite,
                    "--",
                    *COMMON,
                    "-vv",
                    *group,
                    "--basetemp=" + str(private / f"shard-{index}-{shard}"),
                ],
                raw / "validation",
                deadline=240,
            )
            receipt["collected_node_ids"] = group
            work["family_receipts"].append(receipt)
            work["checks"].append(receipt)
    work["union"] = e.exact_union(all_nodes, work["family_receipts"])
    work["checks"].append(dict(name="original_union", passed=work["union"]["passed"]))
    commands = [
        (
            "owned_tests",
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                *COMMON,
                "-q",
                e.TEST,
                "--basetemp=" + str(private / "owned"),
            ],
        ),
        ("ruff_check", [str(e.ROOT / ".venv/bin/ruff"), "check", *e.OWNED, e.TEST]),
        ("ruff_format", [str(e.ROOT / ".venv/bin/ruff"), "format", "--check", *e.OWNED, e.TEST]),
        (
            "strict_mypy",
            [str(e.ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *e.OWNED],
        ),
        ("spec_coverage", [sys.executable, "scripts/check_spec_coverage.py", "--files", e.TEST]),
    ]
    for name, argv in commands:
        work["checks"].append(invoke(name, argv, raw / "validation", deadline=360))


def measurement(work: Json, root: Path, raw: Path, private: Path) -> None:
    """Reproduce the timeout on stored bytes before using the240-second bound."""
    primary, original = e.authenticate(root)
    work.update(
        history=e.history_receipts(primary),
        episode_hashes=e.episode_hashes(original),
        sources={str(root / e.ORIGINAL): e.PIN},
        historical_model_provenance=primary["historical_model_provenance"],
    )
    candidate = e.private_history(primary, original, private / "stored")
    old_cli = str(e.ROOT / e.panel.CLI)
    timeout = invoke(
        "old90_reproduction",
        [sys.executable, "-u", old_cli, "--cold-replay", str(candidate)],
        raw / "reproduction",
        deadline=90,
    )
    work["reproductions"].append(
        dict(
            check="old90_timeout",
            passed=timeout["timed_out"] and timeout["exit_code"] == -9,
            child=timeout,
        )
    )
    replay = invoke(
        "original_terminal_replay",
        [sys.executable, "-u", old_cli, "--cold-replay", str(candidate)],
        raw / "measurement",
        deadline=240,
    )
    work["checks"].append(replay)
    work["terminal_replays"].append(replay)
    args = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    replay = invoke(
        "primitive_replay",
        [*args, "--primitives", str(raw / "replay_rows.json")],
        raw / "measurement",
        deadline=240,
    )
    work["checks"].append(replay)
    work["terminal_replays"].append(replay)
    if replay["passed"]:
        work["replay_rows"] = json.loads((raw / "replay_rows.json").read_bytes())
    control = invoke(
        "original_fixtures",
        [*args, "--fixtures", str(raw / "fixtures.json")],
        raw / "measurement",
        deadline=120,
    )
    work["checks"].append(control)
    if control["passed"]:
        work["fixtures"] = json.loads((raw / "fixtures.json").read_bytes())
    partial = raw / "partial.ini"
    partial.write_text(
        "[run]\nparallel=true\ndata_file="
        + str(raw / "reproduction" / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in e.panel.OWNED)
    )
    reproduction = invoke(
        "unpropagated_coverage",
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(partial),
            "-m",
            "pytest",
            *COMMON,
            "-q",
            e.panel.TESTS[1] + "::test_full_primitive_replay_uses_validation_reserve",
            "--basetemp=" + str(private / "partial"),
        ],
        raw / "reproduction",
        deadline=90,
    )
    for command in ("combine", "json"):
        invoke(
            "partial_" + command,
            [
                sys.executable,
                "-m",
                "coverage",
                command,
                "--rcfile=" + str(partial),
                *(["-o", str(raw / "partial_coverage.json")] if command == "json" else ["--keep"]),
            ],
            raw / "reproduction",
            deadline=30,
        )
    report = json.loads((raw / "partial_coverage.json").read_bytes())
    work["reproductions"].append(
        dict(
            check="missing_child_coverage",
            passed=reproduction["passed"] and report["totals"]["percent_covered"] < 100,
            observed_percent=report["totals"]["percent_covered"],
            child=reproduction,
            original_percent=24,
        )
    )
    work["checks"].append(
        dict(name="failure_reproductions", passed=all(r["passed"] for r in work["reproductions"]))
    )
    work["failures"] = []


def publish(value: Json, output: Path, raw: Path) -> Json:
    """The unchanged publisher exposes only exact bytes accepted by fresh consumers."""

    def validate(candidate: Path) -> Json:
        receipts = [
            invoke(name, argv, raw / "terminal", deadline=240 if name == "terminal_replay" else 90)
            for name, argv in [
                (
                    "terminal_replay",
                    [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(candidate)],
                ),
                (
                    "adversarial",
                    [sys.executable, "scripts/adversarial_verify.py", "--json", str(candidate)],
                ),
                (
                    "strict_rows",
                    [
                        sys.executable,
                        "scripts/verdict_row_consistency_lint.py",
                        "--strict",
                        str(candidate),
                    ],
                ),
            ]
        ]
        report = json.loads(Path(receipts[1]["stdout_path"]).read_bytes())
        findings = [f for row in report["reports"] for f in row.get("flags", [])]
        return dict(
            passed=all(r["passed"] for r in receipts) and findings == value["adversarial_findings"],
            checks=receipts,
            findings=findings,
        )

    publication = publish_primary(output, value, validate)
    bound = read_bound_sidecar(output, Path(publication["sidecar_path"]))
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="arc_panel_execution_ready_score",
        expected=value["arc_panel_execution_ready_score"],
    )
    side = dict(
        publication=publication,
        terminal=bound["report"],
        consumer=consumer,
        passed=bound["report"]["passed"] and consumer["passed"],
    )
    atomic_json(Path(value["terminal_validation_sidecar_path"]), side)
    return side


def run(
    root: Path, output: Path, private: Path, *, control: bool = False, failed: bool = False
) -> int:
    """Reserve validation time and retain failures instead of promoting incomplete work."""
    started = time.monotonic()
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    work = empty_work(private)
    coverage_data = private / "coverage_data"
    coverage_data.mkdir(parents=True, exist_ok=True)
    config = private / "coverage.ini"
    config.write_text(
        "[run]\ncore=sysmon\nparallel=true\npatch=subprocess,_exit\ndata_file="
        + str(coverage_data / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in e.OWNED)
    )
    work["coverage_config"] = str(config)
    collector = coverage.Coverage(config_file=str(config))
    if not control:
        collector.start()
        plan = yaml.safe_load((root / "research-roadmap.yaml").read_bytes())
        task = next((t for t in plan["tasks"] if t["id"] == e.TASK), None)
        resource = work["preconditions"]
        resource_gate_failed = (
            e.canonical_hash(task) != e.TASK_PIN
            or resource["filesystem"] in ("tmpfs", "ramfs", "unknown")
            or not all(resource["tools"].values())
            or min(resource["available_memory_bytes"], resource["free_disk_bytes"]) < 2**30
        )
        if resource_gate_failed:
            work["failures"][0].update(
                check="authority_disk_memory_tools",
                field="original_authority_and_resources",
                expected=True,
                observed=resource,
            )
        else:
            e.progress("measurement_before")
            measurement(work, root, raw, private)
            e.progress("measurement_after", len(work["replay_rows"]), 0)
        work["phase_spans"].append(dict(phase="measurement", duration_s=time.monotonic() - started))
        if time.monotonic() - started > 3600:
            work["checks"].append(dict(name="measurement_deadline", passed=False))
        validation(work, raw, private, config)
    if failed:
        work["checks"].append(dict(name="owned_failure", passed=False))
    path = raw / "work.json"
    atomic_json(path, work)
    provisional = e.build(path, output)
    tampered = deepcopy(provisional)
    if work["replay_rows"]:
        altered = deepcopy(work)
        altered["replay_rows"][0]["observation"]["actual_disabled_attempts"] += 1
        atomic_json(raw / "tampered_work.json", altered)
        tampered = e.build(raw / "tampered_work.json", output)
    else:
        tampered["completed_count"] = 999
        tampered["reproducibility_checksum"] = e.canonical_hash(tampered)
    atomic_json(raw / "tampered.json", tampered)
    for name, args, expected in [
        ("missing", ["--cold-replay", str(raw / "absent")], 1),
        ("intentional", ["--deliberate-error"], 7),
        ("rehashed_tamper", ["--cold-replay", str(raw / "tampered.json")], 1),
    ]:
        work["checks"].append(
            invoke(
                name,
                [sys.executable, "-u", str(e.ROOT / e.CLI), *args],
                raw / "controls",
                deadline=240,
                expected=expected,
            )
        )
    if not control:
        collector.stop()
        collector.save()
        preserved = raw / "coverage_before_reporting"
        shutil.copytree(coverage_data, preserved)
        for name, args in [
            ("coverage_combine", ["combine", "--keep"]),
            ("coverage_report", ["report", "--fail-under=100"]),
        ]:
            work["checks"].append(
                invoke(
                    name,
                    [
                        sys.executable,
                        "-m",
                        "coverage",
                        args[0],
                        "--rcfile=" + str(config),
                        *args[1:],
                    ],
                    raw / "validation",
                    deadline=60,
                )
            )
        for shard in preserved.glob(".coverage.*"):
            shutil.copyfile(shard, coverage_data / shard.name)
        stored_data = raw / "coverage_data"
        shutil.copytree(coverage_data, stored_data)
        stored_config = raw / "coverage.ini"
        stored_config.write_text(config.read_text().replace(str(coverage_data), str(stored_data)))
        work["coverage_config"] = str(stored_config)
        work["coverage"] = e.coverage_receipt(stored_config, work["code_hashes"])
        work["checks"].append(dict(name="exact_owned_coverage", passed=work["coverage"]["passed"]))
    work["duration_s"] = time.monotonic() - started
    work["phase_spans"].append(
        dict(
            phase="validation",
            duration_s=work["duration_s"] - sum(r["duration_s"] for r in work["phase_spans"]),
        )
    )
    work["raw_hashes"] = {
        str(p): sha256_file(p) for p in (raw / "coverage_data").glob(".coverage*") if p.is_file()
    }
    atomic_json(path, work)
    value = e.build(path, output)
    e.progress("publication_before")
    side = publish(value, output, raw)
    e.progress("publication_after", value["completed_count"], 0)
    return int(not side["passed"])


def main(argv: list[str] | None = None) -> int:
    """Direct CLI controls exercise the same import and publication boundaries."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[e.RUN_DATE], default=e.RUN_DATE)
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--owned-failure", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--primitives", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--suite", choices=HISTORICAL)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.suite:
        return historical_suite(
            args.suite, args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
        )
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        try:
            passed = e.replay(json.loads(args.cold_replay.read_bytes()))
        except (OSError, ValueError):
            passed = False
        e.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    if args.primitives:
        _, work = e.authenticate(args.root)
        atomic_json(args.primitives, e.reduce_history(work))
        return 0
    if args.fixtures:
        value = fixtures.collect(args.fixtures.parent / "fixture_operands")
        atomic_json(args.fixtures, value)
        return int(not fixtures.replay(value))
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT / "results"):
        parser.error("private controls require an output outside repository results")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8548-", dir="/var/tmp") as directory:
            return run(
                args.root,
                output,
                Path(directory),
                control=args.private_control,
                failed=args.owned_failure,
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)

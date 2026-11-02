"""REQ-VERIFY-8663: retain bounded measurements and publish exact checked bytes."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import arc_reset_8663 as e
from carnot.reporting import arc_redirect_runner_8509 as validation
from carnot.reporting import arc_reset_witness_8663 as witness
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
TASK_PIN = "sha256:d652f78bf6686f139160b8b04664a9ea70f39d9210d8f5f5461a9e64274ec237"
HEAD = "results/experiment_4629_live_frame_change_cnn.pt"
HEAD_PIN = "sha256:ee308cc8f2a1c9182cdf3bdbef8e25f0561b63ba2bca115fa7b51f17f006a78a"


def measure(root: Path, raw: Path, private: Path, date: str) -> Json:
    """Independent CPU work needs no optional hardware or future-roadmap result."""
    start = time.monotonic_ns()
    with (
        patch.multiple(validation, e=e, TESTS=e.TESTS),
        patch.object(e, "TASK_PIN", TASK_PIN, create=True),
    ):
        failures, resources, hashes = validation.preconditions(root, private)
    history = reproduction = None
    try:
        history = witness.authenticate(root)
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        failures.append(
            validation.gate(
                "historical_public_streams",
                root / next(iter(e.PINS)),
                "pinned_original_evidence",
                next(iter(e.PINS.values())),
                str(exc) if (root / next(iter(e.PINS))).exists() else None,
            )
        )
    head = root / HEAD
    actual = sha256_file(head) if head.is_file() else None
    if actual != HEAD_PIN:
        failures.append(validation.gate("numerical_checkpoint", head, "sha256", HEAD_PIN, actual))
    registry = e.ROOT / "ops/arc_solve_registry.yaml"
    roster = yaml.safe_load(registry.read_bytes())["games"]
    precheck = dict(
        path=str(registry),
        sha256=sha256_file(registry),
        checked_before_execution=True,
        policy_received_registry=False,
        existing_game_count=len(roster),
        real_solve_targets=[],
        duplicate_solve_targets=[],
        witness_game="ar25-0c556536",
        headline_solve_credit=0,
    )
    if not failures:
        assert history is not None
        e.progress("exact_exception_reproduction_before")
        reproduction = witness.reproduce(history, raw)
        e.progress(
            "exact_exception_reproduction_after",
            len(reproduction["exception_reproduction_rows"]),
            0,
        )
        if not reproduction["passed"]:
            failures.append(
                validation.gate(
                    "unreproduced_observed_failure",
                    raw / "reproduction.json",
                    "passed",
                    True,
                    False,
                )
            )
    dependencies = [
        "python/carnot/agentic/arc_competition_agent.py",
        "python/carnot/agentic/arc_outcome_observer.py",
        "python/carnot/agentic/arc_live_panel_runtime_8384.py",
        "python/carnot/reporting/arc_reset_live_8649.py",
        "python/carnot/reporting/arc_observer_live_8637.py",
        "tests/python/test_arc_reset_provenance_8649.py",
        *validation.HISTORICAL,
    ]
    snapshots = {}
    for name in [
        *hashes,
        *(str(e.ROOT / p) for p in dependencies),
        *(str(root / p) for p in e.PINS if (root / p).is_file()),
        *([str(head)] if actual == HEAD_PIN else []),
    ]:
        source = Path(name)
        target = raw / "source_closure" / source.relative_to(e.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        snapshots[str(target)] = sha256_file(target)
    references, receipts = [], []
    for i, arm in enumerate(("off", "on") if not failures else ()):
        if (time.monotonic_ns() - start) / 1e9 > 3450:
            raise TimeoutError("measurement_cap")
        directory = raw / "episodes" / arm
        unit = dict(
            episode_id="bounded-" + arm,
            game="ar25-0c556536",
            seed=11,
            arm="on",
            reset_provenance=arm,
            max_actions=16,
            max_seconds=60,
            capability_mask=True,
            directory=str(directory),
        )
        unit_path = directory / "unit.json"
        atomic_json(unit_path, unit)
        receipts.append(
            child(
                "sdk_" + arm,
                [sys.executable, "-u", str(e.ROOT / e.CLI), "--episode", str(unit_path)],
                raw / "measurement",
                deadline=90,
            )
        )
        stream = directory / "witness.json"
        if stream.is_file():
            references.append(dict(path=str(stream), sha256=sha256_file(stream)))
        e.progress("witness_arm_complete", i + 1, 1 - i)
    end = time.monotonic_ns()
    raw_hashes = {
        str(p): sha256_file(p)
        for p in raw.rglob("*")
        if p.is_file() and "source_closure" not in p.parts
    }
    return dict(
        run_date=date,
        failures=failures,
        references=references,
        historical=history,
        reproduction=reproduction,
        registry_precheck=precheck,
        measurement_receipts=receipts,
        preconditions_checked=resources,
        source_artifact_hashes=snapshots,
        code_config_hashes={
            str(raw / "source_closure" / p): sha256_file(e.ROOT / p) for p in e.OWNED + dependencies
        },
        raw_file_hashes=raw_hashes,
        historical_model_provenance=(history or {}).get("historical_model_provenance", {}),
        target_model=dict(
            path=str(head), sha256=HEAD_PIN, language_model=False, memory_withheld=True
        ),
        duration_s=(end - start) / 1e9,
        phase_spans=[
            dict(
                phase="measurement",
                started_monotonic_ns=start,
                ended_monotonic_ns=end,
                duration_s=(end - start) / 1e9,
            )
        ],
    )


def publish(value: Json, output: Path, raw: Path) -> Json:
    """Use unchanged validators and both publication readers on exact final bytes."""

    def validate(candidate: Path) -> Json:
        checks = [
            child(name, argv, raw / "terminal", deadline=120)
            for name, argv in [
                (
                    "cold_replay",
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
        with TemporaryDirectory(prefix="exp8663-readers-", dir="/var/tmp") as directory:
            private = Path(directory) / output.name
            private.write_bytes(candidate.read_bytes())
            reader = reader_receipt(
                e.TASK,
                private.parent,
                field="reset_provenance_ready_score",
                expected=value["reset_provenance_ready_score"],
            )
        report = dict(
            passed=all(r["passed"] for r in checks) and reader["passed"],
            checks=checks,
            consumer=reader,
            adversarial_report=json.loads(Path(checks[1]["stdout_path"]).read_bytes()),
        )
        atomic_json(raw / "publication_attempt.json", report)
        return report

    publication = publish_primary(output, value, validate)
    side = dict(publication=publication, passed=True)
    atomic_json(Path(value["terminal_validation_sidecar_path"]), side)
    return side


def validate_checks(raw: Path, private: Path) -> list[Json]:
    """Keep role-specific raw coverage before combining only current owned copies."""
    logs = raw / "validation"
    private.mkdir(parents=True, exist_ok=True)
    config = private / "owned.ini"
    config.write_text(
        "[run]\nparallel=True\npatch=subprocess\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in e.OWNED)
    )
    receipts = []
    suites = validation.HISTORICAL + ["tests/python/test_arc_reset_provenance_8649.py"]
    collect = child(
        "collect_once",
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            *suites,
        ],
        logs,
        deadline=90,
    )
    receipts.append(collect)
    nodes = [
        line.strip()
        for line in Path(collect["stdout_path"]).read_text().splitlines()
        if line.startswith("tests/python/") and "::" in line
    ]
    shards = [nodes[i : i + 4] for i in range(0, len(nodes), 4)]
    union_ok = (
        bool(nodes)
        and collect["passed"]
        and len(nodes) == len(set(nodes))
        and sorted(n for s in shards for n in s) == sorted(nodes)
    )
    union_path = logs / "node_union.json"
    atomic_json(union_path, dict(collected=nodes, shards=shards, disjoint=union_ok))
    for index, shard in enumerate(shards):
        role = private / f"role{index}"
        role.mkdir()
        role_config = role / "coverage.ini"
        role_config.write_text(
            "[run]\nparallel=True\npatch=subprocess\ndata_file="
            + str(role / ".coverage")
            + "\nsource="
            + str(e.ROOT / "python/carnot/reporting")
            + ","
            + str(e.ROOT / "python/carnot/agentic")
            + "\n"
        )
        receipts.append(
            child(
                f"historical{index}",
                [
                    sys.executable,
                    "-m",
                    "coverage",
                    "run",
                    "--rcfile=" + str(role_config),
                    "-m",
                    "pytest",
                    "-q",
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    *shard,
                    "--basetemp=" + str(role / "temp"),
                    "--junitxml=" + str(logs / f"historical{index}.xml"),
                ],
                logs,
                deadline=180,
                scope="consumer_or_semantic",
            )
        )
        destination = logs / "raw_coverage" / f"historical{index}"
        destination.mkdir(parents=True)
        for path in role.glob(".coverage*"):
            shutil.copyfile(path, destination / path.name)
    receipts.append(
        dict(
            name="historical_union",
            passed=bool(union_ok and all(r["passed"] for r in receipts)),
            duration_s=0,
            path=str(union_path),
            sha256=sha256_file(union_path),
        )
    )
    old = witness.OLD / "current_source_closure/python/carnot/agentic/arc_reset_provenance.py"
    harness = private / "sealed_semantics.py"
    harness.write_text(
        "import importlib.util,sys,pytest,carnot.agentic\n"
        + f"base={str(old.parent)!r}\n"
        + "for name in ['arc_outcome_observer','arc_reset_provenance','arc_competition_agent']:\n"
        + " s=importlib.util.spec_from_file_location('carnot.agentic.'+name,base+'/'+name+'.py')\n"
        + " m=importlib.util.module_from_spec(s)\n sys.modules[s.name]=m\n s.loader.exec_module(m)\n setattr(carnot.agentic,name,m)\n"
        + "raise SystemExit(pytest.main(sys.argv[1:]))\n"
    )
    original_dir = private / "original"
    original_dir.mkdir()
    original_config = original_dir / "coverage.ini"
    original_config.write_text(
        "[run]\nparallel=True\npatch=subprocess\ndata_file="
        + str(original_dir / ".coverage")
        + "\ninclude="
        + str(old.parent / "*.py")
        + "\n"
    )
    receipts.append(
        child(
            "original_semantics",
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(original_config),
                str(harness),
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                suites[-1],
                "--basetemp=" + str(original_dir / "temp"),
                "--junitxml=" + str(logs / "original.xml"),
            ],
            logs,
            deadline=180,
        )
    )
    destination = logs / "raw_coverage" / "original"
    destination.mkdir(parents=True)
    for path in original_dir.glob(".coverage*"):
        shutil.copyfile(path, destination / path.name)
    receipts.append(
        dict(name="compatible_semantics", passed=all(r["passed"] for r in receipts), duration_s=0)
    )
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
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                *e.TESTS,
                "--basetemp=" + str(private / "owned"),
                "--junitxml=" + str(logs / "owned.xml"),
            ],
            240,
        ),
        ("ruff_check", [str(e.ROOT / ".venv/bin/ruff"), "check", *e.OWNED, *e.TESTS], 60),
        (
            "ruff_format",
            [str(e.ROOT / ".venv/bin/ruff"), "format", "--check", *e.OWNED, *e.TESTS],
            60,
        ),
        (
            "strict_mypy",
            [str(e.ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=skip", *e.OWNED],
            120,
        ),
        (
            "spec_coverage",
            [sys.executable, "scripts/check_spec_coverage.py", "--files", *e.TESTS],
            60,
        ),
    ]
    for name, argv, deadline in commands:
        receipts.append(child(name, argv, logs, deadline=deadline))
    destination = logs / "raw_coverage" / "owned"
    destination.mkdir(parents=True)
    for path in private.glob(".coverage*"):
        shutil.copyfile(path, destination / path.name)
    for name, arguments in [
        ("coverage_combine", ["combine", "--keep", str(destination)]),
        ("coverage_json", ["json", "-o", str(logs / "owned_coverage.json")]),
        ("coverage_report", ["report", "--fail-under=100", "--show-missing"]),
    ]:
        receipts.append(
            child(
                name,
                [
                    sys.executable,
                    "-m",
                    "coverage",
                    *arguments[:1],
                    "--rcfile=" + str(config),
                    *arguments[1:],
                ],
                logs,
                deadline=60,
            )
        )
    receipts.append(
        dict(
            name="validation_operand_custody",
            passed=True,
            duration_s=0,
            artifact_hashes={str(p): sha256_file(p) for p in logs.rglob("*") if p.is_file()},
        )
    )
    return receipts


def main(argv: list[str] | None = None) -> int:
    """Private controls and replay use the actual CLI without starting another panel."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_language_model_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261019")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--owned-failure", action="store_true")
    args = parser.parse_args(argv)
    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        parser.error("--date requires YYYYMMDD")
    if args.cold_replay:
        try:
            passed = e.replay(json.loads(args.cold_replay.read_bytes()))
        except (OSError, ValueError, KeyError, TypeError, AssertionError):
            passed = False
        e.progress("cold_replay_complete", int(passed), 0)
        return int(not passed)
    if args.episode:
        return witness.episode(args.episode)
    if args.private_control and args.root.exists():
        parser.error("private controls require absent external inputs")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)


def run(args: Any) -> int:
    """Separate measurement from validation and keep scratch on disk."""
    raw = args.output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    with TemporaryDirectory(prefix="carnot8663-", dir="/var/tmp") as directory:
        private = Path(directory)
        work = measure(args.root, raw, private, args.date)
        path = raw / "work.json"
        atomic_json(path, work)
        atomic_json(path.with_name("measurement_seal.json"), dict(work_sha256=sha256_file(path)))
        e.progress("measurement_after_validation_before")
        if args.private_control:
            receipts = [
                child(
                    "private_control",
                    [
                        sys.executable,
                        "-c",
                        "raise SystemExit(" + str(int(args.owned_failure)) + ")",
                    ],
                    raw / "validation",
                )
            ]
        else:
            receipts = validate_checks(raw, private / "checks")
        receipts.extend(work["measurement_receipts"])
        value = e.build(path, receipts, args.output)
        e.progress("publication_before")
        side = publish(value, args.output, raw)
        e.progress("publication_after", value["completed_count"], 2 - value["completed_count"])
        return int(not side["passed"])

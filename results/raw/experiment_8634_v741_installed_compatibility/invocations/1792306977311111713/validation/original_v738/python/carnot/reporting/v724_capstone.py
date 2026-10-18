"""REQ-VERIFY-8415: existing validators qualify bounded immutable capstone evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import historical_consumer_runner_8402 as history
from carnot.reporting import historical_consumer_roots_8402 as contract
from carnot.reporting import v721_capstone as base
from carnot.reporting import v724_capstone_evidence as e
from carnot.reporting import v717_contract_runner as validation
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Freeze genuine child coverage, historical rotations and named E2E checks before measurement."""
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    with patch.object(validation, "m", e):
        plan = list(validation.manifest(private))
    plan[0]["deadline"] = 900
    next(p for p in plan if p["name"] == "coverage_combine")["argv"].append("--keep")
    fixtures = e.load(
        e.freeze(next((e.ROOT / "results").glob("experiment_8402_*.json")), private / "sources")
    )["historical_fixture_hashes"]
    if not contract.check_fixtures(fixtures):
        raise ValueError("historical_fixture_custody")
    fixture_manifest = private / "historical-fixtures.json"
    atomic_json(fixture_manifest, fixtures)
    history.rotations(e.ROOT, private)
    original = contract.historical_logs(private / "sources")[-1]["failed_node_ids"]
    for rotation in ["pinned", "rotated"]:
        plan.append(
            dict(
                name="unchanged_historical_consumers_" + rotation,
                argv=[
                    str(e.ROOT / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-v",
                    "-p",
                    "carnot.testing.historical_roots_8402",
                    "--historical-fixture-manifest=" + str(fixture_manifest),
                    "--historical-current-root=" + str(private / "current" / rotation),
                    *original,
                    "--basetemp=" + str(private / ("historical-" + rotation)),
                ],
                expected=0,
                deadline=180,
                scope="historical",
            )
        )
    for name, tests in [
        ("private_E2E021", ["tests/python/test_restricted_decision_audit_8210.py"]),
        ("full_contract_mutations", ["tests/python/test_historical_consumer_roots_8402.py"]),
        (
            "unchanged_qualified_consumers",
            ["tests/python/test_v718_contract_replay_8318.py::test_finding_policy"],
        ),
    ]:
        plan.append(
            dict(
                name=name,
                argv=[
                    str(e.ROOT / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    *tests,
                    "--basetemp=" + str(private / name),
                ],
                expected=0,
                deadline=300,
                scope="owned",
            )
        )
    return plan


def terminal_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
    """Complete cold replay includes all branch workers under its own bounded deadline."""
    if name == "terminal_cold":
        kwargs["deadline"] = 240
    return dict(base.child(name, argv, logs, **kwargs))


def controls(value: Json, raw: Path) -> list[Json]:
    """Actual missing and deliberate-error children supplement shipped positive and tamper controls."""
    receipts = list(base.controls(value, raw))
    for name, args, expected in [
        ("cold_missing", ["--cold-replay", str(raw / "absent.json")], 1),
        ("cold_deliberate_error", ["--deliberate-error"], 7),
    ]:
        receipts.append(
            base.child(
                name,
                [sys.executable, "-u", str(e.ROOT / e.CLI), *args],
                raw / "additional",
                expected=expected,
                deadline=60,
            )
        )
    return receipts


def run(
    root: Path, output: Path, private: Path, *, control: bool = False, failed: bool = False
) -> int:
    """Reuse shipped control, audit and atomic publication without changing validator policies."""
    original_execute, original_controls = base.execute, base.controls

    def execute(plan: list[Json], logs: Path) -> list[Json]:
        receipts = original_execute(plan, logs)
        if not control:
            target = logs.parent / "coverage_shards"
            target.mkdir(parents=True, exist_ok=True)
            for source in private.glob(".coverage*"):
                shutil.copyfile(source, target / source.name)
            shutil.copyfile(private / "coverage.ini", target / "coverage.ini")
        return list(receipts)

    original_measure = e.measure

    def measure(root: Path, raw: Path) -> Json:
        configuration = [
            e.freeze(path, raw / "configuration")
            for path in [
                private / "coverage.ini",
                private / "historical-fixtures.json",
                *sorted((private / "current").rglob("*")),
            ]
            if path.is_file()
        ]
        work = original_measure(root, raw)
        work["extra_refs"].extend(configuration)
        work["invocation_argv"] = [sys.executable, *sys.argv]
        return dict(work)

    original_build = e.build

    def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
        work["historical_receipts"] = [r for r in receipts if r.get("scope") == "historical"]
        work["coverage_shards"] = [
            dict(path=str(path), sha256=sha256_file(path))
            for path in sorted((raw / "coverage_shards").glob("*"))
        ]
        return dict(original_build(work, receipts, raw, output))

    def owned_controls(value: Json, raw: Path) -> list[Json]:
        with patch.object(base, "controls", original_controls):
            return controls(value, raw)

    with (
        patch.multiple(base, e=e, manifest=manifest, execute=execute, controls=owned_controls),
        patch.multiple(e, measure=measure, build=build),
        patch.object(base.terminal, "child", terminal_child),
    ):
        return int(base.run(root, output, private, control=control, failed=failed))


def main(argv: list[str] | None = None) -> int:
    """Private real controls retain old date contracts while this invocation accepts 20261011."""
    e.progress("start_no_model_load_LLM_loads_0_generation_calls_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261011"], default="20261011")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--failed-child", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--worker-request", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args(argv)
    if args.worker_request:
        return e.worker_process(args.worker_request, args.worker_output)
    if args.deliberate_error:
        e.progress("deliberate_error")
        return 7
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        parser.error("private controls must publish outside the repository")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8415-owned-", dir="/var/tmp") as directory:
            return run(
                args.root,
                output,
                Path(directory),
                control=args.private_control,
                failed=args.failed_child,
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)

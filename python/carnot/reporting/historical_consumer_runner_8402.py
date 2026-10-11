"""REQ-VERIFY-8402: bounded children seal historical outcomes before atomic publication."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import runpy
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import historical_consumer_roots_8402 as e
from carnot.reporting import v717_contract_runner as validation
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Freeze invocation coverage, real consumers and strict checks before evidence reading."""
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    with patch.object(validation, "m", e):
        plan = list(validation.manifest(private))
    plan[0]["deadline"] = 600
    next(p for p in plan if p["name"] == "coverage_combine")["argv"].append("--keep")
    old_test = "tests/python/test_v722_contract_methods_8374.py"
    config = private / "coverage.ini"
    config.write_text(
        config.read_text().replace("[report]", "    " + str(e.ROOT / old_test) + "\n[report]")
    )
    for spec in plan:
        if spec["name"] in ["coverage_report", "coverage_json"]:
            spec["argv"].append("--include=" + ",".join(str(e.ROOT / p) for p in e.OWNED))
        if spec["name"] in ["ruff_check", "ruff_format", "spec_coverage"]:
            spec["argv"].append(old_test)
    plan.append(
        dict(
            name="repaired_fixture_statements",
            argv=[
                str(e.ROOT / ".venv/bin/python"),
                "-u",
                "-c",
                "import coverage,pathlib; c=coverage.Coverage(config_file="
                + repr(str(config))
                + "); c.load(); p=pathlib.Path("
                + repr(str(e.ROOT / old_test))
                + "); lines=c.get_data().lines(str(p)); owned=[i+1 for i,s in enumerate(p.read_text().splitlines()) if 'import historical_tasks' in s or 'tasks = historical_tasks(722)' in s]; assert len(owned)==2 and all(i in lines for i in owned); print('repaired fixture statements: 2/2, 100 percent')",
            ],
            expected=0,
            deadline=30,
            scope="owned",
        )
    )
    plan.append(
        dict(
            name="private_E2E021",
            argv=[
                str(e.ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_restricted_decision_audit_8210.py",
                "--basetemp=" + str(private / "e2e021"),
            ],
            expected=0,
            deadline=300,
            scope="owned",
        )
    )
    return plan


def family_commands(raw: Path) -> dict[str, list[str]]:
    """Keep the original historical test files and nodes; rotation changes only dependency access."""
    return dict(
        direct=[
            "tests/python/test_hard_exit_learning_qualification_8206.py",
            "tests/python/test_local_consumer_qualification_8347.py",
            "tests/python/test_threshold_guard_8362.py",
            "tests/python/test_direct_atomic_state_8376.py",
            "tests/python/test_v721_capstone_frozen_aliases_8373.py",
            "tests/python/test_v721_capstone_worker_memory_8373.py",
        ],
        runtime=[
            "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            "tests/python/test_v722_contract_methods_8374.py::test_private_e2e018",
        ],
    )


def history_plan(raw: Path) -> list[Json]:
    """Freeze both input rotations and exact identical node selectors before running children."""
    return [
        dict(
            name=f"{family}_{rotation}",
            family=family,
            rotation=rotation,
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
                "--historical-fixture-manifest=" + str(raw / "fixtures/manifest.json"),
                "--historical-current-root=" + str(raw / "current" / rotation),
                *nodes,
                "--basetemp=" + str(raw / "scratch" / f"{family}-{rotation}"),
            ],
            expected=0,
            deadline=900 if family == "direct" else 180,
            scope="historical",
        )
        for family, nodes in family_commands(raw).items()
        for rotation in ["pinned", "rotated"]
    ]


def rotations(root: Path, raw: Path) -> list[Json]:
    """Private current authorities contain valid unrelated bytes while the historical roots stay immutable."""
    import yaml

    rows = []
    old_design = e.ROOT / "openspec/change-proposals/research-roadmap-v723-preserved-20261010.md"
    text = e.checked(
        old_design, "sha256:2bf4aa445d6a69f36ff4667c7a7060759420dea5155a4820a486c44a0fa60cde"
    ).read_text()
    old_tasks = e.parse_design(text, milestone="2026.10.723")[1]
    for rotation in ["pinned", "rotated"]:
        directory = raw / "current" / rotation
        for name in [e.DESIGN, e.ACTIVE, e.STAGED]:
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            source = root / (e.ACTIVE if name == e.STAGED and not (root / name).is_file() else name)
            data = (
                source.read_bytes()
                if rotation == "pinned"
                else text.encode()
                if name == e.DESIGN
                else yaml.safe_dump(dict(milestone="2026.10.723", tasks=old_tasks)).encode()
            )
            path.write_bytes(data)
            rows.append(dict(rotation=rotation, role=name, **e.reference(path)))
    return rows


def controls(value: Json, raw: Path) -> list[Json]:
    """Actual cold children accept sealed bytes and reject missing, erroneous and rehashed claims."""
    rows = []
    for label, expected in [("positive", 0), ("tamper", 1), ("missing", 1), ("error", 7)]:
        candidate = raw / (label + ".json")
        changed = deepcopy(value)
        if label == "tamper":
            changed["completed_count"] += 1
            changed["reproducibility_checksum"] = canonical_hash(
                {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
            )
        if label in ["positive", "tamper"]:
            atomic_json(candidate, changed)
        args = ["--deliberate-error"] if label == "error" else ["--cold-replay", str(candidate)]
        rows.append(
            child(
                "cold_" + label,
                [sys.executable, "-u", str(e.ROOT / e.CLI), *args],
                raw / "controls",
                expected=expected,
                deadline=60,
            )
        )
    return rows


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Separate owned validation from measured family failures and retain their exact terminal seals."""
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    (raw / "scratch").mkdir(parents=True, exist_ok=True, mode=0o700)
    plan = (
        manifest(private)
        if not control
        else [
            dict(
                name="private_child",
                argv=[sys.executable, "-u", "-c", "print('actual private child', flush=True)"],
                deadline=10,
                expected=0,
                scope="owned",
            )
        ]
    )
    history = history_plan(raw)
    # Pytest must create its directories outside the historical reader's
    # protected results namespace. Its evidence is copied into raw afterwards.
    historical_scratch = private / "historical_scratch"
    historical_scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
    for spec in history:
        spec["argv"] = [
            "--basetemp=" + str(historical_scratch / spec["name"])
            if arg.startswith("--basetemp=")
            else arg
            for arg in spec["argv"]
        ]
    atomic_json(
        raw / "execution_manifest.json",
        dict(
            owned=e.OWNED,
            commands=plan,
            historical_commands=history,
            consumer_assertion_sources=e.assertion_sources(
                [node for nodes in family_commands(raw).values() for node in nodes]
            ),
            coverage_shards="invocation-only genuine Coverage.py patch=subprocess",
            task_cap_s=4800,
            child_poll_s=30,
        ),
    )
    e.progress("preconditions_before")
    work = e.measure(root, raw)
    work["execution_manifest_reference"] = e.reference(raw / "execution_manifest.json")
    e.progress("preconditions_after", 14 if work["contract"]["activated"] else 0, 0)
    if work["fixtures"] and not control:
        work["current_file_substitution_rows"] = rotations(root, raw)
        e.progress("historical_families_before", 0, len(history))
        measured = execute(history, raw / "historical")
        for name, nodes in family_commands(raw).items():
            runs = [r for r in measured if r["name"].startswith(name + "_")]
            work["families"][name] = e.family(name, runs, nodes)
        work["family_refs"] = {}
        for name, row in work["families"].items():
            assertions = e.assertion_sources(row["consumer_node_ids"])
            seal = raw / "families" / (name + ".json")
            atomic_json(
                seal,
                dict(
                    classification=row,
                    terminal_hash=canonical_hash(row),
                    assertion_sources=assertions,
                    historical_fixture_hashes=work["fixtures"],
                    canonical_tasks_sha256=e.TASKS_PIN,
                ),
            )
            work["family_refs"][name] = e.reference(seal)
        e.progress("historical_families_after", len(history), 0)
        shutil.copytree(historical_scratch, raw / "historical_scratch")
    receipts = execute(plan, raw / "validation")
    if (private / "coverage.json").is_file():
        atomic_json(raw / "coverage.json", json.loads((private / "coverage.json").read_bytes()))
        work["owned_coverage_reference"] = e.reference(raw / "coverage.json")
        shards = raw / "coverage_shards"
        shards.mkdir()
        for path in private.glob(".coverage*"):
            shutil.copyfile(path, shards / path.name)
        work["coverage_shards"] = [e.reference(path) for path in sorted(shards.iterdir())]
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["phase_spans"] = [dict(name=r["name"], duration_s=r["duration_s"]) for r in receipts]
    work["invocation_argv"] = [str(e.ROOT / e.CLI), *sys.argv[1:]]
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, receipts, raw, output)
    receipts.extend(controls(value, raw))
    value = e.build(work, receipts, raw, output)
    publish(value, output, raw)
    return 0


def publish(value: Json, output: Path, raw: Path) -> None:
    """Unchanged validators check exact candidate bytes; their logs remain bound to the terminal."""
    reports: list[Json] = []

    def validate(candidate: Path) -> Json:
        for name, args in [
            ("terminal_cold", [str(e.ROOT / e.CLI), "--cold-replay", str(candidate)]),
            ("adversarial", ["scripts/adversarial_verify.py", "--json", str(candidate)]),
            (
                "strict_rows",
                ["scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
            ),
        ]:
            reports.append(
                child(name, [sys.executable, "-u", *args], raw / "terminal", deadline=60)
            )
        return dict(passed=all(r["passed"] for r in reports), checks=reports)

    publication = publish_primary(output, value, validate)
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="required_checks_passed",
        expected=value["required_checks_passed"],
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            checks=reports,
            consumer=consumer,
            normal_process_completion=True,
        ),
    )
    e.progress("published", 1, 0)


def main(argv: list[str] | None = None) -> int:
    """Keep the new date contract separate while all historical runners retain their dates."""
    e.progress("start_no_model_load_LLM_loads_0_generation_calls_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261011"], default="20261011")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--historical-exec", type=Path)
    parser.add_argument("--historical-version", choices=["720", "721", "722"])
    parser.add_argument("--historical-argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.historical_exec:
        return historical_exec(args.historical_exec, args.historical_version, args.historical_argv)
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
        with TemporaryDirectory(prefix="exp8402-owned-", dir="/var/tmp") as directory:
            return run(args.root, output, Path(directory), control=args.private_control)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)


def historical_exec(manifest: Path, version: str, argv: list[str]) -> int:
    """Run original child code with its original argv and the independently sealed dependency root."""
    from carnot.testing.historical_roots_8402 import bound_popen

    fixtures = json.loads(manifest.read_bytes())
    if not e.check_fixtures(fixtures):
        raise ValueError("historical_fixture_custody")
    original_argv = sys.argv
    sys.argv = argv
    try:
        with TemporaryDirectory(prefix="exp8402-historical-child-", dir="/var/tmp") as directory:
            with (
                e.inject(fixtures[version], Path(directory)),
                patch.object(subprocess, "Popen", bound_popen(manifest, version, subprocess.Popen)),
            ):
                try:
                    runpy.run_path(argv[0], run_name="__main__")
                except SystemExit as error:
                    if error.code is None:
                        return 0
                    if isinstance(error.code, int):
                        return error.code
                    print(error.code, file=sys.stderr, flush=True)
                    return 1
        return 0
    finally:
        sys.argv = original_argv

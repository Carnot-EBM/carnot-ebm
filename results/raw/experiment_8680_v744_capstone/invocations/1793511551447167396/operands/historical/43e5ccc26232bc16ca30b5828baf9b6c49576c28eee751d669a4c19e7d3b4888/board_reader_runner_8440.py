"""REQ-VERIFY-8440: reuse bounded children and the existing historical adapter."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import board_reader_qualification_8440 as e
from carnot.reporting import board_operation_boundary_8413 as old
from carnot.reporting import board_operation_runner_8413 as historical
from carnot.reporting import hardware_operation_runner_8371 as shipped
from carnot.reporting import v717_contract_runner as qualified
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import execute

Json = dict[str, Any]


def bound_popen(original: Any) -> Any:
    """Keep the qualified Exp8399 adapter and add only direct Exp8413 child rooting."""
    existing = historical.bound_popen(original)

    def start(argv: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            isinstance(argv, (list, tuple))
            and str(e.ROOT / old.CLI) in argv
            and "--historical-argv" not in argv
        ):
            index = argv.index(str(e.ROOT / old.CLI))
            argv = [
                *argv[:index],
                str(e.ROOT / e.CLI),
                "--historical-argv",
                *argv[index:],
            ]
        return existing(argv, *args, **kwargs)

    return start


def historical_main(argv: list[str]) -> int:
    """Original Exp8413 CLI and nested validators receive its independent V724 closure."""
    if argv[0] != str(e.ROOT / old.CLI):
        raise ValueError("historical_cli_identity")
    sys.argv = argv
    with (
        e.historical_bindings(724),
        patch.object(subprocess, "Popen", bound_popen(subprocess.Popen)),
    ):
        try:
            runpy.run_path(argv[0], run_name="__main__")
        except SystemExit as code:
            return int(code.code or 0)
    raise RuntimeError("historical_cli_did_not_exit")


def manifest(private: Path) -> list[Json]:
    """Existing bound_popen propagates authentic roots; each family keeps its original assertions."""
    with patch.object(qualified, "m", e):
        plan = qualified.manifest(private)
    plan[0]["deadline"] = 600
    config = private / "coverage.ini"
    text = config.read_text().replace(
        "[report]", "".join("    " + str(e.ROOT / p) + "\n" for p in old.OWNED[:2]) + "[report]"
    )
    config.write_text(text)
    legacy = private / "historical-plan"
    legacy.mkdir(parents=True, exist_ok=True)
    original = historical.manifest(legacy)[-1]
    plan.insert(
        2,
        dict(
            name="historical_V724_original_assertions",
            argv=[
                str(e.ROOT / ".venv/bin/python"),
                "-u",
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "carnot.reporting.board_reader_runner_8440",
                "--historical-tests",
                "724",
                "--basetemp=" + str(private / "v724"),
            ],
            expected=0,
            deadline=600,
            scope="owned",
        ),
    )
    plan.insert(
        3,
        dict(
            original,
            argv=[
                str(e.ROOT / ".venv/bin/python"),
                "-u",
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "carnot.reporting.board_reader_runner_8440",
                "--historical-tests",
                "723",
                "--basetemp=" + str(private / "v723"),
            ],
        ),
    )
    next(p for p in plan if p["name"] == "coverage_combine")["argv"].append("--keep")
    next(p for p in plan if p["name"] == "coverage_report")["argv"].extend(e.OWNED)
    return list(plan)


def historical_tests(version: int, args: list[str]) -> int:
    """Run unchanged tests in their own sealed family rather than assigning today's authority."""
    import pytest

    name = (
        "tests/python/test_board_operation_boundary_"
        + ("8413" if version == 724 else "8399")
        + ".py"
    )
    with (
        e.historical_bindings(version),
        patch.object(subprocess, "Popen", bound_popen(subprocess.Popen)),
    ):
        return int(
            pytest.main(
                [
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    name,
                    e.TEST + "::test_original_missing_branches",
                    *args,
                ]
            )
        )


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Save real coverage databases before shipped scratch cleanup and terminal publication."""

    def retained(plan: list[Json], logs: Path) -> list[Json]:
        """Owned child outcomes and actual missing lines determine readiness."""
        receipts: list[Json] = execute(plan, logs)
        for source in private.glob(".coverage*"):
            if source.is_file():
                destination = logs.parent / "coverage_shards" / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
        path = private / "coverage.json"
        if path.is_file():
            coverage = json.loads(path.read_bytes())
            work_path = logs.parent / "measurement.json"
            work = json.loads(work_path.read_bytes())
            work["owned_coverage"] = {
                p: r["summary"] for p, r in coverage["files"].items() if p in e.OWNED
            }
            work["historical_missing_after"] = {
                p: [line for line in lines if line in coverage["files"][p]["missing_lines"]]
                for p, lines in work["history_defects"]["missing_statements"].items()
            }
            work["coverage_shards"] = [
                e.base.reference(p)
                for p in sorted((logs.parent / "coverage_shards").glob(".coverage*"))
            ]
            receipts.append(
                dict(
                    name="historical_missing_statements_executed",
                    passed=not any(work["historical_missing_after"].values()),
                    coverage_reference=e.base.reference(path),
                    scope="owned",
                )
            )
            atomic_json(work_path, work)
        return receipts

    measure = e.measure
    captured: list[Json] = []

    def measured(root: Path, raw: Path) -> Json:
        """Share the measured object with the retention callback, without changing shipped lifecycle."""
        value = measure(root, raw)
        captured.append(value)
        return value

    def retained_bound(plan: list[Json], logs: Path) -> list[Json]:
        """Copy retained coverage into the object the shipped builder will publish."""
        receipts = retained(plan, logs)
        captured[0].update(json.loads((logs.parent / "measurement.json").read_bytes()))
        return list(receipts)

    with (
        patch.object(shipped, "e", e),
        patch.object(shipped, "manifest", manifest),
        patch.object(shipped, "execute", retained_bound),
        patch.object(e, "measure", measured),
    ):
        return int(shipped.run(root, output, private, control=control))


if __name__ == "__main__":
    raise SystemExit(historical_tests(int(sys.argv[2]), sys.argv[3:]))

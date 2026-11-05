"""REQ-VERIFY-8827: execute unchanged historical validation under sealed fixtures."""

from functools import partial
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
from unittest.mock import patch

from coverage import Coverage
import yaml

from carnot.reporting import v755_branch_contract as old
from carnot.reporting import v755_branch_runner as original
from carnot.reporting import v756_branch_contract as c
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
MISSING = [136, 137, 138, 139, 140, 141, 142, 143, 151, 152, 157, 205, 206, 207]


def original_fixture(root: Path) -> Path:
    """Use the preserved design bytes; the mutable next design is never an old fixture."""
    design = root / old.DESIGN
    design.parent.mkdir(parents=True, exist_ok=True)
    design.write_bytes((c.ROOT / c.ARCHIVE).read_bytes())
    tasks = c.parse_design(design.read_text(), milestone=old.MILESTONE)[1]
    (root / old.ACTIVE).write_text(yaml.safe_dump(dict(milestone=old.MILESTONE, tasks=tasks)))
    return root


def pytest_collection_modifyitems(items: list[Any]) -> None:
    """Repair only fixture custody after rollover, retaining every original assertion."""
    for item in items:
        if item.module.__name__.endswith("test_v755_branch_contract_8814"):
            item.module.tree = original_fixture


def measured(paths: list[Path], owned: list[str], output: Path) -> Json:
    """Combine actual SQLite data and derive statement membership from Coverage.py."""
    output.mkdir(parents=True, exist_ok=True)
    tracer = Coverage(data_file=str(output / ".coverage"), config_file=False)
    tracer.set_option("report:exclude_lines", [])
    for i, path in enumerate(paths):
        shutil.copyfile(path, output / (".coverage.input" + str(i)))
    tracer.combine(data_paths=[str(output)], strict=True, keep=True)
    tracer.save()
    report = output / "coverage.json"
    tracer.json_report(outfile=str(report), include=[str(c.ROOT / p) for p in owned])
    document = json.loads(report.read_bytes())
    rows = []
    for name in owned:
        actual = next((v for k, v in document["files"].items() if k.endswith(name)), {})
        rows.append(
            dict(
                path=name,
                statements=actual.get("summary", {}).get("num_statements", 0),
                missing=actual.get("missing_lines", ["unmeasured"]),
                excluded=actual.get("excluded_lines", []),
            )
        )
    complete = all(r["statements"] and not r["missing"] and not r["excluded"] for r in rows)
    runner = next(
        (v for k, v in document["files"].items() if k.endswith("v755_branch_runner.py")), {}
    )
    return dict(
        complete=bool(complete),
        percent=document["totals"]["percent_covered"],
        files=rows,
        rows=[
            dict(
                path=old.OWNED[1],
                line=n,
                executed=n in runner.get("executed_lines", []),
                excluded=n in runner.get("excluded_lines", []),
            )
            for n in MISSING
        ],
        report_reference=c.reference(report),
        data_reference=c.reference(output / ".coverage"),
    )


def verify_measured(report: Json, owned: list[str]) -> bool:
    """Read SQLite directly; a rehashed JSON percentage cannot invent executed lines."""
    try:
        ref = report["data_reference"]
        if c.sha256_file(Path(ref["path"])) != ref["sha256"]:
            return False
        tracer = Coverage(data_file=ref["path"], config_file=False)
        tracer.set_option("report:exclude_lines", [])
        tracer.load()
        rows = []
        for name in owned:
            _, statements, excluded, missing, _ = tracer.analysis2(str(c.ROOT / name))
            rows.append(
                dict(path=name, statements=len(statements), excluded=excluded, missing=missing)
            )
        return rows == report["files"] and all(
            r["statements"] and not r["missing"] and not r["excluded"] for r in rows
        )
    except (OSError, ValueError, KeyError):
        return False


def qualify(raw: Path) -> Json:
    """Real failure and warmup children supply measurements before original qualification."""
    raw.mkdir(parents=True, exist_ok=True)
    root = original_fixture(raw / "authority")
    config = raw / "original.ini"
    config.write_text(
        "[run]\nparallel = true\ninclude =\n"
        + "".join("    " + str(c.ROOT / p) + "\n" for p in old.OWNED)
        + "[report]\nexclude_lines =\n"
    )
    plugin = "carnot.reporting.v756_branch_qualification"
    receipts, paths = [], []
    environment = dict(
        PYTEST_PLUGINS=",".join(filter(None, [os.environ.get("PYTEST_PLUGINS"), plugin])),
    )
    with patch.dict(os.environ, environment):
        for mode in ["rejection", "coverage_warmup", "qualification"]:
            logs = raw / mode
            logs.mkdir()
            for i, path in enumerate(paths):
                shutil.copyfile(path, logs / (".coverage.seed" + str(i)))
            output = (
                raw
                / mode
                / "results"
                / (old.NAME + ".json" if mode == "qualification" else "invalid.json")
            )
            output.parent.mkdir(parents=True)
            argv = [
                sys.executable,
                "-u",
                "-m",
                "coverage",
                "run",
                "--rcfile",
                str(config),
                "--data-file",
                str(logs / ".coverage.child"),
                "--parallel-mode",
                str(c.ROOT / old.CLI),
                "--date",
                "20261104",
                "--root",
                str(root),
                "--output",
                str(output),
                "--raw-dir",
                str(logs),
            ]
            c.progress("original_" + mode + "_before")
            control = (
                dict(COVERAGE_CORE="pytrace", PYTEST_ADDOPTS="--invalid-8827-negative-control")
                if mode == "rejection"
                else dict(COVERAGE_CORE="ctrace")
            )
            with (
                patch.dict(os.environ, control),
                patch.object(
                    original, "child", partial(child, expected=0 if mode == "qualification" else 1)
                ),
            ):
                receipt = original.run_check("original_" + mode, argv, raw / "logs", 600)
            receipts.append(receipt)
            paths += sorted(logs.glob(".coverage.child.*"))
            c.progress("original_" + mode + "_after", len(receipts))
    paths += sorted(raw.glob("*/.coverage"))
    result = measured(paths, old.OWNED, raw / "combined")
    validation = json.loads((raw / "qualification/validation.json").read_bytes())
    result.update(
        complete=result["complete"]
        and all(r["passed"] for r in receipts)
        and all(r["passed"] for r in validation["receipts"]),
        receipts=receipts + validation["receipts"],
        original_assertions_unchanged=True,
        fixture_references=[c.reference(root / p) for p in [old.DESIGN, old.ACTIVE]],
        code_references=[c.reference(c.ROOT / p) for p in old.OWNED + [old.TEST]],
    )
    c.atomic_json(raw / "qualification.json", result)
    return result

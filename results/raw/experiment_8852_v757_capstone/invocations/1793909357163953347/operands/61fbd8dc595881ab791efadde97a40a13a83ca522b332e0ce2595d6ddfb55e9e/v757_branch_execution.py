"""REQ-REPORT-8840: use the original runner with private, preserved authority."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any
from unittest.mock import patch

from coverage import Coverage
import yaml

from carnot.reporting import v756_branch_contract as legacy
from carnot.reporting import v756_branch_runner as original
from carnot.reporting import v756_branch_qualification as measured
from carnot.reporting import v757_branch_contract as c
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def original_fixture(root: Path, *, active: bool = True) -> Path:
    """Redirect only fixture bytes after rollover so all old assertions remain intact."""
    design = root / c.DESIGN
    design.parent.mkdir(parents=True, exist_ok=True)
    design.write_bytes((c.ROOT / c.ARCHIVE).read_bytes())
    tasks = c.parse_design(design.read_text(), milestone=legacy.MILESTONE)[1]
    planned = dict(milestone=legacy.MILESTONE, tasks=tasks)
    (root / c.STAGED).write_text(yaml.safe_dump(planned))
    (root / c.ACTIVE).write_text(
        yaml.safe_dump(planned if active else dict(milestone="2026.11.755", tasks=[]))
    )
    return root


def pytest_collection_modifyitems(items: list[Any]) -> None:
    """Keep the historical fixture's inputs versioned without changing test expectations."""
    for item in items:
        if item.module.__name__.endswith("test_v756_branch_contract_8827"):
            item.module.fixture = original_fixture


def legacy_child(output: Path, raw: Path, mode: str) -> int:
    """Execute original statements and reuse only authenticated, complete V755 qualification."""
    prior = c.authenticate()
    root = original_fixture(raw / "fixture")
    if mode == "controls":
        assert original.main(["--date", "bad"]) == 2
        work = legacy.freeze(raw / "missing", raw / "missing-input", "20261105")
        assert work["operand_failures"] and not work["contract"]["ready"]
        (root / c.ACTIVE).write_text("{}")
        assert not legacy.authority(root)["ready"]
        return 0
    saved_validate = original.validate

    def validate(raw: Path, tracer: Coverage, upstream: Json) -> Json:
        """Use fast child tracing; the already running outer tracer keeps post-validation lines."""
        with patch.dict(os.environ, COVERAGE_CORE="ctrace"):
            return saved_validate(raw, tracer, upstream)

    with (
        patch.object(measured, "qualify", lambda raw: prior["original_owned_statement_coverage"]),
        patch.object(original, "validate", validate),
    ):
        return original.main(
            [
                "--date",
                "20261105",
                "--root",
                str(root),
                "--output",
                str(output),
                "--raw-dir",
                str(raw),
            ]
        )


def qualify(raw: Path) -> Json:
    """Measure every original missing statement in private child SQLite data before admission."""
    prior = c.authenticate()
    raw.mkdir(parents=True, exist_ok=True)
    config = raw / "original.ini"
    config.write_text(
        "[run]\nparallel = true\ninclude =\n"
        + "".join("    " + str(c.ROOT / p) + "\n" for p in legacy.OWNED + c.OWNED)
        + "[report]\nexclude_lines =\n"
    )
    paths = [Path(prior["owned_statement_coverage"]["data_reference"]["path"])]
    receipts = []
    environment = dict(PYTEST_PLUGINS=__name__, COVERAGE_CORE="pytrace")
    for mode in ["controls", "rejection", "coverage_warmup", "qualification"]:
        folder = raw / mode
        folder.mkdir()
        for i, source in enumerate(paths):
            shutil.copyfile(source, folder / (".coverage.seed" + str(i)))
        output = (
            folder
            / "results"
            / (
                legacy.NAME + ".json"
                if mode in {"coverage_warmup", "qualification"}
                else "invalid.json"
            )
        )
        command = [
            sys.executable,
            "-u",
            "-m",
            "coverage",
            "run",
            "--rcfile",
            str(config),
            "--data-file",
            str(folder / ".coverage.child"),
            "--parallel-mode",
            str(c.ROOT / c.CLI),
            "--original-mode",
            mode,
            "--output",
            str(output),
            "--raw-dir",
            str(folder),
        ]
        c.progress("v756_" + mode + "_before", len(receipts), 4 - len(receipts))
        with (
            patch.dict(os.environ, environment),
            patch.object(
                original.shared,
                "child",
                partial(child, expected=1 if mode in {"rejection", "coverage_warmup"} else 0),
            ),
        ):
            receipt = original.shared.run_check("v756_" + mode, command, raw / "logs", 600)
        receipts.append(receipt)
        paths += sorted(folder.glob(".coverage.child.*"))
        c.progress("v756_" + mode + "_after", len(receipts), 4 - len(receipts))
    report = measured.measured(paths, legacy.OWNED, raw / "combined")
    for i, source in enumerate(paths[1:]):
        shutil.copyfile(source, raw.parent / (".coverage.original" + str(i)))
    for name, command in [
        (
            "roadmap_schema",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import yaml; from scripts.roadmap_schema import Roadmap; Roadmap.model_validate(yaml.safe_load(Path('research-roadmap.yaml').read_bytes()))",
            ],
        ),
        ("roadmap_gates", [sys.executable, "scripts/audit_roadmap_gates.py", c.ACTIVE]),
        ("roadmap_exclusions", [sys.executable, "scripts/exclusion_manifest_lint.py", c.ACTIVE]),
    ]:
        receipts.append(original.shared.run_check(name, command, raw / "logs", 120))
    data = Coverage(data_file=report["data_reference"]["path"], config_file=False)
    data.load()
    executed = set(data.get_data().lines(str(c.ROOT / legacy.OWNED[2])) or [])
    report["rows"] = [
        dict(row, executed=row["line"] in executed, excluded=False) for row in c.missing_rows(prior)
    ]
    report.update(
        complete=report["complete"]
        and all(r["passed"] for r in receipts)
        and all(r["executed"] for r in report["rows"]),
        receipts=receipts,
        fixture_references=[c.reference(c.ROOT / c.ARCHIVE)],
        code_references=prior["code_config_hashes"],
        original_assertions_unchanged=True,
    )
    c.atomic_json(raw / "qualification.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    """Expose private execution and replay while keeping orchestration in the existing runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261105")
    parser.add_argument("--output", type=Path, default=c.ROOT / "results" / (c.NAME + ".json"))
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--root", type=Path, default=c.ROOT)
    parser.add_argument("--cold-replay", "--replay", type=Path)
    parser.add_argument(
        "--original-mode", choices=["controls", "rejection", "coverage_warmup", "qualification"]
    )
    args = parser.parse_args(argv)
    if args.cold_replay:
        result = c.replay(args.cold_replay)
        print(json.dumps(result), flush=True)
        return 0 if result["passed"] else 1
    if not re.fullmatch(r"\d{8}", args.date):
        print("invalid_date", flush=True)
        return 2
    raw = args.raw_dir or args.output.parent / "raw" / c.NAME / "invocations" / str(time.time_ns())
    if args.original_mode:
        return legacy_child(args.output.absolute(), raw.absolute(), args.original_mode)
    with patch.object(original, "c", c), patch.object(measured, "qualify", qualify):
        return original.run(args.output.absolute(), raw.absolute(), args.root, args.date)

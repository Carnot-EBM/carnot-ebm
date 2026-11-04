"""REQ-VERIFY-8800: bound children, retained validation and single checked publication."""

import argparse
from pathlib import Path
import sys
import time
from typing import Any

from carnot.reporting import v753_capstone as c
from carnot.reporting import v752_capstone_runner as shared
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v748_collection_contract import namespace
from scripts import publication_gate

Json = dict[str, Any]
_validator = namespace(shared, "v753_capstone_validator")
_validator.c = c
validate, terminal = _validator.validate, _validator.terminal


def validation_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
    """Owned strict checks must expose type errors disabled in legacy repo policy."""
    if name == "strict_mypy":
        argv = [argv[0], "--config-file=/dev/null", *argv[1:]]
    return child(name, argv, logs, **kwargs)


_validator.child = validation_child


def main(argv: list[str] | None = None) -> int:
    """Production checks cover all new code; isolated controls test actual child failures."""
    c.progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261104"], default="20261104")
    parser.add_argument("--root", type=Path, default=c.ROOT)
    parser.add_argument("--output", type=Path, default=c.ROOT / "results" / (c.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--failed-child", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = c.replay(args.cold_replay)
        c.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.resolve().is_relative_to(c.ROOT.resolve()):
        parser.error("private output must be outside the checkout")
    raw = output.parent / "raw" / c.NAME / "invocations" / str(time.time_ns())
    c.progress("preconditions_and_freeze_before")
    work = c.freeze(args.root, raw, args.date)
    receipt = child(
        "publication_gate",
        [sys.executable, "scripts/publication_gate.py", "--json"],
        raw / "checks",
        deadline=60,
    )
    work["publication_gates"] = c.load(c.reference(Path(receipt["stdout_path"])))
    work["references"] += [c.reference(Path(receipt[s + "_path"])) for s in ["stdout", "stderr"]]
    publication_sources = [
        publication_gate.STATE_PATH,
        publication_gate.TECH_REPORT,
        publication_gate.PAPER_TEX,
    ]
    for gate in work["publication_gates"]["gates"].values():
        if "source" in gate:
            publication_sources.append(c.ROOT / "results" / gate["source"])
    work["publication_gate_source_hashes"] = [c.reference(p) for p in publication_sources]
    work["references"] += [r for r in work["publication_gate_source_hashes"] if r["sha256"]]
    c.progress("freeze_after", 12, 0)
    if args.private_control:
        source = raw / "test_private_control.py"
        source.write_text(
            "def test_private():\n    assert " + ("False" if args.failed_child else "True") + "\n"
        )
        config = raw / "private_coverage.ini"
        config.write_text(f"[run]\ndata_file={raw}/.coverage.private\n[report]\nexclude_lines=\n")
        checked = child(
            "private_owned_control",
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                "-c",
                "/dev/null",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                str(source),
            ],
            raw / "checks",
            deadline=60,
        )
        validation = dict(
            receipts=[checked],
            union=dict(passed=checked["passed"]),
            coverage=dict(
                complete=checked["passed"],
                scope="private assertion child only; not production qualification",
            ),
            ended_at_utc=c.now(),
        )
    else:
        health = child(
            "full_python_suite",
            [".venv/bin/pytest", "tests/python", "-q"],
            raw / "repository_health",
            deadline=180,
            scope="repository_health",
        )
        work["repository_health_reference"] = dict(
            receipt=health, full_suite_pass=health["passed"], acceptance_dependency=False
        )
        c.progress("scoped_validation_before")
        validation = validate(raw)
        c.progress("scoped_validation_after")
    for name, value in [
        ("work", work),
        ("frozen_inputs", work),
        ("validation", validation),
        ("frozen_validation", validation),
    ]:
        c.atomic_json(raw / (name + ".json"), value)
    c.atomic_json(
        raw / "independent_seal.json",
        dict(
            primitive_reference=c.reference(raw / "work.json"),
            validation_reference=c.reference(raw / "validation.json"),
        ),
    )
    value = c.build(work, validation, output)
    c.progress("terminal_validation_before")
    published = publish_primary(output, value, lambda candidate: terminal(candidate, raw))
    c.atomic_json(raw / "publication_receipt.json", published)
    c.atomic_json(
        raw / "reader_receipt.json",
        reader_receipt(
            c.TASK,
            output.parent,
            field="reconciliation_ready_score",
            expected=value["reconciliation_ready_score"],
        ),
    )
    c.progress("published_" + value["verdict_class"], 12, 0)
    return int(value["verdict_class"] == "disqualified")

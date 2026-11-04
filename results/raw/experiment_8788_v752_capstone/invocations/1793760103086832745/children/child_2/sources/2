"""REQ-VERIFY-8788: bounded serial checks and immutable terminal-byte validation."""

import argparse
from pathlib import Path
import sys
import shutil
import tempfile
import time
from typing import Any

from coverage import Coverage

from carnot.reporting import v752_capstone as c
from carnot.reporting import v746_retained_evidence as retained
from carnot.reporting.collection_contract import collect
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def validate(raw: Path) -> Json:
    """Exact JUnit nodes and retained SQLite keep passing commands auditable."""
    scratch = Path(tempfile.mkdtemp(prefix="capstone8788-checks-", dir="/var/tmp"))
    plan, collection = collect(scratch, raw, c.TESTS + c.CONSUMERS, c.TESTS, child)
    receipts, children = [collection], []
    raw_refs = []
    for spec in plan:
        receipt = child(spec["name"], spec["argv"], raw / "checks", deadline=600)
        receipts.append(receipt)
        for source in Path(spec["scratch"]).rglob("original-family*"):
            if source.is_file():
                target = raw / "archived_family" / source.name
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(source, target)
                raw_refs.append(c.reference(target))
        children.append(
            retained.retain(
                receipt,
                spec["nodes"],
                Path(spec["scratch"]),
                raw / "children" / spec["name"],
                [c.reference(c.ROOT / p) for p in c.OWNED + c.TESTS],
                spec["role"],
            )
        )
    union = retained.aggregate(children, [n for s in plan for n in s["nodes"]])
    combined = retained.combine(children, raw / "combined")
    cov = Coverage(data_file=combined["combined_reference"]["path"], config_file=False)
    cov.load()
    files = []
    for name in c.OWNED:
        _, statements, excluded, missing, _ = cov.analysis2(str(c.ROOT / name))
        files.append(
            dict(path=name, statements=len(statements), excluded=excluded, missing=missing)
        )
    total, missing_count = (
        sum(r["statements"] for r in files),
        sum(len(r["missing"]) for r in files),
    )
    coverage = dict(
        complete=total > 0 and missing_count == 0 and not any(r["excluded"] for r in files),
        percent=100 * (total - missing_count) / max(1, total),
        files=files,
        combined_reference=combined["combined_reference"],
    )
    for name, argv in [
        ("ruff_check", [".venv/bin/ruff", "check", *c.OWNED, *c.TESTS]),
        ("ruff_format", [".venv/bin/ruff", "format", "--check", *c.OWNED, *c.TESTS]),
        ("strict_mypy", [".venv/bin/mypy", "--strict", "--follow-imports=silent", *c.OWNED]),
        ("spec_coverage", [sys.executable, "scripts/check_spec_coverage.py", *c.TESTS]),
    ]:
        receipts.append(child(name, argv, raw / "checks", deadline=120))
    raw_refs.append(coverage["combined_reference"])
    for receipt in receipts:
        raw_refs += [
            dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
            for stream in ["stdout", "stderr"]
        ]
    for ref in children:
        retained_row = retained.load(ref)
        command = retained_row["command_receipt"]
        raw_refs += [
            ref,
            *retained_row["sqlite_references"],
            *retained_row["source_hashes"],
            retained_row["junit_reference"],
        ]
        raw_refs += [
            dict(path=command[stream + "_path"], sha256=command[stream + "_sha256"])
            for stream in ["stdout", "stderr"]
        ]
    shutil.rmtree(scratch)
    return dict(
        receipts=receipts,
        union=union,
        coverage=coverage,
        raw_references=raw_refs,
        ended_at_utc=c.now(),
    )


def terminal(candidate: Path, raw: Path) -> Json:
    """Unchanged validators inspect the same candidate bytes that become the primary."""
    receipts = []
    for name, argv in [
        (
            "adversarial",
            [sys.executable, "scripts/adversarial_verify.py", "--json", str(candidate)],
        ),
        (
            "row_consistency",
            [sys.executable, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
        ),
    ]:
        receipts.append(child(name, argv, raw / "terminal_checks", deadline=120))
    value = dict(
        passed=all(r["passed"] for r in receipts) and c.replay(candidate),
        receipts=receipts,
        cold_replay_passed=c.replay(candidate),
        primary_sha256=c.reference(candidate)["sha256"],
    )
    c.atomic_json(raw / "terminal_validation.json", value)
    return value


def main(argv: list[str] | None = None) -> int:
    """Private controls use actual pytest children and never write production results."""
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
    c.progress("freeze_before")
    work = c.freeze(args.root, raw, args.date)
    publication = child(
        "publication_gate",
        [sys.executable, "scripts/publication_gate.py", "--json"],
        raw / "checks",
        deadline=60,
    )
    work["publication_gates"] = c.load(c.reference(Path(publication["stdout_path"])))
    work["references"] += [
        c.reference(Path(publication[s + "_path"])) for s in ["stdout", "stderr"]
    ]
    c.progress("freeze_after", 13, 0)
    if args.private_control:
        source = raw / "test_private_control.py"
        source.write_text(
            "def test_private():\n    assert " + ("False" if args.failed_child else "True") + "\n"
        )
        receipt = child(
            "private_owned_control",
            [
                sys.executable,
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
            receipts=[receipt],
            union=dict(passed=receipt["passed"]),
            coverage=dict(complete=False),
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
    c.atomic_json(raw / "work.json", work)
    c.atomic_json(raw / "frozen_inputs.json", work)
    c.atomic_json(raw / "validation.json", validation)
    c.atomic_json(raw / "frozen_validation.json", validation)
    c.atomic_json(
        raw / "independent_seal.json",
        dict(
            primitive_reference=c.reference(raw / "work.json"),
            validation_reference=c.reference(raw / "validation.json"),
        ),
    )
    value = c.build(work, validation, output)
    c.progress("terminal_validation_before")
    receipt = publish_primary(output, value, lambda candidate: terminal(candidate, raw))
    c.atomic_json(raw / "publication_receipt.json", receipt)
    c.atomic_json(
        raw / "reader_receipt.json",
        reader_receipt(
            c.TASK,
            output.parent,
            field="reconciliation_ready_score",
            expected=value["reconciliation_ready_score"],
        ),
    )
    c.progress("published_" + value["verdict_class"], 13, 0)
    return int(value["verdict_class"] == "disqualified")

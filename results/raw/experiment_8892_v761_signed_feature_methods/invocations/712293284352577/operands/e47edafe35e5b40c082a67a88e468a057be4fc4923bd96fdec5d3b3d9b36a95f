"""REQ-VERIFY-8776: bounded private checks and atomic terminal publication."""

import argparse
from datetime import datetime, UTC
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any

from coverage import Coverage

from carnot.reporting import v752_branch_methods as a
from carnot.reporting import v752_branch_report as report
from carnot.reporting import v750_archive_runner, v749_contract_runner
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
checks = namespace(v750_archive_runner, "v752_scoped_checks")
checks.a = SimpleNamespace(
    ROOT=a.ROOT,
    OWNED=a.OWNED,
    TESTS=a.TESTS,
    CONSUMERS=a.CONSUMERS + ["tests/python/test_v752_archive_8776.py"],
    reference=a.reference,
)
terminal = namespace(v749_contract_runner, "v752_terminal")
terminal.CLI = a.CLI


def now() -> str:
    """Use a fresh wall clock rather than importing an earlier invocation's epoch."""
    return datetime.now(UTC).isoformat()


def validate(raw: Path, tracer: Any) -> Json:
    """Real private results isolate unchanged historical assertions from the current roadmap."""
    with TemporaryDirectory(prefix="exp8776-archives-") as directory:
        private = Path(directory)
        suites = []
        for version in [748, 749]:
            a.progress(f"archive_{version}_before")
            root = a.archive.archived_root(private / f"archive_{version}", version)
            suites.append(a.archive.historical_suite(root, raw / f"original_{version}", version))
            a.progress(f"archive_{version}_after")
        root = a.archive.archive750(private / "archive_750")
        suites.append(a.archive.suite750(root, raw / "original_750"))
        value = dict(checks.validate(raw, tracer, suites))
        value["archive_test_references"] = [
            a.reference(p) for s in suites for p in [Path(s), Path(s).parent / "conftest.py"]
        ]
        value.setdefault("raw_references", []).extend(value["archive_test_references"])
        value["required_suites"] = checks.a.TESTS + checks.a.CONSUMERS + suites
        return value


def main(argv: list[str] | None = None) -> int:
    """Publish only checked terminal bytes; private controls cannot write in the checkout."""
    a.progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261103"], default="20261103")
    parser.add_argument("--output", type=Path, default=a.ROOT / "results" / (a.NAME + ".json"))
    parser.add_argument("--root", type=Path, default=a.ROOT)
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = report.replay(args.cold_replay)
        a.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.resolve().is_relative_to(a.ROOT.resolve()):
        parser.error("private outputs must remain outside the checkout")
    raw = output.parent / "raw" / a.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    tracer = Coverage(data_file=str(raw / "producer.sqlite"), config_file=False)
    if not args.private_control:
        tracer.start()
    a.progress("freeze_before")
    work = a.freeze(args.root, raw, args.date)
    work["invocation_id"] = raw.name
    a.progress("freeze_after", 13, 0)
    if args.private_control:
        failed_node = raw / "test_private_failure.py"
        failed_node.write_text("def test_owned_failure():\n    assert False\n")
        receipt = child(
            "private_owned_failure",
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
                str(failed_node),
            ],
            raw / "checks",
            deadline=60,
        )
        validation = dict(
            receipts=[receipt],
            union=dict(passed=False, node_ids=[]),
            coverage=dict(complete=False),
            raw_references=[],
            ended_at_utc=now(),
        )
    else:
        work["mutation_rows"] = a.mutations(Path(work["root"]), raw / "mutations")
        a.progress("scoped_validation_before")
        validation = validate(raw, tracer)
        a.progress("scoped_validation_after")
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "frozen_inputs.json", work)
    atomic_json(raw / "validation.json", validation)
    atomic_json(
        raw / "independent_seal.json",
        dict(
            primitive_reference=a.reference(raw / "work.json"),
            validation_reference=a.reference(raw / "validation.json"),
        ),
    )
    value = report.build(work, validation, output)
    a.progress("terminal_validation_before")
    published = publish_primary(output, value, lambda candidate: terminal.terminal(candidate, raw))
    atomic_json(raw / "publication_receipt.json", published)
    atomic_json(
        raw / "reader_receipt.json",
        reader_receipt(
            a.TASK,
            output.parent,
            field="contract_ready_score",
            expected=value["contract_ready_score"],
        ),
    )
    a.progress("published_" + value["verdict_class"], 13, 0)
    return 0

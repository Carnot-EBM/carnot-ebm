"""REQ-VERIFY-8801: combine actual child coverage and publish checked terminal bytes."""

import argparse
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Any

from coverage import Coverage

from carnot.reporting import v753_qualification_runner as original
from carnot.reporting import v752_capstone_runner as shared
from carnot.reporting import v754_memory_history as historical
from carnot.reporting import v754_repaired_memory as a
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child, pytest_controls
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]


def validate(raw: Path, failure: Json) -> Json:
    """Shared validators keep exact nodes and SQLite; historical fixture bindings retain assertions."""
    validator = namespace(shared, "v754_validation")
    validator.c = SimpleNamespace(**vars(a))
    repaired, old = historical.suite(raw), original.original_suite(raw)
    control = historical.repaired_control(raw)
    validator.c.TESTS = a.TESTS + [repaired, old]
    validator.c.CONSUMERS = a.CONSUMERS + ["tests/python/test_v752_archive_8776.py"]

    def suite_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
        """Canonical source paths retain the unchanged original files' Ruff policy."""
        if name in {"ruff_check", "ruff_format"}:
            argv = [
                a.previous.TESTS[0] if p == repaired else a.previous.old.TESTS[0] if p == old else p
                for p in argv
            ]
        return child(name, argv, logs, **kwargs)

    validator.child = suite_child
    result = dict(validator.validate(raw))
    joint = raw / "original_joint"
    joint.mkdir()
    for i, ref in enumerate(
        [
            result["coverage"]["combined_reference"],
            failure["sqlite_reference"],
            control["sqlite_reference"],
        ]
    ):
        shutil.copyfile(ref["path"], joint / f".coverage.{i}")
    cov = Coverage(data_file=str(joint / ".coverage"), config_file=False)
    cov.combine(keep=True)
    cov.save()
    result["original_coverage"] = original.coverage_summary(
        joint / ".coverage", a.previous.old.OWNED
    )
    result["repaired_coverage"] = original.coverage_summary(joint / ".coverage", a.previous.OWNED)
    result["original_node_inventory"] = [n for n in result["union"]["node_ids"] if "family" in n]
    result["receipts"].append(
        dict(name="repaired_original_private_cli", passed=control["passed"], control=control)
    )
    result["raw_references"] += [
        a.reference(joint / ".coverage"),
        *failure["references"],
        *control["references"],
    ]
    return result


def unqualified() -> Json:
    """Absent owned receipts cannot manufacture passing coverage or readiness."""
    return dict(
        receipts=[dict(name="not_qualified", passed=False)],
        union=dict(passed=False, node_ids=[]),
        coverage=dict(complete=False),
        original_coverage=dict(complete=False),
        repaired_coverage=dict(complete=False),
        raw_references=[],
        ended_at_utc=a.now(),
    )


terminal_runner = namespace(shared, "v754_terminal")
terminal_runner.c = a


def main(argv: list[str] | None = None) -> int:
    """Only immutable scanner-checked candidates become reader-visible primaries."""
    a.progress("start_no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261104"], default="20261104")
    parser.add_argument("--root", type=Path, default=a.ROOT)
    parser.add_argument("--output", type=Path, default=a.ROOT / "results" / (a.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = a.replay(args.cold_replay)
        a.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.resolve().is_relative_to(a.ROOT.resolve()):
        parser.error("private output must be outside the checkout")
    raw = output.parent / "raw" / a.NAME / "invocations" / str(time.time_ns())
    work = a.freeze(args.root, raw, args.date)
    a.progress("custody_and_protocol_frozen", 13, 0)
    if args.private_control or not work["preconditions"]["projection_fits"]:
        validation = unqualified()
        if args.private_control:
            controls = pytest_controls(raw / "controls", raw / "control_checks")
            work["private_control_receipts"] = controls
    else:
        work["mutation_rows"] = a.mutations(Path(work["root"]), raw / "mutations")
        a.progress("original_failure_before")
        work["original_failure"] = historical.original_failure(raw)
        a.progress("original_failure_after")
        validation = validate(raw, work["original_failure"])
    for name in ["work", "frozen_inputs"]:
        a.atomic_json(raw / (name + ".json"), work)
    a.atomic_json(raw / "validation.json", validation)
    a.atomic_json(
        raw / "independent_seal.json",
        dict(
            primitive_reference=a.reference(raw / "work.json"),
            validation_reference=a.reference(raw / "validation.json"),
        ),
    )
    value = a.build(work, validation, output)
    a.atomic_json(raw / "terminal_candidate.json", value)
    a.progress("terminal_validation_before")
    receipt = publish_primary(
        output, value, lambda candidate: terminal_runner.terminal(candidate, raw)
    )
    a.atomic_json(raw / "publication_receipt.json", receipt)
    a.progress("published_" + value["verdict_class"], 13, 0)
    return 0

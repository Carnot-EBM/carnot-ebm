"""REQ-VERIFY-8591: bounded serial children qualify one exact atomic primary.

Only new statements are covered. Private end-to-end consumers retain their
original collected identities and do not rerun historical validation controllers.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from collections.abc import Iterator
from datetime import datetime
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import signal
import sys
from tempfile import mkdtemp
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v736_contract_runner as shipped
from carnot.reporting import v738_contract_methods as e
from carnot.reporting import v738_contract_report as report
from carnot.reporting.v709_execution import execute
from carnot.reporting.current_validator_8580 import xml_nodes

Json = dict[str, Any]
collection = shipped.collection


@contextmanager
def binding() -> Iterator[None]:
    """Temporary current bindings leave original historical consumers intact afterward."""
    with patch.multiple(shipped, e=e, report=report):
        yield


def manifest(private: Path) -> list[Json]:
    """Retain the original bounded plan, including real private E2E-018 and E2E-021."""
    with binding():
        plan = list(shipped.manifest(private))
    next(r for r in plan if r["name"] == "owned_tests")["deadline"] = 450
    owned = [
        n
        for spec in plan
        if spec["name"].startswith("collect_")
        for n in spec["collected_node_ids"]
        if n.split("::")[0] in e.TESTS
    ]
    for spec in plan:
        if spec["name"] == "owned_tests":
            spec["collected_node_ids"] = owned
        if "collected_node_ids" in spec and not spec["name"].startswith("collect_"):
            spec["junit_path"] = str(private / (spec["name"] + ".xml"))
            spec["argv"] += ["--junitxml=" + spec["junit_path"]]
    return plan


def exact_union(plan: list[Json]) -> Json:
    """Actual JUnit identities must equal the disjoint frozen node lists exactly once."""
    original = [
        n
        for spec in plan
        if spec["name"].startswith("collect_")
        for n in spec["collected_node_ids"]
    ]
    groups = [spec["collected_node_ids"] for spec in plan if "junit_path" in spec]
    paths = [Path(spec["junit_path"]) for spec in plan if "junit_path" in spec]
    actual: list[str] = []
    try:
        for path in paths:
            actual += xml_nodes(path)
        passed = bool(
            original
            and len(original) == len(set(original))
            and sorted(original) == sorted(n for group in groups for n in group) == sorted(actual)
            and all(len(row) == 0 for path in paths for row in ET.parse(path).iter("testcase"))
        )
    except (OSError, ValueError, ET.ParseError):
        passed = False
    return dict(
        name="exact_node_union",
        passed=passed,
        original_node_ids=original,
        child_node_ids=groups,
        executed_node_ids=actual,
        junit_references=[e.reference(p) for p in paths if p.is_file()],
    )


def date_value(text: str) -> str:
    """Validate a real calendar date without turning the planning date into a gate."""
    try:
        parsed = datetime.strptime(text, "%Y%m%d")
        if parsed.strftime("%Y%m%d") != text:
            raise ValueError("date width")
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a calendar date YYYYMMDD") from error
    return text


def main(argv: list[str] | None = None) -> int:
    """Direct CLI children expose real acceptance and rejection while keeping a finite clock."""
    e.progress("start_no_model_load_calls_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date_value, default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--canary-binding", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.canary_binding:
        from carnot.reporting.v738_contract_inputs import accept_canary

        return int(not accept_canary(args.canary_binding))
    if args.cold_replay:
        passed = report.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        parser.error("private controls require an output outside the checkout")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
        work = e.measure(args.root, raw, args.date)
        private = Path(mkdtemp(prefix="exp8591-validation-", dir="/var/tmp"))
        plan = (
            [
                dict(
                    name="private_child",
                    argv=[sys.executable, "-c", "print('private CPU control')"],
                    deadline=10,
                    expected=0,
                    scope="owned",
                )
            ]
            if args.private_control
            else manifest(private)
        )
        e.atomic_json(
            raw / "execution_manifest.json",
            dict(commands=plan, private_control=args.private_control, private_scratch=str(private)),
        )
        work["execution_manifest_reference"] = e.reference(raw / "execution_manifest.json")
        e.atomic_json(raw / "measurement.json", work)
        receipts = execute(plan, raw / "checks")
        if not args.private_control:
            receipts.append(exact_union(plan))
            for name in ["coverage.json", "original_node_union.json"]:
                e.atomic_json(raw / name, json.loads((private / name).read_bytes()))
        validation = raw / "validation.json"
        e.atomic_json(validation, dict(receipts=receipts))
        value = report.build(work, e.reference(validation), raw, output)
        with binding():
            receipts += shipped.controls(value, raw)
        e.atomic_json(validation, dict(receipts=receipts))
        value = report.build(work, e.reference(validation), raw, output)
        with binding():
            shipped.publish(value, output, raw)
        return 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)

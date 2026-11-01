"""REQ-VERIFY-8629: original validation authority precedes atomic publication."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import signal
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import streaming_source_8629 as e
from carnot.reporting import triviaplus_extraction_runner_8620 as shipped
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v738_contract_runner import date_value

Json = dict[str, Any]


def run_checks(raw: Path) -> Json:
    """Reuse bounded original-node collection without modifying historical code."""
    with patch.object(shipped, "e", e):
        validation = shipped.run_checks(raw)
    for receipt in list(validation["receipts"]):
        if receipt.get("name") == "coverage_combine":
            directory = Path(receipt["argv"][-1])
            retained = raw / "original_coverage_shards"
            retained.mkdir(parents=True, exist_ok=True)
            refs = []
            for path in sorted(directory.glob(".coverage.*")):
                content = path.read_bytes()
                destination = retained / (e.sha256_file(path).split(":")[1] + ".coverage")
                destination.write_bytes(content)
                refs.append(e.reference(destination))
            validation["receipts"].append(
                dict(
                    name="original_coverage_retention",
                    passed=bool(refs)
                    and all(e.sha256_file(Path(r["path"])) == r["sha256"] for r in refs),
                    retained_shards=refs,
                )
            )
    return validation


def validate_candidate(path: Path, raw: Path) -> Json:
    """The shared validators and cold child must inspect the exact candidate bytes."""
    with patch.object(shipped, "e", e):
        return shipped.validate_candidate(path, raw)


def main(argv: list[str] | None = None) -> int:
    """Measure fixtures, qualify owned work and publish only checked terminal bytes."""
    e.progress("start_no_model_load_calls_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date_value, default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--measure-only", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        return int(not e.replay(args.cold_replay))
    previous = signal.setitimer(signal.ITIMER_REAL, 3600)
    started = time.monotonic()
    try:
        output = args.output.absolute()
        raw = (
            args.measure_only.parent / "primitives"
            if args.measure_only
            else output.parent / "raw" / output.stem
        )
        work = e.measure(args.root, raw, args.date)
        if args.measure_only:
            e.atomic_json(args.measure_only, work)
            return 0
        signal.setitimer(signal.ITIMER_REAL, max(1, 4800 - (time.monotonic() - started)))
        validation = run_checks(raw)
        e.atomic_json(raw / "validation.json", validation)
        candidate = raw / "private_candidate.json"
        value = e.build(work, validation, output)
        for mode in ("missing", "rehashed"):
            changed = deepcopy(value)
            if mode == "missing":
                changed["primitive_reference"]["path"] = str(raw / "absent.json")
            else:
                changed["streaming_extractor_ready_score"] = 999
                changed["reproducibility_checksum"] = e.canonical_hash(
                    {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
                )
            control = raw / (mode + ".json")
            e.atomic_json(control, changed)
            validation["receipts"].append(
                child(
                    "reject_" + mode,
                    [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(control)],
                    raw / "controls",
                    expected=1,
                    deadline=60,
                )
            )
        e.atomic_json(candidate, e.build(work, validation, output))
        checked = validate_candidate(candidate, raw)
        validation["receipts"] += checked["receipts"]
        validation["adversarial_findings"] = checked["adversarial_findings"]
        e.atomic_json(raw / "validation.json", validation)
        value = e.build(work, validation, output)

        def terminal(path: Path) -> Json:
            """A failed check may publish an honest disqualification, never readiness."""
            report = validate_candidate(path, raw)
            safe = value["verdict_class"] == "disqualified" and all(
                r["passed"] for r in report["receipts"] if r["name"] != "adversarial"
            )
            return dict(
                report,
                passed=report["passed"] or safe,
                validation_passed=report["passed"],
                failed_result_publication=safe,
            )

        result = publish_primary(output, value, terminal)
        readers = reader_receipt(
            e.TASK,
            output.parent,
            field="streaming_extractor_ready_score",
            expected=value["streaming_extractor_ready_score"],
        )
        e.atomic_json(raw / "terminal_validation.json", dict(result, reader_receipt=readers))
        e.progress("published", value["completed_count"], value["censored_count"])
        return 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)

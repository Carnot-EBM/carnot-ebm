"""REQ-REPORT-8621: publish only atomically validated current qualification bytes."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import lean_qualification_8621 as e
from carnot.reporting import lean_qualification_8609 as previous
from carnot.reporting.lean_validation_8621 import validate
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt, read_bound_sidecar
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
monotonic = time.monotonic


def terminal(path: Path, raw: Path) -> Json:
    """Keep actual unchanged validator exits and findings bound to the candidate."""
    receipts = [
        child(name, [sys.executable, "-u", *argv], raw, deadline=90)
        for name, argv in [
            ("cold_replay", [e.CLI, "--cold-replay", str(path)]),
            ("adversarial", ["scripts/adversarial_verify.py", "--json", str(path)]),
            ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)]),
        ]
    ]
    text = Path(receipts[1]["stdout_path"]).read_text()
    return dict(
        passed=all(r["passed"] for r in receipts),
        receipts=receipts,
        actual_findings=json.loads(text) if text.strip().startswith("{") else {},
    )


def run(output: Path, run_date: str) -> int:
    """Execute fresh private CPU mechanics with no inherited import environment."""
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    began = monotonic()
    old_cwd, old_path = Path.cwd(), os.environ.pop("PYTHONPATH", None)
    try:
        with TemporaryDirectory(prefix="carnot8621-execution-", dir="/var/tmp") as scratch:
            os.chdir(scratch)
            work = e.measure(raw, run_date)
    finally:
        os.chdir(old_cwd)
        if old_path is not None:
            os.environ["PYTHONPATH"] = old_path
    measurement_in_budget = monotonic() - began < 3600
    receipts = validate(work, raw)
    receipts.append(dict(name="measurement_deadline", passed=measurement_in_budget))
    atomic_json(Path(work["raw"]) / "work.json", work)
    value = e.artifact(work, receipts, output)
    changed = deepcopy(value)
    changed["completed_count"] += 1
    failed = e.artifact(work, [*receipts, dict(passed=False)], output)
    failed["lean_measurement_ready_score"] = 1
    for name, candidate, expected in [
        ("cold", value, 0),
        ("missing", None, 1),
        ("rehashed_tamper", changed, 1),
        ("owned_failure", failed, 1),
    ]:
        path = raw / "controls" / (name + ".json")
        if candidate is not None:
            candidate["reproducibility_checksum"] = previous.checksum(candidate)
            atomic_json(path, candidate)
        receipts.append(
            child(
                name,
                [sys.executable, "-u", e.CLI, "--cold-replay", str(path)],
                raw / "controls",
                expected=expected,
                deadline=90,
            )
        )
    private_output = raw / "private" / output.name
    value = e.artifact(work, receipts, output)
    publication = publish_primary(
        private_output, value, lambda p: terminal(p, raw / "private_checks")
    )
    reader = reader_receipt(
        e.TASK,
        private_output.parent,
        field="lean_measurement_ready_score",
        expected=value["lean_measurement_ready_score"],
    )
    receipts.append(
        dict(
            name="private_publication_readers",
            passed=reader["passed"],
            publication=publication,
            reader=reader,
        )
    )
    value = e.artifact(work, receipts, output)
    published = publish_primary(output, value, lambda p: terminal(p, raw / "terminal_checks"))
    atomic_json(
        raw / "terminal_validation.json",
        read_bound_sidecar(output, Path(published["sidecar_path"])),
    )
    progress("8621_complete_" + value["verdict_class"], 1, 0)
    return 0 if value["required_checks_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """Bound both real execution and replay while accepting the requested historical date."""
    progress("8621_start_no_model_load_MODEL_SPECS=[]_current_llm_calls=0")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261017")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    timer = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        if args.cold_replay:
            e.replay(args.cold_replay)
            return 0
        if datetime.strptime(args.date, "%Y%m%d").strftime("%Y%m%d") != args.date:
            raise ValueError("date_width")
        return run(args.output.absolute(), args.date)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp8621] error={error}", file=sys.stderr, flush=True)
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)

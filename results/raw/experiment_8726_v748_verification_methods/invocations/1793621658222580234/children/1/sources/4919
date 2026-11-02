"""REQ-VERIFY-8622: private replay and checked publication use the same candidate bytes."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import signal
import sys
import time
from typing import Any

from carnot.reporting import lean_cost_8622 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt, read_bound_sidecar
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]


def terminal(path: Path, raw: Path) -> Json:
    """Keep original validator exits and findings bound to the exact checked bytes."""
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


def run(output: Path, run_date: str, input_root: Path, validation: Path | None) -> int:
    """Measure first, then use bounded validation with900 seconds reserved by design."""
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = e.measure(input_root, raw, run_date)
    return finalize(work, output, validation)


def finalize(work: Json, output: Path, validation: Path | None = None) -> int:
    """Recount sealed frames without selecting or rerunning measured cells.

    Separate receipt directories preserve earlier reporting attempts when an
    accounting correction needs fresh checks on the same immutable clocks.
    """
    raw = Path(work["raw"]) / "finalizations" / str(time.time_ns())
    if validation:
        receipts = json.loads(validation.read_bytes())["receipts"]
    else:
        from carnot.reporting.lean_cost_validation_8622 import validate

        receipts = validate(raw)
    value = e.artifact(work, receipts, output)
    changed = deepcopy(value)
    changed["completed_count"] += 1
    failed = deepcopy(value)
    failed["lean_cost_ready_score"] = 1
    for name, candidate, expected in [
        ("cold", value, 0),
        ("missing", None, 1),
        ("rehashed_tamper", changed, 1),
        ("owned_failure", failed, 1),
    ]:
        path = raw / "controls" / (name + ".json")
        if candidate is not None:
            candidate["reproducibility_checksum"] = e.checksum(candidate)
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
    private = raw / "private" / output.name
    value = e.artifact(work, receipts, output)
    publication = publish_primary(private, value, lambda p: terminal(p, raw / "private_checks"))
    reader = reader_receipt(
        e.TASK,
        private.parent,
        field="lean_cost_ready_score",
        expected=value["lean_cost_ready_score"],
    )
    receipts.append(
        dict(
            name="private_readers", passed=reader["passed"], reader=reader, publication=publication
        )
    )
    value = e.artifact(work, receipts, output)
    published = publish_primary(output, value, lambda p: terminal(p, raw / "terminal_checks"))
    atomic_json(
        Path(work["raw"]) / "terminal_validation.json",
        read_bound_sidecar(output, Path(published["sidecar_path"])),
    )
    progress(
        "8622_complete_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return int(not value["required_checks_passed"] and not work["gate_check_summary"])


def main(argv: list[str] | None = None) -> int:
    """A private root enables real missing-input controls without changing live producers."""
    progress("8622_start_no_model_load_MODEL_SPECS=[]_language_model_calls=0")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261017")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--validation-receipts", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    timer = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        if args.cold_replay:
            e.replay(args.cold_replay)
            return 0
        if (
            datetime.strptime(args.date, "%Y%m%d").strftime("%Y%m%d") != args.date
            or (args.input_root and not args.private_output)
            or (args.private_output and args.private_output.resolve().is_relative_to(e.ROOT))
            or (args.validation_receipts and not args.private_output)
        ):
            raise ValueError("date_or_private_boundary")
        return run(
            args.private_output or e.ROOT / "results" / (e.NAME + ".json"),
            args.date,
            args.input_root or e.ROOT,
            args.validation_receipts,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[8622] error={error}", file=sys.stderr, flush=True)
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)

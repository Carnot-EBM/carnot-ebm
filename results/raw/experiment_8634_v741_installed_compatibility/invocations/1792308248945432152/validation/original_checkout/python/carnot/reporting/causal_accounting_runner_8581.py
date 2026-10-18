"""REQ-VERIFY-8581: supervise fresh accounting without promoting old evidence.

The existing preparation executes numerical controls and genuine restart
children. The new revision is an explicit input to this producer alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import causal_accounting_8581 as e
from carnot.reporting import causal_accounting_runner_8568 as previous
from carnot.reporting import causal_accounting_validation_8581 as validation
from carnot.reporting import v733_update_accounting as consumer
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from scripts.adversarial_verify import verify_artifact

Json = dict[str, Any]


def progress(phase: str) -> None:
    """Flush each CPU phase because model heartbeats do not exist for this run."""
    print("[exp8581] " + phase, flush=True)


def prepare(raw: Path, scratch: Path, *, private: bool = False) -> Json:
    """Authenticate new inputs before reusing unchanged numerical and event mechanics."""
    progress("start date=20261016 no_model_load MODEL_SPECS=[] current_llm_calls=0")
    binding, refs, failures = e.authority(e.ROOT)
    work = previous.prepare(raw, scratch, private=private)
    original_failures = [r for r in work["failures"] if r["field"].startswith("validator_revision")]
    work["failures"] = [r for r in work["failures"] if r not in original_failures] + failures
    work.update(
        binding=binding,
        upstream_failures=failures,
        original_revision_failures=original_failures,
        historical_accounting_reference=e.old.reference(
            e.ROOT / "results" / (e.previous.NAME + ".json")
        ),
        validator_reference=binding.get("current_validator_reference"),
        sources=work["sources"] + refs,
        code=[
            e.old.reference(e.ROOT / p)
            for p in [*e.OWNED, *e.TESTS, *e.previous.OWNED, *e.old.OWNED]
        ],
    )
    atomic_json(raw / "v737_preliminary.json", work)
    work["preliminary_reference"] = e.old.reference(raw / "v737_preliminary.json")
    atomic_json(raw / "final.json", work)
    progress("preliminary_sealed completed=1 pending=1")
    return work


def terminal(path: Path, raw: Path) -> Json:
    """Check final bytes in fresh children and retain every adversarial finding."""
    progress("before_terminal_validation")
    scan = verify_artifact(path)
    audit = child(
        "final_adversarial",
        [sys.executable, "scripts/adversarial_verify.py", "--json", str(path)],
        raw / "terminal_logs",
        expected=int(bool(scan["flags"])),
        deadline=90,
    )
    rows = child(
        "final_rows",
        [sys.executable, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)],
        raw / "terminal_logs",
        deadline=90,
    )
    cold = child(
        "final_cold",
        [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)],
        raw / "terminal_logs",
        deadline=120,
    )
    atomic_json(raw / "final_adversarial.json", scan)
    progress("after_terminal_validation")
    return dict(
        passed=e.replay(path)
        and audit["passed"]
        and rows["passed"]
        and cold["passed"]
        and not any(r["severity"] == "critical" for r in scan["flags"]),
        adversarial=scan,
        receipts=[audit, rows, cold],
    )


def run(output: Path, *, private: bool = False) -> int:
    """Keep owned failures disqualified and publish only replayable exact bytes."""
    start = time.monotonic()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    with TemporaryDirectory(prefix="carnot-8581-", dir="/var/tmp") as directory:
        work = prepare(raw, Path(directory), private=private)
        if time.monotonic() - start >= 3600:
            work["failures"].append(
                e.old.gate(raw, "measurement_deadline", "<3600", time.monotonic() - start)
            )
        validation.configuration(raw)
        if not private:
            progress("before_scoped_validation")
            validation.validate(work)
            progress("after_scoped_validation")
        work["duration_s"] = time.monotonic() - start
        work["phases"].append(
            dict(
                phase="validation", duration_s=work["duration_s"] - work["phases"][0]["duration_s"]
            )
        )
        atomic_json(raw / "final.json", work)
        candidate = raw / "preterminal.json"
        atomic_json(candidate, e.build(work, output))
        work["findings"] = verify_artifact(candidate)["flags"]
        atomic_json(raw / "final.json", work)
        value = e.build(work, output)
        publication = publish_primary(output, value, lambda path: terminal(path, raw))
        with patch.multiple(consumer, TASK=e.TASK, replay=e.replay):
            selected = consumer.stream_consumer(output, Path(publication["sidecar_path"]))
        atomic_json(
            raw / "terminal_validation.json",
            dict(
                publication=publication,
                stream_consumer=selected,
                downstream_rejection_expected=value["online_adapter_ready_score"] == 0,
                selected_identity=reader_receipt(
                    e.TASK,
                    output.parent,
                    field="online_adapter_ready_score",
                    expected=value["online_adapter_ready_score"],
                ),
            ),
        )
    progress("normal_exit verdict=" + value["verdict_class"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Keep private writes outside the checkout and enforce the invocation deadline."""
    progress("entry")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261016")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.date != "20261016" or (
        args.private_output and args.private_output.resolve().is_relative_to(e.ROOT)
    ):
        return 2
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        return int(not e.replay(args.cold_replay))
    timer = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(
            args.private_output or e.ROOT / "results" / (e.NAME + ".json"),
            private=args.private_output is not None,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)

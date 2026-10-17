"""REQ-VERIFY-8582: supervised current execution and exact-byte publication.

Private runs omit recursive validation and consequently remain disqualified.
The public run must earn every scoped test, coverage and consumer receipt.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import causal_brier_stream_8582 as e
from carnot.reporting import brier_online_runner_8517 as legacy
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from scripts.adversarial_verify import verify_artifact

Json = dict[str, Any]


def validation_child(
    name: str, argv: list[str], logs: Path, *, deadline: float = 180, **options: Any
) -> Json:
    """Reserve the explicit 900-second validation budget without changing test operands."""
    return dict(child(name, argv, logs, deadline=900 if name == "owned" else deadline, **options))


def terminal(path: Path, raw: Path) -> Json:
    """Retain the actual scanner, strict row lint and fresh-process replay findings."""
    expected_findings = json.loads(path.read_bytes())["adversarial_findings"]
    receipts = []
    for name, argv, expected in [
        (
            "adversarial",
            ["scripts/adversarial_verify.py", "--json", str(path)],
            int(bool(expected_findings)),
        ),
        ("rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)], 0),
        ("cold", [str(e.ROOT / e.CLI), "--cold-replay", str(path)], 0),
    ]:
        receipts.append(
            child(
                name,
                [sys.executable, "-u", *argv],
                raw / "terminal_logs",
                expected=expected,
                deadline=300,
            )
        )
    scan = json.loads(Path(receipts[0]["stdout_path"]).read_bytes())["reports"][0]
    return dict(
        passed=scan["flags"] == expected_findings
        and all(r["passed"] for r in receipts)
        and not any(f["severity"] == "critical" for f in scan["flags"]),
        adversarial=scan,
        receipts=receipts,
    )


def run(output: Path, *, private: bool = False, measurement: Path | None = None) -> int:
    """Finish CPU measurements before validation and atomically publish only checked bytes."""
    start = time.monotonic()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    with TemporaryDirectory(prefix="carnot-8582-", dir="/var/tmp") as directory:
        e.old.progress("preconditions_before_no_model_load_MODEL_SPECS=[]_current_llm_calls=0")
        if measurement:
            work = json.loads(measurement.read_bytes())
            previous_raw = Path(work["raw"])
            for name in (
                "stream_rows.json",
                "released_feedback.json",
                "retention_rows.json",
                "coefficient_transitions.json",
            ):
                atomic_json(raw / name, json.loads((previous_raw / name).read_bytes()))
            work.update(raw=str(raw), receipts=[], owned_ready=False)
        else:
            work = e.measure(e.ROOT, raw, Path(directory))
        if time.monotonic() - start >= 3600:
            work["failures"].append(
                e.accounting.old.gate(
                    raw, "measurement_deadline", "<3600", time.monotonic() - start
                )
            )
        e.old.progress("measurement_sealed_before_validation", 20, 0)
        atomic_json(raw / "measurement.json", work)
        if not private:
            with (
                patch.object(e.validation, "e", e),
                patch.object(e.validation, "child", validation_child),
                patch.dict(os.environ, CARNOT_8582_MEASUREMENT=str(raw / "measurement.json")),
            ):
                e.validation.validate(work)
        work["ended_monotonic_ns"] = time.monotonic_ns()
        atomic_json(raw / "measurement.json", work)
        candidate = raw / "preterminal.json"
        atomic_json(candidate, e.build(work, output))
        work["adversarial_findings"] = verify_artifact(candidate)["flags"]
        atomic_json(raw / "measurement.json", work)
        value = e.build(work, output)
        publication = publish_primary(output, value, lambda path: terminal(path, raw))
        selected = reader_receipt(
            e.TASK,
            output.parent,
            field="causal_learning_rows_ready_score",
            expected=value["causal_learning_rows_ready_score"],
        )
        atomic_json(
            Path(value["terminal_validation_sidecar_path"]),
            dict(publication=publication, selected_identity=selected),
        )
    e.old.progress("normal_exit_" + value["verdict_class"], 1, 0)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Keep worker modes on the original learner, with this producer's explicit CLI binding."""
    args = sys.argv[1:] if argv is None else argv
    print("[exp8582] start no_model_load MODEL_SPECS=[] current_llm_calls=0", flush=True)
    if any(flag in args for flag in ("--worker", "--release", "--retention")):
        with patch.object(e.stream, "CLI", e.CLI), patch.object(e.stream.shipped, "CLI", e.CLI):
            return int(legacy.main(args))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261016")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--resume-measurement", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parsed = parser.parse_args(args)
    if parsed.date != "20261016" or (
        parsed.private_output and parsed.private_output.resolve().is_relative_to(e.ROOT)
    ):
        return 2
    if parsed.deliberate_error:
        return 7
    if parsed.cold_replay:
        return int(not e.replay(parsed.cold_replay))
    timer = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(
            parsed.private_output or e.ROOT / "results" / (e.NAME + ".json"),
            private=parsed.private_output is not None,
            measurement=parsed.resume_measurement,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)

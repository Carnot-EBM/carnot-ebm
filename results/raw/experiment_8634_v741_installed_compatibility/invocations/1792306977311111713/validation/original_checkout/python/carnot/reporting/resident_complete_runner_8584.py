"""REQ-VERIFY-8584: supervise bounded validation and publish the exact checked candidate."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import resident_complete_cost_8584 as e
from carnot.reporting import resident_complete_worker_8584 as worker
from carnot.reporting import resident_validation_8572 as validation
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, progress
from scripts.adversarial_verify import verify_artifact

Json = dict[str, Any]


def controls(work: Json, receipts: list[Json], raw: Path, output: Path) -> list[Json]:
    """Execute valid, absent, failed-owned and repaired-hash candidates in fresh processes."""
    value = e.artifact(work, receipts, output)
    candidates = [("cold", value, 0), ("missing", None, 1)]
    failed = e.artifact(
        work, [*receipts, dict(name="deliberate_owned_failure", passed=False)], output
    )
    failed["resident_cost_ready_score"] = 1
    failed["reproducibility_checksum"] = e.checksum(failed)
    candidates.append(("owned_failure", failed, 1))
    changed = deepcopy(value)
    changed["completed_count"] += 1
    changed["reproducibility_checksum"] = e.checksum(changed)
    candidates.append(("rehashed_tamper", changed, 1))
    result = []
    for name, candidate, expected in candidates:
        path = raw / "controls" / (name + ".json")
        if candidate is not None:
            atomic_json(path, candidate)
        result.append(
            child(
                name,
                [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)],
                raw / "control_logs",
                expected=expected,
                deadline=180,
            )
        )
    return result


def terminal(path: Path, raw: Path) -> Json:
    """Require fresh replay and retain unchanged scanner and row-lint findings."""
    findings = json.loads(path.read_bytes())["adversarial_findings"]
    receipts = []
    for name, argv, expected in [
        (
            "adversarial",
            ["scripts/adversarial_verify.py", "--json", str(path)],
            int(bool(findings)),
        ),
        ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)], 0),
        ("cold", [e.CLI, "--cold-replay", str(path)], 0),
    ]:
        receipts.append(
            child(
                name,
                [sys.executable, "-u", *argv],
                raw / "terminal_logs",
                expected=expected,
                deadline=180,
            )
        )
    scan = json.loads(Path(receipts[0]["stdout_path"]).read_bytes())["reports"][0]
    return dict(
        passed=all(r["passed"] for r in receipts)
        and scan["flags"] == findings
        and not any(f["severity"] == "critical" for f in findings),
        receipts=receipts,
        adversarial=scan,
    )


def run(output: Path, *, measurement: Path | None = None, private: bool = False) -> int:
    """Measurements stop before terminal tests; private runs cannot assert earned readiness."""
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    if measurement:
        work = json.loads(measurement.read_bytes())
        original_raw = Path(work["raw"])
        work["measurement_reference"] = e.reference(measurement)
        work["measurement_code_config_hashes"] = [
            e.reference(p) for p in sorted((original_raw / "measurement_sources").glob("*.py"))
        ]
        work["raw"] = str(raw)
        if work.get("first_cell_forecast"):
            work["preliminary_first_cell_forecast"] = work["first_cell_forecast"]
            work["first_cell_forecast"] = dict(
                work["first_cell_forecast"],
                projected_total_s=work["first_cell_forecast"]["observed_s"] * 240,
                projection_unit="960 launches / first four launches",
            )
        atomic_json(raw / "measurement.json", work)
    else:
        work = e.measure(e.ROOT, raw)
    receipts = []
    if not private:
        suites = [e.TEST, *validation.SUITES[1:]]
        with (
            patch.object(validation, "e", e),
            patch.object(validation, "SUITES", suites),
            patch.dict(os.environ, CARNOT_8584_MEASUREMENT=str(raw / "measurement.json")),
        ):
            receipts.extend(validation.validate(work, raw))
            receipts.extend(validation.finish(work, raw))
    work["ended_ns"] = time.monotonic_ns()
    atomic_json(raw / "measurement.json", work)
    receipts.extend(controls(work, receipts, raw, output))
    preliminary = raw / "preterminal.json"
    atomic_json(preliminary, e.artifact(work, receipts, output))
    work["adversarial_findings"] = verify_artifact(preliminary)["flags"]
    for name, argv in [
        ("adversarial", ["scripts/adversarial_verify.py", "--json", str(preliminary)]),
        ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(preliminary)]),
    ]:
        receipts.append(
            child(name, [sys.executable, "-u", *argv], raw / "preterminal_logs", deadline=180)
        )
    atomic_json(raw / "measurement.json", work)
    value = e.artifact(work, receipts, output)
    publication = publish_primary(output, value, lambda p: terminal(p, raw))
    selected = reader_receipt(
        e.TASK,
        output.parent,
        field="resident_cost_ready_score",
        expected=value["resident_cost_ready_score"],
    )
    downstream = (
        e.consume(output)
        if value["resident_cost_ready_score"]
        else dict(passed=False, reason="unqualified_cost")
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(publication=publication, selected_identity=selected, cost_consumer=downstream),
    )
    progress("8584_complete_" + value["verdict_class"], 240, 0)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Bound direct execution and expose real worker and cold-replay routes for tests."""
    args = sys.argv[1:] if argv is None else argv
    # Workers reserve stdout for framed protocol bytes.
    if "--worker" not in args:
        progress("8584_start_no_model_load_MODEL_SPECS=[]_current_llm_calls=0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261016")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--config-hash", default="")
    parser.add_argument("--arm", choices=["python", "native"], default="python")
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--consume", type=Path)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--resume-measurement", type=Path)
    parsed = parser.parse_args(args)
    if parsed.date != "20261016" or (
        parsed.private_output and parsed.private_output.resolve().is_relative_to(e.ROOT)
    ):
        return 2
    if parsed.worker:
        return worker.serve(parsed.worker, parsed.config_hash, parsed.arm)
    if parsed.cold_replay:
        return int(not e.replay(parsed.cold_replay))
    if parsed.consume:
        print(json.dumps(e.consume(parsed.consume)), flush=True)
        return 0
    timer = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(
            parsed.private_output or e.ROOT / "results" / (e.NAME + ".json"),
            measurement=parsed.resume_measurement,
            private=parsed.private_output is not None,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)

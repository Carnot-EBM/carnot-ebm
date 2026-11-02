"""REQ-VERIFY-8593: unchanged terminal validators decide exact-byte publication."""

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

from carnot.reporting import resident_provenance_8593 as e
from carnot.reporting import resident_validation_8593 as validation
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary, reader_receipt, read_bound_sidecar
from carnot.reporting.v709_execution import child, progress
from scripts.adversarial_verify import verify_artifact

Json = dict[str, Any]


def terminal(path: Path, raw: Path, *, replay: bool = True) -> Json:
    """A warning exit remains a failure; final acceptance never filters findings."""
    plan = [
        ("adversarial", ["scripts/adversarial_verify.py", "--json", str(path)]),
        ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)]),
    ]
    if replay:
        plan.append(("terminal_replay", [e.CLI, "--cold-replay", str(path)]))
    receipts = [child(name, [sys.executable, "-u", *argv], raw, deadline=60) for name, argv in plan]
    return dict(
        passed=all(r["passed"] for r in receipts)
        and not verify_artifact(path)["flags"]
        and (replay or e.check_descriptor(json.loads(path.read_bytes()).get("target_model"))),
        receipts=receipts,
        findings=verify_artifact(path)["flags"],
    )


def controls(work: Json, receipts: list[Json], raw: Path, output: Path) -> list[Json]:
    """Repaired envelopes cannot erase owned failure or replace the numerical head."""
    value = e.artifact(work, receipts, output)
    cases = [("cold", value, 0), ("missing", None, 1)]
    failed = e.artifact(
        work, [*receipts, dict(name="deliberate_owned_failure", passed=False)], output
    )
    failed["resident_measurement_ready_score"] = 1
    changed = deepcopy(value)
    changed["completed_count"] += 1
    cases.extend([("owned_failure", failed, 1), ("rehashed_tamper", changed, 1)])
    for name, target in [
        ("absent_descriptor", None),
        ("wrong_head", dict(value.get("target_model") or {}, parameter_count=108)),
        (
            "inconsistent_scaler",
            dict(value.get("target_model") or {}, scaler_sha256="sha256:inconsistent"),
        ),
    ]:
        mutated = deepcopy(value)
        mutated["target_model"] = target
        if target is not None:
            target["descriptor_sha256"] = canonical_hash(
                {k: v for k, v in target.items() if k != "descriptor_sha256"}
            )
        cases.append((name, mutated, 1))
    result = []
    for name, candidate, expected in cases:
        path = raw / "controls" / (name + ".json")
        if candidate is not None:
            candidate["reproducibility_checksum"] = e.checksum(candidate)
            atomic_json(path, candidate)
        result.append(
            child(
                name,
                [
                    sys.executable,
                    "-u",
                    str(e.ROOT / e.CLI),
                    "--consume"
                    if name in {"absent_descriptor", "wrong_head", "inconsistent_scaler"}
                    else "--cold-replay",
                    str(path),
                ],
                raw / "control_logs",
                expected=expected,
                deadline=60,
            )
        )
    work["descriptor_mutation_rows"] = [
        dict(
            name=r["name"],
            passed=r["passed"],
            exit_code=r["exit_code"],
            log_reference=dict(path=r["stdout_path"], sha256=r["stdout_sha256"]),
        )
        for r in result[4:]
    ]
    return result


def run(output: Path, run_date: str) -> int:
    """Evidence qualification executes no new serving benchmark or language model."""
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = e.measure(raw, run_date)
    if work["target_model"] is not None:
        atomic_json(raw / "head_descriptor.json", work["target_model"])
    receipts = validation.validate(work, raw)
    receipts.extend(validation.finish(work, raw))
    if work["reproduction"]:
        corrected = work["reproduction"]["corrected"]
        path = Path(corrected["candidate"]["path"])
        private_output = Path(corrected["publication"]["primary_path"])
        corrected["publication"] = publish_primary(
            private_output,
            json.loads(path.read_bytes()),
            lambda p: terminal(p, raw / "private_checks", replay=False),
        )
        report = read_bound_sidecar(private_output, Path(corrected["publication"]["sidecar_path"]))[
            "report"
        ]
        accepted = child(
            "private_descriptor_consumer",
            [sys.executable, "-u", str(e.ROOT / e.CLI), "--check-descriptor", str(path)],
            raw / "private_checks",
            deadline=60,
        )
        report["passed"] = (
            report["passed"] and accepted["passed"] and corrected["publication"] is not None
        )
        work["private_publication_receipts"] = [
            dict(publication=corrected["publication"], validators=report, consumer=accepted)
        ]
        receipts.extend(
            [
                *report["receipts"],
                accepted,
                dict(name="private_publication", passed=report["passed"]),
            ]
        )
    atomic_json(raw / "work.json", work)
    receipts.extend(controls(work, receipts, raw, output))
    work["ended_ns"] = time.monotonic_ns()
    atomic_json(raw / "work.json", work)
    value = e.artifact(work, receipts, output)
    publication = publish_primary(output, value, lambda p: terminal(p, raw / "terminal_logs"))
    downstream = (
        e.consume(output)
        if value["resident_measurement_ready_score"]
        else dict(passed=False, reason="unqualified_provenance")
    )
    selected = reader_receipt(
        e.TASK,
        output.parent,
        field="resident_measurement_ready_score",
        expected=value["resident_measurement_ready_score"],
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(publication=publication, selected_identity=selected, provenance_consumer=downstream),
    )
    progress("8593_complete_" + value["verdict_class"], 2, 0)
    return 0


def date_value(value: str) -> str:
    """Runtime dates bind this invocation without limiting future valid executions."""
    try:
        if datetime.strptime(value, "%Y%m%d").strftime("%Y%m%d") != value:
            raise ValueError("date_width")
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected YYYYMMDD calendar date") from error
    return value


def main(argv: list[str] | None = None) -> int:
    """Expose genuine replay and consumer routes under one bounded entry point."""
    progress("8593_start_no_model_load_MODEL_SPECS=[]_current_llm_calls=0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date_value, default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--consume", type=Path)
    parser.add_argument("--check-descriptor", type=Path)
    parser.add_argument("--private-output", type=Path)
    parsed = parser.parse_args(argv)
    if parsed.cold_replay:
        return int(not e.replay(parsed.cold_replay))
    try:
        if parsed.consume:
            print(json.dumps(e.consume(parsed.consume)), flush=True)
            return 0
        if parsed.check_descriptor:
            return int(
                not e.check_descriptor(
                    json.loads(parsed.check_descriptor.read_bytes()).get("target_model")
                )
            )
    except (OSError, ValueError, KeyError, TypeError):
        return 1
    timer = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(parsed.private_output or e.ROOT / "results" / (e.NAME + ".json"), parsed.date)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)

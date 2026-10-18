"""REQ-VERIFY-8568: keep measurement, immutable snapshots and publication bounded.

The controller uses an explicit coverage file and never accesses a live tracer.
Private runs exercise the actual paths without nesting qualification controllers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch
from typing import Any

from carnot.reporting import causal_accounting_8568 as e
from carnot.reporting import causal_accounting_validation_8568 as validation
from carnot.reporting import canonical_update_runner_8555 as previous
from carnot.reporting import objective_publication_8516 as objective
from carnot.reporting import v733_update_accounting as consumer
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from scripts.adversarial_verify import verify_artifact

Json = dict[str, Any]


def progress(phase: str) -> None:
    """Flush phase boundaries so CPU work and supervised children remain observable."""
    print("[exp8568] " + phase, flush=True)


def prepare(raw: Path, scratch: Path, *, private: bool = False) -> Json:
    """Check real tools and original authority before numerical controls or subprocesses."""
    start = time.monotonic()
    progress("start date=20261016 no_model_load MODEL_SPECS=[] current_llm_calls=0")
    raw.mkdir(parents=True, exist_ok=True)
    pre = previous.resources.resources(scratch)
    pre["passed"] = (
        pre["filesystem"] not in {"tmpfs", "ramfs"}
        and all(pre["tools"].values())
        and pre["available_memory_bytes"] >= 268435456
        and pre["free_disk_bytes"] >= 104857600
    )
    historical, refs, failures = e.authority(e.ROOT)
    if not pre["passed"]:
        failures.append(e.old.gate(scratch, "disk_memory_tools", True, pre))
    progress("authenticate_complete reduce_events")
    events, sources = (e.old.controls(), []) if private else e.old.history()
    atomic_json(raw / "events.json", events)
    progress("before_independent_finite_differences")
    numerical = e.old.numerical(scratch / "numerical")
    progress("after_independent_finite_differences before_restart_children")
    restarts, restarted = previous.restart(raw / "restart")
    progress("after_restart_children")
    primary = objective.ROOT / "results" / (objective.NAME + ".json")
    original = e.old.read(dict(path=str(primary), sha256=previous.old.old.UPSTREAM_PIN))
    terminal = e.old.read(e.old.reference(Path(original["terminal_validation_sidecar_path"])))
    upstream = objective.consumer(
        primary, Path(terminal["publication"]["sidecar_path"]), replay=False
    )
    union = historical.get("exact_union_receipt", dict(passed=False))
    # The qualified historical artifact owns this union; no historical tests are rerun.
    union = dict(
        passed=union.get("passed", False),
        original_node_ids=e.old.historical.original()[1]["nodes"],
        sealed_receipt=union,
        authority_by_suite=historical.get("original_authority_by_suite"),
    )
    work: Json = dict(
        raw=str(raw.absolute()),
        private=private,
        historical=historical,
        failures=failures,
        events=e.old.reference(raw / "events.json"),
        sources=[*refs, *sources, e.old.reference(primary)],
        code=[e.old.reference(e.ROOT / p) for p in [*e.OWNED, *e.TESTS]],
        historical_reference=dict(
            path=str(e.ROOT / "results" / (e.old.HISTORICAL_NAME + ".json")),
            sha256=e.old.HISTORICAL_PIN,
        ),
        validator_reference=dict(
            path=str(e.ROOT / "results" / (e.validator.NAME + ".json")), sha256=e.VALIDATOR_PIN
        ),
        preconditions=pre,
        numerical=numerical,
        restart_receipts=restarts,
        restart_passed=restarted,
        receipts=restarts.copy(),
        diagnostics=e.diagnostics(e.ROOT),
        historical_union=union,
        controls=e.old.controls(),
        upstream_consumer=upstream,
        owned_ready=False,
        duration_s=time.monotonic() - start,
        phases=[],
        findings=[],
    )
    work["phases"] = [dict(phase="measurement", duration_s=work["duration_s"])]
    atomic_json(raw / "preliminary.json", work)
    work["preliminary_reference"] = e.old.reference(raw / "preliminary.json")
    atomic_json(raw / "final.json", work)
    return work


def terminal(path: Path, raw: Path) -> Json:
    """Retain all scanner findings and validate exact bytes before atomic replacement."""
    progress("validate_final_bytes")
    scan = verify_artifact(path)
    value = json.loads(path.read_bytes())
    critical = any(f["severity"] == "critical" for f in scan["flags"])
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
    return dict(
        passed=e.replay(path)
        and rows["passed"]
        and audit["passed"]
        and cold["passed"]
        and not critical,
        adversarial=scan,
        receipts=[audit, rows, cold],
        readiness=value["online_adapter_ready_score"],
    )


def run(output: Path, *, private: bool = False) -> int:
    """Keep old failures diagnostic while requiring fresh current checks for readiness."""
    start = time.monotonic()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    with TemporaryDirectory(prefix="carnot-8568-", dir="/var/tmp") as directory:
        work = prepare(raw, Path(directory), private=private)
        if time.monotonic() - start >= 3600:
            work["failures"].append(
                e.old.gate(raw, "measurement_deadline", "<3600", time.monotonic() - start)
            )
        validation.configuration(raw)
        if not private:
            validation.validate(work)
        work["duration_s"] = time.monotonic() - start
        work["phases"].append(
            dict(
                phase="validation", duration_s=work["duration_s"] - work["phases"][0]["duration_s"]
            )
        )
        atomic_json(raw / "final.json", work)
        preliminary_candidate = raw / "preterminal.json"
        atomic_json(preliminary_candidate, e.build(work, output))
        work["findings"] = verify_artifact(preliminary_candidate)["flags"]
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
    """Reject date drift and keep private fixtures outside the repository results."""
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
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(
            args.private_output or e.ROOT / "results" / (e.NAME + ".json"),
            private=args.private_output is not None,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)

"""REQ-REPORT-8407: bounded real consumers check terminal bytes before publication."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import python_transaction_cost_8407 as e
from carnot.reporting import continuous_direct_runner_8406 as inherited
from carnot.reporting import direct_atomic_runner_8376 as shipped
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import execute, child as base_child

Json = dict[str, Any]
SCRATCH = Path.home() / ".cache/carnot-exp8407-private"


def plan(private: Path) -> list[Json]:
    """Reuse scoped coverage and the independently pinned private historical consumers."""
    with patch.object(inherited, "e", e):
        result = inherited.plan(private)
    return [r for r in result if r["name"] != "private_E2E021"]


def bounded_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
    """A complete panel can require more replay time than an individual control."""
    if "--cold-replay" in argv:
        kwargs["deadline"] = 600
        argv = [*argv, "--cold-reduction-output", str(logs / "cold_reduction.json")]
    return dict(base_child(name, argv, logs, **kwargs))


def validate(candidate: Path, raw: Path) -> Json:
    """The unchanged typed consumer resolves findings; this adapter grants no exemptions."""
    with patch.object(shipped, "e", e), patch.object(shipped, "child", bounded_child):
        return dict(shipped.validate(candidate, raw))


def main(argv: list[str] | None = None) -> int:
    """Bound the invocation, record actual children and atomically expose checked bytes."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261011"], default="20261011")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--cold-reduction-output", type=Path)
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 1
    if args.cold_replay:
        passed = e.replay(args.cold_replay, args.cold_reduction_output)
        e.progress("replay_passed" if passed else "replay_failed")
        return int(not passed)
    began = time.monotonic()
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        SCRATCH.mkdir(parents=True, exist_ok=True, mode=0o700)
        with TemporaryDirectory(dir=SCRATCH, prefix="invocation-") as folder:
            private = Path(folder)
            os.environ["TMPDIR"] = str(private)
            raw = args.output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
            raw.mkdir(parents=True, mode=0o700)
            frozen = plan(private)
            atomic_json(raw / "validation_plan.json", dict(commands=frozen))
            work = e.measure(args.root, raw, private)
            work["code_config_hashes"] = [e.reference(e.ROOT / p) for p in [*e.OWNED, e.TEST]] + [
                e.reference(raw / "validation_plan.json")
            ]
            receipts = execute(frozen, raw / "validation")
            work["coverage_shards"] = []
            for source in private.glob(".coverage*"):
                if source.is_file():
                    destination = raw / "coverage_shards" / source.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(source.read_bytes())
                    work["coverage_shards"].append(e.reference(destination))
            report = private / "coverage.json"
            if report.is_file():
                saved = raw / "coverage.json"
                saved.write_bytes(report.read_bytes())
                work["owned_coverage"] = dict(
                    json.loads(saved.read_bytes())["totals"], reference=e.reference(saved)
                )
            atomic_json(Path(work["work_path"]), work)
            candidate = private / (e.NAME + ".json")
            atomic_json(candidate, e.build(work, receipts))
            with patch.object(shipped, "e", e), patch.object(shipped, "child", bounded_child):
                receipts.extend(shipped.controls(candidate, raw / "controls"))
            initial = validate(candidate, raw / "initial_terminal")
            receipts.extend(initial["checks"])
            work["adversarial_findings"] = initial["adversarial"]["findings"]
            work["duration_s"] = time.monotonic() - began
            work["phase_spans"].extend(
                dict(phase=r["name"], duration_s=r["duration_s"]) for r in receipts
            )
            atomic_json(Path(work["work_path"]), work)
            value = e.build(work, receipts)
            publication = publish_primary(
                args.output, value, lambda p: validate(p, raw / "terminal")
            )
            atomic_json(Path(value["terminal_validation_sidecar_path"]), publication)
            e.progress("published_" + value["verdict_class"], 1, 0)
            return int(not value["required_checks_passed"])
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        e.progress("owned_failure_" + type(error).__name__ + "_" + str(error))
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)

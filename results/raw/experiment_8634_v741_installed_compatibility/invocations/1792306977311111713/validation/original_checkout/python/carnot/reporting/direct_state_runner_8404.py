"""REQ-VERIFY-8404: bounded children publish only atomically checked terminal bytes."""

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

from carnot.reporting import direct_state_qualification_8404 as e
from carnot.reporting import direct_state_qualification_8390 as old
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
SCRATCH = Path.home() / ".cache/carnot-exp8404-private"


def main(argv: list[str] | None = None) -> int:
    """Freeze commands before measurement and retain failed checks without granting readiness."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261011"], default="20261011")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--state-directory", type=Path)
    parser.add_argument("--barrier", choices=["none", *e.BARRIERS], default="none")
    parser.add_argument("--two-readers", action="store_true")
    parser.add_argument("--probe", nargs=3)
    args = parser.parse_args(argv)
    with e.bindings(), patch.object(old.runner, "e", e):
        if args.deliberate_error:
            return 1
        if args.cold_replay:
            passed = e.replay(args.cold_replay)
            e.progress("replay_passed" if passed else "replay_failed")
            return int(not passed)
        if args.probe:
            frozen, state, mode = args.probe
            return int(old.probe(Path(frozen), Path(state), mode))
        if args.worker:
            if args.state_directory is None:
                parser.error("--worker requires --state-directory")
            return int(e.worker(args.worker, args.state_directory, args.barrier, args.two_readers))
        began = time.monotonic()
        previous = signal.setitimer(signal.ITIMER_REAL, 4800)
        try:
            SCRATCH.mkdir(parents=True, exist_ok=True, mode=0o700)
            SCRATCH.chmod(0o700)
            with TemporaryDirectory(dir=SCRATCH, prefix="invocation-") as directory:
                private = Path(directory)
                os.environ["TMPDIR"] = str(private)
                raw = args.output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
                raw.mkdir(parents=True, mode=0o700)
                frozen_plan = e.plan(private)
                atomic_json(raw / "validation_plan.json", dict(commands=frozen_plan))
                work = e.measure(args.root, raw, private)
                work["code_config_hashes"].append(e.reference(raw / "validation_plan.json"))
                atomic_json(Path(work["work_path"]), work)
                receipts = []
                for index, command in enumerate(frozen_plan):
                    e.progress("validation", index, len(frozen_plan) - index)
                    receipts.append(
                        child(
                            command["name"],
                            command["argv"],
                            raw / "validation",
                            deadline=command["deadline_s"],
                            scope=command["scope"],
                        )
                    )
                for name in ("coverage.ini", "coverage.json", ".coverage"):
                    source = private / name
                    if source.is_file():
                        target = raw / name
                        target.write_bytes(source.read_bytes())
                        work["code_config_hashes"].append(e.reference(target))
                        if name == "coverage.json":
                            report = json.loads(target.read_bytes())
                            work["owned_coverage"] = dict(
                                report["totals"], reference=e.reference(target)
                            )
                work["coverage_shards"] = []
                for source in private.glob(".coverage.*"):
                    target = raw / "coverage_shards" / source.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read_bytes())
                    work["coverage_shards"].append(e.reference(target))
                atomic_json(Path(work["work_path"]), work)
                candidate = private / (e.NAME + ".json")
                atomic_json(candidate, e.build(work, receipts))
                receipts.extend(e.controls(candidate, raw / "controls"))
                receipts.extend(e.state_controls(work, raw / "state_controls"))
                if work["historical_consumer_receipt"]:
                    receipts.append(
                        child(
                            "original_cold_replay",
                            [
                                sys.executable,
                                "-u",
                                str(e.ROOT / e.CLI.replace("8404_v724", "8390_v723")),
                                "--cold-replay",
                                str(e.ROOT / e.ORIGINAL),
                            ],
                            raw / "original_controls",
                            deadline=120,
                        )
                    )
                initial = old.runner.validate(candidate, raw / "initial_terminal")
                receipts.extend(initial["checks"])
                work["adversarial_findings"] = initial["adversarial"]["findings"]
                work["duration_s"] = time.monotonic() - began
                work["phase_spans"].extend(
                    dict(phase=r["name"], duration_s=r["duration_s"]) for r in receipts
                )
                atomic_json(Path(work["work_path"]), work)
                value = e.build(work, receipts)
                publication = publish_primary(
                    args.output, value, lambda p: old.runner.validate(p, raw / "terminal")
                )
                atomic_json(Path(work["terminal_validation_sidecar_path"]), publication)
                e.progress("published_" + value["verdict_class"], 1, 0)
                return int(not value["required_checks_passed"])
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            e.progress("owned_failure_" + type(error).__name__ + "_" + str(error))
            return 1
        finally:
            signal.setitimer(signal.ITIMER_REAL, *previous)

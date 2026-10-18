"""REQ-VERIFY-8537: use unchanged bounded validators and publication consumers."""

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

from carnot.reporting import arc_capability_panel_8537 as e
from carnot.reporting import arc_effect_panel_runner_8523 as shipped
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def validate_checks(raw: Path, private: Path) -> list[Json]:
    """Preserve original collected E2E nodes and scope coverage to new files."""
    with (
        patch.multiple(shipped.old, e=e, TESTS=e.TESTS),
        patch.multiple(
            shipped,
            e=e,
            HISTORICAL=[
                *shipped.old.HISTORICAL,
                "tests/python/test_arc_capability_8536.py",
                "tests/python/test_arc_capability_qualification_8536.py",
                "tests/python/test_arc_capability_runner_8536.py",
            ],
        ),
    ):
        return list(shipped.validate_checks(raw, private))


def controls(value: Json, raw: Path) -> list[Json]:
    """The original direct-receipt tamper is also invalid in two applied arms."""
    with patch.object(shipped, "e", e):
        return list(shipped.controls(value, raw))


def publish(value: Json, output: Path, raw: Path) -> Json:
    """Validate exact candidate bytes before atomic publication and actual reading."""

    def reader(results: Path) -> Json:
        return dict(
            reader_receipt(
                e.TASK,
                results,
                field="arc_capability_panel_ready_score",
                expected=value["arc_capability_panel_ready_score"],
            )
        )

    def validate(candidate: Path) -> Json:
        checks = [
            child(name, argv, raw / "terminal", deadline=240 if name == "terminal_replay" else 90)
            for name, argv in [
                (
                    "terminal_replay",
                    [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(candidate)],
                ),
                (
                    "adversarial",
                    [sys.executable, "scripts/adversarial_verify.py", "--json", str(candidate)],
                ),
                (
                    "strict_rows",
                    [
                        sys.executable,
                        "scripts/verdict_row_consistency_lint.py",
                        "--strict",
                        str(candidate),
                    ],
                ),
            ]
        ]
        with TemporaryDirectory(prefix="exp8537-consumer-", dir="/var/tmp") as directory:
            primary = Path(directory) / output.name
            primary.write_bytes(candidate.read_bytes())
            consumer = reader(primary.parent)
        report = dict(
            passed=all(r["passed"] for r in checks) and consumer["passed"],
            checks=checks,
            consumer_before_publication=consumer,
            exact_candidate_sha256=sha256_file(candidate),
        )
        atomic_json(raw / "terminal_report.json", report)
        return report

    publication = publish_primary(output, value, validate)
    bound = read_bound_sidecar(output, Path(publication["sidecar_path"]))
    consumer = reader(output.parent)
    side = dict(
        publication=publication,
        terminal=bound["report"],
        consumer=consumer,
        passed=bound["report"]["passed"] and consumer["passed"],
    )
    atomic_json(Path(value["terminal_validation_sidecar_path"]), side)
    return side


def main(argv: list[str] | None = None) -> int:
    """Bound collection to3600 seconds and reserve validation before the4800 cap."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261015"], default="20261015")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.episode:
        return e.episode(args.episode)
    if args.cold_replay:
        try:
            passed = e.replay(json.loads(args.cold_replay.read_bytes()))
        except (OSError, ValueError):
            passed = False
        e.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    started = time.monotonic()
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    output = args.output.absolute()
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(prefix="exp8537-", dir="/var/tmp") as directory:
            work = e.measure(args.root, raw, Path(directory))
            path = raw / "work.json"
            atomic_json(path, work)
            e.progress("validation_before")
            receipts = validate_checks(raw, Path(directory) / "validation")
            receipts.extend(controls(e.build(path, receipts, output), raw))
            work["phase_spans"].append(
                dict(phase="validation", duration_s=time.monotonic() - started - work["duration_s"])
            )
            work["duration_s"] = time.monotonic() - started
            atomic_json(path, work)
            value = e.build(path, receipts, output)
            e.progress("publication_before")
            side = publish(value, output, raw)
            e.progress("publication_after", value["completed_count"], 0)
            return int(not side["passed"])
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)

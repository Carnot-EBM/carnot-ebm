"""REQ-VERIFY-8523: bounded child evidence qualifies exact terminal publication."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import arc_effect_panel_8523 as e
from carnot.reporting import arc_redirect_runner_8509 as old
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
HISTORICAL = [
    *old.HISTORICAL,
    "tests/python/test_arc_decision_telemetry.py",
    "tests/python/test_experiment_7491_e6_timed_live_profile.py",
    "tests/python/test_experiment_7531_b2_induction_gate_measurement.py",
    "tests/python/test_semif_arc_readout_eval.py",
    "tests/python/test_arc_effect_receipts_8522.py",
]


def validate_checks(raw: Path, private: Path) -> list[Json]:
    """Collect every historical node before executing disjoint unchanged assertion shards."""
    collected = {
        f"collect_{i}": child(
            f"collect_{i}",
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                suite,
            ],
            raw / "validation",
            deadline=90,
        )
        for i, suite in enumerate(HISTORICAL)
    }
    if any(not r["passed"] for r in collected.values()):
        return list(collected.values())

    def checked_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
        return collected[name] if name in collected else dict(child(name, argv, logs, **kwargs))

    with patch.multiple(old, e=e, TESTS=e.TESTS, HISTORICAL=HISTORICAL, child=checked_child):
        return list(old.validate_checks(raw, private))


def controls(value: Json, raw: Path) -> list[Json]:
    """Rehashed terminal claims still fail independent reduction in fresh children."""
    work = deepcopy(json.loads(Path(value["work_reference"]["path"]).read_bytes()))
    changed = deepcopy(value)
    if work["references"]:
        ref = work["references"][0]
        directory = raw / "tampered_episode"
        directory.mkdir(parents=True, exist_ok=True)
        original = Path(ref["directory"])
        terminal = json.loads((original / "episode.json").read_bytes())
        terminal["supervisor_receipt"]["direct_effect_receipts"][0]["applied"] = True
        ref["directory"] = str(directory)
        ref["unit"]["directory"] = str(directory)
        terminal["unit"] = ref["unit"]
        atomic_json(directory / "episode.json", terminal)
        (directory / "steps.jsonl").write_bytes((original / "steps.jsonl").read_bytes())
        ref["terminal_sha256"] = sha256_file(directory / "episode.json")
        ref["steps_sha256"] = sha256_file(directory / "steps.jsonl")
    else:
        work["failures"] = []
    atomic_json(raw / "tampered_work.json", work)
    changed["work_reference"] = dict(
        path=str(raw / "tampered_work.json"), sha256=sha256_file(raw / "tampered_work.json")
    )
    changed["reproducibility_checksum"] = e.canonical_hash(work)
    atomic_json(raw / "self_consistently_rehashed_tamper.json", changed)
    return [
        child(
            name,
            [sys.executable, "-u", str(e.ROOT / e.CLI), *args],
            raw / "controls",
            deadline=90,
            expected=expected,
        )
        for name, args, expected in [
            ("missing", ["--cold-replay", str(raw / "absent.json")], 1),
            ("intentional", ["--deliberate-error"], 7),
            (
                "rehashed_tamper",
                ["--cold-replay", str(raw / "self_consistently_rehashed_tamper.json")],
                1,
            ),
        ]
    ]


def publish(value: Json, output: Path, raw: Path) -> Json:
    """The unchanged validators see terminal bytes before the unchanged consumers do."""

    def validate(candidate: Path) -> Json:
        checks = [
            child(name, argv, raw / "terminal", deadline=90)
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
        with TemporaryDirectory(prefix="exp8523-consumer-", dir="/var/tmp") as directory:
            primary = Path(directory) / output.name
            primary.write_bytes(candidate.read_bytes())
            consumer = reader_receipt(
                e.TASK,
                primary.parent,
                field="arc_effect_panel_ready_score",
                expected=value["arc_effect_panel_ready_score"],
            )
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
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="arc_effect_panel_ready_score",
        expected=value["arc_effect_panel_ready_score"],
    )
    side = dict(
        publication=publication,
        terminal=bound["report"],
        consumer=consumer,
        passed=bound["report"]["passed"] and consumer["passed"],
    )
    atomic_json(Path(value["terminal_validation_sidecar_path"]), side)
    return side


def main(argv: list[str] | None = None) -> int:
    """The dated parent bounds measurement and reserves time for required validation."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261014"], default="20261014")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--episode", type=Path)
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
        with TemporaryDirectory(prefix="exp8523-", dir="/var/tmp") as directory:
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

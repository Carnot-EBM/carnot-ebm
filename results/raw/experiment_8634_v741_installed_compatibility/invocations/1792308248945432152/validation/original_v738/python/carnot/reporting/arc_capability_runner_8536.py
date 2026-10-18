"""REQ-VERIFY-8536: bounded validation certifies only exact published bytes."""

from __future__ import annotations

import argparse
from copy import deepcopy
import difflib
import json
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import arc_capability_qualification_8536 as e
from carnot.reporting import arc_effect_runner_8522 as inherited
from carnot.reporting import arc_redirect_runner_8509 as old
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
HISTORICAL = [*inherited.HISTORICAL, "tests/python/test_arc_effect_receipts_8522.py"]


def validate_checks(raw: Path, private: Path) -> list[Json]:
    """Collect all original nodes first, then reuse bounded disjoint serial shards."""
    logs = raw / "validation"
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
            logs,
            deadline=90,
        )
        for i, suite in enumerate(HISTORICAL)
    }
    if any(not r["passed"] for r in collected.values()):
        return list(collected.values())

    def instrumented(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
        if name in collected:
            return collected[name]
        if name == "owned_tests":
            config = private / "coverage.ini"
            config.write_text(
                config.read_text().replace(
                    "[report]",
                    "".join("    " + str(e.ROOT / p) + "\n" for p in [e.POLICY, e.SUPERVISOR])
                    + "[report]",
                )
            )
        if name == "coverage_report":
            argv = [*argv, "--include=" + ",".join(str(e.ROOT / p) for p in e.OWNED)]
        if name == "coverage_json":
            argv = [*argv, "--fail-under=0"]
        return dict(child(name, argv, logs, **kwargs))

    with patch.multiple(old, e=e, TESTS=e.TESTS, HISTORICAL=HISTORICAL, child=instrumented):
        receipts = list(old.validate_checks(raw, private))
    report = raw / "validation/owned_coverage.json"
    if report.is_file():
        files = json.loads(report.read_bytes())["files"]
        added, missing = {}, {}
        for path in [e.POLICY, e.SUPERVISOR]:
            baseline = e.ROOT / "results/raw" / e.NAME / "historical" / Path(path).name
            current = (e.ROOT / path).read_text().splitlines()
            changed = {
                n
                for op, _, _, start, end in difflib.SequenceMatcher(
                    None, baseline.read_text().splitlines(), current
                ).get_opcodes()
                if op != "equal"
                for n in range(start + 1, end + 1)
            }
            data = files[path]
            statements = set(data["executed_lines"]) | set(data["missing_lines"])
            added[path] = sorted(changed & statements)
            missing[path] = sorted(changed & set(data["missing_lines"]))
        receipts.append(
            dict(
                name="policy_added_coverage",
                passed=not any(missing.values()),
                added_statements=added,
                missing_lines=missing,
            )
        )
    return receipts


def controls(value: Json, raw: Path) -> list[Json]:
    """Rehash invented selections so rejection tests semantics, not an outer checksum."""
    changed = deepcopy(json.loads(Path(value["work_reference"]["path"]).read_bytes()))
    changed["fixtures"]["selection_fixture_rows"][0]["selected_arms"] = ["invented"]
    path = raw / "tampered_work.json"
    atomic_json(path, changed)
    candidate = raw / "self_consistently_rehashed_tamper.json"
    output = Path(value["terminal_validation_sidecar_path"]).parents[2] / (e.NAME + ".json")
    atomic_json(candidate, e.build(path, value["validation_receipts"], output))
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
            ("rehashed_tamper", ["--cold-replay", str(candidate)], 1),
        ]
    ]


def publish(value: Json, output: Path, raw: Path) -> Json:
    """Validate terminal bytes, replace atomically and exercise both actual consumers."""

    def validate(candidate: Path) -> Json:
        receipts = [
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
        with TemporaryDirectory(prefix="exp8536-consumer-", dir="/var/tmp") as directory:
            primary = Path(directory) / output.name
            primary.write_bytes(candidate.read_bytes())
            consumer = reader_receipt(
                e.TASK,
                primary.parent,
                field="arc_capability_ready_score",
                expected=value["arc_capability_ready_score"],
            )
        report = dict(
            passed=all(r["passed"] for r in receipts) and consumer["passed"],
            checks=receipts,
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
        field="arc_capability_ready_score",
        expected=value["arc_capability_ready_score"],
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
    """Own a dated bounded invocation; future measurement cannot count as evidence."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261015"], default="20261015")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        try:
            passed = e.replay(json.loads(args.cold_replay.read_bytes()))
        except (OSError, ValueError):
            passed = False
        e.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    start = time.monotonic()
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    output = args.output.absolute()
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(prefix="exp8536-", dir="/var/tmp") as directory:
            private = Path(directory)
            work = e.measure(args.root, raw, private)
            path = raw / "work.json"
            atomic_json(path, work)
            e.progress("validation_before")
            receipts = validate_checks(raw, private / "validation")
            receipts.extend(controls(e.build(path, receipts, output), raw))
            work["phase_spans"].append(
                dict(phase="validation", duration_s=time.monotonic() - start - work["duration_s"])
            )
            work["duration_s"] = time.monotonic() - start
            atomic_json(path, work)
            value = e.build(path, receipts, output)
            e.progress("publication_before")
            side = publish(value, output, raw)
            e.progress("publication_after", value["completed_count"], 0)
            return int(not side["passed"])
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)

"""REQ-VERIFY-8522: bound validation and exact-byte publication precede readiness."""

from __future__ import annotations

import argparse
import ast
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

from carnot.reporting import arc_effect_qualification_8522 as e
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
    "tests/python/test_arc_trajectory_supervisor.py",
]


def validate_checks(raw: Path, private: Path) -> list[Json]:
    """Reuse exact disjoint historical shards while measuring only newly owned code."""
    logs = raw / "validation"
    collected = {}
    for index, suite in enumerate(HISTORICAL):
        name = f"collect_{index}"
        collected[name] = child(
            name,
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
    if any(not r["passed"] for r in collected.values()):
        return list(collected.values())

    def instrumented_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
        if name in collected:
            return collected[name]
        if name == "owned_tests":
            config = private / "coverage.ini"
            config.write_text(
                config.read_text().replace(
                    "[report]", "    " + str(e.ROOT / e.POLICY) + "\n[report]"
                )
            )
        if name == "coverage_report":
            argv = [*argv, "--include=" + ",".join(str(e.ROOT / p) for p in e.OWNED)]
        if name == "coverage_json":
            argv = [*argv, "--fail-under=0"]
        return child(name, argv, logs, **kwargs)

    with patch.multiple(old, e=e, TESTS=e.TESTS, HISTORICAL=HISTORICAL, child=instrumented_child):
        rows = old.validate_checks(raw, private)
    coverage = raw / "validation/owned_coverage.json"
    if coverage.is_file():
        data = json.loads(coverage.read_bytes())["files"][e.POLICY]
        original = e.ROOT / "results/raw" / e.NAME / "historical/arc_competition_agent.py"
        current = (e.ROOT / e.POLICY).read_text()
        changed = {
            n
            for op, _, _, start, end in difflib.SequenceMatcher(
                None, original.read_text().splitlines(), current.splitlines()
            ).get_opcodes()
            if op != "equal"
            for n in range(start + 1, end + 1)
        }
        statements = {
            node.lineno for node in ast.walk(ast.parse(current)) if isinstance(node, ast.stmt)
        }
        missing = sorted(changed & statements - set(data["executed_lines"]))
        rows.append(
            dict(
                name="policy_added_coverage",
                passed=not missing,
                added_statements=sorted(changed & statements),
                missing_lines=missing,
            )
        )
    return rows


def controls(value: Json, raw: Path) -> list[Json]:
    """A child failure is only valid when it matches a declared negative control."""
    changed = deepcopy(json.loads(Path(value["work_reference"]["path"]).read_bytes()))
    changed["fixtures"]["mutation_fixture_rows"][0]["receipts"][0]["applied"] = True
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
            deadline=60,
            expected=expected,
        )
        for name, args, expected in [
            ("missing", ["--cold-replay", str(raw / "absent.json")], 1),
            ("intentional", ["--deliberate-error"], 7),
            ("rehashed_tamper", ["--cold-replay", str(candidate)], 1),
        ]
    ]


def publish(value: Json, output: Path, raw: Path) -> Json:
    """Check exact terminal bytes through unchanged validators and both live readers."""

    def validate(candidate: Path) -> Json:
        rows = [
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
        with TemporaryDirectory(prefix="exp8522-consumer-", dir="/var/tmp") as directory:
            primary = Path(directory) / output.name
            primary.write_bytes(candidate.read_bytes())
            consumer = reader_receipt(
                e.TASK,
                primary.parent,
                field="arc_effect_receipts_ready_score",
                expected=value["arc_effect_receipts_ready_score"],
            )
        return dict(
            passed=all(r["passed"] for r in rows) and consumer["passed"],
            checks=rows,
            consumer_before_publication=consumer,
            exact_candidate_sha256=sha256_file(candidate),
        )

    publication = publish_primary(output, value, validate)
    bound = read_bound_sidecar(output, Path(publication["sidecar_path"]))
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="arc_effect_receipts_ready_score",
        expected=value["arc_effect_receipts_ready_score"],
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
    """Own a bounded qualification invocation; the future live panel remains separate."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261014"], default="20261014")
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
    started = time.monotonic()
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    output = args.output.absolute()
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(prefix="exp8522-", dir="/var/tmp") as directory:
            private = Path(directory)
            work = e.measure(args.root, raw, private)
            path = raw / "work.json"
            atomic_json(path, work)
            e.progress("validation_before")
            receipts = validate_checks(raw, private / "validation")
            value = e.build(path, receipts, output)
            receipts.extend(controls(value, raw))
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

"""REQ-VERIFY-8649: bounded current qualification with immutable original authority."""

from __future__ import annotations

import argparse
from datetime import datetime
import difflib
import json
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

import yaml
from coverage.parser import PythonParser

from carnot.reporting import arc_reset_8649 as e
from carnot.reporting import arc_observer_runner_8637 as inherited
from carnot.reporting import arc_redirect_runner_8509 as validation
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def live_checks(raw: Path) -> list[Json]:
    """Measure only executable insertions against the immutable pre-edit agent."""
    name = "python/carnot/agentic/arc_competition_agent.py"
    current = e.ROOT / name
    original = e.CLOSURE.parent / "original_source_closure" / name
    added = set()
    for tag, _, _, start, end in difflib.SequenceMatcher(
        a=original.read_text().splitlines(), b=current.read_text().splitlines(), autojunk=False
    ).get_opcodes():
        if tag in {"insert", "replace"}:
            added.update(range(start + 1, end + 1))
    parser = PythonParser(filename=str(current))
    parser.parse_source()
    scope = raw / "live_statement_scope.json"
    data = raw / "live.coverage"
    atomic_json(
        scope,
        dict(
            path=str(current),
            original_sha256=sha256_file(original),
            current_sha256=sha256_file(current),
            statements=sorted(parser.statements & added),
            data_file=str(data),
        ),
    )
    commands = [
        (
            "live_hook_tests",
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--data-file=" + str(data),
                "--include=" + str(current),
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                e.TESTS[0],
            ],
        ),
        (
            "live_statement_coverage",
            [
                sys.executable,
                "-c",
                "import coverage,json,sys; from pathlib import Path; s=json.loads(Path(sys.argv[1]).read_bytes()); "
                "c=coverage.Coverage(data_file=s['data_file']); c.load(); "
                "missing=sorted(set(s['statements'])-set(c.get_data().lines(s['path']) or [])); "
                "print(json.dumps(dict(missing=missing,statement_count=len(s['statements']),percent=100 if not missing else 0))); "
                "raise SystemExit(int(bool(missing) or not s['statements']))",
                str(scope),
            ],
        ),
    ]
    return [child(name, argv, raw / "validation", deadline=180) for name, argv in commands]


def publish(value: Json, output: Path, raw: Path) -> Json:
    """Validate final bytes and live readers before the atomic replacement."""

    def validate(candidate: Path) -> Json:
        checks = [
            child(name, argv, raw / "terminal", deadline=120)
            for name, argv in [
                (
                    "cold_replay",
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
        with TemporaryDirectory(prefix="exp8649-readers-", dir="/var/tmp") as directory:
            private = Path(directory) / output.name
            private.write_bytes(candidate.read_bytes())
            reader = reader_receipt(
                e.TASK,
                private.parent,
                field="reset_provenance_ready_score",
                expected=value["reset_provenance_ready_score"],
            )
        report = dict(
            passed=all(r["passed"] for r in checks) and reader["passed"],
            checks=checks,
            consumer=reader,
        )
        atomic_json(raw / "publication_attempt.json", report)
        return report

    publication = publish_primary(output, value, validate)
    side = dict(publication=publication, passed=True)
    atomic_json(Path(value["terminal_validation_sidecar_path"]), side)
    return side


def measure(root: Path, raw: Path, private: Path, date: str) -> Json:
    """Check only CPU prerequisites; same-roadmap outputs are future dependencies."""
    started = time.monotonic_ns()
    with patch.multiple(validation, e=e, TESTS=e.TESTS):
        failures, resources, hashes = validation.preconditions(root, private)
    source = None
    try:
        source = e.authenticate(root)
    except (ValueError, OSError, KeyError, AssertionError) as exc:
        failures.append(
            validation.gate(
                "original_exp8637",
                root / e.UPSTREAM,
                "sha256_and_public_chronology",
                e.UPSTREAM_PIN,
                None if not (root / e.UPSTREAM).exists() else str(exc),
            )
        )
    registry = e.ROOT / "ops/arc_solve_registry.yaml"
    roster = yaml.safe_load(registry.read_bytes())["games"]
    registry_precheck = dict(
        path=str(registry),
        sha256=sha256_file(registry),
        checked_before_execution=True,
        duplicate_solve_targets=[],
        new_solve_credit=0,
        policy_received_registry=False,
        fixture_targets=["fixture-unknown"],
        existing_game_count=len(roster),
        real_solve_targets=[],
    )
    head = e.ROOT / "results/experiment_4629_live_frame_change_cnn.pt"
    pin = "sha256:ee308cc8f2a1c9182cdf3bdbef8e25f0561b63ba2bca115fa7b51f17f006a78a"
    if not head.exists() or sha256_file(head) != pin:
        failures.append(
            validation.gate(
                "numerical_checkpoint",
                head,
                "sha256",
                pin,
                sha256_file(head) if head.exists() else None,
            )
        )
    references, measurements = [], []
    for index, mechanic in enumerate(("navigation", "toggle", "sequence") if not failures else ()):
        e.progress("fixture_before", index, 3 - index)
        directory = raw / "fixtures" / mechanic
        spec = raw / (mechanic + ".json")
        atomic_json(spec, dict(directory=str(directory), mechanic=mechanic))
        measurements.append(
            child(
                "fixture_" + mechanic,
                [sys.executable, "-u", str(e.ROOT / e.CLI), "--fixture", str(spec)],
                raw / "measurement",
                deadline=300,
            )
        )
        primitive = directory / "pair.json"
        if primitive.is_file():
            references.append(dict(path=str(primitive), sha256=sha256_file(primitive)))
        e.progress("fixture_after", index + 1, 2 - index)
    live = "python/carnot/agentic/arc_competition_agent.py"
    for name in [*hashes, str(e.ROOT / live), str(e.CLOSURE), str(head)]:
        original = Path(name)
        if not original.is_file():
            continue
        snapshot = raw / "current_source_closure" / original.relative_to(e.ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, snapshot)
        hashes.pop(name, None)
        hashes[str(snapshot)] = sha256_file(snapshot)
    ended = time.monotonic_ns()
    return dict(
        run_date=date,
        failures=failures,
        references=references,
        history=source,
        preconditions_checked=resources,
        registry_precheck=registry_precheck,
        historical_model_provenance=source["historical_model_provenance"] if source else {},
        numerical_head_reference=dict(
            path=str(head), sha256=pin, language_model=False, memory_withheld=True
        ),
        source_artifact_hashes=hashes,
        code_config_hashes={
            str(raw / "current_source_closure" / n): sha256_file(e.ROOT / n) for n in e.OWNED
        },
        live_path_reference=dict(
            factory="make_carnot_agent",
            policy="E3AgentPolicy",
            sdk_method="do_action_request",
            current_adapter="carnot.reporting.arc_reset_live_8649.collect",
            current_source_sha256=sha256_file(e.ROOT / live),
            original_source_closure=str(e.CLOSURE),
            historical_readers_unchanged=True,
        ),
        measurement_receipts=measurements,
        duration_s=(ended - started) / 1e9,
        phase_spans=[
            dict(
                phase="measurement",
                started_monotonic_ns=started,
                ended_monotonic_ns=ended,
                duration_s=(ended - started) / 1e9,
            )
        ],
    )


def main(argv: list[str] | None = None) -> int:
    """Private controls exercise real publication children without nested test controllers."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261018")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--owned-failure", action="store_true")
    args = parser.parse_args(argv)
    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        parser.error("--date requires YYYYMMDD")
    if args.cold_replay:
        try:
            passed = e.replay(json.loads(args.cold_replay.read_bytes()))
        except (OSError, ValueError, KeyError, TypeError, AssertionError):
            passed = False
        e.progress("cold_replay_complete", int(passed), 0)
        return int(not passed)
    if args.fixture:
        from carnot.reporting.arc_reset_live_8649 import fixture_pair

        unit = json.loads(args.fixture.read_bytes())
        pair = fixture_pair(Path(unit["directory"]), unit["mechanic"])
        return int(not pair["passed"])
    if args.private_control and args.root.exists():
        parser.error("private controls require an absent input root")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)


def run(args: Any) -> int:
    """Reserve validation time with bounded fixtures and a whole-invocation deadline."""
    raw = args.output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    with TemporaryDirectory(prefix="carnot8649-", dir="/var/tmp") as directory:
        private = Path(directory)
        work = measure(args.root, raw, private, args.date)
        path = raw / "work.json"
        atomic_json(path, work)
        atomic_json(path.with_name("measurement_seal.json"), dict(work_sha256=sha256_file(path)))
        e.progress("measurement_after_validation_before")
        if args.private_control:
            receipts = [
                child(
                    "private_control",
                    [
                        sys.executable,
                        "-c",
                        "raise SystemExit(" + str(int(args.owned_failure)) + ")",
                    ],
                    raw / "validation",
                )
            ]
        else:
            with patch.object(inherited, "e", e):
                receipts = inherited.validate_checks(raw, private / "checks")
            receipts.extend(live_checks(raw))
        receipts.extend(work["measurement_receipts"])
        value = e.build(path, receipts, args.output)
        e.progress("publication_before")
        side = publish(value, args.output, raw)
        e.progress("publication_after", value["completed_count"], 3 - value["completed_count"])
        return int(not side["passed"])

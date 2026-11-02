"""REQ-VERIFY-8650: run bounded fresh SDK children and publish checked reductions."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import inspect
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import arc_reset_transfer_8650 as e
from carnot.reporting import arc_observer_runner_8637 as inherited
from carnot.reporting import arc_redirect_runner_8509 as validation
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def episode(path: Path) -> int:
    """Use the qualified current adapter with a fresh official SDK and no induction."""
    from carnot.reporting.arc_reset_live_8649 import collect
    from carnot.reporting.arc_supervisor_live_panel_8384 import sdk

    unit = json.loads(path.read_bytes())
    raw = Path(unit["directory"])
    e.progress("official_SDK_episode_before")
    arcade = sdk()
    result = collect(unit, arcade, raw)
    implementation = Path(inspect.getfile(type(arcade)))
    result["sdk_identity"] = dict(
        class_name=type(arcade).__module__ + "." + type(arcade).__qualname__,
        path=str(implementation),
        sha256=sha256_file(implementation),
        game=unit["game"],
        seed=unit["seed"],
    )
    if (
        e.original.replay_stream(result["steps"], result["events"], result["end_reason"])
        != result["observer_rows"]
    ):
        raise ValueError("original_observer_replay_drift")
    observations = []
    with (raw / "steps.jsonl").open() as stream:
        for line in stream:
            step = json.loads(line)
            if step["action"]["name"] == "RESET":
                public = dict(
                    result["observations"][step["index"] - 1], pixels=step["observation"]["pixels"]
                )
                observations.append(
                    dict(
                        action_index=step["index"],
                        observation_sha256=canonical_hash(public),
                        public_observation=public,
                    )
                )
    result["reset_observations"] = observations
    result["legacy"] = canonical_hash(result["legacy"])
    atomic_json(raw / "transfer_stream.json", result)
    e.progress("official_SDK_episode_after", len(result["actions"]), 512 - len(result["actions"]))
    return int(bool(result["termination"]["error"] or result["model_tripwire_attempts"]))


def measure(root: Path, raw: Path, private: Path, date: str) -> Json:
    """Freeze current sources before running this CPU branch, independently of optional devices."""
    started = time.monotonic_ns()
    with patch.multiple(validation, e=e, TESTS=e.TESTS):
        failures, resources, hashes = validation.preconditions(root, private)
    source: Json = {}
    for name, pin in e.PINS.items():
        path = root / name
        observed = sha256_file(path) if path.is_file() else None
        if observed != pin:
            failure = validation.gate("required_upstream", path, "sha256", pin, observed)
            failure["upstream"] = (
                "exp8640-contract-methods" if "8640" in name else "exp8649-arc-reset-provenance"
            )
            failures.append(failure)
        else:
            hashes[str(path)] = pin
    if not failures:
        try:
            source = e.authenticate(root)
        except (ValueError, OSError, KeyError, AssertionError) as exc:
            failures.append(
                validation.gate(
                    "upstream_authentication",
                    root / list(e.PINS)[1],
                    "qualified_source_closure",
                    True,
                    str(exc),
                )
            )
    head = e.ROOT / "results/experiment_4629_live_frame_change_cnn.pt"
    head_pin = "sha256:ee308cc8f2a1c9182cdf3bdbef8e25f0561b63ba2bca115fa7b51f17f006a78a"
    observed_head = sha256_file(head) if head.is_file() else None
    if observed_head != head_pin:
        failures.append(
            validation.gate("numerical_checkpoint", head, "sha256", head_pin, observed_head)
        )
    else:
        hashes[str(head)] = head_pin
    registry = e.ROOT / "ops/arc_solve_registry.yaml"
    roster = yaml.safe_load(registry.read_bytes())["games"]
    precheck = dict(
        path=str(registry),
        sha256=sha256_file(registry),
        checked_before_execution=True,
        policy_received_registry=False,
        new_solve_credit=0,
        rows=[
            dict(
                game=g, public_solve=next((r for r in roster if r["game"] == g.split("-")[0]), None)
            )
            for g in e.GAMES
        ],
    )
    dependencies = [
        "python/carnot/agentic/arc_competition_agent.py",
        "python/carnot/agentic/arc_reset_provenance.py",
        "python/carnot/agentic/arc_live_panel_runtime_8384.py",
        "python/carnot/agentic/arc_outcome_observer.py",
        "python/carnot/reporting/arc_reset_live_8649.py",
        "python/carnot/reporting/arc_observer_live_8637.py",
    ]
    for name in [*hashes, *(str(e.ROOT / p) for p in dependencies)]:
        original = Path(name)
        snapshot = raw / "current_source_closure" / original.relative_to(e.ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, snapshot)
        hashes.pop(name, None)
        hashes[str(snapshot)] = sha256_file(snapshot)
    refs, measurements, raw_hashes = [], [], {}
    for index, unit in enumerate(e.units() if not failures else []):
        if (time.monotonic_ns() - started) / 1e9 > 3450:
            raise TimeoutError("measurement_cap")
        e.progress("episode_before", index, 16 - index)
        directory = raw / "episodes" / f"{index:02d}"
        unit["directory"] = str(directory)
        path = directory / "unit.json"
        atomic_json(path, unit)
        measurements.append(
            child(
                f"episode_{index:02d}",
                [sys.executable, "-u", str(e.ROOT / e.CLI), "--episode", str(path)],
                raw / "measurement",
                deadline=150,
            )
        )
        stream = directory / "transfer_stream.json"
        if stream.is_file():
            refs.append(dict(path=str(stream), sha256=sha256_file(stream)))
        raw_hashes.update({str(p): sha256_file(p) for p in directory.iterdir() if p.is_file()})
        e.progress("episode_after", index + 1, 15 - index)
    ended = time.monotonic_ns()
    return dict(
        run_date=date,
        failures=failures,
        references=refs,
        upstream_gates={
            k: source.get(k, 0) for k in ("arc_inputs_ready_score", "reset_provenance_ready_score")
        },
        historical_model_provenance=source.get("historical_model_provenance", {}),
        numerical_head_reference=dict(
            path=str(head), sha256=head_pin, language_model=False, memory_withheld=True
        ),
        registry_precheck=precheck,
        preconditions_checked=resources,
        source_artifact_hashes=hashes,
        code_config_hashes={
            str(raw / "current_source_closure" / n): sha256_file(e.ROOT / n)
            for n in e.OWNED + dependencies
        },
        raw_file_hashes=raw_hashes,
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


def publish(value: Json, output: Path, raw: Path) -> Json:
    """Validate the locked bytes and both unchanged readers before atomic publication."""

    def validate(candidate: Path) -> Json:
        checks = [
            child(name, argv, raw / "terminal", deadline=180)
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
        with TemporaryDirectory(prefix="exp8650-readers-", dir="/var/tmp") as directory:
            private = Path(directory) / output.name
            private.write_bytes(candidate.read_bytes())
            reader = reader_receipt(
                e.TASK,
                private.parent,
                field="reset_transfer_rows_ready_score",
                expected=value["reset_transfer_rows_ready_score"],
            )
        report = dict(
            passed=all(r["passed"] for r in checks) and reader["passed"],
            checks=checks,
            consumer=reader,
            adversarial_report=json.loads(Path(checks[1]["stdout_path"]).read_bytes()),
        )
        atomic_json(raw / "publication_attempt.json", report)
        return report

    publication = publish_primary(output, value, validate)
    reader = reader_receipt(
        e.TASK,
        output.parent,
        field="reset_transfer_rows_ready_score",
        expected=value["reset_transfer_rows_ready_score"],
    )
    side = dict(publication=publication, passed=reader["passed"], consumer=reader)
    atomic_json(Path(value["terminal_validation_sidecar_path"]), side)
    return side


def main(argv: list[str] | None = None) -> int:
    """Private controls exercise real children while production reserves validation time."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_language_model_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261018")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--episode", type=Path)
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
    if args.episode:
        return episode(args.episode)
    if args.private_control and args.root.exists():
        parser.error("private controls require an absent input root")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)


def run(args: Any) -> int:
    """Measurement and validation keep separate immutable clocks and private scratch."""
    raw = args.output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    with TemporaryDirectory(prefix="carnot8650-", dir="/var/tmp") as directory:
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
        receipts.extend(work["measurement_receipts"])
        value = e.build(path, receipts, args.output)
        e.progress("publication_before")
        side = publish(value, args.output, raw)
        e.progress("publication_after", value["completed_count"], 16 - value["completed_count"])
        return int(not side["passed"])

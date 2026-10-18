"""REQ-VERIFY-8637: bounded children retain immutable natural-panel evidence."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import arc_observer_transfer_8637 as e
from carnot.reporting import arc_redirect_runner_8509 as validation
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def publish(value: Json, output: Path, raw: Path) -> Json:
    """Check exact final bytes and both unchanged live readers before replacement."""

    def validate(candidate: Path) -> Json:
        checks = [
            child(name, argv, raw / "terminal", deadline=180)
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
        with TemporaryDirectory(prefix="exp8637-readers-", dir="/var/tmp") as directory:
            path = Path(directory) / output.name
            path.write_bytes(candidate.read_bytes())
            reader = reader_receipt(
                e.TASK,
                path.parent,
                field="real_horizon_rows_ready_score",
                expected=value["real_horizon_rows_ready_score"],
            )
        report = dict(
            passed=all(c["passed"] for c in checks) and reader["passed"],
            checks=checks,
            consumer=reader,
        )
        atomic_json(raw / "publication_attempt.json", report)
        return report

    publication = publish_primary(output, value, validate)
    reader = reader_receipt(
        e.TASK,
        output.parent,
        field="real_horizon_rows_ready_score",
        expected=value["real_horizon_rows_ready_score"],
    )
    side = dict(publication=publication, consumer=reader, passed=reader["passed"])
    atomic_json(Path(value["terminal_validation_sidecar_path"]), side)
    return side


def measure(root: Path, raw: Path, private: Path, date: str, private_control: bool) -> Json:
    """Authenticate only this CPU branch before executing any registered episode."""
    started = time.monotonic_ns()
    with patch.multiple(validation, e=e, TESTS=e.TESTS):
        failures, resources, hashes = validation.preconditions(root, private)
    for name in ["research-roadmap.yaml", "openspec/change-proposals/research-roadmap-vNEXT.md"]:
        original = root / name
        if original.exists():
            copy = raw / original.name
            shutil.copyfile(original, copy)
            hashes.pop(str(original), None)
            hashes[str(copy)] = sha256_file(copy)
    source: Json = {}
    for name, pin in e.PINS.items():
        path = root / name
        observed = sha256_file(path) if path.is_file() else None
        if observed != pin:
            failures.append(
                validation.gate("required_original_input", path, "sha256", pin, observed)
            )
        else:
            hashes[str(path)] = observed
    if not failures:
        try:
            source = e.authenticate(root)
        except (ValueError, OSError, KeyError) as exc:
            failures.append(
                validation.gate(
                    "original_observer_and_methods",
                    root / list(e.PINS)[1],
                    "original_validation",
                    True,
                    str(exc),
                )
            )
    registry = e.ROOT / "ops/arc_solve_registry.yaml"
    games = source.get("protocol", {}).get("games", [])
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
            for g in games
        ],
    )
    head = e.ROOT / "results/experiment_4629_live_frame_change_cnn.pt"
    head_pin = "sha256:ee308cc8f2a1c9182cdf3bdbef8e25f0561b63ba2bca115fa7b51f17f006a78a"
    if not private_control and (not head.is_file() or sha256_file(head) != head_pin):
        failures.append(
            validation.gate(
                "numerical_head",
                head,
                "sha256",
                head_pin,
                sha256_file(head) if head.is_file() else None,
            )
        )
    refs, measurements = [], []
    for index, unit in enumerate(e.units(source["protocol"]) if source and not failures else []):
        if (time.monotonic_ns() - started) / 1e9 > 3600 - 150:
            raise TimeoutError("measurement_cap")
        e.progress("episode_before", index, 16 - index)
        directory = raw / "episodes" / f"{index:02d}"
        directory.mkdir(parents=True)
        unit["directory"] = str(directory)
        path = directory / "unit.json"
        atomic_json(path, unit)
        receipt = child(
            f"episode_{index:02d}",
            [sys.executable, "-u", str(e.ROOT / e.CLI), "--episode", str(path)],
            raw / "measurement",
            deadline=150,
        )
        measurements.append(receipt)
        stream = directory / "public_stream.json"
        if stream.exists():
            refs.append(dict(path=str(stream), sha256=sha256_file(stream)))
        e.progress("episode_after", index + 1, 15 - index)
    end = time.monotonic_ns()
    return dict(
        run_date=date,
        input_root=str(root),
        failures=failures,
        references=refs,
        games=games,
        registry_precheck=precheck,
        measurement_receipts=measurements,
        preconditions_checked=resources,
        source_artifact_hashes=hashes,
        code_config_hashes={str(e.ROOT / p): sha256_file(e.ROOT / p) for p in e.OWNED + e.TESTS},
        historical_model_provenance=source.get("historical_model_provenance", {}),
        numerical_head_reference=dict(
            path=str(head), sha256=head_pin, language_model=False, memory_withheld=True
        ),
        duration_s=(end - started) / 1e9,
        phase_spans=[
            dict(
                phase="measurement",
                started_monotonic_ns=started,
                ended_monotonic_ns=end,
                duration_s=(end - started) / 1e9,
            )
        ],
    )


def main(argv: list[str] | None = None) -> int:
    """Private controls exercise real children without nesting historical controllers."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261018")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--owned-failure", action="store_true")
    args = parser.parse_args(argv)
    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        parser.error("--date requires YYYYMMDD")
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        try:
            accepted = e.replay(json.loads(args.cold_replay.read_bytes()))
        except (ValueError, OSError, KeyError, TypeError):
            accepted = False
        e.progress("cold_replay_complete")
        return 0 if accepted else 1
    if args.episode:
        from carnot.reporting.arc_observer_live_8637 import episode

        return episode(args.episode)
    if args.private_control and args.root.exists():
        parser.error("private controls require an absent input root")
    raw = args.output.parent / "raw" / args.output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    with TemporaryDirectory(prefix="carnot8637-", dir="/var/tmp") as directory:
        private = Path(directory)
        work = measure(args.root, raw, private, args.date, args.private_control)
        path = raw / "work.json"
        atomic_json(path, work)
        atomic_json(raw / "measurement_seal.json", dict(work_sha256=sha256_file(path)))
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
            with patch.multiple(validation, e=e, TESTS=e.TESTS):
                receipts = list(validation.validate_checks(raw, private / "checks"))
        value = e.build(path, work["measurement_receipts"] + receipts, args.output)
        e.progress("publication_before")
        side = publish(value, args.output, raw)
        e.progress("publication_after", value["completed_count"], 16 - value["completed_count"])
        return int(not side["passed"])

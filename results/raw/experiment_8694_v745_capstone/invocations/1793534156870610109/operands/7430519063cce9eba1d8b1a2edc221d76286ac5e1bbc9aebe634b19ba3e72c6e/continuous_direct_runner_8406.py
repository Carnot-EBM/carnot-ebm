"""REQ-REPORT-8406: bounded real children check terminal bytes before publication."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import continuous_direct_learning_8406 as e
from carnot.reporting import direct_atomic_runner_8376 as shipped
from carnot.reporting import v717_contract_runner as commands
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child, execute
from carnot.verify import continuous_direct_learning_8406 as k

Json = dict[str, Any]
SCRATCH = Path.home() / ".cache/carnot-exp8406-private"


def plan(private: Path) -> list[Json]:
    """Freeze scoped checks and real child coverage without launching global repository health."""
    with patch.object(commands, "m", e):
        result = commands.manifest(private)
    config = private / "coverage.ini"
    config.write_text(config.read_text().replace("patch=subprocess", "patch=subprocess,_exit"))
    os.environ["COVERAGE_RCFILE"] = str(config)
    os.environ["COVERAGE_FILE"] = str(private / ".coverage")
    result[0]["deadline"] = 900
    combine = next(c for c in result if c["name"] == "coverage_combine")
    combine["argv"].append("--keep")
    for name, test in [
        ("E2E020", "test_hard_exit_learning_qualification_8206.py"),
        ("E2E021", "test_restricted_decision_audit_8210.py"),
    ]:
        result.insert(
            2,
            dict(
                name="private_" + name,
                argv=[
                    str(e.ROOT / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    "tests/python/" + test,
                    "--basetemp=" + str(private / name),
                ],
                expected=0,
                deadline=900,
                scope="owned",
            ),
        )
    source = e.ROOT / "results/experiment_8402_v724_historical_consumer_roots.json"
    fixture = json.loads(source.read_bytes())["historical_fixture_hashes"]
    historical = next(c for c in result if c["name"] == "private_E2E020")
    historical["argv"].extend(
        [
            "-p",
            "carnot.testing.historical_roots_8402",
            "--historical-fixture-manifest="
            + str(Path(fixture["720"]["root"]).parent / "manifest.json"),
            "--historical-current-root="
            + str(Path(fixture["720"]["root"]).parents[1] / "current/pinned"),
        ]
    )
    return list(result)


def validate(candidate: Path, raw: Path) -> Json:
    """The unchanged verifier and typed consumer decide whether findings are resolved."""
    with patch.object(shipped, "e", e):
        return dict(shipped.validate(candidate, raw))


def controls(candidate: Path, raw: Path) -> list[Json]:
    """A repaired hash cannot turn a changed coefficient into a causal observation."""
    with patch.object(shipped, "e", e):
        receipts = list(shipped.controls(candidate, raw))
    value = json.loads(candidate.read_bytes())
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    if work["streams"]:
        bad = deepcopy(value)
        original = Path(work["streams"][0]["directory"])
        destination = raw / "tampered_primitives"
        shutil.copytree(original, destination)
        step = destination / "steps/096.json"
        changed = json.loads(step.read_bytes())
        changed["state"]["head"]["coefficients"][2] += 0.1
        atomic_json(step, changed)
        atomic_json(destination / "final.json", changed["state"])
        work["streams"][0]["directory"] = str(destination)
        work["work_path"] = str(raw / "tampered_work.json")
        atomic_json(Path(work["work_path"]), work)
        bad["work_reference"] = e.reference(Path(work["work_path"]))
        for ref in bad["raw_shard_hashes"]:
            if Path(ref["path"]).is_relative_to(original):
                ref.update(e.reference(destination / Path(ref["path"]).relative_to(original)))
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        path = raw / "rehashed_primitive_tamper.json"
        atomic_json(path, bad)
        receipts.append(
            child(
                "rehashed_primitive_tamper",
                [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)],
                raw / "primitive_control",
                expected=1,
                deadline=60,
                heartbeat=20,
            )
        )
    return receipts


def main(argv: list[str] | None = None) -> int:
    """Use no LLM loads; process exits, validation and publication stay within the task cap."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261011"], default="20261011")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--state-directory", type=Path)
    parser.add_argument("--variant", choices=k.VARIANTS, default=k.VARIANTS[0])
    parser.add_argument("--crash", type=int, choices=[0, 32, 64], default=0)
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 1
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_failed")
        return int(not passed)
    if args.worker:
        if args.state_directory is None:
            parser.error("--worker requires --state-directory")
        k.run(
            json.loads(args.worker.read_bytes()),
            args.state_directory,
            args.variant,
            crash=args.crash,
        )
        return 0
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
                    target = raw / "coverage_shards" / source.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read_bytes())
                    work["coverage_shards"].append(e.reference(target))
            source = private / "coverage.json"
            if source.is_file():
                target = raw / "coverage.json"
                target.write_bytes(source.read_bytes())
                work["owned_coverage"] = dict(
                    json.loads(target.read_bytes())["totals"], reference=e.reference(target)
                )
            atomic_json(Path(work["work_path"]), work)
            candidate = private / (e.NAME + ".json")
            atomic_json(candidate, e.build(work, receipts))
            receipts.extend(controls(candidate, raw / "controls"))
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

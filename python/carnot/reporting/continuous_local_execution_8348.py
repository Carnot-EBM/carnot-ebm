"""REQ-REPORT-8348: freeze bounded validation before atomic primary publication."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import continuous_local_learning_8348 as e
from carnot.reporting import methods_stream_execution_8111 as commands
from carnot.reporting import v718_replay_runner as findings
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def cli() -> list[str]:
    """Owned workers and CLI controls save measured subprocess coverage too."""
    prefix = [str(e.ROOT / ".venv/bin/python"), "-u"]
    config = os.environ.get("COVERAGE_RCFILE")
    if config:
        prefix += ["-m", "coverage", "run", "--rcfile=" + config]
    return prefix + [str(e.ROOT / e.CLI)]


def manifest(private: Path, candidate: Path) -> Json:
    """Reuse the qualified scoped harness without invoking repository-wide tests."""
    with patch.object(commands, "e", e), patch.object(commands, "OWNED", e.OWNED):
        plan = commands.manifest(private, candidate)
    plan.pop("repository_health")
    config = private / "coverage.ini"
    config.write_text(config.read_text() + "patch = _exit\n")
    tests = plan["commands"][0]
    tests["argv"].remove("-s")
    tests["argv"].append("--basetemp=" + str(private / "owned-tests"))
    tests["deadline_s"] = 600
    consumers = plan["commands"][1]
    consumers.update(name="consumer_and_E2E020_021", deadline_s=600)
    consumers["argv"] = [
        str(e.ROOT / ".venv/bin/pytest"),
        "-n",
        "0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        "--basetemp=" + str(private / "consumers"),
        "tests/python/test_local_update_isolation_8306.py",
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_hard_exit_learning_qualification_8206.py",
        "tests/python/test_restricted_decision_audit_8210.py",
    ]
    plan["commands"][-1]["argv"].insert(2, "--files")
    return dict(
        plan,
        owned=e.OWNED,
        scientific_protocol_sha256=e.PINS[e.original.PROTOCOL],
        heartbeat_s=20,
        no_full_repository_suite=True,
    )


def check(spec: Json, raw: Path) -> Json:
    """Keep exact argv, exits, timing and durable log hashes for every child."""
    return child(
        spec["name"],
        spec["argv"],
        raw,
        deadline=spec["deadline_s"],
        expected=spec.get("expected_exit", 0),
        heartbeat=20,
    )


def publish(value: Json, output: Path, raw: Path, proof: Json) -> None:
    """All unchanged terminal validators inspect the exact locked candidate bytes."""

    def validate(candidate: Path) -> Json:
        cold = check(
            dict(
                name="terminal_cold", argv=cli() + ["--cold-replay", str(candidate)], deadline_s=120
            ),
            raw / "terminal",
        )
        found = findings.audit(candidate, raw / "terminal_findings", proof)
        rows = check(
            dict(
                name="strict_rows",
                argv=[
                    str(e.ROOT / ".venv/bin/python"),
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ],
                deadline_s=60,
            ),
            raw / "terminal",
        )
        return dict(
            passed=all(r["passed"] for r in [cold, found["receipt"], rows]),
            checks=[cold, found["receipt"], rows],
            adversarial=found,
        )

    receipt = publish_primary(output, value, validate)
    atomic_json(
        raw / "terminal_validation.json", dict(publication=receipt, normal_process_completion=True)
    )
    e.progress("published", 1, 0)


def main(argv: list[str] | None = None) -> int:
    """The standalone command runs workers, private controls and terminal replay."""
    e.progress("start")
    began = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261009"], default="20261009")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--worker-dir", type=Path)
    parser.add_argument("--crash", type=int, choices=[0, 32, 64], default=0)
    parser.add_argument("--mutate-from", type=int, default=0)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "reduction_drift")
        return int(not passed)
    if args.worker:
        if args.worker_dir is None:
            parser.error("--worker-dir is required for workers")
        e.k.run(
            json.loads(args.worker.read_bytes()),
            args.worker_dir,
            crash=args.crash,
            mutate_from=args.mutate_from,
        )
        return 0
    output = args.output.absolute()
    if args.private and output.is_relative_to(e.ROOT):
        parser.error("private control output must stay outside the repository")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, mode=0o700)
    with TemporaryDirectory(prefix="carnot-8348-") as directory:
        private = Path(directory)
        probe = private / "probe"
        probe.write_bytes(b"private_scratch")
        plan = manifest(private, private / "candidate.json")
        atomic_json(raw / "execution_manifest.json", plan)
        work = e.measure(args.root, raw)
        receipts = []
        if not args.private:
            for i, spec in enumerate(plan["commands"]):
                receipts.append(check(spec, raw / "validation"))
                e.progress("validation", i + 1, len(plan["commands"]) - i - 1)
        else:
            receipts.append(
                check(
                    dict(
                        name="private_child",
                        argv=[
                            str(e.ROOT / ".venv/bin/python"),
                            "-u",
                            "-c",
                            "print('private execution control')",
                        ],
                        deadline_s=10,
                    ),
                    raw / "validation",
                )
            )
        if (private / "coverage.json").exists():
            atomic_json(
                raw / "owned_coverage.json", json.loads((private / "coverage.json").read_bytes())
            )
            work["owned_coverage_reference"] = e.reference(raw / "owned_coverage.json")
        work.update(
            duration_s=time.monotonic() - began,
            execution_manifest=e.reference(raw / "execution_manifest.json"),
        )
        atomic_json(raw / "measurement.json", work)
        value = e.build(work, raw, receipts)
        for name, bad, expected in [
            ("valid", value, 0),
            ("negative", {}, 1),
            ("rehashed", dict(value, trajectory_ready_score=99), 1),
        ]:
            candidate = deepcopy(bad)
            candidate.pop("reproducibility_checksum", None)
            candidate["reproducibility_checksum"] = canonical_hash(candidate)
            path = raw / (name + ".json")
            atomic_json(path, candidate)
            receipts.append(
                check(
                    dict(
                        name=name,
                        argv=cli() + ["--cold-replay", str(path)],
                        deadline_s=120,
                        expected_exit=expected,
                    ),
                    raw / "controls",
                )
            )
        proof = e.k.numeric_proof(work)
        candidate = private / "candidate.json"
        value = e.build(work, raw, receipts)
        atomic_json(candidate, value)
        found = findings.audit(candidate, raw / "finding_controls", proof)
        work["adversarial_findings"] = found["findings"]
        work["finding_dispositions"] = found["dispositions"]
        receipts.append(found["receipt"])
        atomic_json(raw / "measurement.json", work)
        value = e.build(work, raw, receipts)
        publish(value, output, raw, proof)
    return 0

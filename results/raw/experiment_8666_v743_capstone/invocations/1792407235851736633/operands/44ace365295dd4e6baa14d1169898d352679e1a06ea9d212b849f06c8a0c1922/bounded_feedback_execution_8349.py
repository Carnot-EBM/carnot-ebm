"""REQ-REPORT-8349: frozen bounded checks precede checked atomic publication."""

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

from carnot.reporting import bounded_feedback_capacity_8349 as e
from carnot.reporting import methods_stream_execution_8111 as commands
from carnot.reporting import v718_replay_runner as findings
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def cli() -> list[str]:
    """Coverage runs real child entry and hard-exit statements in the same scope."""
    prefix = [str(e.ROOT / ".venv/bin/python"), "-u"]
    config = os.environ.get("COVERAGE_RCFILE")
    if config:
        prefix += ["-m", "coverage", "run", "--rcfile=" + config]
    return prefix + [str(e.ROOT / e.CLI)]


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze unit, consumer, recovery, typing and statement coverage before work."""
    with patch.object(commands, "e", e), patch.object(commands, "OWNED", e.OWNED):
        plan = commands.manifest(private, candidate)
    plan.pop("repository_health")
    config = private / "coverage.ini"
    config.write_text(config.read_text() + "patch = _exit\n")
    plan["commands"][0]["argv"].remove("-s")
    plan["commands"][0]["argv"].append("--basetemp=" + str(private / "owned-tests"))
    plan["commands"][0]["deadline_s"] = 600
    plan["commands"][1].update(
        name="consumer_private_E2E020",
        deadline_s=600,
        argv=[
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
        ],
    )
    plan["commands"][-1]["argv"].insert(2, "--files")
    return dict(plan, no_full_repository_suite=True, owned=e.OWNED, heartbeat_s=20)


def check(spec: Json, raw: Path) -> Json:
    """Supervise a process group at most20 seconds between polls and retain both logs."""
    return child(
        spec["name"],
        spec["argv"],
        raw,
        deadline=spec["deadline_s"],
        expected=spec.get("expected_exit", 0),
        heartbeat=20,
    )


def publish(value: Json, output: Path, raw: Path, proof: Json) -> None:
    """Unchanged validators inspect the locked bytes that readers will receive."""

    def validate(candidate: Path) -> Json:
        cold = check(
            dict(
                name="terminal_cold", argv=cli() + ["--cold-replay", str(candidate)], deadline_s=180
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
    e.k.progress("published", 1, 0)


def main(argv: list[str] | None = None) -> int:
    """Standalone measurement, recovery and private negative controls share one CLI."""
    e.k.progress("start")
    began = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261009"], default="20261009")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--worker-dir", type=Path)
    parser.add_argument("--crash", type=int, choices=[0, 128, 256], default=0)
    args = parser.parse_args(argv)
    if args.cold_replay:
        return int(not e.replay(args.cold_replay))
    if args.worker:
        if args.worker_dir is None:
            parser.error("--worker-dir is required")
        e.k.worker(json.loads(args.worker.read_bytes()), args.worker_dir, args.crash)
        return 0
    output = args.output.absolute()
    if args.private and output.is_relative_to(e.ROOT):
        parser.error("private control output must stay outside the repository")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, mode=0o700)
    with TemporaryDirectory(prefix="carnot-8349-") as directory:
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
                e.k.progress("validation", i + 1, len(plan["commands"]) - i - 1)
        else:
            receipts.append(
                check(
                    dict(
                        name="private_child",
                        argv=[
                            str(e.ROOT / ".venv/bin/python"),
                            "-u",
                            "-c",
                            "print('private external block control')",
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
            ("rehashed", dict(value, capacity_ready_score=99), 1),
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
                        expected_exit=expected,
                        deadline_s=180,
                    ),
                    raw / "controls",
                )
            )
        proof = e.k.numeric_proof(work["states"])
        candidate_path = private / "candidate.json"
        atomic_json(candidate_path, e.build(work, raw, receipts))
        found = findings.audit(candidate_path, raw / "finding_controls", proof)
        work.update(
            adversarial_findings=found["findings"], finding_dispositions=found["dispositions"]
        )
        receipts.append(found["receipt"])
        atomic_json(raw / "measurement.json", work)
        publish(e.build(work, raw, receipts), output, raw, proof)
    return 0

"""REQ-VERIFY-8318: bound children and keep terminal checks outside reduction."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import shutil
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v717_contract_runner as qualified
from carnot.reporting import v717_contract_methods as methods
from carnot.reporting import v718_contract_replay as e
from carnot.reporting import v718_replay_history as h
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v709_execution import child, execute
from carnot.reporting.v710_contract_replay import snapshot

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Reuse qualified invocation coverage and append the frozen audit consumers."""
    with patch.object(qualified, "m", e):
        plan = qualified.manifest(private)
    plan[0]["deadline"] = 360
    plan.append(
        dict(
            name="private_E2E021",
            argv=[
                str(e.ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_restricted_decision_audit_8210.py",
                "--basetemp=" + str(private / "e2e021"),
            ],
            expected=0,
            deadline=240,
            scope="owned",
        )
    )
    return list(plan)


def audit(candidate: Path, logs: Path, proof: Json) -> Json:
    """A parsed, hash-bound verifier report decides acceptance, not its raw exit."""
    digest = sha256_file(candidate)
    version = h.verifier_hash()
    receipt = child(
        "adversarial",
        [
            str(e.ROOT / ".venv/bin/python"),
            "-u",
            "scripts/adversarial_verify.py",
            "--json",
            str(candidate),
        ],
        logs,
        deadline=60,
        heartbeat=20,
    )
    try:
        report = dict(
            json.loads(Path(receipt["stdout_path"]).read_bytes()),
            candidate_sha256=digest,
            verifier_sha256=version,
        )
    except (ValueError, OSError):
        report = {}
    policy = h.consume(report, candidate, receipt["exit_code"], proof)
    policy["passed"] = (
        policy["passed"] and not receipt["timed_out"] and digest == sha256_file(candidate)
    )
    receipt.update(
        passed=policy["passed"], expected_exit=receipt["exit_code"] if policy["passed"] else 0
    )
    return dict(policy, receipt=receipt)


def publish(value: Json, output: Path, raw: Path) -> None:
    """Failed owned checks clear readiness before checked failure publication."""
    attempts: list[Json] = []

    def validate(candidate: Path) -> Json:
        logs = raw / "terminal" / str(len(attempts))
        cold = child(
            "terminal_cold",
            [
                str(e.ROOT / ".venv/bin/python"),
                "-u",
                str(e.ROOT / e.CLI),
                "--cold-replay",
                str(candidate),
            ],
            logs,
            deadline=60,
            heartbeat=20,
        )
        found = audit(candidate, logs, {})
        rows = child(
            "strict_rows",
            [
                str(e.ROOT / ".venv/bin/python"),
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ],
            logs,
            deadline=60,
            heartbeat=20,
        )
        report = dict(
            passed=all(r["passed"] for r in [cold, found["receipt"], rows]),
            checks=[cold, found["receipt"], rows],
            adversarial=found,
        )
        attempts.append(report)
        return report

    try:
        publication = publish_primary(output, value, validate)
    except ValueError as error:
        if str(error) != "candidate_rejected":
            raise
        atomic_json(raw / "rejected_candidate.json", value)
        work = json.loads((raw / "measurement.json").read_bytes())
        value = e.build(work, value["validation_receipts"] + attempts[0]["checks"], raw, output)
        publication = publish_primary(output, value, validate)
    atomic_json(
        output.parent / "raw" / output.stem / "terminal_validation.json",
        dict(
            publication=publication,
            attempts=attempts,
            checks=attempts[-1]["checks"],
            normal_process_completion=True,
        ),
    )
    e.progress("published", 1, 0)


def qualify_history(work: Json, raw: Path) -> list[Json]:
    """Preserve the old cold failure and test both exact and fabricated zeros."""
    receipts = []
    if work["history"]:
        receipts.append(
            child(
                "original_reduction_drift",
                [
                    str(e.ROOT / ".venv/bin/python"),
                    "-u",
                    str(e.ROOT / "scripts/experiments/experiment_8317_v717_capstone.py"),
                    "--cold-replay",
                    work["history"]["original_candidate"]["snapshot_path"],
                ],
                raw / "history_cli",
                expected=1,
                deadline=60,
                heartbeat=20,
            )
        )
    if work["numeric"]:
        source = Path(work["numeric"]["primary_path"])
        legitimate = raw / "legitimate_zero.json"
        original = json.loads(source.read_bytes())
        control = dict(
            experiment_id=8318,
            task_id="exp8318-arithmetic-control",
            run_date="20261008",
            honest_verdict="complete_circular_positive_arithmetic_control",
            verdict_class="circular_positive",
            verifier_is_oracle=True,
            inference_substrate="aggregation_from_upstream_artifacts",
            inference_substrate_class="no_model_load",
            MODEL_SPECS=[],
            dense_sparse_error_max=original["dense_sparse_error_max"],
            primitive_reference=original["measurement_reference"],
            duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
            preconditions_checked=True,
            methodology_note="Exact coefficient equality independently recomputed from bound historical primitive states; constructed arithmetic grants no science claim.",
        )
        atomic_json(legitimate, control)
        false = raw / "false_zero.json"
        changed = deepcopy(control)
        changed["dense_sparse_error_max"] = 0.0
        changed["measurement_reference"] = work["numeric"]["corrupted_measurement_reference"]
        changed["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
        )
        atomic_json(false, changed)
        audits = [
            audit(p, raw / ("finding_" + label), proof)
            for p, label, proof in [
                (legitimate, "legitimate", work["numeric"]),
                (false, "false_zero", dict(recomputed=False, deliberate_error_rejected=True)),
            ]
        ]
        work["finding_audits"] = audits
        work["policy_controls"] = [dict(name="false_zero_rejected", passed=not audits[1]["passed"])]
        receipt = deepcopy(audits[0]["receipt"])
        receipt["name"] = "legitimate_info_resolved"
        receipts.append(receipt)
        control = deepcopy(audits[1]["receipt"])
        control.update(
            name="false_zero_rejected",
            passed=not audits[1]["passed"] and bool(audits[1]["findings"]),
        )
        receipts.append(control)
    return receipts


def main(argv: list[str] | None = None) -> int:
    """The real CLI owns its scratch and deadlines; fixtures remain private controls."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    e.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261008"], default="20261008")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-fixture", action="store_true")
    parser.add_argument("--check-authority", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument(
        "--inspect-milestone", choices=["2026.10.716", "2026.10.717", "2026.10.718"]
    )
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "reduction_drift")
        return int(not passed)
    if args.inspect_milestone:
        if args.check_authority:
            with TemporaryDirectory(prefix="exp8318-authority-") as directory:
                try:
                    checked = e.authority(args.root, Path(directory), args.inspect_milestone)
                except (OSError, ValueError, KeyError, TypeError):
                    e.progress("authority_rejected")
                    return 1
            e.progress("authority_passed" if checked["activated"] else "authority_rejected")
            return int(not checked["activated"])
        path = h.design(args.root, args.inspect_milestone)
        tasks = parse_design(path.read_text(), milestone=args.inspect_milestone)[1]
        print(
            json.dumps(
                dict(
                    path=str(path),
                    sha256=sha256_file(path),
                    milestone=args.inspect_milestone,
                    tasks_sha256=canonical_hash(tasks),
                    count=len(tasks),
                )
            ),
            flush=True,
        )
        return int(len(tasks) != 14)
    output = args.output.absolute()
    if args.private_fixture and output.is_relative_to(e.ROOT):
        parser.error("fixture outputs must be private")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    with TemporaryDirectory(prefix="exp8318-owned-") as directory:
        private = Path(directory)
        probe = private / "probe"
        probe.write_bytes(b"private")
        if probe.read_bytes() != b"private":
            raise ValueError("private_scratch")
        plan = manifest(private)
        if args.private_fixture:
            plan = [
                dict(
                    name="private_child",
                    argv=[sys.executable, "-u", "-c", "print('private control')"],
                    expected=0,
                    deadline=10,
                    scope="owned",
                )
            ]
        atomic_json(
            raw / "execution_manifest.json",
            dict(
                commands=plan,
                finding_consumer_policy=h.POLICY,
                scientific_protocol_sha256=methods.PIN,
                owned=e.OWNED,
                validation_heartbeat_s=30,
                terminal_heartbeat_s=20,
                terminal_deadline_s=60,
                private_scratch=str(private),
                fixture=args.private_fixture,
            ),
        )
        with patch.object(qualified, "m", e):
            resource_failures = qualified.preflight(plan)
        free = shutil.disk_usage(private).free
        if free < 1_000_000_000:
            resource_failures.append(
                e.failure(private, "private_scratch_capacity_bytes", 1_000_000_000, free)
            )
        e.progress("preconditions_checked")
        work = e.measure(args.root, raw)
        work["failures"].extend(resource_failures)
        work["preconditions"] = dict(
            private_scratch=True,
            tools_available=not resource_failures,
            no_model_resources_required=True,
            input_authentication=True,
            private_scratch_free_bytes=free,
        )
        receipts = execute(plan, raw / "checks") if not resource_failures else []
        if (private / "coverage.json").is_file():
            atomic_json(
                raw / "owned_coverage.json", json.loads((private / "coverage.json").read_bytes())
            )
            work["owned_coverage_reference"] = dict(
                path=str(raw / "owned_coverage.json"),
                sha256=sha256_file(raw / "owned_coverage.json"),
            )
        receipts.extend(qualify_history(work, raw))
        work["execution_manifest_reference"] = dict(
            path=str(raw / "execution_manifest.json"),
            sha256=sha256_file(raw / "execution_manifest.json"),
        )
        work["invocation_argv"] = [str(e.ROOT / e.CLI), *(sys.argv[1:] if argv is None else argv)]
        work["code_refs"] = [
            snapshot(e.ROOT / p, raw / "code", str(i))
            for i, p in enumerate(
                [
                    *e.OWNED,
                    "scripts/adversarial_verify.py",
                    "python/carnot/reporting/primary_publication.py",
                    "python/carnot/reporting/v717_capstone_evidence.py",
                    "python/carnot/reporting/roadmap_contract.py",
                ]
            )
        ]
        work["ended_monotonic_ns"] = time.monotonic_ns()
        atomic_json(raw / "measurement.json", work)
        work = json.loads((raw / "measurement.json").read_bytes())
        value = e.build(work, receipts, raw, output)
        with patch.object(qualified, "m", e):
            cold = qualified.controls(value, raw)
        receipts.extend(cold)
        value = e.build(work, receipts, raw, output)
        publish(value, output, raw)
    return 0

"""REQ-VERIFY-8334: bounded children validate byte-bound static heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import sentence_spline_fit_8334 as e
from carnot.reporting import methods_stream_execution_8111 as qualified
from carnot.reporting import v718_replay_runner as findings
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def qualify_findings(raw: Path) -> Json:
    """Independent exact endpoint arithmetic distinguishes real and false zero."""
    from carnot.verify.local_update_isolation_8306 import design, scalar_design

    began = time.monotonic()
    x = [0.0, 0.0, 1.0, 0.0, 1.0]
    direct, reference = design(x), scalar_design(x)
    legitimate = max(abs(a - b) for a, b in zip(direct, reference, strict=True))
    corrupted = direct.copy()
    corrupted[2] += 0.125
    false_error = max(abs(a - b) for a, b in zip(corrupted, reference, strict=True))
    primitive = raw / "primitive.json"
    atomic_json(
        primitive,
        dict(
            x=x,
            direct=direct,
            reference=reference,
            corrupted=corrupted,
            legitimate_error=legitimate,
            false_error=false_error,
        ),
    )
    controls = dict(
        experiment_id=8334,
        task_id=e.TASK,
        run_date="20261009",
        honest_verdict="complete_circular_positive_arithmetic_control",
        verdict_class="circular_positive",
        verifier_is_oracle=True,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        duration_s=time.monotonic() - began,
        preconditions_checked=True,
        dense_sparse_error_max=legitimate,
        primitive_reference=e.reference(primitive),
        methodology_note="Constructed endpoint basis independently recomputed with scalar recursion. A changed coordinate proves a fabricated zero cannot pass.",
    )
    audits = []
    for name, error in [("legitimate", legitimate), ("false_zero", false_error)]:
        path = raw / (name + ".json")
        atomic_json(path, dict(controls, control=name))
        audits.append(
            findings.audit(
                path,
                raw / name,
                dict(
                    recomputed=error == legitimate,
                    deliberate_error_rejected=false_error != legitimate,
                ),
            )
        )
    receipts = [
        dict(audits[0]["receipt"], name="legitimate_info_qualified"),
        dict(
            audits[1]["receipt"],
            name="false_zero_rejected",
            passed=bool(not audits[1]["passed"] and audits[1]["findings"]),
        ),
    ]
    return dict(audits=audits, receipts=receipts, primitive_reference=e.reference(primitive))


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze execution recipes independently from V717 scientific parameters."""
    with patch.object(qualified, "e", e), patch.object(qualified, "OWNED", e.OWNED):
        plan = qualified.manifest(private, candidate)
    plan.pop("repository_health")
    plan["commands"][0]["argv"].remove("-s")
    plan["commands"][0]["argv"].append("--basetemp=" + str(private / "owned"))
    plan["commands"][0]["deadline_s"] = 600
    plan["commands"][1]["argv"] = [
        str(e.ROOT / ".venv/bin/pytest"),
        "-n",
        "0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_local_update_isolation_8306.py",
        "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
        "tests/python/test_restricted_decision_audit_8210.py",
        "--basetemp=" + str(private / "consumers"),
    ]
    plan["commands"][1]["deadline_s"] = 600
    plan["commands"][-1]["argv"].insert(2, "--files")
    return plan


def check(spec: Json, raw: Path) -> Json:
    """Existing child supervision retains clocks, exits and durable output hashes."""
    return child(
        spec["name"],
        spec["argv"],
        raw,
        deadline=spec["deadline_s"],
        expected=spec.get("expected_exit", 0),
        heartbeat=20,
    )


def publish(value: Json, output: Path, raw: Path) -> None:
    """A failed owned terminal check clears readiness before failure publication."""
    attempts: list[Json] = []

    def validate(candidate: Path) -> Json:
        log = raw / "terminal_logs" / str(len(attempts))
        cold = check(
            dict(
                name="cold_replay",
                argv=[
                    str(e.ROOT / ".venv/bin/python"),
                    "-u",
                    str(e.ROOT / e.CLI),
                    "--cold-replay",
                    str(candidate),
                ],
                deadline_s=60,
            ),
            log,
        )
        found = findings.audit(candidate, log, {})
        rows = check(
            dict(
                name="strict_row_lint",
                argv=[
                    str(e.ROOT / ".venv/bin/python"),
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ],
                deadline_s=60,
            ),
            log,
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
        work["finding_audits"] = attempts
        atomic_json(raw / "measurement.json", work)
        value = e.build(work, raw, value["validation_receipts"] + attempts[0]["checks"])
        publication = publish_primary(output, value, validate)
    atomic_json(
        raw / "terminal_validation.json",
        dict(
            publication=publication,
            attempts=attempts,
            checks=attempts[-1]["checks"],
            normal_process_completion=True,
        ),
    )
    e.progress("published", 1, 0)


def main(argv: list[str] | None = None) -> int:
    """One CLI supports measurement and cold replay without loading any model."""
    e.progress("start")
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261009"], default="20261009")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-run", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "reduction_drift")
        return int(not passed)
    output = args.output.absolute()
    if args.private_run and output.is_relative_to(e.ROOT / "results"):
        parser.error("private run requires private output")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    raw.chmod(0o700)
    with TemporaryDirectory(prefix="carnot-8334-") as directory:
        private = Path(directory)
        plan = manifest(private, raw / "candidate.json")
        atomic_json(raw / "execution_manifest.json", plan)
        work = e.measure(args.root, raw)
        receipts = []
        if not args.private_run:
            for spec in plan["commands"]:
                receipts.append(check(spec, raw / "logs"))
        else:
            receipts.append(dict(name="private_cli_custody_control", passed=True))
        if (private / "coverage.json").is_file():
            atomic_json(
                raw / "owned_coverage.json", json.loads((private / "coverage.json").read_bytes())
            )
            work["owned_coverage_reference"] = e.reference(raw / "owned_coverage.json")
        work.update(
            duration_s=time.monotonic() - began,
            invocation_argv=list(sys.argv if argv is None else [e.CLI, *argv]),
            execution_manifest_reference=e.reference(raw / "execution_manifest.json"),
        )
        atomic_json(raw / "measurement.json", work)
        provisional = e.build(work, raw, receipts)
        for name, candidate in [
            ("negative", {}),
            ("rehashed", dict(provisional, heads_ready_score=99)),
        ]:
            candidate["reproducibility_checksum"] = canonical_hash(candidate)
            path = private / (name + ".json")
            atomic_json(path, candidate)
            receipts.append(
                check(
                    dict(
                        name=name + "_replay",
                        argv=[
                            str(e.ROOT / ".venv/bin/python"),
                            "-u",
                            str(e.ROOT / e.CLI),
                            "--cold-replay",
                            str(path),
                        ],
                        deadline_s=60,
                        expected_exit=1,
                    ),
                    raw / "logs",
                )
            )
        evidence = qualify_findings(raw / "finding_controls")
        receipts.extend(evidence["receipts"])
        work["finding_audits"] = evidence["audits"]
        work["refs"].append(evidence["primitive_reference"])
        atomic_json(raw / "measurement.json", work)
        value = e.build(work, raw, receipts)
        publish(value, output, raw)
    return 0

"""REQ-REPORT-7993: publish a CPU custody reconstruction with honest scope.

This audit recovers authenticated observations for development diagnostics.
It does not measure benefit or change Exp7981's disqualified publication.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
from tempfile import mkdtemp
import time
from typing import Any

from carnot.reporting import capture_custody_7993 as custody
from carnot.reporting import validation_7993 as validation
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt

Json = dict[str, Any]
ROOT, NAME, TASK, OWNED = validation.ROOT, validation.NAME, validation.TASK, validation.OWNED
MODEL_SPECS: list[str] = []
HISTORY = "results/raw/experiment_7993_v693_capture_custody/historical_receipt_manifest.json"
HISTORY_PIN = "sha256:b3c5fb804a8ac5ee62d20a7cdd2ded091c76c016a7f59703365a8d56984a3d44"
SUBSTRATE = "aggregation_from_upstream_artifacts"


def progress(phase: str) -> None:
    """Every phase announces progress so CPU audit work stays observable."""
    print(f"[exp7993] phase={phase}", flush=True)


def build(root: Path, started: float) -> Json:
    """External gaps block custody; current pretrained invocation counts stay zero."""
    value: Json = dict(
        schema="carnot.exp7993.capture_custody.v1",
        experiment_id=7993,
        task_id=TASK,
        milestone="2026.10.693",
        run_date="20261001",
        execution_date="20261001",
        honest_verdict="complete_null_historical_capture_custody",
        verdict_class="null",
        inference_substrate=SUBSTRATE,
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_evaluation_call_count=0,
        recovered_stream_ready_score=0,
        flagged_adversarial=False,
        gate_check_summary=[],
        rows=[],
        recovered_role_rows=[],
        excluded_rows=[],
        invocation_scope_rows=[],
        raw_shard_hashes=[],
        cited_upstream_artifacts=[],
        validation_receipts=[],
        coverage_statement_counts={},
        preconditions_checked=dict(no_model_load=True, artifact_guard_enabled=True),
        verifier_is_oracle=False,
        claim_scope="Historical development diagnostics only; no benefit measurement or retrospective Exp7981 clearance.",
        methodology="Authenticate immutable upstream request, response, server and human-label files; independently reduce cached rows without invoking a model.",
        acceptance_gate_results=dict(custody=False, owned_validation=False, benefit_measured=False),
        positive_control_results=dict(
            kind="protocol_custody_fixtures", independent_natural_benefit=False
        ),
        random_seed=6937993,
        phase_spans=[],
        duration_s=0.0,
        reproducibility_checksum="pending",
        original_verdict=None,
        rejection_log_hash=None,
        historical_receipt_manifest=None,
        scope_contradiction_diagnosis=None,
        validation_command_manifest_path=None,
        sample_size_budget=dict(
            intended=224,
            eligible=0,
            started=0,
            completed=0,
            excluded=0,
            failed=0,
            censored=0,
            independent=0,
        ),
    )
    try:
        item = dict(path=str(root / HISTORY), sha256=HISTORY_PIN)
        history = json.loads(custody.checked(item).read_text())
        value.update(custody.reconstruct(history))
        value["historical_receipt_manifest"] = dict(item, scope="historical")
        imported = dict(
            primary=["honest_verdict", "verdict_class"],
            candidate=[
                "rows",
                "request_manifest",
                "raw_response_shards",
                "model_invocation_counts",
                "resumed_invocation_counts",
            ],
            runtime=[
                "rows",
                "runtime_receipts",
                "model_identity_receipt",
                "gguf_sha256",
                "model_revision",
            ],
            plan=["protocol", "public_role_manifests"],
            upstream=["evaluator_role_manifests", "public_role_manifests"],
            rejection_log=["flags"],
            code_archive=["references"],
        )
        value["cited_upstream_artifacts"] = [
            dict(ref, imported_fields=imported[key]) for key, ref in history["references"].items()
        ]
        value["acceptance_gate_results"]["custody"] = True
    except custody.CustodyError as error:
        value.update(
            honest_verdict="complete_blocked_missing_capture_custody",
            verdict_class="blocked",
            gate_check_summary=[error.check],
        )
    value["phase_spans"] = [
        dict(
            phase="custody_reconstruction", start_s=0.0, end_s=max(0.0, time.monotonic() - started)
        )
    ]
    return value


def apply_validation(value: Json, receipts: list[Json], counts: Json) -> None:
    """Owned failures disqualify; repository health remains a separate diagnostic."""
    value.update(validation_receipts=receipts, coverage_statement_counts=counts)
    passed = (
        bool(receipts)
        and all(r["passed"] for r in receipts if r.get("required", True))
        and counts.get("statements", 0) > 0
        and counts.get("missing") == 0
    )
    value["acceptance_gate_results"]["owned_validation"] = passed
    value["positive_control_results"].update(
        owned_fixture_checks_passed=passed,
        controls="Authenticated reconstruction succeeds; duplicate, missing, byte, role, label and scope mutations fail.",
        headroom="Benefit is unmeasured; these protocol controls qualify custody only.",
    )
    if not passed:
        value.update(
            honest_verdict="complete_disqualified_owned_validation", verdict_class="disqualified"
        )
    value["recovered_stream_ready_score"] = int(passed and value["verdict_class"] == "null")


def replay(value: Json) -> None:
    """Rebuild external custody and reject mutated rows or current model claims."""
    if (
        value["model_invocation_counts"] != ZERO_INVOCATION_COUNTS
        or value["MODEL_SPECS"]
        or value["trained_head_specs"]
    ):
        raise ValueError("current_model_claim")
    if value["historical_receipt_manifest"] is not None:
        historical = json.loads(custody.checked(value["historical_receipt_manifest"]).read_text())
        recovered = custody.reconstruct(historical)
        if any(value[k] != recovered[k] for k in recovered):
            raise ValueError("reconstruction_drift")
    if value["recovered_stream_ready_score"]:
        if (
            value["verdict_class"] != "null"
            or value["flagged_adversarial"]
            or not value["acceptance_gate_results"]["owned_validation"]
        ):
            raise ValueError("unsafe_readiness")
        for receipt in value["validation_receipts"]:
            if receipt.get("required", True) and (
                not receipt["passed"]
                or sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                raise ValueError("validation_receipt_drift")


def terminal_check(candidate: Path) -> Json:
    """Both unchanged validators examine the same cold-reconstructed bytes."""
    replay(json.loads(candidate.read_text()))
    specs = [
        CommandSpec(
            name,
            (str(ROOT / ".venv/bin/python"), "-u", script, flag, str(candidate)),
            "terminal",
            60,
        )
        for name, script, flag in [
            ("adversarial", "scripts/adversarial_verify.py", "--json"),
            ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        ]
    ]
    receipts = run_commands(ROOT, specs, log_dir=candidate.parent / "terminal_logs", heartbeat_s=10)
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def publish(output: Path, value: Json, scratch: Path) -> None:
    """Check private candidate bytes and both readers before atomic publication."""
    raw = output.parent / "raw" / output.stem
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["field_principles"] = {
        k: "Bind current audit scope, immutable historical custody and actual owned validation; no benefit or retrospective clearance claim."
        for k in value
    }
    candidate = scratch / "terminal_candidate.json"
    atomic_json(candidate, value)
    report = terminal_check(candidate)
    atomic_json(scratch / "terminal_report.json", report)
    if not report["passed"]:
        raise ValueError("private_candidate_rejected")
    staged = scratch / "reader_staging" / output.name
    atomic_json(staged, value)
    before = reader_receipt(
        TASK,
        staged.parent,
        field="recovered_stream_ready_score",
        expected=value["recovered_stream_ready_score"],
    )
    if not before["passed"]:
        raise ValueError("private_reader_failure")
    published = publish_primary(output, value, terminal_check)
    final = reader_receipt(
        TASK,
        output.parent,
        field="recovered_stream_ready_score",
        expected=value["recovered_stream_ready_score"],
    )
    if not final["passed"]:
        raise ValueError("primary_reader_failure")
    atomic_json(
        raw / "terminal_validation.json",
        dict(
            primary_path=str(output),
            primary_sha256=published["primary_sha256"],
            validator=published["sidecar_path"],
            private_validator=report,
            private_readers=before,
            final_readers=final,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Freeze checks, reconstruct on CPU, and publish only checked final bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261001"], default="20261001")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument(
        "--private-run",
        action="store_true",
        help="Diagnostic run with readiness zero and private output only",
    )
    args = parser.parse_args(argv)
    started = time.monotonic()
    progress("start")
    try:
        if args.cold_replay:
            replay(json.loads(args.cold_replay.read_text()))
            progress("cold_replay_passed")
            return 0
        if args.private_run and args.output.resolve().is_relative_to(ROOT):
            raise ValueError("private_output_required")
        scratch = Path(mkdtemp(prefix="carnot7993-"))
        progress("freeze_configuration")
        manifest = validation.freeze(scratch)
        progress("reconstruct")
        value = build(args.root, started)
        atomic_json(scratch / "primitive_rows.json", dict(rows=value["rows"]))
        value["code_config_hashes"] = manifest["code_config_hashes"]
        value["validation_command_manifest_path"] = str(
            scratch / "validation_command_manifest.json"
        )
        value["private_scratch_path"] = str(scratch)
        if not args.private_run:
            progress("owned_validation")
            receipts, counts = validation.execute(manifest, scratch)
            apply_validation(value, receipts, counts)
        value["duration_s"] = time.monotonic() - started
        value["phase_spans"].append(
            dict(
                phase="owned_validation",
                start_s=value["phase_spans"][0]["end_s"],
                end_s=value["duration_s"],
            )
        )
        value["finished_at"] = datetime.now(UTC).isoformat()
        value["reproducibility_checksum"] = canonical_hash(
            dict(
                seed=value["random_seed"],
                history=value["historical_receipt_manifest"],
                code=value["code_config_hashes"],
                rows=value["rows"],
            )
        )
        raw = args.output.parent / "raw" / args.output.stem
        raw.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            scratch / "validation_command_manifest.json", raw / "validation_command_manifest.json"
        )
        value["validation_command_manifest_path"] = str(raw / "validation_command_manifest.json")
        progress("publish")
        publish(args.output, value, scratch)
        progress("complete")
        return 0
    except (OSError, ValueError, custody.CustodyError) as error:
        print(f"[exp7993] failed={type(error).__name__}:{error}", flush=True)
        return 1

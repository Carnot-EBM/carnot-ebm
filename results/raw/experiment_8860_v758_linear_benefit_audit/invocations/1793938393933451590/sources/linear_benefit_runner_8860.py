"""REQ-REPORT-8860: publish a read-only audit with bounded checks and raw custody."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import linear_benefit_audit_8860 as audit
from carnot.reporting import learning_signal_runner_8815 as validation
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
BUDGET = 4200


def validate(raw: Path) -> Json:
    """Reuse bounded children and real unit/child coverage; global health stays separate."""
    with patch.multiple(validation, OWNED=audit.OWNED, TESTS=audit.TESTS, CLI=audit.CLI):
        return validation.validate(raw)


def terminal(candidate: Path, raw: Path) -> Json:
    """Run unchanged scanners and a fresh outside-checkout replay without PYTHONPATH."""
    with patch.object(validation, "CLI", audit.CLI):
        return validation.terminal(candidate, raw)


def main(argv: list[str] | None = None) -> int:
    """Private output changes custody only; the date and research gates stay fixed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261106")
    parser.add_argument(
        "--output",
        "--private-output",
        type=Path,
        default=audit.ROOT / "results" / (audit.NAME + ".json"),
    )
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--source", type=Path, default=audit.SOURCE)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        try:
            print(json.dumps(audit.replay(args.replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            print(f"replay_rejected:{error}", flush=True)
            return 1
    if args.date != "20261106":
        print("date_rejected: expected 20261106", flush=True)
        return 1
    started = time.monotonic_ns()
    output = args.output.absolute()
    raw = (
        args.raw_dir or output.parent / "raw" / audit.NAME / "invocations" / str(time.time_ns())
    ).absolute()
    raw.mkdir(parents=True, exist_ok=False)
    raw.chmod(0o700)
    work: Json = {}
    pre: Json = {}
    refs: list[Json] = []
    failures = []
    reduced = audit.empty("external_input_unavailable")
    verdict, admitted = "null", False
    audit.progress("preconditions_before", 0, 1)
    try:
        work, pre, refs = audit.preflight(raw, args.source)
        admitted = True
        audit.progress("preconditions_after", 1)
        audit.progress("reconstruction_before", 0, 24)
        reduced = audit.reduce(work)
        audit.progress("reconstruction_after", 24)
        if reduced["causal_benefit_score"]:
            verdict = "circular_positive"
    except (OSError, ValueError, KeyError, RuntimeError, StopIteration) as error:
        external = not admitted and str(error) != "internal_resources"
        verdict = (
            "partial"
            if isinstance(error, TimeoutError)
            else "blocked"
            if external
            else "disqualified"
        )
        path = Path(getattr(error, "filename", None) or args.source)
        reason = (
            "missing_file"
            if isinstance(error, FileNotFoundError)
            else "missing_field"
            if isinstance(error, KeyError)
            else "observed_zero"
            if "observed_zero" in str(error)
            else "rejected_producer"
            if external
            else "owned_failure"
        )
        failures.append(
            dict(
                check="sealed_stream_authority" if external else "owned_integrity",
                upstream="exp8856-linear-causal-streams",
                artifact_field="stream_rows_ready_score",
                expected=1,
                observed=0 if reason == "observed_zero" else str(error),
                path=str(path),
                hash=sha256_file(path) if path.is_file() else None,
                status=reason,
            )
        )
        reduced = audit.empty(reason)
    atomic_json(raw / "evidence.json", work)
    atomic_json(raw / "preconditions.json", pre)
    sources = raw / "sources"
    sources.mkdir()
    for name in audit.OWNED + audit.TESTS:
        shutil.copyfile(audit.ROOT / name, sources / Path(name).name)
    audit.progress("validation_before", 0, 1)
    checked = validate(raw)
    audit.progress("validation_after", 1)
    if not checked["passed"]:
        verdict = "disqualified"
    if verdict in {"blocked", "disqualified", "partial"}:
        reduced.update({k: 0 for k in reduced if k.endswith("_score")})
    ended = time.monotonic_ns()
    spans = [
        dict(
            phase="preconditions_reconstruction_validation",
            started_monotonic_ns=started,
            ended_monotonic_ns=ended,
            duration_s=(ended - started) / 1e9,
        )
    ]
    substrate = "aggregation_from_upstream_artifacts" if admitted else "blocked_no_run"
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=substrate,
        inference_substrate_class="no_model_load",
        execution_venue="host",
        inference_substrate_details=dict(backend="cpu_read_only", generator_weights_updated=False),
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=spans,
    )
    value = dict(
        reduced,
        experiment_id=8860,
        task_id="exp8860-linear-benefit-audit",
        milestone="2026.11.758",
        run_date=args.date,
        honest_verdict="complete_" + verdict + "_linear_benefit_audit",
        verdict_class=verdict,
        inference_substrate=substrate,
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="constructed signed records and exact labels; no natural-language or generator-training claim",
        independent_generalization_score=0,
        duration_s=(ended - started) / 1e9,
        phase_spans=spans,
        preconditions_checked=pre,
        random_seed=8860,
        reproducibility_checksum=canonical_hash([audit.PIN, audit.CONTRASTS, 8860]),
        source_artifact_hashes=refs,
        code_config_hashes=[
            audit.reference(audit.ROOT / p)
            for p in audit.OWNED
            + audit.TESTS
            + [
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/learning_signal_runner_8815.py",
                "scripts/experiment_template.py",
            ]
        ],
        raw_shard_hashes=[audit.reference(raw / "evidence.json")]
        + [audit.reference(p) for p in sorted(sources.iterdir())],
        gate_check_summary=failures,
        acceptance_gates=audit.a.GATES,
        required_checks_passed=checked["passed"] and verdict in {"null", "circular_positive"},
        validation_receipts=checked["receipts"],
        owned_statement_coverage=checked["coverage"],
        current_work_receipt=receipt,
        repository_health_receipts=[
            r for r in checked["receipts"] if r["scope"] == "repository_health"
        ],
        methodology_note="Queries25-48;10000 paired whole-stream draws; simultaneous99% six-contrast future family and four-endpoint combined retention family. Immediate feedback diagnoses delay only. No fit or selection.",
    )
    value["field_principles"] = {
        k: "Bind actual audit work to immutable rows; completion does not imply scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        causal_benefit_score="Integrity, future gain, addition-specific gain, true-label specificity and both retention endpoints must all pass.",
        addition_benefit_score="Combined improves calibration-only by the registered margin; retention is reported separately.",
        retention_pass_score="Four simultaneous upper bounds compare combined to its own untouched checkpoint0.",
        learning_audit_ready_score="A valid null remains ready for the capstone; benefit never selects readiness.",
        movement_intervals="Parameter path lengths are reconstructed from sealed coefficient differences; no optimizer is called.",
        template_reuse_rows="Each later prediction joins an earlier admission and its actual binding witnesses.",
        delay_subgroups="Fixed and jitter delay descriptives retain paired streams and immediate feedback is diagnostic only.",
    )
    candidate = raw / "candidate.json"
    atomic_json(candidate, value)
    audit.progress("terminal_before", 0, 1)
    final = terminal(candidate, raw)
    audit.progress("terminal_after", 1)
    value["validation_receipts"].extend(final["receipts"])
    finished = time.monotonic_ns()
    spans.append(
        dict(
            phase="terminal_scanners_cold_replay",
            started_monotonic_ns=ended,
            ended_monotonic_ns=finished,
            duration_s=(finished - ended) / 1e9,
        )
    )
    value["duration_s"] = (finished - started) / 1e9
    receipt.update(ended_monotonic_ns=finished, duration_s=value["duration_s"], phase_spans=spans)
    if not final["passed"] or value["duration_s"] > BUDGET:
        value.update(
            honest_verdict="complete_disqualified_terminal_validation",
            verdict_class="disqualified",
            required_checks_passed=False,
        )
        value.update({k: 0 for k in reduced if k.endswith("_score")})
    atomic_json(raw / "terminal_validation.json", final)
    value["terminal_validation"] = audit.reference(raw / "terminal_validation.json")
    value["field_principles"]["terminal_validation"] = (
        "Bind unchanged scanners and cold replay to actual output logs."
    )
    value = normalize_artifact_for_template_write(value)
    if value["verdict_class"] == "partial":
        atomic_json(output, value)
    else:
        publish_primary(
            output, value, lambda path: dict(passed=json.loads(path.read_bytes()) == value)
        )
    audit.progress("published", value["completed_count"], value["censored_count"])
    return 0

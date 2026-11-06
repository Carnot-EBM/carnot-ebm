"""REQ-REPORT-8846: publish a bounded read-only audit with actual owned validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import future_benefit_audit_8846 as audit
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
    """Reuse streamed children, actual statement coverage and separate repository health."""
    with patch.multiple(validation, OWNED=audit.OWNED, TESTS=audit.TESTS, CLI=audit.CLI):
        return validation.validate(raw)


def terminal(candidate: Path, raw: Path) -> Json:
    """Use unchanged scanners and replay in a fresh outside-checkout process."""
    with patch.object(validation, "CLI", audit.CLI):
        return validation.terminal(candidate, raw)


def main(argv: list[str] | None = None) -> int:
    """Keep custody options separate from frozen scientific inputs and acceptance gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261105")
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
    if args.date != "20261105":
        print("date_rejected: expected 20261105", flush=True)
        return 1
    started = time.monotonic_ns()
    output = args.output.absolute()
    raw = (
        args.raw_dir or output.parent / "raw" / audit.NAME / "invocations" / str(time.time_ns())
    ).absolute()
    work: Json = {}
    pre: Json = {}
    refs: list[Json] = []
    failures = []
    reduced: Json = dict(
        rows=audit.producer.census(audit.producer.manifest(), "external_input_unavailable"),
        independent_count=0,
        intended_count=12096,
        completed_count=0,
        failed_count=0,
        censored_count=12096,
        excluded_count=0,
        learning_audit_ready_score=0,
        causal_benefit_score=0,
        addition_benefit_score=0,
        retention_pass_score=0,
        categorical_noninferiority_score=0,
        per_stream_results=[],
        paired_interval_rows=[],
        retention_interval_rows=[],
        causal_integrity_rows=[],
        untouched_retention_rows=[],
        template_reuse_rows=[],
        influence_rows=[],
        subgroup_descriptives=[],
        next_head_recommendation=dict(
            read_only=True, next_head="undetermined", reason="external input unavailable"
        ),
    )
    verdict = "null"
    admitted = False
    audit.progress("preconditions_before", 0, 1)
    try:
        work, pre, refs = audit.preflight(raw, args.source)
        admitted = True
        audit.progress("preconditions_after", 1)
        audit.progress("independent_reduction_before", 0, 24)
        reduced = audit.reduce(work)
        audit.progress("independent_reduction_after", 24)
        if reduced["causal_benefit_score"] and reduced["retention_pass_score"]:
            verdict = "circular_positive"
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        external = not admitted and isinstance(
            error, (FileNotFoundError, PermissionError, KeyError)
        )
        verdict = "blocked" if external else "disqualified"
        path = Path(getattr(error, "filename", None) or args.source)
        failures.append(
            dict(
                check="sealed_stream_authority" if external else "owned_integrity",
                upstream="exp8843-causal-streams",
                artifact_field="stream_rows_ready_score",
                expected=1,
                observed=str(error),
                path=str(path),
                hash=sha256_file(path) if path.is_file() else None,
                status="missing_file"
                if isinstance(error, FileNotFoundError)
                else "missing_field"
                if isinstance(error, KeyError)
                else "observed_zero"
                if "observed_zero" in str(error)
                else "rejected_producer"
                if external
                else "internal_failure",
            )
        )
    atomic_json(raw / "evidence.json", work)
    atomic_json(raw / "preconditions.json", pre)
    audit.progress("validation_before", 0, 1)
    checked = validate(raw)
    audit.progress("validation_after", 1)
    if not checked["passed"]:
        verdict = "disqualified"
    if verdict in {"blocked", "disqualified"}:
        reduced.update({k: 0 for k in reduced if k.endswith("_score")})
    ended = time.monotonic_ns()
    spans = [
        dict(
            phase="preconditions_reduction_validation",
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
        experiment_id=8846,
        task_id="exp8846-future-benefit-audit",
        milestone="2026.11.757",
        run_date=args.date,
        honest_verdict="complete_" + verdict + "_future_benefit_audit",
        verdict_class=verdict,
        inference_substrate=substrate,
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="constructed signed worlds and exact symbolic labels; no natural-language or hidden-game generalization",
        independent_generalization_score=0,
        duration_s=(ended - started) / 1e9,
        random_seed=8846,
        reproducibility_checksum=canonical_hash([audit.PIN, audit.CONTRASTS, 8846]),
        phase_spans=spans,
        preconditions_checked=pre,
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
        raw_shard_hashes=[audit.reference(raw / "evidence.json")],
        gate_check_summary=failures,
        acceptance_gates=dict(
            frozen_brier_improvement=0.01,
            addition_brier_improvement=0.005,
            shuffled_improvement_lower_bound=0,
            retention_upper_increase=0.01,
            decisive_error_upper_increase=0.01,
            categorical_noninferiority_margin=0.005,
            bootstrap_draws=10000,
            brier_familywise_confidence=0.99,
        ),
        required_checks_passed=checked["passed"] and verdict not in {"blocked", "disqualified"},
        validation_receipts=checked["receipts"],
        owned_statement_coverage=checked["coverage"],
        current_work_receipt=receipt,
        repository_health_receipts=[
            r for r in checked["receipts"] if r["scope"] == "repository_health"
        ],
        methodology_note="Queries25-48; paired whole-stream bootstrap10000 draws; Bonferroni99% familywise five Brier endpoints. Separate simultaneous retention family. Decisive errors use scheduled-row incidence. Log probabilities floored at1e-15. No training, threshold selection or serving change.",
    )
    value["field_principles"] = {
        k: "Bind measured audit work to immutable operands; completion does not imply benefit."
        for k in value
    }
    value["field_principles"].update(
        paired_interval_rows="Five preregistered Brier contrasts share whole-stream draws and Bonferroni99% familywise intervals.",
        retention_interval_rows="Separate simultaneous family bounds Brier and scheduled decisive-error incidence changes at24/48.",
        influence_rows="Leave-one-stream-out signs describe sensitivity and cannot choose a post-hoc winner.",
        template_reuse_rows="Join each later prediction to an independently reconstructed earlier admission.",
        next_head_recommendation="Read-only complexity recommendation applies the frozen0.005 margin without changing a live policy.",
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
        "Bind unchanged scanners and outside-checkout replay to actual output logs."
    )
    value = normalize_artifact_for_template_write(value)
    publish_primary(output, value, lambda path: dict(passed=json.loads(path.read_bytes()) == value))
    audit.progress("published", value["completed_count"], value["censored_count"])
    return 0

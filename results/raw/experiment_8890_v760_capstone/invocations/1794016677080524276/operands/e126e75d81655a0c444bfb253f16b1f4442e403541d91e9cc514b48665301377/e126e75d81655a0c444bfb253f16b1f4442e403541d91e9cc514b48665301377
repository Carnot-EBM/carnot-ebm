"""REQ-REPORT-8867: keep diagnosis private until bounded terminal validation passes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import feedback_information_8867 as d
from carnot.reporting import linear_benefit_runner_8860 as existing
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
    """Run owned checks while retaining an authenticated prior health attempt."""
    prior = d.ROOT / "results" / (d.NAME + ".json")
    retained = []
    if prior.is_file():
        retained = [
            receipt
            for receipt in json.loads(prior.read_bytes()).get("repository_health_receipts", [])
            if receipt["name"] == "global_python_health" and receipt["scope"] == "repository_health"
        ]
    for receipt in retained:
        d.require(
            sha256_file(d.ROOT / receipt["log_path"]) == receipt["log_sha256"],
            "repository_health_log",
        )
    commands = existing.validation.commands

    def reuse(plan: list[tuple[str, list[str], str, float]], logs: Path) -> list[Json]:
        return (
            commands(
                [entry for entry in plan if not (retained and entry[0] == "global_python_health")],
                logs,
            )
            + retained
        )

    with (
        patch.object(existing, "audit", d),
        patch.object(existing.validation, "commands", reuse),
    ):
        return existing.validate(raw)


def terminal(candidate: Path, raw: Path) -> Json:
    """Use unchanged scanners and actual outside-checkout replay before publication."""
    with patch.object(existing, "audit", d):
        return existing.terminal(candidate, raw)


def main(argv: list[str] | None = None) -> int:
    """Output overrides change custody; original authorities and controls stay fixed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261106")
    parser.add_argument(
        "--output", "--private-output", type=Path, default=d.ROOT / "results" / (d.NAME + ".json")
    )
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--source", type=Path, default=d.SOURCE)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        try:
            print(json.dumps(d.replay(args.replay)), flush=True)
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
        args.raw_dir or output.parent / "raw" / d.NAME / "invocations" / str(time.time_ns())
    ).absolute()
    raw.mkdir(parents=True, exist_ok=False)
    raw.chmod(0o700)
    evidence: Json = {}
    pre: Json = {}
    refs: list[Json] = []
    failures: list[Json] = []
    spans: list[Json] = []
    admitted = False
    verdict = "null"
    reduced = d.empty("external_input_unavailable")
    began = started
    d.progress("preconditions_before", 0, 1)
    try:
        work, pre, refs = d.preflight(raw, args.source)
        admitted = True
        ended = time.monotonic_ns()
        spans.append(
            dict(
                phase="preconditions",
                started_monotonic_ns=began,
                ended_monotonic_ns=ended,
                duration_s=(ended - began) / 1e9,
            )
        )
        d.progress("preconditions_after", 1)
        began = ended
        d.progress("historical_before", 0, 24)
        historic = d.historical(work)
        atomic_json(raw / "historical_checkpoint.json", historic)
        ended = time.monotonic_ns()
        spans.append(
            dict(
                phase="historical_reconstruction",
                started_monotonic_ns=began,
                ended_monotonic_ns=ended,
                duration_s=(ended - began) / 1e9,
            )
        )
        d.progress("historical_after", 24)
        began = ended
        d.progress("development_before", 0, 6)
        dev = d.development(work)
        evidence = dict(historical=historic, development=dev, preconditions=pre, refs=refs)
        reduced = d.reduce(evidence)
        d.require(reduced["diagnosis_ready_score"] == 1, "owned_controls")
        ended = time.monotonic_ns()
        spans.append(
            dict(
                phase="paired_development_controls",
                started_monotonic_ns=began,
                ended_monotonic_ns=ended,
                duration_s=(ended - began) / 1e9,
            )
        )
        pre["pilot_projection"] = dict(
            measured_research_s=(ended - started) / 1e9,
            validation_reserve_s=1200,
            projected_total_s=2 * (ended - started) / 1e9 + 1200,
            cap_s=4800,
            target_s=BUDGET,
        )
        d.require(pre["pilot_projection"]["projected_total_s"] < BUDGET, "execution_budget")
        d.progress("development_after", 6)
    except (OSError, ValueError, KeyError, RuntimeError, StopIteration) as error:
        external = not admitted and str(error) != "internal_resources"
        verdict = (
            "partial"
            if isinstance(error, TimeoutError)
            else "blocked"
            if external
            else "disqualified"
        )
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
        path = Path(getattr(error, "filename", None) or args.source)
        failures.append(
            dict(
                check="historical_authority" if external else "owned_controls",
                upstream="exp8860-linear-benefit-audit",
                artifact_field="learning_audit_ready_score",
                expected=1,
                observed=str(error),
                path=str(path),
                hash=sha256_file(path) if path.is_file() else None,
                status=reason,
            )
        )
        reduced = d.empty(reason)
    atomic_json(raw / "evidence.json", evidence)
    atomic_json(raw / "preconditions.json", pre)
    sources = raw / "sources"
    sources.mkdir()
    for name in d.OWNED + d.TESTS:
        shutil.copyfile(d.ROOT / name, sources / Path(name).name)
    began = time.monotonic_ns()
    d.progress("validation_before", 0, 1)
    checked = validate(raw)
    d.progress("validation_after", 1)
    if not checked["passed"]:
        verdict = "disqualified"
    if verdict in {"blocked", "disqualified", "partial"}:
        reduced["diagnosis_ready_score"] = 0
    ended = time.monotonic_ns()
    spans.append(
        dict(
            phase="validation",
            started_monotonic_ns=began,
            ended_monotonic_ns=ended,
            duration_s=(ended - began) / 1e9,
        )
    )
    substrate = "deterministic_verifier" if admitted else "blocked_no_run"
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=substrate,
        inference_substrate_class="no_model_load",
        execution_venue="host",
        inference_substrate_details=dict(
            backend="JAX CPU numerical linear heads", generator_weights_updated=False
        ),
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=spans,
        small_ebm_training=dict(
            performed=reduced["numerical_training_calls"] > 0,
            numerical_training_calls=reduced["numerical_training_calls"],
        ),
    )
    value = dict(
        reduced,
        experiment_id=8867,
        task_id="exp8867-feedback-information-diagnosis",
        milestone="2026.11.759",
        run_date=args.date,
        producer_invocation=dict(
            executable=sys.executable,
            argv=sys.argv if argv is None else [d.CLI, *argv],
            working_directory=str(Path.cwd()),
            JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS"),
        ),
        honest_verdict="complete_" + verdict + "_feedback_information_diagnosis",
        verdict_class=verdict,
        inference_substrate=substrate,
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        verifier_is_oracle=True,
        exposure_scope="constructed exact signed records; oracle labels cannot establish natural-language or oracle-distinct generalization",
        independent_generalization_score=0,
        duration_s=(ended - started) / 1e9,
        phase_spans=spans,
        preconditions_checked=pre,
        random_seed=8867,
        reproducibility_checksum=canonical_hash([d.PIN, d.GATES, 8867]),
        source_artifact_hashes=refs,
        code_config_hashes=[
            d.reference(d.ROOT / p)
            for p in d.OWNED
            + d.TESTS
            + [
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/learning_signal_runner_8815.py",
                "scripts/experiment_template.py",
            ]
        ],
        raw_shard_hashes=[
            d.reference(raw / "evidence.json"),
            d.reference(raw / "preconditions.json"),
        ]
        + [d.reference(p) for p in sorted(sources.iterdir())],
        gate_check_summary=failures,
        acceptance_gates=d.GATES,
        required_checks_passed=checked["passed"] and verdict == "null",
        validation_receipts=checked["receipts"],
        owned_statement_coverage=checked["coverage"],
        current_work_receipt=receipt,
        repository_health_receipts=[
            r for r in checked["receipts"] if r["scope"] == "repository_health"
        ],
        methodology_note="All 24 original streams reconstructed without fitting; six separate development streams with 8 complete release batches each; deterministic within-batch permutation after predictions seal. Frozen query-only admission uses syntax only. Historical attribution is descriptive, not causal proof. No headline benefit or milestone activation.",
    )
    value["field_principles"] = {
        k: "Bind actual diagnosis work to immutable inputs; completion does not establish benefit."
        for k in value
    }
    value["field_principles"].update(
        diagnosis_ready_score="Authenticated complete diagnosis and valid controls release methods regardless of benefit.",
        template_cardinality_rows="Normalized query syntax distinguishes actual topology from unused marker branches and entity spelling.",
        label_assignment_rows="Historical prefix sampling may reuse labels and alter admission; rows retain actual labels and release events.",
        exposure_difference_rows="Saved-state differences are descriptive attribution, not causal proof.",
        matched_control_rows="True and permuted learned arms share verified admission, releases, features and optimizer counts.",
        control_sensitivity="Injected separable signal must be detected while zero-signal loss remains null; no headline claim.",
        forward_method="Release-matched Exp8869 arms and explicit limits on the supported mechanism.",
        numerical_training_calls="Actual new numerical SGD calls only; historical reconstruction never fits.",
        independent_count="24 historical stream clusters; repeated queries and five reused initialization seeds do not increase n.",
    )
    candidate = raw / "candidate.json"
    atomic_json(candidate, value)
    d.progress("terminal_before", 0, 1)
    final = terminal(candidate, raw)
    d.progress("terminal_after", 1)
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
            diagnosis_ready_score=0,
        )
    atomic_json(raw / "terminal_validation.json", final)
    value["terminal_validation"] = d.reference(raw / "terminal_validation.json")
    value["field_principles"]["terminal_validation"] = (
        "Unchanged scanners and real cold replay bind executed logs."
    )
    value = normalize_artifact_for_template_write(value)
    if value["verdict_class"] == "partial":
        atomic_json(output, value)
    else:
        publish_primary(
            output, value, lambda path: dict(passed=json.loads(path.read_bytes()) == value)
        )
    for path in raw.rglob("*"):
        if path.is_file():
            path.chmod(0o400)
    d.progress("published", value["completed_count"], value["censored_count"])
    return 0

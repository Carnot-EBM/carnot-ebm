"""REQ-REPORT-8922: publish validated continuous learning evidence, not a benefit claim."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import guarded_causal_runner_8909 as inherited
from carnot.reporting import interpolated_causal_streams_8922 as m
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_runner_8816 import unqualified
from carnot.reporting.v746_retained_evidence import reference
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate, normalize_artifact_for_template_write

Json = dict[str, Any]
checks = namespace(inherited, "interpolated8922_checks")
checks.m = m
CONTEXT = [
    ("tests_first", Path("/tmp/carnot-8922-tests-first.log")),
    ("tests_first", Path("/tmp/carnot-8922-runner-tests-first.log")),
] + [
    ("repository_health", Path("/tmp/carnot-8922-health/full_pytest." + suffix))
    for suffix in ("receipt.json", "stdout", "stderr")
]


def replay(path: Path) -> Json:
    """Recompute trajectories and readiness so a replacement checksum cannot bless changed evidence."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    m.require(canonical_hash(value) == checksum, "artifact_checksum")
    m.require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8922, m.TASK, "2026.11.763", "20261108"),
        "producer_identity",
    )
    p = m.protocol()
    m.require(
        value["protocol"] == p
        and value["protocol_hash"] == canonical_hash(p)
        and value["cluster_manifest"] == p["frozen_evaluation_manifest"],
        "protocol_drift",
    )
    m.require(
        value["source_artifact_hashes"] == value["preconditions_checked"]["references"]
        and value["producer_invocation"]["executed_sources"]
        == [reference(m.ROOT / n) for n in m.OWNED],
        "source_inventory",
    )
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        m.require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_hash")
    receipt = value["current_work_receipt"]
    m.require(
        not validate_current_work_receipt(receipt)
        and value["duration_s"] == receipt["duration_s"]
        and value["model_invocation_counts"] == receipt["invocation_counts"]
        and value["numerical_training_calls"]
        == receipt["small_ebm_training"]["numerical_training_calls"]
        and value["verifier_is_oracle"] is True
        and value["independent_generalization_score"] == 0,
        "compute_scope",
    )
    raw = Path(value["raw_directory"])
    measured = json.loads((raw / "measurement.json").read_bytes())
    for key, expected in measured.items():
        m.require(value[key] == expected, "reduction_drift:" + key)
    m.require(measured == m.recover(raw, value["censor_reason"]), "retained_reduction_drift")
    counts = Counter(r["status"] for r in value["rows"])
    clusters = Counter(r["cluster_id"] for r in value["rows"] if r["status"] == "complete")
    m.require(
        value["intended_count"] == len(value["rows"]) == 18432
        and value["completed_count"] == counts["complete"]
        and value["failed_count"] == counts["failed"]
        and value["censored_count"] == counts["censored"]
        and value["excluded_count"] == 0
        and value["independent_count"] == sum(n == 768 for n in clusters.values()),
        "count_drift",
    )
    m.require(value["required_checks_passed"] == checks.qualification(value), "validation_drift")
    ready = checks.readiness(value)
    verdict = (
        "blocked"
        if not value["preconditions_checked"]["external_ready"]
        else "null"
        if ready
        else "disqualified"
    )
    m.require(
        type(value["causal_rows_ready_score"]) is int
        and value["causal_rows_ready_score"] == ready
        and value["verdict_class"] == verdict
        and value["honest_verdict"] == f"complete_{verdict}_interpolated_causal_streams",
        "readiness_drift",
    )
    m.require(verdict != "blocked" or value["gate_check_summary"], "missing_gate_diagnostic")
    if value["preconditions_checked"]["external_ready"]:
        source = next(
            r for r in value["source_artifact_hashes"] if Path(r["path"]).stem == m.SOURCE
        )
        with TemporaryDirectory(prefix="carnot8922-replay-") as directory:
            pre = m.preconditions(Path(directory) / "auth", Path(source["path"]).parent)
            m.require(pre["external_ready"], "original_producer_rejected")
            for index, ref in enumerate(value["stream_refs"]):
                saved = json.loads(Path(ref["path"]).read_bytes())
                plan = next(
                    s
                    for s in p["frozen_evaluation_manifest"]
                    if s["stream_id"] == saved["plan"]["stream_id"]
                )
                m.verify(saved, plan)
                m.progress("cold_replay_before", index, len(value["stream_refs"]) - index)
                rebuilt = m.run_stream(
                    Path(directory) / plan["stream_id"], plan, pre["heads"][str(plan["seed"])]
                )
                m.require(saved == rebuilt, "causal_stream_drift")
                m.progress("cold_replay_after", index + 1, len(value["stream_refs"]) - index - 1)
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Retain actual invocation receipts and publish only after owned and terminal checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261108"], default="20261108")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--sources", type=Path, default=m.ROOT / "results")
    parser.add_argument("--replay", "--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        try:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError, TypeError) as error:
            print("replay_rejected: " + str(error), flush=True)
            return 1
    output = (args.private_output or args.output).absolute()
    if args.private_output and output.resolve().is_relative_to(m.ROOT):
        parser.error("private outputs must remain outside checkout")
    started = time.monotonic_ns()
    raw = output.parent / "raw" / m.NAME / "invocations" / str(started)
    raw.mkdir(parents=True, mode=0o700)
    invocation = dict(
        executable=sys.executable,
        arguments=argv if argv is not None else sys.argv[1:],
        cwd=os.getcwd(),
        pid=os.getpid(),
        executed_sources=[reference(m.ROOT / n) for n in m.OWNED],
    )
    atomic_json(raw / "invocation.json", invocation)
    context: list[Json] = []
    for scope, source in CONTEXT:
        if source.is_file():
            target = raw / source.name
            shutil.copyfile(source, target)
            context.append(dict(reference(target), scope=scope, original_path=str(source)))
    template = ExperimentTemplate(
        8922,
        "Interpolated causal streams",
        "results/" + m.NAME + ".json",
        repo_root=m.ROOT,
        seed=m.SEED,
    )
    m.progress("preconditions_before", 0, 1)
    pre = m.preconditions(raw, args.sources)
    atomic_json(raw / "preconditions.json", pre)
    m.progress("preconditions_after", 1, 0)
    errors: list[str] = []
    reason = "external_or_owned_precondition"
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            with template.phase("interpolated_causal_streams"):
                m.measure(raw, pre)
            reason = "unstarted"
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            reason = str(error)
            errors.append(reason)
    result = m.recover(raw, reason)
    atomic_json(raw / "measurement.json", result)
    measured = time.monotonic_ns()
    m.progress("validation_before", 0, 1)
    with TemporaryDirectory(prefix="carnot8922-validation-") as directory:
        validation = (
            unqualified()
            if args.private_output
            else checks.validate(raw / "validation", Path(directory))
        )
    m.progress("validation_after", 1, 0)
    ended = time.monotonic_ns()
    spans = [
        dict(phase="preconditions_and_measurement", start_s=0.0, end_s=(measured - started) / 1e9),
        dict(phase="validation", start_s=(measured - started) / 1e9, end_s=(ended - started) / 1e9),
    ]
    counts = Counter(r["status"] for r in result["rows"])
    p = m.protocol()
    value = dict(
        result,
        experiment_id=8922,
        task_id=m.TASK,
        milestone="2026.11.763",
        run_date=args.date,
        inference_substrate="cpu_signed_linear_normalized_energy_head",
        inference_substrate_class="no_model_load"
        if result["numerical_training_calls"]
        else "blocked_no_run",
        execution_venue="host",
        MODEL_SPECS=[],
        verifier_is_oracle=True,
        exposure_scope="24 fresh constructed paired streams; oracle truth; research retention read-only",
        independent_generalization_score=0,
        preconditions_checked=pre,
        random_seed=m.SEED,
        producer_invocation=invocation,
        source_artifact_hashes=pre["references"],
        code_config_hashes=[reference(m.ROOT / n) for n in m.OWNED + m.TESTS + methods_files()],
        raw_shard_hashes=[reference(path) for path in sorted(raw.rglob("*.json"))]
        + validation["raw_references"],
        raw_directory=str(raw),
        context_receipts=context,
        intended_count=18432,
        completed_count=counts["complete"],
        failed_count=counts["failed"],
        censored_count=counts["censored"],
        excluded_count=0,
        independent_count=sum(
            n == 768
            for n in Counter(
                r["cluster_id"] for r in result["rows"] if r["status"] == "complete"
            ).values()
        ),
        gate_check_summary=pre["diagnostics"],
        acceptance_gates=dict(
            methods_ready_score=1,
            candidate_ready_score=1,
            causal_rows=18432,
            paired_streams=24,
            owned_statement_coverage=100,
            benefit_required=False,
            benefit_owner="Exp8925",
        ),
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        test_union=validation["union"],
        owned_errors=errors,
        protocol=p,
        protocol_hash=canonical_hash(p),
        cluster_manifest=p["frozen_evaluation_manifest"],
        censor_reason=reason,
        cited_upstream_artifacts=[
            dict(
                experiment_id=8919,
                sha256=m.PIN,
                fields_imported=["methods_ready_score", "candidate_ready_score", "protocol"],
            )
        ],
        methodology="Frozen24 paired48-query streams, delay8, four matched arms and48 read-only retention queries at0/24/48.",
        methodology_note="Original three-class Brier loss; readiness measures causal custody only. Exp8925 owns benefit inference.",
        flagged_adversarial=False,
    )
    value["raw_shard_hashes"] += context
    value["required_checks_passed"] = checks.qualification(value)
    value["causal_rows_ready_score"] = checks.readiness(value)
    verdict = (
        "blocked"
        if not pre["external_ready"]
        else "null"
        if value["causal_rows_ready_score"]
        else "disqualified"
    )
    value.update(
        verdict_class=verdict, honest_verdict=f"complete_{verdict}_interpolated_causal_streams"
    )

    def refresh(final: int) -> None:
        receipt = build_current_work_receipt(
            run_id=str(started),
            owner_pid=os.getpid(),
            events=[],
            inference_substrate=value["inference_substrate"],
            inference_substrate_class=value["inference_substrate_class"],
            execution_venue="host",
            inference_substrate_details=dict(backend="JAX CPU", pretrained_loads=0),
            started_monotonic_ns=started,
            ended_monotonic_ns=final,
            phase_spans=spans,
            small_ebm_training=dict(
                performed=result["numerical_training_calls"] > 0,
                numerical_training_calls=result["numerical_training_calls"],
            ),
        )
        value.update(
            duration_s=receipt["duration_s"],
            phase_spans=spans,
            current_work_receipt=receipt,
            model_invocation_counts=receipt["invocation_counts"],
        )

    refresh(ended)
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Actual invocation metadata or byte-bound constructed evidence; no independent natural-language claim."
        for k in value
    }
    value["field_principles"].update(
        causal_rows_ready_score="Complete causal evidence and validated replay only; zero benefit inference here.",
        independent_count="One independent paired stream per complete768-row cluster;24 scheduled, no query inflation.",
        gradient_call_ledger="Actual gradients in every arm, including all rejected candidates; replay compute is verification only.",
        optimizer_sequences="Both fast and slow persist in the same atomic transaction as pending labels and head versions.",
        release_rows="Due ordered feedback follows durable predictions; retention never reaches this interface.",
        restart_rows="At query24, actual reopened state includes pending feedback and both optimizer sequences.",
        acceptance_gates="Method and development eligibility precede future labels; Exp8925 owns the joint benefit test.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    if not args.private_output:
        m.progress("terminal_before", 0, 3)
        candidate = raw / "prospective.json"
        atomic_json(candidate, value)
        reports = checks.terminal(candidate, raw)
        value["validation_receipts"] += reports
        value["raw_shard_hashes"] += [
            reference(Path(r[k + "_path"])) for r in reports for k in ("stdout", "stderr")
        ]
        if not all(r["passed"] for r in reports):
            value.update(
                honest_verdict="complete_disqualified_interpolated_causal_streams",
                verdict_class="disqualified",
                causal_rows_ready_score=0,
                required_checks_passed=False,
            )
        final = time.monotonic_ns()
        spans.append(
            dict(phase="terminal", start_s=(ended - started) / 1e9, end_s=(final - started) / 1e9)
        )
        refresh(final)
        value.pop("reproducibility_checksum")
        value["reproducibility_checksum"] = canonical_hash(value)
        m.progress("terminal_after", 3, 0)
    m.require((time.monotonic_ns() - started) / 1e9 < 4200, "total_work_budget")
    publish_primary(output, value, replay)
    m.progress(
        "published_" + value["verdict_class"], counts["complete"], 18432 - counts["complete"]
    )
    return 0


def methods_files() -> list[str]:
    """Bind reused loops, optimizer, custody helpers and unchanged validators to this invocation."""
    return list(m.methods.OWNED) + [
        "python/carnot/reporting/guarded_causal_streams_8909.py",
        "python/carnot/reporting/guarded_causal_runner_8909.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "scripts/experiment_template.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
    ]

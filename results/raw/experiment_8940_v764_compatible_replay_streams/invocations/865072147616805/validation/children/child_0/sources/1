"""REQ-REPORT-8940: publish complete replay measurements after owned checks."""

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

from carnot.reporting import compatible_replay_streams_8940 as m
from carnot.reporting import compatible_replay_runner_8933 as inherited
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
from scripts.experiment_template import ExperimentTemplate, normalize_artifact_for_template_write

Json = dict[str, Any]
CONTEXT = [
    ("tests_first", Path("/tmp/carnot-8940-tests-first.log")),
    ("tests_first", Path("/tmp/carnot-8940-runner-tests-first.log")),
    ("tests_first", Path("/tmp/carnot-8940-gradient-receipt-tests-first.log")),
    (
        "superseded_owned_validation",
        m.ROOT
        / "results/raw/experiment_8940_v764_compatible_replay_streams/invocations/863783737405454/terminal/adversarial.stdout",
    ),
] + [
    ("repository_health", Path("/tmp/carnot-8940-health") / name)
    for name in ("receipt.json", "full_pytest.stdout", "full_pytest.stderr")
]


def adapter() -> Any:
    """Reuse measured unit/child coverage, private E2E checks and unchanged scanners."""
    runtime = inherited.adapter()
    runtime.m = m
    return runtime


def qualification(value: Json) -> bool:
    """Require owned statement coverage with no missing or excluded statements."""
    coverage = value["owned_statement_coverage"]
    return bool(
        coverage["complete"]
        and coverage["percent"] == 100
        and {r["path"] for r in coverage["files"]} == set(m.OWNED)
        and not any(r["missing"] or r["excluded"] for r in coverage["files"])
        and value["test_union"]["passed"]
        and all(r["passed"] for r in value["validation_receipts"])
    )


def readiness(value: Json) -> int:
    """Qualify causal measurement independently of loss improvement or promotion."""
    return int(
        qualification(value)
        and value["preconditions_checked"]["external_ready"]
        and value["preconditions_checked"]["resources"]["passed"]
        and not value["owned_errors"]
        and value["completed_count"] == 23040
        and len(value["stream_refs"]) == 24
        and len(value["attempt_rows"]) == 24
        and all(r["status"] == "complete" for r in value["attempt_rows"])
        and len(value["pending_feedback_rows"]) == 960
        and bool(value["control_rows"])
        and all(r["passed"] for r in value["control_rows"])
    )


def replay(path: Path) -> Json:
    """Rebuild complete trajectories so replacing a checksum cannot bless false rows."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    m.require(canonical_hash(value) == checksum, "artifact_checksum")
    m.require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8940, m.TASK, "2026.11.764", "20261108"),
        "producer_identity",
    )
    p = m.protocol()
    m.require(
        value["protocol"] == p
        and value["protocol_hash"] == canonical_hash(p)
        and value["stream_manifest"] == p["frozen_evaluation_manifest"],
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
    m.require(
        measured == m.recover(raw, value["missing_reason"] or "unstarted"),
        "retained_reduction_drift",
    )
    counts = Counter(r["status"] for r in value["rows"])
    clusters = Counter(r["cluster_id"] for r in value["rows"] if r["status"] == "complete")
    m.require(
        value["intended_count"] == len(value["rows"]) == 23040
        and value["completed_count"] == counts["complete"]
        and value["failed_count"] == counts["failed"]
        and value["censored_count"] == counts["censored"]
        and value["excluded_count"] == 0
        and value["independent_count"] == sum(n == 960 for n in clusters.values()),
        "count_drift",
    )
    m.require(value["required_checks_passed"] == qualification(value), "validation_drift")
    ready = readiness(value)
    verdict = (
        "blocked"
        if not value["preconditions_checked"]["external_ready"]
        else "null"
        if ready
        else "disqualified"
    )
    m.require(
        type(value["replay_rows_ready_score"]) is int
        and value["replay_rows_ready_score"] == ready
        and value["verdict_class"] == verdict
        and value["honest_verdict"] == f"complete_{verdict}_compatible_replay_streams",
        "readiness_drift",
    )
    m.require(verdict != "blocked" or value["gate_check_summary"], "missing_gate_diagnostic")
    if value["preconditions_checked"]["external_ready"]:
        original = next(
            r for r in value["source_artifact_hashes"] if Path(r["path"]).stem == m.SOURCE
        )
        with TemporaryDirectory(prefix="carnot8940-replay-") as directory:
            pre = m.preconditions(Path(directory) / "auth", Path(original["path"]).parent)
            m.require(pre["external_ready"], "original_producer_rejected")
            for i, ref in enumerate(value["stream_refs"]):
                saved = json.loads(Path(ref["path"]).read_bytes())
                plan = next(
                    s
                    for s in p["frozen_evaluation_manifest"]
                    if s["stream_id"] == saved["plan"]["stream_id"]
                )
                m.verify(saved, plan)
                m.progress("cold_replay_before", i, len(value["stream_refs"]) - i)
                rebuilt = m.run_stream(
                    Path(directory) / plan["stream_id"],
                    plan,
                    pre["heads"][str(plan["initial_head_seed"])],
                )
                m.require(saved == rebuilt, "causal_stream_drift")
                m.progress("cold_replay_after", i + 1, len(value["stream_refs"]) - i - 1)
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Keep all evidence private until owned checks and semantic cold replay pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261108"], default="20261108")
    parser.add_argument("--sources", type=Path, default=m.ROOT / "results")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", type=Path)
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
    template = ExperimentTemplate(
        8940,
        "Compatible replay future streams",
        "results/" + m.NAME + ".json",
        repo_root=m.ROOT,
        seed=m.SEED,
    )
    m.progress("preconditions_before", 0, 1)
    pre = m.preconditions(raw, args.sources)
    atomic_json(raw / "preconditions.json", pre)
    m.progress("preconditions_after", 1, 0)
    reason, errors = "external_or_owned_precondition", []
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            with template.phase("future_streams"):
                m.measure(raw, pre["heads"])
            reason = "unstarted"
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            reason = str(error)
            errors.append(reason)
    result = m.recover(raw, reason)
    atomic_json(raw / "measurement.json", result)
    measured = time.monotonic_ns()
    m.progress("validation_before", 0, 1)
    with TemporaryDirectory(prefix="carnot8940-validation-") as directory:
        validation = (
            unqualified()
            if args.private_output
            else adapter().validate(raw / "validation", Path(directory))
        )
    m.progress("validation_after", 1, 0)
    ended = time.monotonic_ns()
    context: list[Json] = []
    for scope, source in CONTEXT:
        if source.is_file():
            target = raw / "context" / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
            context.append(dict(reference(target), scope=scope, original_path=str(source)))
    counts = Counter(r["status"] for r in result["rows"])
    p = m.protocol()
    value = dict(
        result,
        experiment_id=8940,
        task_id=m.TASK,
        milestone="2026.11.764",
        run_date=args.date,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load"
        if result["numerical_training_calls"]
        else "blocked_no_run",
        execution_venue="host",
        MODEL_SPECS=[],
        verifier_is_oracle=True,
        exposure_scope="24 constructed independent histories; oracle truth; read-only retention",
        independent_generalization_score=0,
        preconditions_checked=pre,
        random_seed=m.SEED,
        producer_invocation=invocation,
        source_artifact_hashes=pre["references"],
        code_config_hashes=[
            reference(m.ROOT / n)
            for n in m.OWNED
            + m.TESTS
            + m.methods.OWNED
            + [
                "python/carnot/memory/guarded_memory_8905.py",
                "python/carnot/memory/signed_memory_8892.py",
                "python/carnot/reporting/guarded_update_methods_8905.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
                "scripts/experiment_template.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ],
        raw_shard_hashes=[reference(path) for path in sorted(raw.rglob("*.json"))]
        + validation["raw_references"],
        raw_directory=str(raw),
        intended_count=23040,
        completed_count=counts["complete"],
        failed_count=counts["failed"],
        censored_count=counts["censored"],
        excluded_count=0,
        independent_count=sum(
            n == 960
            for n in Counter(
                r["cluster_id"] for r in result["rows"] if r["status"] == "complete"
            ).values()
        ),
        gate_check_summary=pre["diagnostics"],
        acceptance_gates=dict(
            replay_methods_ready_score=1,
            comparison_rows=23040,
            paired_streams=24,
            terminal_pending_per_arm=8,
            owned_statement_coverage=100,
            benefit_required=False,
            benefit_owner="Exp8941",
        ),
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        test_union=validation["union"],
        owned_errors=errors,
        protocol=p,
        protocol_hash=canonical_hash(p),
        stream_manifest=p["frozen_evaluation_manifest"],
        missing_reason=reason if counts["complete"] < 23040 else None,
        cited_upstream_artifacts=[
            dict(
                producer=m.SOURCE,
                sha256=m.PIN,
                artifact_field="replay_methods_ready_score",
                expected=1,
            )
        ],
        methodology="24 sealed streams, five arms,48 predictions, delay8, retention at0/24/48; exact query24 restart against uninterrupted controls.",
        methodology_note="23040 Brier query comparisons retain typed error alongside each score;46080 endpoint cells. Exp8941 owns benefit. Oracle graphs establish no natural-language generalization.",
        flagged_adversarial=False,
        context_receipts=context,
    )
    value["required_checks_passed"] = qualification(value)
    value["replay_rows_ready_score"] = readiness(value)
    verdict = (
        "blocked"
        if not pre["external_ready"]
        else "null"
        if value["replay_rows_ready_score"]
        else "disqualified"
    )
    value.update(
        verdict_class=verdict, honest_verdict=f"complete_{verdict}_compatible_replay_streams"
    )
    spans = [
        dict(phase="preconditions_and_measurement", start_s=0.0, end_s=(measured - started) / 1e9),
        dict(phase="validation", start_s=(measured - started) / 1e9, end_s=(ended - started) / 1e9),
    ]
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Actual invocation or byte-bound constructed evidence; no benefit or natural-language claim."
        for k in value
    }
    value["field_principles"].update(
        replay_rows_ready_score="Complete causal rows, unchanged methods and owned checks; loss improvement is not a gate.",
        rows="One paired query comparison with Brier numerator/denominator and typed error;24 independent histories.",
        numerical_training_calls="Actual comparison and private-control gradients; upstream, validation children and cold verification remain separate receipts.",
        private_control_rows="Per-arm uninterrupted control receipts retain actual gradient counts; they do not add independent samples.",
        pending_feedback_rows="Eight sealed examples per arm remain unreleased at query48; no terminal training.",
        replay_rows="Every prior TRAIN buffer candidate distance and selected source id; labels cannot select examples.",
        retention_rows="Research-only reads at0/24/48; exact before/after store bytes must match.",
        restart_rows="Complete query24 reopen state, uninterrupted final state and subsequent prediction equality.",
        independent_count="One independent world/history seed per complete960-row stream; maximum24.",
    )

    def seal(final: int) -> None:
        receipt = build_current_work_receipt(
            run_id=str(started),
            owner_pid=os.getpid(),
            events=[],
            inference_substrate=value["inference_substrate"],
            inference_substrate_class=value["inference_substrate_class"],
            execution_venue="host",
            inference_substrate_details=dict(
                backend="JAX CPU signed small head", pretrained_loads=0
            ),
            started_monotonic_ns=started,
            ended_monotonic_ns=final,
            phase_spans=spans,
            small_ebm_training=dict(
                performed=result["numerical_training_calls"] > 0,
                numerical_training_calls=result["numerical_training_calls"],
            ),
        )
        value.update(
            current_work_receipt=receipt,
            model_invocation_counts=receipt["invocation_counts"],
            duration_s=receipt["duration_s"],
            phase_spans=spans,
        )
        value.pop("reproducibility_checksum", None)
        for key in list(value) + ["reproducibility_checksum"]:
            value["field_principles"].setdefault(
                key, "Current invocation evidence with actual timing and compute counters."
            )
        value["reproducibility_checksum"] = canonical_hash(value)

    seal(ended)
    if not args.private_output:
        m.progress("terminal_before", 0, 3)
        candidate = raw / "prospective.json"
        atomic_json(candidate, value)
        reports = adapter().terminal(candidate, raw)
        value["validation_receipts"] += reports
        value["raw_shard_hashes"] += [
            reference(Path(r[k + "_path"])) for r in reports for k in ("stdout", "stderr")
        ]
        if not all(r["passed"] for r in reports):
            value.update(
                honest_verdict="complete_disqualified_compatible_replay_streams",
                verdict_class="disqualified",
                replay_rows_ready_score=0,
                required_checks_passed=False,
            )
        final = time.monotonic_ns()
        spans.append(
            dict(phase="terminal", start_s=(ended - started) / 1e9, end_s=(final - started) / 1e9)
        )
        seal(final)
        m.progress("terminal_after", 3, 0)
    m.require((time.monotonic_ns() - started) / 1e9 < 4200, "total_work_budget")
    publish_primary(output, value, replay)
    m.progress(
        "published_" + value["verdict_class"], counts["complete"], 23040 - counts["complete"]
    )
    return 0

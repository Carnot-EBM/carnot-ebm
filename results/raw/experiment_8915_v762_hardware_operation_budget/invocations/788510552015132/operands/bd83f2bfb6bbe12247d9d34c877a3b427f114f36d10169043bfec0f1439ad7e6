"""REQ-REPORT-8909: publish one terminal artifact after private causal validation."""

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

from carnot.reporting import guarded_causal_streams_8909 as m
from carnot.reporting import topology_memory_runner_8881 as inherited
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_runner_8816 import stream, unqualified
from carnot.reporting.v746_retained_evidence import reference
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate, normalize_artifact_for_template_write

Json = dict[str, Any]


def validate(raw: Path, private: Path) -> Json:
    """Measure only owned statements and reuse untouched private E2E-018/021 checks."""
    runner = namespace(inherited, "guarded8909_validation")
    runner.m = m
    runner.stream = lambda name, argv, logs, **kwargs: stream(
        name,
        argv,
        logs,
        **dict(kwargs, deadline=180 if name == "collect_once" else kwargs.get("deadline", 600)),
    )
    return dict(runner.validate(raw, private))


def terminal(candidate: Path, raw: Path) -> list[Json]:
    """Run unchanged scanners and the real CLI from outside the checkout without PYTHONPATH."""
    runner = namespace(inherited, "guarded8909_terminal")
    runner.m = m
    return list(runner.terminal(candidate, raw))


def qualification(value: Json) -> bool:
    """Readiness requires complete owned validation, with no coverage exclusions."""
    coverage = value["owned_statement_coverage"]
    return bool(
        coverage["complete"]
        and coverage["percent"] == 100
        and {r["path"] for r in coverage["files"]} == set(m.OWNED)
        and not any(r["excluded"] or r["missing"] for r in coverage["files"])
        and value["test_union"]["passed"]
        and all(r["passed"] for r in value["validation_receipts"])
    )


def readiness(value: Json) -> int:
    """Release complete measurement regardless of promotions or scientific effect."""
    return int(
        value["preconditions_checked"]["external_ready"]
        and value["preconditions_checked"]["resources"]["passed"]
        and qualification(value)
        and not value["owned_errors"]
        and value["completed_count"] == 18432
        and len(value["stream_refs"]) == 24
        and len(value["attempt_rows"]) == 24
        and all(r["status"] == "complete" for r in value["attempt_rows"])
        and not value["pending_feedback_rows"]
        and all(r["passed"] for r in value["control_rows"])
    )


def replay(path: Path) -> Json:
    """Reconstruct numerical trajectories so a new checksum cannot authorize false rows."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    m.require(canonical_hash(value) == checksum, "artifact_checksum")
    m.require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8909, m.TASK, "2026.11.762", "20261107"),
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
        len(value["rows"]) == value["intended_count"] == 18432
        and value["completed_count"] == counts["complete"]
        and value["failed_count"] == counts["failed"]
        and value["censored_count"] == counts["censored"]
        and value["excluded_count"] == 0
        and value["independent_count"] == sum(n == 768 for n in clusters.values()),
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
        type(value["causal_rows_ready_score"]) is int
        and value["causal_rows_ready_score"] == ready
        and value["verdict_class"] == verdict
        and value["honest_verdict"] == f"complete_{verdict}_guarded_causal_streams",
        "readiness_drift",
    )
    m.require(verdict != "blocked" or value["gate_check_summary"], "missing_gate_diagnostic")
    if value["preconditions_checked"]["external_ready"]:
        original = next(
            r for r in value["source_artifact_hashes"] if Path(r["path"]).stem == m.SOURCE
        )
        m.require(original["sha256"] == m.PIN, "producer_pin")
        with TemporaryDirectory(prefix="carnot8909-replay-") as directory:
            pre = m.preconditions(Path(directory) / "authentication", Path(original["path"]).parent)
            m.require(pre["external_ready"], "original_producer_rejected")
            for ref in value["stream_refs"]:
                saved = json.loads(Path(ref["path"]).read_bytes())
                plan = next(
                    s
                    for s in p["frozen_evaluation_manifest"]
                    if s["stream_id"] == saved["plan"]["stream_id"]
                )
                m.verify(saved, plan)
                m.progress("cold_replay_before_" + plan["stream_id"], 0, 1)
                rebuilt = m.run_stream(
                    Path(directory) / plan["stream_id"], plan, pre["heads"][str(plan["seed"])]
                )
                m.require(saved == rebuilt, "causal_stream_drift")
                m.progress("cold_replay_after_" + plan["stream_id"], 1, 0)
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Keep checkpoints private until complete owned and terminal validation finishes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261107"], default="20261107")
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
    template = ExperimentTemplate(
        8909, "Guarded causal streams", "results/" + m.NAME + ".json", repo_root=m.ROOT, seed=m.SEED
    )
    m.progress("preconditions_before", 0, 1)
    pre = m.preconditions(raw, args.sources)
    atomic_json(raw / "preconditions.json", pre)
    m.progress("preconditions_after", 1, 0)
    errors: list[str] = []
    reason = "external_or_owned_precondition"
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            with template.phase("guarded_causal_streams"):
                m.measure(raw, pre)
            reason = "unstarted"
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            reason = str(error)
            errors.append(reason)
    result = m.recover(raw, reason)
    atomic_json(raw / "measurement.json", result)
    measured = time.monotonic_ns()
    m.progress("validation_before", 0, 1)
    with TemporaryDirectory(prefix="carnot8909-validation-") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
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
        experiment_id=8909,
        task_id=m.TASK,
        milestone="2026.11.762",
        run_date=args.date,
        inference_substrate="cpu_signed_linear_normalized_energy_head",
        inference_substrate_class="no_model_load"
        if result["numerical_training_calls"]
        else "blocked_no_run",
        execution_venue="host",
        MODEL_SPECS=[],
        verifier_is_oracle=True,
        exposure_scope="eight constructed online and eight retention topology families; fresh seeds only",
        independent_generalization_score=0,
        preconditions_checked=pre,
        random_seed=m.SEED,
        producer_invocation=invocation,
        source_artifact_hashes=pre["references"],
        code_config_hashes=[
            reference(m.ROOT / n)
            for n in m.OWNED
            + m.TESTS
            + [
                "python/carnot/memory/guarded_memory_8905.py",
                "python/carnot/reporting/guarded_update_methods_8905.py",
                "python/carnot/memory/signed_memory_8892.py",
                "scripts/experiment_template.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ],
        raw_shard_hashes=[reference(path) for path in sorted(raw.rglob("*.json"))]
        + validation["raw_references"],
        raw_directory=str(raw),
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
            guarded_methods_ready_score=1,
            candidate_ready_score=1,
            causal_rows=18432,
            paired_streams=24,
            owned_statement_coverage=100,
            benefit_required=False,
            benefit_owner="Exp8911",
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
                experiment_id=8905,
                sha256=m.PIN,
                fields_imported=[
                    "guarded_methods_ready_score",
                    "candidate_ready_score",
                    "protocol",
                    "anchor_manifest",
                    "frozen_evaluation_manifest",
                ],
            )
        ],
        methodology="Frozen24 paired48-query streams, four arms,48 read-only retention queries at0/24/48. Identical feature/template exposure and numerical work; delayed labels and atomic candidate deployment.",
        methodology_note="Exact constructed truth is oracle supervision. Repeated operational guards are not independent risk certificates. Readiness is complete measurement; Exp8911 owns independent intervals and benefit.",
        flagged_adversarial=False,
    )
    value["required_checks_passed"] = qualification(value)
    value["causal_rows_ready_score"] = readiness(value)
    verdict = (
        "blocked"
        if not pre["external_ready"]
        else "null"
        if value["causal_rows_ready_score"]
        else "disqualified"
    )
    value.update(verdict_class=verdict, honest_verdict=f"complete_{verdict}_guarded_causal_streams")
    health = Path("/tmp/carnot-8909-health/receipt.json")
    if health.is_file():
        record = json.loads(health.read_bytes())
        copies = []
        for suffix in ("stdout", "stderr"):
            target = raw / ("repository_health." + suffix)
            shutil.copyfile(record[suffix + "_path"], target)
            copies.append(reference(target))
        value["repository_health"] = dict(
            scope="repository_health",
            invocation_receipt=record,
            retained_logs=copies,
            interpretation="One bounded whole-suite run; failures are separate from owned qualification.",
        )
        value["raw_shard_hashes"] += copies + [reference(health)]
    tests_first = Path("/tmp/carnot-8909-tests-first.log")
    if tests_first.is_file():
        shutil.copyfile(tests_first, raw / "tests_first.log")
        value["raw_shard_hashes"].append(reference(raw / "tests_first.log"))

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
        k: "Actual invocation metadata or byte-bound finite constructed evidence; no independent natural-language claim."
        for k in value
    }
    value["field_principles"].update(
        causal_rows_ready_score="Complete trustworthy18432 scheduled outcomes release measurement regardless of promotions or effect.",
        numerical_training_calls="Actual current evaluation and development restart-pair SGD steps; validation and reader reconstructions are separately logged verification work.",
        independent_count="Each completely measured paired stream counts once; repeated queries and five head seeds add no units.",
        prediction_rows="Durable parameter/template hashes and sequence precede due-time label access.",
        retention_rows="Untouched48 queries at0/24/48; durable state bytes remain identical across each read-only checkpoint.",
        promotion_rows="Every scheduled candidate is retained, including ineligible and rejected deployment decisions.",
        protocol_hash="Original producer protocol stays frozen; seed novelty establishes no unseen-topology claim.",
        reproducibility_checksum="Canonical artifact hash plus cold numerical reconstruction rejects rehashed scientific changes.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    candidate = raw / "prospective.json"
    atomic_json(candidate, value)
    if not args.private_output:
        m.progress("terminal_before", 0, 3)
        reports = terminal(candidate, raw)
        value["validation_receipts"] += reports
        value["raw_shard_hashes"] += [
            reference(Path(r[k + "_path"])) for r in reports for k in ("stdout", "stderr")
        ]
        if not all(r["passed"] for r in reports):
            value.update(
                honest_verdict="complete_disqualified_guarded_causal_streams",
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

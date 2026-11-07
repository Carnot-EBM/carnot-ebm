"""REQ-REPORT-8895: expose one terminal artifact after measured private validation."""

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

from carnot.reporting import signed_causal_streams_8895 as m
from carnot.reporting import topology_causal_runner_8884 as inherited
from carnot.reporting import topology_memory_runner_8881 as prior_runner
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_runner_8816 import unqualified
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate, normalize_artifact_for_template_write

Json = dict[str, Any]


def validate(raw: Path, private: Path) -> Json:
    """Reuse original unit/child coverage, strict checks and private E2E-018/021 routes."""
    runner = namespace(inherited, "signed8895_validation")
    runner.m = m
    return dict(runner.validate(raw, private))


def terminal(candidate: Path, raw: Path) -> list[Json]:
    """The original scanners and outside-checkout CLI inspect prospective bytes."""
    runner = namespace(prior_runner, "signed8895_terminal")
    runner.m = m
    return list(runner.terminal(candidate, raw))


def replay(path: Path) -> Json:
    """Rebuild actual trajectories so rehashed false causal rows cannot gain readiness."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    m.require(canonical_hash(value) == checksum, "artifact_checksum")
    m.require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8895, m.TASK, "2026.11.761", "20261107"),
        "producer_identity",
    )
    m.require(
        value["frozen_protocol"] == m.methods.protocol()
        and value["feature_schema"] == m.methods.protocol()["feature_schema"],
        "protocol_drift",
    )
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        load_ref = reference(Path(ref["path"]))
        m.require(load_ref["sha256"] == ref["sha256"], "operand_hash")
    m.upstream().verify_inputs(value)
    receipt = value["current_work_receipt"]
    m.require(
        not validate_current_work_receipt(receipt)
        and value["duration_s"] == receipt["duration_s"]
        and value["model_invocation_counts"] == receipt["invocation_counts"]
        and value["verifier_is_oracle"] is True
        and value["independent_generalization_score"] == 0,
        "compute_scope",
    )
    raw = Path(value["raw_directory"])
    result = load(reference(raw / "measurement.json"))
    for key, expected in result.items():
        m.require(canonical_hash(value[key]) == canonical_hash(expected), "reduction_drift:" + key)
    rebuilt = m.recover(raw, value["censor_reason"])
    m.require(canonical_hash(result) == canonical_hash(rebuilt), "retained_reduction_drift")
    counts = {
        s: sum(r["status"] == s for r in value["rows"]) for s in ("complete", "failed", "censored")
    }
    cluster_counts = Counter(r["cluster_id"] for r in value["rows"] if r["status"] == "complete")
    complete_clusters = {k for k, n in cluster_counts.items() if n == 1536}
    m.require(
        len(value["rows"]) == value["intended_count"] == 36864
        and all(
            value[s + "_count"] == counts[t]
            for s, t in (("completed", "complete"), ("failed", "failed"), ("censored", "censored"))
        )
        and value["excluded_count"] == 0
        and value["independent_count"] == len(complete_clusters),
        "count_drift",
    )
    valid = bool(
        value["owned_statement_coverage"]["complete"]
        and value["owned_statement_coverage"]["percent"] == 100
        and value["test_union"]["passed"]
        and all(r["passed"] for r in value["validation_receipts"])
    )
    m.require(value["required_checks_passed"] == valid, "validation_drift")
    ready = int(
        value["preconditions_checked"]["external_ready"]
        and value["preconditions_checked"]["resources"]["passed"]
        and counts["complete"] == 36864
        and len(result["stream_refs"]) == 24
        and not result["partial_refs"]
        and len(result["attempt_rows"]) == 24
        and all(a["status"] == "complete" for a in result["attempt_rows"])
        and valid
        and not value["owned_errors"]
        and all(r["passed"] for r in value["control_rows"])
    )
    verdict = (
        "blocked"
        if not value["preconditions_checked"]["external_ready"]
        else "null"
        if ready
        else "disqualified"
    )
    m.require(
        value["causal_rows_ready_score"] == ready
        and value["verdict_class"] == verdict
        and value["honest_verdict"] == f"complete_{verdict}_signed_causal_streams",
        "readiness_or_verdict_drift",
    )
    if result["stream_refs"]:
        heads = load(value["preconditions_checked"]["fitted_heads"])["heads"]
        p = m.methods.protocol()
        with TemporaryDirectory(prefix="carnot8895-replay-") as directory:
            for ref in result["stream_refs"]:
                original = load(ref)
                plans = {s["stream_id"]: s for s in p["cohort_manifest"]}
                plan = plans[original["arms"][0]["stream"]["stream_id"]]
                m.require(
                    all(a["stream"] == plan for a in original["arms"]), "registered_stream_drift"
                )
                m.verify(original)
                reproduced = m.run_stream(Path(directory) / plan["stream_id"], plan, heads, p)
                m.require(
                    canonical_hash(m.causal.semantic(original))
                    == canonical_hash(m.causal.semantic(reproduced)),
                    "causal_stream_drift",
                )
    return dict(passed=True, status="replay_passed", **reference(path))


def main(argv: list[str] | None = None) -> int:
    """Run fixed work, retain all failures and publish only after terminal validation."""
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
        8895, "Signed causal streams", "results/" + m.NAME + ".json", repo_root=m.ROOT, seed=m.SEED
    )
    m.progress("preconditions_before", 0, 1)
    pre = m.preconditions(raw, args.sources)
    atomic_json(raw / "preconditions.json", pre)
    m.progress("preconditions_after", 1, 0)
    errors, reason = [], "external_input_unavailable"
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            with template.phase("causal_streams"):
                m.measure(raw, pre)
            reason = "unstarted"
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            reason = str(error)
            errors.append(reason)
    result = m.recover(raw, reason)
    atomic_json(raw / "measurement.json", result)
    measured = time.monotonic_ns()
    m.progress("validation_before", 0, 1)
    with TemporaryDirectory(prefix="carnot8895-validation-") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
    valid = bool(
        validation["coverage"]["complete"]
        and validation["union"]["passed"]
        and all(r["passed"] for r in validation["receipts"])
    )
    completed = sum(r["status"] == "complete" for r in result["rows"])
    ready = int(
        pre["external_ready"]
        and pre["resources"]["passed"]
        and valid
        and not errors
        and completed == 36864
        and len(result["stream_refs"]) == 24
        and not result["partial_refs"]
        and len(result["attempt_rows"]) == 24
        and all(a["status"] == "complete" for a in result["attempt_rows"])
        and all(r["passed"] for r in result["control_rows"])
    )
    verdict = "blocked" if not pre["external_ready"] else "null" if ready else "disqualified"
    ended = time.monotonic_ns()
    spans = [
        dict(phase="preconditions_and_measurement", start_s=0.0, end_s=(measured - started) / 1e9),
        dict(phase="validation", start_s=(measured - started) / 1e9, end_s=(ended - started) / 1e9),
    ]
    receipt = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_linear_normalized_energy_head",
        inference_substrate_class="no_model_load"
        if result["numerical_training_calls"]
        else "blocked_no_run",
        execution_venue="host",
        inference_substrate_details=dict(backend="JAX CPU", pretrained_loads=0),
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=spans,
        small_ebm_training=dict(
            performed=result["numerical_training_calls"] > 0,
            numerical_training_calls=result["numerical_training_calls"],
        ),
    )
    helpers = [
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "python/carnot/reporting/topology_causal_streams_8884.py",
        "python/carnot/reporting/topology_causal_runner_8884.py",
        "python/carnot/reporting/topology_memory_runner_8881.py",
    ]
    value = dict(
        result,
        experiment_id=8895,
        task_id=m.TASK,
        milestone="2026.11.761",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_signed_causal_streams",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        verifier_is_oracle=True,
        exposure_scope="finite constructed signed relational queries only",
        independent_generalization_score=0,
        duration_s=receipt["duration_s"],
        phase_spans=spans,
        preconditions_checked=pre,
        random_seed=m.SEED,
        producer_invocation=invocation,
        current_work_receipt=receipt,
        source_artifact_hashes=pre["references"],
        code_config_hashes=[reference(m.ROOT / n) for n in m.OWNED + m.TESTS + helpers],
        raw_shard_hashes=[reference(p) for p in sorted(raw.rglob("*.json"))]
        + validation["raw_references"],
        raw_directory=str(raw),
        intended_count=36864,
        completed_count=completed,
        failed_count=sum(r["status"] == "failed" for r in result["rows"]),
        censored_count=sum(r["status"] == "censored" for r in result["rows"]),
        excluded_count=0,
        independent_count=sum(
            n == 1536
            for n in Counter(
                r["cluster_id"] for r in result["rows"] if r["status"] == "complete"
            ).values()
        ),
        gate_check_summary=pre["diagnostics"],
        acceptance_gates=dict(
            causal_rows_required=36864,
            paired_streams=24,
            owned_statement_coverage=100,
            benefit_required=False,
            benefit_owner="Exp8898",
        ),
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        test_union=validation["union"],
        owned_errors=errors,
        causal_rows_ready_score=ready,
        feature_schema=m.methods.protocol()["feature_schema"],
        frozen_protocol=m.methods.protocol(),
        censor_reason=reason,
        cited_upstream_artifacts=[
            dict(
                experiment_id=8892,
                artifact_field="signed_methods_ready_score",
                expected=1,
                sha256=m.PIN,
            )
        ],
        methodology="Fixed24 paired streams and eight arms. Frozen signed heads, due-time matched labels, atomic admissions, untouched retention and paired restart controls. No refit or threshold search.",
        methodology_note="Exact constructed labels are oracle supervision. Readiness releases causal rows; Exp8898 owns benefit decisions. No natural-language generalization or physical acceleration is measured.",
        flagged_adversarial=False,
    )
    health = Path("/tmp/carnot-8895-repository-health.log")
    if health.is_file():
        shutil.copyfile(health, raw / "repository_health.log")
        value["repository_health"] = dict(
            scope="repository_health",
            command=".venv/bin/pytest tests/python -q",
            **reference(raw / "repository_health.log"),
            interpretation="Whole-suite health is separate from owned qualification.",
        )
        value["raw_shard_hashes"].append(reference(raw / "repository_health.log"))
        outcome = Path("/tmp/carnot-8895-repository-health.json")
        if outcome.is_file():
            value["repository_health"].update(json.loads(outcome.read_bytes()))
            shutil.copyfile(outcome, raw / "repository_health.json")
            value["raw_shard_hashes"].append(reference(raw / "repository_health.json"))
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Measured producer field or immutable operand; scope is the finite constructed query family."
        for k in value
    }
    value["field_principles"].update(
        causal_rows_ready_score="Complete sealed trajectories and owned validation release the audit, regardless of effect.",
        independent_count="Paired streams count once; five head seeds and repeated topologies add no independent units.",
        numerical_training_calls="Actual current optimizer calls, including private paired pilots and sensitivity controls; imported fitting is excluded.",
        prediction_rows="Parameters and template hashes are sealed durably before any due-time label becomes available.",
        retention_rows="Same untouched48 queries at0/24/48; no retention-driven updates.",
        restart_receipts="Registered query24 reopens must match the private uninterrupted pilot, including event acknowledgments.",
        reproducibility_checksum="Canonical checksum plus cold numerical reconstruction detects rehashed scientific tampering.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    candidate = raw / "prospective.json"
    atomic_json(candidate, value)
    if not args.private_output:
        m.progress("terminal_validation_before", 0, 3)
        reports = terminal(candidate, raw)
        value["validation_receipts"] += reports
        value["raw_shard_hashes"] += [
            reference(Path(r[k + "_path"])) for r in reports for k in ("stdout", "stderr")
        ]
        if not all(r["passed"] for r in reports):
            value.update(
                honest_verdict="complete_disqualified_signed_causal_streams",
                verdict_class="disqualified",
                causal_rows_ready_score=0,
                required_checks_passed=False,
            )
        final = time.monotonic_ns()
        spans.append(
            dict(
                phase="terminal_validation",
                start_s=(ended - started) / 1e9,
                end_s=(final - started) / 1e9,
            )
        )
        receipt.update(
            ended_monotonic_ns=final, duration_s=(final - started) / 1e9, phase_spans=spans
        )
        value.update(
            duration_s=receipt["duration_s"], phase_spans=spans, current_work_receipt=receipt
        )
        value.pop("reproducibility_checksum")
        value["reproducibility_checksum"] = canonical_hash(value)
        m.progress("terminal_validation_after", 3, 0)
    publish_primary(output, value, replay)
    for path in raw.rglob("*"):
        if path.is_file():
            path.chmod(0o400)
    m.progress("published_" + value["verdict_class"], completed, 36864 - completed)
    return 0

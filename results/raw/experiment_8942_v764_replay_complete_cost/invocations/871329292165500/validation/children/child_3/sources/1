"""REQ-REPORT-8942: publish qualified complete costs independently of benefit."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import replay_complete_cost_8942 as m
from carnot.reporting import topology_memory_runner_8881 as shared_validation
from carnot.reporting import topology_causal_runner_8884 as shared_terminal
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
from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]
validation_runtime = namespace(shared_validation, "replay8942_validation")
validation_runtime.m = m
terminal_runtime = namespace(shared_terminal, "replay8942_terminal")
terminal_runtime.m = m


def validate(raw: Path, private: Path) -> Json:
    """Reuse measured unit/child coverage and unchanged private lifecycle CLI controls."""
    result = dict(validation_runtime.validate(raw, private))
    result["receipts"].append(terminal_runtime.e2e018(raw / "e2e018", private / "e2e018"))
    result["raw_references"] += [
        reference(p) for p in sorted((raw / "e2e018").rglob("*")) if p.is_file()
    ]
    return result


def terminal(candidate: Path, raw: Path) -> list[Json]:
    """Run unchanged scanners and cold reduction without repeating any measurement."""
    return list(terminal_runtime.terminal(candidate, raw))


def qualification(value: Json) -> bool:
    """Require measured coverage for every owned statement and all owned receipts."""
    c = value["owned_statement_coverage"]
    return bool(
        c["complete"]
        and c["percent"] == 100
        and {r["path"] for r in c["files"]} == set(m.OWNED)
        and not any(r["missing"] or r["excluded"] for r in c["files"])
        and value["test_union"]["passed"]
        and value["validation_receipts"]
        and all(r["passed"] for r in value["validation_receipts"])
    )


def semantics(value: Json) -> tuple[bool, str]:
    """A valid complete null qualifies; absent inputs and failed controls do not."""
    ready = bool(
        qualification(value)
        and value["preconditions_checked"]["external_ready"]
        and value["preconditions_checked"]["resources"]["passed"]
        and not value["owned_errors"]
        and value["completed_count"] == 288
        and value["independent_count"] == 24
        and len(value["durability_rows"]) == 96
        and all(r["passed"] for r in value["durability_rows"])
    )
    return ready, "blocked" if not value["preconditions_checked"][
        "external_ready"
    ] else "null" if ready else "disqualified"


def recommendation(value: Json) -> bool:
    """Permit a recommendation only with authenticated benefit and a whole-service saving."""
    quality = value["quality_reference"]
    return bool(
        value["cost_rows_ready_score"]
        and quality["authenticated"]
        and type(quality["compatible_benefit_score"]) is int
        and quality["compatible_benefit_score"] == 1
        and any(
            r["arm"] == "compatible_replay"
            and r["metric"] == "complete_process_s"
            and r["independent_count"] == 24
            and r["upper"] < 0
            for r in value["paired_cost_intervals"]
        )
    )


def seal(value: Json, started: int, measured: int) -> None:
    """Bind actual work and outer-clock provenance without importing upstream compute."""
    ended = time.monotonic_ns()
    spans = [
        dict(phase="authentication_and_service", start_s=0.0, end_s=(measured - started) / 1e9),
        dict(
            phase="owned_validation",
            start_s=(measured - started) / 1e9,
            end_s=(ended - started) / 1e9,
        ),
    ]
    receipt = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load"
        if value["numerical_training_calls"]
        else "blocked_no_run",
        execution_venue="host",
        inference_substrate_details=dict(
            scope="CPU small signed energy heads on finite constructed streams; paired durable service and separate uninterrupted controls; validation fixtures excluded from research counters"
        ),
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=spans,
        small_ebm_training=dict(
            performed=value["numerical_training_calls"] > 0,
            numerical_training_calls=value["numerical_training_calls"],
            generator_weights_frozen=True,
        ),
    )
    value.update(
        current_work_receipt=receipt,
        duration_s=receipt["duration_s"],
        phase_spans=spans,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        model_invocation_counts=receipt["invocation_counts"],
    )
    value["field_principles"] = {
        k: "Actual invocation or authenticated finite service evidence; no natural-language generalization or acceleration claim."
        for k in value
    }
    value["field_principles"].update(
        cost_rows_ready_score="Complete costs and valid controls; no speedup or learning prerequisite.",
        rows="One process/arm/metric; seconds per whole window, request/update denominators48.",
        process_seed_rows="24 independent paired process/world seeds; repeats never increase n.",
        paired_cost_intervals="Nine paired process t intervals, simultaneous99 percent; finite workloads only.",
        operation_counts="Derived outside timed windows from retained service state; controls counted separately in numerical_training_calls.",
        hardware_compatibility_rows="Existing limits and candidate operation mapping only; no board probe or repair.",
        removable_fraction="Null because no qualified component timing exists.",
        service_100x_projection="Null because no measured removable fraction exists.",
        quality_reference="Authenticated Exp8941 only gates deployment; never gates cost measurement.",
        deployment_recommendation="Requires cost readiness, quality benefit1 and a negative complete-service upper bound.",
        numerical_training_calls="Actual small-head gradient steps in compared and separate uninterrupted service controls; zero pretrained calls.",
        reproducibility_checksum="Canonical artifact identity; cold replay rebuilds all costs from authenticated retained states and bounds.",
    )
    value.pop("reproducibility_checksum", None)
    value["reproducibility_checksum"] = canonical_hash(value)


def replay(path: Path) -> Json:
    """Rebuild every headline from retained costs; never benchmark again during replay."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    m.require(canonical_hash(value) == checksum, "artifact_checksum")
    m.require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8942, m.TASK, "2026.11.764", "20261109"),
        "producer_identity",
    )
    m.require(
        value["roster"] == m.roster()
        and value["producer_invocation"]["executed_sources"]
        == [reference(m.ROOT / n) for n in m.OWNED],
        "executed_protocol",
    )
    refs = value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    for i, ref in enumerate(refs):
        m.require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_hash")
        if i % 200 == 0:
            m.progress("cold_authenticate", i, len(refs) - i)
    rebuilt = m.reduce(Path(value["raw_directory"]), value["missing_reason"])
    for key, expected in rebuilt.items():
        m.require(value[key] == expected, "reduction_drift:" + key)
    receipt = value["current_work_receipt"]
    m.require(
        not validate_current_work_receipt(receipt)
        and value["duration_s"] == receipt["duration_s"]
        and value["phase_spans"] == receipt["phase_spans"]
        and value["model_invocation_counts"] == receipt["invocation_counts"]
        and value["numerical_training_calls"]
        == receipt["small_ebm_training"]["numerical_training_calls"]
        and value["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
        and value["inference_substrate_class"] == receipt["inference_substrate_class"]
        and value["verifier_is_oracle"] is True
        and value["independent_generalization_score"] == 0,
        "current_compute_scope",
    )
    ready, verdict = semantics(value)
    m.require(
        type(value["cost_rows_ready_score"]) is int
        and value["cost_rows_ready_score"] == int(ready)
        and value["verdict_class"] == verdict
        and value["honest_verdict"] == "complete_" + verdict + "_replay_complete_cost"
        and value["required_checks_passed"] == qualification(value),
        "readiness_drift",
    )
    m.require(value["deployment_recommendation"] == recommendation(value), "deployment_drift")
    m.require(verdict != "blocked" or value["gate_check_summary"], "missing_gate_diagnostic")
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Keep prospective bytes private until owned validation and cold replay pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261109"], default="20261109")
    parser.add_argument("--sources", type=Path, default=m.ROOT / "results")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--replay", "--cold-replay", type=Path)
    parser.add_argument("--segment", type=Path)
    parser.add_argument("--stop", type=int, choices=[24, 48], default=48)
    args = parser.parse_args(argv)
    if args.segment:
        m.worker(args.segment, args.stop)
        return 0
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
        executed_sources=[reference(m.ROOT / p) for p in m.OWNED],
    )
    atomic_json(raw / "invocation.json", invocation)
    template = ExperimentTemplate(
        8942,
        "Complete replay service costs",
        "results/" + m.NAME + ".json",
        repo_root=m.ROOT,
        seed=m.SEED,
    )
    m.progress("preconditions_before", 0, 1)
    pre = m.preconditions(raw, args.sources)
    m.progress("preconditions_after", 1, 0)
    reason, errors = "external_or_owned_precondition", []
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            with template.phase("complete_service_windows"):
                m.measure(raw, pre)
            reason = "unfinished"
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            reason = str(error)
            errors.append(reason)
    result = m.reduce(raw, reason)
    atomic_json(raw / "measurement.json", result)
    measured = time.monotonic_ns()
    m.progress("validation_before", 0, 1)
    with TemporaryDirectory(prefix="carnot8942-validation-") as directory:
        try:
            validation = (
                unqualified()
                if args.private_output
                else validate(raw / "validation", Path(directory))
            )
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            validation = unqualified()
            errors.append(str(error))
    m.progress("validation_after", 1, 0)
    context = []
    for path in sorted(Path("/tmp").glob("carnot-8942-*first.log")) + sorted(
        Path("/tmp/carnot-8942-health").glob("*")
    ):
        if path.is_file():
            target = raw / "context" / path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            context.append(reference(target))
    helpers = (
        m.causal.OWNED
        + m.methods.OWNED
        + [
            "python/carnot/memory/guarded_memory_8905.py",
            "python/carnot/reporting/guarded_durable_cost_8914.py",
            "python/carnot/reporting/linear_durable_cost_8862.py",
            "python/carnot/reporting/durable_cost_qualification_8926.py",
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "scripts/experiment_template.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "research-hardware-wishlist.md",
        ]
    )
    value = dict(
        result,
        experiment_id=8942,
        task_id=m.TASK,
        milestone="2026.11.764",
        run_date=args.date,
        execution_venue="host",
        MODEL_SPECS=[],
        verifier_is_oracle=True,
        exposure_scope="Finite constructed recurring topology streams only; no natural-language generalization, verifier moat or generator learning.",
        independent_generalization_score=0,
        preconditions_checked=pre,
        random_seed=m.SEED,
        clock_contract=dict(
            clock="time.monotonic_ns",
            implementation=time.get_clock_info("monotonic").implementation,
            authority="Exp8926 qualified whole-service outer clocks",
            observer_subtraction=False,
            component_clocks=False,
            counter_collection="retained state outside timed windows",
        ),
        producer_invocation=invocation,
        source_artifact_hashes=pre["references"],
        code_config_hashes=[reference(m.ROOT / p) for p in m.OWNED + m.TESTS + helpers],
        raw_shard_hashes=[reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
        + validation["raw_references"],
        raw_directory=str(raw),
        gate_check_summary=pre["diagnostics"],
        acceptance_gates=dict(
            process_seeds=24,
            trainable_arms=4,
            cycles=48,
            restart_prediction=24,
            owned_statement_percent=100,
            exclusions=0,
            per_phase_clocks=False,
        ),
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        test_union=validation["union"],
        owned_errors=errors,
        missing_reason=reason,
        quality_reference=pre["quality_reference"],
        cited_upstream_artifacts=[
            dict(
                producer=n, sha256=p, scope="opaque original producer; no current compute imported"
            )
            for n, (p, _) in m.PINS.items()
        ],
        roster=m.roster(),
        methodology="24 fresh paired independent process/world seeds; rotating four trainable arms;48 predictions and delayed releases; cold restart after prediction24; terminal feedback drain; complete uninstrumented outer monotonic windows; retained-state counters outside timing. Separate uninterrupted controls. No observer subtraction or component attribution.",
        repository_health=dict(
            command=".venv/bin/pytest tests/python -q",
            retained=context,
            scope="Separate unrelated repository health; cannot grant cost readiness",
        ),
        flagged_adversarial=False,
        field_principles={},
    )
    ready, verdict = semantics(value)
    value.update(
        required_checks_passed=qualification(value),
        cost_rows_ready_score=int(ready),
        verdict_class=verdict,
        honest_verdict="complete_" + verdict + "_replay_complete_cost",
    )
    value["deployment_recommendation"] = recommendation(value)
    value["deployment_quality_gate"] = dict(
        upstream=m.QUALITY,
        artifact_field="compatible_benefit_score",
        expected=1,
        observed=pre["quality_reference"]["compatible_benefit_score"],
        authenticated=pre["quality_reference"]["authenticated"],
    )
    seal(value, started, measured)

    def checked(candidate: Path) -> Json:
        reports = [] if args.private_output else terminal(candidate, raw)
        return dict(
            passed=all(r["passed"] for r in reports) and replay(candidate)["passed"],
            receipts=reports,
        )

    publish_primary(output, value, checked)
    m.progress("published", value["completed_count"], value["censored_count"])
    return 0

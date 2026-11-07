"""REQ-REPORT-8906: publish proof qualification without claiming label savings."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import finite_budget_methods_8906 as m
from carnot.reporting import shared_terminal_runner_8893 as previous
from carnot.reporting import topology_memory_runner_8881 as terminal_runner
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_runner_8816 import unqualified
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate, normalize_artifact_for_template_write

Json = dict[str, Any]
SOURCE = "experiment_8896_v761_shared_terminal_evaluation"
PIN = "sha256:4874e84d0b36fc1b7891e1ccaf93b53b60e75821bab156a4ace0c4f7307bc5a8"
FIELD = "certificate_rows_ready_score"
HEALTH = Path("/tmp/carnot-8906-work/repository-health.json")


def require(condition: Any, reason: str) -> None:
    """A new checksum cannot authorize false scientific or producer assertions."""
    if not condition:
        raise ValueError(reason)


def preconditions(raw: Path, sources: Path) -> Json:
    """Authenticate only imported original operands, preserving every producer assertion."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    path = sources / (SOURCE + ".json")
    diagnostic: Json = dict(
        check="missing_file",
        upstream=SOURCE,
        artifact_field=FIELD,
        expected=1,
        observed=None,
        path=str(path),
        hash=None,
    )
    refs, diagnostics, population = [], [], None
    try:
        diagnostic["hash"] = sha256_file(path)
        diagnostic["check"] = "rejected_producer"
        value = json.loads(path.read_bytes())
        require(isinstance(value, dict), "producer_object")
        diagnostic.update(check="missing_field", observed=value.get(FIELD))
        score = value[FIELD]
        diagnostic["check"] = "observed_zero"
        require(type(score) is int and score == 1, "readiness_zero")
        diagnostic.update(
            check="rejected_producer",
            artifact_field="sha256",
            expected=PIN,
            observed=diagnostic["hash"],
        )
        require(
            diagnostic["hash"] == PIN
            and value["experiment_id"] == 8896
            and value["task_id"] == "exp8896-shared-terminal-evaluation"
            and value["verdict_class"] in ["null", "circular_positive"]
            and value["required_checks_passed"] is True,
            "original_producer",
        )
        refs.append(reference(path))
        operands = value["code_config_hashes"] + value["raw_shard_hashes"]
        for index, ref in enumerate(operands):
            diagnostic.update(
                path=ref["path"], expected=ref["sha256"], artifact_field="imported_operand"
            )
            diagnostic["observed"] = sha256_file(Path(ref["path"]))
            diagnostic["hash"] = diagnostic["observed"]
            require(diagnostic["observed"] == ref["sha256"], "original_operand_drift")
            refs.append(ref)
            if Path(ref["path"]).name == "population.json":
                population = ref
            if index % 50 == 0:
                m.progress("authenticate_original_operands", index, len(operands) - index)
        if population is None:
            raise ValueError("original_population_missing")
        require(
            sum(x["split"] == "population" for x in load(population)["inputs"]) == 4096,
            "original_census_size",
        )
        shutil.copyfile(path, raw / "original_producer.json")
    except (OSError, ValueError, KeyError, TypeError) as error:
        diagnostics.append(
            dict(
                diagnostic,
                error=str(error),
                check="missing_file"
                if isinstance(error, OSError)
                else "missing_field"
                if isinstance(error, KeyError)
                else diagnostic["check"],
            )
        )
    return dict(
        external_ready=not diagnostics,
        diagnostics=diagnostics,
        references=refs,
        census_reference=population,
        explicit_execution_gates=[],
        resources=dict(passed=shutil.disk_usage(raw).free > 128 * 1024 * 1024),
    )


def validate(raw: Path, private: Path) -> Json:
    """Reuse owned unit/child coverage, strict checks and private E2E018/021."""
    runner = namespace(previous, "finite8906_validation")
    runner.m = m
    return dict(runner.validate(raw, private))


def scheduled(reason: str, status: str = "censored") -> Json:
    """Retain the scheduled proof when an external input or owned check fails."""
    return dict(
        rows=[dict(m.row("proof_not_run", False), status=status, missing_reason=reason)],
        proof_obligations=dict(proof=m.PROOF, passed=False),
        exhaustive_path_rows=[],
        bound_monotonicity_rows=[],
        randomized_state_rows=[],
        cost_identity_rows=[],
        family_alpha_ledger=m.alpha_ledger(),
        unsafe_control={},
        numerical_training_calls=0,
        scientific_benefit_claim=False,
    )


def replay(path: Path) -> Json:
    """Recompute all proof cases so rehashed semantic tampering cannot grant readiness."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    require(canonical_hash(value) == checksum, "artifact_checksum")
    require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8906, m.TASK, "2026.11.762", "20261107"),
        "producer_identity",
    )
    require(canonical_hash(value["protocol"]) == canonical_hash(m.protocol()), "protocol_drift")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_hash")
    require(not validate_current_work_receipt(value["current_work_receipt"]), "current_receipt")
    require(
        value["verifier_is_oracle"] is True
        and value["independent_generalization_score"] == 0
        and value["independent_count"] == 0
        and value["numerical_training_calls"] == 0
        and value["duration_s"] == value["current_work_receipt"]["duration_s"]
        and value["model_invocation_counts"] == value["current_work_receipt"]["invocation_counts"],
        "compute_scope_drift",
    )
    for status in ["completed", "failed", "censored"]:
        require(
            value[status + "_count"] == sum(x["status"] == status for x in value["rows"]),
            "count_drift",
        )
    require(
        value["intended_count"]
        == len(value["rows"])
        == sum(value[s + "_count"] for s in ["completed", "failed", "censored"])
        and value["excluded_count"] == 0,
        "count_drift",
    )
    if value["completed_count"]:
        result = m.qualify()
        require(
            all(canonical_hash(value[k]) == canonical_hash(v) for k, v in result.items()),
            "reduction_drift",
        )
        require(
            load(value["calibration_manifest"]) == m.calibration_manifest(), "calibration_drift"
        )
    coverage = value["owned_statement_coverage"]
    valid = (
        coverage["complete"]
        and coverage["percent"] == 100
        and {x["path"] for x in coverage["files"]} == set(m.OWNED)
        and all(not x["missing"] and not x["excluded"] for x in coverage["files"])
        and all(x["passed"] for x in value["validation_receipts"])
    )
    require(
        type(value["finite_budget_ready_score"]) is int
        and value["finite_budget_ready_score"]
        == int(
            value["completed_count"] > 0
            and value["proof_obligations"]["passed"]
            and value["required_checks_passed"]
            and valid
            and value["verdict_class"] == "null"
        ),
        "readiness_drift",
    )
    require(not value["required_checks_passed"] or valid, "qualification_drift")
    require(
        value["verdict_class"] != "blocked" or value["gate_check_summary"],
        "missing_gate_diagnostic",
    )
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Keep checkpoints private until owned qualification and terminal validators finish."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261107"], default="20261107")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--sources", type=Path, default=m.ROOT / "results")
    parser.add_argument("--cold-replay", "--replay", dest="replay", type=Path)
    args = parser.parse_args(argv)
    m.progress("start_no_model_load")
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
    template = ExperimentTemplate(
        8906,
        "Finite budget refusal proof",
        "results/" + m.NAME + ".json",
        repo_root=m.ROOT,
        seed=m.SEED,
    )
    started = time.monotonic_ns()
    raw = output.parent / "raw" / m.NAME / "invocations" / str(started)
    m.progress("before_preconditions")
    pre = preconditions(raw, args.sources)
    atomic_json(raw / "preconditions.json", pre)
    atomic_json(raw / "registered_protocol.json", m.protocol())
    atomic_json(
        raw / "invocation.json",
        dict(
            executable=sys.executable,
            arguments=argv if argv is not None else sys.argv[1:],
            cwd=os.getcwd(),
            pid=os.getpid(),
            executed_sources=[reference(m.ROOT / p) for p in m.OWNED],
        ),
    )
    m.progress("after_preconditions_before_proof")
    result, errors, manifest = scheduled("external_precondition"), [], None
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            with template.phase("finite_proof"):
                result = m.qualify()
                require(result["proof_obligations"]["passed"], "owned_proof_failure")
                atomic_json(raw / "calibration_manifest.json", m.calibration_manifest())
                manifest = reference(raw / "calibration_manifest.json")
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            errors.append(str(error))
            result = scheduled(str(error), "failed")
    measured = time.monotonic_ns()
    m.progress("after_proof_before_owned_validation")
    with TemporaryDirectory(prefix="exp8906-validation-") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
    valid = bool(
        validation["coverage"]["complete"]
        and validation["union"]["passed"]
        and all(x["passed"] for x in validation["receipts"])
    )
    m.progress("after_owned_validation")
    require((time.monotonic_ns() - started) / 1e9 < 4200, "owned_validation_deadline")
    verdict = (
        "disqualified"
        if errors or not pre["resources"]["passed"]
        else "blocked"
        if not pre["external_ready"]
        else "null"
        if valid
        else "disqualified"
    )
    ended = time.monotonic_ns()
    phases = [
        dict(phase="preconditions_and_proof", start_s=0, end_s=(measured - started) / 1e9),
        dict(
            phase="owned_validation",
            start_s=(measured - started) / 1e9,
            end_s=(ended - started) / 1e9,
        ),
    ]
    completed = sum(x["status"] == "completed" for x in result["rows"])
    receipt = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_exact_binomial_finite_budget_proof",
        inference_substrate_class="no_model_load" if completed else "blocked_no_run",
        inference_substrate_details=dict(pretrained_loads=0, numerical_training_calls=0),
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=phases,
    )
    atomic_json(raw / "current_work_receipt.json", receipt)
    health = load(reference(HEALTH)) if HEALTH.is_file() else None
    if health:
        require(
            sha256_file(Path(health["log"]["path"])) == health["log"]["sha256"],
            "repository_health_log_drift",
        )
        shutil.copyfile(health["log"]["path"], raw / "repository_health.log")
        health["log"] = reference(raw / "repository_health.log")
    value = dict(
        result,
        experiment_id=8906,
        task_id=m.TASK,
        milestone="2026.11.762",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_finite_budget_methods",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        target_model="seven frozen numerical policies; proof only, no model load or training",
        verifier_is_oracle=True,
        exposure_scope="finite constructed proof states; evaluator-only census",
        independent_generalization_score=0,
        duration_s=receipt["duration_s"],
        phase_spans=phases,
        preconditions_checked=pre,
        random_seed=m.SEED,
        current_work_receipt=receipt,
        source_artifact_hashes=pre["references"],
        code_config_hashes=[
            reference(m.ROOT / p)
            for p in m.OWNED
            + m.TESTS
            + [
                "scripts/experiment_template.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/shared_terminal_methods_8893.py",
                "python/carnot/reporting/sequential_certificate_methods_8880.py",
                "python/carnot/reporting/conditional_risk_methods_8868.py",
                "python/carnot/reporting/split_certificate_methods_8857.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ],
        raw_shard_hashes=[reference(p) for p in raw.rglob("*.json")]
        + validation["raw_references"]
        + ([reference(raw / "repository_health.log")] if health else []),
        intended_count=len(result["rows"]),
        completed_count=completed,
        failed_count=sum(x["status"] == "failed" for x in result["rows"]),
        censored_count=sum(x["status"] == "censored" for x in result["rows"]),
        excluded_count=0,
        independent_count=0,
        gate_check_summary=pre["diagnostics"],
        owned_errors=errors,
        acceptance_gates=dict(
            owned_statement_coverage=100, proof_controls=True, savings_required=False
        ),
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        finite_budget_ready_score=int(completed > 0 and valid and verdict == "null"),
        cited_upstream_artifacts=[dict(producer=SOURCE, sha256=PIN, readiness_field=FIELD)],
        protocol=m.protocol(),
        calibration_manifest=manifest,
        census_reference=pre["census_reference"],
        repository_health=health,
        methodology="Exact finite-binomial monotonicity proof; exhaustive small paths and seeded full-size states; no calibration labels opened.",
        methodology_note="Methods readiness is separate from cost benefit. Fixed and sequential arms share the final test. Proof cases are not independent scientific replicates.",
        flagged_adversarial=False,
    )
    value["field_principles"] = {
        k: "Exact frozen or measured operand; proof qualification does not establish savings or generalization."
        for k in value
    }
    value["field_principles"].update(
        finite_budget_ready_score="All proof controls and owned validation qualify this evaluator, without a savings gate.",
        independent_count="Zero: deterministic proof cases are not independent calibration replicates.",
        calibration_manifest="Forty fresh draw streams frozen without opening labels; downstream evaluation remains unmeasured.",
        family_alpha_ledger="84 early tests at1/16800 and28 shared final tests at1/5600 sum to.01 per replicate.",
        cost_identity_rows="Synthetic accounting controls count request ordinals once across policies, retaining replacement repeats.",
        exhaustive_path_rows="Every eight-draw binary error path and ternary rejection/correct/error path compares all three arms at the frozen small-budget tests.",
        bound_monotonicity_rows="Finite checks test both directions of exact-bound monotonicity and every full-budget integer acceptance threshold.",
        randomized_state_rows="4096 seeded feasible cumulative sufficient states compare sequential decisions and fixed-final inclusion; these are proof controls, not science replicates.",
        proof_obligations="A binomial coupling proves optimistic continuation dominates every attainable future state; qualification requires all retained controls.",
        unsafe_control="The empirical-risk stopping rule deliberately loses a later valid certificate and must be detected.",
        census_reference="Original4096-world input bytes stay bound; this task does not open census truth or measure service risk.",
        repository_health="One full-suite invocation is separate from qualification of newly owned code.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    candidate = raw / "prospective.json"
    atomic_json(candidate, value)
    if not args.private_output:
        m.progress("before_terminal_subprocesses")
        runner = namespace(terminal_runner, "finite8906_terminal")
        runner.m = m
        reports = runner.terminal(candidate, raw)
        value["validation_receipts"] += reports
        value["raw_shard_hashes"] += [
            reference(Path(x[s + "_path"])) for x in reports for s in ["stdout", "stderr"]
        ]
        if not all(x["passed"] for x in reports):
            value.update(
                honest_verdict="complete_disqualified_finite_budget_methods",
                verdict_class="disqualified",
                finite_budget_ready_score=0,
                required_checks_passed=False,
            )
        value.pop("reproducibility_checksum")
        value["reproducibility_checksum"] = canonical_hash(value)
        m.progress("after_terminal_subprocesses")
    require((time.monotonic_ns() - started) / 1e9 < 4800, "hard_deadline")
    publish_primary(output, value, replay)
    m.progress("published_" + value["verdict_class"], completed, len(result["rows"]) - completed)
    return 0

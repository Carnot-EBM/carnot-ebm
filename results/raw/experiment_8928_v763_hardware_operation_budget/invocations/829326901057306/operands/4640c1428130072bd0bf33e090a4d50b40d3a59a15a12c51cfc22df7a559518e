"""REQ-REPORT-8910: authenticate original proof bytes before bounded evaluation."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import finite_budget_evaluation_8910 as m
from carnot.reporting import shared_terminal_evaluation_runner_8896 as previous
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
SOURCE = "experiment_8906_v762_finite_budget_methods"
PIN = "sha256:568e18254f50b1214a3624ee8286ee96ea5e472e5fe18d883104d295c5bf8a9c"
FIELD = "finite_budget_ready_score"
DEADLINE = 1800
HEALTH = Path("/tmp/carnot-8910-work/repository-health.json")


def require(condition: Any, reason: str) -> None:
    """Checksums cannot authorize changed producer assertions or scientific claims."""
    if not condition:
        raise ValueError(reason)


def preconditions(raw: Path, sources: Path) -> Json:
    """Bind every imported byte while distinguishing absence, zero and invalid producers."""
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
    refs, diagnostics, proof = [], [], None
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
            and value["experiment_id"] == 8906
            and value["task_id"] == m.frozen.TASK
            and value["verdict_class"] == "null"
            and value["required_checks_passed"] is True,
            "original_producer",
        )
        require(
            canonical_hash(value["protocol"]) == canonical_hash(m.frozen.protocol())
            and value["proof_receipt_hash"] == canonical_hash(value["rows"])
            and value["proof_obligations"]["passed"],
            "frozen_proof",
        )
        refs.append(reference(path))
        operands = (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        )
        for index, ref in enumerate(operands):
            diagnostic.update(
                path=ref["path"], expected=ref["sha256"], artifact_field="imported_operand"
            )
            diagnostic["observed"] = sha256_file(Path(ref["path"]))
            require(diagnostic["observed"] == ref["sha256"], "original_operand_drift")
            refs.append(ref)
            if index % 50 == 0:
                m.progress("authenticate_original_operands", index, len(operands) - index)
        require(
            load(value["calibration_manifest"]) == m.frozen.calibration_manifest(),
            "fresh_draw_manifest",
        )
        require(
            sum(u["split"] == "population" for u in load(value["census_reference"])["inputs"])
            == 4096,
            "frozen4096_population",
        )
        proof = value["proof_receipt_hash"]
        shutil.copyfile(path, raw / path.name)
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
        proof_receipt_hash=proof,
        resources=dict(passed=shutil.disk_usage(raw).free > 128 * 1024 * 1024),
    )


def validate(raw: Path, private: Path) -> Json:
    """Reuse measured owned unit/child coverage and original private E2E018/021 suites."""
    runner = namespace(previous, "finite8910_validation")
    runner.m = m
    return dict(runner.validate(raw, private))


def measurement(raw: Path, *, replaying: bool = False) -> Json:
    """Seal service decisions before opening evaluator-only full census truth."""
    started, events = time.monotonic(), []

    def seal(stage: str, value: Json) -> Json:
        require(time.monotonic() - started < DEADLINE, "measurement_deadline")
        path = raw / (stage + ".json")
        if replaying:
            require(
                canonical_hash(load(reference(path))) == canonical_hash(value),
                "measurement_drift:" + stage,
            )
        else:
            atomic_json(path, value)
            events.append(dict(stage=stage, monotonic_ns=time.monotonic_ns(), **reference(path)))
            atomic_json(raw / "access_order.json", dict(events=events))
        m.progress("replayed_" + stage if replaying else "sealed_" + stage)
        return value

    seal("protocol", m.protocol())
    source = load(reference(raw / (SOURCE + ".json")))
    operands = {
        stage: next(
            x for x in source["source_artifact_hashes"] if Path(x["path"]).name == stage + ".json"
        )
        for stage in ["population", "scores", "fit"]
    }
    operands["draws"] = source["calibration_manifest"]
    seal("operands", operands)
    cohort, scores, trace = (
        load(operands["population"]),
        load(operands["scores"])["rows"],
        load(operands["draws"]),
    )
    controls_start = time.monotonic()
    controls = m.shared.controls()
    require(all(x["passed"] for x in controls), "owned_control_failure")
    if not replaying:
        atomic_json(
            raw / "pilot_runtime.json",
            dict(projected_total_s=(time.monotonic() - controls_start) * 40 + 1200),
        )
    require(
        load(reference(raw / "pilot_runtime.json"))["projected_total_s"] < 4200, "pilot_runtime"
    )
    data, indices = m.base.registry(cohort, scores)
    units = {u["unit_id"]: u for u in cohort["inputs"]}
    population = [units[u] for u in data[0]]
    truth: Json = {}
    for rep in trace["replicates"]:
        for group in m.base.GROUPS:
            truth.update(
                m.base.old.old.p.labels(
                    [units[data[0][indices[group][i]]] for i in rep["groups"][group]["selection"]]
                )
            )
        m.progress(
            "selection_fetch", rep["replicate"] + 1, len(trace["replicates"]) - rep["replicate"] - 1
        )
    seal("selection_labels", dict(labels=truth))
    selection = seal("selection", m.shared.select(cohort, scores, truth, trace))
    m.progress("before_certification_benchmark", 0, len(trace["replicates"]))
    cert = seal(
        "certificates",
        m.certify(
            cohort,
            scores,
            trace,
            selection,
            lambda arm, rep, group, stage, requested: requested_labels(truth, requested),
        ),
    )
    m.progress("after_certification_before_census", len(cert["rows"]), 0)
    census = seal("census_labels", m.base.old.old.p.labels(population))
    m.progress("before_census_benchmark")
    result = m.evaluate(cohort, scores, census, trace, cert)
    result.update(
        positive_negative_controls=controls,
        historical_training_calls=load(operands["fit"])["numerical_training_calls"],
    )
    seal("evaluation", result)
    m.progress("after_census_benchmark", len(trace["replicates"]), 0)
    return result


def reduction(raw: Path) -> Json:
    """Recompute from sealed operands so rehashed aggregate claims cannot pass."""
    events = load(reference(raw / "access_order.json"))["events"]
    require(
        [x["stage"] for x in events] == m.protocol()["access_order"]
        and all(a["monotonic_ns"] <= b["monotonic_ns"] for a, b in zip(events, events[1:])),
        "access_order_drift",
    )
    for event in events:
        load(event)
    return measurement(raw, replaying=True)


def scheduled(reason: str, status: str = "censored") -> Json:
    """Preserve all scheduled cells when external inputs or owned checks prevent measurement."""
    rows = [
        dict(
            unit_id=f"{rep}/{name}/{p}/{g}",
            cluster_id=f"rep/{rep}",
            replicate=rep,
            protocol=name,
            arm=arm,
            seed=seed,
            group=g,
            metric=None,
            denominator=960,
            status=status,
            missing_reason=reason,
        )
        for rep in range(40)
        for name in m.ARMS
        for p, (arm, seed) in enumerate(m.base.POLICIES)
        for g in m.base.GROUPS
    ]
    return dict(
        rows=rows,
        stopping_rows=[],
        shared_union_costs=[],
        standalone_costs=[],
        paid_label_ledger=[],
        evaluator_label_ledger=[],
        paired_intervals=[],
        census_risk_rows=[],
        pathwise_inclusion_violations=0,
        decision_equality_violations=0,
        shared_union_benefit_score=0,
        standalone_label_benefit_score=0,
        numerical_training_calls=0,
    )


def replay(path: Path) -> Json:
    """Require intact bytes, exact reconstruction, and qualification before granting readiness."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    require(canonical_hash(value) == checksum, "artifact_checksum")
    require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8910, m.TASK, "2026.11.762", "20261107"),
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
        and value["numerical_training_calls"] == 0
        and value["duration_s"] == value["current_work_receipt"]["duration_s"]
        and value["model_invocation_counts"] == value["current_work_receipt"]["invocation_counts"],
        "compute_scope_drift",
    )
    require(value["verdict_class"] != "positive", "oracle_cannot_be_positive")
    counts = {
        s: sum(x["status"] == s for x in value["rows"]) for s in ["completed", "failed", "censored"]
    }
    require(
        value["intended_count"] == len(value["rows"]) == sum(counts.values()) == 3360
        and value["excluded_count"] == 0
        and all(value[k + "_count"] == n for k, n in counts.items())
        and value["independent_count"] == (40 if counts["completed"] == 3360 else 0),
        "count_drift",
    )
    if counts["completed"]:
        raw = Path(value["measurement_manifest"]["path"]).parent
        require(sha256_file(raw / (SOURCE + ".json")) == PIN, "original_producer_drift")
        result = reduction(raw)
        require(
            all(canonical_hash(value[k]) == canonical_hash(v) for k, v in result.items()),
            "reduction_drift",
        )
    ready = value["certificate_rows_ready_score"]
    require(
        type(ready) is int
        and ready
        == int(
            counts["completed"] == 3360
            and value["required_checks_passed"]
            and value["verdict_class"] in ["null", "circular_positive"]
        ),
        "readiness_drift",
    )
    if value["required_checks_passed"]:
        coverage = value["owned_statement_coverage"]
        require(
            coverage["complete"]
            and coverage["percent"] == 100
            and {x["path"] for x in coverage["files"]} == set(m.OWNED)
            and all(not x["missing"] and not x["excluded"] for x in coverage["files"])
            and all(x["passed"] for x in value["validation_receipts"]),
            "qualification_drift",
        )
    require(
        not value["shared_union_benefit_score"]
        or counts["completed"] == 3360
        and value["verdict_class"] in ["circular_positive", "disqualified"],
        "benefit_drift",
    )
    require(
        value["verdict_class"] != "blocked" or value["gate_check_summary"],
        "missing_gate_diagnostic",
    )
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Keep checkpoints private and atomically publish only a validated terminal artifact."""
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
    started = time.monotonic_ns()
    template = ExperimentTemplate(
        8910,
        "Finite budget evaluation",
        "results/" + m.NAME + ".json",
        repo_root=m.ROOT,
        seed=m.SEED,
    )
    raw = output.parent / "raw" / m.NAME / "invocations" / str(started)
    m.progress("before_preconditions")
    pre = preconditions(raw, args.sources)
    atomic_json(raw / "preconditions.json", pre)
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
    errors: list[str] = []
    result = scheduled("external_precondition")
    m.progress("after_preconditions_before_measurement")
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            with template.phase("measurement"):
                result = measurement(raw)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            errors.append(str(error))
            result = scheduled(str(error), "censored" if "deadline" in str(error) else "failed")
    measured = time.monotonic_ns()
    m.progress("before_owned_validation")
    with TemporaryDirectory(prefix="exp8910-validation-") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
    valid = bool(
        validation["coverage"]["complete"]
        and validation["union"]["passed"]
        and all(x["passed"] for x in validation["receipts"])
    )
    m.progress("after_owned_validation")
    require((time.monotonic_ns() - started) / 1e9 < 4200, "validation_deadline")
    completed = sum(x["status"] == "completed" for x in result["rows"])
    verdict = (
        "disqualified"
        if errors or not pre["resources"]["passed"] or (not valid and not args.private_output)
        else "blocked"
        if not pre["external_ready"]
        else "disqualified"
        if not valid
        else "circular_positive"
        if result["shared_union_benefit_score"]
        else "null"
    )
    ended = time.monotonic_ns()
    phases = [
        dict(phase="preconditions_and_measurement", start_s=0, end_s=(measured - started) / 1e9),
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
        inference_substrate="cpu_frozen_heads_finite_budget_evaluation",
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
            sha256_file(Path(health["log"]["path"])) == health["log"]["sha256"], "health_log_drift"
        )
        shutil.copyfile(health["log"]["path"], raw / "repository_health.log")
        health["log"] = reference(raw / "repository_health.log")
    value = dict(
        result,
        experiment_id=8910,
        task_id=m.TASK,
        milestone="2026.11.762",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_finite_budget_evaluation",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        target_model="seven frozen numerical policies; no current model load or training",
        verifier_is_oracle=True,
        exposure_scope="constructed stationary4096-world population; evaluator-only census",
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
                "python/carnot/reporting/finite_budget_methods_8906.py",
                "python/carnot/reporting/shared_terminal_methods_8893.py",
                "python/carnot/reporting/sequential_label_cost_8885.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ],
        raw_shard_hashes=[reference(p) for p in raw.rglob("*.json")]
        + validation["raw_references"]
        + ([reference(raw / "repository_health.log")] if health else []),
        intended_count=3360,
        completed_count=completed,
        failed_count=sum(x["status"] == "failed" for x in result["rows"]),
        censored_count=sum(x["status"] == "censored" for x in result["rows"]),
        excluded_count=0,
        independent_count=40 if completed == 3360 else 0,
        gate_check_summary=pre["diagnostics"],
        owned_errors=errors,
        acceptance_gates=dict(
            owned_statement_coverage=100,
            primary_contrast="shared_union/label_savings",
            shared_union_lower99_gt=0,
            risk_ceiling=0.05,
            pathwise_inclusion_violations=0,
            decision_equality_violations=0,
            benefit_required_for_measurement=False,
        ),
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        certificate_rows_ready_score=int(
            completed == 3360 and valid and verdict in ["null", "circular_positive"]
        ),
        cited_upstream_artifacts=[dict(producer=SOURCE, sha256=PIN, readiness_field=FIELD)],
        proof_receipt_hash=pre["proof_receipt_hash"],
        protocol=m.protocol(),
        repository_health=health,
        measurement_manifest=reference(raw / "evaluation.json") if completed else None,
        methodology="Forty fresh paired calibration replicates on the original4096 worlds and seven frozen policies. Selection240 is independent of the120/240/480/960 certification looks. Reuse exact registered tests and conservative23-contrast bounded Hoeffding paired intervals. The sole primary contrast is all-policy union requests saved by refusal against early acceptance. Repeats and all policy input evaluations count. Comparator labels and post-decision census work are evaluator costs.",
        methodology_note="Logical equality and inclusion are pathwise controls. The unchanged finite test family supplies per-replicate error<=.01 under stationary iid sampling; forty replicates do not establish rare-event guarantees. Census safety and label benefit are separate. Oracle truth cannot establish independent natural-language generalization. Deterministic oracle computation reuses labels already requested during selection or certification. Every ordinal fetch, including cache hits, is paid. Full wall time is descriptive, with no speed benefit claim.",
        flagged_adversarial=False,
    )
    value["field_principles"] = {
        k: "Frozen or measured operand bound to original bytes; readiness does not imply scientific benefit or generalization."
        for k in value
    }
    value["field_principles"].update(
        shared_union_benefit_score="Only the primary paired union saving lower99>0 with no decision or inclusion loss and safe census risks establishes benefit.",
        standalone_label_benefit_score="A separate learned-policy cost diagnostic cannot establish shared union benefit.",
        paid_label_ledger="Deployed refusal arm only: selection and actual fetched prefixes retain ordinal repeats and distinct world counts.",
        evaluator_label_ledger="Comparator certification fetches and post-decision census are evaluator work. Each protocol is charged selection240, while actual comparator selection fetches are zero because common selection labels are reused.",
        evaluator_census_labels="Full population labels are evaluator-only and opened after every service stop.",
        standalone_costs="Each policy's selection and stopped prefix are reported even when it never certifies.",
        independent_count="Forty paired calibration streams; policies, groups, mixtures and repeated worlds cannot inflate n.",
        repository_health="One separate full-suite invocation; unrelated failures do not redefine owned qualification.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    candidate = raw / "prospective.json"
    atomic_json(candidate, value)
    if not args.private_output:
        m.progress("before_terminal_subprocesses")
        runner = namespace(terminal_runner, "finite8910_terminal")
        runner.m = m
        reports = runner.terminal(candidate, raw)
        value["validation_receipts"] += reports
        value["raw_shard_hashes"] += [
            reference(Path(x[s + "_path"])) for x in reports for s in ["stdout", "stderr"]
        ]
        if not all(x["passed"] for x in reports):
            value.update(
                honest_verdict="complete_disqualified_finite_budget_evaluation",
                verdict_class="disqualified",
                certificate_rows_ready_score=0,
                required_checks_passed=False,
            )
        m.progress("after_terminal_subprocesses")
    final_end = time.monotonic_ns()
    phases.append(
        dict(
            phase="terminal_validation_and_preparation",
            start_s=(ended - started) / 1e9,
            end_s=(final_end - started) / 1e9,
        )
    )
    receipt = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        inference_substrate_details=receipt["inference_substrate_details"],
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=final_end,
        phase_spans=phases,
    )
    atomic_json(raw / "terminal_current_work_receipt.json", receipt)
    value.update(
        current_work_receipt=receipt,
        duration_s=receipt["duration_s"],
        phase_spans=phases,
        full_wall_costs=dict(
            through_terminal_validation_s=receipt["duration_s"],
            publication_reconstruction_included=False,
            speed_benefit_claim=False,
        ),
    )
    value["raw_shard_hashes"].append(reference(raw / "terminal_current_work_receipt.json"))
    value["field_principles"]["full_wall_costs"] = (
        "Descriptive actual wall time includes measurement, owned qualification and terminal scanners; final publication reconstruction is outside this receipt. No speed benefit follows."
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    require((time.monotonic_ns() - started) / 1e9 < 4800, "hard_deadline")
    publish_primary(output, value, replay)
    m.progress("published_" + value["verdict_class"], completed, 3360 - completed)
    return 0


def requested_labels(cache: Json, units: list[Json]) -> Json:
    """Reuse deterministic oracle computation while exposing only requested identifiers."""
    missing = {u["unit_id"]: u for u in units if u["unit_id"] not in cache}
    if missing:
        cache.update(m.base.old.old.p.labels(list(missing.values())))
    return {u["unit_id"]: cache[u["unit_id"]] for u in units}

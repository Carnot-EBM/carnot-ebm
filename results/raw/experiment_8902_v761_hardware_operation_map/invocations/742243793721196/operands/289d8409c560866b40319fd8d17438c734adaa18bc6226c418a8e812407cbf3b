"""REQ-REPORT-8896: authenticate frozen operands and publish a reconstructed measurement."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import shared_terminal_evaluation_8896 as m
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
SOURCE = "experiment_8893_v761_shared_terminal_methods"
UPSTREAM = {
    SOURCE: (
        "sha256:f1320c074d34bfde274cbb85e25410fd0da50a65d947acb5d4cb32e8484c5ead",
        "shared_terminal_ready_score",
    ),
    "experiment_8885_v760_sequential_label_cost": previous.UPSTREAM[
        "experiment_8885_v760_sequential_label_cost"
    ],
}
DEADLINE = 1800
HEALTH = Path("/tmp/carnot-8896-work/repository-health.json")


def require(condition: Any, reason: str) -> None:
    """Reject repaired checksums when their scientific or provenance meaning has changed."""
    if not condition:
        raise ValueError(reason)


def preconditions(raw: Path, sources: Path) -> Json:
    """Bind original claims and exact operands; known metadata rollover never changes a pin."""
    # The inherited checker accepts null producers; the exact original pin below
    # also authenticates the successful oracle producer without rewriting it.
    checker = namespace(previous, "evaluation8896_preconditions")
    checker.UPSTREAM = {k: v for k, v in UPSTREAM.items() if k != SOURCE}
    pre = dict(checker.preconditions(raw, sources))
    refs = {x["path"]: x for x in pre["references"]}
    path = sources / (SOURCE + ".json")
    diagnostic: Json = dict(
        check="missing_file",
        upstream=SOURCE,
        artifact_field=UPSTREAM[SOURCE][1],
        expected=1,
        observed=None,
        path=str(path),
        hash=None,
    )
    try:
        diagnostic["hash"] = sha256_file(path)
        value = json.loads(path.read_bytes())
        diagnostic.update(check="missing_field", observed=value.get(UPSTREAM[SOURCE][1]))
        score = value[UPSTREAM[SOURCE][1]]
        diagnostic["check"] = "observed_zero"
        require(type(score) is int and score == 1, "readiness_zero")
        diagnostic.update(
            check="rejected_producer",
            artifact_field="sha256",
            expected=UPSTREAM[SOURCE][0],
            observed=sha256_file(path),
        )
        require(
            sha256_file(path) == UPSTREAM[SOURCE][0]
            and value["experiment_id"] == 8893
            and value["verdict_class"] in ["null", "circular_positive"],
            "producer_identity",
        )
        checked: dict[Path, str] = {}
        operands = [reference(path)] + [
            x
            for k in ["source_artifact_hashes", "code_config_hashes", "raw_shard_hashes"]
            for x in value[k]
        ]
        for i, ref in enumerate(operands):
            resolved = previous.PRESERVED.get(Path(ref["path"]), Path(ref["path"]))
            diagnostic.update(path=str(resolved), expected=ref["sha256"])
            if resolved not in checked:
                checked[resolved] = sha256_file(resolved)
            diagnostic["observed"] = checked[resolved]
            require(checked[resolved] == ref["sha256"], "original_operand_drift")
            refs[str(resolved)] = dict(path=str(resolved), sha256=checked[resolved])
            if i % 500 == 0:
                m.progress("authenticate_original_operands", i, len(operands) - i)
        shutil.copyfile(path, raw / path.name)
    except (OSError, ValueError, KeyError, TypeError) as error:
        pre["diagnostics"].insert(
            0,
            dict(
                diagnostic,
                error=str(error),
                check="missing_file"
                if isinstance(error, OSError)
                else "missing_field"
                if isinstance(error, KeyError)
                else diagnostic["check"],
            ),
        )
    pre.update(references=list(refs.values()), external_ready=not pre["diagnostics"])
    for diagnostic in pre["diagnostics"]:
        diagnostic.setdefault("hash", diagnostic.get("sha256"))
    return pre


def validate(raw: Path, private: Path) -> Json:
    """Reuse real unit and child statement coverage and the private E2E018/021 suites."""
    checker = namespace(previous, "evaluation8896_validation")
    checker.m = m
    return dict(checker.validate(raw, private))


def measurement(raw: Path, *, replaying: bool = False) -> Json:
    """Reexecute frozen paired draws while sealing each decision before opening census truth."""
    started, events = time.monotonic(), []

    def seal(stage: str, value: Json) -> Json:
        require(time.monotonic() - started <= DEADLINE, "measurement_deadline")
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
        m.progress("sealed_" + stage)
        return value

    seal("protocol", m.protocol())
    source = load(reference(raw / (SOURCE + ".json")))
    retained = {}
    for stage in [
        "population",
        "draws",
        "fit",
        "scores",
        "selection_labels",
        "certification_labels",
    ]:
        ref = next(x for x in source["raw_shard_hashes"] if Path(x["path"]).name == stage + ".json")
        path = raw / "retained_inputs" / (stage + ".json")
        if not replaying:
            path.parent.mkdir(exist_ok=True)
            shutil.copyfile(ref["path"], path)
        require(sha256_file(path) == ref["sha256"], "original_operand_drift")
        retained[stage] = reference(path)
    seal("frozen_operands", retained)
    cohort, trace = load(retained["population"]), load(retained["draws"])
    scores = load(retained["scores"])["rows"]
    controls = m.frozen.controls()
    require(all(x["passed"] for x in controls), "owned_control_failure")
    selection = seal(
        "selection",
        m.frozen.select(cohort, scores, load(retained["selection_labels"])["labels"], trace),
    )
    m.progress("before_certification_benchmark", 0, 2240)
    cert = m.frozen.certify(
        cohort, scores, load(retained["certification_labels"])["labels"], trace, selection
    )
    require(
        canonical_hash(cert["rows"]) == canonical_hash(source["stopping_rows"]),
        "frozen_certificate_drift",
    )
    seal("certificates", cert)
    m.progress("after_certification_before_census", len(cert["rows"]), 0)
    population = [u for u in cohort["inputs"] if u["split"] == "population"]
    truth = seal("census_labels", m.frozen.base.old.old.p.labels(population))
    m.progress("before_census_benchmark", 0, len(trace["replicates"]))
    result = m.evaluate(cohort, scores, truth, trace, cert)
    result.update(
        historical_training_calls=source["historical_training_calls"],
        positive_negative_controls=controls,
    )
    seal("evaluation", result)
    m.progress("after_census_benchmark", len(trace["replicates"]), 0)
    return result


def reduction(raw: Path) -> Json:
    """Reconstruct all science after checking sealed phase order and original bytes."""
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
    """Preserve all scheduled cells when external absence or owned failure prevents measurement."""
    return dict(
        previous.scheduled(reason, status),
        stopping_rows=[],
        bound_rows=[],
        inclusion_rows=[],
        coverage_rows=[],
        paid_label_ledger={},
        theoretical_family_error=m.frozen.protocol()["alpha_allocation"],
        standalone_label_benefit_score=0,
        shared_union_benefit_score=0,
    )


def replay(path: Path) -> Json:
    """Original producer pins and independent reduction prevent rehashed false claims."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    require(canonical_hash(value) == checksum, "artifact_checksum")
    require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8896, m.TASK, "2026.11.761", "20261107"),
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
        and value["duration_s"] == value["current_work_receipt"]["duration_s"]
        and value["model_invocation_counts"] == value["current_work_receipt"]["invocation_counts"],
        "compute_scope_drift",
    )
    counts = {
        s: sum(x["status"] == s for x in value["rows"]) for s in ["completed", "failed", "censored"]
    }
    require(
        len(value["rows"]) == value["intended_count"] == sum(counts.values()) == 2240
        and value["excluded_count"] == 0
        and all(value[s + "_count"] == n for s, n in counts.items())
        and value["independent_count"] == (40 if counts["completed"] == 2240 else 0),
        "count_drift",
    )
    if counts["completed"]:
        raw = Path(value["measurement_manifest"]["path"]).parent
        require(
            all(sha256_file(raw / (name + ".json")) == pin for name, (pin, _) in UPSTREAM.items()),
            "original_producer_drift",
        )
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
            counts["completed"] == 2240
            and value["required_checks_passed"]
            and value["verdict_class"] in ["null", "circular_positive"]
        ),
        "readiness_drift",
    )
    if ready:
        coverage = value["owned_statement_coverage"]
        require(
            coverage["complete"]
            and coverage["percent"] == 100
            and {x["path"] for x in coverage["files"]} == set(m.OWNED)
            and all(not x["missing"] and not x["excluded"] for x in coverage["files"])
            and all(x["passed"] for x in value["validation_receipts"]),
            "qualification_drift",
        )
    benefit = value["standalone_label_benefit_score"] or value["shared_union_benefit_score"]
    require(
        not benefit
        or counts["completed"] == 2240
        and value["verdict_class"] in ["circular_positive", "disqualified"],
        "benefit_drift",
    )
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Keep checkpoints private until validation and publish only one checked terminal artifact."""
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
        8896,
        "Shared terminal label-cost evaluation",
        "results/" + m.NAME + ".json",
        repo_root=m.ROOT,
        seed=m.SEED,
    )
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    started = time.monotonic_ns()
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
    errors = []
    result = scheduled(pre["diagnostics"][0]["check"] if pre["diagnostics"] else "owned_failure")
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
    with TemporaryDirectory(prefix="exp8896-validation-") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
    valid = bool(
        validation["coverage"]["complete"]
        and validation["union"]["passed"]
        and all(x["passed"] for x in validation["receipts"])
    )
    m.progress("after_owned_validation")
    completed = sum(x["status"] == "completed" for x in result["rows"])
    verdict = (
        "disqualified"
        if errors or not pre["resources"]["passed"]
        else "blocked"
        if not pre["external_ready"]
        else "disqualified"
        if not valid
        else "circular_positive"
        if result["standalone_label_benefit_score"] or result["shared_union_benefit_score"]
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
        inference_substrate="cpu_retained_heads_shared_terminal_evaluation",
        inference_substrate_class="no_model_load" if completed else "blocked_no_run",
        inference_substrate_details=dict(pretrained_loads=0, fresh_training_calls=0),
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=phases,
    )
    atomic_json(raw / "current_work_receipt.json", receipt)
    health = None
    if HEALTH.is_file():
        health = load(reference(HEALTH))
        source = Path(health["log"]["path"])
        require(sha256_file(source) == health["log"]["sha256"], "repository_health_log_drift")
        target = raw / "repository_health.log"
        shutil.copyfile(source, target)
        health["log"] = reference(target)
        atomic_json(raw / "repository_health.json", health)
    value = dict(
        result,
        experiment_id=8896,
        task_id=m.TASK,
        milestone="2026.11.761",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_shared_terminal_evaluation",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        target_model="seven frozen numerical policies: two baselines and five retained heads; no pretrained model",
        verifier_is_oracle=True,
        exposure_scope="constructed stationary finite population only",
        independent_generalization_score=0,
        duration_s=receipt["duration_s"],
        phase_spans=phases,
        preconditions_checked=pre,
        random_seed=m.SEED,
        current_work_receipt=receipt,
        source_artifact_hashes=pre["references"],
        repository_health=health,
        code_config_hashes=[
            reference(m.ROOT / p)
            for p in m.OWNED
            + m.TESTS
            + [
                "scripts/experiment_template.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/shared_terminal_methods_8893.py",
                "python/carnot/reporting/shared_terminal_runner_8893.py",
                "python/carnot/reporting/sequential_label_cost_8885.py",
                "python/carnot/reporting/sequential_label_cost_runner_8885.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ],
        raw_shard_hashes=[reference(p) for p in raw.rglob("*.json")]
        + validation["raw_references"]
        + ([reference(raw / "repository_health.log")] if health else []),
        intended_count=2240,
        completed_count=completed,
        failed_count=sum(x["status"] == "failed" for x in result["rows"]),
        censored_count=sum(x["status"] == "censored" for x in result["rows"]),
        excluded_count=0,
        independent_count=40 if completed == 2240 else 0,
        gate_check_summary=pre["diagnostics"],
        owned_errors=errors,
        acceptance_gates=dict(
            owned_statement_coverage=100,
            risk_ceiling=0.05,
            pathwise_inclusion_violations=0,
            standalone_savings_lower_gt=0,
            shared_union_savings_lower_gt=0,
            benefit_required_for_measurement=False,
        ),
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        certificate_rows_ready_score=int(
            completed == 2240 and valid and verdict in ["null", "circular_positive"]
        ),
        cited_upstream_artifacts=[
            dict(producer=n, sha256=p, readiness_field=f) for n, (p, f) in UPSTREAM.items()
        ],
        protocol=m.protocol(),
        measurement_manifest=reference(raw / "evaluation.json") if completed else None,
        methodology="Reexecute the frozen4096-world population and forty paired independent draw replicates. Seven policies, four groups and five mixtures stay fixed. Exact upper bounds share one final event. Selection labels and all candidate draws including repeats are paid. Five learned seeds average within each replicate. Bonferroni bounded Hoeffding intervals use99-percent simultaneous confidence over23 frozen contrasts. Standalone and shared union costs are separate.",
        methodology_note="Constructed oracle truth is circular evidence. Per-replicate theoretical family error<=.01 assumes stationary iid group draws independent of selection; forty draws do not prove it. Census risk and logical coverage inclusion are separate checks. A measured null is terminal.",
        flagged_adversarial=False,
        field_principles={},
    )
    value["field_principles"] = {
        k: "Frozen or measured operand bound to original bytes; completion and readiness do not imply benefit or generalization."
        for k in value
    }
    value["field_principles"].update(
        standalone_label_benefit_score="Learned seed costs average inside each replicate; savings lower99>0 also requires inclusion and every served group risk<=.05.",
        shared_union_benefit_score="A separate interval uses the longest unfinished policy prefix per group. Standalone savings never transfer to shared work.",
        certificate_rows_ready_score="Complete authenticated measurement and owned qualification remain useful when either benefit score is zero.",
        theoretical_family_error="Distinct early and shared final events sum to .01 per replicate; empirical violations do not establish this guarantee.",
        bound_rows="Every cumulative look retains n, errors, alpha, bound and visited status; unvisited looks remain audit operands.",
        inclusion_rows="Every paired cell checks common final alpha, prefix, cutoff, certification and served-set inclusion.",
        paid_label_ledger="All240 selection labels per group and every scanned candidate including repeats are paid; census and historical training are separate.",
        repository_health="The full repository test outcome is retained separately from scoped owned validation.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    candidate = raw / "prospective.json"
    atomic_json(candidate, value)
    if not args.private_output:
        m.progress("before_terminal_checks")
        runner = namespace(terminal_runner, "evaluation8896_terminal")
        runner.m = m
        reports = runner.terminal(candidate, raw)
        value["validation_receipts"] += reports
        value["raw_shard_hashes"] += [
            reference(Path(x[s + "_path"])) for x in reports for s in ["stdout", "stderr"]
        ]
        if not all(x["passed"] for x in reports):
            value.update(
                honest_verdict="complete_disqualified_shared_terminal_evaluation",
                verdict_class="disqualified",
                certificate_rows_ready_score=0,
                required_checks_passed=False,
            )
        value.pop("reproducibility_checksum")
        value["reproducibility_checksum"] = canonical_hash(value)
        m.progress("after_terminal_checks")
    publish_primary(output, value, replay)
    m.progress("published_" + value["verdict_class"], completed, 2240 - completed)
    return 0

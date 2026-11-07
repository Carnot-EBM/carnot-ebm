"""REQ-REPORT-8870: authenticate sealed methods and publish checked mixture evaluation."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import conditional_risk_evaluation_8870 as m
from carnot.reporting import conditional_risk_runner_8868 as previous
from carnot.reporting import selective_methods_runner_8829 as inherited
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_runner_8816 import stream, unqualified
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
PIN = "sha256:af45370397fc99a00514d511a3c81d29790a879e3bb6bafcf55701e0088d5207"
SOURCE = "experiment_8868_v759_conditional_risk_methods"
HEALTH = Path("/tmp/carnot-8870-repository-health/receipt.json")
DEADLINE = 1800


def preconditions(raw: Path, sources: Path = m.ROOT / "results") -> Json:
    """The existing authority checker authenticates original producers and all byte operands."""
    checker = namespace(previous, "conditional8870_preconditions")
    checker.UPSTREAM = {
        SOURCE: (PIN, "conditional_methods_ready_score"),
        "experiment_8859_v758_split_certificate_evaluation": previous.UPSTREAM[
            "experiment_8859_v758_split_certificate_evaluation"
        ],
    }
    return dict(checker.preconditions(raw, sources))


def validate(raw: Path, private: Path) -> Json:
    """Reuse actual owned unit/child coverage, private E2E018/021 and immutable logs."""
    checker = namespace(inherited, "conditional8870_validation")
    checker.m = m
    return dict(checker.validate(raw, private))


def measurement(raw: Path, *, replaying: bool = False) -> Json:
    """Seal each selection and certificate before opening its later label operands."""
    started, events = time.monotonic(), []
    source = json.loads((raw / (SOURCE + ".json")).read_bytes())
    origin = Path(source["population_manifest"]["path"]).parent

    def seal(stage: str, value: Json) -> Json:
        if time.monotonic() - started > DEADLINE:
            raise ValueError("measurement_deadline")
        if not replaying:
            atomic_json(raw / (stage + ".json"), value)
            events.append(
                dict(
                    stage=stage,
                    monotonic_ns=time.monotonic_ns(),
                    **reference(raw / (stage + ".json")),
                )
            )
            atomic_json(raw / "access_order.json", dict(events=events))
        elif canonical_hash(load(reference(raw / (stage + ".json")))) != canonical_hash(value):
            raise ValueError("measurement_drift:" + stage)
        m.progress("replayed_" + stage if replaying else "sealed_" + stage, len(events), 0)
        return value

    def get(stage: str) -> Json:
        return load(
            next(
                ref
                for ref in source["raw_shard_hashes"]
                if Path(ref["path"]) == origin / (stage + ".json")
            )
        )

    seal("protocol", m.protocol())
    cohort, fitted, trace = get("population"), get("fit"), get("draws")
    if trace != m.base.draws(1024) or len(trace["replicates"]) != 40:
        raise ValueError("sealed_draw_drift")
    for u in cohort["inputs"]:
        if m.base.group(u["input"]["query"]) != u["group"]:
            raise ValueError("group_assignment_drift")
    seal(
        "inputs",
        dict(population=source["population_manifest"], fit=source["fit_manifest"], draws=trace),
    )
    m.progress("before_prediction_no_model_load", 0, 7)
    m.base.old.old.audit_fit(cohort, fitted)
    scores = m.base.old.old.predict(cohort, fitted)
    if canonical_hash(scores) != canonical_hash(get("scores")["rows"]):
        raise ValueError("prediction_drift")
    seal("scores", dict(rows=scores))
    m.progress("after_prediction_no_model_load", 7, 0)
    selected_truth = seal("selection_labels", get("selection_labels"))["labels"]
    selection = m.base.select(cohort, scores, selected_truth, trace)
    if selection != get("selection"):
        raise ValueError("selection_drift")
    seal("selection", selection)
    certification_truth = seal("certification_labels", get("certification_labels"))["labels"]
    certificates = m.base.certify(cohort, scores, certification_truth, trace, selection)
    if certificates != get("certificates"):
        raise ValueError("certificate_drift")
    seal("certificates", certificates)
    truth = m.base.old.old.p.labels([u for u in cohort["inputs"] if u["split"] == "population"])
    if truth != get("census_labels")["labels"] or any(
        truth[k] != v for labels in [selected_truth, certification_truth] for k, v in labels.items()
    ):
        raise ValueError("label_drift")
    seal("census_labels", dict(labels=truth))
    m.progress("before_census_benchmark", 0, 4096)
    result = m.evaluate(cohort, scores, truth, certificates)
    m.progress("after_census_benchmark", 4096, 0)
    m.progress("before_corruption_benchmark", 0, 4096)
    corruption = get("corruption")
    corrupted = m.base.corrupt(cohort)
    stress_scores = m.base.old.old.predict(corrupted, fitted)
    stress_truth = m.base.old.old.p.labels(
        [u for u in corrupted["inputs"] if u["split"] == "population"]
    )
    if (
        canonical_hash(stress_scores) != canonical_hash(corruption["scores"])
        or stress_truth != corruption["labels"]
    ):
        raise ValueError("corruption_drift")
    stress = m.evaluate(corrupted, stress_scores, stress_truth, certificates, corrupted=True)
    m.progress("after_corruption_benchmark", 4096, 0)
    summary = m.summarize(result["rows"], result["group_count_rows"])
    return seal(
        "measurement",
        dict(
            summary,
            rows=result["rows"] + stress["rows"],
            corruption_rows=stress["rows"],
            corruption_group_rows=stress["group_count_rows"],
            numerical_training_calls=0,
            historical_training_calls=fitted["numerical_training_calls"],
        ),
    )


def measure(raw: Path, pre: Json) -> Json:
    """External readiness is checked before starting any scientific work."""
    if not pre["external_ready"]:
        raise ValueError("external_not_ready")
    return measurement(raw)


def reduction(raw: Path) -> Json:
    """Cold replay authenticates event order and independently reconstructs all counts."""
    events = load(reference(raw / "access_order.json"))["events"]
    if [e["stage"] for e in events] != [
        "protocol",
        "inputs",
        "scores",
        "selection_labels",
        "selection",
        "certification_labels",
        "certificates",
        "census_labels",
        "measurement",
    ] or any(a["monotonic_ns"] > b["monotonic_ns"] for a, b in zip(events, events[1:])):
        raise ValueError("access_order_drift")
    for event in events:
        if event["stage"] != "measurement":
            load(event)
    return measurement(raw, replaying=True)


def scheduled(reason: str, status: str = "censored") -> Json:
    """Missing external inputs and failed owned work retain the entire scheduled roster."""
    value = previous.scheduled(reason)
    value["rows"] = [dict(r, status=status) for r in value["rows"]]
    value.update(
        {
            k: []
            for k in [
                "certificate_event_rows",
                "group_count_rows",
                "census_risk_rows",
                "mixture_rows",
                "corruption_rows",
                "paired_replicate_rows",
                "simultaneous_intervals",
            ]
        }
    )
    value.update(conditional_utility_score=0, robustness_advantage_score=0, comparison="unmeasured")
    return value


def replay(path: Path) -> Json:
    """Identity, original bytes and rederived scientific results govern safe publication."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    if canonical_hash(value) != checksum:
        raise ValueError("artifact_checksum")
    if (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"]) != (
        8870,
        m.TASK,
        "2026.11.759",
        "20261106",
    ):
        raise ValueError("producer_identity")
    if value["protocol_hash"] != canonical_hash(m.protocol()) or value["protocol"] != json.loads(
        json.dumps(m.protocol())
    ):
        raise ValueError("protocol_drift")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("operand_hash")
    if validate_current_work_receipt(value["current_work_receipt"]):
        raise ValueError("current_receipt")
    completed = sum(r["status"] == "completed" for r in value["rows"])
    if (
        len(value["rows"]) != 3360
        or value["completed_count"] != completed
        or completed + value["failed_count"] + value["censored_count"] != 3360
        or value["excluded_count"] != 0
        or value["independent_count"] != (40 if completed == 3360 else 0)
    ):
        raise ValueError("count_drift")
    if completed:
        raw = Path(value["measurement_manifest"]["path"]).parent
        if sha256_file(raw / (SOURCE + ".json")) != PIN:
            raise ValueError("original_producer_drift")
        result = reduction(raw)
        if any(canonical_hash(value[k]) != canonical_hash(v) for k, v in result.items()):
            raise ValueError("reduction_drift")
    elif value["conditional_utility_score"] or value["robustness_advantage_score"]:
        raise ValueError("unmeasured_benefit")
    ready = value["conditional_rows_ready_score"]
    if ready not in [0, 1] or (
        ready
        and (
            completed != 3360
            or not value["required_checks_passed"]
            or value["verdict_class"] not in ["null", "circular_positive"]
        )
    ):
        raise ValueError("readiness_drift")
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Execute research without activating a milestone or loading pretrained weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261106"], default="20261106")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--sources", type=Path, default=m.ROOT / "results")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    m.progress("start_no_model_load")
    if args.cold_replay:
        try:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError, TypeError) as error:
            print("replay_rejected: " + str(error), flush=True)
            return 1
    output = (args.private_output or args.output).absolute()
    if args.private_output and output.resolve().is_relative_to(m.ROOT):
        parser.error("private outputs must remain outside checkout")
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    started = time.monotonic_ns()
    pre = preconditions(raw, args.sources)
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
    atomic_json(raw / "preconditions.json", pre)
    result = scheduled(pre["diagnostics"][0]["check"] if pre["diagnostics"] else "owned_failure")
    errors = []
    m.progress("after_preconditions_before_measurement")
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            result = measure(raw, pre)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            errors.append(str(error))
            result = scheduled(str(error), "censored" if "deadline" in str(error) else "failed")
    measured = time.monotonic_ns()
    m.progress("before_owned_validation")
    with TemporaryDirectory(prefix="exp8870-validation-") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
    valid = bool(
        validation["coverage"]["complete"]
        and validation["union"]["passed"]
        and all(r["passed"] for r in validation["receipts"])
    )
    m.progress("after_owned_validation")
    completed = sum(r["status"] == "completed" for r in result["rows"])
    verdict = (
        "disqualified"
        if errors or not pre["resources"]["passed"]
        else "blocked"
        if not pre["external_ready"]
        else "disqualified"
        if not valid
        else "circular_positive"
        if result["robustness_advantage_score"]
        else "null"
    )
    ended = time.monotonic_ns()
    phases = [
        dict(phase="preconditions_and_measurement", start_s=0.0, end_s=(measured - started) / 1e9),
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
        inference_substrate="cpu_retained_energy_head_predictions_and_exact_conditional_certificates",
        inference_substrate_class="no_model_load" if completed else "blocked_no_run",
        execution_venue="host",
        inference_substrate_details=dict(
            scope="retained numerical heads; zero fresh training or pretrained loads"
        ),
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=phases,
    )
    atomic_json(raw / "current_work_receipt.json", receipt)
    health = None
    if HEALTH.is_file():
        health = load(reference(HEALTH))
        for suffix in ["stdout", "stderr"]:
            target = raw / ("repository_health." + suffix)
            shutil.copyfile(health[suffix + "_path"], target)
            health[suffix + "_path"] = str(target)
        atomic_json(raw / "repository_health.json", health)
    value = dict(
        result,
        experiment_id=8870,
        task_id=m.TASK,
        milestone="2026.11.759",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_conditional_risk_evaluation",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        verifier_is_oracle=True,
        exposure_scope="constructed signed relational worlds; exact oracle labels; no natural-language or oracle-distinct generalization",
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
            ]
        ],
        raw_shard_hashes=[reference(p) for p in raw.glob("*.json")]
        + validation["raw_references"]
        + [reference(p) for p in raw.glob("repository_health.std*")],
        repository_health=health,
        intended_count=3360,
        completed_count=completed,
        failed_count=sum(r["status"] == "failed" for r in result["rows"]),
        censored_count=sum(r["status"] == "censored" for r in result["rows"]),
        excluded_count=0,
        independent_count=40 if completed == 3360 else 0,
        gate_check_summary=pre["diagnostics"],
        owned_errors=errors,
        acceptance_gates=dict(
            coverage_lower_99=0.30,
            certified_group_risk_max=0.05,
            robustness_paired_lower_99_gt=0,
            pooled_utility_requires_parity=True,
        ),
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        conditional_rows_ready_score=int(
            valid and completed == 3360 and verdict in ["null", "circular_positive"]
        ),
        protocol=m.protocol(),
        protocol_hash=canonical_hash(m.protocol()),
        measurement_manifest=reference(raw / "measurement.json") if completed else None,
        methodology="Reexecute all40 sealed paired calibration replicates on authenticated fixed scores and groups; selection before certification labels, certificates before census labels; compute exact accepted/error masses for five fixed mixtures. Bonferroni bounded Hoeffding intervals over36 fixed quantities. Seeds average within replicate. Separate corruption changes within-group evidence/outcome distribution and has no guarantee.",
        methodology_note="Conditional measurement, utility and comparative benefit are separate. Retained heads incur no fresh training. Oracle labels establish no independent generalization.",
        flagged_adversarial=False,
        field_principles={},
    )
    value["field_principles"] = {
        k: "Frozen protocol or measured operand bound to original bytes; administrative custody is not scientific evidence."
        for k in value
    }
    value["field_principles"].update(
        certificate_event_rows="Every replicate/protocol/policy/group certificate, bound and refusal remains visible.",
        group_count_rows="Exact served and selected-rule accepted/error counts retain refused groups and their population denominators.",
        census_risk_rows="Finite-census served risk divides errors by accepted worlds; no service retains null risk.",
        mixture_rows="Coverage=sum(weight*accepted/population); risk=sum(weight*errors/population)/coverage, never an unweighted risk average.",
        corruption_rows="Opposite signed evidence changes within-group evidence/outcome distributions and breaks certification/serving exchangeability; stress has no guarantee.",
        corruption_group_rows="Every corruption group retains its served counts, refusals and selected-rule risks outside the guarantee.",
        paired_replicate_rows="One row per independent replicate averages seeds inside each fixed arm family before pairing protocols.",
        simultaneous_intervals="Bonferroni allocates total error .01 over36 frozen bounded Hoeffding quantities; shared seeds and reweightings do not increase n.",
        policy_summaries="Each fixed arm family retains its own coverage/safety utility gates and comparative conclusion.",
        certificate_availability="Exact binomial 99-percent intervals divide issued group certificates by all40 scheduled replicates for each policy.",
        calibration_family_violation_count="Count replicates with any selected-rule population risk exceeding its calibration upper bound, including refused rules.",
        calibration_family_violation_rows="The56 bounds form one within-replicate family;40 replicate observations are not a universal guarantee.",
        worst_served_group_risk="Maximum census risk over actually served groups; absent service retains null.",
        comparison="Pooled utility forces parity; low coverage is a terminal null and cannot trigger retries.",
        historical_training_calls="Retained Exp8868 fitting operations belong to that producer and never count as current training.",
        conditional_rows_ready_score="Complete authenticated measurements and owned checks, independent of utility or advantage.",
        conditional_utility_score="Predeclared learned-energy family meets lower99 coverage in every mixture and no unsafe certified group.",
        robustness_advantage_score="Utility plus strictly positive paired excess-risk reduction; pooled utility forces parity.",
        independent_count="Forty replicate pairs; policy seeds and mixture reweightings add no independent observations.",
        numerical_training_calls="Zero current training calls; historical fitting counts are separately attributed.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    prospective = raw / "prospective.json"
    atomic_json(prospective, value)
    if not args.private_output:
        m.progress("before_terminal_checks")
        commands = [
            (
                "adversarial",
                [
                    sys.executable,
                    "-u",
                    str(m.ROOT / "scripts/adversarial_verify.py"),
                    "--json",
                    str(prospective),
                ],
            ),
            (
                "rows",
                [
                    sys.executable,
                    "-u",
                    str(m.ROOT / "scripts/verdict_row_consistency_lint.py"),
                    "--strict",
                    str(prospective),
                ],
            ),
            (
                "cold_replay",
                [
                    "env",
                    "-u",
                    "PYTHONPATH",
                    "--chdir=/tmp",
                    sys.executable,
                    "-u",
                    str(m.ROOT / m.CLI),
                    "--cold-replay",
                    str(prospective),
                ],
            ),
        ]
        reports = [
            stream(name, command, raw / "checks", deadline=300) for name, command in commands
        ]
        value["validation_receipts"] += reports
        value["raw_shard_hashes"] += [
            reference(Path(r[s + "_path"])) for r in reports for s in ["stdout", "stderr"]
        ]
        if not all(r["passed"] for r in reports):
            value.update(
                honest_verdict="complete_disqualified_conditional_risk_evaluation",
                verdict_class="disqualified",
                conditional_rows_ready_score=0,
                required_checks_passed=False,
            )
        del value["reproducibility_checksum"]
        value["reproducibility_checksum"] = canonical_hash(value)
        m.progress("after_terminal_checks")
    publish_primary(output, value, replay)
    m.progress("published_" + value["verdict_class"], completed, 3360 - completed)
    return 0

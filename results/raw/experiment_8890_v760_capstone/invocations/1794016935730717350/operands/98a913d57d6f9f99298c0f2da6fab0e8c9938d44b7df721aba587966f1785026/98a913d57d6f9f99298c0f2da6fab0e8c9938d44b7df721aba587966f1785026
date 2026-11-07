"""REQ-REPORT-8885: authenticate frozen methods and publish checked label-cost evidence."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import sequential_label_cost_8885 as m
from carnot.reporting import sequential_certificate_runner_8880 as upstream
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
from scripts.experiment_template import ExperimentTemplate, normalize_artifact_for_template_write

Json = dict[str, Any]
SOURCE = "experiment_8880_v760_sequential_certificate_methods"
UPSTREAM = {
    SOURCE: (
        "sha256:482065c936b96cd56ffec0fb41abb0a683e17f98b750d7de2ba5479016a5c91e",
        "sequential_methods_ready_score",
    )
}
HEALTH = Path("/tmp/carnot-8885-health/receipt.json")
DEADLINE = 1800


def preconditions(raw: Path, sources: Path = m.ROOT / "results") -> Json:
    """Original producer pins bind authority, configuration and every retained raw operand."""
    checker = namespace(upstream, "cost8885_preconditions")
    checker.UPSTREAM = UPSTREAM
    return dict(checker.preconditions(raw, sources))


def validate(raw: Path, private: Path) -> Json:
    """Shared validation retains real unit/child coverage and private E2E018/021 outcomes."""
    checker = namespace(upstream, "cost8885_validation")
    checker.m = m
    return dict(checker.validate(raw, private))


def measurement(raw: Path, *, replaying: bool = False) -> Json:
    """Reexecute frozen draws, sealing both certificates before requesting any census labels."""
    started, events = time.monotonic(), []

    def seal(stage: str, value: Json) -> Json:
        if time.monotonic() - started > DEADLINE:
            raise ValueError("measurement_deadline")
        path = raw / (stage + ".json")
        if replaying:
            if canonical_hash(load(reference(path))) != canonical_hash(value):
                raise ValueError("measurement_drift:" + stage)
        else:
            atomic_json(path, value)
            events.append(dict(stage=stage, monotonic_ns=time.monotonic_ns(), **reference(path)))
            atomic_json(raw / "access_order.json", dict(events=events))
        m.progress("replayed_" + stage if replaying else "sealed_" + stage, len(events), 0)
        return value

    seal("protocol", m.protocol())
    source = load(reference(raw / (SOURCE + ".json")))
    retained = {}
    for stage in ["population", "draws", "scores", "selection_labels", "certification_labels"]:
        ref = next(x for x in source["raw_shard_hashes"] if Path(x["path"]).name == stage + ".json")
        path = raw / "retained_inputs" / (stage + ".json")
        if not replaying:
            path.parent.mkdir(exist_ok=True)
            shutil.copyfile(ref["path"], path)
        if sha256_file(path) != ref["sha256"]:
            raise ValueError("original_operand_drift")
        retained[stage] = reference(path)
    seal("frozen_operands", retained)
    cohort, trace = load(retained["population"]), load(retained["draws"])
    scores = load(retained["scores"])["rows"]
    selection = seal(
        "selection",
        m.frozen.select(cohort, scores, load(retained["selection_labels"])["labels"], trace),
    )
    m.progress("before_certification_benchmark", 0, 2240)
    cert = m.frozen.certify(
        cohort, scores, load(retained["certification_labels"])["labels"], trace, selection
    )
    if canonical_hash(cert["rows"]) != canonical_hash(source["rows"]):
        raise ValueError("frozen_certificate_drift")
    seal("certificates", cert)
    m.progress("after_certification_before_census", 2240, 4096)
    population = [u for u in cohort["inputs"] if u["split"] == "population"]
    truth = seal("census_labels", m.frozen.base.old.old.p.labels(population))
    m.progress("before_census_benchmark", 0, 40)
    result = m.evaluate(
        cohort,
        scores,
        truth,
        trace,
        cert,
        calibration_truth=load(retained["certification_labels"])["labels"],
    )
    result["actual_label_ledger"].update(
        retained_acquisition_manifests=[
            retained["selection_labels"],
            retained["certification_labels"],
        ],
        census_acquisition_manifest=reference(raw / "census_labels.json"),
        historical_training_calls=source["historical_training_calls"],
    )
    seal("evaluation", result)
    m.progress("after_census_benchmark", 40, 0)
    return result


def reduction(raw: Path) -> Json:
    """Phase order and reconstruction reject semantic changes even after hashes are updated."""
    events = load(reference(raw / "access_order.json"))["events"]
    if [x["stage"] for x in events] != m.protocol()["access_order"] or any(
        a["monotonic_ns"] > b["monotonic_ns"] for a, b in zip(events, events[1:])
    ):
        raise ValueError("access_order_drift")
    for event in events:
        load(event)
    return measurement(raw, replaying=True)


def scheduled(reason: str, status: str = "censored") -> Json:
    """Every scheduled policy/group remains visible when external or owned work cannot finish."""
    return dict(
        rows=upstream.scheduled(reason, status)["rows"],
        stopping_rows=[],
        per_replicate_results=[],
        actual_label_ledger={},
        standalone_costs=[],
        shared_union_costs=[],
        certificate_availability=[],
        family_violation_rows=[],
        census_risk_rows=[],
        paired_intervals=[],
        label_savings_score=0,
        numerical_training_calls=0,
    )


def replay(path: Path) -> Json:
    """Only reconstructed scientific results and qualified complete rows can grant readiness."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    if canonical_hash(value) != checksum:
        raise ValueError("artifact_checksum")
    if (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"]) != (
        8885,
        m.TASK,
        "2026.11.760",
        "20261106",
    ):
        raise ValueError("producer_identity")
    if canonical_hash(value["protocol"]) != canonical_hash(m.protocol()):
        raise ValueError("protocol_drift")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("operand_hash")
    if validate_current_work_receipt(value["current_work_receipt"]):
        raise ValueError("current_receipt")
    counts = {
        s: sum(x["status"] == s for x in value["rows"]) for s in ["completed", "failed", "censored"]
    }
    if (
        len(value["rows"]) != 2240
        or sum(counts.values()) != 2240
        or value["excluded_count"] != 0
        or any(value[s + "_count"] != n for s, n in counts.items())
        or value["independent_count"] != (40 if counts["completed"] == 2240 else 0)
    ):
        raise ValueError("count_drift")
    if counts["completed"]:
        raw = Path(value["measurement_manifest"]["path"]).parent
        if sha256_file(raw / (SOURCE + ".json")) != UPSTREAM[SOURCE][0]:
            raise ValueError("original_producer_drift")
        result = reduction(raw)
        if any(canonical_hash(value[k]) != canonical_hash(v) for k, v in result.items()):
            raise ValueError("reduction_drift")
    ready = value["sequential_rows_ready_score"]
    if ready not in [0, 1] or ready != int(
        counts["completed"] == 2240
        and value["required_checks_passed"]
        and value["verdict_class"] in ["null", "circular_positive"]
    ):
        raise ValueError("readiness_drift")
    if ready and (
        not value["owned_statement_coverage"]["complete"]
        or value["owned_statement_coverage"]["percent"] != 100
        or {x["path"] for x in value["owned_statement_coverage"]["files"]} != set(m.OWNED)
        or any(x["missing"] or x["excluded"] for x in value["owned_statement_coverage"]["files"])
        or not all(x["passed"] for x in value["validation_receipts"])
    ):
        raise ValueError("qualification_drift")
    if value["label_savings_score"] and (
        not ready or value["verdict_class"] != "circular_positive"
    ):
        raise ValueError("benefit_drift")
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Publish one terminal artifact; private fixtures cannot authorize production readiness."""
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
    template = ExperimentTemplate(
        8885,
        "Sequential certification label cost",
        "results/" + m.NAME + ".json",
        repo_root=m.ROOT,
        seed=m.SEED,
    )
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
    with TemporaryDirectory(prefix="exp8885-validation-") as directory:
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
        if result["label_savings_score"]
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
        inference_substrate="cpu_retained_heads_exact_sequential_certificate_cost",
        inference_substrate_class="no_model_load" if completed else "blocked_no_run",
        execution_venue="host",
        inference_substrate_details=dict(
            scope="retained scores; zero fresh training or pretrained loads"
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
        experiment_id=8885,
        task_id=m.TASK,
        milestone="2026.11.760",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_sequential_label_cost",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        verifier_is_oracle=True,
        exposure_scope="constructed stationary population only; no within-group corruption or natural-text validity",
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
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ],
        raw_shard_hashes=[reference(p) for p in raw.rglob("*.json")]
        + validation["raw_references"]
        + [reference(p) for p in raw.glob("repository_health.std*")],
        repository_health=health,
        intended_count=2240,
        completed_count=completed,
        failed_count=sum(x["status"] == "failed" for x in result["rows"]),
        censored_count=sum(x["status"] == "censored" for x in result["rows"]),
        excluded_count=0,
        independent_count=40 if completed == 2240 else 0,
        gate_check_summary=pre["diagnostics"],
        owned_errors=errors,
        acceptance_gates=dict(
            total_label_savings_lower_gt=0,
            coverage_difference_lower_ge=-0.01,
            maximum_served_group_risk=0.05,
            owned_coverage=100,
        ),
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        sequential_rows_ready_score=int(
            completed == 2240 and valid and verdict in ["null", "circular_positive"]
        ),
        cited_upstream_artifacts=[
            dict(
                experiment_id=8880, readiness_field=UPSTREAM[SOURCE][1], sha256=UPSTREAM[SOURCE][0]
            )
        ],
        protocol=m.protocol(),
        measurement_manifest=reference(raw / "evaluation.json") if completed else None,
        methodology="Forty frozen independent paired replicates; all candidate draws including repeats are paid. Exact oracle census follows fixed thresholds and certificates. Five learned seeds average within replicate. Bonferroni bounded Hoeffding intervals cover23 fixed contrasts with family error .01. Null savings is terminal.",
        methodology_note="Theoretical per-replicate family error and observed violations are separate. Forty clean replicates do not prove a universal guarantee. Constructed oracle truth establishes no independent generalization.",
        flagged_adversarial=False,
        field_principles={},
    )
    value["field_principles"] = {
        k: "Frozen or measured operand bound to original bytes; readiness is independent of benefit and oracle truth is not generalization."
        for k in value
    }
    value["field_principles"].update(
        actual_label_ledger="Actual joint selection/certification draws count repeats; census and retained historical training remain separate.",
        standalone_costs="Each policy pays its own selection and certification candidates in this counterfactual deployment.",
        shared_union_costs="Seven policies share selection and the longest required candidate prefix in each group.",
        family_violation_rows="Test every potential calibration bound, including refused and unvisited looks; forty observations are not a universal guarantee.",
        paired_intervals="Two-sided99-percent simultaneous bounded intervals over23 fixed paired-replicate quantities; learned seeds add no independent observations.",
        label_savings_score="Learned-family cost savings require retained coverage in every original mixture and safe served census groups.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    prospective = raw / "prospective.json"
    atomic_json(prospective, value)
    if not args.private_output:
        m.progress("before_terminal_checks")
        reports = []
        for name, script, options in [
            ("adversarial", "adversarial_verify.py", ["--json"]),
            ("rows", "verdict_row_consistency_lint.py", ["--strict"]),
        ]:
            reports.append(
                stream(
                    name,
                    [
                        sys.executable,
                        "-u",
                        str(m.ROOT / "scripts" / script),
                        *options,
                        str(prospective),
                    ],
                    raw / "checks",
                    deadline=180,
                )
            )
        reports.append(
            stream(
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
                raw / "checks",
                deadline=300,
            )
        )
        value["validation_receipts"] += reports
        value["raw_shard_hashes"] += [
            reference(Path(x[s + "_path"])) for x in reports for s in ["stdout", "stderr"]
        ]
        if not all(x["passed"] for x in reports):
            value.update(
                honest_verdict="complete_disqualified_sequential_label_cost",
                verdict_class="disqualified",
                sequential_rows_ready_score=0,
                required_checks_passed=False,
            )
        value.pop("reproducibility_checksum")
        value["reproducibility_checksum"] = canonical_hash(value)
        m.progress("after_terminal_checks")
    publish_primary(output, value, replay)
    m.progress("published_" + value["verdict_class"], completed, 2240 - completed)
    return 0

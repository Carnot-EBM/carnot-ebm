"""REQ-REPORT-8880: authenticate original inputs and publish finite-look methods."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import sequential_certificate_methods_8880 as m
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
from scripts.experiment_template import ExperimentTemplate, normalize_artifact_for_template_write

Json = dict[str, Any]
SOURCE = "experiment_8868_v759_conditional_risk_methods"
UPSTREAM = {
    SOURCE: (
        "sha256:af45370397fc99a00514d511a3c81d29790a879e3bb6bafcf55701e0088d5207",
        "conditional_methods_ready_score",
    ),
    "experiment_8870_v759_conditional_risk_evaluation": (
        "sha256:5dc1020bfc6fa45d40a1465c2353555291479b250d71f0f7cf2242458a9a8960",
        "conditional_rows_ready_score",
    ),
    "experiment_8878_v760_branch_contract": (
        "sha256:697c14905bb02ed4f2fce51028575e7a71bb8ad58b0e5eb8a9bb532f53532a21",
        "contract_ready_score",
    ),
}
DEADLINE = 1800
HEALTH = Path("/tmp/carnot-8880-health/receipt.json")


def preconditions(raw: Path, sources: Path = m.ROOT / "results") -> Json:
    """Existing producer checks bind original assertions and retained raw operands."""
    checker = namespace(previous, "sequential8880_preconditions")
    checker.UPSTREAM = UPSTREAM
    checker.DESIGN = "openspec/change-proposals/research-methods-v760-20261106.json"
    checker.DESIGN_PIN = "sha256:c99296bd441c10200bad04e77643649bd65f62fb2199029302b24473af6da3d7"
    return dict(checker.preconditions(raw, sources))


def validate(raw: Path, private: Path) -> Json:
    """Retain actual owned unit/child coverage and private original E2E018/021 checks."""
    checker = namespace(inherited, "sequential8880_validation")
    checker.m = m
    return dict(checker.validate(raw, private))


def measurement(raw: Path, pre: Json, *, replaying: bool = False) -> Json:
    """Seal cutoffs before certification labels; census labels are never requested here."""
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
    historical = load(source["population_manifest"])
    fitted = load(source["fit_manifest"])
    reserved = previous.exposed(pre)
    pilot_start = time.monotonic()
    pilot = m.freeze(reserved, count=2)
    controls = m.controls()
    if not all(x["passed"] for x in controls):
        raise ValueError("pilot_control_failure")
    projected = (time.monotonic() - pilot_start) * 512 + 1200
    pilot_result = dict(population=pilot, controls=controls)
    if not replaying:
        atomic_json(
            raw / "pilot_runtime.json",
            dict(duration_s=time.monotonic() - pilot_start, projected_total_s=projected),
        )
    if load(reference(raw / "pilot_runtime.json"))["projected_total_s"] > 4200:
        raise ValueError("pilot_runtime")
    seal("pilot", pilot_result)
    reserved |= {s for w in pilot["worlds"] for s in w["relevant_structures"]}
    cohort = m.freeze(reserved)
    cohort["inputs"] += [u for u in historical["inputs"] if u["split"] == "fit"]
    seal("population", cohort)
    trace = seal("draws", m.draws(1024))
    m.base.old.old.audit_fit(cohort, fitted)
    seal("fit", fitted)
    m.progress("before_prediction_no_model_load", 0, 4096)
    scores = m.base.old.old.predict(cohort, fitted)
    seal("scores", dict(rows=scores))
    m.progress("after_prediction_no_model_load", 4096, 0)
    _, indices = m.base.registry(cohort, scores)
    by_id = {u["unit_id"]: u for u in cohort["inputs"]}
    units = sorted({s["unit_id"] for s in scores})

    def release(stage: str) -> Json:
        requested = [
            dict(
                draw_id=f"{rep['replicate']}/{g}/{stage}/{j}",
                replicate=rep["replicate"],
                group=g,
                candidate_index=i,
                unit_id=units[indices[g][i]],
            )
            for rep in trace["replicates"]
            for g in m.base.GROUPS
            for j, i in enumerate(rep["groups"][g][stage])
        ]
        unique = {r["unit_id"] for r in requested}
        truth = m.base.old.old.p.labels([by_id[u] for u in sorted(unique)])
        seal(
            stage + "_labels",
            dict(labels=truth, acquired_draws=requested, paid_labels=len(requested)),
        )
        return truth

    selection = seal("selection", m.select(cohort, scores, release("selection"), trace))
    m.progress("before_certification_benchmark", 0, 2240)
    result = m.certify(cohort, scores, release("certification"), trace, selection)
    seal("certificates", result)
    m.progress("after_certification_benchmark", len(result["rows"]), 0)
    return dict(
        result,
        positive_negative_controls=controls,
        historical_training_calls=fitted["numerical_training_calls"],
    )


def reduction(raw: Path, pre: Json) -> Json:
    """Rehashed aggregates cannot replace original draws, predictions or phase order."""
    events = load(reference(raw / "access_order.json"))["events"]
    if [e["stage"] for e in events] != m.protocol()["access_order"] or any(
        a["monotonic_ns"] > b["monotonic_ns"] for a, b in zip(events, events[1:])
    ):
        raise ValueError("access_order_drift")
    for e in events:
        load(e)
    return measurement(raw, pre, replaying=True)


def scheduled(reason: str, status: str = "censored") -> Json:
    """External absence or owned failure retains every scheduled policy/group cell."""
    rows = [
        dict(
            unit_id=f"{rep}/{name}/{arm}/{seed}/{g}",
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
        for name in m.ALPHAS
        for arm, seed in m.base.POLICIES
        for g in m.base.GROUPS
    ]
    return dict(
        rows=rows,
        standalone_costs=[],
        shared_union_costs=[],
        actual_research_labels=0,
        numerical_training_calls=0,
        positive_negative_controls=[],
        historical_training_calls=0,
    )


def replay(path: Path) -> Json:
    """Only authenticated operands and independently reconstructed results can grant readiness."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    if canonical_hash(value) != checksum:
        raise ValueError("artifact_checksum")
    if (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"]) != (
        8880,
        m.TASK,
        "2026.11.760",
        "20261106",
    ):
        raise ValueError("producer_identity")
    if canonical_hash(value["protocol"]) != canonical_hash(m.protocol()):
        raise ValueError("protocol_drift")
    for field in [
        "look_schedule",
        "alpha_allocation",
        "bound_proof",
        "label_cost_definition",
        "exact_sample_requirements",
    ]:
        if canonical_hash(value[field]) != canonical_hash(m.protocol()[field]):
            raise ValueError("frozen_contract_drift")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("operand_hash")
    if validate_current_work_receipt(value["current_work_receipt"]):
        raise ValueError("current_receipt")
    rows = value["rows"]
    counts = {s: sum(x["status"] == s for x in rows) for s in ["completed", "failed", "censored"]}
    if (
        len(rows) != 2240
        or any(value[s + "_count"] != n for s, n in counts.items())
        or value["excluded_count"] != 0
        or sum(counts.values()) != 2240
        or value["independent_count"] != (40 if counts["completed"] == 2240 else 0)
    ):
        raise ValueError("count_drift")
    if counts["completed"]:
        raw = Path(value["population_manifest"]["path"]).parent
        for name, (pin, _) in UPSTREAM.items():
            if sha256_file(raw / (name + ".json")) != pin:
                raise ValueError("original_producer_drift")
        result = reduction(raw, value["preconditions_checked"])
        if any(canonical_hash(value[k]) != canonical_hash(v) for k, v in result.items()):
            raise ValueError("reduction_drift")
    ready = value["sequential_methods_ready_score"]
    if ready not in [0, 1] or ready != int(
        counts["completed"] == 2240
        and value["required_checks_passed"]
        and value["verdict_class"] == "null"
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
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Execute methods research with private checkpoints and one checked terminal publication."""
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
        8880,
        "Finite-look certificate methods",
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
                result = measurement(raw, pre)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            errors.append(str(error))
            result = scheduled(str(error), "censored" if "deadline" in str(error) else "failed")
    measured = time.monotonic_ns()
    m.progress("before_owned_validation")
    with TemporaryDirectory(prefix="exp8880-validation-") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
    valid = bool(
        validation["coverage"]["complete"]
        and validation["union"]["passed"]
        and all(x["passed"] for x in validation["receipts"])
    )
    m.progress("after_owned_validation")
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
        dict(phase="preconditions_and_measurement", start_s=0.0, end_s=(measured - started) / 1e9),
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
        inference_substrate="cpu_retained_heads_and_finite_look_exact_certificates",
        inference_substrate_class="no_model_load" if completed else "blocked_no_run",
        execution_venue="host",
        inference_substrate_details=dict(
            scope="retained numerical heads; zero new training or pretrained loads"
        ),
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=phases,
    )
    atomic_json(raw / "current_work_receipt.json", receipt)
    health = None
    if HEALTH.is_file():
        health = load(reference(HEALTH))
        target = raw / "repository_health.stdout"
        shutil.copyfile(health["stdout_path"], target)
        health.update(stdout_path=str(target), stdout_sha256=sha256_file(target))
        atomic_json(raw / "repository_health.json", health)
    value = dict(
        result,
        experiment_id=8880,
        task_id=m.TASK,
        milestone="2026.11.760",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_sequential_certificate_methods",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        verifier_is_oracle=True,
        exposure_scope="constructed stationary signed relational population only; exact oracle labels; no natural-language generalization",
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
        raw_shard_hashes=[reference(p) for p in raw.glob("*.json")]
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
            valid_controls=True, owned_coverage=100, fixed_population=True, benefit_required=False
        ),
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        sequential_methods_ready_score=int(verdict == "null" and completed == 2240),
        cited_upstream_artifacts=[
            dict(experiment_id=int(n.split("_")[1]), readiness_field=f, sha256=p)
            for n, (p, f) in UPSTREAM.items()
        ],
        protocol=m.protocol(),
        **{
            k: m.protocol()[k]
            for k in [
                "look_schedule",
                "alpha_allocation",
                "bound_proof",
                "label_cost_definition",
                "exact_sample_requirements",
            ]
        },
        **{
            k + "_manifest": reference(raw / (s + ".json")) if completed else None
            for k, s in [
                ("population", "population"),
                ("draw", "draws"),
                ("selection", "selection"),
                ("fit", "fit"),
            ]
        },
        methodology="Frozen query groups, retained policy heads, independent selection/certification replacement streams; first valid alpha-spent exact upper bound or final refusal. All acquired labels and policy unions are charged. No census or cost-benefit inference; evaluation belongs to Exp8885.",
        methodology_note="Methods readiness is independent of savings. Family error .01 applies per replicate. Oracle controls establish no independent generalization.",
        flagged_adversarial=False,
        field_principles={},
    )
    value["field_principles"] = {
        k: "Frozen method or measured operand bound to original bytes; methods readiness does not claim label savings or independent generalization."
        for k in value
    }
    value["field_principles"].update(
        actual_research_labels="Joint selection plus full fixed candidate streams; repeats count as acquisitions, not independent worlds.",
        standalone_costs="Counterfactual standalone per-policy protocol costs; shared labels are not charged seven times in the actual research ledger.",
        shared_union_costs="Selection once per group plus candidates through the last unfinished policy; joint research also includes fixed-arm candidates.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    prospective = raw / "prospective.json"
    atomic_json(prospective, value)
    if not args.private_output:
        m.progress("before_terminal_checks")
        reports = [
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
            for name, script, options in [
                ("adversarial", "adversarial_verify.py", ["--json"]),
                ("rows", "verdict_row_consistency_lint.py", ["--strict"]),
            ]
        ]
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
                honest_verdict="complete_disqualified_sequential_certificate_methods",
                verdict_class="disqualified",
                sequential_methods_ready_score=0,
                required_checks_passed=False,
            )
        value.pop("reproducibility_checksum")
        value["reproducibility_checksum"] = canonical_hash(value)
        m.progress("after_terminal_checks")
    publish_primary(output, value, replay)
    m.progress("published_" + value["verdict_class"], completed, 2240 - completed)
    return 0

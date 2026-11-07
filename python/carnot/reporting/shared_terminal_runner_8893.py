"""REQ-REPORT-8893: authenticate original evidence and publish a checked terminal method."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import shared_terminal_methods_8893 as m
from carnot.reporting import sequential_certificate_runner_8880 as upstream
from carnot.reporting import conditional_risk_runner_8868 as inherited
from carnot.reporting import topology_memory_runner_8881 as reused
from carnot.reporting import v761_branch_contract as authority
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
SOURCE = upstream.SOURCE
HEALTH = Path("/tmp/carnot-8893-work/repository-health.json")
UPSTREAM = {
    "experiment_8885_v760_sequential_label_cost": (
        "sha256:4372f5a0a50918da1fe848350a1386208aa0697208a9a907b7f90e99b0b56c53",
        "sequential_rows_ready_score",
    ),
    SOURCE: (
        "sha256:af45370397fc99a00514d511a3c81d29790a879e3bb6bafcf55701e0088d5207",
        "conditional_methods_ready_score",
    ),
}
PRESERVED = {
    m.ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md": m.ROOT
    / "openspec/change-proposals/research-roadmap-v760-preserved-20261106.md",
    m.ROOT / "research-roadmap.yaml": m.ROOT
    / "openspec/change-proposals/research-roadmap-v760-queue-preserved-20261106.yaml",
    m.ROOT / "research-references.md": m.ROOT
    / "results/raw/experiment_8889_v760_hardware_operation_map/invocations/694907840058934/operands/11526c782c4077c670fb9dbd3fb741bbb9e6f91196a78d423547cff795629958",
}


def preconditions(raw: Path, sources: Path) -> Json:
    """Never repair original producer assertions when their bound bytes no longer authenticate."""
    checker = namespace(inherited, "shared8893_preconditions")
    checker.UPSTREAM = UPSTREAM
    checker.DESIGN = authority.METHODS
    checker.DESIGN_PIN = "sha256:e02928f8cce705aa5e62d57ad3a591f9b3cb9cda1c011fddd1867245fb0efe88"
    checked: dict[Path, str] = {}

    def authenticated_hash(path: Path) -> str:
        resolved = PRESERVED.get(path, path)
        if resolved not in checked:
            checked[resolved] = sha256_file(resolved)
            if len(checked) % 500 == 0:
                m.progress("authenticate_original_operands", len(checked), 1)
        return checked[resolved]

    # Resolve rolled-over metadata to exact original bytes, preserving every expected hash.
    checker.sha256_file = authenticated_hash

    def original_reference(path: Path) -> Json:
        return reference(PRESERVED.get(path, path))

    checker.reference = original_reference
    pre = dict(checker.preconditions(raw, sources))
    for ref in pre["references"]:
        original = Path(ref["path"])
        if original in PRESERVED:
            target = raw / "retained_originals" / ref["sha256"].split(":")[1]
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(PRESERVED[original], target)
            ref.update(reference(target), original_path=str(original))
    contract = authority.authority(m.ROOT)
    pre["authority"] = contract
    if not contract["ready"]:
        pre["diagnostics"].append(
            dict(
                check="rejected_producer",
                upstream="V761",
                artifact_field="execution_authority",
                expected=True,
                observed=False,
                path=str(m.ROOT / authority.DESIGN),
                hash=sha256_file(m.ROOT / authority.DESIGN),
            )
        )
    for name in UPSTREAM:
        path = sources / (name + ".json")
        if path.is_file():
            target = raw / path.name
            shutil.copyfile(path, target)
            pre["references"].append(dict(reference(target), original_path=str(path)))
    for diagnostic in pre["diagnostics"]:
        original = Path(diagnostic["path"])
        resolved = PRESERVED.get(original, original)
        diagnostic.update(
            original_path=str(original),
            path=str(resolved),
            hash=sha256_file(resolved) if resolved.is_file() else None,
        )
    pre["external_ready"] = not pre["diagnostics"]
    return pre


def validate(raw: Path, private: Path) -> Json:
    """Reuse measured unit/child coverage, strict checks and private E2E018/021 routes."""
    runner = namespace(upstream, "shared8893_validation")
    runner.m = m
    return dict(runner.validate(raw, private))


def measurement(raw: Path, pre: Json, *, replaying: bool = False) -> Json:
    """Reuse the frozen-head runner and open census truth only after the final certificates."""
    runner = namespace(upstream, "shared8893_measurement")
    runner.m = m
    cert = runner.measurement(raw, pre, replaying=replaying)
    cohort = load(reference(raw / "population.json"))
    scores = load(reference(raw / "scores.json"))["rows"]
    trace = load(reference(raw / "draws.json"))
    truth = m.base.old.old.p.labels([u for u in cohort["inputs"] if u["split"] == "population"])
    result = m.evaluate(cohort, scores, truth, trace, cert)
    result["historical_training_calls"] = cert["historical_training_calls"]
    for stage, value in [("census_labels", truth), ("evaluation", result)]:
        path = raw / (stage + ".json")
        if replaying:
            if canonical_hash(load(reference(path))) != canonical_hash(value):
                raise ValueError("measurement_drift:" + stage)
        else:
            atomic_json(path, value)
            order = load(reference(raw / "access_order.json"))
            order["events"].append(
                dict(stage=stage, monotonic_ns=time.monotonic_ns(), **reference(path))
            )
            atomic_json(raw / "access_order.json", order)
        m.progress("sealed_" + stage)
    return result


def reduction(raw: Path, pre: Json) -> Json:
    """Recompute science from sealed inputs so a new aggregate checksum grants no authority."""
    events = load(reference(raw / "access_order.json"))["events"]
    if [x["stage"] for x in events] != m.protocol()["access_order"] or any(
        a["monotonic_ns"] > b["monotonic_ns"] for a, b in zip(events, events[1:])
    ):
        raise ValueError("access_order_drift")
    for event in events:
        load(event)
    return measurement(raw, pre, replaying=True)


def scheduled(reason: str, status: str = "censored") -> Json:
    """Every unmeasured policy/group remains in the denominator with its failure reason."""
    return dict(
        upstream.scheduled(reason, status),
        fixed_alpha_sensitivity_rows=[],
        census_risk_rows=[],
        family_violation_rows=[],
        paired_intervals=[],
        per_replicate_results=[],
        actual_label_ledger={},
        label_savings_score=0,
        pathwise_inclusion_violations=0,
        standalone_learned_policy_costs=[],
    )


def replay(path: Path) -> Json:
    """Readiness requires exact identity, intact original bytes and qualified complete rows."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    if canonical_hash(value) != checksum:
        raise ValueError("artifact_checksum")
    if (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"]) != (
        8893,
        m.TASK,
        "2026.11.761",
        "20261107",
    ):
        raise ValueError("producer_identity")
    if canonical_hash(value["protocol"]) != canonical_hash(m.protocol()):
        raise ValueError("protocol_drift")
    if any(
        canonical_hash(value[k]) != canonical_hash(m.protocol()[k])
        for k in ["alpha_allocation", "coverage_inclusion_proof", "fixed_final_test"]
    ):
        raise ValueError("protocol_declaration_drift")
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
        or value["intended_count"] != 2240
        or sum(counts.values()) != 2240
        or value["excluded_count"] != 0
        or any(value[s + "_count"] != n for s, n in counts.items())
        or value["independent_count"] != (40 if counts["completed"] == 2240 else 0)
    ):
        raise ValueError("count_drift")
    if counts["completed"]:
        raw = Path(value["measurement_manifest"]["path"]).parent
        if any(sha256_file(raw / (name + ".json")) != pin for name, (pin, _) in UPSTREAM.items()):
            raise ValueError("original_producer_drift")
        result = reduction(raw, value["preconditions_checked"])
        if any(canonical_hash(value[k]) != canonical_hash(v) for k, v in result.items()):
            raise ValueError("reduction_drift")
    ready = value["shared_terminal_ready_score"]
    if ready != int(
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
        counts["completed"] != 2240
        or value["verdict_class"] not in ["circular_positive", "disqualified"]
    ):
        raise ValueError("benefit_drift")
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Keep checkpoints private until validation; completion does not imply measured benefit."""
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
        8893,
        "Shared terminal certification",
        "results/" + m.NAME + ".json",
        repo_root=m.ROOT,
        seed=m.SEED,
    )
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    started = time.monotonic_ns()
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
    with TemporaryDirectory(prefix="exp8893-validation-") as directory:
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
        inference_substrate="cpu_frozen_heads_shared_terminal_methods",
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
        if sha256_file(source) != health["log"]["sha256"]:
            raise ValueError("repository_health_log_drift")
        target = raw / "repository_health.log"
        shutil.copyfile(source, target)
        health["log"] = reference(target)
        atomic_json(raw / "repository_health.json", health)
    p = m.protocol()
    value = dict(
        result,
        experiment_id=8893,
        task_id=m.TASK,
        milestone="2026.11.761",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_shared_terminal_methods",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        target_model="seven frozen numerical policies: two baselines and five retained heads; no pretrained model",
        model_invocation_counts=receipt["invocation_counts"],
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
            reference(m.ROOT / f)
            for f in m.OWNED
            + m.TESTS
            + [
                "scripts/experiment_template.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/sequential_certificate_runner_8880.py",
                "python/carnot/reporting/conditional_risk_methods_8868.py",
                "python/carnot/reporting/sequential_label_cost_8885.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ],
        raw_shard_hashes=[reference(f) for f in raw.rglob("*.json")]
        + ([reference(raw / "repository_health.log")] if health else [])
        + validation["raw_references"],
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
            pathwise_coverage_inclusion=True,
            scientific_savings_lower_gt=0,
        ),
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        shared_terminal_ready_score=int(
            completed == 2240 and valid and verdict in ["null", "circular_positive"]
        ),
        cited_upstream_artifacts=[
            dict(producer=name, sha256=pin, readiness_field=field)
            for name, (pin, field) in UPSTREAM.items()
        ],
        protocol=p,
        alpha_allocation=p["alpha_allocation"],
        coverage_inclusion_proof=p["coverage_inclusion_proof"],
        sufficient_statistic_controls=m.controls(),
        fixed_final_test=p["fixed_final_test"],
        candidate_draw_manifest=reference(raw / "draws.json") if completed else None,
        frozen_policies=p["policy_menu"],
        label_cost_definition=p["label_cost_definition"],
        measurement_manifest=reference(raw / "evaluation.json") if completed else None,
        methodology="Local finite-look union bound with one shared terminal event. Fixed implies sequential certification pathwise. Exact census follows frozen certification. Selection and every candidate draw, including repeats, are charged. Forty independent paired draws; five learned seeds average within replicate. Bootstrap-free bounded Hoeffding intervals use 99 percent simultaneous confidence over23 frozen contrasts.",
        methodology_note="Oracle construction establishes no natural-language generalization. Statistical risk needs iid stationary within-group draws. Inclusion is logical; methods readiness and scientific benefit remain separate. Missing original provenance blocks measurement.",
        flagged_adversarial=False,
        field_principles={},
    )
    value["field_principles"] = {
        k: "Frozen or measured operand; exact source bytes bind scope. Completion and readiness do not imply scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        coverage_inclusion_proof="Logical inclusion holds on every draw sequence; it does not establish answer correctness.",
        alpha_allocation="The shared final event is counted once; stationary iid calibration gives .01 error per replicate.",
        fixed_alpha_sensitivity_rows="The stronger fixed-only alpha arm is descriptive and outside the joint risk allocation.",
        standalone_learned_policy_costs="Each learned seed pays selection and all candidates; seeds average inside each independent replicate.",
        shared_union_costs="All seven policies share selection; the longest unfinished prefix determines paid union labels.",
        paired_intervals="99-percent simultaneous bounded Hoeffding intervals use forty paired replicates; descriptive coverage intervals do not replace the inclusion proof.",
        repository_health="The bounded full-suite outcome is retained separately and grants no owned qualification.",
        target_model="Names the actual retained numerical policies without claiming any current pretrained load or fresh training.",
        duration_s="Measured preconditions, measurement and owned validation; later terminal checks retain their own child durations.",
        shared_terminal_ready_score="Proof, controls and complete owned qualification authorize measurement even when union savings are zero.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    candidate = raw / "prospective.json"
    atomic_json(candidate, value)
    if not args.private_output:
        m.progress("before_terminal_checks")
        runner = namespace(reused, "shared8893_terminal")
        runner.m = m
        reports = runner.terminal(candidate, raw)
        value["validation_receipts"] += reports
        value["raw_shard_hashes"] += [
            reference(Path(x[s + "_path"])) for x in reports for s in ["stdout", "stderr"]
        ]
        if not all(x["passed"] for x in reports):
            value.update(
                honest_verdict="complete_disqualified_shared_terminal_methods",
                verdict_class="disqualified",
                shared_terminal_ready_score=0,
                required_checks_passed=False,
            )
        value.pop("reproducibility_checksum")
        value["reproducibility_checksum"] = canonical_hash(value)
        m.progress("after_terminal_checks")
    publish_primary(output, value, replay)
    m.progress("published_" + value["verdict_class"], completed, 2240 - completed)
    return 0

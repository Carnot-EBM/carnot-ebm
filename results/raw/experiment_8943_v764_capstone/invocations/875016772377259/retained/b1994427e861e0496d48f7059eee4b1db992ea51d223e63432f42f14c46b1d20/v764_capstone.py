"""REQ-REPORT-8943: reconcile evidence without turning completion into benefit."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from carnot.reporting import depth_control_8934 as depth
from carnot.reporting import compatible_replay_benefit_8941 as benefit
from carnot.reporting import replay_complete_cost_8942 as cost
from carnot.reporting import hardware_operation_runner_8825 as checks
from carnot.reporting import v764_capstone_sources as sources_module
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace
from carnot.reporting.v763_capstone import checksum
from carnot.reporting.v763_capstone_reduction import require, compare
from carnot.reporting.v764_capstone_sources import ROOT, PINS, admission, collect, progress
from scripts.experiment_template import ExperimentTemplate

__all__ = ["PINS", "admission", "collect", "checksum"]
Json = dict[str, Any]
NAME, TASK = "experiment_8943_v764_capstone", "exp8943-capstone"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/v764_capstone_sources.py",
    "python/carnot/reporting/v764_capstone.py",
    CLI,
]
TEST = "tests/python/test_v764_capstone_8943.py"
NEXT = {
    8930: "Require a new frozen full fourteen-task contract for any changed authority.",
    8931: "Retain final-answer graph labels; changed source design requires a new sealed split and depth protocol.",
    8932: "Retain historical pilot format and semantic errors separately; no prefix localization or independent detector evaluation.",
    8933: "Require a changed signature, representation or promotion mechanism and fresh development; preserve all ten bounds.",
    8934: "Retire unchanged k2/k3 source gate: 39/50 errors at k3 is not greater than42/50 at k2. Reopen only with separately authorized changed graph/depth methods and new paired controls.",
    8935: "Require qualified changed depth_source_ready_score=1 and signal_capture_ready_score=1 before separately authorized fresh shard A.",
    8936: "Require qualified changed source gate and separate scheduler task identity before separately authorized fresh shard B.",
    8937: "Methods qualification is development only; require authenticated disjoint real train/calibration/test corpus and class support.",
    8938: "Require two authenticated real160-graph shards and independent labels with all corpus support gates.",
    8939: "Require corpus_ready_score=1 and frozen held-out energy/logistic/prevalence comparisons. No detector benefit exists here.",
    8940: "Do not repeat unchanged replay streams; require a changed informative candidate and freshly sealed independent streams.",
    8941: "Retire unchanged signed-local-v1 TRAIN buffer32 two-gradient SGD delay8 guard and48-query schedule. New evidence must pass all ten simultaneous bounds.",
    8942: "Retain complete costs separately from quality; deployment requires compatible_benefit_score=1. Hardware requires parity plus measured transfer and removable fractions.",
    8943: "Retain unconditional accounting; new evidence requires a new authorized contract, never hidden branch retries.",
}


def reductions(evidence: dict[int, Json]) -> Json:
    """Use original reducers so every displayed comparison still follows original rows."""
    result: Json = dict(
        collection=dict(
            planned_graphs=320,
            observed_graphs=None,
            observed_errors=None,
            status="blocked",
            missing_reason="depth_source_ready_score=0; no collection executed",
        ),
        detector=dict(
            benefit=None,
            independent_test_graphs=None,
            status="blocked",
            missing_reason="no admitted collection corpus",
        ),
    )
    if 8934 in evidence and 8931 in evidence:
        p, v = evidence[8931], evidence[8934]
        protocol = {
            k: p[k] for k in ("graph_manifest", "feature_schema", "generator_settings", "bounds")
        }
        measured = depth.reduce(depth.ordered_units(protocol), v["rows"])
        compare(v, measured)
        result["depth"] = measured
        result["depth"]["rows"] = v["rows"]
    if 8932 in evidence:
        v = evidence[8932]
        result["historical_pilot"] = {
            k: v[k]
            for k in (
                "error_type_counts",
                "total_error_rate",
                "semantic_error_rate",
                "independent_count",
            )
        }
    if 8941 in evidence:
        measured = benefit.summarize(evidence[8941]["rows"])
        compare(evidence[8941], measured)
        result["learning"] = measured
        result["learning"]["rows"] = evidence[8941]["rows"]
    if 8942 in evidence:
        measured = cost.reduce(Path(evidence[8942]["raw_directory"]), "")
        compare(evidence[8942], measured)
        result["cost"] = {k: v for k, v in measured.items() if k != "numerical_training_calls"}
    return result


def build(work: Json, validation: Json) -> Json:
    """Fourteen accounted slots can be ready while science branches remain blocked."""
    tasks, slots = work["tasks"], work["slots"]
    require(
        canonical_hash(tasks) == sources_module.authority.PIN
        and [s["task_id"] for s in slots] == [t["id"] for t in tasks[:-1]],
        "ordered_slots",
    )
    owned = validation["passed"] and validation["coverage"].get("complete") is True
    ready = bool(owned and len(slots) == 13 and all(s["accounted"] for s in slots))
    verdict = "blocked" if owned else "disqualified"
    outcomes, branches, denominators, continuations, retirements = [], [], [], [], []
    evidence = {}
    for task, slot in zip(tasks, slots):
        n, ref = slot["experiment_id"], slot["reference"]
        producer = load(ref) if ref and n in PINS else {}
        if producer and slot["accounted"] and n != 8935:
            evidence[n] = producer
        prior = producer.get("verdict_class", "blocked")
        outcomes.append(
            dict(
                task_id=task["id"],
                experiment_id=n,
                disposition=slot["disposition"],
                accounted=slot["accounted"],
                verdict_class=prior,
                honest_verdict=producer.get("honest_verdict"),
                primary_reference=ref,
                scheduler_record=slot["scheduler_record"],
                gate_check_summary=slot["gate_check_summary"],
            )
        )
        assertions = {
            k: producer[k]
            for k in producer
            if k.endswith("_score")
            or k
            in (
                "honest_verdict",
                "verdict_class",
                "acceptance_gates",
                "required_checks_passed",
                "flagged_adversarial",
                "gate_check_summary",
            )
        }
        branches.append(
            dict(
                task_id=task["id"],
                original_assertions=assertions,
                verdict_class=prior,
                upstream_compute_reference=dict(
                    scope="opaque_original_producer_no_current_compute", producer_reference=ref
                ),
                gate_check_summary=slot["gate_check_summary"],
            )
        )
        denominators.append(
            dict(
                task_id=task["id"],
                producer_counts={
                    k: producer.get(k)
                    for k in (
                        "intended_count",
                        "completed_count",
                        "failed_count",
                        "censored_count",
                        "excluded_count",
                        "independent_count",
                    )
                },
                scope="50 paired control graphs;320 collection graphs planned only;24 learning streams reused by audit;24 separate cost worlds. Never sum branch denominators.",
            )
        )
        continuations.append(
            dict(
                task_id=task["id"],
                required_new_evidence=NEXT[n],
                activation_authorized=False,
                hidden_work_scheduled=False,
            )
        )
    outcomes.append(
        dict(
            task_id=TASK,
            experiment_id=8943,
            disposition="self_slot",
            accounted=True,
            verdict_class=verdict,
        )
    )
    continuations.append(
        dict(
            task_id=TASK,
            required_new_evidence=NEXT[8943],
            activation_authorized=False,
            hidden_work_scheduled=False,
        )
    )
    for task in tasks:
        for prior_failure in task["prior_failures"]:
            n = int(task["id"][3:7])
            retirements.append(
                dict(
                    task_id=task["id"],
                    prior_failure=prior_failure,
                    exact_scope=task["title"],
                    recommendation="retire_unchanged_mechanism"
                    if n in (8934, 8941)
                    else "retain_terminal_block_without_retry"
                    if n in (8935, 8936, 8938, 8939)
                    else "retain_qualified_scope_without_new_scientific_credit",
                    current_honest_verdict=next(
                        r.get("honest_verdict") for r in outcomes if r["experiment_id"] == n
                    ),
                    exact_prior_verdict_recurred=next(
                        r.get("honest_verdict") for r in outcomes if r["experiment_id"] == n
                    )
                    == prior_failure["verdict"],
                    retire_if_same_verdict=prior_failure["retire_if_same_verdict"],
                    reopen_requires=NEXT[n],
                    addressed_by=prior_failure["addressed_by"],
                    exclusion_manifest_edit_authorized=False,
                    broader_hypothesis_retired=False,
                )
            )
    progress("scientific_reductions_before", 0, 3)
    science = reductions(evidence)
    progress("scientific_reductions_after", 3, 0)
    rows = [
        dict(r, producer_experiment_id=n)
        for key, n in [("depth", 8934), ("learning", 8941), ("cost", 8942)]
        for r in science.get(key, {}).get("rows", [])
    ]
    duration = (work["ended_ns"] - work["started_ns"]) / 1e9
    receipt = build_current_work_receipt(
        run_id=str(work["started_ns"]),
        owner_pid=work.get("owner_pid", os.getpid()),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(operation="authenticate and reduce original evidence"),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=work["started_ns"],
        ended_monotonic_ns=work["ended_ns"],
        phase_spans=[dict(phase="reconciliation_and_owned_checks", start_s=0, end_s=duration)],
    )
    value = dict(
        experiment_id=8943,
        task_id=TASK,
        milestone="2026.11.764",
        run_date="20261109",
        honest_verdict="complete_" + verdict + "_v764_capstone",
        verdict_class=verdict,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="Constructed graph final-answer labels and synthetic continual streams; no natural-language generalization or verifier moat.",
        independent_generalization_score=0,
        duration_s=duration,
        phase_spans=receipt["phase_spans"],
        current_work_receipt=receipt,
        random_seed=7648943,
        preconditions_checked=dict(
            authenticated_before_reduction=True,
            generator_weights_frozen=True,
            arc_suspended=True,
            external_publication_authorized=False,
        ),
        source_artifact_hashes=[s["reference"] for s in slots if s["reference"]],
        code_config_hashes=work["code"],
        raw_shard_hashes=work["references"],
        cited_upstream_artifacts=[
            dict(s["reference"], scope="opaque_original_producer_no_current_compute")
            for s in slots
            if s["reference"]
        ],
        rows=rows,
        outcome_rows=outcomes,
        branch_conclusions=branches,
        independent_denominator_rows=denominators,
        continuation_rows=continuations,
        retirement_recommendations=retirements,
        scientific_reductions=science,
        closed_unchanged_scopes=[
            dict(
                scope="Exp8920 label-demand methods disqualified; Exp8923 skipped",
                reopen_requires="Repair failed owned adversarial acceptance, pass unchanged validators, then qualify label_demand_ready_score=1 before a separately authorized40-draw evaluation.",
            ),
            dict(
                scope="Exp8921 inline observer failed; Exp8927 skipped",
                reopen_requires="Changed observer must pass every simultaneous overhead upper bound<=0.05 on fresh development worlds before separately authorized complete request attribution.",
            ),
        ],
        intended_count=14,
        completed_count=sum(s["accounted"] for s in slots) + 1,
        failed_count=0,
        censored_count=sum(not s["accounted"] for s in slots),
        excluded_count=0,
        independent_count=0,
        gate_check_summary=[d for s in slots for d in s["gate_check_summary"]],
        acceptance_gates=dict(
            complete_fourteen_slot_accounting=ready,
            owned_checks=bool(owned),
            scientific_benefit=False,
        ),
        reconciliation_ready_score=int(ready),
        required_checks_passed=bool(owned),
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        validation_state=validation,
        work_reference=reference(Path(work["raw"]) / "work.json"),
        flagged_adversarial=False,
        paper_ready=False,
        paper_ready_scope="No separately authorized publication process; accounting only.",
        unmet_gates=[
            "qualified_changed_depth_source",
            "admitted320_graph_corpus",
            "held_out_detector_benefit",
            "ten_bound_future_learning_benefit",
            "independent_natural_language_generalization",
            "authorized_publication_process",
        ],
        repository_health=dict(
            scope="authenticated historical V763 debt; not rerun or counted as current owned checks",
            producer_reference=work["history_reference"],
            receipts=work["repository_health"],
        ),
        methodology_note="Unconditional fourteen-slot reconciliation; missing external work is terminal blocked. No step localization, ARC progress, hardware acceleration, roadmap activation or external publication. Costs and source gate success cannot establish benefit.",
    )
    value["field_principles"] = {
        k: "Original producer evidence and local denominators only; readiness is accounting, never benefit. Current compute excludes all upstream activity."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild from independently pinned producers, rejecting even rehashed claim drift."""
    value = json.loads(path.read_bytes())
    require(value["reproducibility_checksum"] == checksum(value), "checksum_drift")
    work = load(value["work_reference"])
    for i, ref in enumerate(work["references"] + work["code"]):
        require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_drift")
        if i % 500 == 0:
            progress("replay_authentication", i, len(work["references"]) + len(work["code"]) - i)
    for slot in work["slots"]:
        ref, n = slot["reference"], slot["experiment_id"]
        require(ref is not None or slot["accounted"] is False, "missing_primary_accounting")
        if ref and n in PINS:
            require(ref["sha256"] == "sha256:" + PINS[n][1], "producer_pin")
            producer = load(ref)
            require(
                slot["accounted"]
                == (n == 8935 or admission(producer, n) in (None, "disqualified_primary")),
                "accounting_drift",
            )
        if n in (8936, 8938, 8939):
            require(
                ref["sha256"] == sources_module.LOG_PIN
                and sha256_file(Path(ref["path"])) == sources_module.LOG_PIN
                and slot["scheduler_record"] in Path(ref["path"]).read_text().splitlines(),
                "scheduler_record_drift",
            )
    require(value == build(work, value["validation_state"]), "artifact_reduction_drift")
    return dict(passed=True, status="replay_passed", slots=14)


def run_checks(root: Path, commands: list[CommandSpec], **kwargs: Any) -> list[Json]:
    """Run all owned checks; historical global health remains separate authenticated debt."""
    return run_commands(root, [s for s in commands if s.scope != "repository_health"], **kwargs)


checker = namespace(checks, "capstone8943_checks")
checker.m = sys.modules[__name__]
checker.run_commands = run_checks


def main(argv: list[str] | None = None) -> int:
    """Publish checked bytes atomically; private runs test plumbing without readiness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261109"], default="20261109")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--sources", type=Path, default=ROOT / "results")
    parser.add_argument("--replay", "--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        try:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(json.dumps(dict(passed=False, reason=str(error))), flush=True)
            return 1
    output = (args.private_output or ROOT / "results" / (NAME + ".json")).absolute()
    raw = output.parent / "raw" / NAME / "invocations" / str(time.monotonic_ns())
    template = ExperimentTemplate(
        8943, "V764 fourteen-slot capstone", output.name, repo_root=ROOT, seed=7648943
    )
    with template.phase("authenticate_and_reduce"):
        work = collect(raw, sources=args.sources)
    work["owner_pid"] = os.getpid()
    for p in [
        *OWNED,
        TEST,
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
    ]:
        work["code"].append(sources_module.retain(ROOT / p, raw, sha256_file(ROOT / p)))
    progress("owned_validation_before", 0, 1)
    validation = dict(passed=False, coverage=dict(complete=False), receipts=[])
    if not args.private_output:
        validation = checker.validate(raw / "validation")
    progress("owned_validation_after", 1, 0)
    work["ended_ns"] = time.monotonic_ns()
    atomic_json(raw / "work.json", work)
    candidate = raw / "prospective.json"
    atomic_json(candidate, build(work, validation))
    if not args.private_output:
        progress("terminal_checks_before", 0, 3)
        terminal = checker.terminal(candidate, raw)
        validation["receipts"] += terminal["receipts"]
        validation["passed"] &= terminal["passed"]
        work["ended_ns"] = time.monotonic_ns()
        work["references"] += [
            reference(p)
            for directory in ("validation", "terminal")
            for p in (raw / directory).rglob("*")
            if p.is_file()
        ]
        atomic_json(raw / "work.json", work)
        progress("terminal_checks_after", 3, 0)
    require((time.monotonic_ns() - work["started_ns"]) / 1e9 < 4200, "fixed_deadline")
    publish_primary(output, build(work, validation), replay)
    progress("complete", 14, 0)
    return 0

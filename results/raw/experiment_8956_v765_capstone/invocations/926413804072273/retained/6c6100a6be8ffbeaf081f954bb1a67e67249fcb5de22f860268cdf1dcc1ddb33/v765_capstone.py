"""REQ-REPORT-8956: complete accounting cannot turn branch nulls into global benefit."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from carnot.reporting import hardware_operation_runner_8825 as checks
from carnot.reporting import v765_capstone_sources as s
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
from carnot.reporting.v763_capstone_reduction import require
from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]
ROOT = s.ROOT
NAME, TASK = "experiment_8956_v765_capstone", "exp8956-capstone"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/v765_capstone.py",
    "python/carnot/reporting/v765_capstone_sources.py",
    CLI,
]
TEST = "tests/python/test_v765_capstone_8956.py"
NEXT = {
    8944: "Retain immutable thirteen-task authority; any changed scope needs a new authorized contract.",
    8945: "Close unchanged unequal-opportunity depth failure. Repair quarantined provenance controls before admitting this diagnosis; no causal explanation is established.",
    8946: "Retain equal-cardinality source and frozen calibration methods; a new population needs fresh sealed roles and controls.",
    8947: "Require a changed gradient selection mechanism and fresh development before any new learning evaluation.",
    8948: "Retain the measured matched depth rise in its constructed graph scope; old unmatched depth failure stays closed.",
    8949: "Retain first-response shard A; no hidden retries or extra collection.",
    8950: "Retain first-response shard B; no hidden retries or extra collection.",
    8951: "Retain source-bound 320-graph admission and class support; this grants no detector benefit.",
    8952: "Close unchanged confidence-only calibrated selector on this population; changed features or mechanism need fresh frozen roles and all original bounds.",
    8953: "Close unchanged released-gradient selector, buffer32, two-gradient SGD, delay8, guard and48-query schedule; fresh streams require a changed informative mechanism.",
    8954: "Retire unchanged gradient-alignment replay in the exact TRAIN/retention scope. A changed mechanism must pass all11 simultaneous bounds including signature comparison.",
    8955: "Retain complete service costs; deployment still requires qualified gradient benefit and favorable costs. Hardware needs equivalent kernel parity and measured host, transfer and durability fractions.",
    8956: "Retain unconditional accounting; no automatic experiment, manifest change or external publication is authorized.",
}


def build(work: Json, validation: Json) -> Json:
    """Keep scientific scope and upstream failures separate from owned qualification."""
    tasks, slots = work["tasks"], work["slots"]
    require(
        canonical_hash(tasks) == s.authority.PIN
        and [r["task_id"] for r in slots] == [t["id"] for t in tasks[:-1]],
        "ordered_slots",
    )
    owned = bool(validation["passed"] and validation["coverage"].get("complete") is True)
    ready = owned and len(slots) == 12 and all(r["accounted"] and r["qualified"] for r in slots)
    verdict = "disqualified" if not owned else "null" if ready else "blocked"
    outcomes, branches, denominators, continuations, rows = [], [], [], [], []
    for task, slot in zip(tasks, slots):
        n, ref = slot["experiment_id"], slot["reference"]
        producer = load(ref) if ref else {}
        outcomes.append(
            dict(
                slot,
                verdict_class=producer.get("verdict_class", "blocked"),
                honest_verdict=producer.get("honest_verdict"),
                verifier_is_oracle=producer.get("verifier_is_oracle"),
                flagged_adversarial=producer.get("flagged_adversarial"),
            )
        )
        assertions = {
            k: v
            for k, v in producer.items()
            if k.endswith("_score")
            or k
            in (
                "honest_verdict",
                "verdict_class",
                "flagged_adversarial",
                "verifier_is_oracle",
                "acceptance_gates",
                "required_checks_passed",
                "gate_check_summary",
            )
        }
        branches.append(
            dict(
                task_id=task["id"],
                original_assertions=assertions,
                admitted_to_scientific_reductions=slot["qualified"],
                upstream_compute_reference=dict(
                    scope="opaque_original_producer_no_current_compute", reference=ref
                ),
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
                scope="50 paired depth graphs;320 corpus graphs reused by admission and80 held-out calibration graphs;24 learning streams reused by audit;24 separate cost worlds. Never add these denominators.",
            )
        )
        if (
            slot["qualified"]
            and slot["disposition"] == "qualified_primary"
            and n in (8948, 8949, 8950, 8952, 8954, 8955)
        ):
            rows.extend(
                dict(
                    r,
                    producer_experiment_id=n,
                    condition=r.get("condition", r.get("split", "original_producer_condition")),
                    censor_reason=r.get("censor_reason", r.get("missing_reason")),
                )
                for r in producer["rows"]
            )
    outcomes.append(
        dict(
            task_id=TASK,
            experiment_id=8956,
            disposition="self_slot",
            accounted=True,
            qualified=owned,
            verdict_class=verdict,
            honest_verdict=f"complete_{verdict}_v765_capstone",
        )
    )
    for task, outcome in zip(tasks, outcomes, strict=True):
        n = int(task["id"][3:7])
        continuations.append(
            dict(
                task_id=task["id"],
                required_new_evidence=NEXT[n],
                activation_authorized=False,
                hidden_work_scheduled=False,
            )
        )
        rows.append(
            dict(
                unit_id=task["id"],
                cluster_id="administrative_contract",
                arm="reconciliation",
                condition="task_accounting",
                metric="task_slot_accounted",
                value=int(outcome["accounted"]),
                numerator=int(outcome["accounted"]),
                denominator=1,
                status="completed" if outcome["accounted"] else "censored",
                censor_reason=None if outcome["accounted"] else "external_evidence",
                independent_count=0,
            )
        )
    retirements = [
        dict(
            task_id=t["id"],
            prior_failure=p,
            current_honest_verdict=o.get("honest_verdict"),
            exact_prior_verdict_recurred=o.get("honest_verdict") == p["verdict"],
            recommendation="close_unchanged_exact_scope"
            if int(t["id"][3:7]) in (8945, 8952, 8953, 8954)
            else "retain_original_scope_and_disposition",
            reopen_requires=NEXT[int(t["id"][3:7])],
            broader_hypothesis_retired=False,
            exclusion_manifest_edit_authorized=False,
        )
        for t, o in zip(tasks, outcomes, strict=True)
        for p in t["prior_failures"]
    ]
    duration = (work["ended_ns"] - work["started_ns"]) / 1e9
    receipt = build_current_work_receipt(
        run_id=str(work["started_ns"]),
        owner_pid=work.get("owner_pid", os.getpid()),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(
            operation="authenticate and reduce retained evidence only"
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=work["started_ns"],
        ended_monotonic_ns=work["ended_ns"],
        phase_spans=[dict(phase="reconciliation_and_owned_checks", start_s=0, end_s=duration)],
    )
    gate = work["publication_gate"]
    value = dict(
        experiment_id=8956,
        task_id=TASK,
        milestone="2026.11.765",
        run_date="20261109",
        random_seed=7658956,
        honest_verdict=f"complete_{verdict}_v765_capstone",
        verdict_class=verdict,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        current_work_receipt=receipt,
        duration_s=duration,
        phase_spans=receipt["phase_spans"],
        verifier_is_oracle=True,
        exposure_scope="Constructed graph final answers and synthetic continual streams only.",
        independent_generalization_score=0,
        preconditions_checked=dict(
            authenticated_before_reduction=True,
            generator_weights_frozen=True,
            arc_suspended=True,
            no_science_rerun=True,
            external_publication_authorized=False,
        ),
        intended_count=13,
        completed_count=sum(r["accounted"] for r in outcomes),
        failed_count=0,
        censored_count=sum(not r["accounted"] for r in outcomes),
        excluded_count=0,
        independent_count=0,
        rows=rows,
        outcome_rows=outcomes,
        branch_conclusions=branches,
        independent_denominator_rows=denominators,
        continuation_rows=continuations,
        retirement_recommendations=retirements,
        scientific_reductions=work["scientific_reductions"],
        closed_unchanged_scopes=[
            dict(scope=NEXT[n], experiment_id=n) for n in (8945, 8952, 8953, 8954)
        ],
        open_research_claims=[
            "natural_language_verification",
            "actual_generative_self_learning",
            "10x_Rust_throughput",
            "100x_hardware_acceleration",
        ],
        arc_status="suspended",
        gate_check_summary=[c for r in slots for c in r["gate_check_summary"]],
        acceptance_gates=dict(
            complete_thirteen_slot_accounting=all(r["accounted"] for r in outcomes),
            upstream_evidence_qualified=all(r["qualified"] for r in slots),
            owned_checks=owned,
            scientific_benefit=False,
        ),
        reconciliation_ready_score=int(ready),
        required_checks_passed=owned,
        validation_receipts=validation["receipts"] + work["publication_receipts"],
        owned_statement_coverage=validation["coverage"],
        validation_state=validation,
        global_test_debt=dict(
            current=work.get("global_test_debt", []),
            inherited=work["inherited_global_test_debt"],
            scope="Global repository health is separate from owned qualification.",
        ),
        source_artifact_hashes=[r["reference"] for r in slots if r["reference"]],
        code_config_hashes=work["code"],
        raw_shard_hashes=work["references"],
        work_reference=reference(Path(work["raw"]) / "work.json"),
        flagged_adversarial=False,
        g1=gate["gates"]["G1"],
        g2=gate["gates"]["G2"],
        g3=gate["gates"]["G3"],
        g4=gate["gates"]["G4"],
        paper_ready=gate["paper_ready"],
        unmet_gates=gate["unmet_gates"],
        paper_ready_scope="FoVer paper only; stable G1-G4 do not certify V765 findings or authorize publication.",
        methodology_note="Unconditional13-slot reconciliation. Original classes, oracle provenance and adversarial flags remain inherited. Quarantined producers cannot establish science readiness. Costs do not establish benefit; no model, learner, benchmark or replacement experiment runs here.",
        reproduction_commands=[
            f"PYTHONPATH=python .venv/bin/python -u {CLI} --date 20261109",
            f".venv/bin/python -u {CLI} --cold-replay <primary-path>",
        ],
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Pinned retained rows and local denominators only. Current compute is zero; administrative completeness and FoVer readiness grant no V765 scientific credit.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild claims from pinned producers so rehashing cannot bless altered conclusions."""
    value = json.loads(path.read_bytes())
    require(value["reproducibility_checksum"] == checksum(value), "checksum_drift")
    work = load(value["work_reference"])
    for index, ref in enumerate(work["references"] + work["code"]):
        require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_drift")
        if index % 500 == 0:
            s.progress(
                "replay_authentication", index, len(work["references"]) + len(work["code"]) - index
            )
    evidence = {}
    for task, slot in zip(work["tasks"], work["slots"]):
        ref, n = slot["reference"], slot["experiment_id"]
        require(ref is not None or not slot["accounted"], "missing_primary_accounting")
        if ref:
            require(ref["sha256"] == s.PINS[n][1], "producer_pin")
            producer = load(ref)
            if producer.get("blocked_at_layer") == "conductor_pre_gate":
                upstream = int(producer["failed_upstream"][3:7])
                upstream_ref = next(
                    r["reference"] for r in work["slots"] if r["experiment_id"] == upstream
                )
                check = s.scheduler_check(
                    producer, task, {upstream: load(upstream_ref)}, {upstream: upstream_ref}
                )
                require(
                    slot["qualified"] and slot["gate_check_summary"] == [check],
                    "scheduler_receipt_drift",
                )
            else:
                require(
                    slot["qualified"] == (s.admission(producer, task) is None),
                    "qualification_drift",
                )
                if slot["qualified"]:
                    evidence[n] = producer
    require(value == build(work, value["validation_state"]), "artifact_reduction_drift")
    s.progress("replay_reductions_before", 0, 5)
    require(work["scientific_reductions"] == s.reductions(evidence), "scientific_reduction_drift")
    return dict(passed=True, status="replay_passed", slots=13)


def run_checks(root: Path, commands: list[CommandSpec], **kwargs: Any) -> list[Json]:
    """The single full-suite attempt is retained separately; all owned checks run fresh."""
    return run_commands(root, [c for c in commands if c.scope != "repository_health"], **kwargs)


checker = namespace(checks, "capstone8956_checks")
checker.m = sys.modules[__name__]
checker.run_commands = run_checks


def main(argv: list[str] | None = None) -> int:
    """Only checked candidate bytes reach the primary; private fixtures cannot claim readiness."""
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
    if args.private_output and output.is_relative_to(s.ROOT):
        parser.error("private output requires scratch outside the checkout")
    raw = output.parent / "raw" / NAME / "invocations" / str(time.monotonic_ns())
    template = ExperimentTemplate(
        8956, "V765 thirteen-slot capstone", output.name, repo_root=ROOT, seed=7658956
    )
    with template.phase("authenticate_and_reduce"):
        work = s.collect(raw, sources=args.sources)
    work["owner_pid"] = os.getpid()
    for name in [
        *OWNED,
        TEST,
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
    ]:
        work["code"].append(s.retain(ROOT / name, raw, sha256_file(ROOT / name)))
    health = os.environ.get("CARNOT_V765_GLOBAL_HEALTH_RECEIPT")
    if health:
        ref = s.retain(Path(health), raw, sha256_file(Path(health)))
        work["references"].append(ref)
        work["global_test_debt"] = json.loads(Path(ref["path"]).read_bytes())
        for receipt in work["global_test_debt"]:
            work["references"].append(
                s.retain(Path(receipt["log_path"]), raw, receipt["log_sha256"])
            )
    s.progress("owned_validation_before", 0, 1)
    validation = dict(passed=False, coverage=dict(complete=False), receipts=[])
    if not args.private_output:
        validation = checker.validate(raw / "validation")
    s.progress("owned_validation_after", 1, 0)
    work["ended_ns"] = time.monotonic_ns()
    atomic_json(raw / "work.json", work)
    candidate = raw / "prospective.json"
    atomic_json(candidate, build(work, validation))
    if not args.private_output:
        s.progress("terminal_checks_before", 0, 3)
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
        s.progress("terminal_checks_after", 3, 0)
    require((time.monotonic_ns() - work["started_ns"]) / 1e9 < 4200, "fixed_deadline")
    require(args.private_output is not None or validation["passed"], "owned_validation_failed")
    publish_primary(output, build(work, validation), replay)
    s.progress("complete", 13, 0)
    return 0

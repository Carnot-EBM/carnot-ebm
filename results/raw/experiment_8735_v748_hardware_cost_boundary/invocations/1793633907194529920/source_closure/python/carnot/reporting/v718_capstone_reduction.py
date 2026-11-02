"""REQ-REPORT-8331: separate administrative validity from unavailable science.

Each branch keeps its frozen denominator and its own next observation. Imported
failures describe earlier work; only a failed current owned check disqualifies
this invocation. No cached or constructed result receives generalization credit.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting import v718_capstone_evidence as e
from carnot.reporting import v716_capstone_evidence as reader
from carnot.reporting import v717_capstone_evidence as old
from carnot.reporting import v718_replay_history as replay_history
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v710_contract_replay import require_reference

Json = dict[str, Any]
NEXT = [
    "Pass deterministic full-object authority and mixed-history replay on unchanged primitives.",
    "Qualify sparse/dense primitive parity and reject false-zero evidence under Exp8318 policy.",
    "Produce frozen equal-information static heads and fit-only action controls with original support.",
    "Seal all 128 reserved predictions before opening evaluator labels.",
    "Seal 96 delayed-feedback stream slots with real crash32/64 and exact dense/sparse parity.",
    "Seal capacity invariants and matched feedback-loss controls under the separate capacity protocol.",
    "Measure H1 on >=80 sources and >=8 per class: mean>=.02, nominal lower bound>0, Brier degradation<=.01.",
    "Measure H2 on >=64 later sources and >=8 per class with all four fixed retention windows and final bounds.",
    "Qualify an explicit V716 authority reader; authenticate a changed runtime before another GPU probe.",
    "Authenticate context/copy parity and changed runtime before one bounded canary; never reopen V713 capture.",
    "Authenticate new ARC outcomes with >=3 overlapping games and >=5 firings per shared arm cell.",
    "Measure complete CPU costs and a compatible k<=5 SSH KV260 operation with input/output parity.",
    "Authenticate dated physical change, GM1Ax IDCODE0x20000001, then n16 flash and sample/hash smoke.",
    "Supply qualified sealed H1/H2 primitives before another capstone science decision.",
]


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Reconstruct every disposition; never substitute an audit summary for a seal."""
    for ref in work["references"]:
        require_reference(ref)
    checked = e.authority(work)
    tasks = checked.get("tasks", work["tasks"])
    if tasks != work["tasks"] or len(work["inputs"]) != 13:
        raise ValueError("contract_primitive_drift")
    failures = deepcopy(work["failures"])
    protocol_ok = work["references"][3]["sha256"] == e.PROTOCOL_PIN
    if not protocol_ok:
        failures.append(
            e.failure(
                Path(work["root"]) / e.PROTOCOL,
                "protocol_sha256",
                e.PROTOCOL_PIN,
                work["references"][3]["sha256"],
            )
        )
    rows, sources = [], {}
    for identity, task, item in zip(range(8318, 8331), tasks[:-1], work["inputs"], strict=True):
        if item["reference"] not in work["references"]:
            raise ValueError("input_reference_drift")
        row, failed, source = reader.outcome(task, identity, item)
        if row["missing"] or row["disposition"] == "conductor_pre_gate":
            row["honest_verdict"] = None
        if row["disposition"] == "conductor_pre_gate":
            for gate in old.read(item["reference"])["gates_evaluated"]:
                bound: Json = next(
                    (r for r in work["references"] if r["path"] == gate["artifact_path"]), {}
                )
                if bound.get("sha256") != gate["artifact_sha256"]:
                    row.update(missing=True, disposition="unbound_pre_gate", numerator=0)
                    failed.append(
                        e.failure(
                            Path(item["reference"]["path"]),
                            "conductor_receipt_bound_sha256",
                            gate["artifact_sha256"],
                            bound.get("sha256"),
                        )
                    )
        rows.append(row)
        sources[identity] = source
        failures.extend(failed)
    for ref in work["references"]:
        if ref.get("expected_sha256") and ref["sha256"] != ref["expected_sha256"]:
            failures.append(
                e.failure(
                    Path(ref["path"]), "primitive_sha256", ref["expected_sha256"], ref["sha256"]
                )
            )
    historical = {}
    if sources[8318].get("work_reference"):
        operand = next(
            r for r in work["references"] if r["path"] == sources[8318]["work_reference"]["path"]
        )
        history = json.loads(Path(operand["snapshot_path"]).read_bytes()).get("history", {})
        if history:
            require_reference(history["work_reference"])
            primitive = json.loads(Path(history["work_reference"]["snapshot_path"]).read_bytes())
            historical = replay_history.restore_locations(
                old.reduce(primitive, history["historical_receipts"]), history["relocation_map"]
            )
            if canonical_hash(historical) != history["reduction_sha256"]:
                raise ValueError("historical_reduction_drift")
    owned = [r for r in receipts + work["audits"] if r.get("scope") == "owned"]
    passed = (
        bool(owned)
        and all(r["passed"] and r.get("normal_exit", True) for r in owned)
        and not any(r["disposition"] == "owned_reader_exception" for r in rows)
    )
    ready = int(
        passed
        and checked["activated"]
        and protocol_ok
        and work.get("owned_validation_complete", True)
    )
    resource_blocked = (
        bool(work["failures"])
        and not work.get("owned_validation_complete", True)
        and all(r["passed"] for r in owned)
    )
    kind = "blocked" if passed or resource_blocked else "disqualified"
    verdict = (
        "complete_"
        + kind
        + (
            "_" + work["failures"][0]["upstream"]
            if resource_blocked
            else "_upstream_science"
            if passed
            else "_owned_validation"
        )
    )
    for r in receipts + work["audits"]:
        if not r["passed"]:
            failures.append(
                e.failure(
                    Path(r["stdout_path"]), "normal_exit", r["expected_exit"], r["actual_exit"]
                )
            )
    for identity in [8321, 8322, 8324, 8325]:
        failures.append(
            e.failure(
                Path(work["root"]) / tasks[identity - 8318]["deliverable"],
                "sealed_primitive_science_rows",
                "complete independently replayable V717 protocol rows",
                None,
            )
        )
    rows.append(
        dict(
            experiment_id=8331,
            task_id=e.TASK,
            unit_id=e.TASK,
            arm="task_disposition",
            condition="terminal_accounting",
            metric="owned_execution_readiness",
            numerator=ready,
            denominator=1,
            status="completed",
            completed=True,
            missing=False,
            producer_executed=True,
            eligible=bool(ready),
            excluded=not ready,
            failed=kind == "disqualified",
            censored=False,
            honest_verdict=verdict,
            verdict_class=kind,
            disposition="self_owned_completion",
            evidence_type="current_capstone",
        )
    )
    retirements = []
    for task, row, condition in zip(tasks, rows, NEXT, strict=True):
        matched = [
            p
            for p in task["prior_failures"]
            if p["retire_if_same_verdict"] and p["verdict"] == row["honest_verdict"]
        ]
        evidence = []
        for prior in matched:
            item = work["history"].get(prior["experiment_id"])
            source = old.read(item["reference"]) if item else {}
            evidence.append(
                dict(
                    authenticated=bool(
                        source.get("task_id") == prior["experiment_id"]
                        and source.get("honest_verdict") == prior["verdict"]
                        and item
                        and not item["sidecar"].get("binding_error")
                    ),
                    references=[item["reference"]] if item else [],
                )
            )
        retirements.append(
            dict(
                task_id=task["id"],
                predecessors=deepcopy(task["prior_failures"]),
                same_verdict_entries=matched,
                prior_evidence=evidence,
                decision="retire_exact_repeated_scope" if matched else "await_eligible_evidence",
                permanent=bool(matched),
                scope="unchanged evidence/probe only; unmeasured scientific hypothesis remains open",
                reopening_condition=condition,
            )
        )
    arms = ["spline34", "RBF34", "linear6", "scalar2", "holistic_probability"]
    h1 = dict(
        status="blocked_unmeasured",
        intended_count=128,
        completed_count=None,
        statistics=None,
        alpha=0.025,
        interval_scope="descriptive exposed development only",
        arm_comparisons=[
            dict(
                arm=a,
                comparator="spline34",
                intended_count=128,
                paired_cost_gain=None,
                brier_degradation=None,
            )
            for a in arms[1:]
        ],
        support=None,
        typed_action_control=None,
        oracle_headroom=None,
    )
    h2 = dict(
        status="blocked_unmeasured",
        intended_count=88,
        stream_intended_count=96,
        retention_intended_count=32,
        retention_windows=[0, 32, 64, 96],
        completed_count=None,
        statistics=None,
        alpha=0.025,
        arm_comparisons=[
            dict(arm=a, comparator="frozen", intended_count=88, paired_cost_gain=None)
            for a in ["local_sparse", "local_dense", "calibration_only", "released_label_shuffle"]
        ],
        retention=[
            dict(window=w, intended_count=32, cost_degradation=None, brier_degradation=None)
            for w in [0, 32, 64, 96]
        ],
        support=None,
        action_control=None,
        oracle_headroom=None,
    )
    capacity = dict(
        status="blocked_unmeasured",
        experiment_id=8323,
        included_in_H1_H2=False,
        pending_capacity=None,
        memory_invariants=None,
        feedback_loss_controls=None,
        natural_benefit=False,
        regret_guarantee=False,
    )
    return dict(
        honest_verdict=verdict,
        verdict_class=kind,
        rows=rows,
        task_dispositions=deepcopy(rows),
        intended_count=14,
        completed_count=14,
        failed_count=sum(r["failed"] for r in rows),
        censored_count=0,
        excluded_count=sum(r["excluded"] for r in rows),
        independent_count=0,
        actual_executed_task_count=sum(r["producer_executed"] for r in rows),
        pre_gate_count=sum(r["disposition"] == "conductor_pre_gate" for r in rows),
        missing_output_count=sum(r["missing"] for r in rows),
        required_checks_passed=passed and work.get("owned_validation_complete", True),
        flagged_adversarial=False,
        capstone_execution_ready_score=ready,
        current_contract_ready_score=ready,
        science_ready_score=0,
        h1_development_signal_score=0,
        h2_development_signal_score=0,
        static_ready_score=int(
            rows[2]["eligible"] and sources[8320].get("static_ready_score") == 1
        ),
        durable_local_correctness_score=int(
            rows[1]["eligible"] and sources[8319].get("local_kernel_ready_score") == 1
        ),
        actual_later_source_improvement_score=0,
        H1=h1,
        H2=h2,
        capacity_scope=capacity,
        verifier_is_oracle=False,
        exposure_scope="exposed_cached_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        gate_check_summary=failures + checked["gate_check_summary"],
        task_contract=tasks,
        canonical_tasks_sha256=canonical_hash(tasks)[7:],
        full_task_authority_equal=checked["activated"],
        branch_replay_receipts=work["audits"],
        historical_v717=historical,
        historical_model_provenance=sources[8318].get("historical_model_provenance", []),
        historical_outcomes_preserved=True,
        board_obligations=[
            dict(
                board=b,
                current=sources[i].get(key),
                terminal_scope="board-local Linux CPU only"
                if b == "PolarFire"
                else "unmet obligation",
                next_evidence_condition=condition,
            )
            for b, i, key, condition in [
                (
                    "PolarFire",
                    8329,
                    "polarfire_graduation",
                    "Preserve Exp8259 authenticated CPU dispatch/output parity; no fabric claim.",
                ),
                ("KV260", 8329, "kv260_obligation", NEXT[11]),
                ("GateMate", 8330, "gatemate_obligation", NEXT[12]),
            ]
        ],
        arc_support=sources[8328],
        live_call_accounting=dict(
            current_capstone_calls=0,
            canary=sources[8327].get("model_invocation_counts"),
            imported_calls_are_current_execution=False,
            canary_is_H1=False,
            V713_capture_reopened=False,
        ),
        retirements=retirements,
        next_evidence_conditions=NEXT,
        three_prd_gaps=[
            dict(gap=g, requirements=req, closed=False, next_evidence_condition=c)
            for g, req, c in [
                ("useful_verified_decisions", ["FR-06", "FR-12"], NEXT[6]),
                ("later_learning_and_retention", ["FR-11"], NEXT[7]),
                ("request_scale_deployment", ["FR-05", "FR-08", "NFR-01"], NEXT[11]),
            ]
        ],
        continuation_decision="not_earned_unmeasured; continue only after qualified sealed science operands",
        acceptance_gates=dict(
            owned_validation=passed,
            full_task_authority=checked["activated"],
            independent_science=False,
        ),
        sample_size_budget=dict(
            task_slots=14,
            independent_scientific_observations=0,
            H1=128,
            H2_later=88,
            stream=96,
            retention=32,
        ),
    )

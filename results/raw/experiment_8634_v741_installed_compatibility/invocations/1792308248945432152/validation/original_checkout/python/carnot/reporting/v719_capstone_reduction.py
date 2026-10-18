"""REQ-REPORT-8345: administrative completion is separate from scientific utility."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting import v719_capstone_evidence as e
from carnot.reporting.v718_capstone_reduction import NEXT
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v710_contract_replay import require_reference

Json = dict[str, Any]


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Rebuild fourteen dispositions from authenticated compact source reductions.

    Absent science preserves its registered denominators. The audit tasks are
    dependencies for an eventual sealed study, not measurements of negative benefit.
    """
    for ref in work["references"]:
        require_reference(ref)
    checked = e.authority(work)
    if checked["tasks"] != work["tasks"] or len(work["inputs"]) != 13:
        raise ValueError("contract_primitive_drift")
    rows, failures = [], list(work["failures"]) + checked["gate_check_summary"]
    for task, item in zip(work["tasks"][:-1], work["inputs"], strict=True):
        if (
            item["reference"] not in work["references"]
            or canonical_hash(item["summary"]) != item["summary_sha256"]
        ):
            raise ValueError("compact_primitive_drift")
        row = deepcopy(item["summary"]["row"])
        for gate in item["summary"]["conductor_gates"]:
            bound: Json = next(
                (r for r in work["references"] if r["path"] == gate["artifact_path"]), {}
            )
            if bound.get("sha256") != gate["artifact_sha256"]:
                row.update(missing=True, disposition="unbound_pre_gate", numerator=0)
        rows.append(row)
        failures.extend(item["summary"]["failures"])
    for ref in work["references"]:
        if "expected_sha256" in ref and ref["sha256"] != ref["expected_sha256"]:
            failures.append(
                e.failure(
                    Path(ref["path"]), "primitive_sha256", ref["expected_sha256"], ref["sha256"]
                )
            )
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
    owned = [r for r in receipts + work["audits"] if r.get("scope") == "owned"]
    measured = work["memory_measurements"]
    memory_ok = (
        measured["parent_growth_mb"] <= 500
        and measured["retained_payload_bytes"] <= 8_000_000
        and all(r["passed"] for r in measured["workers"])
        and measured.get("owned_parent_after", measured["parent_after"])["peak_rss_mb"]
        - measured["parent_before"]["peak_rss_mb"]
        <= 500
        and all(r["memory_passed"] for r in measured.get("historical_workers", []))
    )
    passed = (
        bool(receipts)
        and bool(owned)
        and all(r["passed"] and r.get("normal_exit", True) for r in owned)
        and memory_ok
        and work.get("repository_suite_attempt", {"passed": True})["passed"]
    )
    ready = int(
        passed
        and checked["activated"]
        and protocol_ok
        and work.get("owned_validation_complete", True)
    )
    kind = "blocked" if passed or (work["failures"] and not owned) else "disqualified"
    verdict = (
        "complete_"
        + kind
        + (
            "_upstream_science"
            if passed
            else "_" + work["failures"][0]["upstream"]
            if kind == "blocked"
            else "_owned_validation"
        )
    )
    for receipt in owned:
        if not receipt["passed"]:
            failures.append(
                e.failure(
                    Path(receipt["stdout_path"]),
                    "normal_exit",
                    receipt["expected_exit"],
                    receipt["actual_exit"],
                )
            )
    if not memory_ok:
        failures.append(e.failure(Path(work["root"]), "memory_growth_mb", "<=500", measured))
    for index in [3, 4, 6, 7]:
        failures.append(
            e.failure(
                Path(work["root"]) / work["tasks"][index]["deliverable"],
                "sealed_primitive_science_rows",
                "complete independently replayable V717 protocol rows",
                None,
            )
        )
    if work.get("repository_suite_attempt") and not work["repository_suite_attempt"]["passed"]:
        suite = work["repository_suite_attempt"]
        failures.append(
            e.failure(
                Path(suite.get("stdout_path", work["root"])),
                "repository_suite_normal_exit",
                0,
                suite["actual_exit"],
            )
        )
    failures = [dict(g, artifact_field=g.get("artifact_field", g.get("field"))) for g in failures]
    rows.append(
        dict(
            experiment_id=8345,
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
    for task, row, condition in zip(work["tasks"], rows, NEXT, strict=True):
        matched = [
            p
            for p in task["prior_failures"]
            if p["retire_if_same_verdict"] and p["verdict"] == row["honest_verdict"]
        ]
        prior_evidence = []
        for prior in matched:
            item = work["history"].get(prior["experiment_id"])
            source = e.read(item["reference"]) if item else {}
            prior_evidence.append(
                dict(
                    authenticated=bool(
                        item
                        and source.get("task_id") == prior["experiment_id"]
                        and source.get("honest_verdict") == prior["verdict"]
                        and not item["sidecar"].get("binding_error")
                    ),
                    references=[item["reference"]] if item else [],
                )
            )
        retirements.append(
            dict(
                task_id=task["id"],
                predecessors=task["prior_failures"],
                same_verdict_entries=matched,
                prior_evidence=prior_evidence,
                permanent=bool(matched),
                scope="unchanged probe only; unmeasured hypotheses remain open",
                decision="retire_exact_repeated_scope" if matched else "await_eligible_evidence",
                reopening_condition=condition,
            )
        )
    h1 = dict(
        status="blocked_unmeasured",
        intended_count=128,
        completed_count=None,
        statistics=None,
        alpha=0.025,
        interval_scope="descriptive exposed development only",
        support=None,
        typed_action_control=None,
        oracle_headroom=None,
        arm_comparisons=[
            dict(
                arm=a,
                comparator="spline34",
                intended_count=128,
                paired_cost_gain=None,
                brier_degradation=None,
            )
            for a in ["RBF34", "linear6", "scalar2", "holistic_probability"]
        ],
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
        support=None,
        action_control=None,
        oracle_headroom=None,
        arm_comparisons=[
            dict(arm=a, comparator="frozen", intended_count=88, paired_cost_gain=None)
            for a in ["local_sparse", "local_dense", "calibration_only", "released_label_shuffle"]
        ],
        retention=[
            dict(window=w, intended_count=32, cost_degradation=None, brier_degradation=None)
            for w in [0, 32, 64, 96]
        ],
    )
    selected = [i["summary"]["selected"] for i in work["inputs"]]
    h1["sealed_predictions"] = dict(
        qualified=bool(rows[3]["eligible"] and selected[3].get("predictions_ready_score") == 1),
        completed_source_count=rows[3]["source_counts"]["completed_count"],
        intended_source_count=128,
        arm_count=selected[3].get("sample_size_budget", {}).get("arms"),
        arms=[
            "spline34",
            "RBF34",
            "linear6",
            "scalar2",
            "frozen_holistic_probability",
            "sigmoid34",
        ],
        evaluator_access_count=sum(
            r["evaluator_access_count"] for r in selected[3].get("label_access_ledger", [])
        )
        if selected[3].get("label_access_ledger")
        else None,
        independent_parity=selected[3].get("prediction_parity"),
        reference=work["inputs"][3]["reference"],
        purpose="Sealed predictions are available; evaluator rows and protocol scientific controls remain a separate blocked audit dependency.",
    )
    return dict(
        honest_verdict=verdict,
        verdict_class=kind,
        rows=rows,
        task_dispositions=rows,
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
            rows[2]["eligible"]
            and selected[2].get("heads_ready_score", selected[2].get("static_ready_score")) == 1
        ),
        durable_local_correctness_score=int(
            rows[1]["eligible"] and selected[1].get("local_kernel_ready_score") == 1
        ),
        actual_later_source_improvement_score=0,
        H1=h1,
        H2=h2,
        capacity_scope=dict(
            status="blocked_unmeasured",
            experiment_id=8337,
            included_in_H1_H2=False,
            memory_invariants=None,
            feedback_loss_controls=None,
            natural_benefit=False,
            regret_guarantee=False,
        ),
        verifier_is_oracle=False,
        exposure_scope="exposed_cached_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        gate_check_summary=failures,
        task_contract=work["tasks"],
        canonical_tasks_sha256=canonical_hash(work["tasks"])[7:],
        full_task_authority_equal=checked["activated"],
        branch_replay_receipts=work["audits"],
        historical_v717=dict(
            disposition="historical_synthesis_only",
            references=work["historical_fixture_hashes"][:2],
        ),
        historical_model_provenance=selected[0].get("historical_model_provenance", []),
        historical_outcomes_preserved=True,
        board_obligations=[
            dict(
                board=b,
                current=selected[i].get(key),
                terminal_scope=scope,
                next_evidence_condition=condition,
            )
            for b, i, key, scope, condition in [
                (
                    "PolarFire",
                    11,
                    "polarfire_graduation",
                    "board-local Linux CPU only",
                    "Preserve Exp8259 CPU parity; no fabric claim.",
                ),
                ("KV260", 11, "kv260_obligation", "quadratic Ising k<=5 only", NEXT[11]),
                ("GateMate", 12, "gatemate_obligation", "physical change required", NEXT[12]),
            ]
        ],
        arc_support=dict(
            reference=work["inputs"][10]["reference"], disposition=rows[10]["disposition"]
        ),
        live_call_accounting=dict(
            current_capstone_calls=0,
            canary=selected[9].get("model_invocation_counts"),
            imported_calls_are_current_execution=False,
            canary_is_H1=False,
            V713_capture_reopened=False,
        ),
        retirements=retirements,
        next_evidence_conditions=NEXT,
        three_prd_gaps=[
            dict(gap=g, requirements=req, closed=False, next_evidence_condition=NEXT[i])
            for g, req, i in [
                ("useful_verified_decisions", ["FR-06", "FR-12"], 6),
                ("later_learning_and_retention", ["FR-11"], 7),
                ("request_scale_deployment", ["FR-05", "FR-08", "NFR-01"], 11),
            ]
        ],
        continuation_decision="not_earned_unmeasured; require qualified sealed science operands",
        acceptance_gates=dict(
            owned_validation=passed,
            full_task_authority=checked["activated"],
            independent_science=False,
            memory=memory_ok,
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

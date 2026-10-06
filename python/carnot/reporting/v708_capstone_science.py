"""REQ-VERIFY-8204: cached source reductions retain their original claim limits.

A completed source null is usable evidence. Missing learning and natural demand
stay blocked, and exposed development cannot close the general verifier moat.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.v685_authority_lifecycle import tasks_digest
from carnot.reporting.v703_capstone_inputs import deduplicate

Json = dict[str, Any]


def primitive(value: Json, number: int) -> Json:
    """Reopen frozen primitives and compare every reconstructed scientific field."""
    if number == 8197 and value.get("measurement_reference"):
        from carnot.verify.selective_decision_audit_8197 import reduce as reducer

        ref = value["measurement_reference"]
        result = reducer(json.loads(checked(ref).read_bytes())["evidence"])
    elif number == 8200 and value.get("primitive_inventory_path"):
        from carnot.verify.request_trace_8200 import census

        ref = next(
            r for r in value["raw_shard_hashes"] if r["path"] == value["primitive_inventory_path"]
        )
        inventory = json.loads(checked(ref).read_bytes())
        result = census(inventory["records"], inventory["replay_identity"])
    elif number == 8203 and value.get("replay_input_reference"):
        from carnot.reporting.hardware_decision_8203 import costs

        ref = value["replay_input_reference"]
        data = json.loads(checked(ref).read_bytes())
        rows, bounds = costs(data["history"].get("workload_rows", []))
        result = dict(board_rows=data["boards"], workload_rows=rows, amdahl_bounds=bounds)
    elif number in (8174, 8188) and value.get("raw_shard_hashes"):
        from carnot.verify import complete_request_8174, exact_request_8188

        ref = next(
            r for r in value["raw_shard_hashes"] if Path(r["path"]).name == "primitive_rows.json"
        )
        reducer = complete_request_8174.reduce if number == 8174 else exact_request_8188.reduce
        result = reducer(json.loads(checked(ref).read_bytes()))
    else:
        return dict(available=False)
    if result != {k: value[k] for k in result}:
        raise ValueError("primitive_reduction_drift")
    return dict(available=True, reference=ref, references=[ref], result=result)


def operand(
    path: str, field: str, expected: Any, observed: Any, op: str = "==", digest: str | None = None
) -> Json:
    """Each scientific block names an actual operand instead of a vague gap label."""
    passed = (
        (
            observed > expected
            if op == ">"
            else observed >= expected
            if op == ">="
            else observed <= expected
            if op == "<="
            else observed == expected
        )
        if observed is not None
        else False
    )
    return dict(
        check=field,
        upstream="V708_science",
        path=path,
        hash=digest,
        artifact_field=field,
        op=op,
        expected=expected,
        observed=observed,
        passed=bool(passed),
    )


def reduce(data: Json) -> Json:
    """Keep the two fixed alpha allocations and all three open PRD gaps separate."""
    tasks, rows = data["tasks"], deepcopy(data["dispositions"])
    audit = data["audits"].get(tasks[5]["id"], {})
    stats = audit.get("result", {}) if rows[5]["qualified"] else {}
    source_path = audit.get("reference", {}).get("path", rows[5]["path"])
    h1_stat = stats.get("H1", {})
    h1 = dict(
        status="completed_signal"
        if stats.get("h1_development_signal_score")
        else "completed_null"
        if stats
        else "blocked",
        alpha=0.025,
        completed_count=sum(r["complete_pair"] for r in stats.get("per_source_results", [])),
        intended_count=128,
        statistics=stats,
        logistic_equivalence=stats.get("equivalent_logistic_parity", {}),
        original_missing_mask=[not r["complete_pair"] for r in stats.get("per_source_results", [])],
        scope="exposed development; original source clusters; no independent benefit",
    )
    h1_checks = [
        operand(
            source_path, field, expected, observed, op, audit.get("reference", {}).get("sha256")
        )
        for field, expected, observed, op in (
            ("qualified_H1_primitives", True, bool(stats), "=="),
            (
                "lower_one_sided_975",
                0.02,
                h1_stat.get("interval", {}).get("lower_one_sided_975"),
                ">",
            ),
            ("extra_false_accepts", 0, h1_stat.get("extra_false_accepts"), "<="),
            ("improved_sources", 5, h1_stat.get("improved_sources"), ">="),
            ("brier_increase", 0.01, h1_stat.get("brier_increase"), "<="),
            ("support_sufficient", True, h1_stat.get("support_sufficient"), "=="),
        )
    ]
    h2_checks = [
        operand(
            rows[6]["path"],
            "learning_trajectory_ready_score",
            1,
            data["primaries"].get(tasks[6]["id"], {}).get("learning_trajectory_ready_score"),
            digest=rows[6]["sha256"],
        ),
        operand(
            rows[7]["path"],
            "qualified_H2_primitive_source_rows",
            True,
            data["audits"].get(tasks[7]["id"], {}).get("available", False),
            digest=rows[7]["sha256"],
        ),
    ]
    h2 = dict(
        status="blocked",
        alpha=0.025,
        statistics={},
        intended_count=192,
        completed_count=0,
        learning_execution=rows[6]["eligible"],
        center_growth_benefit=None,
        shared_calibration_effect=None,
        retention_benefit=None,
        gate_check_summary=h2_checks,
        scope="no qualified V708 natural trajectory or later learning audit",
    )
    trace = data["audits"].get(tasks[8]["id"], {}).get("result", {})
    hardware = data["audits"].get(tasks[11]["id"], {}).get("result", {})
    service = data.get("historical_service", {})
    complete = service.get("8174", {}).get("result", {})
    service_checks = [
        operand(
            rows[8]["path"],
            "reconstructable_request_count",
            96,
            trace.get("completed_count", 0),
            ">=",
            rows[8]["sha256"],
        ),
        operand(rows[9]["path"], "qualified_complete_service_lower95_ratio", 10, None, ">="),
    ]
    failed = deduplicate(
        [g for g in data["failures"] if not g.get("passed")]
        + [g for g in h1_checks + h2_checks + service_checks if not g["passed"]]
    )
    verdict = "complete_blocked_" + str(failed[0]["check"])
    rows.append(
        dict(
            task_id=tasks[-1]["id"],
            unit_id=tasks[-1]["id"],
            source_cluster_id="owned_accounting",
            arm="task_disposition",
            condition="terminal_accounting",
            metric="eligible_branch",
            numerator=0,
            denominator=1,
            status="completed",
            completed=True,
            eligible=False,
            excluded=True,
            failed=False,
            censored=False,
            honest_verdict=verdict,
            verdict_class="blocked",
            disposition="owned_capstone",
            exclusion_reason="external_science_block",
        )
    )
    retirements = []
    for task, row in zip(tasks, rows, strict=True):
        for prior in task["prior_failures"]:
            old = data["prior_evidence"].get(prior["experiment_id"], {})
            authentic = old.get("sha256") == old.get("expected_sha256") and bool(old.get("sha256"))
            authentic &= old.get("honest_verdict") == prior["verdict"]
            same = (
                authentic
                and row.get("primary_present", False)
                and row["honest_verdict"] == prior["verdict"]
            )
            retirements.append(
                dict(
                    prior,
                    task_id=task["id"],
                    prior_artifact=old,
                    prior_verdict_matches_artifact=authentic,
                    same_verdict=same,
                    retire_exact_configuration=bool(same and prior["retire_if_same_verdict"]),
                    retire_method_family=False,
                    documented_scope=prior["addressed_by"],
                    scientific_hypothesis_tested=row.get("eligible", False),
                    decision="retire_only_matching_scope"
                    if same
                    else "preserve_changed_or_untested_scope",
                )
            )
    retirements.append(
        dict(
            task_id=tasks[5]["id"],
            mechanism_scope="V708 local_set exposed128 continuation",
            evidence_path=source_path,
            retire_exact_configuration=h1["status"] == "completed_null",
            retire_method_family=False,
            unchanged_capture_or_threshold_retry_authorized=False,
            decision="retire_exact_exposed_selective_continuation"
            if stats
            else "untested_external_block",
        )
    )
    gaps = {}
    for name, requirements, checks, action in (
        (
            "source_decisions",
            ["FR-06", "FR-12"],
            h1_checks,
            "Retire the exact exposed local_set continuation; propose independent labels only after a changed safe mechanism qualifies.",
        ),
        (
            "later_learning_retention",
            ["FR-11"],
            h2_checks,
            "Repair owned calibrated-head qualification before a changed natural trajectory; independently test center growth beyond shared calibration.",
        ),
        (
            "service_deployment",
            ["FR-05", "FR-08", "NFR-01"],
            service_checks,
            "Authenticate request issue clocks and identities before measuring cold-inclusive Python/Rust complete-service10x and deployed demand.",
        ),
    ):
        gaps[name] = dict(
            closed=False,
            requirements=requirements,
            evidence_paths=[c["path"] for c in checks],
            unmet_operands=[c for c in checks if not c["passed"]],
            next_action=action,
            independent_validation_proposal_only=True,
        )
    return dict(
        honest_verdict=verdict,
        verdict_class="blocked",
        rows=rows,
        task_dispositions=rows,
        task_contract=tasks,
        canonical_tasks_sha256=tasks_digest(tasks),
        authority=data["authority"],
        H1=h1,
        H2=h2,
        multiplicity=dict(
            family=["H1", "H2"],
            method="fixed Bonferroni allocation",
            alpha_per_hypothesis=dict(H1=0.025, H2=0.025),
            alpha_transfer=False,
        ),
        h1_development_signal_score=int(stats.get("h1_development_signal_score", 0)),
        h2_development_signal_score=0,
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        verifier_is_oracle=data.get("verifier_is_oracle", False),
        claim_scope="terminal accounting and cached exposed-development reductions",
        exposure_scope="all historical cohorts exposed development",
        science_ready_score=0,
        capstone_execution_ready_score=0,
        current_scheduling_ready_score=int(data["authority"].get("activated", False)),
        intended_count=13,
        completed_count=13,
        eligible_count=sum(r["eligible"] for r in rows),
        excluded_count=sum(r["excluded"] for r in rows),
        failed_count=sum(r["failed"] for r in rows),
        censored_count=0,
        independent_count=0,
        sample_size_budget=dict(tasks=13, H1=128, H2=192, retention=64, seeds_are_sources=False),
        preconditions_checked=True,
        gate_check_summary=failed,
        independent_reductions=data["audits"],
        retirement_decisions=retirements,
        gap_decisions=gaps,
        board_obligations=hardware.get("board_rows", []),
        literature_mapping=dict(
            frozen_scan=data["literature_mapping"],
            result_mapping=dict(
                class_conditional_sets=source_path,
                ARM_EBM_equivalence=source_path,
                KAC_continual_calibration=tasks[7]["deliverable"],
                delayed_feedback=tasks[6]["deliverable"],
                KANtize_hardware=tasks[11]["deliverable"],
                neural_constraints="semantic extraction remains fallible",
                energy_guided_generation="deferred; no safe utility",
                hardware_sampling="board projections only",
            ),
        ),
        historical_hash_failures=data["historical_hash_failures"],
        current_hash_failures=[g for g in failed if "sha256" in g["check"]],
        calibration_only_effect=dict(
            qualification=data["primaries"].get(tasks[1]["id"], {}).get("verdict_class"),
            exposed_fixture_success=data["primaries"]
            .get(tasks[1]["id"], {})
            .get("future_decision_fixture_score"),
            shared_calibration=None,
            center_growth_benefit=None,
            natural_trajectory_qualified=False,
        ),
        service_evidence_scope=dict(
            observed_trace=trace,
            historical_complete_service=complete,
            conditional_exact_repeat=service.get("8188", {}).get("result", {}),
            whole_service_qualified=False,
            nfr01_met=False,
            nfr01_required_lower95_ratio=10,
            current_observed_service_measurement=None,
            hardware_bounds=hardware.get("amdahl_bounds", []),
            scope="research workload only; historical generation-inclusive ratios remain descriptive; conditional cache hits do not measure deployed demand",
        ),
        arc_evidence=dict(
            disposition=rows[10]["disposition"],
            scientific_rerun=False,
            standing_inventory=True,
            new_solve_credit=0,
            policy_changed=False,
            failed_prerequisites=rows[10]["gate_check_summary"],
        ),
        oracle_distinct_corrigendum=dict(
            date="2026-09-28",
            gap="GAP-ORACLE-DISTINCT",
            closed=False,
            evidence_path="research-references.md:48101",
            invalidated=[4245, 5151, 5160, 5171],
            reasons=["leaked confidence", "self-labeling", "wrong confidence-interval unit"],
            independent_labels_required=True,
            source_family_intervals_required=True,
            generator_training_authorized=False,
            unseen_generator_transfer_established=False,
        ),
        publication_scope="FoVer G1-G4 only; paper_ready does not certify H1, H2 or deployment",
    )

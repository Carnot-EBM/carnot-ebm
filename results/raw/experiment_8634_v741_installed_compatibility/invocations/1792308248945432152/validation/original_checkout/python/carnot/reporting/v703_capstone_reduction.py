"""REQ-VERIFY-8135: keep source, learning and service conclusions independent.

Each scientific branch supplies its own source support. Repeated seeds and
qualified fixture timings cannot pay another branch's missing evidence debt.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v699_capstone_reduction import holm
from carnot.reporting.v701_capstone_reduction import hypothesis
from carnot.reporting.v703_capstone_inputs import deduplicate
from carnot.reporting.v685_authority_lifecycle import tasks_digest

Json = dict[str, Any]


def scientific(data: Json, index: int, name: str) -> Json:
    """Recompute source gains and a conditional bootstrap tail on eligible pairs only."""
    primary = data["primaries"].get(data["tasks"][index]["id"], {})
    eligible = data["dispositions"][index]["eligible"]
    rows = primary.get("decision_rows", []) if eligible else []
    result = hypothesis(rows, name)
    groups: dict[str, list[float]] = {}
    for row, mask in zip(rows, result["sample_masks"], strict=True):
        if mask["eligible"]:
            groups.setdefault(str(row["source_id"]), []).append(
                row["control_cost"] - row["treatment_cost"]
            )
    p = 1.0
    if result["status"] == "completed":
        gains = np.array([np.mean(g) for g in groups.values()])
        rng = np.random.default_rng(703)
        if name == "H2":
            slots = {
                row["slot"]: float(np.mean(groups[str(row["source_id"])]))
                for row in rows
                if str(row.get("source_id")) in groups
            }
            windows = [
                [slots[s] for s in range(start, start + 16) if s in slots]
                for start in range(65, 242)
            ]
            totals, counts = (
                np.array([sum(w) for w in windows]),
                np.array([len(w) for w in windows]),
            )
            choices = rng.integers(len(windows), size=(10000, 12))
            denominator = counts[choices].sum(axis=1)
            draws = totals[choices].sum(axis=1)[denominator > 0] / denominator[denominator > 0]
        else:
            draws = gains[rng.integers(len(gains), size=(10000, len(gains)))].mean(axis=1)
        p = float((1 + np.sum(draws <= 0.02)) / 10001)
    result.update(
        raw_p_value=p,
        support_passed=result["status"] == "completed",
        safety_passed=bool(result["development_signal_score"]),
        beneficial_changed_sources=result["beneficial_sources"],
        intended_source_count=128 if name == "H1" else 192,
        unavailable_source_count=max(0, (128 if name == "H1" else 192) - len(groups)),
        protocol_conformant=eligible,
        uncertainty_scope="Conditional exposed-development bootstrap; no independent confirmation",
    )
    return result


def reduce(data: Json) -> Json:
    """Finish all thirteen dispositions while every unresolved branch keeps its own gap."""
    tasks = data["tasks"]
    h1, h2 = scientific(data, 5, "H1"), scientific(data, 8, "H2")
    learning = (
        data["primaries"].get(tasks[8]["id"], {}) if data["dispositions"][8]["eligible"] else {}
    )
    retained = {
        str(row["source_id"]): row
        for row in learning.get("retention_rows", [])
        if row.get("status") == "completed"
    }
    support = {
        str(label): sum(row["label"] == label for row in retained.values()) for label in (0, 1)
    }
    safe = (
        len(retained) >= 48
        and min(support.values()) >= 8
        and all(
            row["retention_cost_increase"] <= 0.02
            and row["retention_brier_increase"] <= 0.01
            and not row["extra_false_accept"]
            for row in retained.values()
        )
    )
    h2["retention"] = dict(
        source_count=len(retained),
        class_support=support,
        safe=safe,
        intended_source_count=64,
        unavailable_source_count=64 - len(retained),
    )
    h2["safety_passed"] = bool(h2["safety_passed"] and safe)
    family = holm(
        h1, h2, (data["dispositions"][5]["eligible"], data["dispositions"][8]["eligible"])
    )
    for h, corrected in zip((h1, h2), family, strict=True):
        h["development_signal_score"] = int(corrected["positive_claim"])
    blocked = bool(data["failures"]) or any(not row["eligible"] for row in data["dispositions"])
    blocked = blocked or any(h["status"] == "blocked" for h in (h1, h2))
    state = (
        "blocked"
        if blocked
        else "positive"
        if any(h["development_signal_score"] for h in (h1, h2))
        else "null"
    )
    oracle = bool(data.get("verifier_is_oracle", False))
    if state == "positive" and oracle:
        state = "circular_positive"
    first = next((g["check"] for g in data["failures"] if not g.get("passed")), "science_support")
    verdict = "complete_" + state + "_" + (str(first) if state == "blocked" else "v703_capstone")
    rows = deepcopy(data["dispositions"])
    rows.append(
        dict(
            unit_id=tasks[-1]["id"],
            task_id=tasks[-1]["id"],
            source_cluster_id="owned_accounting",
            arm="task_disposition",
            condition="terminal_accounting",
            metric="qualified_branch",
            numerator=int(not blocked),
            denominator=1,
            raw_numerator=int(not blocked),
            raw_denominator=1,
            eligible=not blocked,
            excluded=blocked,
            completed=True,
            status="completed",
            failed=False,
            censored=False,
            honest_verdict=verdict,
            verdict_class=state,
            disposition="owned_capstone",
            exclusion_reason="external_science_block" if blocked else None,
            gate_check_summary=[],
        )
    )
    for row in rows:
        row.setdefault("source_cluster_id", row["unit_id"])
        row.setdefault("arm", "task_disposition")
        row.setdefault("condition", "terminal_accounting")
        row.setdefault("metric", "qualified_branch")
    retirements = []
    for task, row in zip(tasks, rows, strict=True):
        for prior in task["prior_failures"]:
            evidence = data.get("prior_evidence", {}).get(prior["experiment_id"], {})
            same = prior["verdict"] == row["honest_verdict"]
            retirements.append(
                dict(
                    prior,
                    task_id=task["id"],
                    observed_verdict=row["honest_verdict"],
                    same_verdict=same,
                    prior_path=evidence.get("path"),
                    prior_sha256=evidence.get("sha256"),
                    prior_artifact_verdict=evidence.get("honest_verdict"),
                    prior_verdict_matches_artifact=evidence.get("honest_verdict")
                    == prior["verdict"],
                    task_config_sha256=canonical_hash(task),
                    retire_exact_configuration=False,
                    decision="preserve_history_no_authenticated_unchanged_configuration",
                    environmental_block_retires_method_family=False,
                    reopen_condition="Changed mechanism and qualified evidence; unchanged external blocks remain terminal.",
                )
            )
    service = data["primaries"].get(tasks[9]["id"], {})
    service_qualified = data["dispositions"][9].get(
        "qualified", data["dispositions"][9]["eligible"]
    )
    host = bool(service_qualified and service.get("host_service_ready_score") == 1)
    complete = bool(service_qualified and service.get("complete_service_ready_score") == 1)
    hardware = data["primaries"].get(tasks[11]["id"], {})
    arc = data["primaries"].get(tasks[10]["id"], {})
    next_actions = {
        "source_decisions": "Qualify Exp8124 identity and capture protocol, then capture fit/reserved paired evidence and audit Exp8128 with >=96 sources and >=12 per class.",
        "later_learning_retention": "Qualify Exp8129 full-size protocol receipt; run Exp8130 sealed 20-slot/12-label schedule, then audit >=128 later sources, >=8 per class and >=48 retained sources.",
        "service_deployment": "Join qualified natural update, acquisition, transport and recovery costs on identical request/state keys; test one-sided matched whole-service lower bound >10.",
        "arc_reader_frontier": "Authenticate the Exp8120 validation manifest or append a new independently qualified reader receipt; then scan only unseen redirect IDs and require >=30 outcomes across >=5 games.",
        "independent_generalization": "Acquire a separately collected, human-labeled corpus and freeze untouched source roles before making an independent generalization claim.",
    }
    boards = []
    for board in hardware.get("board_rows", []):
        boards.append(
            dict(
                board=board["board"],
                custody_valid=board.get("custody_valid"),
                current_execution=board.get("current_hardware_execution", False),
                processor_class=board.get("processor_class"),
                k_max=board.get("k_max"),
                source_path=board.get("source_path"),
                source_hash=board.get("source_hash"),
                next_action=board.get("next_operator_or_device_change"),
                falsifiable_evidence=board.get("next_missing_prerequisite"),
                deployed_benefit=False,
            )
        )
    decisions = {
        name: dict(closed=False, requirements=reqs, next_action=next_actions[name])
        for name, reqs in [
            ("source_decisions", ["FR-06", "FR-12"]),
            ("later_learning_retention", ["FR-11"]),
            ("service_deployment", ["FR-05", "FR-08", "NFR-01"]),
            ("arc_reader_frontier", ["FR-06", "FR-12"]),
            ("independent_generalization", ["FR-06", "FR-11"]),
        ]
    }
    return dict(
        honest_verdict=verdict,
        verdict_class=state,
        rows=rows,
        task_dispositions=rows,
        task_contract=tasks,
        canonical_tasks_sha256=tasks_digest(tasks),
        authority=data["authority"],
        H1=h1,
        H2=h2,
        primary_hypothesis_results=family,
        multiplicity=dict(family=["H1", "H2"], method="Holm", alpha=0.05, unavailable_family_p=1),
        h1_development_signal_score=h1["development_signal_score"],
        h2_development_signal_score=h2["development_signal_score"],
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        verifier_is_oracle=oracle,
        claim_scope="exposed development and administrative accounting",
        exposure_scope="Previously exposed RAGTruth; fixture host heads supply protocol evidence only",
        science_ready_score=int(not blocked),
        capstone_execution_ready_score=0,
        intended_count=13,
        completed_count=13,
        eligible_count=sum(r["eligible"] for r in rows),
        excluded_count=sum(r["excluded"] for r in rows),
        failed_count=sum(r["failed"] for r in rows),
        censored_count=0,
        independent_count=0,
        sample_size_budget=dict(tasks=13, H1=128, H2=192, retention=64, seeds_are_sources=False),
        gate_check_summary=deduplicate(data["failures"]),
        preconditions_checked=data["preconditions"],
        independent_reductions=data["independent_reductions"],
        retirement_decisions=retirements,
        retirement_history=[
            p.get("retirement_decisions", p.get("retirement_rows", []))
            for p in data["primaries"].values()
        ]
        + [
            dict(
                prior_task_id=key,
                path=p["path"],
                sha256=p["sha256"],
                preserved_retirements=p.get("retirement_history", []),
            )
            for key, p in data.get("prior_evidence", {}).items()
            if p.get("retirement_history")
        ],
        gap_decisions=decisions,
        next_actions=next_actions,
        service_evidence_scope=dict(
            host_qualified=host,
            whole_service_qualified=complete,
            oracle_fixture=bool(service.get("verifier_is_oracle")),
            natural_update_ready=service.get("natural_update_cost_ready_score", 0),
            nfr01_met=bool(complete and service.get("nfr01_met")),
            scope="Current host fixture timings; historical acquisition is modeled; no deployed speedup",
        ),
        arc_evidence=dict(
            reader_ready=arc.get("supervisor_reader_ready_score", 0),
            new_outcome_count=arc.get("new_outcome_count"),
            frontier=arc.get("current_frontier"),
            solve_credit=0,
            policy_changed=False,
        ),
        board_obligations=boards,
    )


def qualify(value: Json, passed: bool) -> None:
    """Owned failure disqualifies accounting and cannot preserve a positive readiness bit."""
    value.update(required_checks_passed=passed, capstone_execution_ready_score=int(passed))
    if not passed:
        value.update(
            honest_verdict="complete_disqualified_owned_validation",
            verdict_class="disqualified",
            science_ready_score=0,
            h1_development_signal_score=0,
            h2_development_signal_score=0,
        )
        value["rows"][-1].update(
            honest_verdict=value["honest_verdict"],
            verdict_class="disqualified",
            numerator=0,
            raw_numerator=0,
            eligible=False,
            excluded=True,
        )

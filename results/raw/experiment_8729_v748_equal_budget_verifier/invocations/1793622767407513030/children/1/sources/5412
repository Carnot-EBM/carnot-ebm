"""REQ-VERIFY-8191: cached branch reductions retain original scientific limits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting import v706_capstone_evidence as previous
from carnot.reporting.v699_capstone_reduction import holm

Json = dict[str, Any]


def primitive(value: Json, number: int) -> Json:
    """Reopen sealed primitive bytes with the producer's qualified equations."""
    if number == 8185 and value.get("measurement_reference"):
        from carnot.verify.sentence_decision_audit_8185 import reduce as reducer

        ref = value["measurement_reference"]
        result = reducer(json.loads(checked(ref).read_bytes())["evidence"])
    elif number == 8188 and value.get("raw_shard_hashes"):
        from carnot.verify.exact_request_8188 import reduce as reducer

        ref = next(
            r for r in value["raw_shard_hashes"] if Path(r["path"]).name == "primitive_rows.json"
        )
        result = reducer(json.loads(checked(ref).read_bytes()))
    elif number == 8190 and value.get("replay_input_reference"):
        from carnot.reporting.hardware_service_8190 import reduce as reducer

        ref = value["replay_input_reference"]
        reduced = reducer(json.loads(checked(ref).read_bytes()))
        result = {
            k: reduced[k]
            for k in ("board_rows", "amdahl_bounds", "quantization_rows", "workload_rows")
        }
    else:
        return dict(available=False)
    if result != {k: value[k] for k in result}:
        raise ValueError("primitive_reduction_drift")
    return dict(available=True, reference=ref, references=[ref], result=result)


def reduce(data: Json) -> Json:
    """Keep completed H1 nulls usable while absent H2 science stays blocked."""
    value = previous.reduce(data)
    tasks, rows = data["tasks"], value["rows"]
    audit = data["audits"].get(tasks[7]["id"], {})
    stats = audit.get("result", {}) if rows[7]["eligible"] else {}
    value["H2"]["learning_execution"] = rows[8]["eligible"]
    gains = [r["h1_gain"] for r in stats.get("per_source_results", []) if r["h1_gain"] is not None]
    p_value = 1.0
    if gains:
        print("[exp8191] phase=before_benchmark_H1_margin completed=0 pending=10000", flush=True)
        samples = np.random.default_rng(7078185).integers(0, len(gains), size=(10000, len(gains)))
        means = np.asarray(gains)[samples].mean(axis=1)
        p_value = float((1 + np.count_nonzero(means <= 0.02)) / 10001)
        print("[exp8191] phase=after_benchmark_H1_margin completed=10000 pending=0", flush=True)
    h1 = dict(
        status="completed_signal"
        if stats.get("h1_development_signal_score")
        else "completed_null"
        if stats
        else "blocked",
        raw_p_value=p_value,
        raw_p_value_method="source_bootstrap_margin_0.02",
        support_passed=stats.get("support_sufficient", False),
        safety_passed=stats.get("safety_passed", False),
        observed_gain=stats.get("paired_intervals", {}).get("all_slot", {}).get("mean_gain"),
        beneficial_changed_sources=stats.get("improved_sources", 0),
        intended_count=128,
        completed_count=stats.get("eligible_count", 0),
        original_missing_mask=[not r["complete_pair"] for r in stats.get("per_source_results", [])],
        statistics=stats,
        scope="exposed development; original source clusters",
        logistic_equivalence=stats.get("equivalent_logistic_parity", {}),
    )
    family = holm(h1, value["H2"], (bool(stats), False))
    service = data["audits"].get(tasks[10]["id"], {}).get("result", {})
    hardware = data["audits"].get(tasks[12]["id"], {}).get("result", {})
    failures = [
        g for r in rows[1:-1] for g in r.get("gate_check_summary", []) if not g.get("passed")
    ]
    check = str(failures[0]["check"]) if failures else "H2_unavailable"
    verdict = "complete_blocked_" + check
    rows[-1]["honest_verdict"] = verdict
    value.update(
        honest_verdict=verdict,
        H1=h1,
        primary_hypothesis_results=family,
        h1_development_signal_score=int(family[0]["positive_claim"]),
        h2_development_signal_score=0,
        current_scheduling_ready_score=int(data["authority"].get("activated", False)),
        historical_hash_failures=data["historical_hash_failures"],
        calibration_only_effect=dict(
            status="blocked",
            reason="Disqualified method and absent natural learning audit",
            observed_effect=None,
            center_growth_benefit=None,
            retained_performance=None,
        ),
        board_obligations=hardware.get("board_rows", []),
        service_evidence_scope=dict(
            conditional_exact_repeat=service,
            complete_request_timing="Qualified cold, repeat, refresh and changed-source requests; imported host timing only",
            measured_repeat_frequency=data["primaries"]
            .get(tasks[10]["id"], {})
            .get("measured_repeat_frequency"),
            whole_service_qualified=False,
            nfr01_met=service.get("nfr01_met", False),
            hardware_bounds=hardware,
            scope="Conditional exact-repeat benefit; unknown natural repeat frequency; no deployed board timing",
        ),
        oracle_distinct_corrigendum=dict(
            date="2026-09-28",
            gap="GAP-ORACLE-DISTINCT",
            closed=False,
            prior_data_exposure=True,
            generator_training_authorized=False,
            unseen_generator_transfer_established=False,
        ),
    )
    for decision in value["retirement_decisions"]:
        row = next(r for r in rows if r["task_id"] == decision["task_id"])
        tested = row.get("primary_present", False)
        old = data["prior_evidence"][decision["experiment_id"]]
        matches = old["sha256"] == old["expected_sha256"]
        decision["prior_sha256_matches_snapshot"] = matches
        decision["prior_verdict_matches_artifact"] &= matches
        decision["retire_exact_configuration"] &= tested and matches
        decision["scientific_hypothesis_tested"] = tested and row["eligible"]
        if decision["task_id"] == tasks[-1]["id"]:
            decision.update(
                current_honest_verdict=verdict, same_verdict=False, retire_exact_configuration=False
            )
    value["next_actions"]["later_learning_retention"] = (
        "Repair owned method validation, then test fresh causal later-source labels. Require center-growth cost gain >.02 beyond calibration-only, >=128 source pairs, >=48 retention sources and strict restart parity."
    )
    value["next_actions"]["service_deployment"] = (
        "Measure independent natural repeat frequency and complete cold-inclusive requests. Require lower95 speed ratio >1 and <=100ms end-to-end before NFR-01 closure; authenticate each board dispatch."
    )
    for name, gap in value["gap_decisions"].items():
        gap["next_action"] = value["next_actions"][name]
    return value

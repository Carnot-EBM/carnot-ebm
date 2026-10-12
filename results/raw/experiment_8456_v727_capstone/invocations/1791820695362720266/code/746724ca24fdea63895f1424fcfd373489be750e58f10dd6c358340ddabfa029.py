"""REQ-REPORT-8456: qualification never converts a missing experiment to a null."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_capstone_report as old
from carnot.reporting import v727_capstone_evidence as e

Json = dict[str, Any]
_BUILD = old.build
CLAIMS = {
    "source_information": (
        8446,
        "continue_source_controls_after_qualification",
        "Restore full independent V727 authority; freeze scalar logistic, scalar spline, full, no-source and target-blind permuted-source heads. Require both corrected source Brier lower bounds>0, typed-loss noninferiority and an independent sealed cohort before generalization.",
    ),
    "objective_learning": (
        8448,
        "continue_changed_objective_after_qualification",
        "Qualify the identical delayed-feedback Brier/log-loss/frozen/shuffled stream. Both later-question Brier lower bounds must exceed0, typed losses must not increase, and the simultaneous upper95 bound across all five retention windows must be<=0.01.",
    ),
    "durable_service": (
        8451,
        "continue_changed_representation_after_durability",
        "First qualify causal snapshot/journal equivalence, idempotent acknowledged recovery and crash durability; then measure paired complete setup/request/update/tail clocks at identical work. Arithmetic-only10x remains closed.",
    ),
    "current_qwen": (
        8452,
        "continue_only_after_qualified_runtime_repair",
        "Require an authenticated CUDA repair and bounded context/copy parity plus source controls before optional current-Qwen calls.",
    ),
    "arc_cross_game_supervision": (
        8453,
        "blocked_independent_support",
        "Restore original ARC publication custody and three adapter-withheld overlapping games with at least five resolved firings per shared cell; preserve censored outcomes.",
    ),
    "board_operations": (
        8454,
        "blocked_execution_evidence",
        "Require supported installed board operations and complete board/transport clocks. KV260 quadratic Ising k<=5; PolarFire authenticated Linux CPU dispatch only.",
    ),
    "gatemate_execution": (
        8455,
        "blocked_physical_and_source_evidence",
        "Recover exact source hashes, dated physical cable/port/power change, IDCODE0x20000001, n16 flash and device sample/hash smoke before repeating.",
    ),
}


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """The shipped accounting reducer retains its checks; only this contract and claims change."""
    with patch.multiple(old, e=e, CLAIMS=CLAIMS):
        value = dict(_BUILD(work, receipts, raw, output))
    value.update(
        experiment_id=8456,
        random_seed=7278456,
        honest_verdict=value["honest_verdict"].replace("v726", "v727"),
        invocation_argv=work["invocation_argv"],
        methodology_note="Read-only bounded aggregation of V727 slots and original sealed V726 primitives. Public-label results remain retrospective. Source information, objective learning and complete durable journal speed require measured qualified branches; absence is not zero effect. No model, game or board is invoked. Historical FoVer readiness does not establish V727 success.",
    )
    value["rows"][-1]["experiment_id"] = 8456
    for claim in ["source_information", "objective_learning", "durable_service"]:
        c = value["scientific_claims"][claim]
        c.update(
            qualified_retrospective_benefit=None,
            missing_reason=None if c["qualified_producer"] else "required_producer_not_qualified",
        )
    value["scientific_claims"]["objective_learning"]["retention_windows"] = [
        dict(window=i, metric=None, missing_reason="Exp8448 objective audit not measured")
        for i in range(5)
    ]
    value["scientific_claims"]["durable_service"].update(
        complete_journal_speedup=None, durability_qualified=False
    )
    preserved = {}
    for row in work["historical_slots"]:
        preserved[str(row["experiment_id"])] = dict(
            qualified=row["eligible"],
            original_verdict=row["honest_verdict"],
            metrics=row["selected"],
            primary=row["primary_reference"],
            decision="keep_retrospective_scalar_gain"
            if row["experiment_id"] == 8432
            else "keep_learning_null_and_all_retention_windows"
            if row["experiment_id"] == 8435
            else "keep_complete_service_partition_and_arithmetic_ceiling_closed",
        )
    value["scientific_claims"]["V726_preserved"] = preserved
    value["cited_upstream_artifacts"] += [
        dict(r["primary_reference"], producer=r["task_id"], imported_fields=list(r["selected"]))
        for r in work["historical_slots"]
    ]
    value["retirement_recommendations"] += [
        dict(
            task_id=r["task_id"],
            predecessor=r["primary_reference"],
            exact_verdict=r["honest_verdict"],
            recommendation="retire_exact_unchanged_repeat",
            hypothesis_retired=False,
            reopening_condition=CLAIMS[
                "durable_service" if r["experiment_id"] == 8437 else "objective_learning"
            ][2],
        )
        for r in work["historical_slots"]
        if r["experiment_id"] in [8435, 8437]
    ]
    for claim, (number, _, condition) in CLAIMS.items():
        row = next(r for r in work["slots"] if r["experiment_id"] == number)
        if row.get("producer_executed") and row["verdict_class"] in {"blocked", "disqualified"}:
            value["retirement_recommendations"].append(
                dict(
                    task_id=row["task_id"],
                    predecessor=row["primary_reference"],
                    exact_verdict=row["honest_verdict"],
                    recommendation="retire_exact_unchanged_repeat",
                    hypothesis_retired=False,
                    scope=claim,
                    reopening_condition=condition,
                )
            )
    value["field_principles"].update(
        {
            k: "Bind actual invocation and sealed original producer scope."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = e.checksum(value)
    return value

"""REQ-REPORT-8539: qualification, benefit and publication have separate meanings."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_capstone_report as shipped
from carnot.reporting import v733_capstone_evidence as e
from carnot.reporting import v733_contract_methods as methods

Json = dict[str, Any]
CLAIMS = {
    "source_information": (
        8529,
        "retire_unchanged_source_contrasts",
        "Require new disjoint independently labeled questions or new source information; do not tune the same exposed raw/scalar/permuted contrasts.",
    ),
    "objective_learning": (
        8532,
        "blocked_no_valid_learning_null",
        "Repair exact accounting/validation failures and qualify real versus shuffled feedback plus simultaneous retention on the original stream. Change objective only after a valid learning null.",
    ),
    "nine_input_inference": (
        8534,
        "blocked_native_qualification_and_complete_costs",
        "Qualify the actual nine-input native binary and numerical/typed decision parity, then measure complete paired native request speed for NFR-01. Do not queue another five-input append benchmark.",
    ),
    "current_qwen": (
        8535,
        "blocked_unchanged_runtime",
        "Require an authenticated actual CUDA/runtime repair and successful bounded context/copy probe before generation.",
    ),
    "arc_capability": (
        8537,
        "disqualified_panel_no_generalization",
        "Repair owned panel validation; require model-enabled qualified capability or a new general discovery mechanism with resolved direct effects, never unchanged re-solves.",
    ),
    "board_operations": (
        8538,
        "blocked_installed_operations_and_physical_change",
        "Require compatible installed operations, complete native/device/transport clocks and actual runtime or physical change. Preserve GateMate source hashes then cable/port/power, IDCODE, flash and smoke obligations.",
    ),
}


def prior_pins() -> Json:
    """Follow sealed ancestors so older native failures retain their original bytes."""
    protocol = methods.protocol()
    wanted = {
        p["experiment_id"] for t in protocol["task_map_v733"] for p in t.get("prior_failures", [])
    }
    pending = [protocol["v732_capstone_reference"], *protocol["source_references"]]
    pins: Json = {}
    seen = set()
    while pending and not wanted <= pins.keys():
        ref = pending.pop(0)
        name = Path(ref.get("source_path", ref["path"])).name
        if "capstone" not in name or not name.endswith(".json") or ref["sha256"] in seen:
            continue
        seen.add(ref["sha256"])
        value = e.load(ref)
        pins[value["task_id"]] = ref
        for row in [
            *value.get("rows", []),
            *value.get("historical_reductions", []),
            *value.get("prior_failure_comparisons", []),
        ]:
            source = row.get("primary_reference", row.get("reference"))
            if source and source.get("exists", True):
                pins[row.get("predecessor", row["task_id"])] = source
        pending.extend(value.get("source_artifact_hashes", []))
    return pins


def history_summary(prior: Json, ref: Json) -> Json:
    """Keep the outcome table small while immutable source seals retain all historical rows."""
    return dict(
        primary=ref,
        task_count=prior["task_count"],
        rows=[
            {
                k: r.get(k)
                for k in [
                    "task_id",
                    "experiment_id",
                    "disposition",
                    "honest_verdict",
                    "verdict_class",
                    "eligible",
                    "primary_reference",
                ]
            }
            for r in prior["rows"]
        ],
        five_input_host_service={
            k: v for k, v in prior["scientific_claims"]["append_service"].items() if k != "metrics"
        }
        | {
            "metrics": {
                k: prior["scientific_claims"]["append_service"]["metrics"][k]
                for k in [
                    "pooled_speedup",
                    "primary_speedup_ci95",
                    "primary_p95_ratio",
                    "NFR_01_status",
                    "durability_scope",
                ]
            }
        },
        scientific_claims={
            k: {
                f: v.get(f) for f in ["decision", "original_verdict", "qualified_producer", "scope"]
            }
            for k, v in prior["scientific_claims"].items()
        },
        authority_retroactively_repaired=False,
    )


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Keep exact prior verdicts and descriptive operands without promoting failed science."""
    with patch.multiple(shipped, e=e, CLAIMS=CLAIMS):
        value = dict(shipped.build(work, receipts, raw, output))
    prior = e.load(work["predecessors"][0])
    value.update(
        experiment_id=8539,
        run_date="20261015",
        random_seed=7338539,
        honest_verdict=f"complete_{value['verdict_class']}_v733_capstone",
        canonical_tasks_sha256=work["authority"].get("canonical_tasks_sha256"),
        invocation_argv=work["invocation_argv"],
        administrative_complete=len(value["rows"]) == 14,
        publication_authorized=False,
        V732_history=history_summary(prior, work["predecessors"][0]),
        historical_exposure_and_quarantines=methods.protocol()["retirement_ledger"],
        prior_owned_failed_attempts=[
            dict(e.load(r), reference=r) for r in work["prior_owned_failed_attempts"]
        ],
    )
    value["rows"][-1].update(
        experiment_id=8539,
        honest_verdict=value["honest_verdict"],
        verdict_class=value["verdict_class"],
    )
    value["failed_count"] = value["disqualified_producer_count"]
    value["acceptance_gates"]["all_slots_accounted"] = len(value["rows"]) == 14
    for name, (number, decision, _) in CLAIMS.items():
        row = next(r for r in value["rows"] if r["experiment_id"] == number)
        value["scientific_claims"][name].update(
            decision=decision,
            qualified_retrospective_benefit=None,
            verdict_class=row["verdict_class"],
            missing_reason=None if row["eligible"] else "required_science_absent_or_disqualified",
        )
    value["scientific_claims"]["objective_learning"]["feedback_controls"] = [
        dict(
            arm=a,
            metric=None,
            missing_reason="Exp8530 disqualified; Exp8531 pre-gated; Exp8532 cascade-skipped",
        )
        for a in ["real_feedback", "shuffled_feedback", "frozen", "log_loss"]
    ]
    value["scientific_claims"]["objective_learning"]["retention_windows"] = [
        dict(window=i, metric=None, missing_reason="no_qualified_causal_stream") for i in range(5)
    ]
    value["scientific_claims"]["nine_input_inference"].update(
        complete_speedup=None,
        scope="nine-input inference only; Exp8520 five-input host-service speed does not transfer",
    )
    value["inherited_adversarial_findings"] = [
        dict(
            producer=r["task_id"],
            primary=r["primary_reference"],
            findings=r.get("upstream_adversarial_findings", []),
            failed_receipts=r.get("upstream_failed_receipts", []),
        )
        for r in work["slots"]
        if r.get("upstream_adversarial_findings") or r.get("upstream_failed_receipts")
    ]
    pins = prior_pins()
    comparisons = []
    for task in work["tasks"]:
        current = next(r for r in value["rows"] if r["task_id"] == task["id"])
        for p in task.get("prior_failures", []):
            ref = pins.get(p["experiment_id"])
            original = e.load(ref) if ref and ref.get("exists", True) else None
            observed = original.get("honest_verdict") if original else None
            comparisons.append(
                dict(
                    task_id=task["id"],
                    predecessor=p["experiment_id"],
                    reference=ref,
                    expected_verdict=p["verdict"],
                    observed_verdict=observed,
                    exact_match=observed == p["verdict"] if original else None,
                    current_verdict=current.get("honest_verdict"),
                    same_result=observed == current.get("honest_verdict") if original else None,
                    retire_if_same_verdict=p.get("retire_if_same_verdict"),
                    missing_reason=None if original else "authenticated_original_bytes_unavailable",
                )
            )
    value["prior_failure_comparisons"] = comparisons
    for name, (number, _, condition) in CLAIMS.items():
        row = next(r for r in value["rows"] if r["experiment_id"] == number)
        value["retirement_recommendations"].append(
            dict(
                task_id=row["task_id"],
                predecessor=row["primary_reference"],
                exact_verdict=row["honest_verdict"],
                recommendation="retire_exact_unchanged_attempt",
                hypothesis_retired=False,
                scope=name,
                reopening_condition=condition,
            )
        )
    value["methodology_note"] = (
        "No-model aggregation under exact sealed V733 authority. Cold-replay shipped producer primitives; preserve failed validation and censored branches. Fourteen scheduling outcomes are not fourteen successful studies. Exposed labels and oracle fixtures grant no generalization. Publication readiness is independent and grants no submission authority."
    )
    value["field_principles"].update(
        {
            k: "Bind exact invocation and original producer seals; administrative completion grants no scientific benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = e.checksum(value)
    return value

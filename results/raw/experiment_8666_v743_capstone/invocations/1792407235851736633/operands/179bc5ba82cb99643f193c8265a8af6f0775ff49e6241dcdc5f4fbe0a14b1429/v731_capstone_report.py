"""REQ-REPORT-8511: separate administrative completion from each branch's scientific eligibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_capstone_report as old
from carnot.reporting import v731_capstone_evidence as e

Json = dict[str, Any]
CLAIMS = {
    "source_information": (
        8502,
        "unmeasured_source_contrasts",
        "Require positive raw/scalar/permutation paired Brier lower bounds and typed-cost noninferiority; exposed history earns at most circular_positive. Otherwise new independently labeled disjoint questions or a registered changed mechanism; stop unchanged tuning.",
    ),
    "objective_learning": (
        8505,
        "unmeasured_later_benefit",
        "Qualify Exp8503/8504; preserve V726 initial head and delayed feedback, require later Brier gains versus frozen and log-loss, typed costs, simultaneous retention upper95<=.01 and a weaker shuffled control.",
    ),
    "append_service": (
        8507,
        "unmeasured_qualified_service_speed",
        "Qualify explicit import and paired PRIMARY full clocks with lower95>1 and p95<=1.05. Five-input host storage does not establish nine-input inference or Rust/Python NFR-01; physical power loss remains open.",
    ),
    "current_qwen": (
        8508,
        "unmeasured_runtime",
        "Require a changed authenticated causal live CUDA runtime repair and bounded context/copy receipts before current Qwen generation.",
    ),
    "arc_cross_game_supervision": (
        8509,
        "qualified_effect_audit_null",
        "Require resolved applied-arm support across independent adapter-withheld games. Shadow redirects and censored outcomes cannot authorize refinements or solve credit.",
    ),
    "board_operations": (
        8510,
        "blocked_operations",
        "Require installed actual supported operations and complete device/transport clocks. GateMate also requires exact missing sources, dated cable/port/power change, IDCODE0x20000001, n16 flash and sample/hash smoke.",
    ),
}


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """The shipped schema retains gates and provenance; missing measurements receive no zero gain."""
    with patch.multiple(old, e=e, CLAIMS=CLAIMS):
        value = dict(old.build(work, receipts, raw, output))
    value.update(
        experiment_id=8511,
        run_date="20261014",
        random_seed=7318511,
        honest_verdict=f"complete_{value['verdict_class']}_v731_capstone",
        intended_count=13,
        censored_count=13 - value["completed_count"],
        task_count=13,
        canonical_tasks_sha256=work["authority"].get("canonical_tasks_sha256"),
        invocation_argv=work["invocation_argv"],
    )
    value["rows"][-1]["experiment_id"] = 8511
    value["acceptance_gates"]["all_slots_accounted"] = len(value["rows"]) == 13
    value["sample_size_budget"]["intended_scheduling_slots"] = 13
    for name, claim in value["scientific_claims"].items():
        if name in CLAIMS:
            claim.update(
                qualified_retrospective_benefit=None,
                missing_reason="required_scientific_measurement_absent"
                if name not in {"arc_cross_game_supervision", "board_operations"}
                else "no_resolved_benefit_or_physical_operation",
            )
    value["scientific_claims"]["objective_learning"]["retention_windows"] = [
        dict(window=i, metric=None, missing_reason="Exp8504/8505 not executed") for i in range(5)
    ]
    value["scientific_claims"]["source_information"]["primary_contrasts"] = [
        dict(control=c, gain=None, missing_reason="Exp8501/8502 not executed")
        for c in ["raw_p2", "scalar", "source_permutation"]
    ]
    previous = e.load(work["predecessors"][-1])
    value["scientific_claims"]["V730_preserved"] = dict(
        source=work["predecessors"][-1],
        original_verdict=previous["honest_verdict"],
        scientific_claims=previous["scientific_claims"],
        paper_ready_does_not_grant_science=previous["paper_ready"],
    )
    value["historical_reductions"] = work["historical_slots"]
    value["inherited_adversarial_findings"] = [
        dict(
            producer=r["task_id"],
            flagged_adversarial=r.get("upstream_flagged_adversarial"),
            findings=r.get("upstream_adversarial_findings", []),
            primary=r["primary_reference"],
        )
        for r in work["historical_slots"]
        if r.get("upstream_flagged_adversarial") or r.get("upstream_adversarial_findings")
    ]
    value["administrative_complete"] = len(value["rows"]) == 13
    value["scientific_eligibility"] = "blocked_required_upstream_science_unavailable"
    value["publication_authorized"] = False
    value["prior_failure_comparisons"] = []
    pins = {
        r["task_id"]: r["primary_reference"]
        for r in [*work["historical_slots"], *previous["rows"]]
        if r.get("primary_reference")
    }
    pins[previous["task_id"]] = work["predecessors"][-1]
    for ref in previous["source_artifact_hashes"]:
        name = Path(ref["path"]).name
        if name.startswith("experiment_") and int(name.split("_")[1]) in e.PREDECESSOR_PINS:
            if ref["sha256"] != e.PREDECESSOR_PINS[int(name.split("_")[1])]:
                raise ValueError("predecessor_pin")
            ancestor = e.load(ref)
            pins[ancestor["task_id"]] = ref
            pins.update(
                {
                    r["task_id"]: r["primary_reference"]
                    for r in ancestor["rows"]
                    if r.get("primary_reference")
                }
            )
    for task in work["tasks"]:
        for prior in task.get("prior_failures", []):
            ref = pins.get(prior["experiment_id"])
            original = e.load(ref) if ref and ref.get("exists", True) else None
            comparison = dict(
                task_id=task["id"],
                predecessor=prior["experiment_id"],
                reference=ref,
                expected_verdict=prior["verdict"],
                observed_verdict=original.get("honest_verdict") if original else None,
                exact_match=original.get("honest_verdict") == prior["verdict"]
                if original
                else None,
                retire_if_same_verdict=prior.get("retire_if_same_verdict"),
                missing_reason=None if original else "authenticated_predecessor_bytes_unavailable",
            )
            value["prior_failure_comparisons"].append(comparison)
    for row in value["rows"][:-1]:
        value["retirement_recommendations"].append(
            dict(
                task_id=row["task_id"],
                predecessor=row["primary_reference"],
                exact_verdict=row["honest_verdict"],
                recommendation="retire_exact_unchanged_attempt",
                hypothesis_retired=False,
                reopening_condition="Supply the branch's next-evidence conditions; unmeasured hypotheses remain open.",
            )
        )
    value["methodology_note"] = (
        "Read-only no-model aggregation with original sealed authorities and bounded primitive replay. Source panel remains exposed; new source contrasts, later learning and qualified append speed are missing. ARC effect nulls grant no solve credit. Generator weights, external services, conductor and active YAML are unchanged. Publication gate readiness is separate and grants no submission authorization."
    )
    value["field_principles"].update(
        {
            k: "Bind exact producer bytes and this invocation; administrative completion grants no scientific benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = e.checksum(value)
    return value

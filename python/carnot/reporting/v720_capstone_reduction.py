"""REQ-REPORT-8359: administrative readiness and observed benefit are separate."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v719_capstone_reduction as old
from carnot.reporting import v720_capstone_evidence as e

Json = dict[str, Any]
NEXT = [
    "Reuse authenticated heads and prediction seals only while their hashes and complete task authority match.",
    "Continue only with unchanged dense/sparse, crash and consumer checks under frozen historical operands.",
    "Require sealed later utility gain >=.02, lower97.5% bound >0 and retention within .02 cost/.01 Brier under the registered update budget.",
    "Require persistent capacity and loss invariants on a new registered trace; constructed gain remains outside natural H1/H2.",
    "Qualify the missing owned coverage statements before accepting the full128-slot H1 audit; preserve its independently replayed descriptive null.",
    "Qualify audit coverage and all sealed retention windows before accepting H2; new positive evidence must meet unchanged later-source and retention gates.",
    "Continue only if table fidelity holds for all registered resolutions and refresh costs remain explicit after a changed head.",
    "Require byte-bound historical authority consumers and changed CUDA substrate checks before a runtime readiness claim.",
    "Require qualified changed-runtime evidence before a bounded Qwen canary; V713 capture stays parked.",
    "Require new authenticated cross-game supervisor outcomes with overlapping support before any generalization claim.",
    "Require complete request-scale cost and transfer measurements plus actual supported KV260 device evidence; CPU timing alone cannot close NFR-01.",
    "Recover exact missing producer closure bytes; then require dated physical change, IDCODE0x20000001, n16 flash and device sample/hash smoke.",
    "Require complete immutable producer closures; successful replay preserves the producer's original failed or blocked disposition.",
    "Continue only with qualified sealed H1/H2 audits or new authenticated runtime/device evidence; exact repeated failed scope alone retires.",
]


def science(selected: Json, baseline: Json, eligible: bool) -> Json:
    """Keep absent observations absent and distinguish replayed diagnostics from qualification."""
    reduced = selected.get("independent_reduction")
    if reduced is None:
        return dict(
            baseline,
            status="blocked_unmeasured",
            primitive_reduction=None,
            qualified=False,
            development_signal_score=0,
        )
    return dict(
        reduced,
        status="qualified_measured" if eligible else "measured_unqualified_diagnostic",
        qualified=eligible,
        development_signal_score=int(
            eligible
            and reduced.get(
                "h1_development_signal_score", reduced.get("h2_development_signal_score", 0)
            )
        ),
        primitive_reduction=reduced,
        science_controls=selected["science_controls"],
        sealed_before_labels=selected["sealed_before_labels"],
        scope="descriptive exposed development; historical qualification failure preserved",
    )


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Extend the bounded V719 reducer using the same task and memory assertions.

    Scientific primitive diagnostics are replayed independently. Their producer
    qualification stays failed until its owned checks pass, so a diagnostic null
    does not erase a historical execution failure or retire a hypothesis.
    """
    with patch.object(old, "NEXT", NEXT), patch.object(old.e, "authority", e.authority):
        value = dict(old.reduce(work, receipts))
    rows = value["rows"]
    rows[-1].update(experiment_id=8359, task_id=e.TASK, unit_id=e.TASK)
    selected = [i["summary"]["selected"] for i in work["inputs"]]
    value["H1"] = science(selected[4], value["H1"], rows[4]["eligible"])
    value["H2"] = science(selected[5], value["H2"], rows[5]["eligible"])
    value["H1"].update(intended_count=128)
    value["H2"].update(
        intended_count=88,
        stream_intended_count=96,
        retention_intended_count=32,
        retention_windows=[0, 32, 64, 96],
    )
    value.update(
        science_ready_score=int(rows[4]["eligible"] and rows[5]["eligible"]),
        h1_development_signal_score=value["H1"]["development_signal_score"],
        h2_development_signal_score=value["H2"]["development_signal_score"],
        static_ready_score=int(
            rows[0]["eligible"] and selected[0].get("frozen_heads_ready_score") == 1
        ),
        durable_local_correctness_score=int(
            rows[1]["eligible"] and selected[1].get("local_kernel_ready_score") == 1
        ),
        actual_later_source_improvement_score=value["H2"]["development_signal_score"],
        next_evidence_conditions=NEXT,
        historical_v717=dict(
            disposition="historical_synthesis_only",
            references=work["historical_fixture_hashes"][:2],
        ),
        historical_v719=dict(
            disposition="historical_failed_replays_preserved",
            reference=work["historical_fixture_hashes"][2],
        ),
    )
    value["gate_check_summary"] = [
        g
        for g in value["gate_check_summary"]
        if g.get("artifact_field") != "sealed_primitive_science_rows"
    ]
    for index, field in [(4, "qualified_H1_primitive_rows"), (5, "qualified_H2_primitive_rows")]:
        if not rows[index]["eligible"]:
            value["gate_check_summary"].append(
                e.failure(
                    Path(work["root"]) / work["tasks"][index]["deliverable"],
                    field,
                    "sealed rows and passing owned producer qualification",
                    rows[index]["honest_verdict"],
                )
            )
    value["capacity_scope"] = dict(
        selected[3], included_in_H1_H2=False, natural_benefit=False, regret_guarantee=False
    )
    value["engineering_results"] = dict(
        capacity=selected[3],
        table_fidelity=selected[6],
        cpu_costs=selected[10],
        natural_decision_gain_proven=False,
    )
    value["board_obligations"] = [
        dict(
            board="KV260",
            terminal_scope="quadratic Ising k<=5 only",
            current=selected[10].get("board_obligations"),
            next_evidence_condition=NEXT[10],
        ),
        dict(
            board="PolarFire",
            terminal_scope="board-local Linux CPU only",
            current=selected[10].get("polarfire_graduation"),
            next_evidence_condition="Preserve authenticated Exp8259 CPU dispatch/output parity; fabric remains unmeasured.",
        ),
        dict(
            board="GateMate",
            terminal_scope="physical change required; history incomplete",
            current=selected[11].get("board_rows"),
            next_evidence_condition=NEXT[11],
        ),
    ]
    value["arc_support"] = dict(
        reference=work["inputs"][9]["reference"],
        disposition=rows[9]["disposition"],
        evidence=selected[9],
    )
    value["live_call_accounting"].update(
        canary=selected[8].get("model_invocation_counts"), bounded_generation_required=True
    )
    value["three_prd_gaps"] = [
        dict(gap=g, requirements=req, closed=False, next_evidence_condition=NEXT[i])
        for g, req, i in [
            ("useful_verified_decisions", ["FR-06", "FR-12"], 4),
            ("later_learning_and_retention", ["FR-11"], 5),
            ("request_scale_deployment", ["FR-05", "FR-08", "NFR-01"], 10),
        ]
    ]
    value["continuation_decision"] = (
        "Require qualified audit execution; replayed diagnostics do not establish natural decision gain."
    )
    for row, task, retirement in zip(rows, work["tasks"], value["retirements"], strict=True):
        retirement["comparisons"] = [
            dict(
                prior=p,
                observed_producer_verdict=row["honest_verdict"],
                exact_repeat=p["verdict"] == row["honest_verdict"],
                hypothesis_retired=False,
            )
            for p in task["prior_failures"]
        ]
        retirement["permanent"] = bool(retirement["same_verdict_entries"]) and all(
            p["authenticated"] for p in retirement["prior_evidence"]
        )
    value["gate_check_summary"] = [
        dict(
            g, operator=g.get("operator", g.get("op", "==")), sha256=g.get("sha256", g.get("hash"))
        )
        for g in value["gate_check_summary"]
    ]
    return value

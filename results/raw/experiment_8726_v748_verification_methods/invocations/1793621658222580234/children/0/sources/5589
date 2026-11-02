"""REQ-REPORT-8614: small deterministic reductions preserve scientific limits.

These calculations inspect historical operands. They do not load a language
model, repeat training, benchmark a service or create independent questions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import v739_capstone_evidence as e

Json = dict[str, Any]


def reduce() -> Json:
    """Recompute fixed samples and retain missing release and cost evidence as null."""
    from carnot.reporting import arc_outcome_identifiability_8599 as arc

    p = e.methods.protocol()
    learning = e.read(p["original_producers"]["8582"])
    predictions = e.read(learning["prediction_rows_reference"])["rows"][:2]
    losses = [
        dict(
            unit_id=r["unit_id"],
            brier=(r["probability"] - r["evaluator_target"]) ** 2,
            matches_original=(r["probability"] - r["evaluator_target"]) ** 2 == r["brier"],
            qualified=False,
        )
        for r in predictions
    ]
    ref = learning["retention_rows_reference"]
    if e.methods.sha256_file(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("retention_hash")
    retention = json.loads(Path(ref["path"]).read_bytes())
    maximum = max((r["probability"] - r["target"]) ** 2 - r["baseline_brier"] for r in retention)
    old_maximum = max(r["brier"] - r["baseline_brier"] for r in retention)
    lean = e.read(e.reference(e.ROOT / "results/experiment_8608_v739_lean_resident_prototype.json"))
    sessions = e.read(lean["work_reference"])["sessions"]
    clocks = []
    for session in sessions:
        spans = [r["spans"] for r in session["responses"]]
        total = sum(s["end_ns"] - s["start_ns"] for group in spans for s in group)
        union = sum(group[-1]["end_ns"] - group[0]["start_ns"] for group in spans)
        clocks.append(
            dict(
                arm=session["label"],
                operation_subtotal_ns=total,
                contiguous_interval_ns=union,
                matches_original=total == union,
                scope="prototype_operation_subtotal_not_complete_session_cost",
                qualified_cost=False,
            )
        )
    resident = e.read(p["original_producers"]["8596"])
    cost_row = resident["paired_session_rows"][0]
    primitive = e.read(cost_row["primitive_reference"])
    total = sum((s["end_ns"] - s["start_ns"]) / 1e9 for s in primitive["sessions"])
    panel = e.read(p["original_producers"]["8588"])
    panel_work = e.read(panel["work_reference"])
    original_ref = panel_work["references"][0]
    _, events, _ = arc.panel.panel.reduce_episode(original_ref)
    steps = [
        json.loads(line)
        for line in (Path(original_ref["directory"]) / "steps.jsonl").read_text().splitlines()
    ]
    horizons = arc.windows(steps, events)
    observer = e.read(p["original_producers"]["8599"])
    original = [
        r for r in observer["censoring_rows"] if r["episode_id"] == panel["rows"][0]["episode_id"]
    ]
    matches = all(all(r[k] == old[k] for k in r) for r, old in zip(horizons, original, strict=True))
    deadline = e.read(
        e.reference(e.ROOT / "results/experiment_8603_v739_deadline_edge_qualification.json")
    )
    deadline_work = e.read(deadline["primitive_reference"])
    coverage = e.read(deadline_work["coverage_reference"])
    runner = coverage["files"]["python/carnot/reporting/causal_brier_runner_8582.py"]
    return dict(
        causal_losses=losses,
        retention_maximum=dict(
            source=ref,
            count=len(retention),
            maximum_delta=maximum,
            matches_original=maximum == old_maximum,
            qualified=False,
        ),
        triviaplus_coverage=dict(
            intended_questions=128,
            observed_questions=None,
            missing_reason="Exp8604 pre-gated; Exp8607 primary absent",
            factual_correctness_established=False,
            energy_head_transfer_allowed=False,
            judge_confidence_available=False,
        ),
        complete_lean_cost=dict(
            total_s=None, missing_reason="Exp8610 pre-gated; no complete lean-session measurement"
        ),
        lean_prototype_sessions=clocks,
        historical_resident_total=dict(
            total_s=total,
            matches_original=total == cost_row["total_session_wall_s"],
            source=cost_row["primitive_reference"],
        ),
        arc_horizons=dict(
            rows=horizons,
            resolved_count=sum(r["resolved_by_levelup"] for r in horizons),
            matches_original=matches,
            original_refinement_eligible_score=observer["refinement_eligible_score"],
            observer_cannot_authorize_panel=True,
        ),
        deadline_failure=dict(
            primary=e.reference(
                e.ROOT / "results/experiment_8603_v739_deadline_edge_qualification.json"
            ),
            coverage_reference=deadline_work["coverage_reference"],
            missing_lines=runner["missing_lines"],
            failed_assertion="all(not f['summary']['missing_lines'] for f in coverage['files'].values())",
            field="files.python/carnot/reporting/causal_brier_runner_8582.py.summary.missing_lines",
            operator="==",
            expected=0,
            observed=runner["summary"]["missing_lines"],
        ),
        independent_units_added=0,
    )

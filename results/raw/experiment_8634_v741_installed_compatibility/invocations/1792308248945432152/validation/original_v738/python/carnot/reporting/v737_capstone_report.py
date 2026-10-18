"""REQ-REPORT-8590: qualification and benefit remain separate scientific decisions."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import v736_capstone_report as base
from carnot.reporting import v737_capstone_evidence as e
from carnot.reporting.v736_historical_inputs import checked

Json = dict[str, Any]
DECISIONS: Json = dict(
    fixed_brier_learning=dict(
        decision="close_unchanged_retry_unqualified_benefit",
        qualified_benefit=None,
        reopen="Isolate the runner deadline clock from the numerical kernel clock so original test_production_dispatch_and_deadline no longer exhausts its mocked iterator (StopIteration). Qualify the unchanged fixed Brier stream before Exp8583 benefit/retention audit. Exp8580/8581 do not repair Exp8567/8568 historical hashes.",
    ),
    resident_native_boundary=dict(
        decision="retain_qualified_parity_close_unqualified_cost_retry",
        qualified_benefit=None,
        reopen="Resolve Exp8584 METHODOLOGY_MISSING model_specs/target_model provenance rejection consistently with actual no-model CPU execution; qualify original inclusive resident operands through unchanged validators. Keep Exp8572 parity; quarantined speed cannot revise Exp8559 cold null.",
    ),
    fixed_arc_panel=dict(
        decision="close_qualified_null",
        qualified_benefit=False,
        reopen="Require a diagnosed changed supervisor mechanism or separately qualified causal CUDA/model-enabled runtime plus a fresh registered study; Exp8588 qualified null and refinement eligibility zero forbid unchanged retries.",
    ),
)


def build(work: Json, validation: Json, raw: Path, output: Path) -> Json:
    """Reuse the original row schema while retaining only evidence-qualified continuation."""
    with patch.object(base, "e", e):
        value = dict(base.build(work, validation, raw, output))
    slots = work["slots"]
    by = {r["experiment_id"]: r for r in slots}
    prior = e.read(work["protocol"]["original_producers"]["8578"])
    value.update(
        experiment_id=8590,
        run_date="20261017",
        random_seed=7378590,
        intended_count=12,
        task_count=12,
        milestone_status="twelve_scheduling_slots_accounted_scientific_branches_blocked_or_closed",
        quarantined_inputs=prior["quarantined_inputs"]
        + [r for r in slots if r["verdict_class"] == "disqualified"],
        preserved_nulls=prior["preserved_nulls"],
        qualified_scientific_results=prior["preserved_nulls"]
        + [r for r in slots if r["eligible"] and r["verdict_class"] == "null"],
        continuation_decisions=DECISIONS,
        retirement_recommendations=[dict(scope=k, **v) for k, v in DECISIONS.items()],
        count_revision_chain=dict(
            historical_rejection_preserved=True,
            historical_references=[
                work["protocol"]["original_producers"][str(n)] for n in [8567, 8568]
            ],
            current_references=[by[n]["primary_reference"] for n in [8580, 8581]],
            historical_hashes_repaired=False,
        ),
        conditional_exposed_benefits=dict(
            learning=None,
            resident_native=None,
            arc=False,
            status="learning_and_resident_disqualified_arc_qualified_null",
            generalization_credit=0,
        ),
        methodology_note="Twelve V737 slots reconcile original terminal flags and immutable operands. Current validator qualification preserves historical revision rejection. Small original learning, resident-clock and current ARC arithmetic is diagnostic only. Exp8582 learning and Exp8584 speed remain quarantined; Exp8588 is a qualified null. HalluTruthQA audit readiness grants no independent evaluation or deployment permission. No model or hardware benchmark runs. G1-G4 retain original definitions and grant no external publication permission.",
    )
    own = value["rows"][-1]
    own.update(
        experiment_id=8590, honest_verdict=f"complete_{value['verdict_class']}_v737_capstone"
    )
    value["honest_verdict"] = own["honest_verdict"]
    value["sample_size_budget"]["intended_scheduling_slots"] = 12
    value["acceptance_gates"].update(
        all_slots_accounted=len(value["rows"]) == 12,
        qualified_arc_panel=by[8588]["eligible"],
    )
    release = e.read(by[8585]["primary_reference"])
    value["hallutruth_release_eligibility"] = {
        k: release[k]
        for k in [
            "release_audit_ready_score",
            "human_label_evidence_score",
            "unseen_question_eligible_score",
            "unseen_generator_eligible_score",
            "deployment_permission_score",
            "natural_error_evidence_score",
            "rights_reference",
            "annotation_process",
            "future_protocol_conditions",
        ]
    }
    value["hallutruth_release_eligibility"].update(
        primary_reference=by[8585]["primary_reference"],
        qualified=by[8585]["eligible"],
        limitations=release["source_metadata"]["limitations"],
        language="Arabic",
        single_historical_generator=True,
        controlled_construction=True,
        usable_independent_truth=False,
    )
    value["historical_model_provenance"] = value["historical_model_provenance"] + [
        dict(experiment_id=r["experiment_id"], provenance=r.get("historical_model_provenance", []))
        for r in slots
    ]
    value["cited_upstream_artifacts"] += [r["primary_reference"] for r in slots]
    value["field_principles"].update(
        {
            k: "Bind this invocation to original operands and qualification; audit completion, repeated observations and exposed nulls grant no independent truth or benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = e.checksum(value)
    return value


def replay(path: Path) -> bool:
    """Reconstruct original claims so repaired envelope hashes cannot manufacture evidence."""
    try:
        value = json.loads(path.read_bytes())
        work = e.read(value["primitive_reference"])
        if (
            e.checksum(value) != value["reproducibility_checksum"]
            or e.methods.protocol() != work["protocol"]
            or work["duration_s"] != (work["end_ns"] - work["start_ns"]) / 1e9
            or not 0 <= work["duration_s"] < 3600
        ):
            return False
        for ref in work["source_refs"] + work["code_refs"]:
            if ref.get("exists", True) and not checked(
                dict(ref, path=ref.get("snapshot_path", ref["path"]))
            ):
                return False
        if e.methods.authority(Path(work["root"])) != work["contract"]:
            return False
        if (
            json.loads(Path(work["publication_receipt"]["stdout_path"]).read_bytes())
            != work["publication"]
        ):
            return False
        if value != build(
            work,
            value["validation_reference"],
            Path(value["primitive_reference"]["path"]).parent,
            Path(value["publication_output"]),
        ):
            return False
        tasks = work["protocol"]["task_map_v737"]
        log = next(
            r
            for r in work["source_refs"]
            if r["path"].endswith("ops/conductor-log.md") and "snapshot_path" in r
        )
        lines = Path(log["snapshot_path"]).read_text().splitlines()
        for row, task in zip(work["slots"], tasks[:-1], strict=True):
            ref = row["primary_reference"]
            summary = e.outcome(task, ref, Path(ref["path"]).parent) if ref["exists"] else None
            cascade = next(
                (
                    s
                    for s in lines
                    if "GATE_BLOCK" in s and s.split("|")[2].strip() == task["title"][:50].strip()
                ),
                None,
            )
            if row != e.old.slot(task, ref, cascade, summary):
                return False
        with TemporaryDirectory(prefix="exp8590-replay-", dir="/var/tmp") as directory:
            diagnostic, receipt = e.samples(Path(directory))
        if not receipt["passed"] or work["samples"] != diagnostic:
            return False
        for receipt in [
            *work["branch_receipts"],
            *value["validation_receipts"],
            work["publication_receipt"],
        ]:
            original = Path(receipt["stdout_path"]).with_name(receipt["name"] + ".receipt.json")
            if json.loads(original.read_bytes()) != receipt:
                return False
            for stream in ["stdout", "stderr"]:
                if not checked(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                ):
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def consumer(output: Path) -> Json:
    """Both unchanged readers must select exactly the published primary bytes."""
    with patch.object(base, "e", e):
        return dict(base.consumer(output))

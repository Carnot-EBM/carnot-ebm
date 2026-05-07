"""Tests for Exp 1458 HardNet++/DSP repair-stack consolidation.

Spec: REQ-REPORT-044, SCENARIO-REPORT-044.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.hardnet_dsp_repair_stack_retirement import (
    EXCLUSION_MARKER,
    REQUIRED_ARTIFACT_FIELDS,
    _manifest_contains_block,
    _relative_path,
    build_artifact,
    ensure_manifest_block,
    render_consolidation_note,
    run,
    write_in_progress_artifact,
)


def _review_rows() -> list[dict[str, object]]:
    return [
        {
            "experiment_id": "exp1147",
            "title": "HardNet++-Style Projection Repair Layer",
            "artifact_path": "results/experiment_1147_hardnet_projection_repair.json",
            "verdict": "projection_accurate_and_fast",
            "evidence": "20/20 violations repaired with projection_repair_accuracy=1.0",
            "lesson": "Hard projection can certify feasibility in numeric repair cases.",
        },
        {
            "experiment_id": "exp1275",
            "title": "FSNet Feasibility Step",
            "artifact_path": "results/experiment_1275_fsnet_feasibility_step_continuous_ebm.json",
            "verdict": "feasibility_step_viable",
            "evidence": "violation_count_mean dropped from 5.0 to 0.0",
            "lesson": "Feasibility seeking belongs in the repair loop before text polish.",
        },
        {
            "experiment_id": "exp1276",
            "title": "SnareNet Repair Layer",
            "artifact_path": "results/experiment_1276_snarenet_repair_layer_gated.json",
            "verdict": "adaptive_repair_improves_fsnet",
            "evidence": "repair_delta_over_fsnet=0.2199604492292856",
            "lesson": "Adaptive repair can help, but it remained a local repair operator.",
        },
        {
            "experiment_id": "exp1291",
            "title": "HardNet++ Nonlinear Repair Benchmark",
            "artifact_path": "results/experiment_1291_hardnetpp_nonlinear_repair_benchmark.json",
            "verdict": "hardnetpp_nonlinear_repair_viable",
            "evidence": "hardnetpp_delta_over_snarenet=1.2207222442957435",
            "lesson": "Nonlinear projection beats repeated local-linear repair on residual cases.",
        },
        {
            "experiment_id": "exp1292",
            "title": "DSP Feasibility-Channel Diagnostic",
            "artifact_path": "results/experiment_1292_dsp_feasibility_channel_diagnostic.json",
            "verdict": "feasibility_channel_predictive_marginal",
            "evidence": "feasibility_channel_auc=0.6604651162790698 and false_continue_rate=0.7714",
            "lesson": "DSP phi is useful telemetry but not a decisive learned stop rule.",
        },
        {
            "experiment_id": "exp1305",
            "title": "HardNet++ + DSP Feasibility Stop Policy",
            "artifact_path": "results/experiment_1305_hardnetpp_dsp_feasibility_stop_policy.json",
            "verdict": "conservative replay policy useful, DSP marginal",
            "evidence": "policy_stop_accuracy=1.0 via conservative replay",
            "lesson": "Conservative replay is the retained operator gate.",
        },
        {
            "experiment_id": "exp1318",
            "title": "HardNet++/DSP Learned Stop Policy",
            "artifact_path": "results/experiment_1318_hardnetpp_dsp_learned_stop_policy.json",
            "verdict": "learned policy matched conservative replay",
            "evidence": "hardnetpp_delta_over_replay_policy=0.0",
            "lesson": "The learned policy did not prove broad generalization beyond replay.",
        },
    ]


def _papers() -> list[dict[str, str]]:
    return [
        {
            "name": "HardNet++",
            "citation": "arXiv:2604.19669",
            "lesson": "Nonlinear hard projection validates feasibility-first repair.",
        },
        {
            "name": "KKT-Hardnet",
            "citation": "arXiv:2507.08124",
            "lesson": "KKT projection is a future alternative if equality/inequality residuals recur.",
        },
        {
            "name": "SnareNet",
            "citation": "arXiv:2602.09317",
            "lesson": "Adaptive repair layers belong in continuous-latent experiments, not variants.",
        },
        {
            "name": "Differentiable Symbolic Planning",
            "citation": "arXiv:2604.02350",
            "lesson": "Feasibility channels are useful signals but need non-replay validation.",
        },
    ]


def test_scenario_report_044_builds_artifact_manifest_and_note() -> None:
    """SCENARIO-REPORT-044: Exp 1458 retires HardNet++/DSP variants."""

    manifest, block_added = ensure_manifest_block("retired_extras:\n")
    artifact = build_artifact(
        review_rows=_review_rows(),
        consolidation_note_path="ops/lineage-retirements/hardnet_dsp_repair_stack_retired.md",
        manifest_text=manifest,
        manifest_block_added=block_added,
        cited_recent_constraint_papers=_papers(),
    )
    note = render_consolidation_note(_review_rows(), artifact)

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["hardnet_dsp_experiments_reviewed"] == [
        "exp1147",
        "exp1275",
        "exp1276",
        "exp1291",
        "exp1292",
        "exp1305",
        "exp1318",
    ]
    assert artifact["hardnet_dsp_lineage_retired"] is True
    assert artifact["exclusion_manifest_updated"] is True
    assert artifact["exclusion_manifest_block_added"] is True
    assert "conservative replay" in " ".join(artifact["lessons_retained"]).lower()
    assert "operator explicitly reopens" in " ".join(artifact["future_reopen_conditions"])
    assert artifact["honest_verdict"] == (
        "hardnet_dsp_lineage_retired_conservative_replay_retained_no_new_variants"
    )

    assert EXCLUSION_MARKER in manifest
    assert "HardNet++/DSP" in manifest
    assert "operator explicitly reopens the line" in manifest
    assert "Hard Constraint Lesson" in note
    assert "arXiv:2604.19669" in note
    assert "conservative replay" in note


def test_req_report_044_manifest_block_is_idempotent() -> None:
    """REQ-REPORT-044: exclusion manifest receives one durable scope block."""

    manifest, first_added = ensure_manifest_block("retired_extras:\n")
    manifest_again, second_added = ensure_manifest_block(manifest)
    blank_manifest, blank_added = ensure_manifest_block("")

    assert first_added is True
    assert second_added is False
    assert manifest_again == manifest
    assert manifest_again.count(EXCLUSION_MARKER) == 1
    assert _manifest_contains_block(manifest_again) is True
    assert _manifest_contains_block("retired_extras:\n") is False
    assert blank_added is True
    assert blank_manifest.startswith("retired_extras:")


def test_req_report_044_run_writes_bootstrap_note_manifest_and_terminal_artifact(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-044: run writes bootstrap, note, manifest block, and JSON."""

    out_path = tmp_path / "results" / "experiment_1458_hardnet_dsp_repair_stack_consolidation.json"
    note_path = tmp_path / "ops" / "lineage-retirements" / "hardnet_dsp_repair_stack_retired.md"
    manifest_path = tmp_path / "ops" / "exclusion_manifest.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("retired_extras:\n", encoding="utf-8")

    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    artifact = run(
        root=tmp_path,
        out_path=out_path,
        note_path=note_path,
        manifest_path=manifest_path,
        review_rows=_review_rows(),
        cited_recent_constraint_papers=_papers(),
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    note = note_path.read_text(encoding="utf-8")
    manifest = manifest_path.read_text(encoding="utf-8")

    assert artifact == written
    assert written["status"] == "complete"
    assert written["consolidation_note_path"] == (
        "ops/lineage-retirements/hardnet_dsp_repair_stack_retired.md"
    )
    assert written["hardnet_dsp_lineage_retired"] is True
    assert EXCLUSION_MARKER in manifest
    assert "Future Reopen Conditions" in note
    assert _relative_path(note_path) == (
        "ops/lineage-retirements/hardnet_dsp_repair_stack_retired.md"
    )
    assert (
        _relative_path(out_path)
        == "results/experiment_1458_hardnet_dsp_repair_stack_consolidation.json"
    )
    assert _relative_path(tmp_path / "loose.md") == "loose.md"

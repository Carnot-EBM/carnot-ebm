"""Tests for the Exp 1467 `.113` activation manifest.

Spec: REQ-REPORT-048, SCENARIO-REPORT-048.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_113_activation_manifest import (
    ALLOWED_113_TRACKS,
    FORBIDDEN_REOPEN_TRACKS,
    REQUIRED_ARTIFACT_FIELDS,
    _read_json,
    _read_text,
    _relative_path,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _retro_payload() -> dict[str, object]:
    return {
        "status": "complete",
        "milestone": "2026.04.112",
        "criteria_met": 14,
        "criteria_total": 14,
        "scope_reduction_compliance_met": True,
        "honest_verdict": "milestone_112_14_of_14_criteria_met_scope_reduction_satisfied",
        "retired_lineages": [
            {
                "lineage": "GRPO/VPRM",
                "source_experiment": "exp1456",
                "evidence_path": (
                    "results/experiment_1456_grpo_vprm_lineage_consolidation_retirement.json"
                ),
                "honest_verdict": "grpo_vprm_lineage_retired_no_v15_without_operator_reopen",
            },
            {
                "lineage": "WOPR puzzle cartridges",
                "source_experiment": "exp1457",
                "evidence_path": "results/experiment_1457_wopr_puzzle_cartridge_retirement.json",
                "honest_verdict": (
                    "wopr_puzzle_lineage_retired_demo_assets_preserved_no_new_gallery_work"
                ),
            },
            {
                "lineage": "HardNet++/DSP repair stack",
                "source_experiment": "exp1458",
                "evidence_path": "results/experiment_1458_hardnet_dsp_repair_stack_consolidation.json",
                "honest_verdict": (
                    "hardnet_dsp_lineage_retired_conservative_replay_retained_no_new_variants"
                ),
            },
            {
                "lineage": "repair executor validation-error context",
                "source_experiment": "exp1464",
                "evidence_path": "results/experiment_1464_repair_validation_error_context_ab.json",
                "honest_verdict": "complete_no_retry_context_improvement_repair_executor_retired",
            },
        ],
        "carry_forward_tracks": [
            {
                "track": "runtime",
                "status": "ready",
                "source_experiment": "exp1463",
                "rule": "Preserve local SOTA GGUF runtime as a precondition.",
            },
            {
                "track": "self_learning",
                "status": "pivot_selected",
                "source_experiment": "exp1459",
                "rule": "Allow only one bounded exp1447-style fresh verified growth follow-up.",
            },
            {
                "track": "benchmark_adoption",
                "source_experiment": "exp1465",
                "rule": "Adopt at most the one minimal BEAVER-style bounds smoke task.",
            },
        ],
        "research_roadmap_yaml_modified": False,
        "scripts_research_conductor_modified": False,
    }


def _research_complete_with_112() -> str:
    return """
- id: 2026.04.111
  title: Prior
- id: 2026.04.112
  title: Scope Reduction + Local SOTA Runtime Repair + Gated Repair Salvage
  completed: '2026-05-07'
"""


def _conductor_log_text() -> str:
    rows = []
    titles = {
        1453: ".112 Scope-Reduction Activation Manifest",
        1454: "Experiment Artifact Signal/Noise Classifier",
        1455: "Known-Issues Mandatory Priority Audit",
        1456: "GRPO/VPRM Lineage Consolidation + Retirement",
        1457: "WOPR Puzzle Cartridge Retirement",
        1458: "HardNet++/DSP Repair Stack Consolidation",
        1459: "Self-Learning Non-Headline Lineage Decision",
        1460: "Hardware Portfolio Narrowing",
        1461: "Comparator Integration Cite/Retire Audit",
        1462: "Paper-v6 Anchored Claims Narrowing",
        1463: "Local SOTA GGUF Runtime Repair",
        1464: "Repair Validation-Error-as-Context A/B",
        1465: "External Verifier Benchmark Fit Audit",
        1466: "Milestone .112 Retrospective",
    }
    for exp_id, title in titles.items():
        rows.append(
            f"| 2026-05-07 05:00 UTC | {title} exp{exp_id} | OK | focused tests passed |"
        )
    return "\n".join(rows)


def _exclusion_manifest_text() -> str:
    return """
retired:
  - id: grpo_vprm_v15_scope_closed
    blocked_patterns: ["GRPO v15", "VPRM v15"]
    retired_milestone: "2026.04.112"
    operator_reopen_required: true
  - id: wopr_puzzle_cartridge_research_scope_closed
    blocked_patterns: ["WOPR puzzle cartridge"]
    retired_milestone: "2026.04.112"
    operator_reopen_required: true
  - id: hardnet_dsp_repair_stack_scope_closed
    blocked_patterns: ["HardNet++", "DSP feasibility channel"]
    retired_milestone: "2026.04.112"
    operator_reopen_required: true
"""


def test_scenario_report_048_activates_113_and_preserves_retirements() -> None:
    """SCENARIO-REPORT-048: .113 activation preserves .112 retired-lineage blocks."""

    artifact, manifest = build_artifact(
        retro=_retro_payload(),
        conductor_log_text=_conductor_log_text(),
        research_complete_text=_research_complete_with_112(),
        research_roadmap_text="milestone: 2026.04.113\n",
        exclusion_manifest_text=_exclusion_manifest_text(),
        active_priorities_text="Active priority count: `7`",
        manifest_path="ops/milestone_113_activation_manifest.md",
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.04.113"
    assert artifact["predecessor_milestone"] == "2026.04.112"
    assert artifact["criteria_met"] == 14
    assert artifact["criteria_total"] == 14
    assert artifact["research_complete_has_112_entry"] is True
    assert artifact["research_complete_archive_update_needed"] is False
    assert artifact["activation_manifest_complete"] is True
    assert artifact["retired_lineages_preserved"] is True
    assert [track["track"] for track in artifact["allowed_113_tracks"]] == [
        track["track"] for track in ALLOWED_113_TRACKS
    ]
    assert [track["track"] for track in artifact["forbidden_reopen_tracks"]] == [
        track["track"] for track in FORBIDDEN_REOPEN_TRACKS
    ]
    assert artifact["conductor_log_exp1453_to_exp1466"]["missing_experiments"] == []
    assert artifact["no_change_confirmations"] == {
        "research-roadmap.yaml": "unchanged_by_exp1467_activation_workflow",
        "scripts/research_conductor.py": "unchanged_by_exp1467_activation_workflow",
    }
    assert "Live SOTA Telemetry" in manifest
    assert "GRPO/VPRM" in manifest
    assert "operator-reopened" in manifest


def test_req_report_048_records_research_complete_archive_gap() -> None:
    """REQ-REPORT-048: absent .112 archive row is reported as an explicit gap."""

    artifact, manifest = build_artifact(
        retro=_retro_payload(),
        conductor_log_text=_conductor_log_text(),
        research_complete_text="- id: 2026.04.111\n",
        research_roadmap_text="milestone: 2026.04.113\n",
        exclusion_manifest_text=_exclusion_manifest_text(),
        active_priorities_text="",
        manifest_path="ops/milestone_113_activation_manifest.md",
    )

    assert artifact["status"] == "complete"
    assert artifact["research_complete_has_112_entry"] is False
    assert artifact["research_complete_archive_update_needed"] is True
    assert artifact["archive_gap"] == {
        "missing_milestone": "2026.04.112",
        "recommended_action": (
            "append .112 archive row to research-complete.yaml without modifying research-roadmap.yaml"
        ),
    }
    assert "Archive gap: `research-complete.yaml` lacks `2026.04.112`." in manifest


def test_req_report_048_blocks_when_retirement_blocks_are_missing() -> None:
    """REQ-REPORT-048: missing retired-lineage evidence blocks activation completion."""

    retro = _retro_payload()
    retro["retired_lineages"] = []
    artifact, manifest = build_artifact(
        retro=retro,
        conductor_log_text="exp1453 OK\n",
        research_complete_text=_research_complete_with_112(),
        research_roadmap_text="",
        exclusion_manifest_text="retired: []",
        active_priorities_text="",
        manifest_path="ops/milestone_113_activation_manifest.md",
    )

    assert artifact["status"] == "blocked"
    assert artifact["activation_manifest_complete"] is False
    assert artifact["retired_lineages_preserved"] is False
    assert "missing retired-lineage blocks" in artifact["blocked_reasons"][0]
    assert "Manifest blocked" in manifest


def test_req_report_048_run_writes_bootstrap_manifest_and_terminal_json(tmp_path: Path) -> None:
    """REQ-REPORT-048: run writes bootstrap, markdown, and terminal artifact."""

    out_path = tmp_path / "results" / "experiment_1467_112_completion_archive_113_activation.json"
    manifest_path = tmp_path / "ops" / "milestone_113_activation_manifest.md"

    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    _write_json(tmp_path / "results" / "experiment_1466_milestone_112_retro.json", _retro_payload())
    (tmp_path / "ops").mkdir(exist_ok=True)
    (tmp_path / "ops" / "conductor-log.md").write_text(_conductor_log_text(), encoding="utf-8")
    (tmp_path / "ops" / "exclusion_manifest.yaml").write_text(
        _exclusion_manifest_text(),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "active-priorities.md").write_text(
        "Active priority count: `7`\n",
        encoding="utf-8",
    )
    (tmp_path / "research-complete.yaml").write_text(
        _research_complete_with_112(),
        encoding="utf-8",
    )
    (tmp_path / "research-roadmap.yaml").write_text(
        "milestone: 2026.04.113\n",
        encoding="utf-8",
    )

    artifact = run(root=tmp_path, out_path=out_path, manifest_path=manifest_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    manifest = manifest_path.read_text(encoding="utf-8")

    assert artifact == written
    assert written["status"] == "complete"
    assert written["manifest_path"] == "ops/milestone_113_activation_manifest.md"
    assert written["source_inputs_read"]["ops/active-priorities.md"]["exists"] is True
    assert "Allowed .113 Tracks" in manifest
    assert "Forbidden Reopen Tracks" in manifest


def test_req_report_048_defensive_helpers_and_incomplete_retro(tmp_path: Path) -> None:
    """REQ-REPORT-048: helpers and incomplete predecessor evidence stay explicit."""

    assert _read_json(tmp_path / "missing.json") is None
    assert _read_text(tmp_path / "missing.md") == ""
    assert _relative_path(tmp_path / "ops" / "manifest.md") == "ops/manifest.md"
    assert _relative_path(tmp_path / "results" / "artifact.json") == "results/artifact.json"
    assert _relative_path(tmp_path / "loose.txt") == "loose.txt"

    artifact, manifest = build_artifact(
        retro={"milestone": "2026.04.112", "criteria_met": 13, "criteria_total": 14},
        conductor_log_text="",
        research_complete_text="",
        research_roadmap_text="",
        exclusion_manifest_text="",
        active_priorities_text="",
        manifest_path="ops/milestone_113_activation_manifest.md",
    )

    assert artifact["status"] == "blocked"
    assert artifact["criteria_met"] == 13
    assert artifact["criteria_total"] == 14
    assert artifact["conductor_log_exp1453_to_exp1466"]["ok_count"] == 0
    assert "predecessor retro criteria not complete" in artifact["blocked_reasons"]
    assert "Manifest blocked" in manifest

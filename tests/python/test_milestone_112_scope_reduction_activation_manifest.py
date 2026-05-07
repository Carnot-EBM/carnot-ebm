"""Tests for the Exp 1453 `.112` scope-reduction activation manifest.

Spec: REQ-REPORT-039, SCENARIO-REPORT-039.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_112_scope_reduction_activation_manifest import (
    REQUIRED_ARTIFACT_FIELDS,
    SCOPE_REQUIREMENTS,
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
        "milestone": "2026.04.111",
        "criteria_met": 10,
        "criteria_total": 14,
        "honest_verdict": (
            "milestone_111_10_of_14_criteria_met_threshold_not_met_live_sota_"
            "runtime_gate_blocked_repair_scale_carry_forward"
        ),
        "carry_forward_tracks": [
            {
                "id": "live_sota_runtime_repair_gate",
                "title": "Fix live local SOTA GGUF runtime before repair v3.",
                "next_rule": (
                    "Do not launch repair v3, energy reranking, or 100-case scale-up "
                    "until a mandated local SOTA GGUF model completes live inference."
                ),
                "prior_failures": [
                    {
                        "experiment_id": "exp1442",
                        "verdict": "blocked_no_live_sota_runtime",
                        "evidence_path": (
                            "results/experiment_1442_live_sota_repair_runtime_preflight.json"
                        ),
                    }
                ],
                "retire_if_same_verdict": True,
            },
            {
                "id": "repair_v3_and_prescale_gated_missing",
                "title": "Treat repair-v3 and pre-scale artifacts as gate-blocked.",
                "next_rule": (
                    "A .112 repair scale task must name the live-runtime fix and cannot "
                    "reuse the same gate-blocked path as success evidence."
                ),
                "prior_failures": [
                    {
                        "experiment_id": "exp1443",
                        "verdict": "missing_artifact_gate_blocked_by_exp1442",
                        "evidence_path": (
                            "results/experiment_1443_live_sota_dccd_semctrl_repair_v3.json"
                        ),
                    },
                    {
                        "experiment_id": "exp1445",
                        "verdict": "missing_artifact_gate_blocked_by_exp1443_exp1444",
                        "evidence_path": (
                            "results/experiment_1445_full_pipeline_v5_100case_prescale.json"
                        ),
                    },
                ],
                "retire_if_same_verdict": True,
            },
            {
                "id": "prm_process_agent_no_improvement",
                "title": "Retire saturated PRM selector pools unless the pool changes.",
                "next_rule": (
                    "Future PRM process-agent work must unsaturate the candidate pool "
                    "or target false-acceptance reduction."
                ),
                "prior_failures": [
                    {
                        "experiment_id": "exp1448",
                        "verdict": (
                            "complete_prmv3_no_headline_improvement_prototype_candidate_"
                            "pool_no_headline_claim"
                        ),
                        "evidence_path": (
                            "results/experiment_1448_prm_v3_online_process_reward_agent.json"
                        ),
                    }
                ],
                "retire_if_same_verdict": True,
            },
        ],
    }


def _known_issues_scope_text() -> str:
    return """
### NEW 2026-05-06 (16:30Z): SCOPE REDUCTION MILESTONE (.111)
The conductor planner reading this entry MUST allocate at least 8 of ~13 tasks
to scope reduction work, NOT to new experiment lineages.

exp_NEXT_SCOPE_A: Experiment artifact classifier
exp_NEXT_SCOPE_B: GRPO lineage consolidation + retirement
exp_NEXT_SCOPE_C: WOPR puzzle cartridge retirement
exp_NEXT_SCOPE_D: known-issues.md MANDATORY priority audit
exp_NEXT_SCOPE_E: Paper-v6 anchored-claims narrowing
exp_NEXT_SCOPE_F: Self-learning `_improved_non_headline` lineage decision
exp_NEXT_SCOPE_G: Hardware portfolio narrowing
exp_NEXT_SCOPE_H: Comparator-integration audit
"""


def _roadmap_text() -> str:
    return " ".join(f"exp{experiment_id}" for experiment_id in range(1453, 1467))


def test_scenario_report_039_maps_every_scope_reduction_requirement() -> None:
    """SCENARIO-REPORT-039: .112 maps the mandatory scope-reduction directive."""

    artifact, manifest = build_artifact(
        retro=_retro_payload(),
        known_issues_text=_known_issues_scope_text(),
        roadmap_text=_roadmap_text(),
        roadmap_next_text="",
        exclusion_manifest_text="retired: []",
        manifest_path="ops/milestone_112_scope_reduction_manifest.md",
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.04.112"
    assert artifact["prior_milestone"] == "2026.04.111"
    assert artifact["scope_reduction_required"] is True
    assert artifact["required_scope_reduction_task_count"] == 8
    assert artifact["planned_scope_reduction_task_count"] == 10
    assert artifact["planned_scope_task_ids"] == [row["task_id"] for row in SCOPE_REQUIREMENTS]
    assert artifact["scope_reduction_manifest_complete"] is True

    task_by_requirement = {
        row["requirement_id"]: row["task_id"] for row in artifact["scope_manifest_rows"]
    }
    assert task_by_requirement["known_issues_priority_audit"] == "exp1455"
    assert task_by_requirement["grpo_vprm_retirement"] == "exp1456"
    assert task_by_requirement["wopr_puzzle_retirement"] == "exp1457"
    assert task_by_requirement["hardware_portfolio_narrowing"] == "exp1460"

    forbidden_ids = {
        item["forbidden_scope_id"] for item in artifact["forbidden_exact_expansions"]
    }
    assert {
        "grpo_v15",
        "wopr_puzzle_cartridges",
        "hardnet_dsp_variants",
        "broad_comparator_hardware_branches",
    } <= forbidden_ids
    assert "| requirement | mapped task id | deliverable path | acceptance field | retire/block rule |" in manifest
    assert "GRPO v15" in manifest
    assert "HardNet++/DSP" in manifest


def test_req_report_039_carries_forward_live_sota_blockers() -> None:
    """REQ-REPORT-039: live-SOTA runtime rules gate repair reruns."""

    artifact, manifest = build_artifact(
        retro=_retro_payload(),
        known_issues_text=_known_issues_scope_text(),
        roadmap_text=_roadmap_text(),
        roadmap_next_text="",
        exclusion_manifest_text="retired: []",
        manifest_path="ops/milestone_112_scope_reduction_manifest.md",
    )

    carryforward_ids = {item["id"] for item in artifact["carryforward_from_111"]}
    assert "live_sota_runtime_repair_gate" in carryforward_ids
    assert "repair_v3_and_prescale_gated_missing" in carryforward_ids
    live_rule_ids = {item["id"] for item in artifact["live_sota_runtime_carryforward_rules"]}
    assert live_rule_ids == {
        "live_sota_runtime_repair_gate",
        "repair_v3_and_prescale_gated_missing",
    }
    assert artifact["live_sota_runtime_carryforward_rules"][0]["prior_failures"][0][
        "verdict"
    ] == "blocked_no_live_sota_runtime"
    assert "Do not launch repair v3, energy reranking, or 100-case scale-up" in manifest
    assert artifact["no_change_confirmations"] == {
        "scripts/research_conductor.py": "unchanged_by_exp1453_activation_workflow",
        "research-roadmap.yaml": "unchanged_by_exp1453_activation_workflow",
    }


def test_req_report_039_blocks_completion_without_scope_directive() -> None:
    """REQ-REPORT-039: missing mandatory directive prevents a complete verdict."""

    artifact, manifest = build_artifact(
        retro=_retro_payload(),
        known_issues_text="no scope directive here",
        roadmap_text="exp1453 exp1454",
        roadmap_next_text="",
        exclusion_manifest_text="",
        manifest_path="ops/milestone_112_scope_reduction_manifest.md",
    )

    assert artifact["status"] == "blocked"
    assert artifact["scope_reduction_required"] is False
    assert artifact["scope_reduction_manifest_complete"] is False
    assert artifact["missing_mapped_task_ids_in_roadmap"] == [
        "exp1455",
        "exp1456",
        "exp1457",
        "exp1458",
        "exp1459",
        "exp1460",
        "exp1461",
        "exp1462",
    ]
    assert "Manifest is blocked" in manifest


def test_req_report_039_run_writes_bootstrap_manifest_and_terminal_json(tmp_path: Path) -> None:
    """REQ-REPORT-039: run writes bootstrap, markdown, and terminal artifact."""

    out_path = tmp_path / "results" / "experiment_1453_112_scope_reduction_activation_manifest.json"
    manifest_path = tmp_path / "ops" / "milestone_112_scope_reduction_manifest.md"

    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    _write_json(tmp_path / "results" / "experiment_1452_milestone_111_retro.json", _retro_payload())
    (tmp_path / "ops").mkdir(exist_ok=True)
    (tmp_path / "ops" / "known-issues.md").write_text(
        _known_issues_scope_text(),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "exclusion_manifest.yaml").write_text("retired: []\n", encoding="utf-8")
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        _roadmap_text(),
        encoding="utf-8",
    )
    (tmp_path / "research-roadmap.yaml").write_text("active roadmap\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "research_conductor.py").write_text("# unchanged\n", encoding="utf-8")

    artifact = run(root=tmp_path, out_path=out_path, manifest_path=manifest_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    manifest = manifest_path.read_text(encoding="utf-8")

    assert artifact == written
    assert written["status"] == "complete"
    assert written["scope_reduction_manifest_path"] == (
        "ops/milestone_112_scope_reduction_manifest.md"
    )
    assert written["source_inputs_read"]["research-roadmap-next.yaml"]["exists"] is False
    assert "Forbidden Exact Expansions" in manifest
    assert "No-Change Confirmation" in manifest


def test_req_report_039_defensive_helpers_and_retro_without_carryforward(tmp_path: Path) -> None:
    """REQ-REPORT-039: missing optional files and empty retro stay explicit."""

    assert _read_json(tmp_path / "missing.json") is None
    assert _read_text(tmp_path / "missing.md") == ""
    assert _relative_path(tmp_path / "ops" / "manifest.md") == "ops/manifest.md"
    assert _relative_path(tmp_path / "results" / "artifact.json") == "results/artifact.json"
    assert _relative_path(tmp_path / "loose.txt") == "loose.txt"

    artifact, manifest = build_artifact(
        retro={"milestone": "2026.04.111", "carry_forward_tracks": []},
        known_issues_text=_known_issues_scope_text(),
        roadmap_text=_roadmap_text(),
        roadmap_next_text="",
        exclusion_manifest_text="",
        manifest_path="ops/milestone_112_scope_reduction_manifest.md",
    )

    assert artifact["status"] == "blocked"
    assert artifact["carryforward_from_111"] == []
    assert artifact["live_sota_runtime_carryforward_rules"] == []
    assert "No live-SOTA runtime carry-forward rules were found" in manifest

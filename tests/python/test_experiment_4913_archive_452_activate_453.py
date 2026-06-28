"""Tests for REQ-REPORT-4913 / SCENARIO-REPORT-4913."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest

from carnot import experiment_4913_archive_452_activate_453 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_json(path: Path, payload: JsonDict | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _capstone_artifact() -> JsonDict:
    return {
        "a1_fork_verdict_trusted": {
            "coverage_migration_count": 0,
            "fork_verdict": "WALL_DEEPER_THAN_VALUE_PREDICTION",
            "honest_verdict": (
                "complete_env_grounded_search_no_first_win_lift_"
                "WALL_DEEPER_THAN_VALUE_PREDICTION"
            ),
            "trust_gate": {
                "a1_numbers_match_fork": True,
                "a1_planner_blind": True,
                "a1_positive_control_non_degenerate": True,
                "a1_trustworthy": True,
                "a1_value_from_real_env": True,
            },
            "trusted": True,
            "value_grounded_first_win_delta_ci95": [-0.04, -0.04],
            "value_grounded_first_win_delta_median": -0.04,
        },
        "capstone_ready": True,
        "honest_verdict": (
            "complete_capstone_v452_escalate_"
            "wall_survives_four_representations_plus_env_grounding"
        ),
        "milestone_scorecard": {
            "a1_env_grounded_search": {
                "coverage_migration_count": 0,
                "fork_verdict": "WALL_DEEPER_THAN_VALUE_PREDICTION",
                "honest_verdict": (
                    "complete_env_grounded_search_no_first_win_lift_"
                    "WALL_DEEPER_THAN_VALUE_PREDICTION"
                ),
                "trusted": True,
                "value_grounded_first_win_delta_ci95": [-0.04, -0.04],
                "value_grounded_first_win_delta_median": -0.04,
            },
            "a1b_latent_action_interface": {
                "fork_verdict": "VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES",
                "honest_verdict": (
                    "complete_latent_action_no_value_lift_"
                    "representation_invariant_4_classes"
                ),
                "latent_action_value_accuracy_delta_median": -0.103162,
                "ran_genuinely_live": True,
            },
            "a2_levelup_bank": {
                "decision": "no_new_level_banked",
                "honest_verdict": "complete_m0r0_no_new_level_residual_duplicate_depth",
                "new_levels_banked": 0,
                "reproducible_total_levels_after": 68,
                "target_game": "m0r0",
            },
            "a3_self_play_checkpoint": {
                "checkpoint_path": "models/arc_verifier_vc33.json",
                "decision": "checkpoint_refreshed",
                "target_game": "vc33",
                "verifier_checkpoint_refreshed": True,
            },
            "a4_fresh_live_heldout": {
                "reason": "flagged_adversarial",
                "status": "skipped_flagged_adversarial",
                "true_honest_verdict": (
                    "complete_heldout_first_win_0.05_ci_lower_0_soft_budget_partial_live"
                ),
            },
            "b2_submission_package": {
                "decision": "package_ready_operator_only",
                "operator_only": True,
                "peak_vram_gb": 15.146,
                "submission_package_ready": True,
                "submits": False,
            },
            "c_hardware": {
                "decision": "kv260_continuity_ok",
                "kv260_ssh_reachable": True,
            },
            "d_v453_handoff": {
                "flagged_for_v453": [
                    {
                        "candidate": "causal_state_abstraction_wall_diagnostic",
                        "priority": 1,
                        "source_ids": ["2401.12497"],
                    },
                    {
                        "candidate": "distributional_energy_verifier_pivot",
                        "priority": 2,
                        "source_ids": ["2605.18871"],
                    },
                ],
                "selected_branch": "wall_survives_four_representations_plus_env_grounding",
            },
        },
        "post_sprint_pivot": {
            "do_not_queue": "representation_5",
            "selected_branch": "wall_survives_four_representations_plus_env_grounding",
        },
        "reproducible_total_levels": 68,
    }


def _make_root(root: Path, *, include_next: bool, active_milestone: str = "2026.06.453") -> None:
    _write_text(root / "research-roadmap.yaml", f"milestone: {active_milestone}\n")
    if include_next:
        _write_text(root / "research-roadmap-next.yaml", "milestone: 2026.06.453\n")
    _write_text(root / mod.REGISTRY_REL_PATH, "schema_version: 1\nreproducible_total_levels: 68\n")
    _write_json(
        root / mod.RETRO_REL_PATH,
        {
            "milestone": "2026.06.452",
            "summary": "The .452 run closed the representation fork as an honest negative.",
        },
    )
    _write_json(root / mod.CAPSTONE_REL_PATH, _capstone_artifact())
    _write_json(
        root / mod.A1_REL_PATH,
        {
            "change_location_prior_used_not_value": True,
            "coverage_migration_count": 0,
            "value_grounded_first_win_delta_ci95": [-0.04, -0.04],
            "value_grounded_first_win_delta_median": -0.04,
        },
    )


def _runner(calls: list[list[str]], *, roadmap_ok: bool, offline_ok: bool, pretest_ok: bool):
    def run(command: list[str], _root: Path) -> mod.CommandResult:
        calls.append(command)
        command_text = " ".join(command)
        if "research-roadmap-next.yaml" in command_text:
            return mod.CommandResult(command, 0 if roadmap_ok else 1, "ok\n", "missing")
        if "offline_arcade" in command_text:
            return mod.CommandResult(command, 0 if offline_ok else 1, "", "arcade")
        return mod.CommandResult(command, 0 if pretest_ok else 1, "passed", "failed")

    return run


def test_req_report_4913_spec_declares_transition_contract() -> None:
    """REQ-REPORT-4913: OpenSpec declares the .452/.453 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert str(mod.OUTPUT_REL_PATH) in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_report_4913_blocked_missing_roadmap_next_writes_deliverable(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4913: blocked roadmap-next still records .452 facts."""

    _make_root(tmp_path, include_next=False)
    calls: list[list[str]] = []
    artifact = mod.run(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=False, offline_ok=True, pretest_ok=True),
        started_s=10.0,
        now_s=10.25,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_next_yaml_missing"
    assert len(calls) == 2
    assert artifact["pretest_gate"] == {
        "green": False,
        "ran": False,
        "reason": "skipped_after_precondition_failure",
    }
    assert artifact["transition_performed"] is False
    assert artifact["leaderboard_submission"] is False
    assert artifact["wall_survives_four_representations_plus_env_grounding"] is True
    assert artifact["energy_program_concluded"] is True
    assert artifact["v453_attacks_closure_diagnostic_not_representation"] is True
    assert artifact["reproducible_total_levels"] == 68
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["passed"] is False
    assert artifact["preconditions_checked"]["offline_arcade"]["passed"] is True
    assert artifact["close_state_452"]["a1"]["value_grounded_first_win_delta_median"] == -0.04
    assert artifact["close_state_452"]["a1b"]["latent_action_value_accuracy_delta_median"] == -0.103162
    assert artifact["close_state_452"]["a2"]["new_levels_banked"] == 0
    assert artifact["close_state_452"]["a3"]["checkpoint_path"] == "models/arc_verifier_vc33.json"
    assert artifact["close_state_452"]["a4"]["status"] == "skipped_flagged_adversarial"
    assert artifact["close_state_452"]["b2"]["peak_vram_gb"] == 15.146
    assert artifact["close_state_452"]["c"]["kv260_ssh_reachable"] is True
    assert artifact["close_state_452"]["d"]["priority_1_arxiv"] == "2401.12497"
    assert json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_4913_complete_transition_records_true_close_state(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4913: complete path records the true .452 state."""

    _make_root(tmp_path, include_next=True)
    calls: list[list[str]] = []
    artifact = mod.run(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
        started_s=20.0,
        now_s=20.125,
    )

    assert artifact["honest_verdict"] == (
        "complete_452_archived_453_activated_closure_diagnostic_recorded"
    )
    assert calls[-1] == [".venv/bin/pytest", "tests/python", "-q"]
    assert artifact["transition_performed"] is True
    assert artifact["active_milestone_confirmed"] == "2026.06.453"
    assert artifact["pretest_gate"]["green"] is True
    assert artifact["poison_test_resolved"] == {"quarantined": False, "test": "", "reason": ""}
    assert artifact["close_state_452"]["capstone"]["honest_verdict"] == (
        "complete_capstone_v452_escalate_wall_survives_four_representations_plus_env_grounding"
    )
    assert artifact["close_state_452"]["d"]["priority_2"] == (
        "distributional_energy_verifier_pivot"
    )
    assert len(artifact["cited_upstream_artifacts"]) == 4
    assert mod.validate_artifact(artifact) == []


def test_validate_artifact_and_helpers_cover_failure_cases(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4913: malformed transition artifacts fail validation."""

    _make_root(tmp_path, include_next=True)
    artifact = mod.build_artifact(
        root=tmp_path,
        honest_verdict="complete_452_archived_453_activated_closure_diagnostic_recorded",
        preconditions_checked={
            "research_roadmap_next_yaml": {"passed": True},
            "offline_arcade": {"passed": True},
        },
        pretest_gate={"ran": True, "green": True},
        transition_performed=True,
        poison_test_resolved={"quarantined": False, "test": "", "reason": ""},
        duration_s=0.0001,
    )

    assert mod.validate_artifact(artifact) == []
    assert "missing_field:honest_verdict" in mod.validate_artifact(
        {key: value for key, value in artifact.items() if key != "honest_verdict"}
    )
    assert "honest_verdict_missing_terminal_prefix" in mod.validate_artifact(
        {**artifact, "honest_verdict": "done"}
    )
    assert "invalid_inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "live_llm"}
    )
    assert "missing_principle:energy_program_concluded" in mod.validate_artifact(
        {**artifact, "field_principles": {}}
    )
    assert "invalid_reproducible_total_levels" in mod.validate_artifact(
        {**artifact, "reproducible_total_levels": "68"}
    )
    assert "invalid_wall_survives_four_representations_plus_env_grounding" in (
        mod.validate_artifact(
            {**artifact, "wall_survives_four_representations_plus_env_grounding": False}
        )
    )
    assert "invalid_energy_program_concluded" in mod.validate_artifact(
        {**artifact, "energy_program_concluded": False}
    )
    assert "invalid_v453_attacks_closure_diagnostic_not_representation" in (
        mod.validate_artifact(
            {**artifact, "v453_attacks_closure_diagnostic_not_representation": False}
        )
    )
    assert "invalid_leaderboard_submission" in mod.validate_artifact(
        {**artifact, "leaderboard_submission": True}
    )
    assert "invalid_reproducibility_checksum" in mod.validate_artifact(
        {**artifact, "reproducibility_checksum": ""}
    )
    assert mod.precondition_blocker(tmp_path, {"research_roadmap_next_yaml": {"passed": False}}) == (
        "blocked_research_roadmap_next_yaml_poison"
    )
    assert mod.precondition_blocker(
        tmp_path / "missing", {"research_roadmap_next_yaml": {"passed": False}}
    ) == "blocked_research_roadmap_next_yaml_missing"
    assert mod.precondition_blocker(
        tmp_path, {"research_roadmap_next_yaml": {"passed": True}, "offline_arcade": {"passed": False}}
    ) == "blocked_offline_arcade_unavailable"
    assert mod.precondition_blocker(
        tmp_path, {"research_roadmap_next_yaml": {"passed": True}, "offline_arcade": {"passed": True}}
    ) == ""
    assert mod.command_summary(mod.CommandResult(["x"], 2, "out", "err")) == {
        "command": ["x"],
        "exit_code": 2,
        "passed": False,
        "stderr_tail": "err",
        "stdout_tail": "out",
    }
    assert mod.duration_from(1.0, 1.0) == 0.0001
    assert mod.duration_from(None, None) >= 0.0001
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_yaml_object(tmp_path / "missing.yaml") == {}
    _write_json(tmp_path / "list.json", [])
    _write_text(tmp_path / "list.yaml", "- x\n")
    assert mod.read_json_object(tmp_path / "list.json") == {}
    assert mod.read_yaml_object(tmp_path / "list.yaml") == {}
    assert mod.file_sha256(tmp_path / "missing.txt") == ""
    no_milestone_root = tmp_path / "no_milestone"
    _write_text(no_milestone_root / "research-roadmap.yaml", "note: no active milestone\n")
    assert mod.read_active_milestone(no_milestone_root) == ("unknown", "research-roadmap.yaml")
    assert mod.read_active_milestone(tmp_path / "missing") == ("unknown", "research-roadmap.yaml")
    assert mod.run_command([sys.executable, "-c", "print('ok')"], tmp_path).exit_code == 0

    inactive_root = tmp_path / "inactive"
    _make_root(inactive_root, include_next=True, active_milestone="2026.06.452")
    inactive_calls: list[list[str]] = []
    inactive = mod.run(
        root=inactive_root,
        command_runner=_runner(inactive_calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
        started_s=1.0,
        now_s=1.1,
    )
    assert inactive["honest_verdict"] == "blocked_453_not_active"
    assert inactive["pretest_gate"] == {
        "green": False,
        "ran": False,
        "reason": "skipped_until_453_active",
    }
    assert len(inactive_calls) == 2

    pretest_fail_root = tmp_path / "pretest_fail"
    _make_root(pretest_fail_root, include_next=True)
    pretest_calls: list[list[str]] = []
    pretest_failed = mod.run(
        root=pretest_fail_root,
        command_runner=_runner(pretest_calls, roadmap_ok=True, offline_ok=True, pretest_ok=False),
        started_s=2.0,
        now_s=2.1,
    )
    assert pretest_failed["honest_verdict"] == "blocked_pretest_gate_failed"
    assert pretest_failed["pretest_gate"]["ran"] is True
    assert pretest_failed["transition_performed"] is False

    main_root = tmp_path / "main"
    _make_root(main_root, include_next=True)
    main_calls: list[list[str]] = []
    assert mod.main(
        root=main_root,
        command_runner=_runner(main_calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
    ) == 0
    assert str(mod.OUTPUT_REL_PATH) in capsys.readouterr().out

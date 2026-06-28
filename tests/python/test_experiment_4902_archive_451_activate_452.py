"""Tests for REQ-REPORT-4902 / SCENARIO-REPORT-4902."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4902_archive_451_activate_452 as mod


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
        "change_value_gap_representation_invariant": True,
        "deadline_lever_scorecard": {
            "a2_bank": {
                "decision": "no_new_level_banked",
                "experiment_id": 4894,
                "new_levels_banked": 0,
                "registry_authoritative_total": 68,
                "registry_update_reason": "duplicate_depth",
                "reproducible_total_levels_after": 68,
                "target_game": "dc22",
            },
            "a3_self_play": {
                "checkpoint_path": "models/arc_verifier_sk48.json",
                "decision": "checkpoint_refreshed",
                "experiment_id": 4895,
                "target_game": "sk48",
                "verifier_checkpoint_refreshed": True,
            },
            "a4_fresh_live_rate": {
                "experiment_id": 4896,
                "heldout_first_win_ci_lower": 0.0,
                "heldout_first_win_rate": 0.052632,
                "status": "soft_budget_partial",
            },
            "b2_package": {
                "decision": "package_ready_operator_only",
                "experiment_id": 4898,
                "operator_only": True,
                "package_builds": True,
                "submission_package_ready": True,
                "vram_estimate_gb": 15.146,
            },
        },
        "fork_verdict_trusted": True,
        "honest_verdict": "complete_capstone_v451_representation_invariant_escalate_operator",
        "operator_escalation_note": "representation-invariant across executable-code, decision-need, and action-prefix latents",
        "representation_fork_verdict": {
            "a1": {
                "delta_ci95": [-0.227708, 0.025266],
                "delta_median": -0.101866,
                "engine_cell_recall_median": 0.727273,
                "fork_verdict": "VALUE_GAP_REPRESENTATION_INVARIANT",
                "honest_verdict": "complete_decision_need_no_value_lift_VALUE_GAP_REPRESENTATION_INVARIANT",
            },
            "a1b": {
                "delta_ci95": [-0.134887, 0.025266],
                "delta_median": 0.0,
                "fork_verdict": "VALUE_GAP_REPRESENTATION_INVARIANT_HARD",
                "honest_verdict": "complete_action_prefix_latent_no_value_lift_representation_invariant_hard",
            },
            "trusted": True,
            "verdict": "representation_invariant_escalate_operator",
        },
        "reproducible_total_levels": 68,
    }


def _make_root(root: Path, *, include_next: bool, active_milestone: str = "2026.06.452") -> None:
    _write_text(root / "research-roadmap.yaml", f"milestone: {active_milestone}\n")
    if include_next:
        _write_text(root / "research-roadmap-next.yaml", "milestone: 2026.06.452\n")
    _write_text(root / mod.REGISTRY_REL_PATH, "schema_version: 1\nreproducible_total_levels: 68\n")
    _write_json(
        root / mod.RETRO_REL_PATH,
        {
            "milestone": "2026.06.451",
            "summary": "The .451 run confirmed representation-invariant change-VALUE gap.",
        },
    )
    _write_json(
        root / mod.PREVIOUS_TRANSITION_REL_PATH,
        {
            "wall_is_executable_code_change_value_representation": True,
            "honest_verdict": "blocked_research_roadmap_next_yaml_missing",
        },
    )
    _write_json(
        root / mod.A1_REL_PATH,
        {
            "coverage_migration": 0,
            "decision_need_value_accuracy_delta_ci95": [-0.227708, 0.025266],
            "decision_need_value_accuracy_delta_median": -0.101866,
            "duration_s": 276.9,
            "engine_cell_recall_median": 0.727273,
            "fork_verdict": "VALUE_GAP_REPRESENTATION_INVARIANT",
            "honest_verdict": "complete_decision_need_no_value_lift_VALUE_GAP_REPRESENTATION_INVARIANT",
        },
    )
    _write_json(
        root / mod.A1B_REL_PATH,
        {
            "action_prefix_value_accuracy_delta_ci95": [-0.134887, 0.025266],
            "action_prefix_value_accuracy_delta_median": 0.0,
            "duration_s": 218.9,
            "fork_verdict": "VALUE_GAP_REPRESENTATION_INVARIANT_HARD",
            "honest_verdict": "complete_action_prefix_latent_no_value_lift_representation_invariant_hard",
        },
    )
    _write_json(root / mod.CAPSTONE_REL_PATH, _capstone_artifact())
    _write_json(
        root / mod.A4_REL_PATH,
        {
            "heldout_first_win_ci_lower": 0.0,
            "heldout_first_win_rate": 0.052632,
            "honest_verdict": "complete_heldout_first_win_0.052632_ci_lower_0_soft_budget_partial",
            "partial": True,
        },
    )
    _write_json(
        root / mod.KV260_REL_PATH,
        {
            "honest_verdict": "success_kv260_continuity_ok",
            "kv260_ssh_reachable": True,
            "next_forward_step": "GRADUATED: KV260 terminal criteria met",
            "board_state": {"uio_device_count": 5},
        },
    )
    _write_json(
        root / mod.SOTA_REL_PATH,
        {
            "honest_verdict": "success_sota_ingestion_v452_frontier_mapped",
            "flagged_for_v452": [
                {
                    "candidate": "latent_action_interface",
                    "priority": 1,
                    "source_ids": ["2503.18938"],
                },
                {
                    "candidate": "reverse_counterfactual_targeter",
                    "priority": 2,
                    "source_ids": ["2505.08073"],
                },
                {
                    "candidate": "verification_calibrated_abstraction",
                    "priority": 3,
                    "source_ids": ["2602.23997"],
                },
            ],
        },
    )


def _runner(calls: list[list[str]], *, roadmap_ok: bool, offline_ok: bool, pretest_ok: bool):
    def run(command: list[str], _root: Path) -> mod.CommandResult:
        calls.append(command)
        command_text = " ".join(command)
        if "research-roadmap-next.yaml" in command_text:
            return mod.CommandResult(command, 0 if roadmap_ok else 1, "ok\n", "" if roadmap_ok else "missing")
        if "offline_arcade" in command_text:
            return mod.CommandResult(command, 0 if offline_ok else 1, "", "" if offline_ok else "arcade")
        return mod.CommandResult(command, 0 if pretest_ok else 1, "passed" if pretest_ok else "failed", "")

    return run


def test_req_report_4902_spec_declares_transition_contract() -> None:
    """REQ-REPORT-4902: OpenSpec declares the .451/.452 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert str(mod.OUTPUT_REL_PATH) in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_report_4902_blocked_missing_roadmap_next_writes_deliverable(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4902: blocked roadmap-next still records .451 facts."""

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
    assert artifact["change_value_gap_representation_invariant_3_classes"] is True
    assert artifact["energy_program_concluded"] is True
    assert artifact["v452_attacks_env_grounding_not_prediction"] is True
    assert artifact["reproducible_total_levels"] == 68
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["passed"] is False
    assert artifact["preconditions_checked"]["offline_arcade"]["passed"] is True
    assert artifact["close_state_451"]["a1"]["decision_need_value_accuracy_delta_median"] == -0.101866
    assert artifact["close_state_451"]["a1b"]["action_prefix_value_accuracy_delta_median"] == 0.0
    assert artifact["close_state_451"]["a2"]["reproducible_total_levels_after"] == 68
    assert artifact["close_state_451"]["a4"]["rerun_required_clean_with_model_specs_and_random_seed"] is True
    assert artifact["close_state_451"]["c"]["kv260_ssh_reachable"] is True
    assert artifact["v452_frontier"]["headline"] == "env_grounding_interleaved_act_and_observe"
    assert json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_4902_complete_transition_records_true_close_state(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4902: complete path records the true .451 state."""

    _make_root(tmp_path, include_next=True)
    calls: list[list[str]] = []
    artifact = mod.run(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
        started_s=20.0,
        now_s=20.125,
    )

    assert artifact["honest_verdict"] == (
        "complete_451_archived_452_activated_env_grounding_frontier_recorded"
    )
    assert calls[-1] == [".venv/bin/pytest", "tests/python", "-q"]
    assert artifact["transition_performed"] is True
    assert artifact["active_milestone_confirmed"] == "2026.06.452"
    assert artifact["pretest_gate"]["green"] is True
    assert artifact["poison_test_resolved"] == {"quarantined": False, "test": "", "reason": ""}
    assert artifact["close_state_451"]["capstone"]["honest_verdict"] == (
        "complete_capstone_v451_representation_invariant_escalate_operator"
    )
    assert artifact["close_state_451"]["d"]["priority_1"] == "latent_action_interface"
    assert artifact["v452_frontier"]["gated_last_representation_swing"] == "latent_action_interface"
    assert len(artifact["cited_upstream_artifacts"]) == 9
    assert mod.validate_artifact(artifact) == []


def test_validate_artifact_and_helpers_cover_failure_cases(tmp_path: Path) -> None:
    """REQ-REPORT-4902: malformed transition artifacts fail validation."""

    _make_root(tmp_path, include_next=True)
    artifact = mod.build_artifact(
        root=tmp_path,
        honest_verdict="complete_451_archived_452_activated_env_grounding_frontier_recorded",
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
    assert "invalid_change_value_gap_representation_invariant_3_classes" in mod.validate_artifact(
        {**artifact, "change_value_gap_representation_invariant_3_classes": False}
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

    inactive_root = tmp_path / "inactive"
    _make_root(inactive_root, include_next=True, active_milestone="2026.06.451")
    inactive_calls: list[list[str]] = []
    inactive = mod.run(
        root=inactive_root,
        command_runner=_runner(inactive_calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
        started_s=1.0,
        now_s=1.1,
    )
    assert inactive["honest_verdict"] == "blocked_452_not_active"
    assert inactive["pretest_gate"] == {
        "green": False,
        "ran": False,
        "reason": "skipped_until_452_active",
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
    assert pretest_failed["pretest_gate"]["green"] is False
    assert pretest_failed["poison_test_resolved"]["quarantined"] is False
    assert pretest_calls[-1] == mod.PRETEST_COMMAND


def test_main_and_default_command_runner(tmp_path: Path) -> None:
    """REQ-REPORT-4902: script entrypoint writes the artifact path."""

    _make_root(tmp_path, include_next=True)
    calls: list[list[str]] = []
    exit_code = mod.main(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
    )
    default_result = mod.run_command([".venv/bin/python", "-c", "print('ok')"], REPO)

    assert exit_code == 0
    assert (tmp_path / mod.OUTPUT_REL_PATH).exists()
    assert default_result.exit_code == 0
    assert default_result.stdout == "ok\n"

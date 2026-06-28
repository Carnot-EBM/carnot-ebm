"""Tests for REQ-REPORT-4891 / SCENARIO-REPORT-4891."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4891_archive_450_activate_451 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _a1_artifact() -> JsonDict:
    return {
        "coverage_migration_count": 0,
        "duration_s": 168.14787244796753,
        "engine_cell_recall_median": 0.727273,
        "fork_verdict": "INDUCER_CEILING_HARD",
        "generator_backend": "gpu0_cuda",
        "honest_verdict": "complete_ttt_dynamics_no_value_lift_INDUCER_CEILING_HARD",
        "inference_substrate": "live_llm_inference",
        "n_games_measured": 9,
        "positive_control_game": "tu93",
        "positive_control_non_degenerate": True,
        "tta_changed_cell_value_accuracy_delta_median": -0.008699,
        "tta_value_accuracy_delta_ci95": [-0.178, 0.0],
    }


def _a1b_artifact() -> JsonDict:
    return {
        "duration_s": 13.697947978973389,
        "flagged_adversarial": True,
        "honest_verdict": "complete_inducer_ceiling_neither_lane_lifts_method_is_ceiling",
        "inducer_ceiling_attribution": "METHOD_IS_CEILING",
        "reference_lane_is_ceiling_only": True,
    }


def _audit_artifact() -> JsonDict:
    return {
        "a1_genuinely_diagnostic": True,
        "a1_positive_control_non_degenerate_confirmed": True,
        "a1_source_fork_verdict": "INDUCER_CEILING_HARD",
        "a1_source_honest_verdict": "complete_ttt_dynamics_no_value_lift_INDUCER_CEILING_HARD",
        "a1_source_n_games_measured": 9,
        "a1b_ab_trustworthy": False,
        "a1b_adversarial_result": {
            "flags": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
            "max_severity": 2,
        },
        "a1b_failure_reasons": [
            "a1b_flagged_adversarial_stamp",
            "a1b_adversarial_verify_flagged",
            "a1b_duration_below_live_floor",
        ],
        "a1b_source_attribution": "METHOD_IS_CEILING",
        "honest_verdict": "complete_a1_a1b_audited",
    }


def _make_root(root: Path, *, include_next: bool) -> None:
    _write_text(
        root / "research-roadmap.yaml",
        "milestone: 2026.06.451\nnote: Energy CONCLUDED; use alternative world models.\n",
    )
    if include_next:
        _write_text(
            root / "research-roadmap-next.yaml",
            "milestone: 2026.06.451\nnote: Energy CONCLUDED.\n",
        )
    _write_text(root / mod.REGISTRY_REL_PATH, "schema_version: 1\nreproducible_total_levels: 68\n")
    _write_json(
        root / mod.RETRO_REL_PATH,
        {
            "honest_verdict": "complete_operational_retro_450",
            "milestone": "2026.06.450",
            "summary": "A1 INDUCER_CEILING_HARD; g50t +1; energy concluded.",
        },
    )
    _write_json(root / mod.A1_REL_PATH, _a1_artifact())
    _write_json(root / mod.A1B_REL_PATH, _a1b_artifact())
    _write_json(root / mod.AUDIT_REL_PATH, _audit_artifact())
    _write_json(
        root / mod.A2_REL_PATH,
        {
            "honest_verdict": "success_g50t_levelup_banked",
            "new_levels_banked": 1,
            "reproducible_total_levels_before": 67,
            "reproducible_total_levels_after": 68,
            "solve_provenance": "live_agent_self_discovery",
            "target_game": "g50t",
        },
    )
    _write_json(
        root / mod.A3_REL_PATH,
        {
            "honest_verdict": "success_self_play_checkpoint_refreshed",
            "target_game": "ls20",
            "verifier_checkpoint_refreshed": True,
        },
    )
    _write_json(
        root / mod.A4_REL_PATH,
        {
            "flagged_adversarial": True,
            "heldout_first_win_ci_lower": 0.0,
            "heldout_first_win_rate": 0.0625,
            "honest_verdict": "complete_heldout_first_win_soft_budget_partial_0.0625_ci_lower_0",
            "live_agent_ran": True,
        },
    )
    _write_json(
        root / mod.B2_REL_PATH,
        {
            "honest_verdict": "success_submission_package_ready_final_pre_deadline",
            "operator_only": True,
            "submission_package_ready": True,
            "vram_estimate_gb": 15.146,
        },
    )
    _write_json(
        root / mod.C_REL_PATH,
        {
            "board_state": {"uio_device_count": 5},
            "honest_verdict": "success_kv260_continuity_ok",
            "kv260_ssh_reachable": True,
        },
    )
    _write_json(
        root / mod.D_REL_PATH,
        {
            "aimed_at_fork_verdict": "INDUCER_CEILING_HARD",
            "flagged_for_v451": [
                {"candidate": "agent_authored_decision_need_targets", "priority": 1},
                {"candidate": "action_prefix_latent_adapter", "priority": 2},
                {"candidate": "latent_action_world_model_adapter", "priority": 3},
            ],
            "honest_verdict": "success_sota_ingestion_v451_frontier_mapped",
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


def test_req_report_4891_spec_declares_transition_contract() -> None:
    """REQ-REPORT-4891: OpenSpec declares the .450/.451 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert str(mod.OUTPUT_REL_PATH) in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_report_4891_blocked_missing_roadmap_next_writes_deliverable(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4891: blocked roadmap-next still records .450 facts."""

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
    assert all(command[:2] == [".venv/bin/python", "-c"] for command in calls)
    assert artifact["pretest_gate"] == {
        "green": False,
        "ran": False,
        "reason": "skipped_after_precondition_failure",
    }
    assert artifact["transition_performed"] is False
    assert artifact["a1_inducer_ceiling_hard_trustworthy"] is True
    assert artifact["a1b_was_fabrication_flagged_non_test"] is True
    assert artifact["wall_is_executable_code_change_value_representation"] is True
    assert artifact["energy_program_concluded"] is True
    assert artifact["reproducible_total_levels"] == 68
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["passed"] is False
    assert artifact["preconditions_checked"]["offline_arcade"]["passed"] is True
    assert artifact["a450_close_state"]["a1"]["engine_cell_recall_median"] == 0.727273
    assert artifact["a450_close_state"]["a1b"]["duration_too_short_flagged"] is True
    assert json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_4891_complete_transition_records_true_close_state(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4891: complete path records the true .450 state."""

    _make_root(tmp_path, include_next=True)
    calls: list[list[str]] = []
    artifact = mod.run(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
        started_s=20.0,
        now_s=20.125,
    )

    assert artifact["honest_verdict"] == (
        "complete_450_archived_451_activated_value_representation_wall_recorded"
    )
    assert calls[-1] == [".venv/bin/pytest", "tests/python", "-q"]
    assert artifact["transition_performed"] is True
    assert artifact["pretest_gate"]["green"] is True
    assert artifact["active_milestone_confirmed"] == "2026.06.451"
    assert artifact["a450_close_state"]["a1"]["fork_verdict"] == "INDUCER_CEILING_HARD"
    assert artifact["a450_close_state"]["a1"]["tta_value_accuracy_delta_ci95"] == [-0.178, 0.0]
    assert artifact["a450_close_state"]["a1b"]["method_is_ceiling_established"] is False
    assert artifact["a450_close_state"]["a2"]["target_game"] == "g50t"
    assert artifact["a450_close_state"]["a2"]["reproducible_total_levels_after"] == 68
    assert artifact["a450_close_state"]["c"]["uio_device_count"] == 5
    assert artifact["a450_close_state"]["d"]["priority_1"] == "agent_authored_decision_need_targets"
    assert len(artifact["cited_upstream_artifacts"]) == 11
    assert mod.validate_artifact(artifact) == []


def test_validate_artifact_and_helpers_cover_failure_cases(tmp_path: Path) -> None:
    """REQ-REPORT-4891: malformed transition artifacts fail validation."""

    _make_root(tmp_path, include_next=True)
    artifact = mod.build_artifact(
        root=tmp_path,
        honest_verdict="complete_450_archived_451_activated_value_representation_wall_recorded",
        preconditions_checked={
            "research_roadmap_next_yaml": {"passed": True},
            "offline_arcade": {"passed": True},
        },
        pretest_gate={"ran": True, "green": True},
        transition_performed=True,
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
    assert "invalid_a1_inducer_ceiling_hard_trustworthy" in mod.validate_artifact(
        {**artifact, "a1_inducer_ceiling_hard_trustworthy": False}
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
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_yaml_object(tmp_path / "missing.yaml") == {}
    assert mod.file_sha256(tmp_path / "missing.txt") == ""
    no_milestone_root = tmp_path / "no_milestone"
    _write_text(no_milestone_root / "research-roadmap.yaml", "note: no active milestone\n")
    assert mod.read_active_milestone(no_milestone_root) == ("unknown", "research-roadmap.yaml")
    assert mod.read_active_milestone(tmp_path / "missing") == ("unknown", "research-roadmap.yaml")

    inactive_root = tmp_path / "inactive"
    _make_root(inactive_root, include_next=True)
    _write_text(
        inactive_root / "research-roadmap.yaml",
        "milestone: 2026.06.450\nnote: Energy CONCLUDED.\n",
    )
    inactive_calls: list[list[str]] = []
    inactive = mod.run(
        root=inactive_root,
        command_runner=_runner(inactive_calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
        started_s=1.0,
        now_s=1.1,
    )
    assert inactive["honest_verdict"] == "blocked_451_not_active"
    assert inactive["pretest_gate"] == {
        "green": False,
        "ran": False,
        "reason": "skipped_until_451_active",
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
    assert pretest_calls[-1] == mod.PRETEST_COMMAND


def test_main_and_default_command_runner(tmp_path: Path) -> None:
    """REQ-REPORT-4891: script entrypoint writes the artifact path."""

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

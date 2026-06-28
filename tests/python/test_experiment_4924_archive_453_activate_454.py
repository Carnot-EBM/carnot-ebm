"""Tests for REQ-REPORT-4924 / SCENARIO-REPORT-4924."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4924_archive_453_activate_454 as mod


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
        "a1_closure_verdict_trusted": {
            "b1_experiment_id": 4918,
            "b1_failure_reasons": [],
            "b1_honest_verdict": "complete_a1_causal_abstraction_audited",
            "closure_verdict": "WALL_IS_HIDDEN_STATE",
            "experiment_id": 4914,
            "hidden_variables_required": ["winning_prefix_order_state"],
            "honest_verdict": (
                "complete_causal_abstraction_hidden_state_representation_invariant_closure"
            ),
            "is_decision_need_table_in_disguise": False,
            "live_path_reachable": True,
            "minimal_abstraction_is_observable_subset": False,
            "n_games_measured": 3,
            "planner_blind_to_banked_answer": True,
            "positive_control_classifies_observable": True,
            "trust_failure_reasons": [],
            "trust_gate": {
                "a1_diagnostic_trustworthy": True,
                "real_transitions": True,
                "not_value_table": True,
                "observable_claims_verified": True,
                "positive_control_observable": True,
                "oracle_distinct_planner_blind": True,
                "numbers_match_fork": True,
            },
            "trusted": True,
            "verifier_is_oracle": False,
        },
        "capstone_ready": True,
        "headline": (
            "ARC CLOSURE: the live first-win wall is representation-invariant by "
            "construction. Deliverable locks to the current ~0.05 first-win agent "
            "plus the publishable FoVer verifier-ensemble paper. Do not queue "
            "representation #5."
        ),
        "honest_verdict": "complete_capstone_v453_wall_is_hidden_state_arc_closure",
        "heldout_first_win_rate": 0.047619,
        "milestone_scorecard": {
            "a1_causal_abstraction_closure": {
                "closure_verdict": "WALL_IS_HIDDEN_STATE",
                "experiment_id": 4914,
                "hidden_variables_required": ["winning_prefix_order_state"],
                "honest_verdict": (
                    "complete_causal_abstraction_hidden_state_representation_invariant_closure"
                ),
                "positive_control_classifies_observable": True,
                "trusted": True,
            },
            "a2_levelup_bank": {
                "decision": "new_level_banked",
                "experiment_id": 4915,
                "honest_verdict": "success_cn04_levelup_banked",
                "new_levels_banked": 1,
                "offline_reproduced": True,
                "registry_authoritative_total": 69,
                "registry_update_reason": "banked_offline_reproduced_level",
                "reproduced_levels": 3,
                "reproducible_total_levels_after": 69,
                "reproducible_total_levels_before": 68,
                "reproduction_gate_passed": True,
                "target_game": "cn04",
            },
            "a3_self_play_checkpoint": {
                "checkpoint_path": "models/arc_verifier_bp35.json",
                "decision": "checkpoint_refreshed",
                "experiment_id": 4916,
                "honest_verdict": "success_self_play_checkpoint_refreshed",
                "target_game": "bp35",
                "verifier_checkpoint_refreshed": True,
            },
            "a4_heldout_go_no_go": {
                "completed_game_count": 21,
                "experiment_id": 4917,
                "flag_resolved": True,
                "heldout_first_win_rate": 0.047619,
                "honest_verdict": (
                    "complete: heldout_first_win_soft_budget_stop_partial_"
                    "21_of_25_games_84_attempts_resume_to_finish"
                ),
                "live_agent_ran": True,
                "partial": True,
                "remaining_game_count": 4,
            },
            "b1_causal_abstraction_audit": {
                "a1_diagnostic_trustworthy": True,
                "a1_failure_reasons": [],
                "checks": {
                    "real_transitions": True,
                    "not_value_table": True,
                    "observable_claims_verified": True,
                    "positive_control_observable": True,
                    "oracle_distinct_planner_blind": True,
                    "numbers_match_fork": True,
                },
                "experiment_id": 4918,
                "honest_verdict": "complete_a1_causal_abstraction_audited",
            },
            "b2_submission_package": {
                "decision": "package_ready_operator_only",
                "experiment_id": 4919,
                "honest_verdict": "success_submission_package_ready_final_pre_deadline",
                "operator_only": True,
                "peak_vram_gb": 15.146,
                "submission_package_ready": True,
                "submits": False,
            },
            "c_kv260": {
                "decision": "kv260_continuity_ok",
                "experiment_id": 4921,
                "honest_verdict": "success_kv260_continuity_ok",
                "kv260_ssh_reachable": True,
            },
            "d_distributional_energy_verifier_pivot": {
                "decision": "pivot_scaffold_executable",
                "experiment_id": 4922,
                "harness_skeleton_path": (
                    "python/carnot/experiment_4922_distributional_energy_verifier_scaffold.py"
                ),
                "honest_verdict": "success_distributional_energy_verifier_pivot_scaffolded",
                "pivot_executable_on_6_30": True,
            },
        },
        "post_sprint_pivot": {
            "decision": "post_6_30_distributional_energy_verifier_pivot",
            "deliverable": (
                "current ~0.05 first-win agent (operator-only package) + "
                "publishable FoVer verifier-ensemble paper"
            ),
            "do_not_queue": "representation_5",
            "paper_ready": True,
        },
        "reproducible_total_levels": 69,
        "submission_package_ready": True,
    }


def _make_root(root: Path, *, include_next: bool, active_milestone: str = "2026.06.454") -> None:
    _write_text(root / "research-roadmap.yaml", f"milestone: {active_milestone}\n")
    if include_next:
        _write_text(root / "research-roadmap-next.yaml", "milestone: 2026.06.454\n")
    _write_text(root / mod.REGISTRY_REL_PATH, "schema_version: 1\nreproducible_total_levels: 69\n")
    _write_json(
        root / mod.RETRO_REL_PATH,
        {
            "milestone": "2026.06.453",
            "summary": (
                "A1 closed WALL_IS_HIDDEN_STATE, B1 trusted all six gates, "
                "A2 banked cn04 68->69, A4 partial 21/25 not flagged."
            ),
            "experiments_completed": 0,
        },
    )
    _write_json(root / mod.CAPSTONE_REL_PATH, _capstone_artifact())


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


def test_req_report_4924_spec_declares_transition_contract() -> None:
    """REQ-REPORT-4924: OpenSpec declares the .453/.454 transition contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-4924") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert str(mod.OUTPUT_REL_PATH) in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_report_4924_blocked_missing_roadmap_next_writes_deliverable(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4924: blocked roadmap-next still records .453 facts."""

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
    assert artifact["arc_first_win_wall_closed_hidden_state"] is True
    assert artifact["deliverable_locked_agent_plus_fover_paper"] is True
    assert artifact["v454_is_submission_maximization_not_new_fork"] is True
    assert artifact["reproducible_total_levels"] == 69
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["passed"] is False
    assert artifact["preconditions_checked"]["offline_arcade"]["passed"] is True
    assert artifact["close_state_453"]["a1"]["closure_verdict"] == "WALL_IS_HIDDEN_STATE"
    assert artifact["close_state_453"]["b1"]["a1_diagnostic_trustworthy"] is True
    assert artifact["close_state_453"]["a2"]["target_game"] == "cn04"
    assert artifact["close_state_453"]["a2"]["reproducible_total_levels_after"] == 69
    assert artifact["close_state_453"]["a3"]["checkpoint_path"] == "models/arc_verifier_bp35.json"
    assert artifact["close_state_453"]["a4"]["completed_game_count"] == 21
    assert artifact["close_state_453"]["a4"]["remaining_game_count"] == 4
    assert artifact["close_state_453"]["b2"]["peak_vram_gb"] == 15.146
    assert artifact["close_state_453"]["c"]["kv260_ssh_reachable"] is True
    assert artifact["close_state_453"]["d"]["decision"] == "pivot_scaffold_executable"
    assert "representation_5" in artifact["close_state_453"]["do_not_queue"]
    assert json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_4924_complete_transition_records_true_close_state(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4924: complete path records the true .453 state."""

    _make_root(tmp_path, include_next=True)
    calls: list[list[str]] = []
    artifact = mod.run(
        root=tmp_path,
        command_runner=_runner(calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
        started_s=20.0,
        now_s=20.125,
    )

    assert artifact["honest_verdict"] == (
        "complete_453_archived_454_activated_submission_maximization_recorded"
    )
    assert calls[-1] == [".venv/bin/pytest", "tests/python", "-q"]
    assert artifact["transition_performed"] is True
    assert artifact["active_milestone_confirmed"] == "2026.06.454"
    assert artifact["pretest_gate"]["green"] is True
    assert artifact["poison_test_resolved"] == {"quarantined": False, "test": "", "reason": ""}
    assert artifact["close_state_453"]["capstone"]["honest_verdict"] == (
        "complete_capstone_v453_wall_is_hidden_state_arc_closure"
    )
    assert artifact["close_state_453"]["deliverable"] == (
        "current ~0.05 first-win agent (operator-only package) + "
        "publishable FoVer verifier-ensemble paper"
    )
    assert len(artifact["cited_upstream_artifacts"]) == 3
    assert mod.validate_artifact(artifact) == []


def test_validate_artifact_and_blocked_branches(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4924-FIELD-PRINCIPLES: malformed artifacts fail validation."""

    _make_root(tmp_path, include_next=True)
    artifact = mod.build_artifact(
        root=tmp_path,
        honest_verdict="complete_453_archived_454_activated_submission_maximization_recorded",
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
    assert "missing_principle:arc_first_win_wall_closed_hidden_state" in (
        mod.validate_artifact({**artifact, "field_principles": {}})
    )
    assert "invalid_reproducible_total_levels" in mod.validate_artifact(
        {**artifact, "reproducible_total_levels": "69"}
    )
    assert "invalid_arc_first_win_wall_closed_hidden_state" in mod.validate_artifact(
        {**artifact, "arc_first_win_wall_closed_hidden_state": False}
    )
    assert "invalid_deliverable_locked_agent_plus_fover_paper" in mod.validate_artifact(
        {**artifact, "deliverable_locked_agent_plus_fover_paper": False}
    )
    assert "invalid_v454_is_submission_maximization_not_new_fork" in mod.validate_artifact(
        {**artifact, "v454_is_submission_maximization_not_new_fork": False}
    )
    assert "invalid_leaderboard_submission" in mod.validate_artifact(
        {**artifact, "leaderboard_submission": True}
    )
    assert "invalid_close_state_453" in mod.validate_artifact(
        {**artifact, "close_state_453": []}
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

    inactive_root = tmp_path / "inactive"
    _make_root(inactive_root, include_next=True, active_milestone="2026.06.453")
    inactive_calls: list[list[str]] = []
    inactive = mod.run(
        root=inactive_root,
        command_runner=_runner(inactive_calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
        started_s=1.0,
        now_s=1.1,
    )
    assert inactive["honest_verdict"] == "blocked_454_not_active"
    assert inactive["pretest_gate"] == {
        "green": False,
        "ran": False,
        "reason": "skipped_until_454_active",
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
    assert pretest_calls[-1] == mod.PRETEST_COMMAND

    main_root = tmp_path / "main"
    _make_root(main_root, include_next=True)
    main_calls: list[list[str]] = []
    assert mod.main(
        root=main_root,
        command_runner=_runner(main_calls, roadmap_ok=True, offline_ok=True, pretest_ok=True),
    ) == 0

"""Tests for Exp 4687 `.431` archive / `.432` activation record.

Spec refs: REQ-CAPSTONE-4687, SCENARIO-CAPSTONE-4687,
SCENARIO-CAPSTONE-4687-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4687-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4687_archive_431_activate_432 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _green_smart_subset(_root: Path) -> mod.CommandResult:
    return mod.CommandResult(command=["pytest", "smart-subset"], exit_code=0, stdout="green", stderr="")


def _red_smart_subset(_root: Path) -> mod.CommandResult:
    return mod.CommandResult(command=["pytest", "smart-subset"], exit_code=1, stdout="red", stderr="failed")


def _a1_4676() -> JsonDict:
    return {
        "honest_verdict": "complete: hierarchical_subgoal_no_new_level_residual_value_head_still_not_separating",
        "wall_diagnosis": "l1_first_contact",
        "generic_agent_reached_level": 0,
        "subgoal_decomposition": [],
        "residual_cause_hypothesis": "value_head_still_not_separating",
        "chosen_submitted_config": "unchanged",
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "generic_first_win_by_config": {
            "explore_budget_200": {
                "first_win_rate": 0.04,
                "first_win_count": 1,
                "variant_attempts_count": 25,
                "variant_attempts": [
                    {"game": "ar25", "first_win": False},
                    {"game": "lp85", "first_win": True},
                ],
            }
        },
    }


def _a2_4677() -> JsonDict:
    return {
        "honest_verdict": "complete: poe_world_factored_planner_no_coverage_gain_residual_logged",
        "candidate_generation_coverage_factored": 0.0,
        "coverage_delta": 0.0,
        "first_win_rate_delta": -0.04,
        "solve_rate_delta": 0.0,
        "residual_bridge_gap": "experts_overfit_prefix",
        "chosen_submitted_config": "unchanged",
        "offline_reproduced": False,
    }


def _a3_4678() -> JsonDict:
    return {
        "honest_verdict": "success: sb26_L2_offline_reproduced",
        "target_game": "sb26",
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "reproduction_gate": {"game": "sb26", "claimed_level": 2, "reached_level": 2, "reproduced": True},
    }


def _a4_4679() -> JsonDict:
    return {
        "honest_verdict": "success: package_refreshed_live_submittable_60_above_33",
        "live_submittable_level_count": 60,
        "ready_for_operator_submit": True,
        "offline_reproduced": True,
    }


def _capstone_4686() -> JsonDict:
    return {
        "honest_verdict": "complete: capability_grew_59_to_60",
        "bridge_crossed_for_solve": False,
        "paper_ready": True,
        "reproducible_total_levels": 60,
        "reproducible_total_levels_delta": 1,
        "live_submittable_level_count": 60,
        "publication_gate": {"paper_ready": True, "frozen_fover_auroc": 0.9131},
        "scorecard": {
            "A1": {"generic_agent_reached_level": 0, "reason": "subgoal_decomposition_missing"},
            "A2": {"coverage_delta": 0.0, "first_win_rate_delta": -0.04},
            "A3": {"registry_authoritative_total_levels": 60, "clean": True},
            "A4": {"live_submittable_level_count": 60, "ready_for_operator_submit": True, "clean": True},
        },
    }


def _write_repo_fixture(
    root: Path,
    *,
    active_milestone: str = "2026.06.432",
    next_present: bool = False,
    registry_total: int = 60,
    upstream_present: bool = True,
) -> None:
    (root / "research-roadmap.yaml").write_text(
        f"milestone: {active_milestone}\n"
        "tasks:\n"
        "  - id: exp4687-phase0\n"
        "    deliverable: results/experiment_4687_archive_431_activate_432.json\n",
        encoding="utf-8",
    )
    if next_present:
        (root / "research-roadmap-next.yaml").write_text(
            "milestone: 2026.06.432\n"
            "tasks:\n"
            "  - id: exp4687-phase0\n"
            "    deliverable: results/experiment_4687_archive_431_activate_432.json\n",
            encoding="utf-8",
        )
    (root / "research-complete.yaml").write_text(
        "milestones:\n"
        "- id: 2026.06.431\n"
        "  finding: prior roadmap archived by conductor\n",
        encoding="utf-8",
    )
    registry = root / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        "schema_version: 1\n"
        "updated: '2026-06-24'\n"
        f"reproducible_total_levels: {registry_total}\n",
        encoding="utf-8",
    )
    proposal = root / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md"
    proposal.parent.mkdir(parents=True, exist_ok=True)
    proposal.write_text("Milestone 2026.06.432 DIRECTED EXPLORATION\n", encoding="utf-8")
    if upstream_present:
        _write_json(root / "results" / "experiment_4676_hierarchical_subgoal_search_live.json", _a1_4676())
        _write_json(root / "results" / "experiment_4677_poe_world_factored_subgoal_planner.json", _a2_4677())
        _write_json(root / "results" / "experiment_4678_levelup_selfplay.json", _a3_4678())
        _write_json(root / "results" / "experiment_4679_refresh_submission_package.json", _a4_4679())
        _write_json(root / "results" / "experiment_4686_capstone_v431.json", _capstone_4686())


def _artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4687_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4687: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4687" in spec
    assert "SCENARIO-CAPSTONE-4687" in spec
    assert "SCENARIO-CAPSTONE-4687-BLOCKED-PRECONDITION" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "DIRECTED EXPLORATION" in spec
    for field, provenance in mod.FIELD_PROVENANCE.items():
        assert field in spec
        assert provenance["principle"] in spec


def test_scenario_capstone_4687_records_true_close_state_when_already_activated(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4687: active `.432` allows a complete record without next YAML."""

    _write_repo_fixture(tmp_path, next_present=False)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    written = json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["honest_verdict"] == "complete: archive_431_activate_432_true_close_state_recorded"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.431",
        "activated_milestone": "2026.06.432",
        "active_milestone_confirmed": True,
        "activation_state": "already_activated_by_conductor",
        "archive_state": "research_complete_contains_2026.06.431",
    }
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["available"] is False
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"][
        "accepted_missing_because_already_active"
    ] is True
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["literal_precondition_passed"] is False
    assert artifact["preconditions_checked"]["active_research_roadmap_yaml"]["milestone"] == "2026.06.432"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is True

    assert artifact["close_state_431"] == {
        "source_capstone_honest_verdict": "complete: capability_grew_59_to_60",
        "a3_level_bank_sb26": {
            "honest_verdict": "success: sb26_L2_offline_reproduced",
            "target_game": "sb26",
            "prior_reproducible_total_levels": 59,
            "reproducible_total_after": 60,
            "reproducible_total_delta": 1,
            "target_level": 2,
            "offline_reproduced": True,
        },
        "a1_hierarchical_subgoal_search": {
            "honest_verdict": "complete: hierarchical_subgoal_no_new_level_residual_value_head_still_not_separating",
            "wall_diagnosis": "l1_first_contact",
            "generic_first_win_rate": 0.04,
            "generic_first_win_count": 1,
            "generic_first_win_games": 25,
            "winning_games": ["lp85"],
            "generic_agent_reached_level": 0,
            "subgoal_decomposition": [],
            "residual": "value_head_still_not_separating",
            "chosen_submitted_config": "unchanged",
        },
        "a2_poe_world_factored_planner": {
            "honest_verdict": "complete: poe_world_factored_planner_no_coverage_gain_residual_logged",
            "candidate_generation_coverage_factored": 0.0,
            "coverage_delta": 0.0,
            "first_win_rate_delta": -0.04,
            "residual": "experts_overfit_prefix",
            "chosen_submitted_config": "unchanged",
        },
        "a4_submission_package": {
            "live_submittable_level_count": 60,
            "beats_submission_baseline": 33,
            "ready_for_operator_submit": True,
        },
        "capstone": {
            "bridge_crossed_for_solve": False,
            "paper_ready": True,
            "frozen_fover_auroc": 0.9131,
        },
    }
    assert artifact["v432_pivot"] == {
        "headline_rationale": "DIRECTED EXPLORATION",
        "operator_frame": "make_a_winning_l1_trajectory_appear_in_the_pool",
        "a1": {
            "lever": "controllable_novelty_e3_proposal_policy",
            "components": ["NGU", "RND", "Strategy-Guided Exploration"],
            "target": "reshape_explorer_action_proposal_distribution",
        },
        "a2": {
            "lever": "program_synthesis_action_effect_proposal_filter",
            "mandatory_gate": "held_out_transition_rejection",
        },
        "a4_retarget": {
            "readiness_lane": "experiment_4605_held_out_first_win",
            "not_replay_package_depth": True,
            "first_scored_submission_baseline": 0.08,
        },
    }
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4687_can_activate_literal_next_roadmap(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4687: present next roadmap is activated onto the active YAML."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.431", next_present=True)

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=2.0,
        now_s=2.5,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8").startswith(
        "milestone: 2026.06.432"
    )
    assert artifact["transition"]["activation_state"] == "activated_from_research_roadmap_next"
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["activation_attempted"] is True
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"][
        "accepted_missing_because_already_active"
    ] is False


def test_scenario_capstone_4687_blockers_are_explicit(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4687-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.431", next_present=False)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=3.0,
        now_s=3.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_432_unavailable"
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is None
    assert artifact["close_state_431"] == {}
    assert artifact["v432_pivot"] == {}
    assert (tmp_path / mod.OUTPUT_REL_PATH).exists()

    checks = _artifact(tmp_path)["preconditions_checked"]
    assert mod._first_blocker(checks) is None

    offline_bad = copy.deepcopy(checks)
    offline_bad["offline_arcade"]["available"] = False
    assert mod._first_blocker(offline_bad) == "offline_arcade"

    smart_bad = copy.deepcopy(checks)
    smart_bad["smart_subset_pretest_gate"]["passed"] = False
    assert mod._first_blocker(smart_bad) == "smart_subset_pretest_gate"

    registry_bad = copy.deepcopy(checks)
    registry_bad["registry"]["reproducible_total_levels"] = 59
    assert mod._first_blocker(registry_bad) == "arc_solve_registry_total_levels_not_60"

    registry_missing = copy.deepcopy(checks)
    registry_missing["registry"]["available"] = False
    assert mod._first_blocker(registry_missing) == "arc_solve_registry"

    for name, expected in {
        "a1_4676": "missing_experiment_4676_hierarchical_subgoal_search_live",
        "a2_4677": "missing_experiment_4677_poe_world_factored_subgoal_planner",
        "a3_4678": "missing_experiment_4678_levelup_selfplay",
        "a4_4679": "missing_experiment_4679_refresh_submission_package",
        "capstone_4686": "missing_experiment_4686_capstone_v431",
        "vnext_design": "missing_research_roadmap_vnext_design",
    }.items():
        bad = copy.deepcopy(checks)
        bad[name]["available"] = False
        assert mod._first_blocker(bad) == expected

    assert mod._command_check(None)["not_run_reason"] == "blocked_before_smart_subset_gate"
    assert mod._float(True, 7.0) == 7.0
    assert mod._float("bad", 9.0) == 9.0
    assert mod._int(False, 2) == 2
    assert mod._int("bad", 3) == 3
    assert mod._generic_first_win_stats(
        {
            "generic_first_win_by_config": {
                "fallback_config": {
                    "first_win_rate": 0.04,
                    "first_win_count": 1,
                    "variant_attempts_count": 25,
                    "variant_attempts": [],
                }
            }
        }
    ) == {"rate": 0.04, "count": 1, "games": 25, "winning_games": ["lp85"]}
    assert mod._registry_total_levels(tmp_path / "missing.yaml") is None
    assert mod._activate_next_roadmap(tmp_path, next_info={"available": False}) == (False, "")

    bad_smart = mod.build_artifact(
        tmp_path,
        started_s=4.0,
        now_s=4.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_red_smart_subset,
    )
    assert bad_smart["honest_verdict"] == "blocked_smart_subset_pretest_gate"


def test_scenario_capstone_4687_field_principle_validation_fails_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4687-FIELD-PRINCIPLES: schema drift fails loudly."""

    valid = _artifact(tmp_path)

    missing = copy.deepcopy(valid)
    del missing["honest_verdict"]
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    bad_verdict = copy.deepcopy(valid)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_substrate = copy.deepcopy(valid)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_provenance = copy.deepcopy(valid)
    bad_provenance["field_provenance"] = {}
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance)

    bad_submission = copy.deepcopy(valid)
    bad_submission["leaderboard_submission"] = True
    with pytest.raises(ValueError, match="leaderboard_submission"):
        mod.validate_artifact(bad_submission)

    blocked = mod._blocked_artifact(
        reason="unit_test",
        preconditions_checked=valid["preconditions_checked"],
        duration_s=0.1,
        cited_upstream_artifacts=valid["cited_upstream_artifacts"],
    )
    blocked["close_state_431"] = {"fabricated": True}
    blocked["reproducibility_checksum"] = "sha256:" + mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    inactive = copy.deepcopy(valid)
    inactive["transition"]["active_milestone_confirmed"] = False
    with pytest.raises(ValueError, match="active .432"):
        mod.validate_artifact(inactive)

    wrong_a3 = copy.deepcopy(valid)
    wrong_a3["close_state_431"]["a3_level_bank_sb26"]["reproducible_total_after"] = 59
    with pytest.raises(ValueError, match="A3"):
        mod.validate_artifact(wrong_a3)

    wrong_a1 = copy.deepcopy(valid)
    wrong_a1["close_state_431"]["a1_hierarchical_subgoal_search"]["wall_diagnosis"] = "selection"
    with pytest.raises(ValueError, match="A1"):
        mod.validate_artifact(wrong_a1)

    wrong_a2 = copy.deepcopy(valid)
    wrong_a2["close_state_431"]["a2_poe_world_factored_planner"]["coverage_delta"] = 1.0
    with pytest.raises(ValueError, match="A2"):
        mod.validate_artifact(wrong_a2)

    wrong_a4 = copy.deepcopy(valid)
    wrong_a4["close_state_431"]["a4_submission_package"]["live_submittable_level_count"] = 33
    with pytest.raises(ValueError, match="A4"):
        mod.validate_artifact(wrong_a4)

    wrong_capstone = copy.deepcopy(valid)
    wrong_capstone["close_state_431"]["capstone"]["bridge_crossed_for_solve"] = True
    with pytest.raises(ValueError, match="capstone"):
        mod.validate_artifact(wrong_capstone)

    wrong_pivot = copy.deepcopy(valid)
    wrong_pivot["v432_pivot"]["headline_rationale"] = "CANDIDATE GENERATION"
    with pytest.raises(ValueError, match="v432 pivot"):
        mod.validate_artifact(wrong_pivot)

    bad_checksum_prefix = copy.deepcopy(valid)
    bad_checksum_prefix["reproducibility_checksum"] = "not-a-checksum"
    with pytest.raises(ValueError, match="sha256-prefixed"):
        mod.validate_artifact(bad_checksum_prefix)

    bad_checksum = copy.deepcopy(valid)
    bad_checksum["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad_checksum)

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("milestone: [\n", encoding="utf-8")
    assert mod._yaml_info(bad_yaml)["parses"] is False
    assert mod._registry_total_levels(bad_yaml) is None

    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- not-a-map\n", encoding="utf-8")
    assert mod._yaml_info(list_yaml)["milestone"] is None
    assert mod._registry_total_levels(list_yaml) is None

    list_json = tmp_path / "list.json"
    list_json.write_text("[1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod._read_json(list_json)

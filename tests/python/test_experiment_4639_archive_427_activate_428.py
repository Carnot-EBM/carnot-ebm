"""Tests for Exp 4639 `.427` archive / `.428` activation.

Spec refs: REQ-CAPSTONE-4639, SCENARIO-CAPSTONE-4639,
SCENARIO-CAPSTONE-4639-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4639-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4639_archive_427_activate_428 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _green_smart_subset(_root: Path) -> mod.CommandResult:
    return mod.CommandResult(
        command=["pytest", "smart-subset"],
        exit_code=0,
        stdout="green",
        stderr="",
    )


def _capstone_4638() -> JsonDict:
    return {
        "honest_verdict": "success: bridge_crossed_live_efficiency_up_1",
        "live_submittable_level_count": 56,
        "ready_for_operator_submit": True,
        "reproducible_total_levels": 56,
        "reproducible_total_levels_delta": 1,
        "first_win_rate_scored": {
            "a2_bare_rate": 0.4072727272727273,
            "a2_delta_vs_bare": 0.1836363636,
            "a2_predictor_rate": 0.5909090909090909,
        },
        "live_action_efficiency": {
            "efficiency_score_term": 1.0,
            "median_actions_to_first_levelup_bare": 2.0,
            "median_actions_to_first_levelup_predictor": 1.0,
            "solve_rate_preserved": True,
            "first_win_rate_bare": 0.4072727272727273,
            "first_win_rate_delta": 0.1836363636,
            "first_win_rate_predictor": 0.5909090909090909,
        },
        "live_solve_rate_delta": {
            "live_solve_rate_bare": 0.04,
            "live_solve_rate_loop": 0.04,
            "solve_rate_delta": 0.0,
            "state_coverage_delta": 2,
        },
        "scorecard": {
            "A1": {
                "live_solve_rate_delta": {
                    "live_solve_rate_bare": 0.04,
                    "live_solve_rate_loop": 0.04,
                    "solve_rate_delta": 0.0,
                    "state_coverage_delta": 2,
                }
            },
            "A2": {
                "live_action_efficiency": {
                    "efficiency_score_term": 1.0,
                    "median_actions_to_first_levelup_bare": 2.0,
                    "median_actions_to_first_levelup_predictor": 1.0,
                    "solve_rate_preserved": True,
                    "first_win_rate_bare": 0.4072727272727273,
                    "first_win_rate_delta": 0.1836363636,
                    "first_win_rate_predictor": 0.5909090909090909,
                },
                "first_win_rate_scored": {
                    "a2_bare_rate": 0.4072727272727273,
                    "a2_delta_vs_bare": 0.1836363636,
                    "a2_predictor_rate": 0.5909090909090909,
                },
            },
            "A3": {
                "banked_plus_one": True,
                "reproduced_levels": 1,
                "registry_reproducible_total_levels": 56,
                "registry_delta_vs_55": 1,
            },
            "A4": {
                "live_submittable_level_count": 56,
                "ready_for_operator_submit": True,
            },
            "A5": {
                "primitive_persisted": {
                    "operator": "persistent_action_effect_memory_operator",
                    "registry_general_gotcha_id": "primitive_persistent_action_effect_memory_operator",
                    "source": "A2_action_effect_candidate_ranker",
                },
                "transfer_games": ["cd82", "sp80", "ka59"],
                "value_added_games": ["cd82", "ka59", "sp80"],
                "verifier_is_oracle": False,
            },
            "A6": {
                "actions_delta_vs_bare": 0.0,
                "included_in_headline": False,
                "live_action_efficiency_integrated": 0.0,
                "live_solve_rate_delta_vs_bare": 0.0,
                "reason": "flagged_adversarial_or_live_critical_excluded",
                "submitted_config_raised_metric_clean": False,
            },
            "headline": {
                "bridge_crossed_by_generation": True,
                "crossing_source": "A2_live_action_efficiency",
            },
        },
        "flagged_artifacts_handled": {
            "excluded_artifacts": ["results/experiment_4633_integration_gate.json"],
            "excluded_details": [
                {
                    "artifact": "results/experiment_4633_integration_gate.json",
                    "critical_flags": [{"kind": "TAUTOLOGY"}],
                    "name": "A6",
                    "reason": "flagged_adversarial_or_live_critical_excluded",
                }
            ],
        },
    }


def _a1_dense_curiosity() -> JsonDict:
    return {
        "honest_verdict": "complete: dense_curiosity_loop_no_live_lift_honest_null_gap_sharpened",
        "live_solve_rate_loop": 0.04,
        "live_solve_rate_bare": 0.04,
        "solve_rate_delta": 0.0,
        "state_coverage_delta": 2,
    }


def _a2_action_effect() -> JsonDict:
    return {
        "honest_verdict": "success: action_effect_predictor_graduated_live_efficiency_up_1",
        "first_win_rate_delta": 0.1836363636,
        "efficiency_score_term": 1.0,
        "live_path_reachable": True,
        "parity_test_green": True,
        "live_measurement": {
            "first_win_rate_bare": 0.4072727272727273,
            "first_win_rate_predictor": 0.5909090909090909,
            "first_win_rate_delta": 0.1836363636,
            "median_actions_to_first_levelup_bare": 2.0,
            "median_actions_to_first_levelup_predictor": 1.0,
            "efficiency_score_term": 1.0,
            "solve_rate_preserved": True,
        },
    }


def _a3_levelup() -> JsonDict:
    return {
        "honest_verdict": "success: ls20_L2_offline_reproduced",
        "target_game": "ls20",
        "reached_level": 2,
        "reproducible_total_levels_before": 55,
        "reproducible_total_levels_after": 56,
        "reproduced_levels": 1,
        "offline_reproduced": True,
    }


def _a4_package() -> JsonDict:
    return {
        "honest_verdict": "success: package_refreshed_live_submittable_56_above_33",
        "live_submittable_count_prev": 55,
        "live_submittable_level_count": 56,
        "count_delta": 1,
        "ready_for_operator_submit": True,
        "offline_reproduced": True,
    }


def _a5_transfer() -> JsonDict:
    return {
        "honest_verdict": "success: primitive_persisted_transfer_sp80_value_added",
        "primitive_persisted": {
            "operator": "persistent_action_effect_memory_operator",
            "registry_general_gotcha_id": "primitive_persistent_action_effect_memory_operator",
            "source": "A2_action_effect_candidate_ranker",
        },
        "transfer_games": ["cd82", "sp80", "ka59"],
        "transfer_value_per_game": {
            "cd82": {"first_win_rate_delta": 0.5, "value_added": True},
            "sp80": {"first_win_rate_delta": 0.4117647059, "value_added": True},
            "ka59": {"first_win_rate_delta": 0.3658536585, "value_added": True},
        },
        "verifier_is_oracle": False,
    }


def _a6_integration() -> JsonDict:
    return {
        "honest_verdict": "success: integrated_action_efficiency_raised_config_shipped",
        "action_efficiency_integrated": {
            "actions_delta_vs_bare": 1.0,
            "efficiency_score_term": 1.0,
            "median_actions_to_first_levelup": 1.0,
            "median_actions_to_first_levelup_bare": 2.0,
        },
        "live_solve_rate_bare": 0.04,
        "live_solve_rate_integrated": 0.04,
        "submitted_config_raised_metric_clean": True,
        "flagged_adversarial": True,
        "corrigendum_pending": [
            {
                "kind": "TAUTOLOGY",
                "severity": "critical",
                "detail": "live_solve_rate_bare=0.04 and live_solve_rate_integrated=0.04 agree.",
            }
        ],
    }


def _write_repo_fixture(root: Path, *, active_milestone: str = "2026.06.428") -> None:
    (root / "research-roadmap.yaml").write_text(
        f"milestone: {active_milestone}\n"
        "tasks:\n"
        "  - id: exp4639-phase0\n"
        "    deliverable: results/experiment_4639_archive_427_activate_428.json\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "milestones:\n"
        "- id: 2026.06.427\n"
        "  finding: prior roadmap archived by conductor\n",
        encoding="utf-8",
    )
    registry = root / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        "schema_version: 1\n"
        "updated: '2026-06-23'\n"
        "reproducible_total_levels: 56\n",
        encoding="utf-8",
    )
    proposal = root / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md"
    proposal.parent.mkdir(parents=True, exist_ok=True)
    proposal.write_text("Milestone 2026.06.428 ENERGY DRIVES GENERATION\n", encoding="utf-8")
    _write_json(root / "results" / "experiment_4638_capstone_v427.json", _capstone_4638())
    _write_json(root / "results" / "experiment_4628_dense_curiosity_progress_loop.json", _a1_dense_curiosity())
    _write_json(root / "results" / "experiment_4629_graduate_action_effect_predictor_live.json", _a2_action_effect())
    _write_json(root / "results" / "experiment_4630_levelup_selfplay.json", _a3_levelup())
    _write_json(root / "results" / "experiment_4631_refresh_submission_package.json", _a4_package())
    _write_json(root / "results" / "experiment_4632_primitive_persist_transfer.json", _a5_transfer())
    _write_json(root / "results" / "experiment_4633_integration_gate.json", _a6_integration())


def _artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4639_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4639: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4639" in spec
    assert "SCENARIO-CAPSTONE-4639" in spec
    assert "SCENARIO-CAPSTONE-4639-BLOCKED-PRECONDITION" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "ENERGY DRIVES GENERATION" in spec
    assert "GAP-ARCH-GOAL-NOT-VERIFIED" in spec
    for field, provenance in mod.FIELD_PROVENANCE.items():
        assert field in spec
        assert provenance["principle"] in spec


def test_scenario_capstone_4639_records_true_close_state_when_already_activated(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4639: consumed next-roadmap still writes .427 close-state."""

    _write_repo_fixture(tmp_path)

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
    assert artifact["honest_verdict"] == "complete: archive_427_activate_428_true_close_state_recorded"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.427",
        "activated_milestone": "2026.06.428",
        "active_milestone_confirmed": True,
        "activation_state": "already_active_roadmap_next_consumed",
        "archive_state": "research_complete_contains_2026.06.427",
    }
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["available"] is False
    assert artifact["preconditions_checked"]["active_research_roadmap_yaml"]["milestone"] == "2026.06.428"
    assert artifact["preconditions_checked"]["offline_arcade"]["available"] is True
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is True

    close = artifact["close_state_427"]
    assert close["source_capstone_honest_verdict"] == "success: bridge_crossed_live_efficiency_up_1"
    assert close["bridge_crossed"] == {
        "crossed": True,
        "source": "A2_live_action_efficiency",
        "capstone_honest_verdict": "success: bridge_crossed_live_efficiency_up_1",
    }
    assert close["a2_action_effect_predictor"] == {
        "honest_verdict": "success: action_effect_predictor_graduated_live_efficiency_up_1",
        "first_win_rate_bare": 0.4072727272727273,
        "first_win_rate_predictor": 0.5909090909090909,
        "first_win_rate_delta": 0.1836363636,
        "median_actions_to_first_levelup_bare": 2.0,
        "median_actions_to_first_levelup_predictor": 1.0,
        "efficiency_score_term": 1.0,
        "solve_rate_preserved": True,
        "live_path_reachable": True,
        "parity_test_green": True,
    }
    assert close["a5_action_effect_transfer"] == {
        "honest_verdict": "success: primitive_persisted_transfer_sp80_value_added",
        "operator": "persistent_action_effect_memory_operator",
        "transfer_games": ["cd82", "sp80", "ka59"],
        "cd82_first_win_delta": 0.5,
        "sp80_value_added": True,
        "value_added_games": ["cd82", "ka59", "sp80"],
        "verifier_is_oracle": False,
    }
    assert close["a1_dense_curiosity_loop"] == {
        "honest_verdict": "complete: dense_curiosity_loop_no_live_lift_honest_null_gap_sharpened",
        "live_solve_rate_loop": 0.04,
        "live_solve_rate_bare": 0.04,
        "solve_rate_delta": 0.0,
        "third_consecutive_solve_rate_null": True,
        "state_coverage_delta": 2,
    }
    assert close["a3_level_bank_ls20"] == {
        "honest_verdict": "success: ls20_L2_offline_reproduced",
        "target_game": "ls20",
        "target_level": 2,
        "reproducible_total_before": 55,
        "reproducible_total_after": 56,
        "reproducible_total_delta": 1,
        "offline_reproduced": True,
    }
    assert close["a4_submission_package"] == {
        "honest_verdict": "success: package_refreshed_live_submittable_56_above_33",
        "live_submittable_level_count": 56,
        "beats_submission_baseline": 33,
        "ready_for_operator_submit": True,
    }
    assert close["a6_action_efficiency_integration"] == {
        "honest_verdict": "success: integrated_action_efficiency_raised_config_shipped",
        "action_efficiency_shipped": True,
        "live_solve_rate_bare": 0.04,
        "live_solve_rate_integrated": 0.04,
        "solve_rate_tautology_quarantined": True,
        "submitted_config_raised_metric_clean": True,
        "capstone_headline_included": False,
    }
    assert close["registry_total_levels"] == 56
    assert close["generation_vs_reranking_lesson"] == "generation_levers_crossed_rerankers_did_not"

    assert artifact["v428_pivot"] == {
        "headline_rationale": "ENERGY DRIVES GENERATION",
        "builds_on": "success: bridge_crossed_live_efficiency_up_1",
        "a1": {
            "lever": "exp4020_graded_is_goal_goal_energy",
            "role": "LIVE goal-ENERGY heuristic",
            "operator_menu": "#1",
            "target": "graph_explore_solve_v2 search heuristic",
            "closes_gap": "GAP-ARCH-GOAL-NOT-VERIFIED",
        },
        "a2": {
            "lever": "action_effect_predictor_search_expansion_prior",
            "previous_role": "candidate_RANKER",
            "new_role": "search_EXPANSION_PRIOR",
        },
    }
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4639_blocks_without_fabricating_missing_a2(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4639-BLOCKED-PRECONDITION: missing source blocks honestly."""

    _write_repo_fixture(tmp_path)
    (tmp_path / "results" / "experiment_4629_graduate_action_effect_predictor_live.json").unlink()

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=2.0,
        now_s=2.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_missing_experiment_4629_graduate_action_effect_predictor_live"
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["preconditions_checked"]["a2_exp4629"]["available"] is False
    assert artifact["close_state_427"] == {}
    assert artifact["v428_pivot"] == {}
    assert (tmp_path / mod.OUTPUT_REL_PATH).exists()
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4639_precondition_order_and_blockers(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4639: blocker classification is explicit."""

    checks = _artifact(tmp_path)["preconditions_checked"]
    assert mod._first_blocker(checks) is None
    assert mod._command_check(None)["not_run_reason"] == "blocked_before_smart_subset_gate"

    next_bad = copy.deepcopy(checks)
    next_bad["research_roadmap_next_yaml"]["available"] = True
    next_bad["research_roadmap_next_yaml"]["parses"] = False
    assert mod._first_blocker(next_bad) == "research_roadmap_next_yaml"

    next_present = copy.deepcopy(checks)
    next_present["research_roadmap_next_yaml"]["available"] = True
    next_present["research_roadmap_next_yaml"]["parses"] = True
    next_present["research_roadmap_next_yaml"]["milestone"] = "2026.06.428"
    assert mod._transition(next_present, complete=True)["activation_state"] == "activated_from_research_roadmap_next"

    active_bad = copy.deepcopy(checks)
    active_bad["active_research_roadmap_yaml"]["milestone"] = "2026.06.427"
    assert mod._first_blocker(active_bad) == "research_roadmap_428_unavailable"

    offline_bad = copy.deepcopy(checks)
    offline_bad["offline_arcade"]["available"] = False
    assert mod._first_blocker(offline_bad) == "offline_arcade"

    smart_bad = copy.deepcopy(checks)
    smart_bad["smart_subset_pretest_gate"]["passed"] = False
    assert mod._first_blocker(smart_bad) == "smart_subset_pretest_gate"

    registry_bad = copy.deepcopy(checks)
    registry_bad["registry"]["available"] = False
    assert mod._first_blocker(registry_bad) == "arc_solve_registry"

    capstone_bad = copy.deepcopy(checks)
    capstone_bad["capstone_4638"]["available"] = False
    assert mod._first_blocker(capstone_bad) == "missing_experiment_4638_capstone_v427"

    vnext_bad = copy.deepcopy(checks)
    vnext_bad["vnext_design"]["available"] = False
    assert mod._first_blocker(vnext_bad) == "missing_research_roadmap_vnext_design"

    assert mod._float(True, 7.0) == 7.0
    assert mod._float("bad", 9.0) == 9.0
    assert mod._int(False, 2) == 2
    assert mod._int("bad", 3) == 3
    assert mod._registry_total_levels(tmp_path / "missing.yaml") is None

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("milestone: [\n", encoding="utf-8")
    assert mod._yaml_info(bad_yaml)["parses"] is False

    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- not-a-map\n", encoding="utf-8")
    assert mod._yaml_info(list_yaml)["milestone"] is None
    assert mod._registry_total_levels(list_yaml) is None

    poisoned_registry = tmp_path / "poisoned_registry.yaml"
    poisoned_registry.write_text("a: : :\n- [\n", encoding="utf-8")
    assert mod._registry_total_levels(poisoned_registry) is None

    list_json = tmp_path / "list.json"
    list_json.write_text("[1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod._read_json(list_json)

    flagged_capstone = {
        "flagged_artifacts_handled": {
            "excluded_details": [{"critical_flags": [{"kind": "TAUTOLOGY"}]}]
        }
    }
    assert mod._has_tautology_flag(flagged_capstone) is True
    assert mod._has_tautology_flag({"corrigendum_pending": [{"kind": "OTHER"}]}) is False


def test_scenario_capstone_4639_field_principle_validation_fails_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4639-FIELD-PRINCIPLES: schema drift fails loudly."""

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
    blocked["close_state_427"] = {"fabricated": True}
    blocked["reproducibility_checksum"] = "sha256:" + mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    inactive = copy.deepcopy(valid)
    inactive["transition"]["active_milestone_confirmed"] = False
    with pytest.raises(ValueError, match="active .428"):
        mod.validate_artifact(inactive)

    wrong_a2 = copy.deepcopy(valid)
    wrong_a2["close_state_427"]["a2_action_effect_predictor"]["first_win_rate_delta"] = 0.0
    with pytest.raises(ValueError, match="A2"):
        mod.validate_artifact(wrong_a2)

    wrong_a5 = copy.deepcopy(valid)
    wrong_a5["close_state_427"]["a5_action_effect_transfer"]["cd82_first_win_delta"] = 0.0
    with pytest.raises(ValueError, match="A5"):
        mod.validate_artifact(wrong_a5)

    wrong_a1 = copy.deepcopy(valid)
    wrong_a1["close_state_427"]["a1_dense_curiosity_loop"]["state_coverage_delta"] = 0
    with pytest.raises(ValueError, match="A1"):
        mod.validate_artifact(wrong_a1)

    wrong_a3 = copy.deepcopy(valid)
    wrong_a3["close_state_427"]["a3_level_bank_ls20"]["reproducible_total_after"] = 55
    with pytest.raises(ValueError, match="A3"):
        mod.validate_artifact(wrong_a3)

    wrong_a4 = copy.deepcopy(valid)
    wrong_a4["close_state_427"]["a4_submission_package"]["live_submittable_level_count"] = 33
    with pytest.raises(ValueError, match="A4"):
        mod.validate_artifact(wrong_a4)

    wrong_a6 = copy.deepcopy(valid)
    wrong_a6["close_state_427"]["a6_action_efficiency_integration"]["solve_rate_tautology_quarantined"] = False
    with pytest.raises(ValueError, match="A6"):
        mod.validate_artifact(wrong_a6)

    wrong_pivot = copy.deepcopy(valid)
    wrong_pivot["v428_pivot"]["headline_rationale"] = "rerank harder"
    with pytest.raises(ValueError, match="v428 pivot"):
        mod.validate_artifact(wrong_pivot)

    bad_checksum_prefix = copy.deepcopy(valid)
    bad_checksum_prefix["reproducibility_checksum"] = "not-a-checksum"
    with pytest.raises(ValueError, match="sha256-prefixed"):
        mod.validate_artifact(bad_checksum_prefix)

    bad_checksum = copy.deepcopy(valid)
    bad_checksum["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad_checksum)

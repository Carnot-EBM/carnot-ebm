"""Tests for Exp 4591 `.423` archive / `.424` activation.

Spec refs: REQ-CAPSTONE-4591, SCENARIO-CAPSTONE-4591,
SCENARIO-CAPSTONE-4591-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4591_archive_423_activate_424 as mod


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


def _capstone_4590() -> JsonDict:
    return {
        "honest_verdict": "success: live_submittable_above_33_feature_router_false_negative_risk_open",
        "live_submittable_level_count": 54,
        "reproducible_total_levels": 54,
        "reproducible_total_levels_delta": 1,
        "generic_transfer_rate_over_variants": 0.04,
        "ready_for_operator_submit": True,
        "live_submittable_moved": {
            "baseline": 33,
            "a1_live_submittable_level_count": 53,
            "a1_count_delta": 20,
            "env_adaptive_resolve_recovered": ["sc25"],
            "moved": True,
            "verifier_is_oracle": False,
        },
        "scorecard": {
            "A1": {
                "live_submittable_baseline": 33,
                "live_submittable_level_count": 53,
                "count_delta": 20,
                "env_adaptive_resolve_recovered": ["sc25"],
                "moved_above_33": True,
            },
            "A2": {
                "reproducible_total_before": 53,
                "reproducible_total_after": 54,
                "reproducible_total_delta": 1,
            },
            "A3": {
                "generic_transfer_moved": False,
                "headline_numbers_aggregated": False,
                "included_in_headline": False,
                "reason": "a3_flagged_false_negative_risk_open",
            },
            "A4": {
                "firstwin_delta_counted": 0,
                "included_in_headline": False,
                "reason": "flagged_adversarial_excluded",
            },
            "A5": {
                "new_levels_banked": 0,
                "primitive_persisted": {
                    "operator": "env_adaptive_resolve_operator",
                    "registry_general_gotcha_id": "primitive_env_adaptive_resolve_operator",
                },
                "transfer_games": ["s5i5", "ft09", "sb26"],
                "value_added_games": ["ft09", "s5i5", "sb26"],
            },
            "A6": {
                "included_in_headline": False,
                "integration_headline_aggregated": False,
                "reason": "flagged_adversarial_excluded",
            },
            "B1": {
                "live_submittable_level_count": 54,
                "reproducible_total_levels": 54,
                "generic_transfer_rate_over_variants": 0.04,
                "generic_transfer_ci": [0.0, 0.1],
                "action_efficiency_score": 1.0,
                "action_efficiency_ci": [1.0, 1.0],
            },
        },
    }


def _feature_router_4582() -> JsonDict:
    return {
        "honest_verdict": "complete: feature_router_no_value_honest_null_transfer_gap_sharpened",
        "generic_transfer_rate_with_router": 0.04,
        "generic_transfer_rate_baseline": 0.04,
        "transfer_delta": 0.0,
        "winner_generated": {
            "attempted_count": 25,
            "generated_count": 1,
            "not_generated_count": 24,
            "with_router": True,
            "without_router": True,
            "random_route": True,
        },
        "random_route_control_passed": False,
        "false_negative_risk_checked": False,
        "missing_verifier_gaps": [
            "feature_router_residual_generation_gap avatar_navigation:goal_distance_astar:variant_wired=False unsolved_count=12",
            "feature_router_residual_generation_gap keyboard_graph:systematic_bfs:variant_wired=True unsolved_count=7",
            "feature_router_residual_generation_gap click_connect:goal_distance_astar:variant_wired=False unsolved_count=3",
            "feature_router_residual_generation_gap click_graph:diversity_graph_explore:variant_wired=True unsolved_count=1",
            "feature_router_residual_generation_gap config_toggle:diversity_graph_explore:variant_wired=True unsolved_count=1",
        ],
    }


def _live_submit() -> JsonDict:
    return {
        "experiment": "arc3_live_submit",
        "run_date": "2026-06-21T17:04:45Z",
        "leaderboard_submitted": True,
        "live_total_levels": 33,
        "claimed_total_levels": 34,
        "games_env_matched": 17,
        "games": 18,
        "per_game": [
            {"game": "sc25", "claimed": 1, "live_level": 0, "env_match": False},
            {"game": "ar25", "claimed": 1, "live_level": 1, "env_match": True},
        ],
    }


def _write_repo_fixture(root: Path, *, active_milestone: str = "2026.06.424") -> None:
    (root / "research-roadmap.yaml").write_text(
        f"milestone: {active_milestone}\n"
        "tasks:\n"
        "  - id: exp4591-phase0\n"
        "    deliverable: results/experiment_4591_archive_423_activate_424.json\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "milestones:\n"
        "- id: 2026.06.423\n"
        "  finding: prior roadmap archived by conductor\n",
        encoding="utf-8",
    )
    registry = root / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        "schema_version: 1\n"
        "updated: '2026-06-22'\n"
        "reproducible_total_levels: 54\n",
        encoding="utf-8",
    )
    _write_json(root / "results" / "experiment_4590_capstone_v423.json", _capstone_4590())
    _write_json(root / "results" / "experiment_4582_feature_router_transfer.json", _feature_router_4582())
    _write_json(root / "results" / "arc3_live_submit.json", _live_submit())


def _artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4591_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4591: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4591" in spec
    assert "SCENARIO-CAPSTONE-4591" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "winner_generated=1/25" in spec
    assert "variant_wired=False" in spec
    for field, provenance in mod.FIELD_PROVENANCE.items():
        assert field in spec
        assert provenance["principle"] in spec


def test_scenario_capstone_4591_records_true_close_state_when_already_activated(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4591: consumed next-roadmap still writes .423 close-state."""

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
    assert artifact["honest_verdict"] == "complete: archive_423_activate_424_true_close_state_recorded"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.423",
        "activated_milestone": "2026.06.424",
        "active_milestone_confirmed": True,
        "activation_state": "already_active_roadmap_next_consumed",
        "archive_state": "research_complete_contains_2026.06.423",
    }
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["available"] is False
    assert artifact["preconditions_checked"]["active_research_roadmap_yaml"]["milestone"] == "2026.06.424"
    assert artifact["preconditions_checked"]["offline_arcade"]["available"] is True
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is True

    close = artifact["close_state_423"]
    assert close["source_capstone_honest_verdict"].startswith("success: live_submittable_above_33")
    assert close["a1_live_submission_gap"] == {
        "baseline": 33,
        "live_submittable_level_count": 53,
        "count_delta": 20,
        "env_adaptive_resolve_recovered": ["sc25"],
        "ready_for_operator_submit": True,
        "verifier_is_oracle": False,
    }
    assert close["a2_levelup_selfplay_ar25"] == {
        "target_game": "ar25",
        "target_level": 2,
        "reproducible_total_before": 53,
        "reproducible_total_after": 54,
        "reproducible_total_delta": 1,
    }
    assert close["a3_feature_router_null"]["generic_transfer_rate_with_router"] == 0.04
    assert close["a3_feature_router_null"]["winner_generated"] == {
        "generated_count": 1,
        "attempted_count": 25,
        "not_generated_count": 24,
    }
    assert close["a3_feature_router_null"]["variant_wired_false_residual"] == {
        "unsolved_count": 15,
        "not_generated_count": 24,
        "summary": "15/24 residual not-generated variants selected unwired approaches",
    }
    assert close["a4_diversity_floor_null"]["firstwin_delta_counted"] == 0
    assert close["a5_env_adaptive_resolve_operator"]["operator"] == "env_adaptive_resolve_operator"
    assert close["a5_env_adaptive_resolve_operator"]["drift_recovery_games"] == [
        "ft09",
        "s5i5",
        "sb26",
    ]
    assert close["a5_env_adaptive_resolve_operator"]["new_levels_banked"] == 0
    assert close["a6_integrated_live_submittable"]["live_submittable_level_count_integrated"] == 54
    assert close["a6_integrated_live_submittable"]["beats_last_submission_gate"] is True
    assert close["live_submission_standing_gate"]["live_total_levels"] == 33
    assert close["registry_total_levels"] == 54
    assert close["generation_not_ranking_diagnosis"]["quadruply_confirmed"] is True

    pivot = artifact["v424_pivot"]
    assert pivot["implementation_target"] == "measure_generic_transfer_over_variants.variant_runner"
    assert pivot["selected_approach_must_run_and_generate"] is True
    assert pivot["winner_generated_target"] == "1/25 -> up"
    assert pivot["residual_to_close"] == "variant_wired_false_generation_gap_15_of_24"
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4591_blocks_without_fabricating_when_424_not_active(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4591: missing activation evidence blocks honestly."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.423")

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=2.0,
        now_s=2.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_424_unavailable"
    assert artifact["preconditions_checked"]["active_research_roadmap_yaml"]["milestone"] == "2026.06.423"
    assert artifact["close_state_423"] == {}
    assert artifact["v424_pivot"] == {}
    assert artifact["transition"]["active_milestone_confirmed"] is False
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4591_precondition_blockers_and_helpers_are_defensive(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4591: missing resources classify without fabricated data."""

    preconditions = _artifact(tmp_path)["preconditions_checked"]

    next_only = copy.deepcopy(preconditions)
    next_only["active_research_roadmap_yaml"]["milestone"] = "2026.06.423"
    next_only["research_roadmap_next_yaml"]["available"] = True
    next_only["research_roadmap_next_yaml"]["parses"] = True
    next_only["research_roadmap_next_yaml"]["milestone"] = "2026.06.424"
    assert mod._first_blocker(next_only) is None
    assert mod._transition(next_only, complete=True)["activation_state"] == (
        "activated_from_research_roadmap_next"
    )

    offline_bad = copy.deepcopy(preconditions)
    offline_bad["offline_arcade"]["available"] = False
    assert mod._first_blocker(offline_bad) == "offline_arcade"

    smart_bad = copy.deepcopy(preconditions)
    smart_bad["smart_subset_pretest_gate"]["passed"] = False
    assert mod._first_blocker(smart_bad) == "smart_subset_pretest_gate"

    registry_bad = copy.deepcopy(preconditions)
    registry_bad["registry"]["available"] = False
    assert mod._first_blocker(registry_bad) == "arc_solve_registry"

    capstone_bad = copy.deepcopy(preconditions)
    capstone_bad["capstone_4590"]["available"] = False
    assert mod._first_blocker(capstone_bad) == "missing_experiment_4590_capstone_v423"

    router_bad = copy.deepcopy(preconditions)
    router_bad["feature_router_4582"]["available"] = False
    assert mod._first_blocker(router_bad) == "missing_experiment_4582_feature_router_transfer"

    live_bad = copy.deepcopy(preconditions)
    live_bad["arc3_live_submit"]["available"] = False
    assert mod._first_blocker(live_bad) == "missing_arc3_live_submit"

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

    bool_registry = tmp_path / "bool_registry.yaml"
    bool_registry.write_text("reproducible_total_levels: true\n", encoding="utf-8")
    assert mod._registry_total_levels(bool_registry) is None
    poisoned_registry = tmp_path / "poisoned_registry.yaml"
    poisoned_registry.write_text("a: : :\n- [\n", encoding="utf-8")
    assert mod._registry_total_levels(poisoned_registry) is None

    list_json = tmp_path / "list.json"
    list_json.write_text("[1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod._read_json(list_json)

    assert mod._variant_wired_false_residual(_feature_router_4582()) == {
        "unsolved_count": 15,
        "not_generated_count": 24,
        "summary": "15/24 residual not-generated variants selected unwired approaches",
    }
    noisy_router = _feature_router_4582()
    noisy_router["missing_verifier_gaps"].append({"not": "a string"})
    assert mod._variant_wired_false_residual(noisy_router)["unsolved_count"] == 15
    assert mod._variant_wired_false_residual({"winner_generated": {"not_generated_count": 0}})[
        "summary"
    ] == "0/0 residual not-generated variants selected unwired approaches"


def test_scenario_capstone_4591_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4591-FIELD-PRINCIPLES: schema drift fails loudly."""

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

    blocked = mod._blocked_artifact(
        reason="unit_test",
        preconditions_checked=valid["preconditions_checked"],
        duration_s=0.1,
        cited_upstream_artifacts=valid["cited_upstream_artifacts"],
    )
    blocked["close_state_423"] = {"fabricated": True}
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    inactive = copy.deepcopy(valid)
    inactive["transition"]["active_milestone_confirmed"] = False
    with pytest.raises(ValueError, match="active .424"):
        mod.validate_artifact(inactive)

    wrong_a1 = copy.deepcopy(valid)
    wrong_a1["close_state_423"]["a1_live_submission_gap"]["count_delta"] = 19
    with pytest.raises(ValueError, match="A1 live-submission gap"):
        mod.validate_artifact(wrong_a1)

    wrong_a2 = copy.deepcopy(valid)
    wrong_a2["close_state_423"]["a2_levelup_selfplay_ar25"]["reproducible_total_after"] = 53
    with pytest.raises(ValueError, match="A2 ar25"):
        mod.validate_artifact(wrong_a2)

    wrong_a3 = copy.deepcopy(valid)
    wrong_a3["close_state_423"]["a3_feature_router_null"]["winner_generated"]["generated_count"] = 2
    with pytest.raises(ValueError, match="A3 feature-router"):
        mod.validate_artifact(wrong_a3)

    wrong_a4 = copy.deepcopy(valid)
    wrong_a4["close_state_423"]["a4_diversity_floor_null"]["firstwin_delta_counted"] = 1
    with pytest.raises(ValueError, match="A4 diversity"):
        mod.validate_artifact(wrong_a4)

    wrong_a5 = copy.deepcopy(valid)
    wrong_a5["close_state_423"]["a5_env_adaptive_resolve_operator"]["new_levels_banked"] = 1
    with pytest.raises(ValueError, match="A5 env-adaptive"):
        mod.validate_artifact(wrong_a5)

    wrong_a6 = copy.deepcopy(valid)
    wrong_a6["close_state_423"]["a6_integrated_live_submittable"]["live_submittable_level_count_integrated"] = 33
    with pytest.raises(ValueError, match="A6 integrated"):
        mod.validate_artifact(wrong_a6)

    wrong_gate = copy.deepcopy(valid)
    wrong_gate["close_state_423"]["live_submission_standing_gate"]["live_total_levels"] = 34
    with pytest.raises(ValueError, match="33 gate"):
        mod.validate_artifact(wrong_gate)

    wrong_generation = copy.deepcopy(valid)
    wrong_generation["close_state_423"]["generation_not_ranking_diagnosis"]["quadruply_confirmed"] = False
    with pytest.raises(ValueError, match="generation-not-ranking"):
        mod.validate_artifact(wrong_generation)

    wrong_pivot = copy.deepcopy(valid)
    wrong_pivot["v424_pivot"]["implementation_target"] = "ranking_only"
    with pytest.raises(ValueError, match="v424 pivot"):
        mod.validate_artifact(wrong_pivot)

    bad_checksum_prefix = copy.deepcopy(valid)
    bad_checksum_prefix["reproducibility_checksum"] = "not-a-checksum"
    with pytest.raises(ValueError, match="sha256-prefixed"):
        mod.validate_artifact(bad_checksum_prefix)

    bad_checksum = copy.deepcopy(valid)
    bad_checksum["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad_checksum)

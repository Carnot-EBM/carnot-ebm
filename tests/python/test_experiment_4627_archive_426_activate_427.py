"""Tests for Exp 4627 `.426` archive / `.427` activation.

Spec refs: REQ-CAPSTONE-4627, SCENARIO-CAPSTONE-4627,
SCENARIO-CAPSTONE-4627-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4627-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4627_archive_426_activate_427 as mod


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


def _capstone_4626() -> JsonDict:
    return {
        "honest_verdict": "complete: bridge_characterized_cause_isolated_no_live_lift",
        "binding_bridge_cause": "compute_cost",
        "reproducible_total_levels": 55,
        "reproducible_total_levels_delta": 0,
        "live_submittable_level_count": 55,
        "ready_for_operator_submit": True,
        "first_win_rate_scored": {
            "actions_delta": 0.0,
            "bare_rate": 0.04,
            "delta_vs_linear_baseline": 0.0,
            "linear_baseline_rate": 0.04,
            "median_actions_to_first_levelup_graduated": 20.0,
            "median_actions_to_first_levelup_linear_baseline": 20.0,
            "solve_rate_graduated": 0.04,
            "solve_rate_linear_baseline": 0.04,
        },
        "scorecard": {
            "A1": {
                "binding_bridge_cause": "compute_cost",
                "indicated_fix": "decision-point-only eval/cached features for live frontier nodes",
                "included_in_headline": True,
            },
            "A2": {
                "included_in_headline": False,
                "reason": "flagged_adversarial_or_live_critical_excluded",
            },
            "A3": {
                "banked_plus_one": False,
                "registry_delta_vs_55": 0,
                "registry_reproducible_total_levels": 55,
                "reproduced_levels": 0,
            },
            "A4": {
                "live_submittable_level_count": 55,
                "ready_for_operator_submit": True,
            },
        },
    }


def _a1_bridge() -> JsonDict:
    return {
        "honest_verdict": "success: bridge_cause_isolated_compute_fix_identified",
        "binding_bridge_cause": "compute_cost",
        "positive_control_passed": True,
        "false_negative_risk_checked": True,
        "indicated_fix": "decision-point-only eval/cached features for live frontier nodes",
    }


def _a2_spatial_value() -> JsonDict:
    return {
        "honest_verdict": "complete: spatial_value_head_graduated_no_live_value_honest_null_gap_sharpened",
        "first_win_delta": 0.0,
        "actions_delta": 0.0,
        "solve_rate_bare": 0.04,
        "solve_rate_graduated": 0.04,
        "solve_rate_linear_baseline": 0.04,
        "first_win_rate_graduated": 0.04,
        "first_win_rate_linear_baseline": 0.04,
        "median_actions_to_first_levelup_graduated": 20.0,
        "median_actions_to_first_levelup_linear_baseline": 20.0,
        "flagged_adversarial": True,
        "false_negative_risk_checked": True,
    }


def _a3_levelup() -> JsonDict:
    return {
        "honest_verdict": "complete: sk48_delta_identified_no_bank",
        "target_game": "sk48",
        "reproduced_levels": 0,
        "offline_reproduced": False,
        "reproduction_gate": {
            "claimed_level": 1,
            "game": "sk48",
            "reached_level": 1,
            "reproduced": True,
        },
    }


def _a4_package() -> JsonDict:
    return {
        "honest_verdict": "complete: package_refreshed_unchanged_depth.",
        "live_submittable_level_count": 55,
        "live_submittable_count_prev": 55,
        "count_delta": 0,
        "ready_for_operator_submit": True,
        "offline_reproduced": True,
    }


def _write_repo_fixture(root: Path, *, active_milestone: str = "2026.06.427") -> None:
    (root / "research-roadmap.yaml").write_text(
        f"milestone: {active_milestone}\n"
        "tasks:\n"
        "  - id: exp4627-phase0\n"
        "    deliverable: results/experiment_4627_archive_426_activate_427.json\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "milestones:\n"
        "- id: 2026.06.426\n"
        "  finding: prior roadmap archived by conductor\n",
        encoding="utf-8",
    )
    registry = root / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        "schema_version: 1\n"
        "updated: '2026-06-23'\n"
        "reproducible_total_levels: 55\n",
        encoding="utf-8",
    )
    note = root / "docs" / "research-notes" / "arc-representation-not-the-bottleneck-2026-06-23.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("representation is NOT the bottleneck; Curiosity-Critic arXiv:2604.18701\n", encoding="utf-8")
    proposal = root / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md"
    proposal.parent.mkdir(parents=True, exist_ok=True)
    proposal.write_text("Milestone 2026.06.427 GENERATE better live exploration\n", encoding="utf-8")
    _write_json(root / "results" / "experiment_4626_capstone_v426.json", _capstone_4626())
    _write_json(root / "results" / "experiment_4616_offline_live_bridge_disambiguation.json", _a1_bridge())
    _write_json(root / "results" / "experiment_4617_graduate_spatial_value_head_live.json", _a2_spatial_value())
    _write_json(root / "results" / "experiment_4618_levelup_selfplay.json", _a3_levelup())
    _write_json(root / "results" / "experiment_4619_refresh_submission_package.json", _a4_package())


def _artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4627_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4627: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4627" in spec
    assert "SCENARIO-CAPSTONE-4627" in spec
    assert "SCENARIO-CAPSTONE-4627-BLOCKED-PRECONDITION" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "value-head-reranker-into-live-search lever twice" in spec
    assert "Curiosity-Critic arXiv:2604.18701" in spec
    for field, provenance in mod.FIELD_PROVENANCE.items():
        assert field in spec
        assert provenance["principle"] in spec


def test_scenario_capstone_4627_records_true_close_state_when_already_activated(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4627: consumed next-roadmap still writes .426 close-state."""

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
    assert artifact["honest_verdict"] == "complete: archive_426_activate_427_true_close_state_recorded"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.426",
        "activated_milestone": "2026.06.427",
        "active_milestone_confirmed": True,
        "activation_state": "already_active_roadmap_next_consumed",
        "archive_state": "research_complete_contains_2026.06.426",
    }
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["available"] is False
    assert artifact["preconditions_checked"]["active_research_roadmap_yaml"]["milestone"] == "2026.06.427"
    assert artifact["preconditions_checked"]["offline_arcade"]["available"] is True
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is True

    close = artifact["close_state_426"]
    assert close["source_capstone_honest_verdict"] == "complete: bridge_characterized_cause_isolated_no_live_lift"
    assert close["capability"] == {
        "reproducible_total_levels_before": 55,
        "reproducible_total_levels_after": 55,
        "reproducible_total_levels_delta": 0,
        "capability_flat": True,
        "consecutive_flat_milestones": 2,
    }
    assert close["a1_bridge_disambiguation"] == {
        "status": "compute_cause_isolated",
        "honest_verdict": "success: bridge_cause_isolated_compute_fix_identified",
        "binding_bridge_cause": "compute_cost",
        "indicated_fix": "decision-point-only eval/cached features for live frontier nodes",
        "positive_control_passed": True,
        "false_negative_risk_checked": True,
    }
    assert close["a2_spatial_value_head_live"] == {
        "status": "honest_null",
        "honest_verdict": "complete: spatial_value_head_graduated_no_live_value_honest_null_gap_sharpened",
        "graduated_to_live_path": True,
        "compute_fix_applied": True,
        "first_win_delta": 0.0,
        "actions_delta": 0.0,
        "solve_rate_bare": 0.04,
        "solve_rate_graduated": 0.04,
        "solve_rate_linear_baseline": 0.04,
        "first_win_rate_graduated": 0.04,
        "first_win_rate_linear_baseline": 0.04,
        "reranker_falsified_twice": True,
        "falsified_milestones": [".425 linear", ".426 SpatialValueNet+compute-fix"],
    }
    assert close["a3_levelup_selfplay"] == {
        "status": "no_bank",
        "target_game": "sk48",
        "attempted_transition": "L1->L2",
        "reached_level": 1,
        "new_levels_banked": 0,
        "offline_reproduced": False,
    }
    assert close["a4_package"] == {
        "live_submittable_level_count": 55,
        "beats_scorecard_baseline": 33,
        "ready_for_operator_submit": True,
    }

    assert artifact["v427_pivot"] == {
        "headline_rationale": "PIVOT from reranking to GENERATING better live exploration",
        "reranking_retired": True,
        "generate_better_live_exploration": True,
        "a1": {
            "lever": "dense_curiosity_learning_progress_loop",
            "target": "live_explorer",
            "source": "live world-model prediction-error improvement",
            "sota_anchor": "Curiosity-Critic arXiv:2604.18701",
        },
        "a2": {
            "lever": "cnn_action_effect_frame_change_predictor",
            "target": "live explorer candidate ranking",
            "source": "leaderboard-proven action-effect predictor",
        },
    }
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4627_blocks_without_fabricating_missing_capstone(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4627-BLOCKED-PRECONDITION: missing source blocks honestly."""

    _write_repo_fixture(tmp_path)
    (tmp_path / "results" / "experiment_4626_capstone_v426.json").unlink()

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=2.0,
        now_s=2.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_missing_experiment_4626_capstone_v426"
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["preconditions_checked"]["capstone_4626"]["available"] is False
    assert artifact["close_state_426"] == {}
    assert artifact["v427_pivot"] == {}
    assert (tmp_path / mod.OUTPUT_REL_PATH).exists()
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4627_precondition_order_and_blockers(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4627: blocker classification is explicit."""

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
    next_present["research_roadmap_next_yaml"]["milestone"] = "2026.06.427"
    assert mod._transition(next_present, complete=True)["activation_state"] == "activated_from_research_roadmap_next"

    active_bad = copy.deepcopy(checks)
    active_bad["active_research_roadmap_yaml"]["milestone"] = "2026.06.426"
    assert mod._first_blocker(active_bad) == "research_roadmap_427_unavailable"

    offline_bad = copy.deepcopy(checks)
    offline_bad["offline_arcade"]["available"] = False
    assert mod._first_blocker(offline_bad) == "offline_arcade"

    smart_bad = copy.deepcopy(checks)
    smart_bad["smart_subset_pretest_gate"]["passed"] = False
    assert mod._first_blocker(smart_bad) == "smart_subset_pretest_gate"

    registry_bad = copy.deepcopy(checks)
    registry_bad["registry"]["available"] = False
    assert mod._first_blocker(registry_bad) == "arc_solve_registry"

    a2_bad = copy.deepcopy(checks)
    a2_bad["a2_exp4617"]["available"] = False
    assert mod._first_blocker(a2_bad) == "missing_experiment_4617_graduate_spatial_value_head_live"

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


def test_scenario_capstone_4627_field_principle_validation_fails_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4627-FIELD-PRINCIPLES: schema drift fails loudly."""

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
    blocked["close_state_426"] = {"fabricated": True}
    blocked["reproducibility_checksum"] = "sha256:" + mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    inactive = copy.deepcopy(valid)
    inactive["transition"]["active_milestone_confirmed"] = False
    with pytest.raises(ValueError, match="active .427"):
        mod.validate_artifact(inactive)

    wrong_capability = copy.deepcopy(valid)
    wrong_capability["close_state_426"]["capability"]["reproducible_total_levels_after"] = 56
    with pytest.raises(ValueError, match="capability flat"):
        mod.validate_artifact(wrong_capability)

    wrong_a1 = copy.deepcopy(valid)
    wrong_a1["close_state_426"]["a1_bridge_disambiguation"]["binding_bridge_cause"] = "calibration"
    with pytest.raises(ValueError, match="A1"):
        mod.validate_artifact(wrong_a1)

    wrong_a2 = copy.deepcopy(valid)
    wrong_a2["close_state_426"]["a2_spatial_value_head_live"]["first_win_delta"] = 0.1
    with pytest.raises(ValueError, match="A2"):
        mod.validate_artifact(wrong_a2)

    wrong_a3 = copy.deepcopy(valid)
    wrong_a3["close_state_426"]["a3_levelup_selfplay"]["new_levels_banked"] = 1
    with pytest.raises(ValueError, match="A3"):
        mod.validate_artifact(wrong_a3)

    wrong_a4 = copy.deepcopy(valid)
    wrong_a4["close_state_426"]["a4_package"]["live_submittable_level_count"] = 33
    with pytest.raises(ValueError, match="A4"):
        mod.validate_artifact(wrong_a4)

    wrong_pivot = copy.deepcopy(valid)
    wrong_pivot["v427_pivot"]["headline_rationale"] = "rerank harder"
    with pytest.raises(ValueError, match="v427 pivot"):
        mod.validate_artifact(wrong_pivot)

    bad_checksum_prefix = copy.deepcopy(valid)
    bad_checksum_prefix["reproducibility_checksum"] = "not-a-checksum"
    with pytest.raises(ValueError, match="sha256-prefixed"):
        mod.validate_artifact(bad_checksum_prefix)

    bad_checksum = copy.deepcopy(valid)
    bad_checksum["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad_checksum)

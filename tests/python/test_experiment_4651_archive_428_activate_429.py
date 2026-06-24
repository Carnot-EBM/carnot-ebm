"""Tests for Exp 4651 `.428` archive / `.429` activation.

Spec refs: REQ-CAPSTONE-4651, SCENARIO-CAPSTONE-4651,
SCENARIO-CAPSTONE-4651-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4651-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4651_archive_428_activate_429 as mod


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


def _capstone_4650() -> JsonDict:
    return {
        "honest_verdict": "complete: capability_grew_56_to_57",
        "uniform_energy_ablation_passed": False,
        "live_submittable_level_count": 57,
        "ready_for_operator_submit": True,
        "reproducible_total_levels": 57,
        "reproducible_total_levels_delta": 1,
        "live_solve_rate_delta": {
            "reason": "uniform_energy_ablation_failed",
            "live_solve_rate_goal_energy": 0.04,
            "live_solve_rate_baseline": 0.04,
            "solve_rate_delta": 0.0,
            "first_win_rate_delta": 0.0,
            "headline_numbers_aggregated": False,
            "uniform_energy_ablation_passed": False,
        },
        "live_multi_level_solve_rate": {
            "reason": "zero_or_inadmissible_multi_level_lift",
            "live_multi_level_solve_rate": 0.0,
            "ranker_baseline_multi_level_rate": 0.0,
            "delta_vs_ranker_baseline": 0.0,
            "depth_of_live_solve_delta": 0.0,
            "first_win_rate_expansion": 1.0,
            "first_win_regressed_vs_427_baseline": False,
        },
        "first_win_rate_scored": {
            "clean_value": 0.590909,
            "coheadline_rate": 0.590909,
            "delta_vs_427_baseline": 0.0,
            "regressed_vs_427_baseline": False,
            "v427_baseline": 0.590909,
        },
        "scorecard": {
            "A1": {"included_in_headline": False, "reason": "uniform_energy_ablation_failed"},
            "A2": {"included_in_headline": True, "reason": "included_clean"},
            "A3": {
                "banked_plus_one": True,
                "registry_reproducible_total_levels": 57,
                "registry_delta_vs_56": 1,
                "reproduced_levels": 1,
                "offline_reproduced": True,
            },
            "A4": {
                "live_submittable_level_count": 57,
                "ready_for_operator_submit": True,
                "levels_folded_in": ["ft09"],
            },
            "A5": {"included_in_headline": True},
            "A6": {
                "included_in_headline": True,
                "live_multi_level_solve_rate_integrated": 0.0,
                "live_submittable_level_count_integrated": 57,
            },
            "B1": {"included_in_headline": True},
            "B2": {"goal_energy_ablation_guard_active": True, "included_in_headline": True},
            "headline": {
                "bridge_extended_by_energy_driven_generation": False,
                "a3_bank_plus_one": True,
                "a4_operator_resubmit_ready_above_33": True,
                "crossing_source": "none",
            },
        },
        "flagged_artifacts_handled": {
            "excluded_artifacts": ["results/experiment_4640_goal_energy_generation_live.json"],
            "excluded_details": [
                {
                    "artifact": "results/experiment_4640_goal_energy_generation_live.json",
                    "name": "A1",
                    "reason": "uniform_energy_ablation_failed",
                }
            ],
        },
    }


def _a1_goal_energy() -> JsonDict:
    return {
        "honest_verdict": "complete: goal_energy_no_live_lift_honest_null_gap_sharpened",
        "uniform_energy_ablation_passed": False,
        "live_solve_rate_goal_energy": 0.04,
        "live_solve_rate_baseline": 0.04,
        "solve_rate_delta": 0.0,
        "first_win_rate_delta": 0.0,
    }


def _a2_expansion_prior() -> JsonDict:
    return {
        "honest_verdict": "complete: action_effect_expansion_prior_no_deeper_solve_honest_null_gap_sharpened",
        "depth_of_live_solve_delta": 0.0,
        "first_win_rate_delta": 0.0,
        "live_multi_level_solve_rate": 0.0,
        "live_measurement": {
            "expansion_prior": {
                "first_win_rate": 1.0,
                "live_solve_rate": 0.0,
                "depth_of_live_solve": 1.0,
            },
            "ranker_baseline": {
                "first_win_rate": 1.0,
                "live_solve_rate": 0.0,
                "depth_of_live_solve": 1.0,
            },
        },
    }


def _a3_levelup() -> JsonDict:
    return {
        "honest_verdict": "success: ft09_L3_offline_reproduced",
        "target_game": "ft09",
        "prior_reproduced_level": 2,
        "reached_level": 3,
        "reproducible_total_levels_before": 56,
        "reproducible_total_levels_after": 57,
        "reproduced_levels": 1,
        "offline_reproduced": True,
    }


def _a4_package() -> JsonDict:
    return {
        "honest_verdict": "success: package_refreshed_live_submittable_57_above_33",
        "live_submittable_count_prev": 56,
        "live_submittable_level_count": 57,
        "count_delta": 1,
        "levels_folded_in": ["ft09"],
        "ready_for_operator_submit": True,
        "offline_reproduced": True,
    }


def _write_repo_fixture(root: Path, *, next_present: bool = True) -> None:
    (root / "research-roadmap.yaml").write_text(
        "milestone: 2026.06.428\n"
        "tasks:\n"
        "  - id: exp4650-capstone\n"
        "    deliverable: results/experiment_4650_capstone_v428.json\n",
        encoding="utf-8",
    )
    if next_present:
        (root / "research-roadmap-next.yaml").write_text(
            "milestone: 2026.06.429\n"
            "tasks:\n"
            "  - id: exp4651-phase0\n"
            "    deliverable: results/experiment_4651_archive_428_activate_429.json\n",
            encoding="utf-8",
        )
    (root / "research-complete.yaml").write_text(
        "milestones:\n"
        "- id: 2026.06.428\n"
        "  finding: prior roadmap archived by conductor\n",
        encoding="utf-8",
    )
    registry = root / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        "schema_version: 1\n"
        "updated: '2026-06-24'\n"
        "reproducible_total_levels: 57\n",
        encoding="utf-8",
    )
    proposal = root / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md"
    proposal.parent.mkdir(parents=True, exist_ok=True)
    proposal.write_text("Milestone 2026.06.429 GENERATION GUIDANCE\n", encoding="utf-8")
    _write_json(root / "results" / "experiment_4650_capstone_v428.json", _capstone_4650())
    _write_json(root / "results" / "experiment_4640_goal_energy_generation_live.json", _a1_goal_energy())
    _write_json(
        root / "results" / "experiment_4641_action_effect_expansion_prior_live.json",
        _a2_expansion_prior(),
    )
    _write_json(root / "results" / "experiment_4642_levelup_selfplay.json", _a3_levelup())
    _write_json(root / "results" / "experiment_4643_refresh_submission_package.json", _a4_package())
    _write_json(
        root / "results" / "experiment_4644_primitive_persist_transfer.json",
        {"honest_verdict": "complete: primitive_persisted_transfer_null_characterized"},
    )
    _write_json(
        root / "results" / "experiment_4645_integration_gate.json",
        {
            "honest_verdict": "success: integrated_live_submittable_raised_config_shipped",
            "live_multi_level_solve_rate_integrated": 0.0,
            "live_submittable_level_count_integrated": 57,
        },
    )
    _write_json(
        root / "results" / "experiment_4646_live_multi_level_solve_rate_metric.json",
        {
            "honest_verdict": "success: live_multi_level_solve_rate_metric_helper_shipped_tests_green",
            "live_multi_level_solve_rate": 0.0,
        },
    )
    _write_json(
        root / "results" / "experiment_4647_adversarial_verify_hardening.json",
        {
            "honest_verdict": "success: adversarial_verify_hardened_goal_energy_ablation_guard_tests_green.",
            "goal_energy_ablation_guard_added": True,
        },
    )


def _artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4651_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4651: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4651" in spec
    assert "SCENARIO-CAPSTONE-4651" in spec
    assert "SCENARIO-CAPSTONE-4651-BLOCKED-PRECONDITION" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "GENERATION GUIDANCE" in spec
    assert "winner_generated" in spec
    for field, provenance in mod.FIELD_PROVENANCE.items():
        assert field in spec
        assert provenance["principle"] in spec


def test_scenario_capstone_4651_records_true_close_state_and_activates_next(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4651: present next-roadmap activates `.429` and records `.428`."""

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
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8").startswith(
        "milestone: 2026.06.429"
    )
    assert artifact["honest_verdict"] == "complete: archive_428_activate_429_true_close_state_recorded"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.428",
        "activated_milestone": "2026.06.429",
        "active_milestone_confirmed": True,
        "activation_state": "activated_from_research_roadmap_next",
        "archive_state": "research_complete_contains_2026.06.428",
    }
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["available"] is True
    assert artifact["preconditions_checked"]["active_research_roadmap_yaml"]["milestone"] == "2026.06.429"
    assert artifact["preconditions_checked"]["offline_arcade"]["available"] is True
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is True

    close = artifact["close_state_428"]
    assert close["source_capstone_honest_verdict"] == "complete: capability_grew_56_to_57"
    assert close["a3_level_bank_ft09"] == {
        "honest_verdict": "success: ft09_L3_offline_reproduced",
        "target_game": "ft09",
        "prior_reproduced_level": 2,
        "target_level": 3,
        "reproducible_total_before": 56,
        "reproducible_total_after": 57,
        "reproducible_total_delta": 1,
        "offline_reproduced": True,
    }
    assert close["a1_goal_energy_generation"] == {
        "honest_verdict": "complete: goal_energy_no_live_lift_honest_null_gap_sharpened",
        "included_in_headline": False,
        "null_reason": "uniform_energy_ablation_failed",
        "uniform_energy_ablation_passed": False,
        "live_solve_rate_goal_energy": 0.04,
        "live_solve_rate_baseline": 0.04,
        "solve_rate_delta": 0.0,
        "first_win_rate_delta": 0.0,
    }
    assert close["a2_action_effect_expansion_prior"] == {
        "honest_verdict": "complete: action_effect_expansion_prior_no_deeper_solve_honest_null_gap_sharpened",
        "null_reason": "no_deeper_solve",
        "depth_of_live_solve_delta": 0.0,
        "live_multi_level_solve_rate": 0.0,
        "ranker_baseline_multi_level_rate": 0.0,
        "first_win_rate_expansion": 1.0,
        "first_win_held_at_or_above_427": True,
    }
    assert close["a4_submission_package"] == {
        "honest_verdict": "success: package_refreshed_live_submittable_57_above_33",
        "live_submittable_level_count": 57,
        "beats_submission_baseline": 33,
        "ready_for_operator_submit": True,
    }
    assert close["a5_a6_b1_b2_shipped"] == {
        "a5_honest_verdict": "complete: primitive_persisted_transfer_null_characterized",
        "a6_honest_verdict": "success: integrated_live_submittable_raised_config_shipped",
        "b1_honest_verdict": "success: live_multi_level_solve_rate_metric_helper_shipped_tests_green",
        "b2_honest_verdict": "success: adversarial_verify_hardened_goal_energy_ablation_guard_tests_green.",
        "goal_energy_ablation_guard_active": True,
    }
    assert close["registry_total_levels"] == 57
    assert close["energy_generation_lesson"] == "generation_guidance_needed_after_energy_levers_nulled"

    assert artifact["v429_pivot"] == {
        "headline_rationale": "GENERATION GUIDANCE",
        "builds_on": "complete: capability_grew_56_to_57",
        "a1": {
            "lever": "productionize_compute_cost_value_routing_fix",
            "fix": "scipy.ndimage.label_connected_components",
            "timing_before_ms": 13.0,
            "timing_after_ms": 0.64,
            "identical_output": True,
            "auroc": 0.725,
            "value_weight_target": "raise_off_0.0",
            "purpose": "discriminator_guides_live_without_timeout",
        },
        "a2": {
            "lever": "energy_as_fitness_qd_evolution",
            "operator_menu": "#2",
            "role": "next_sequenced_generation_lever",
            "gate": "winner_generated",
        },
    }
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4651_blocks_on_missing_literal_next_without_fabrication(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4651-BLOCKED-PRECONDITION: missing next roadmap blocks."""

    _write_repo_fixture(tmp_path, next_present=False)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=2.0,
        now_s=2.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_next_yaml"
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["available"] is False
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is None
    assert artifact["close_state_428"] == {}
    assert artifact["v429_pivot"] == {}
    assert (tmp_path / mod.OUTPUT_REL_PATH).exists()
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4651_precondition_order_and_blockers(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4651: blocker classification is explicit."""

    checks = _artifact(tmp_path)["preconditions_checked"]
    assert mod._first_blocker(checks) is None
    assert mod._command_check(None)["not_run_reason"] == "blocked_before_smart_subset_gate"

    next_missing = copy.deepcopy(checks)
    next_missing["research_roadmap_next_yaml"]["available"] = False
    assert mod._first_blocker(next_missing) == "research_roadmap_next_yaml"

    next_bad = copy.deepcopy(checks)
    next_bad["research_roadmap_next_yaml"]["parses"] = False
    assert mod._first_blocker(next_bad) == "research_roadmap_next_yaml"

    active_bad = copy.deepcopy(checks)
    active_bad["active_research_roadmap_yaml"]["milestone"] = "2026.06.428"
    assert mod._first_blocker(active_bad) == "research_roadmap_429_unavailable"

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
    capstone_bad["capstone_4650"]["available"] = False
    assert mod._first_blocker(capstone_bad) == "missing_experiment_4650_capstone_v428"

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


def test_scenario_capstone_4651_field_principle_validation_fails_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4651-FIELD-PRINCIPLES: schema drift fails loudly."""

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
    blocked["close_state_428"] = {"fabricated": True}
    blocked["reproducibility_checksum"] = "sha256:" + mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    inactive = copy.deepcopy(valid)
    inactive["transition"]["active_milestone_confirmed"] = False
    with pytest.raises(ValueError, match="active .429"):
        mod.validate_artifact(inactive)

    wrong_a3 = copy.deepcopy(valid)
    wrong_a3["close_state_428"]["a3_level_bank_ft09"]["reproducible_total_after"] = 56
    with pytest.raises(ValueError, match="A3"):
        mod.validate_artifact(wrong_a3)

    wrong_a1 = copy.deepcopy(valid)
    wrong_a1["close_state_428"]["a1_goal_energy_generation"]["uniform_energy_ablation_passed"] = True
    with pytest.raises(ValueError, match="A1"):
        mod.validate_artifact(wrong_a1)

    wrong_a2 = copy.deepcopy(valid)
    wrong_a2["close_state_428"]["a2_action_effect_expansion_prior"]["depth_of_live_solve_delta"] = 1.0
    with pytest.raises(ValueError, match="A2"):
        mod.validate_artifact(wrong_a2)

    wrong_a4 = copy.deepcopy(valid)
    wrong_a4["close_state_428"]["a4_submission_package"]["live_submittable_level_count"] = 33
    with pytest.raises(ValueError, match="A4"):
        mod.validate_artifact(wrong_a4)

    wrong_a5 = copy.deepcopy(valid)
    wrong_a5["close_state_428"]["a5_a6_b1_b2_shipped"]["goal_energy_ablation_guard_active"] = False
    with pytest.raises(ValueError, match="A5/A6/B1/B2"):
        mod.validate_artifact(wrong_a5)

    wrong_pivot = copy.deepcopy(valid)
    wrong_pivot["v429_pivot"]["headline_rationale"] = "rerank harder"
    with pytest.raises(ValueError, match="v429 pivot"):
        mod.validate_artifact(wrong_pivot)

    bad_checksum_prefix = copy.deepcopy(valid)
    bad_checksum_prefix["reproducibility_checksum"] = "not-a-checksum"
    with pytest.raises(ValueError, match="sha256-prefixed"):
        mod.validate_artifact(bad_checksum_prefix)

    bad_checksum = copy.deepcopy(valid)
    bad_checksum["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad_checksum)

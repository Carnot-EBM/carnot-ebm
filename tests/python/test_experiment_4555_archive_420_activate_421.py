"""Tests for Exp 4555 `.420` archive / `.421` activation.

Spec refs: REQ-CAPSTONE-4555, SCENARIO-CAPSTONE-4555,
SCENARIO-CAPSTONE-4555-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4555_archive_420_activate_421 as mod


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


def _capstone() -> JsonDict:
    return {
        "honest_verdict": "complete: llm_proposer_null_efficiency_unmoved_barrier_refined",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "efficiency_moved": False,
        "llm_proposer_value_summary": {
            "core_efficiency_baseline": 2.0074,
            "core_efficiency_best": None,
            "diagnosis": {
                "barrier_refinement": (
                    "positive_control_failed: live Qwen proposer did not produce the "
                    "known reachable fixture plan."
                )
            },
            "headline_numbers_aggregated": False,
            "moved": False,
            "positive_control_passed": None,
            "status": "diagnosis_only_null_delta_carve_out",
            "value": {"count": None, "events": [], "opportunities": None, "rate": None},
        },
        "cross_game_discrimination_above_chance": {
            "above_chance": True,
            "chance_auroc": 0.5,
            "in_sample_auroc": 0.8710834214701216,
            "loo_auroc_ci": [0.6058303817975523, 0.7451888709482918],
            "loo_auroc_mean": 0.6744657162333668,
            "loo_ci_excludes_chance": True,
            "positive_control_passed": True,
            "status": "clean_cross_game_discrimination_above_chance",
            "verifier_is_oracle": False,
        },
        "action_efficiency_improved": {
            "improved": False,
            "median_actions_blind": 1.0,
            "median_actions_cnn": 1.0,
            "median_actions_delta": 0.0,
            "positive_control_passed": True,
            "solve_rate_preserved": True,
            "status": "clean_action_efficiency_null",
        },
        "reproducible_total_levels_delta": {
            "prior_total": 51,
            "current_total": 52,
            "delta": 1,
            "banked_levels": 1,
            "capability_grew": True,
        },
        "generic_transfer_rate_over_variants": 0.04,
        "scorecard": {
            "a1_llm_proposer": {
                "core_efficiency_baseline": 2.0074,
                "core_efficiency_best": None,
                "headline_numbers_aggregated": False,
                "moved": False,
                "positive_control_passed": None,
                "status": "diagnosis_only_null_delta_carve_out",
                "value": {"count": None, "events": [], "opportunities": None, "rate": None},
            },
            "a2_cross_game_discrimination": {
                "above_chance": True,
                "loo_auroc_ci": [0.6058303817975523, 0.7451888709482918],
                "loo_auroc_mean": 0.6744657162333668,
                "loo_ci_excludes_chance": True,
                "verifier_is_oracle": False,
            },
            "a3_levelup": {
                "status": "level_up_banked",
                "honest_verdict": "success: su15_L2_offline_reproduced",
                "target_game": "su15",
                "target_level": 2,
                "banked_levels": 1,
                "level_up_banked": True,
                "current_total": 52,
            },
            "a4_frame_change_predictor": {
                "improved": False,
                "median_actions_blind": 1.0,
                "median_actions_cnn": 1.0,
                "median_actions_delta": 0.0,
                "solve_rate_preserved": True,
                "status": "clean_action_efficiency_null",
            },
            "b1_honest_sprint_metric": {
                "generic_transfer_rate_over_variants": 0.04,
                "reproducible_total_levels": 52,
                "variant_attempts_count": 25,
                "variant_solved_count": 1,
            },
            "baseline_core_efficiency": 2.0074,
        },
    }


def _a1_llm_proposer() -> JsonDict:
    return {
        "honest_verdict": "complete: llm_proposer_positive_control_failed_false_negative_risk_open",
        "positive_control_passed": False,
        "positive_control": {
            "dsl_reachable_plan": False,
            "passed": False,
            "reachable_plan": False,
            "source": "live_qwen_known_l2_fixture",
        },
        "llm_proposer_value": {"count": 0, "events": [], "opportunities": 1, "rate": 0.0},
        "core_efficiency_baseline": 2.0074,
        "core_efficiency_best": 2.0074,
        "efficiency_delta": 0.0,
        "barrier_refinement": (
            "positive_control_failed: live Qwen proposer did not produce the known reachable fixture plan."
        ),
        "flagged_adversarial": True,
        "null_delta_methodology_note": (
            "baseline==best because no lever reached a deeper offline-reproduced CORE level "
            "with CORE solves preserved; not a measurement bug."
        ),
    }


def _a2_cross_game() -> JsonDict:
    return {
        "honest_verdict": "success: cross_game_discrimination_loo_auroc_0.674_above_chance",
        "loo_auroc_mean": 0.6744657162333668,
        "loo_auroc_ci": [0.6058303817975523, 0.7451888709482918],
        "loo_ci_excludes_chance": True,
        "verifier_is_oracle": False,
        "positive_control_passed": True,
        "in_sample_auroc": 0.8710834214701216,
    }


def _b1_honest_metric() -> JsonDict:
    return {
        "honest_verdict": "shipped: honest_sprint_metric_variant_transfer_wired",
        "reproducible_total_levels": 52,
        "generic_transfer_rate_over_variants": 0.04,
        "variant_attempts_count": 25,
        "variant_solved_count": 1,
    }


def _write_repo_fixture(root: Path) -> None:
    (root / "research-roadmap.yaml").write_text(
        "milestone: 2026.06.421\n"
        "tasks:\n"
        "  - id: exp4555-phase0\n"
        "    deliverable: results/experiment_4555_archive_420_activate_421.json\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "milestones:\n"
        "- id: 2026.06.420\n"
        "  finding: prior roadmap archived by conductor\n",
        encoding="utf-8",
    )
    registry = root / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        "schema_version: 1\n"
        "updated: '2026-06-21'\n"
        "reproducible_total_levels: 52\n",
        encoding="utf-8",
    )
    _write_json(root / "results" / "experiment_4554_capstone_v420.json", _capstone())
    _write_json(
        root / "results" / "experiment_4544_llm_proposer_reinduction.json",
        _a1_llm_proposer(),
    )
    _write_json(
        root / "results" / "experiment_4545_cross_game_discrimination_v3.json",
        _a2_cross_game(),
    )
    _write_json(
        root / "results" / "experiment_4550_honest_sprint_metric.json",
        _b1_honest_metric(),
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


def test_req_capstone_4555_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4555: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4555" in spec
    assert "SCENARIO-CAPSTONE-4555" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "reproducible_total_levels=52" in spec
    assert "A3 banked `su15` L2" in spec
    for field, provenance in mod.FIELD_PROVENANCE.items():
        assert field in spec
        assert provenance["principle"] in spec


def test_scenario_capstone_4555_records_true_close_state_when_already_activated(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4555: already-activated `.421` still writes `.420` close-state."""

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
    assert artifact["honest_verdict"] == "complete: archive_420_activate_421_true_close_state_recorded"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.420",
        "activated_milestone": "2026.06.421",
        "active_milestone_confirmed": True,
        "activation_state": "already_active_roadmap_next_consumed",
        "archive_state": "research_complete_contains_2026.06.420",
    }
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["available"] is False
    assert artifact["preconditions_checked"]["active_research_roadmap_yaml"]["milestone"] == "2026.06.421"
    assert artifact["preconditions_checked"]["offline_arcade"]["available"] is True
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is True

    close = artifact["close_state_420"]
    assert close["reproducible_total_levels"] == 52
    assert close["efficiency_moved"] is False
    assert close["core_efficiency_baseline"] == 2.0074
    assert close["a1_llm_proposer"]["reinduction_null_streak"] == 3
    assert close["a1_llm_proposer"]["positive_control_passed"] is False
    assert close["a1_llm_proposer"]["llm_proposer_value_count"] == 0
    assert close["a1_llm_proposer"]["free_form_plans_reachable"] is False
    assert close["a2_cross_game_discrimination"]["won"] is True
    assert close["a2_cross_game_discrimination"]["loo_auroc_display"] == 0.674
    assert close["a2_cross_game_discrimination"]["ci_excludes_chance"] is True
    assert close["a2_cross_game_discrimination"]["verifier_is_oracle"] is False
    assert close["a3_levelup"]["target_game"] == "su15"
    assert close["a3_levelup"]["target_level"] == 2
    assert close["a3_levelup"]["banked"] is True
    assert close["a4_cnn_action_efficiency"]["improved"] is False
    assert close["a4_cnn_action_efficiency"]["median_actions_at_floor"] is True
    assert close["b1_honest_sprint_metric"]["generic_transfer_rate_over_variants"] == 0.04
    assert close["net_420"]["score_lever_to_build_next"] == "verifier_router_generic_transfer"
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4555_blocks_without_fabricating_missing_capstone(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4555: missing required close-state input blocks honestly."""

    _write_repo_fixture(tmp_path)
    (tmp_path / "results" / "experiment_4554_capstone_v420.json").unlink()

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=2.0,
        now_s=2.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_missing_experiment_4554_capstone_v420"
    assert artifact["preconditions_checked"]["capstone_4554"]["available"] is False
    assert artifact["close_state_420"] == {}
    assert artifact["transition"]["active_milestone_confirmed"] is False
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4555_records_next_roadmap_activation_state(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4555: an extant next roadmap is recorded as activation input."""

    _write_repo_fixture(tmp_path)
    (tmp_path / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.06.421\ntasks: []\n",
        encoding="utf-8",
    )

    artifact = mod.build_artifact(
        tmp_path,
        started_s=3.0,
        now_s=3.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["transition"]["activation_state"] == "activated_from_research_roadmap_next"
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["parses"] is True


def test_scenario_capstone_4555_precondition_blockers_are_classified(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4555: each required precondition has an honest blocked reason."""

    preconditions = _artifact(tmp_path)["preconditions_checked"]

    active_bad = copy.deepcopy(preconditions)
    active_bad["active_research_roadmap_yaml"]["milestone"] = "2026.06.420"
    active_bad["research_roadmap_next_yaml"]["available"] = False
    active_bad["research_roadmap_next_yaml"]["parses"] = False
    assert mod._first_blocker(active_bad) == "research_roadmap_421_unavailable"

    next_ok = copy.deepcopy(active_bad)
    next_ok["research_roadmap_next_yaml"]["parses"] = True
    next_ok["research_roadmap_next_yaml"]["milestone"] = "2026.06.421"
    assert mod._first_blocker(next_ok) is None

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
    capstone_bad["capstone_4554"]["available"] = False
    assert mod._first_blocker(capstone_bad) == "missing_experiment_4554_capstone_v420"

    a1_bad = copy.deepcopy(preconditions)
    a1_bad["a1_llm_proposer"]["available"] = False
    assert mod._first_blocker(a1_bad) == "missing_experiment_4544_llm_proposer_reinduction"

    a2_bad = copy.deepcopy(preconditions)
    a2_bad["a2_cross_game_discrimination"]["available"] = False
    assert mod._first_blocker(a2_bad) == "missing_experiment_4545_cross_game_discrimination_v3"

    b1_bad = copy.deepcopy(preconditions)
    b1_bad["b1_honest_sprint_metric"]["available"] = False
    assert mod._first_blocker(b1_bad) == "missing_experiment_4550_honest_sprint_metric"


def test_scenario_capstone_4555_parse_helpers_are_defensive(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4555: malformed inputs are detected instead of fabricated."""

    assert mod._list(None) == []
    assert mod._float(True, 7.0) == 7.0
    assert mod._float("bad", 9.0) == 9.0
    assert mod._int(False, 2) == 2
    assert mod._int("bad", 3) == 3
    assert mod._registry_total_levels(tmp_path / "missing.yaml") is None
    assert mod._display_auroc(0.6744657162333668) == 0.674

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("milestone: [\n", encoding="utf-8")
    assert mod._yaml_info(bad_yaml)["parses"] is False

    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- not-a-map\n", encoding="utf-8")
    assert mod._registry_total_levels(list_yaml) is None

    list_json = tmp_path / "list.json"
    list_json.write_text("[1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod._read_json(list_json)


def test_scenario_capstone_4555_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4555-FIELD-PRINCIPLES: schema drift fails loudly."""

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
    blocked["close_state_420"] = {"fabricated": True}
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    inactive = copy.deepcopy(valid)
    inactive["transition"]["active_milestone_confirmed"] = False
    with pytest.raises(ValueError, match="active .421"):
        mod.validate_artifact(inactive)

    wrong_total = copy.deepcopy(valid)
    wrong_total["close_state_420"]["reproducible_total_levels"] = 51
    with pytest.raises(ValueError, match="true .420 close-state"):
        mod.validate_artifact(wrong_total)

    wrong_a1 = copy.deepcopy(valid)
    wrong_a1["close_state_420"]["a1_llm_proposer"]["llm_proposer_value_count"] = 1
    with pytest.raises(ValueError, match="A1 LLM proposer null"):
        mod.validate_artifact(wrong_a1)

    wrong_a2 = copy.deepcopy(valid)
    wrong_a2["close_state_420"]["a2_cross_game_discrimination"]["ci_excludes_chance"] = False
    with pytest.raises(ValueError, match="A2 cross-game verifier win"):
        mod.validate_artifact(wrong_a2)

    wrong_a3 = copy.deepcopy(valid)
    wrong_a3["close_state_420"]["a3_levelup"]["target_game"] = "sp80"
    with pytest.raises(ValueError, match="A3 su15 L2"):
        mod.validate_artifact(wrong_a3)

    wrong_a4 = copy.deepcopy(valid)
    wrong_a4["close_state_420"]["a4_cnn_action_efficiency"]["improved"] = True
    with pytest.raises(ValueError, match="A4 CNN action-efficiency null"):
        mod.validate_artifact(wrong_a4)

    wrong_b1 = copy.deepcopy(valid)
    wrong_b1["close_state_420"]["b1_honest_sprint_metric"][
        "generic_transfer_rate_over_variants"
    ] = 0.05
    with pytest.raises(ValueError, match="B1 generic transfer ceiling"):
        mod.validate_artifact(wrong_b1)

    bad_checksum_prefix = copy.deepcopy(valid)
    bad_checksum_prefix["reproducibility_checksum"] = "not-a-checksum"
    with pytest.raises(ValueError, match="sha256-prefixed"):
        mod.validate_artifact(bad_checksum_prefix)

    bad_checksum_value = copy.deepcopy(valid)
    bad_checksum_value["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad_checksum_value)

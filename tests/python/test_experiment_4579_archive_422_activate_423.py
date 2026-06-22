"""Tests for Exp 4579 `.422` archive / `.423` activation.

Spec refs: REQ-CAPSTONE-4579, SCENARIO-CAPSTONE-4579,
SCENARIO-CAPSTONE-4579-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4579_archive_422_activate_423 as mod


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
        "honest_verdict": "complete: action_efficiency_null_gaps_sharpened",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "action_efficiency_moved": {
            "actions_delta": 0.0,
            "actions_delta_ci": [-0.0186799502, 0.0062266501],
            "moved": False,
            "positive_control_passed": True,
            "false_negative_risk_checked": True,
            "status": "clean_action_efficiency_null",
        },
        "generic_transfer_moved": {
            "baseline": 0.04,
            "coheadline_rate": 0.04,
            "moved": False,
            "reason": "no_clean_verifier_guided_expansion_transfer_above_0.04",
            "status": "false_negative_risk_open",
        },
        "winner_generated_root_cause_addressed": {
            "addressed": False,
            "evidence_status": "false_negative_risk_open",
            "diagnosis_read_as_broken_test_signal": True,
            "prior_root_cause": "winner_not_in_pool",
            "winner_generated_with_expansion": False,
        },
        "reproducible_total_levels_delta": {
            "prior_total": 52,
            "current_total": 53,
            "delta": 1,
            "a3_new_levels_banked": 1,
            "a4_new_levels_banked": 0,
            "capability_grew": True,
        },
        "ready_for_operator_submit": True,
        "scorecard": {
            "a1_clickability_predictor": {
                "actions_delta": 0.0,
                "actions_delta_ci": [-0.0186799502, 0.0062266501],
                "false_negative_risk_checked": True,
                "positive_control_passed": True,
                "status": "clean_action_efficiency_null",
            },
            "a2_verifier_guided_expansion": {
                "false_negative_risk_checked": False,
                "generic_transfer_rate_baseline": 0.04,
                "generic_transfer_rate_with_expansion": None,
                "moved": False,
                "random_priority_control_passed": False,
                "status": "false_negative_risk_open",
                "transfer_delta": None,
                "verifier_is_oracle": None,
            },
            "a3_levelup_attempt": {
                "banked_levels": 1,
                "offline_reproduced": True,
                "status": "level_banked",
                "target_game": "cn04",
                "target_level": 2,
            },
            "a4_hidden_state_probe_ka59": {
                "banked_levels": 0,
                "offline_reproduced": False,
                "status": "no_new_level_banked",
                "target_game": "",
                "target_level": None,
            },
            "a5_integration": {
                "integrated_metric_improved": False,
                "ready_for_operator_submit": False,
                "status": "false_negative_risk_open",
            },
            "a6_transfer": {
                "any_transfer_value_added": True,
                "new_levels_banked": 0,
                "offline_reproduced_new_level": False,
                "primitive_persisted": {"operator": "persistent_action_effect_memory_operator"},
                "status": "transfer_value_added",
                "transfer_games": ["dc22", "m0r0", "ka59"],
                "transfer_value_per_game": {
                    "m0r0": {
                        "actions_reduced": 1.0,
                        "candidate_group_count": 53,
                        "target_candidate_generated": True,
                        "value_added": True,
                        "winner_generated": False,
                    }
                },
            },
            "b1_action_efficiency_coheadline": {
                "action_efficiency_ci": [1.0, 1.0],
                "action_efficiency_score": 1.0,
                "generic_transfer_ci": [0.0, 0.1],
                "generic_transfer_rate_over_variants": 0.04,
                "reproducible_total_levels": 53,
                "status": "clean_action_efficiency_coheadline",
            },
            "last_submitted_levels": 33,
        },
    }


def _a2_expansion() -> JsonDict:
    return {
        "honest_verdict": "complete: verifier_guided_expansion_no_value_honest_null_generation_gap_sharpened",
        "transfer_delta": -0.04,
        "positive_control_passed": None,
        "false_negative_risk_checked": False,
        "random_priority_control_passed": False,
        "winner_generated": False,
    }


def _a4_hidden_state() -> JsonDict:
    return {
        "honest_verdict": "complete: hidden_state_probe_no_new_level_banked",
        "target_game": "ka59",
        "target_level": 2,
        "new_levels_banked": 0,
        "state_disambiguation_control_passed": True,
        "false_negative_risk_checked": True,
    }


def _a5_integration() -> JsonDict:
    return {
        "honest_verdict": "complete: no_lever_raises_a_metric_honest_null",
        "heldout_solve_rate": 0.04,
        "baseline_heldout_solve_rate": 0.04,
        "additivity_checked": {
            "isolated_deltas": {
                "A2_verifier_guided_expansion": {"generic_transfer_delta": -0.04}
            }
        },
    }


def _a6_transfer() -> JsonDict:
    return {
        "honest_verdict": "success: primitive_persisted_transfer_m0r0_value_added",
        "primitive_persisted": True,
        "new_levels_banked": 0,
        "transfer_results": [
            {
                "game": "m0r0",
                "value_added": True,
                "ordering_gain": 1,
                "offline_reproduced_new_level": False,
            }
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
            {"game": "cn04", "claimed": 1, "live_level": 1, "env_match": True},
        ],
    }


def _write_repo_fixture(root: Path, *, active_milestone: str = "2026.06.423") -> None:
    (root / "research-roadmap.yaml").write_text(
        f"milestone: {active_milestone}\n"
        "tasks:\n"
        "  - id: exp4579-phase0\n"
        "    deliverable: results/experiment_4579_archive_422_activate_423.json\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "milestones:\n"
        "- id: 2026.06.422\n"
        "  finding: prior roadmap archived by conductor\n",
        encoding="utf-8",
    )
    registry = root / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        "schema_version: 1\n"
        "updated: '2026-06-21'\n"
        "reproducible_total_levels: 53\n",
        encoding="utf-8",
    )
    _write_json(root / "results" / "experiment_4578_capstone_v422.json", _capstone())
    _write_json(root / "results" / "arc3_live_submit.json", _live_submit())
    _write_json(root / "results" / "experiment_4569_verifier_guided_expansion.json", _a2_expansion())
    _write_json(
        root / "results" / "experiment_4571_hidden_field_state_probe_ka59.json",
        _a4_hidden_state(),
    )
    _write_json(root / "results" / "experiment_4572_integration_gate.json", _a5_integration())
    _write_json(root / "results" / "experiment_4573_primitive_persist_transfer.json", _a6_transfer())


def _artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4579_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4579: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4579" in spec
    assert "SCENARIO-CAPSTONE-4579" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "53` reproducible levels versus `33` submitted levels" in spec
    for field, provenance in mod.FIELD_PROVENANCE.items():
        assert field in spec
        assert provenance["principle"] in spec


def test_scenario_capstone_4579_records_true_close_state_when_already_activated(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4579: consumed next-roadmap still writes .422 close-state."""

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
    assert artifact["honest_verdict"] == "complete: archive_422_activate_423_true_close_state_recorded"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.422",
        "activated_milestone": "2026.06.423",
        "active_milestone_confirmed": True,
        "activation_state": "already_active_roadmap_next_consumed",
        "archive_state": "research_complete_contains_2026.06.422",
    }
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["available"] is False
    assert artifact["preconditions_checked"]["active_research_roadmap_yaml"]["milestone"] == "2026.06.423"
    assert artifact["preconditions_checked"]["offline_arcade"]["available"] is True
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is True

    close = artifact["close_state_422"]
    assert close["source_capstone_honest_verdict"] == "complete: action_efficiency_null_gaps_sharpened"
    assert close["reproducible_total_levels_delta"] == {
        "prior_total": 52,
        "current_total": 53,
        "delta": 1,
        "a3_new_levels_banked": 1,
        "a4_new_levels_banked": 0,
        "capability_grew": True,
    }
    assert close["a1_clickability_ranker"]["actions_delta"] == 0.0
    assert close["a1_clickability_ranker"]["warn_no_efficiency_gain"] is True
    assert close["a2_verifier_guided_expansion"]["transfer_delta"] == -0.04
    assert close["a2_verifier_guided_expansion"]["positive_control_passed"] is None
    assert close["a2_verifier_guided_expansion"]["false_negative_risk_open"] is True
    assert close["a3_levelup_attempt"]["target_game"] == "cn04"
    assert close["a3_levelup_attempt"]["target_level"] == 2
    assert close["a3_levelup_attempt"]["new_levels_banked"] == 1
    assert close["a4_hidden_state_probe_ka59"]["target_game"] == "ka59"
    assert close["a4_hidden_state_probe_ka59"]["state_disambiguation_control_passed"] is True
    assert close["a5_integration"]["heldout_solve_rate"] == 0.04
    assert close["a5_integration"]["heldout_solve_rate_unchanged"] is True
    assert close["a6_primitive_persist_transfer"]["primitive_persisted"] is True
    assert close["a6_primitive_persist_transfer"]["m0r0_cached_pool_value_added"] is True
    assert close["generation_not_ranking_diagnosis"]["triply_confirmed"] is True

    gap = artifact["live_submission_gap"]
    assert gap["reproducible_total_levels"] == 53
    assert gap["live_total_levels"] == 33
    assert gap["gap_levels"] == 20
    assert gap["sc25_env_match"] is False
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4579_blocks_without_fabricating_when_423_not_active(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4579: missing activation evidence blocks honestly."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.422")

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=2.0,
        now_s=2.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_423_unavailable"
    assert artifact["preconditions_checked"]["active_research_roadmap_yaml"]["milestone"] == "2026.06.422"
    assert artifact["close_state_422"] == {}
    assert artifact["live_submission_gap"] == {}
    assert artifact["transition"]["active_milestone_confirmed"] is False
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4579_precondition_blockers_and_helpers_are_defensive(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4579: missing resources classify without fabricated data."""

    preconditions = _artifact(tmp_path)["preconditions_checked"]

    next_only = copy.deepcopy(preconditions)
    next_only["active_research_roadmap_yaml"]["milestone"] = "2026.06.422"
    next_only["research_roadmap_next_yaml"]["available"] = True
    next_only["research_roadmap_next_yaml"]["parses"] = True
    next_only["research_roadmap_next_yaml"]["milestone"] = "2026.06.423"
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
    capstone_bad["capstone_4578"]["available"] = False
    assert mod._first_blocker(capstone_bad) == "missing_experiment_4578_capstone_v422"

    live_bad = copy.deepcopy(preconditions)
    live_bad["arc3_live_submit"]["available"] = False
    assert mod._first_blocker(live_bad) == "missing_arc3_live_submit"

    a2_bad = copy.deepcopy(preconditions)
    a2_bad["a2_verifier_guided_expansion"]["available"] = False
    assert mod._first_blocker(a2_bad) == "missing_experiment_4569_verifier_guided_expansion"

    a4_bad = copy.deepcopy(preconditions)
    a4_bad["a4_hidden_state_probe_ka59"]["available"] = False
    assert mod._first_blocker(a4_bad) == "missing_experiment_4571_hidden_field_state_probe_ka59"

    a5_bad = copy.deepcopy(preconditions)
    a5_bad["a5_integration"]["available"] = False
    assert mod._first_blocker(a5_bad) == "missing_experiment_4572_integration_gate"

    a6_bad = copy.deepcopy(preconditions)
    a6_bad["a6_primitive_persist_transfer"]["available"] = False
    assert mod._first_blocker(a6_bad) == "missing_experiment_4573_primitive_persist_transfer"

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

    list_json = tmp_path / "list.json"
    list_json.write_text("[1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod._read_json(list_json)

    assert mod._a6_m0r0_value_added({}, {"transfer_results": [{"game": "m0r0", "value_added": True}]})
    assert not mod._a6_m0r0_value_added({}, {"transfer_results": [{"game": "dc22", "value_added": True}]})
    assert mod._primitive_persisted({"a6_transfer": {"primitive_persisted": {"operator": "x"}}}, {})
    assert mod._live_submission_gap(live_submit={"live_total_levels": 33, "per_game": []}, registry_total_levels=53)[
        "sc25_env_match"
    ] is None
    assert mod._live_submission_gap(
        live_submit={"live_total_levels": 33, "per_game": [{"game": "dc22"}]},
        registry_total_levels=53,
    )["sc25_env_match"] is None


def test_scenario_capstone_4579_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4579-FIELD-PRINCIPLES: schema drift fails loudly."""

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
    blocked["close_state_422"] = {"fabricated": True}
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    inactive = copy.deepcopy(valid)
    inactive["transition"]["active_milestone_confirmed"] = False
    with pytest.raises(ValueError, match="active .423"):
        mod.validate_artifact(inactive)

    wrong_total = copy.deepcopy(valid)
    wrong_total["close_state_422"]["reproducible_total_levels_delta"]["current_total"] = 52
    with pytest.raises(ValueError, match="true .422 registry delta"):
        mod.validate_artifact(wrong_total)

    wrong_a1 = copy.deepcopy(valid)
    wrong_a1["close_state_422"]["a1_clickability_ranker"]["actions_delta"] = 1.0
    with pytest.raises(ValueError, match="A1 ranker null"):
        mod.validate_artifact(wrong_a1)

    wrong_a2 = copy.deepcopy(valid)
    wrong_a2["close_state_422"]["a2_verifier_guided_expansion"]["positive_control_passed"] = True
    with pytest.raises(ValueError, match="A2 broken control"):
        mod.validate_artifact(wrong_a2)

    wrong_a3 = copy.deepcopy(valid)
    wrong_a3["close_state_422"]["a3_levelup_attempt"]["target_game"] = "sp80"
    with pytest.raises(ValueError, match="A3 cn04"):
        mod.validate_artifact(wrong_a3)

    wrong_a4 = copy.deepcopy(valid)
    wrong_a4["close_state_422"]["a4_hidden_state_probe_ka59"][
        "state_disambiguation_control_passed"
    ] = False
    with pytest.raises(ValueError, match="A4 ka59"):
        mod.validate_artifact(wrong_a4)

    wrong_a5 = copy.deepcopy(valid)
    wrong_a5["close_state_422"]["a5_integration"]["heldout_solve_rate"] = 0.08
    with pytest.raises(ValueError, match="A5 no-lever"):
        mod.validate_artifact(wrong_a5)

    wrong_a6 = copy.deepcopy(valid)
    wrong_a6["close_state_422"]["a6_primitive_persist_transfer"]["new_levels_banked"] = 1
    with pytest.raises(ValueError, match="A6 ordering-only"):
        mod.validate_artifact(wrong_a6)

    wrong_diagnosis = copy.deepcopy(valid)
    wrong_diagnosis["close_state_422"]["generation_not_ranking_diagnosis"][
        "triply_confirmed"
    ] = False
    with pytest.raises(ValueError, match="generation-not-ranking"):
        mod.validate_artifact(wrong_diagnosis)

    wrong_gap = copy.deepcopy(valid)
    wrong_gap["live_submission_gap"]["gap_levels"] = 0
    with pytest.raises(ValueError, match="live-submission gap"):
        mod.validate_artifact(wrong_gap)

    bad_checksum_prefix = copy.deepcopy(valid)
    bad_checksum_prefix["reproducibility_checksum"] = "not-a-checksum"
    with pytest.raises(ValueError, match="sha256-prefixed"):
        mod.validate_artifact(bad_checksum_prefix)

    bad_checksum = copy.deepcopy(valid)
    bad_checksum["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad_checksum)

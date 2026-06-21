"""Tests for Exp 4543 `.419` archive / `.420` activation.

Spec refs: REQ-CAPSTONE-4543, SCENARIO-CAPSTONE-4543,
SCENARIO-CAPSTONE-4543-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4543_archive_419_activate_420 as mod


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
        "honest_verdict": "complete: reinduction_null_efficiency_unmoved_barrier_refined",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "efficiency_moved": False,
        "reproducible_total_levels_delta": {
            "prior_total": 50,
            "current_total": 51,
            "delta": 1,
            "banked_levels": 1,
            "capability_grew": True,
        },
        "scorecard": {
            "a1_reinduction": {
                "status": "diagnosis_only_null_delta_carve_out",
                "moved": False,
                "core_efficiency_baseline": 2.0074,
                "core_efficiency_best": None,
                "diagnosis": {
                    "barrier_refinement": (
                        "post_level_reinduction_triggered_but_no_reachable_l2_plan; "
                        "offline_dsl_attempt_outcomes=['proposer_failed_or_missing_root']."
                    )
                },
            },
            "a2_energy_routing": {
                "status": "excluded_flagged_adversarial_or_live_critical",
                "generalized": False,
                "reason": "headline numbers quarantined",
            },
            "a3_levelup": {
                "status": "level_up_banked",
                "honest_verdict": "success: sp80_L2_offline_reproduced",
                "target_game": "sp80",
                "target_level": 2,
                "banked_levels": 1,
                "level_up_banked": True,
                "current_total": 51,
            },
            "a5_primitive_transfer": {
                "status": "representation_generalized_no_reproducible_level_bank",
                "representation_generalized": True,
                "new_levels_banked": 0,
                "offline_reproduced_new_level": False,
            },
            "baseline_core_efficiency": 2.0074,
        },
        "energy_routing_generalization": {
            "status": "excluded_flagged_adversarial_or_live_critical",
            "generalized": False,
            "reason": "headline numbers quarantined",
        },
        "primitive_transfer_generalization": {
            "representation_generalized": True,
            "new_levels_banked": 0,
            "offline_reproduced_new_level": False,
        },
    }


def _a1_reinduction() -> JsonDict:
    return {
        "honest_verdict": "complete: reinduction_no_deeper_level_barrier_refined_honest_null",
        "barrier_refinement": (
            "post_level_reinduction_triggered_but_no_reachable_l2_plan; "
            "offline_dsl_attempt_outcomes=['proposer_failed_or_missing_root']."
        ),
        "model_specs": "offline_dsl_induction_no_llm",
        "core_efficiency_baseline": 2.0074,
        "core_efficiency_best": 2.0074,
        "efficiency_delta": 0.0,
        "null_delta_methodology_note": (
            "baseline==best because no lever reached a deeper offline-reproduced CORE level "
            "with CORE solves preserved; this is an honest null, not a measurement bug."
        ),
        "deepest_level_reached_per_core_game": {
            "1": {"lp85": 1, "m0r0": 1, "sp80": 1, "vc33": 1}
        },
        "measurements": [
            {
                "per_game": [
                    {
                        "game": "m0r0",
                        "level_up_actions": [3891],
                        "diagnostics": {
                            "model_specs": "offline_dsl_induction_no_llm",
                            "induction_attempts": [
                                {
                                    "skipped": "proposer_failed_or_missing_root",
                                    "planned": False,
                                }
                            ],
                        },
                    }
                ]
            }
        ],
    }


def _write_repo_fixture(root: Path) -> None:
    (root / "research-roadmap.yaml").write_text(
        "milestone: 2026.06.420\n"
        "tasks:\n"
        "  - id: exp4543-phase0\n"
        "    deliverable: results/experiment_4543_archive_419_activate_420.json\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "milestones:\n"
        "- id: 2026.06.419\n"
        "  finding: prior roadmap archived by conductor\n",
        encoding="utf-8",
    )
    registry = root / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        "schema_version: 1\n"
        "updated: '2026-06-21'\n"
        "reproducible_total_levels: 51\n",
        encoding="utf-8",
    )
    _write_json(root / "results" / "experiment_4542_capstone_v419.json", _capstone())
    _write_json(
        root / "results" / "experiment_4533_per_level_goal_reinduction.json",
        _a1_reinduction(),
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


def test_req_capstone_4543_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4543: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4543" in spec
    assert "SCENARIO-CAPSTONE-4543" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "reproducible_total_levels=51" in spec
    assert "A3 banked `sp80` L2" in spec
    for field, provenance in mod.FIELD_PROVENANCE.items():
        assert field in spec
        assert provenance["principle"] in spec


def test_scenario_capstone_4543_records_true_close_state_when_already_activated(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4543: already-activated `.420` still writes the close-state."""

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
    assert artifact["honest_verdict"] == "complete: archive_419_activate_420_true_close_state_recorded"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.419",
        "activated_milestone": "2026.06.420",
        "active_milestone_confirmed": True,
        "activation_state": "already_active_roadmap_next_consumed",
        "archive_state": "research_complete_contains_2026.06.419",
    }
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["available"] is False
    assert artifact["preconditions_checked"]["active_research_roadmap_yaml"]["milestone"] == "2026.06.420"
    assert artifact["preconditions_checked"]["offline_arcade"]["available"] is True
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is True

    close = artifact["close_state_419"]
    assert close["reproducible_total_levels"] == 51
    assert close["efficiency_moved"] is False
    assert close["core_efficiency_baseline"] == 2.0074
    assert close["a1_reinduction"]["triggered_on_level_up"] is True
    assert close["a1_reinduction"]["proposer_is_bottleneck"] is True
    assert close["a1_reinduction"]["proposer_failure"] == "proposer_failed_or_missing_root"
    assert close["a1_reinduction"]["model_specs"] == "offline_dsl_induction_no_llm"
    assert close["a2_energy_routing"]["nulled_because_no_reachable_plan"] is True
    assert close["a3_levelup"]["target_game"] == "sp80"
    assert close["a3_levelup"]["target_level"] == 2
    assert close["a3_levelup"]["banked"] is True
    assert close["a5_primitive_transfer"]["representation_transferred"] is True
    assert close["a5_primitive_transfer"]["new_levels_banked"] == 0
    assert close["net_419"]["score_lever_to_build_next"] == "llm_proposer_reinduction"
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4543_blocks_without_fabricating_missing_capstone(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4543: missing required close-state input blocks honestly."""

    _write_repo_fixture(tmp_path)
    (tmp_path / "results" / "experiment_4542_capstone_v419.json").unlink()

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=2.0,
        now_s=2.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_missing_experiment_4542_capstone_v419"
    assert artifact["preconditions_checked"]["capstone_4542"]["available"] is False
    assert artifact["close_state_419"] == {}
    assert artifact["transition"]["active_milestone_confirmed"] is False
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4543_records_next_roadmap_activation_state(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4543: an extant next roadmap is recorded as activation input."""

    _write_repo_fixture(tmp_path)
    (tmp_path / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.06.420\ntasks: []\n",
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


def test_scenario_capstone_4543_precondition_blockers_are_classified(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4543: each required precondition has an honest blocked reason."""

    preconditions = _artifact(tmp_path)["preconditions_checked"]

    active_bad = copy.deepcopy(preconditions)
    active_bad["active_research_roadmap_yaml"]["milestone"] = "2026.06.419"
    active_bad["research_roadmap_next_yaml"]["available"] = False
    active_bad["research_roadmap_next_yaml"]["parses"] = False
    assert mod._first_blocker(active_bad) == "research_roadmap_420_unavailable"

    next_ok = copy.deepcopy(active_bad)
    next_ok["research_roadmap_next_yaml"]["parses"] = True
    next_ok["research_roadmap_next_yaml"]["milestone"] = "2026.06.420"
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
    capstone_bad["capstone_4542"]["available"] = False
    assert mod._first_blocker(capstone_bad) == "missing_experiment_4542_capstone_v419"

    a1_bad = copy.deepcopy(preconditions)
    a1_bad["a1_goal_reinduction"]["available"] = False
    assert mod._first_blocker(a1_bad) == "missing_experiment_4533_per_level_goal_reinduction"


def test_scenario_capstone_4543_parse_helpers_are_defensive(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4543: malformed inputs are detected instead of fabricated."""

    assert mod._list(None) == []
    assert mod._float(True, 7.0) == 7.0
    assert mod._float("bad", 9.0) == 9.0
    assert mod._int(False, 2) == 2
    assert mod._int("bad", 3) == 3
    assert mod._registry_total_levels(tmp_path / "missing.yaml") is None

    attempt_only = {
        "measurements": [
            {
                "per_game": [
                    {
                        "diagnostics": {
                            "induction_attempts": [{"skipped": "proposer_failed_or_missing_root"}]
                        }
                    }
                ]
            }
        ]
    }
    assert mod._first_proposer_failure(attempt_only) == "proposer_failed_or_missing_root"
    assert mod._reinduction_triggered({"measurements": [{"per_game": [{"level_up_actions": [1]}]}]})
    assert mod._first_proposer_failure({"measurements": [{"per_game": []}]}) == ""
    assert mod._reinduction_triggered({"measurements": [{"per_game": []}]}) is False

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


def test_scenario_capstone_4543_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4543-FIELD-PRINCIPLES: schema drift fails loudly."""

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
    blocked["close_state_419"] = {"fabricated": True}
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    inactive = copy.deepcopy(valid)
    inactive["transition"]["active_milestone_confirmed"] = False
    with pytest.raises(ValueError, match="active .420"):
        mod.validate_artifact(inactive)

    wrong_total = copy.deepcopy(valid)
    wrong_total["close_state_419"]["reproducible_total_levels"] = 50
    with pytest.raises(ValueError, match="true .419 close-state"):
        mod.validate_artifact(wrong_total)

    wrong_a1 = copy.deepcopy(valid)
    wrong_a1["close_state_419"]["a1_reinduction"]["proposer_failure"] = "other"
    with pytest.raises(ValueError, match="proposer bottleneck"):
        mod.validate_artifact(wrong_a1)

    wrong_a2 = copy.deepcopy(valid)
    wrong_a2["close_state_419"]["a2_energy_routing"]["nulled_because_no_reachable_plan"] = False
    with pytest.raises(ValueError, match="A2 energy-routing null"):
        mod.validate_artifact(wrong_a2)

    wrong_a3 = copy.deepcopy(valid)
    wrong_a3["close_state_419"]["a3_levelup"]["target_game"] = "lp85"
    with pytest.raises(ValueError, match="A3 sp80 L2"):
        mod.validate_artifact(wrong_a3)

    wrong_a5 = copy.deepcopy(valid)
    wrong_a5["close_state_419"]["a5_primitive_transfer"]["new_levels_banked"] = 1
    with pytest.raises(ValueError, match="A5 representation transfer"):
        mod.validate_artifact(wrong_a5)

    bad_checksum_prefix = copy.deepcopy(valid)
    bad_checksum_prefix["reproducibility_checksum"] = "not-a-checksum"
    with pytest.raises(ValueError, match="sha256-prefixed"):
        mod.validate_artifact(bad_checksum_prefix)

    bad_checksum_value = copy.deepcopy(valid)
    bad_checksum_value["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad_checksum_value)

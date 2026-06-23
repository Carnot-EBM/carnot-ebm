"""Tests for Exp 4615 `.425` archive / `.426` activation.

Spec refs: REQ-CAPSTONE-4615, SCENARIO-CAPSTONE-4615,
SCENARIO-CAPSTONE-4615-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4615-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4615_archive_425_activate_426 as mod


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


def _capstone_4614() -> JsonDict:
    return {
        "honest_verdict": "complete: pivot_characterized_capability_grew_55_to_55",
        "reproducible_total_levels": 55,
        "reproducible_total_levels_delta": 0,
        "live_submittable_level_count": 55,
        "ready_for_operator_submit": True,
        "world_model_trust_pass_rate": {
            "baseline": 0.0,
            "quarantined_value": 1.0,
            "delta": 1.0,
            "headline_numbers_aggregated": False,
            "trust_pass_numerator": 6,
            "trust_pass_denominator": 6,
            "binary_gate_failures": "0/6",
        },
        "first_win_rate_scored": {
            "bare_rate": 0.04,
            "quarantined_value": 0.04,
            "delta": 0.0,
            "headline_numbers_aggregated": False,
        },
        "scorecard": {
            "A1": {
                "artifact": "results/experiment_4604_world_model_trust_energy.json",
                "included_in_headline": False,
                "reason": "flagged_adversarial_or_live_critical_excluded",
            },
            "A2": {
                "artifact": "results/experiment_4605_live_integration_scored_agent.json",
                "included_in_headline": False,
                "reason": "flagged_adversarial_or_live_critical_excluded",
            },
            "A3": {
                "artifact": "results/experiment_4606_levelup_selfplay.json",
                "included_in_headline": True,
                "offline_reproduced": False,
                "registry_delta_vs_55": 0,
                "registry_reproducible_total_levels": 55,
                "reproduced_levels": 0,
            },
            "A4": {
                "artifact": "results/experiment_4607_refresh_submission_package.json",
                "count_delta": 0,
                "included_in_headline": True,
                "live_submittable_level_count": 55,
                "ready_for_operator_submit": True,
            },
        },
    }


def _a1_world_model() -> JsonDict:
    return {
        "honest_verdict": "success: world_model_trust_energy_pass_rate_up_6_first_win_up",
        "flagged_adversarial": True,
        "duration_s": 0.44,
        "world_model_trust_pass_rate_binary": 0.0,
        "world_model_trust_pass_rate_new": 1.0,
        "first_win_rate_binary": 0.0,
        "first_win_rate_new": 1.0,
        "first_win_delta": 1.0,
    }


def _a2_live_integration() -> JsonDict:
    return {
        "honest_verdict": "complete: live_integration_no_value_honest_null_gap_sharpened",
        "flagged_adversarial": True,
        "first_win_rate_integrated": 0.04,
        "first_win_rate_bare": 0.04,
        "first_win_delta": 0.0,
        "actions_delta": 0.0,
        "solve_rate_integrated": 0.04,
        "solve_rate_bare": 0.04,
    }


def _a3_levelup() -> JsonDict:
    return {
        "honest_verdict": "complete: dc22_delta_identified_no_bank",
        "target_game": "dc22",
        "reproduced_levels": 0,
        "offline_reproduced": False,
        "reproduction_gate": {
            "claimed_level": 2,
            "game": "dc22",
            "reached_level": 1,
            "reproduced": False,
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


def _write_repo_fixture(root: Path, *, include_next: bool = True) -> None:
    (root / "research-roadmap.yaml").write_text(
        "milestone: 2026.06.426\n"
        "tasks:\n"
        "  - id: exp4615-phase0\n"
        "    deliverable: results/experiment_4615_archive_425_activate_426.json\n",
        encoding="utf-8",
    )
    if include_next:
        (root / "research-roadmap-next.yaml").write_text(
            "milestone: 2026.06.426\n"
            "tasks:\n"
            "  - id: exp4616-a1\n",
            encoding="utf-8",
        )
    (root / "research-complete.yaml").write_text(
        "milestones:\n"
        "- id: 2026.06.425\n"
        "  finding: prior roadmap archived by conductor\n",
        encoding="utf-8",
    )
    registry = root / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        "schema_version: 1\n"
        "updated: '2026-06-22'\n"
        "reproducible_total_levels: 55\n",
        encoding="utf-8",
    )
    note = root / "docs" / "research-notes" / "arc-representation-not-the-bottleneck-2026-06-23.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("LOO-AUROC 0.725; OFFLINE -> LIVE bridge.\n", encoding="utf-8")
    proposal = root / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md"
    proposal.parent.mkdir(parents=True, exist_ok=True)
    proposal.write_text("Milestone 2026.06.426 OFFLINE->LIVE BRIDGE.\n", encoding="utf-8")
    _write_json(root / "results" / "experiment_4614_capstone_v425.json", _capstone_4614())
    _write_json(root / "results" / "experiment_4604_world_model_trust_energy.json", _a1_world_model())
    _write_json(root / "results" / "experiment_4605_live_integration_scored_agent.json", _a2_live_integration())
    _write_json(root / "results" / "experiment_4606_levelup_selfplay.json", _a3_levelup())
    _write_json(root / "results" / "experiment_4607_refresh_submission_package.json", _a4_package())


def _complete_artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4615_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4615: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4615" in spec
    assert "SCENARIO-CAPSTONE-4615" in spec
    assert "SCENARIO-CAPSTONE-4615-BLOCKED-PRECONDITION" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "blocked_research_roadmap_next_yaml" in spec
    assert "OFFLINE->LIVE BRIDGE" in spec
    for field, provenance in mod.FIELD_PROVENANCE.items():
        assert field in spec
        assert provenance["principle"] in spec


def test_scenario_capstone_4615_records_true_close_state_when_next_roadmap_exists(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4615: complete path records the honest .425 close-state."""

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
    assert artifact["honest_verdict"] == "complete: archive_425_activate_426_true_close_state_recorded"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.425",
        "activated_milestone": "2026.06.426",
        "active_milestone_confirmed": True,
        "activation_state": "activated_from_research_roadmap_next",
        "archive_state": "research_complete_contains_2026.06.425",
    }
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["milestone"] == "2026.06.426"
    assert artifact["preconditions_checked"]["offline_arcade"]["available"] is True
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is True

    close = artifact["close_state_425"]
    assert close["source_capstone_honest_verdict"] == "complete: pivot_characterized_capability_grew_55_to_55"
    assert close["capability"] == {
        "reproducible_total_levels_before": 55,
        "reproducible_total_levels_after": 55,
        "reproducible_total_levels_delta": 0,
        "capability_flat": True,
    }
    assert close["a1_world_model_trust_energy"] == {
        "status": "quarantined",
        "claimed_trust_pass_rate_binary": 0.0,
        "claimed_trust_pass_rate_new": 1.0,
        "claimed_first_win_rate_binary": 0.0,
        "claimed_first_win_rate_new": 1.0,
        "claimed_first_win_delta": 1.0,
        "flagged_adversarial": True,
        "critical_flag": "DURATION_TOO_SHORT",
        "duration_s": 0.44,
        "capstone_excluded": True,
        "reason": "degenerate_trivially_passing_gate",
    }
    assert close["a2_live_integration_scored_agent"] == {
        "status": "honest_null",
        "first_win_delta": 0.0,
        "actions_delta": 0.0,
        "solve_rate_integrated": 0.04,
        "solve_rate_bare": 0.04,
        "linear_verifier_earns_place": False,
    }
    assert close["a3_levelup_selfplay"] == {
        "status": "no_bank",
        "target_game": "dc22",
        "attempted_transition": "L1->L2",
        "reached_level": 1,
        "reproduced": False,
        "new_levels_banked": 0,
    }
    assert close["a4_package"] == {
        "live_submittable_level_count": 55,
        "beats_scorecard_baseline": 33,
        "ready_for_operator_submit": True,
    }
    assert artifact["v426_pivot"] == {
        "headline_rationale": "PIVOT to the OFFLINE->LIVE BRIDGE",
        "representation_not_bottleneck": True,
        "cross_game_features_v3_loo_auroc": 0.725,
        "candidate_causes_to_disambiguate": [
            "compute_cost",
            "distribution_shift",
            "calibration",
        ],
        "a1": "disambiguate_compute_shift_calibration",
        "a2": "graduate_spatial_value_net_to_live_path",
        "spatial_value_net_offline_lift": "7.6x",
        "replace_linear_verifier": True,
    }
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4615_missing_next_roadmap_blocks_without_fabricating(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4615-BLOCKED-PRECONDITION: missing next roadmap blocks."""

    _write_repo_fixture(tmp_path, include_next=False)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=2.0,
        now_s=2.2,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_next_yaml"
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["transition"]["active_milestone_confirmed"] is False
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["available"] is False
    assert artifact["preconditions_checked"]["offline_arcade"]["available"] is True
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is None
    assert artifact["close_state_425"] == {}
    assert artifact["v426_pivot"] == {}
    assert (tmp_path / mod.OUTPUT_REL_PATH).exists()
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4615_precondition_order_and_blockers(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4615: blocker classification is explicit and deterministic."""

    checks = _complete_artifact(tmp_path)["preconditions_checked"]
    assert mod._first_blocker(checks) is None

    next_bad = copy.deepcopy(checks)
    next_bad["research_roadmap_next_yaml"]["parses"] = False
    assert mod._first_blocker(next_bad) == "research_roadmap_next_yaml"

    active_bad = copy.deepcopy(checks)
    active_bad["active_research_roadmap_yaml"]["milestone"] = "2026.06.425"
    assert mod._first_blocker(active_bad) == "research_roadmap_426_unavailable"

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
    capstone_bad["capstone_4614"]["available"] = False
    assert mod._first_blocker(capstone_bad) == "missing_experiment_4614_capstone_v425"

    a1_bad = copy.deepcopy(checks)
    a1_bad["a1_exp4604"]["available"] = False
    assert mod._first_blocker(a1_bad) == "missing_experiment_4604_world_model_trust_energy"

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


def test_scenario_capstone_4615_field_principle_validation_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4615-FIELD-PRINCIPLES: schema drift fails loudly."""

    valid = _complete_artifact(tmp_path)

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
    blocked["close_state_425"] = {"fabricated": True}
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    inactive = copy.deepcopy(valid)
    inactive["transition"]["active_milestone_confirmed"] = False
    with pytest.raises(ValueError, match="active .426"):
        mod.validate_artifact(inactive)

    wrong_capability = copy.deepcopy(valid)
    wrong_capability["close_state_425"]["capability"]["reproducible_total_levels_after"] = 56
    with pytest.raises(ValueError, match="capability flat"):
        mod.validate_artifact(wrong_capability)

    wrong_a1 = copy.deepcopy(valid)
    wrong_a1["close_state_425"]["a1_world_model_trust_energy"]["flagged_adversarial"] = False
    with pytest.raises(ValueError, match="A1"):
        mod.validate_artifact(wrong_a1)

    wrong_a2 = copy.deepcopy(valid)
    wrong_a2["close_state_425"]["a2_live_integration_scored_agent"]["first_win_delta"] = 0.1
    with pytest.raises(ValueError, match="A2"):
        mod.validate_artifact(wrong_a2)

    wrong_a3 = copy.deepcopy(valid)
    wrong_a3["close_state_425"]["a3_levelup_selfplay"]["reproduced"] = True
    with pytest.raises(ValueError, match="A3"):
        mod.validate_artifact(wrong_a3)

    wrong_a4 = copy.deepcopy(valid)
    wrong_a4["close_state_425"]["a4_package"]["live_submittable_level_count"] = 33
    with pytest.raises(ValueError, match="A4"):
        mod.validate_artifact(wrong_a4)

    wrong_pivot = copy.deepcopy(valid)
    wrong_pivot["v426_pivot"]["headline_rationale"] = "representation"
    with pytest.raises(ValueError, match="v426 pivot"):
        mod.validate_artifact(wrong_pivot)

    bad_checksum_prefix = copy.deepcopy(valid)
    bad_checksum_prefix["reproducibility_checksum"] = "not-a-checksum"
    with pytest.raises(ValueError, match="sha256-prefixed"):
        mod.validate_artifact(bad_checksum_prefix)

    bad_checksum = copy.deepcopy(valid)
    bad_checksum["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad_checksum)

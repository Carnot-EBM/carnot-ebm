"""Tests for Exp 4532 `.418` archive / `.419` activation.

Spec refs: REQ-CAPSTONE-4532, SCENARIO-CAPSTONE-4532,
SCENARIO-CAPSTONE-4532-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4532_archive_418_activate_419 as mod


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
        "honest_verdict": "complete: nav_fix_null_efficiency_unmoved",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "efficiency_moved": False,
        "reproducible_total_levels_delta": {
            "prior_total": 48,
            "current_total": 50,
            "delta": 2,
            "banked_levels": 1,
            "capability_grew": True,
        },
        "scorecard": {
            "core_efficiency": {
                "baseline": 2.0074,
                "integrated": None,
                "moved": False,
                "reason": "integration_excluded_flagged_or_live_critical",
            },
            "stop_after_levelup_delta": {
                "status": "retired_action_trimming_context",
                "median_actions_control": 7761.5,
                "median_actions_best": 2825.5,
                "moves_score": False,
            },
            "a3_levelup": {
                "status": "level_up_banked",
                "target_game": "cd82",
                "target_level": 2,
                "banked_levels": 1,
                "level_up_banked": True,
            },
        },
        "a2_l1_l2_barrier_diagnosis": {
            "status": "excluded_flagged_adversarial",
            "cleanly_reportable": False,
            "what_blocks_deeper_levels": None,
            "what_to_build_next": "not_cleanly_reportable_from_flagged_artifact",
        },
    }


def _a1_forward_walk() -> JsonDict:
    return {
        "honest_verdict": "complete: forward_walk_no_reduction_honest_null",
        "median_actions_on_core_control": 7761.5,
        "median_actions_on_core_best": 7761.5,
        "core_solves_preserved": True,
        "local_gate_budget": 8000,
        "chosen_submitted_config": "unchanged",
        "nav_diagnostics_before_after": {
            "before": {"reset_replay_steps": 4576, "forward_walk_hits": 26},
            "after": {"reset_replay_steps": 4533, "forward_walk_hits": 34},
        },
        "flagged_adversarial": True,
    }


def _a2_reach_deeper_levels() -> JsonDict:
    return {
        "honest_verdict": "complete: l1_l2_barrier_diagnosed_depth_cap_honest_null",
        "core_efficiency_baseline": 2.0074,
        "core_efficiency_best": 2.0074,
        "barrier_diagnosis": {
            "root_cause": "depth_cap",
            "new_win_condition_likely": True,
            "induction_not_engaged": True,
            "target_game": "lp85",
            "evidence": [
                {
                    "l2_win_condition_differs_from_l1": True,
                    "known_l2_transition_in_salience": None,
                    "world_model_induction_invoked": False,
                    "stopped_reason": "depth_cap",
                }
            ],
            "actionable_next_step": (
                "force post-L1 DSL/goal-predicate induction and route lp85 frontier states "
                "toward the level-conditioned L2 predicate."
            ),
        },
        "flagged_adversarial": True,
    }


def _a3_levelup() -> JsonDict:
    return {
        "honest_verdict": "success: cd82_L2_offline_reproduced",
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "target_game": "cd82",
        "target_level": 2,
        "registry_update": {
            "prior_total_declared": 48,
            "new_total_declared": 50,
            "reconciled_total_delta": 2,
            "banked_levels": 1,
        },
    }


def _write_repo_fixture(root: Path) -> None:
    (root / "research-roadmap.yaml").write_text(
        "milestone: 2026.06.419\n"
        "tasks:\n"
        "  - id: exp4532-phase0\n"
        "    deliverable: results/experiment_4532_archive_418_activate_419.json\n"
        "    prompt: target_levels=1 and induction once on stall-and-not-won\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "milestones:\n"
        "- id: 2026.06.418\n"
        "  finding: prior roadmap archived by conductor\n",
        encoding="utf-8",
    )
    registry = root / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        "schema_version: 1\n"
        "updated: '2026-06-21'\n"
        "reproducible_total_levels: 50\n",
        encoding="utf-8",
    )
    _write_json(root / "results" / "experiment_4531_capstone_v418.json", _capstone())
    _write_json(root / "results" / "experiment_4523_forward_walk_navigation.json", _a1_forward_walk())
    _write_json(root / "results" / "experiment_4524_reach_deeper_levels.json", _a2_reach_deeper_levels())
    _write_json(root / "results" / "experiment_4525_levelup_attempt.json", _a3_levelup())


def _artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4532_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4532: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4532" in spec
    assert "SCENARIO-CAPSTONE-4532" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "reproducible_total_levels=50" in spec
    assert "A3 cd82 L2 banked" in spec
    for field, provenance in mod.FIELD_PROVENANCE.items():
        assert field in spec
        assert provenance["principle"] in spec


def test_scenario_capstone_4532_records_true_close_state_when_already_activated(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4532: already-activated `.419` still writes the close-state."""

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
    assert artifact["honest_verdict"] == "complete: archive_418_activate_419_true_close_state_recorded"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.418",
        "activated_milestone": "2026.06.419",
        "active_milestone_confirmed": True,
        "activation_state": "already_active_roadmap_next_consumed",
        "archive_state": "research_complete_contains_2026.06.418",
    }
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["available"] is False
    assert artifact["preconditions_checked"]["active_research_roadmap_yaml"]["milestone"] == "2026.06.419"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is True

    close = artifact["close_state_418"]
    assert close["reproducible_total_levels"] == 50
    assert close["efficiency_moved"] is False
    assert close["nav_action_trimming_dead_score_lever"] is True
    assert close["a1_forward_walk"]["median_actions_on_core_control"] == 7761.5
    assert close["a1_forward_walk"]["median_actions_on_core_best"] == 7761.5
    assert close["a1_forward_walk"]["fixed_transition_budget"] == 8000
    assert close["a2_barrier_diagnosis"]["barrier"] == "per_level_goal_reinduction"
    assert close["a2_barrier_diagnosis"]["l2_win_condition_differs_from_l1"] is True
    assert close["a2_barrier_diagnosis"]["induction_once_on_stall"] is True
    assert close["a2_barrier_diagnosis"]["target_levels"] == 1
    assert close["a3_levelup"]["target_game"] == "cd82"
    assert close["a3_levelup"]["target_level"] == 2
    assert close["a3_levelup"]["banked"] is True
    assert close["net_418"]["submitted_config"] == "unchanged"
    assert close["net_418"]["score_lever_to_build_next"] == "per_level_goal_reinduction"
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4532_blocks_without_fabricating_missing_capstone(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4532: missing required close-state input blocks honestly."""

    _write_repo_fixture(tmp_path)
    (tmp_path / "results" / "experiment_4531_capstone_v418.json").unlink()

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=2.0,
        now_s=2.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_missing_experiment_4531_capstone_v418"
    assert artifact["preconditions_checked"]["capstone_4531"]["available"] is False
    assert artifact["close_state_418"] == {}
    assert artifact["transition"]["active_milestone_confirmed"] is False
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4532_records_next_roadmap_activation_state(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4532: an extant next roadmap is recorded as activation input."""

    _write_repo_fixture(tmp_path)
    (tmp_path / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.06.419\ntasks: []\n",
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


def test_scenario_capstone_4532_precondition_blockers_are_classified(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4532: each required precondition has an honest blocked reason."""

    preconditions = _artifact(tmp_path)["preconditions_checked"]

    active_bad = copy.deepcopy(preconditions)
    active_bad["active_research_roadmap_yaml"]["milestone"] = "2026.06.418"
    active_bad["research_roadmap_next_yaml"]["available"] = False
    active_bad["research_roadmap_next_yaml"]["parses"] = False
    assert mod._first_blocker(active_bad) == "research_roadmap_419_unavailable"

    offline_bad = copy.deepcopy(preconditions)
    offline_bad["offline_arcade"]["available"] = False
    assert mod._first_blocker(offline_bad) == "offline_arcade"

    smart_bad = copy.deepcopy(preconditions)
    smart_bad["smart_subset_pretest_gate"]["passed"] = False
    assert mod._first_blocker(smart_bad) == "smart_subset_pretest_gate"

    registry_bad = copy.deepcopy(preconditions)
    registry_bad["registry"]["available"] = False
    assert mod._first_blocker(registry_bad) == "arc_solve_registry"

    a1_bad = copy.deepcopy(preconditions)
    a1_bad["a1_forward_walk_navigation"]["available"] = False
    assert mod._first_blocker(a1_bad) == "missing_experiment_4523_forward_walk_navigation"

    a2_bad = copy.deepcopy(preconditions)
    a2_bad["a2_reach_deeper_levels"]["available"] = False
    assert mod._first_blocker(a2_bad) == "missing_experiment_4524_reach_deeper_levels"

    a3_bad = copy.deepcopy(preconditions)
    a3_bad["a3_levelup_attempt"]["available"] = False
    assert mod._first_blocker(a3_bad) == "missing_experiment_4525_levelup_attempt"


def test_scenario_capstone_4532_parse_helpers_are_defensive(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4532: malformed inputs are detected instead of fabricated."""

    assert mod._float(True, 7.0) == 7.0
    assert mod._float("bad", 9.0) == 9.0
    assert mod._int(False, 2) == 2
    assert mod._int("bad", 3) == 3
    assert mod._registry_total_levels(tmp_path / "missing.yaml") is None
    assert mod._roadmap_target_levels_one(None) is False
    assert mod._known_l2_salience({"barrier_diagnosis": {"known_l2_transition_in_salience": "fallback"}}) == (
        "fallback"
    )

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


def test_scenario_capstone_4532_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4532-FIELD-PRINCIPLES: schema drift fails loudly."""

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
    blocked["close_state_418"] = {"fabricated": True}
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    inactive = copy.deepcopy(valid)
    inactive["transition"]["active_milestone_confirmed"] = False
    with pytest.raises(ValueError, match="active .419"):
        mod.validate_artifact(inactive)

    wrong_close = copy.deepcopy(valid)
    wrong_close["close_state_418"]["efficiency_moved"] = True
    with pytest.raises(ValueError, match="true .418 close-state"):
        mod.validate_artifact(wrong_close)

    wrong_a2 = copy.deepcopy(valid)
    wrong_a2["close_state_418"]["a2_barrier_diagnosis"]["barrier"] = "depth_cap"
    with pytest.raises(ValueError, match="per-level goal re-induction"):
        mod.validate_artifact(wrong_a2)

    wrong_net = copy.deepcopy(valid)
    wrong_net["close_state_418"]["net_418"]["action_efficiency_moved"] = True
    with pytest.raises(ValueError, match="net_418"):
        mod.validate_artifact(wrong_net)

    bad_checksum_prefix = copy.deepcopy(valid)
    bad_checksum_prefix["reproducibility_checksum"] = "not-a-checksum"
    with pytest.raises(ValueError, match="sha256-prefixed"):
        mod.validate_artifact(bad_checksum_prefix)

    bad_checksum_value = copy.deepcopy(valid)
    bad_checksum_value["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad_checksum_value)

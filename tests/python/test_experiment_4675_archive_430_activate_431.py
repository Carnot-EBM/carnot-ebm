"""Tests for Exp 4675 `.430` archive / `.431` activation record.

Spec refs: REQ-CAPSTONE-4675, SCENARIO-CAPSTONE-4675,
SCENARIO-CAPSTONE-4675-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4675-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4675_archive_430_activate_431 as mod


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


def _capstone_4674() -> JsonDict:
    return {
        "honest_verdict": "complete: capability_grew_58_to_59",
        "bridge_crossed_for_solve": False,
        "reproducible_total_levels": 59,
        "reproducible_total_levels_delta": 1,
        "live_submittable_level_count": 59,
        "paper_ready": True,
        "publication_gate": {"frozen_fover_auroc": 0.9131, "paper_ready": True},
        "a1_generic_agent_reached_l2": {
            "source": "results/experiment_4664_l2_goal_predicate_induction_live.json",
            "generic_agent_deepest_level": 0,
            "goal_predicate_satisfiable": False,
            "offline_reproduced": False,
            "reason": "positive_control_failed",
        },
        "a2_value_routing_live_lift": {
            "source": "results/experiment_4665_dagger_distribution_shift_value_routing.json",
            "distribution_shift_dropped": True,
            "distribution_shift_score_before": 0.699108,
            "distribution_shift_score_after": 0.0,
            "first_win_rate_delta": 0.0,
            "solve_rate_delta": 0.0,
            "live_first_win_rate_corrected": 0.04,
            "live_solve_rate_corrected": 0.0,
        },
        "scorecard": {
            "A3": {
                "honest_verdict": "success: dc22_L2_offline_reproduced",
                "registry_reproducible_total_levels": 59,
                "registry_delta_vs_58": 1,
                "offline_reproduced": True,
                "reproduced_levels": 1,
            },
            "A4": {
                "live_submittable_level_count": 59,
                "ready_for_operator_submit": True,
                "offline_reproduced": True,
            },
        },
    }


def _write_repo_fixture(
    root: Path,
    *,
    active_milestone: str = "2026.06.431",
    next_present: bool = False,
    registry_total: int = 59,
    capstone_present: bool = True,
) -> None:
    (root / "research-roadmap.yaml").write_text(
        f"milestone: {active_milestone}\n"
        "tasks:\n"
        "  - id: exp4675-phase0\n"
        "    deliverable: results/experiment_4675_archive_430_activate_431.json\n",
        encoding="utf-8",
    )
    if next_present:
        (root / "research-roadmap-next.yaml").write_text(
            "milestone: 2026.06.431\n"
            "tasks:\n"
            "  - id: exp4675-phase0\n"
            "    deliverable: results/experiment_4675_archive_430_activate_431.json\n",
            encoding="utf-8",
        )
    (root / "research-complete.yaml").write_text(
        "milestones:\n"
        "- id: 2026.06.430\n"
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
    proposal.write_text("Milestone 2026.06.431 CANDIDATE GENERATION\n", encoding="utf-8")
    if capstone_present:
        _write_json(root / "results" / "experiment_4674_capstone_v430.json", _capstone_4674())


def _artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4675_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4675: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4675" in spec
    assert "SCENARIO-CAPSTONE-4675" in spec
    assert "SCENARIO-CAPSTONE-4675-BLOCKED-PRECONDITION" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "CANDIDATE GENERATION" in spec
    for field, provenance in mod.FIELD_PROVENANCE.items():
        assert field in spec
        assert provenance["principle"] in spec


def test_scenario_capstone_4675_records_true_close_state_when_already_activated(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4675: active `.431` allows a complete record without next YAML."""

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
    assert artifact["honest_verdict"] == "complete: archive_430_activate_431_true_close_state_recorded"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.430",
        "activated_milestone": "2026.06.431",
        "active_milestone_confirmed": True,
        "activation_state": "already_activated_by_conductor",
        "archive_state": "research_complete_contains_2026.06.430",
    }
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["available"] is False
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"][
        "accepted_missing_because_already_active"
    ] is True
    assert artifact["preconditions_checked"]["active_research_roadmap_yaml"]["milestone"] == "2026.06.431"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is True

    close = artifact["close_state_430"]
    assert close["source_capstone_honest_verdict"] == "complete: capability_grew_58_to_59"
    assert close["a3_level_bank_dc22"] == {
        "honest_verdict": "success: dc22_L2_offline_reproduced",
        "target_game": "dc22",
        "prior_reproduced_level": 1,
        "target_level": 2,
        "reproducible_total_before": 58,
        "reproducible_total_after": 59,
        "reproducible_total_delta": 1,
        "offline_reproduced": True,
    }
    assert close["a1_l2_goal_induction"] == {
        "honest_verdict": "complete: l2_goal_induction_no_deepening_residual_single_exemplar_goal_insufficient",
        "null_and_retired": True,
        "win_state_exemplar_injected": False,
        "goal_predicate_satisfiable": {"lp85": False, "sc25": False},
        "bare_control_passed": False,
        "bare_control_note": "sc25 reached only L0 generically",
        "retire_if_same_verdict_fired": True,
    }
    assert close["a2_dagger_lite_value_routing"] == {
        "distribution_shift_score_before": 0.699108,
        "distribution_shift_score_after": 0.0,
        "distribution_shift_corrected": True,
        "first_win_rate_delta": 0.0,
        "solve_rate_delta": 0.0,
        "residual": "missing_verifier_gap_live_frontier_not_separated",
    }
    assert close["generic_fixed_harness_first_win"] == {
        "first_win_rate": 0.04,
        "wins": 1,
        "games": 25,
        "winning_games": ["lp85"],
        "not_assumed_rate": 0.59,
    }
    assert close["a4_submission_package"] == {
        "live_submittable_level_count": 59,
        "beats_submission_baseline": 33,
        "ready_for_operator_submit": True,
    }
    assert close["capstone"] == {
        "bridge_crossed_for_solve": False,
        "paper_ready": True,
        "frozen_fover_auroc": 0.9131,
    }

    assert artifact["v431_pivot"] == {
        "headline_rationale": "CANDIDATE GENERATION",
        "operator_frame": "make-a-winner-appear_not_select",
        "a1": {
            "lever": "hierarchical_subgoal_search_over_live_e3_frontier",
            "goal_induction_role": "subgoal_proposer",
            "value_head_role": "within_subgoal_tie_breaker",
            "step_1_gate": "resolve_0.04_vs_0.59",
        },
        "a2": {"lever": "poe_world_factored_executable_subgoal_planner"},
    }
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4675_can_activate_literal_next_roadmap(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4675: present next roadmap is activated onto the active YAML."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.430", next_present=True)

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=2.0,
        now_s=2.5,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8").startswith(
        "milestone: 2026.06.431"
    )
    assert artifact["transition"]["activation_state"] == "activated_from_research_roadmap_next"
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["activation_attempted"] is True
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"][
        "accepted_missing_because_already_active"
    ] is False


def test_scenario_capstone_4675_blockers_are_explicit(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4675-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.430", next_present=False)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=3.0,
        now_s=3.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_431_unavailable"
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is None
    assert artifact["close_state_430"] == {}
    assert artifact["v431_pivot"] == {}
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
    registry_bad["registry"]["reproducible_total_levels"] = 58
    assert mod._first_blocker(registry_bad) == "arc_solve_registry_total_levels_not_59"

    registry_missing = copy.deepcopy(checks)
    registry_missing["registry"]["available"] = False
    assert mod._first_blocker(registry_missing) == "arc_solve_registry"

    capstone_bad = copy.deepcopy(checks)
    capstone_bad["capstone_4674"]["available"] = False
    assert mod._first_blocker(capstone_bad) == "missing_experiment_4674_capstone_v430"

    design_bad = copy.deepcopy(checks)
    design_bad["vnext_design"]["available"] = False
    assert mod._first_blocker(design_bad) == "missing_research_roadmap_vnext_design"

    assert mod._command_check(None)["not_run_reason"] == "blocked_before_smart_subset_gate"
    assert mod._float(True, 7.0) == 7.0
    assert mod._float("bad", 9.0) == 9.0
    assert mod._int(False, 2) == 2
    assert mod._int("bad", 3) == 3
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


def test_scenario_capstone_4675_field_principle_validation_fails_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4675-FIELD-PRINCIPLES: schema drift fails loudly."""

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
    blocked["close_state_430"] = {"fabricated": True}
    blocked["reproducibility_checksum"] = "sha256:" + mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    inactive = copy.deepcopy(valid)
    inactive["transition"]["active_milestone_confirmed"] = False
    with pytest.raises(ValueError, match="active .431"):
        mod.validate_artifact(inactive)

    wrong_a3 = copy.deepcopy(valid)
    wrong_a3["close_state_430"]["a3_level_bank_dc22"]["reproducible_total_after"] = 58
    with pytest.raises(ValueError, match="A3"):
        mod.validate_artifact(wrong_a3)

    wrong_a1 = copy.deepcopy(valid)
    wrong_a1["close_state_430"]["a1_l2_goal_induction"]["null_and_retired"] = False
    with pytest.raises(ValueError, match="A1"):
        mod.validate_artifact(wrong_a1)

    wrong_a2 = copy.deepcopy(valid)
    wrong_a2["close_state_430"]["a2_dagger_lite_value_routing"]["distribution_shift_score_after"] = 0.2
    with pytest.raises(ValueError, match="A2"):
        mod.validate_artifact(wrong_a2)

    wrong_first_win = copy.deepcopy(valid)
    wrong_first_win["close_state_430"]["generic_fixed_harness_first_win"]["first_win_rate"] = 0.59
    with pytest.raises(ValueError, match="generic first-win"):
        mod.validate_artifact(wrong_first_win)

    wrong_a4 = copy.deepcopy(valid)
    wrong_a4["close_state_430"]["a4_submission_package"]["live_submittable_level_count"] = 33
    with pytest.raises(ValueError, match="A4"):
        mod.validate_artifact(wrong_a4)

    wrong_capstone = copy.deepcopy(valid)
    wrong_capstone["close_state_430"]["capstone"]["bridge_crossed_for_solve"] = True
    with pytest.raises(ValueError, match="capstone"):
        mod.validate_artifact(wrong_capstone)

    wrong_pivot = copy.deepcopy(valid)
    wrong_pivot["v431_pivot"]["headline_rationale"] = "SELECTION"
    with pytest.raises(ValueError, match="v431 pivot"):
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

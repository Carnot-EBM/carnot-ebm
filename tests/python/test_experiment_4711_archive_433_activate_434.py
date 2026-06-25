"""Tests for Exp 4711 `.433` archive / `.434` activation record.

Spec refs: REQ-CAPSTONE-4711, SCENARIO-CAPSTONE-4711,
SCENARIO-CAPSTONE-4711-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4711-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4711_archive_433_activate_434 as mod


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


def _a1_4700() -> JsonDict:
    return {
        "honest_verdict": "complete: object_centric_perception_no_new_level_residual_offpath_calibration_insufficient",
        "generic_agent_reached_level": 0,
        "chosen_submitted_config": "unchanged",
        "perception_is_the_wall": True,
        "proposal_coverage_by_representation": {
            "object_centric": {"coverage": 1.0},
            "order1": {"coverage": 0.75},
        },
        "residual_cause_hypothesis": "offpath_calibration_insufficient",
    }


def _a2_4701() -> JsonDict:
    return {
        "honest_verdict": "complete: amortized_prior_go_explore_no_coverage_gain_residual_logged",
        "coverage_delta": 0.0,
        "first_win_rate_delta": 0.0,
        "chosen_submitted_config": "unchanged",
        "residual_bridge_gap": "archive_expands_dead_cells_no_goal_gradient",
    }


def _a4_4703() -> JsonDict:
    return {
        "honest_verdict": "complete: held_out_first_win_flat_no_leaderboard_change",
        "first_win_rate_integrated": 0.04,
        "first_win_baseline": 0.04,
        "first_win_delta_vs_baseline": 0.0,
        "ready_for_operator_submit": False,
    }


def _arms_4710() -> JsonDict:
    return {
        "experiment": "experiment_4710_arms_summary",
        "honest_verdict": (
            "complete: online_action_learning_no_first_win_lift_null "
            "best_arm=online-warm best_delta=+0.0000 (kill_threshold=+0.05)"
        ),
        "frozen_first_win_rate": 0.04,
        "best_online_arm": "online-warm",
        "best_online_delta_vs_frozen": 0.0,
        "positive_control_passed": True,
    }


def _capstone_4710() -> JsonDict:
    return {
        "honest_verdict": "complete: capability_grew_61_to_62",
        "bridge_crossed_for_solve": False,
        "paper_ready": True,
        "reproducible_total_levels": 62,
        "reproducible_total_levels_delta": 1,
        "publication_gate": {
            "paper_ready": True,
            "frozen_fover_auroc": 0.9131,
            "fover_09131_never_substituted": True,
        },
        "a1_perception_is_the_wall_diagnostic": {
            "perception_is_the_wall": True,
            "object_centric_coverage": 1.0,
            "order1_coverage": 0.75,
        },
        "a1_perception_new_level": {"generic_agent_reached_level": 0},
        "a2_amortized_exploration_coverage_and_lift": {"coverage_delta": 0.0},
        "held_out_first_win_readiness": {
            "first_win_rate_integrated": 0.04,
            "first_win_baseline": 0.04,
            "first_win_delta_vs_baseline": 0.0,
        },
    }


def _write_repo_fixture(
    root: Path,
    *,
    active_milestone: str = "2026.06.434",
    next_present: bool = False,
    registry_total: int = 62,
    upstream_present: bool = True,
) -> None:
    (root / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# test\n", encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        f"milestone: {active_milestone}\n"
        "tasks:\n"
        "  - id: exp4711-phase0\n"
        "    deliverable: results/experiment_4711_archive_433_activate_434.json\n",
        encoding="utf-8",
    )
    if next_present:
        (root / "research-roadmap-next.yaml").write_text(
            "milestone: 2026.06.434\n"
            "tasks:\n"
            "  - id: exp4711-phase0\n"
            "    deliverable: results/experiment_4711_archive_433_activate_434.json\n",
            encoding="utf-8",
        )
    (root / "research-complete.yaml").write_text(
        "milestones:\n- id: 2026.06.433\n  finding: prior roadmap archived by conductor\n",
        encoding="utf-8",
    )
    registry = root / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        "schema_version: 1\n"
        "updated: '2026-06-25'\n"
        f"reproducible_total_levels: {registry_total}\n",
        encoding="utf-8",
    )
    spec = root / "openspec" / "capabilities" / "capstone" / "spec.md"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text("REQ-CAPSTONE-4711\n", encoding="utf-8")
    proposal = root / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md"
    proposal.parent.mkdir(parents=True, exist_ok=True)
    proposal.write_text(
        "Milestone 2026.06.434: perception SOLVED, SURFACING open; silent-bug audit.\n",
        encoding="utf-8",
    )
    if upstream_present:
        _write_json(root / "results" / "experiment_4700_object_centric_perception_proposal_live.json", _a1_4700())
        _write_json(root / "results" / "experiment_4701_amortized_exploration_prior_go_explore_live.json", _a2_4701())
        _write_json(root / "results" / "experiment_4703_held_out_first_win_readiness.json", _a4_4703())
        _write_json(root / "results" / "experiment_4710_arms_summary.json", _arms_4710())
        _write_json(root / "results" / "experiment_4710_capstone_v433.json", _capstone_4710())


def _artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4711_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4711: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4711" in spec
    assert "SCENARIO-CAPSTONE-4711" in spec
    assert "SCENARIO-CAPSTONE-4711-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CAPSTONE-4711-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "perception SOLVED" in spec
    assert "SURFACING open" in spec
    assert "Go-Explore archive `_frame_grid` returned `(1,64,64)`" in spec
    assert "CNN dict-candidate silent bug" in spec
    for field, provenance in mod.FIELD_PROVENANCE.items():
        assert field in spec
        assert provenance["principle"] in spec


def test_scenario_capstone_4711_records_true_close_state_when_already_activated(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4711: active `.434` allows a complete record without next YAML."""

    artifact = _artifact(tmp_path)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(artifact)
    assert artifact["honest_verdict"] == "complete: archive_433_activate_434_true_close_state_recorded"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.433",
        "activated_milestone": "2026.06.434",
        "active_milestone_confirmed": True,
        "activation_state": "already_activated_by_conductor",
        "archive_state": "research_complete_contains_2026.06.433",
    }
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"][
        "accepted_missing_because_already_active"
    ] is True
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["literal_precondition_passed"] is False
    assert artifact["preconditions_checked"]["active_research_roadmap_yaml"]["milestone"] == "2026.06.434"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is True

    assert artifact["close_state_433"] == {
        "source_capstone_honest_verdict": "complete: capability_grew_61_to_62",
        "a3_level_bank": {
            "prior_reproducible_total_levels": 61,
            "reproducible_total_after": 62,
            "reproducible_total_delta": 1,
            "capability_grew_61_to_62": True,
        },
        "a1_object_centric_perception": {
            "honest_verdict": "complete: object_centric_perception_no_new_level_residual_offpath_calibration_insufficient",
            "perception_is_the_wall": True,
            "object_centric_coverage": 1.0,
            "order1_coverage": 0.75,
            "generic_agent_reached_level": 0,
            "winner_rank_baseline": "59/161",
            "residual": "offpath_calibration_insufficient",
            "chosen_submitted_config": "unchanged",
        },
        "a2_amortized_exploration": {
            "honest_verdict": "complete: amortized_prior_go_explore_no_coverage_gain_residual_logged",
            "coverage_delta": 0.0,
            "first_win_rate_delta": 0.0,
            "tested_dead_code": True,
            "dead_code_evidence": "Go-Explore archive _frame_grid returned (1,64,64)",
            "fixed_date": "2026-06-25",
            "null_trustworthy": False,
        },
        "online_action_learning": {
            "honest_verdict": (
                "complete: online_action_learning_no_first_win_lift_null "
                "best_arm=online-warm best_delta=+0.0000 (kill_threshold=+0.05)"
            ),
            "first_win_rate": 0.04,
            "best_online_arm": "online-warm",
            "best_online_delta_vs_frozen": 0.0,
            "cnn_dict_candidate_silent_bug": True,
            "null_trustworthy": False,
        },
        "a4_held_out_first_win": {
            "honest_verdict": "complete: held_out_first_win_flat_no_leaderboard_change",
            "first_win_rate_integrated": 0.04,
            "first_win_baseline": 0.04,
            "first_win_delta_vs_baseline": 0.0,
            "flat_at_0_04": True,
        },
        "capstone": {
            "bridge_crossed_for_solve": False,
            "paper_ready": True,
            "frozen_fover_auroc": 0.9131,
        },
    }
    assert artifact["v434_pivot"] == {
        "headline_rationale": "L1 wall split: perception SOLVED; SURFACING open",
        "surface_present_winner": {
            "lane": "A2",
            "mechanism": "off-path-calibrated oracle-distinct verifier/ranker",
            "input_pool": "object-centric coverage-1.0 proposal pool",
            "baseline_winner_rank": "59/161",
            "goal": "surface present winner to actionable top-k",
        },
        "bank_perception_win": {
            "lane": "A1",
            "target": "lp85 L1->L2",
            "goal": "perception-grounded structural-alignment goal",
            "uses": "detected objects",
        },
        "corrected_online_driver": {
            "lane": "A4",
            "mechanism": "coordinate-head-proposes-clicks online driver",
            "fixes": ["Go-Explore (1,64,64) archive", "CNN dict-candidate bug"],
        },
        "silent_bug_audit": {
            "lane": "B1",
            "scope": ".428-.433 generation-lever nulls",
            "mandate": "classify silent_bug_must_reopen",
        },
    }
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4711_can_activate_literal_next_roadmap(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4711: present next roadmap is activated onto the active YAML."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.433", next_present=True)

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=2.0,
        now_s=2.5,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8").startswith(
        "milestone: 2026.06.434"
    )
    assert artifact["transition"]["activation_state"] == "activated_from_research_roadmap_next"
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["activation_attempted"] is True
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"][
        "accepted_missing_because_already_active"
    ] is False


def test_scenario_capstone_4711_blockers_are_explicit(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4711-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.433", next_present=False)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=3.0,
        now_s=3.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_434_unavailable"
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is None
    assert artifact["close_state_433"] == {}
    assert artifact["v434_pivot"] == {}
    assert (tmp_path / mod.OUTPUT_REL_PATH).exists()

    checks = _artifact(tmp_path)["preconditions_checked"]
    assert mod._first_blocker(checks) is None

    offline_bad = copy.deepcopy(checks)
    offline_bad["offline_arcade"]["available"] = False
    assert mod._first_blocker(offline_bad) == "offline_arcade"

    smart_bad = copy.deepcopy(checks)
    smart_bad["smart_subset_pretest_gate"]["passed"] = False
    assert mod._first_blocker(smart_bad) == "smart_subset_pretest_gate"

    agents_bad = copy.deepcopy(checks)
    agents_bad["agents_md"]["available"] = False
    assert mod._first_blocker(agents_bad) == "missing_agents_md"

    codex_bad = copy.deepcopy(checks)
    codex_bad["codex_or_opencode_md"]["available"] = False
    assert mod._first_blocker(codex_bad) == "missing_codex_or_opencode_md"

    registry_missing = copy.deepcopy(checks)
    registry_missing["registry"]["available"] = False
    assert mod._first_blocker(registry_missing) == "arc_solve_registry"

    registry_bad = copy.deepcopy(checks)
    registry_bad["registry"]["reproducible_total_levels"] = 61
    assert mod._first_blocker(registry_bad) == "arc_solve_registry_total_levels_not_62"

    spec_bad = copy.deepcopy(checks)
    spec_bad["capstone_spec"]["has_req_4711"] = False
    assert mod._first_blocker(spec_bad) == "missing_capstone_spec_req_4711"

    for name, expected in {
        "a1_4700": "missing_experiment_4700_object_centric_perception_proposal_live",
        "a2_4701": "missing_experiment_4701_amortized_exploration_prior_go_explore_live",
        "a4_4703": "missing_experiment_4703_held_out_first_win_readiness",
        "online_4710": "missing_experiment_4710_arms_summary",
        "capstone_4710": "missing_experiment_4710_capstone_v433",
        "vnext_design": "missing_research_roadmap_vnext_design",
    }.items():
        bad = copy.deepcopy(checks)
        bad[name]["available"] = False
        assert mod._first_blocker(bad) == expected

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


def test_scenario_capstone_4711_field_principle_validation_fails_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4711-FIELD-PRINCIPLES: schema drift fails loudly."""

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
    blocked["close_state_433"] = {"fabricated": True}
    blocked["reproducibility_checksum"] = "sha256:" + mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    inactive = copy.deepcopy(valid)
    inactive["transition"]["active_milestone_confirmed"] = False
    with pytest.raises(ValueError, match="active .434"):
        mod.validate_artifact(inactive)

    wrong_a3 = copy.deepcopy(valid)
    wrong_a3["close_state_433"]["a3_level_bank"]["reproducible_total_after"] = 61
    with pytest.raises(ValueError, match="A3"):
        mod.validate_artifact(wrong_a3)

    wrong_a1 = copy.deepcopy(valid)
    wrong_a1["close_state_433"]["a1_object_centric_perception"]["perception_is_the_wall"] = False
    with pytest.raises(ValueError, match="A1"):
        mod.validate_artifact(wrong_a1)

    wrong_a2 = copy.deepcopy(valid)
    wrong_a2["close_state_433"]["a2_amortized_exploration"]["null_trustworthy"] = True
    with pytest.raises(ValueError, match="A2"):
        mod.validate_artifact(wrong_a2)

    wrong_online = copy.deepcopy(valid)
    wrong_online["close_state_433"]["online_action_learning"]["cnn_dict_candidate_silent_bug"] = False
    with pytest.raises(ValueError, match="online-action"):
        mod.validate_artifact(wrong_online)

    wrong_a4 = copy.deepcopy(valid)
    wrong_a4["close_state_433"]["a4_held_out_first_win"]["first_win_rate_integrated"] = 0.08
    with pytest.raises(ValueError, match="A4"):
        mod.validate_artifact(wrong_a4)

    wrong_capstone = copy.deepcopy(valid)
    wrong_capstone["close_state_433"]["capstone"]["bridge_crossed_for_solve"] = True
    with pytest.raises(ValueError, match="capstone"):
        mod.validate_artifact(wrong_capstone)

    wrong_pivot = copy.deepcopy(valid)
    wrong_pivot["v434_pivot"]["headline_rationale"] = "re-run old nulls"
    with pytest.raises(ValueError, match="v434 pivot"):
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

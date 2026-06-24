"""Tests for Exp 4699 `.432` archive / `.433` activation record.

Spec refs: REQ-CAPSTONE-4699, SCENARIO-CAPSTONE-4699,
SCENARIO-CAPSTONE-4699-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4699-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4699_archive_432_activate_433 as mod


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


def _a1_4688() -> JsonDict:
    return {
        "honest_verdict": "complete: controllable_novelty_no_new_level_residual_winning_prefix_still_not_proposed",
        "generic_agent_reached_level": 0,
        "residual_cause_hypothesis": "winning_prefix_still_not_proposed",
        "chosen_submitted_config": "unchanged",
    }


def _a2_4689() -> JsonDict:
    return {
        "honest_verdict": "complete: program_synthesis_filter_no_coverage_gain_residual_heldout_sparse",
        "coverage_delta": 0.0,
        "first_win_rate_delta": -0.04,
        "heldout_programs_kept": 0,
        "residual_bridge_gap": "heldout_transitions_too_sparse",
        "chosen_submitted_config": "unchanged",
    }


def _a3_4690() -> JsonDict:
    return {
        "honest_verdict": "success: lf52_L2_offline_reproduced",
        "target_game": "lf52",
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "reproducible_total_levels_before": 60,
        "reproducible_total_levels_after": 61,
        "reproduction_gate": {"game": "lf52", "claimed_level": 2, "reached_level": 2, "reproduced": True},
    }


def _a4_4691() -> JsonDict:
    return {
        "honest_verdict": "complete: held_out_first_win_flat_no_leaderboard_change",
        "first_win_rate_integrated": 0.04,
        "first_win_baseline": 0.04,
        "first_win_delta_vs_baseline": 0.0,
        "ready_for_operator_submit": False,
        "flagged_adversarial": True,
    }


def _capstone_4698() -> JsonDict:
    return {
        "honest_verdict": "complete: capability_grew_60_to_61",
        "bridge_crossed_for_solve": False,
        "paper_ready": True,
        "reproducible_total_levels": 61,
        "reproducible_total_levels_delta": 1,
        "publication_gate": {"paper_ready": True, "frozen_fover_auroc": 0.9131},
        "a1_controllable_novelty_new_level": {
            "generic_agent_reached_level": 0,
            "reason": "winning_prefix_still_not_proposed",
        },
        "a2_program_synthesis_coverage_and_lift": {
            "coverage_delta": 0.0,
            "first_win_rate_delta": -0.04,
            "heldout_programs_kept": 0,
            "residual": "experts_overfit_prefix",
        },
        "held_out_first_win_readiness": {
            "first_win_rate_integrated": 0.04,
            "first_win_baseline": 0.04,
            "first_win_delta_vs_baseline": 0.0,
            "ready_for_operator_submit": False,
            "reason": "flagged_adversarial_or_live_critical_excluded",
        },
        "flagged_artifacts_handled": {
            "excluded_artifacts": ["results/experiment_4691_held_out_first_win_readiness.json"],
            "excluded_details": [
                {
                    "name": "A4",
                    "artifact": "results/experiment_4691_held_out_first_win_readiness.json",
                    "reason": "flagged_adversarial",
                    "critical_flags": [
                        {
                            "kind": "TAUTOLOGY",
                            "detail": "first_win_baseline=0.04 and first_win_rate_integrated=0.04 agree",
                        }
                    ],
                }
            ],
        },
    }


def _write_repo_fixture(
    root: Path,
    *,
    active_milestone: str = "2026.06.433",
    next_present: bool = False,
    registry_total: int = 61,
    upstream_present: bool = True,
) -> None:
    (root / "research-roadmap.yaml").write_text(
        f"milestone: {active_milestone}\n"
        "tasks:\n"
        "  - id: exp4699-phase0\n"
        "    deliverable: results/experiment_4699_archive_432_activate_433.json\n",
        encoding="utf-8",
    )
    if next_present:
        (root / "research-roadmap-next.yaml").write_text(
            "milestone: 2026.06.433\n"
            "tasks:\n"
            "  - id: exp4699-phase0\n"
            "    deliverable: results/experiment_4699_archive_432_activate_433.json\n",
            encoding="utf-8",
        )
    (root / "research-complete.yaml").write_text(
        "milestones:\n"
        "- id: 2026.06.432\n"
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
    proposal.write_text(
        "Milestone 2026.06.433 PERCEPTION plus AMORTIZED EXPLORATION\n",
        encoding="utf-8",
    )
    if upstream_present:
        _write_json(
            root / "results" / "experiment_4688_controllable_novelty_proposal_policy_live.json",
            _a1_4688(),
        )
        _write_json(
            root / "results" / "experiment_4689_program_synthesis_action_effect_proposal_filter.json",
            _a2_4689(),
        )
        _write_json(root / "results" / "experiment_4690_levelup_selfplay.json", _a3_4690())
        _write_json(
            root / "results" / "experiment_4691_held_out_first_win_readiness.json",
            _a4_4691(),
        )
        _write_json(root / "results" / "experiment_4698_capstone_v432.json", _capstone_4698())


def _artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4699_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4699: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4699" in spec
    assert "SCENARIO-CAPSTONE-4699" in spec
    assert "SCENARIO-CAPSTONE-4699-BLOCKED-PRECONDITION" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "PERCEPTION" in spec
    assert "AMORTIZED EXPLORATION" in spec
    for field, provenance in mod.FIELD_PROVENANCE.items():
        assert field in spec
        assert provenance["principle"] in spec


def test_scenario_capstone_4699_records_true_close_state_when_already_activated(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4699: active `.433` allows a complete record without next YAML."""

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
    assert artifact["honest_verdict"] == "complete: archive_432_activate_433_true_close_state_recorded"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.432",
        "activated_milestone": "2026.06.433",
        "active_milestone_confirmed": True,
        "activation_state": "already_activated_by_conductor",
        "archive_state": "research_complete_contains_2026.06.432",
    }
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["available"] is False
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"][
        "accepted_missing_because_already_active"
    ] is True
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["literal_precondition_passed"] is False
    assert artifact["preconditions_checked"]["active_research_roadmap_yaml"]["milestone"] == "2026.06.433"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is True

    assert artifact["close_state_432"] == {
        "source_capstone_honest_verdict": "complete: capability_grew_60_to_61",
        "a3_level_bank_lf52": {
            "honest_verdict": "success: lf52_L2_offline_reproduced",
            "target_game": "lf52",
            "prior_reproducible_total_levels": 60,
            "reproducible_total_after": 61,
            "reproducible_total_delta": 1,
            "target_level": 2,
            "offline_reproduced": True,
        },
        "a1_controllable_novelty_proposal_policy": {
            "honest_verdict": "complete: controllable_novelty_no_new_level_residual_winning_prefix_still_not_proposed",
            "generic_agent_reached_level": 0,
            "residual": "winning_prefix_still_not_proposed",
            "chosen_submitted_config": "unchanged",
        },
        "a2_program_synthesis_action_effect_proposal_filter": {
            "honest_verdict": "complete: program_synthesis_filter_no_coverage_gain_residual_heldout_sparse",
            "coverage_delta": 0.0,
            "first_win_rate_delta": -0.04,
            "heldout_programs_kept": 0,
            "residuals": ["experts_overfit_prefix", "heldout_transitions_too_sparse"],
            "chosen_submitted_config": "unchanged",
        },
        "a4_held_out_first_win": {
            "honest_verdict": "complete: held_out_first_win_flat_no_leaderboard_change",
            "first_win_rate_integrated": 0.04,
            "first_win_baseline": 0.04,
            "first_win_delta_vs_baseline": 0.0,
            "tautology_flagged": True,
            "null_delta_markers_missing": True,
            "ready_for_operator_submit": False,
        },
        "capstone": {
            "bridge_crossed_for_solve": False,
            "paper_ready": True,
            "frozen_fover_auroc": 0.9131,
        },
    }
    assert artifact["v433_pivot"] == {
        "headline_rationale": "PERCEPTION + AMORTIZED EXPLORATION",
        "perception": {
            "lever": "object_centric_relational_representation",
            "wired_into": "live_PROPOSAL_distribution",
            "diagnostic": "perception_vs_search",
            "operator_named_root_cause": "order_1_features_LOO_chance",
        },
        "amortized_exploration": {
            "cross_game_first_contact_prior": True,
            "go_explore_archive_wired_live": True,
            "source": ".432_sota_ingestion_explicit_.433_bottom_line",
        },
        "a4_fix": {
            "emit_null_delta_markers": True,
            "prevents": "honest_flat_first_win_null_quarantine",
        },
    }
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4699_can_activate_literal_next_roadmap(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4699: present next roadmap is activated onto the active YAML."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.432", next_present=True)

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=2.0,
        now_s=2.5,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8").startswith(
        "milestone: 2026.06.433"
    )
    assert artifact["transition"]["activation_state"] == "activated_from_research_roadmap_next"
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["activation_attempted"] is True
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"][
        "accepted_missing_because_already_active"
    ] is False


def test_scenario_capstone_4699_blockers_are_explicit(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4699-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.432", next_present=False)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=3.0,
        now_s=3.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_433_unavailable"
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is None
    assert artifact["close_state_432"] == {}
    assert artifact["v433_pivot"] == {}
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
    registry_bad["registry"]["reproducible_total_levels"] = 60
    assert mod._first_blocker(registry_bad) == "arc_solve_registry_total_levels_not_61"

    registry_missing = copy.deepcopy(checks)
    registry_missing["registry"]["available"] = False
    assert mod._first_blocker(registry_missing) == "arc_solve_registry"

    for name, expected in {
        "a1_4688": "missing_experiment_4688_controllable_novelty_proposal_policy_live",
        "a2_4689": "missing_experiment_4689_program_synthesis_action_effect_proposal_filter",
        "a3_4690": "missing_experiment_4690_levelup_selfplay",
        "a4_4691": "missing_experiment_4691_held_out_first_win_readiness",
        "capstone_4698": "missing_experiment_4698_capstone_v432",
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
    assert (
        mod._a4_tautology_flagged(
            {"flagged_artifacts_handled": {"excluded_details": ["not-a-mapping"]}},
            {"flagged_adversarial": True, "first_win_delta_vs_baseline": 0.0},
        )
        is True
    )
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


def test_scenario_capstone_4699_field_principle_validation_fails_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4699-FIELD-PRINCIPLES: schema drift fails loudly."""

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
    blocked["close_state_432"] = {"fabricated": True}
    blocked["reproducibility_checksum"] = "sha256:" + mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    inactive = copy.deepcopy(valid)
    inactive["transition"]["active_milestone_confirmed"] = False
    with pytest.raises(ValueError, match="active .433"):
        mod.validate_artifact(inactive)

    wrong_a3 = copy.deepcopy(valid)
    wrong_a3["close_state_432"]["a3_level_bank_lf52"]["reproducible_total_after"] = 60
    with pytest.raises(ValueError, match="A3"):
        mod.validate_artifact(wrong_a3)

    wrong_a1 = copy.deepcopy(valid)
    wrong_a1["close_state_432"]["a1_controllable_novelty_proposal_policy"]["residual"] = "none"
    with pytest.raises(ValueError, match="A1"):
        mod.validate_artifact(wrong_a1)

    wrong_a2 = copy.deepcopy(valid)
    wrong_a2["close_state_432"]["a2_program_synthesis_action_effect_proposal_filter"][
        "coverage_delta"
    ] = 1.0
    with pytest.raises(ValueError, match="A2"):
        mod.validate_artifact(wrong_a2)

    wrong_a4 = copy.deepcopy(valid)
    wrong_a4["close_state_432"]["a4_held_out_first_win"]["tautology_flagged"] = False
    with pytest.raises(ValueError, match="A4"):
        mod.validate_artifact(wrong_a4)

    wrong_capstone = copy.deepcopy(valid)
    wrong_capstone["close_state_432"]["capstone"]["bridge_crossed_for_solve"] = True
    with pytest.raises(ValueError, match="capstone"):
        mod.validate_artifact(wrong_capstone)

    wrong_pivot = copy.deepcopy(valid)
    wrong_pivot["v433_pivot"]["headline_rationale"] = "DIRECTED EXPLORATION"
    with pytest.raises(ValueError, match="v433 pivot"):
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

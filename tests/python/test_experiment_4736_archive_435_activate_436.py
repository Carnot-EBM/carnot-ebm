"""Tests for Exp 4736 `.435` archive / `.436` activation record.

Spec refs: REQ-CAPSTONE-4736, SCENARIO-CAPSTONE-4736,
SCENARIO-CAPSTONE-4736-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4736-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4736_archive_435_activate_436 as mod


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
        stdout="92 passed, 1 warning",
        stderr="",
    )


def _red_smart_subset(_root: Path) -> mod.CommandResult:
    return mod.CommandResult(
        command=["pytest", "smart-subset"],
        exit_code=1,
        stdout="1 failed, 91 passed, 1 warning",
        stderr="test_capstone_expected_clean_a1 failed",
    )


def _b1_4725() -> JsonDict:
    return {
        "honest_verdict": "complete: silent_bug_audit_12_nulls_5_must_reopen",
        "nulls_audited": 12,
        "silent_bug_nulls": [
            {
                "artifact_path": "results/experiment_4640_goal_energy_generation_live.json",
                "null_id": "experiment_4640_goal_energy_generation_live",
                "verdict": "silent_bug_must_reopen",
                "evidence": ["goal_energy arm cloned cached baseline attempts"],
            },
            {
                "artifact_path": "results/experiment_4653_energy_fitness_qd_generation_live.json",
                "null_id": "experiment_4653_energy_fitness_qd_generation_live",
                "verdict": "silent_bug_must_reopen",
                "evidence": ["QD/search/random arms byte-identical"],
            },
            {
                "artifact_path": "results/experiment_4676_hierarchical_subgoal_search_live.json",
                "null_id": "experiment_4676_hierarchical_subgoal_search_live",
                "verdict": "silent_bug_must_reopen",
            },
            {
                "artifact_path": "results/experiment_4701_amortized_exploration_prior_go_explore_live.json",
                "null_id": "experiment_4701_amortized_exploration_prior_go_explore_live",
                "verdict": "silent_bug_must_reopen",
            },
            {
                "artifact_path": "results/experiment_4715_online_action_learning_driver_corrected.json",
                "null_id": "experiment_4715_online_action_learning_driver_corrected",
                "verdict": "silent_bug_must_reopen",
            },
        ],
        "reopen_recommendations": [
            {"lever": "goal_energy_generation_live", "priority": "P2"},
            {"lever": "energy_fitness_qd_generation", "priority": "P3"},
        ],
    }


def _a1_4726() -> JsonDict:
    return {
        "honest_verdict": "complete: online_action_learning_no_first_win_lift_residual_online_signal_genuinely_too_sparse",
        "arms_non_degenerate": True,
        "online_train_steps_executed": 66,
        "per_arm_action_distribution_distinct": True,
        "non_degeneracy_gate": {
            "arms_non_degenerate": True,
            "coordinate_head_differs_from_frozen": True,
            "online_train_steps_executed": 66,
            "per_arm_action_distribution_distinct": True,
        },
        "frozen_first_win": 0.04,
        "online_warm_first_win": 0.04,
        "online_warm_vs_frozen_delta": 0.0,
        "null_delta_methodology_note": "honest no-lift null after non-degeneracy gate",
        "positive_control_passed": True,
        "chosen_submitted_config": "unchanged",
        "verifier_is_oracle": False,
    }


def _a2_4727() -> JsonDict:
    return {
        "honest_verdict": "complete: active_probe_no_new_level_residual_budget_insufficient",
        "active_probe_result": {"reason": "probe_mechanism_did_not_run"},
        "probe_actions_taken": 0,
        "hypothesis_posterior_built": False,
        "posterior_entropy_reduction": 0.0,
        "generic_agent_reached_level": 0,
        "offline_reproduced": False,
        "chosen_submitted_config": "unchanged",
        "verifier_is_oracle": False,
    }


def _a3_4728() -> JsonDict:
    return {
        "honest_verdict": "success: ar25_L3_offline_reproduced",
        "target_game": "ar25",
        "reached_level": 3,
        "reproducible_total_levels_before": 63,
        "reproducible_total_levels": 64,
        "new_levels_banked": 1,
        "offline_reproduced": True,
        "reproduced_levels": 3,
    }


def _capstone_4735() -> JsonDict:
    reopen_list = [
        "results/experiment_4640_goal_energy_generation_live.json",
        "results/experiment_4653_energy_fitness_qd_generation_live.json",
        "results/experiment_4676_hierarchical_subgoal_search_live.json",
        "results/experiment_4701_amortized_exploration_prior_go_explore_live.json",
        "results/experiment_4715_online_action_learning_driver_corrected.json",
    ]
    return {
        "honest_verdict": "blocked_upstream_artifacts",
        "bridge_crossed_for_solve": False,
        "reproducible_total_levels": 64,
        "reproducible_total_levels_delta": 1,
        "b1_silent_bug_reopen_list": reopen_list,
        "next_milestone_fallback": {"b1_reopen_list": reopen_list},
        "publication_gate": {
            "paper_ready": True,
            "frozen_fover_auroc": 0.9131,
            "fover_09131_frozen_never_substituted": True,
        },
    }


def _write_repo_fixture(
    root: Path,
    *,
    active_milestone: str = "2026.06.436",
    next_present: bool = False,
    registry_total: int = 64,
    upstream_present: bool = True,
) -> None:
    (root / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# test\n", encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        f"milestone: {active_milestone}\n"
        "tasks:\n"
        "  - id: exp4736-phase0\n"
        "    deliverable: results/experiment_4736_archive_435_activate_436.json\n",
        encoding="utf-8",
    )
    if next_present:
        (root / "research-roadmap-next.yaml").write_text(
            "milestone: 2026.06.436\n"
            "tasks:\n"
            "  - id: exp4736-phase0\n"
            "    deliverable: results/experiment_4736_archive_435_activate_436.json\n",
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
    spec.write_text("REQ-CAPSTONE-4736\n", encoding="utf-8")
    proposal = root / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md"
    proposal.parent.mkdir(parents=True, exist_ok=True)
    proposal.write_text(
        "Milestone 2026.06.436: valid-test goal-energy and energy-fitness QD generation.\n",
        encoding="utf-8",
    )
    log = root / "ops" / "conductor-log.md"
    log.write_text(
        "| 2026-06-25 07:24 UTC | PHASE B2 | SKIP | "
        "Pre-tests failing, self-heal failed: 1 failed, 91 passed, 1 warning in 6.44s |\n",
        encoding="utf-8",
    )
    if upstream_present:
        _write_json(root / "results" / "experiment_4725_silent_bug_audit.json", _b1_4725())
        _write_json(
            root / "results" / "experiment_4726_online_action_learning_driver_valid_test.json",
            _a1_4726(),
        )
        _write_json(root / "results" / "experiment_4727_active_probe_disambiguation.json", _a2_4727())
        _write_json(root / "results" / "experiment_4728_levelup_selfplay.json", _a3_4728())
        _write_json(root / "results" / "experiment_4735_capstone_v435.json", _capstone_4735())


def _artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4736_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4736: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4736" in spec
    assert "SCENARIO-CAPSTONE-4736" in spec
    assert "SCENARIO-CAPSTONE-4736-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CAPSTONE-4736-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "A3 banked +1 through `ar25` L3 offline reproduction" in spec
    assert "valid-testing the guidance-class generation levers" in spec
    assert "incident_agent_shipped_test_cascade" in spec
    for field, provenance in mod.FIELD_PROVENANCE.items():
        assert field in spec
        assert provenance["principle"] in spec


def test_scenario_capstone_4736_records_true_close_state_when_already_activated(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4736: active `.436` allows a complete record without next YAML."""

    artifact = _artifact(tmp_path)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(artifact)
    assert artifact["honest_verdict"] == "complete: archive_435_activate_436_true_close_state_recorded"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.435",
        "activated_milestone": "2026.06.436",
        "active_milestone_confirmed": True,
        "activation_state": "already_activated_by_conductor",
        "archive_state": "archive_noop_or_already_recorded",
    }
    next_check = artifact["preconditions_checked"]["research_roadmap_next_yaml"]
    assert next_check["accepted_missing_because_already_active"] is True
    assert next_check["literal_precondition_passed"] is False
    assert artifact["preconditions_checked"]["active_research_roadmap_yaml"]["milestone"] == "2026.06.436"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is True

    poison = artifact["poison_pretest_resolved"]
    assert poison["resolved"] is True
    assert poison["current_gate_passed"] is True
    assert poison["poison_tests"] == []
    assert poison["historical_signature_observed"] is True
    assert "1 failed, 91 passed" in poison["historical_signature"]

    close = artifact["close_state_435"]
    assert close["a3_level_bank"] == {
        "prior_reproducible_total_levels": 63,
        "reproducible_total_after": 64,
        "reproducible_total_delta": 1,
        "target_game": "ar25",
        "reached_level": 3,
        "offline_reproduced": True,
        "honest_verdict": "success: ar25_L3_offline_reproduced",
    }
    assert close["a1_online_driver"]["validly_tested"] is True
    assert close["a1_online_driver"]["arms_non_degenerate"] is True
    assert close["a1_online_driver"]["online_train_steps_executed"] == 66
    assert close["a1_online_driver"]["online_warm_vs_frozen_delta"] == 0.0
    assert close["a1_online_driver"]["genuine_null"] is True
    assert close["a1_online_driver"]["retires"] is True
    assert close["a2_active_probe"]["dead_code"] is True
    assert close["a2_active_probe"]["probe_actions_taken"] == 0
    assert close["a2_active_probe"]["reason"] == "probe_mechanism_did_not_run"
    assert close["b1_silent_bug_audit"]["nulls_audited"] == 12
    assert close["b1_silent_bug_audit"]["must_reopen_count"] == 5
    assert close["b1_silent_bug_audit"]["guidance_class_generation_levers"] == [
        "experiment_4640_goal_energy_generation_live",
        "experiment_4653_energy_fitness_qd_generation_live",
    ]
    assert close["capstone"]["bridge_crossed_for_solve"] is False
    assert close["capstone"]["consecutive_false_bridge_crossed_milestones"] == 11
    assert close["capstone"]["paper_ready"] is True
    assert close["capstone"]["frozen_fover_auroc"] == 0.9131

    assert artifact["v436_pivot"] == {
        "headline_rationale": "valid-test the guidance-class generation levers",
        "a1_goal_energy_candidate_generation": {
            "reopens": "experiment_4640_goal_energy_generation_live",
            "mechanism": "score real candidate states with graded goal-energy",
            "non_degeneracy_gate": "distinct candidate scores and candidate pool/ranking differs from baseline",
        },
        "a2_energy_fitness_qd_generation": {
            "reopens": "experiment_4653_energy_fitness_qd_generation_live",
            "mechanism": "distinct QD and random-mutation candidate pools with energy as fitness",
            "non_degeneracy_gate": "byte-distinct QD/random/search pools before lift measurement",
        },
        "null_delta_markers_required": True,
    }
    assert artifact["leaderboard_submission"] is False
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4736_can_activate_literal_next_roadmap(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4736: present next roadmap is activated onto the active YAML."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.435", next_present=True)

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=2.0,
        now_s=2.5,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8").startswith(
        "milestone: 2026.06.436"
    )
    assert artifact["transition"]["activation_state"] == "activated_from_research_roadmap_next"
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["activation_attempted"] is True
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"][
        "accepted_missing_because_already_active"
    ] is False


def test_scenario_capstone_4736_blockers_are_explicit(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4736-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.435", next_present=False)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=3.0,
        now_s=3.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_436_unavailable"
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is None
    assert artifact["poison_pretest_resolved"]["resolved"] is False
    assert artifact["close_state_435"] == {}
    assert artifact["v436_pivot"] == {}
    assert (tmp_path / mod.OUTPUT_REL_PATH).exists()

    checks = _artifact(tmp_path)["preconditions_checked"]
    assert mod._first_blocker(checks) is None

    for key, expected in {
        "agents_md": "missing_agents_md",
        "codex_or_opencode_md": "missing_codex_or_opencode_md",
        "capstone_spec": "missing_capstone_spec_req_4736",
        "registry": "arc_solve_registry",
        "b1_4725": "missing_experiment_4725_silent_bug_audit",
        "a1_4726": "missing_experiment_4726_online_action_learning_driver_valid_test",
        "a2_4727": "missing_experiment_4727_active_probe_disambiguation",
        "a3_4728": "missing_experiment_4728_levelup_selfplay",
        "capstone_4735": "missing_experiment_4735_capstone_v435",
        "conductor_log": "missing_conductor_log",
        "vnext_design": "missing_research_roadmap_vnext_design",
    }.items():
        bad = copy.deepcopy(checks)
        bad[key]["available"] = False
        if key == "capstone_spec":
            bad[key]["has_req_4736"] = False
        assert mod._first_blocker(bad) == expected

    offline_bad = copy.deepcopy(checks)
    offline_bad["offline_arcade"]["available"] = False
    assert mod._first_blocker(offline_bad) == "offline_arcade"

    smart_bad = copy.deepcopy(checks)
    smart_bad["smart_subset_pretest_gate"]["passed"] = False
    assert mod._first_blocker(smart_bad) == "smart_subset_pretest_gate"

    registry_bad = copy.deepcopy(checks)
    registry_bad["registry"]["reproducible_total_levels"] = 63
    assert mod._first_blocker(registry_bad) == "arc_solve_registry_total_levels_not_64"

    bad_smart = mod.build_artifact(
        tmp_path,
        started_s=4.0,
        now_s=4.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_red_smart_subset,
    )
    assert bad_smart["honest_verdict"] == "blocked_smart_subset_pretest_gate"
    assert bad_smart["poison_pretest_resolved"]["poison_tests"] == [
        {
            "id": "test_capstone_expected_clean_a1",
            "reason": "historical poison signature observed but current smart-subset gate is red",
            "action": "blocked_for_manual_fix_or_quarantine",
        }
    ]


def test_scenario_capstone_4736_field_principle_validation_fails_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4736-FIELD-PRINCIPLES: schema drift fails loudly."""

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

    bad_poison = copy.deepcopy(valid)
    bad_poison["poison_pretest_resolved"]["resolved"] = False
    with pytest.raises(ValueError, match="poison"):
        mod.validate_artifact(bad_poison)

    blocked = mod._blocked_artifact(
        reason="unit_test",
        preconditions_checked=valid["preconditions_checked"],
        poison_pretest_resolved=valid["poison_pretest_resolved"],
        duration_s=0.1,
        cited_upstream_artifacts=valid["cited_upstream_artifacts"],
    )
    blocked["close_state_435"] = {"fabricated": True}
    blocked["reproducibility_checksum"] = "sha256:" + mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    inactive = copy.deepcopy(valid)
    inactive["transition"]["active_milestone_confirmed"] = False
    with pytest.raises(ValueError, match="active .436"):
        mod.validate_artifact(inactive)

    wrong_a3 = copy.deepcopy(valid)
    wrong_a3["close_state_435"]["a3_level_bank"]["reproducible_total_after"] = 63
    with pytest.raises(ValueError, match="A3"):
        mod.validate_artifact(wrong_a3)

    wrong_a1 = copy.deepcopy(valid)
    wrong_a1["close_state_435"]["a1_online_driver"]["arms_non_degenerate"] = False
    with pytest.raises(ValueError, match="A1"):
        mod.validate_artifact(wrong_a1)

    wrong_a2 = copy.deepcopy(valid)
    wrong_a2["close_state_435"]["a2_active_probe"]["dead_code"] = False
    with pytest.raises(ValueError, match="A2"):
        mod.validate_artifact(wrong_a2)

    wrong_b1 = copy.deepcopy(valid)
    wrong_b1["close_state_435"]["b1_silent_bug_audit"]["must_reopen_count"] = 4
    with pytest.raises(ValueError, match="B1"):
        mod.validate_artifact(wrong_b1)

    wrong_capstone = copy.deepcopy(valid)
    wrong_capstone["close_state_435"]["capstone"]["bridge_crossed_for_solve"] = True
    with pytest.raises(ValueError, match="capstone"):
        mod.validate_artifact(wrong_capstone)

    wrong_pivot = copy.deepcopy(valid)
    wrong_pivot["v436_pivot"]["headline_rationale"] = "rerun online driver"
    with pytest.raises(ValueError, match="v436 pivot"):
        mod.validate_artifact(wrong_pivot)

    bad_checksum_prefix = copy.deepcopy(valid)
    bad_checksum_prefix["reproducibility_checksum"] = "not-a-checksum"
    with pytest.raises(ValueError, match="sha256-prefixed"):
        mod.validate_artifact(bad_checksum_prefix)

    bad_checksum = copy.deepcopy(valid)
    bad_checksum["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad_checksum)

    assert mod._float(True, 7.0) == 7.0
    assert mod._float("bad", 9.0) == 9.0
    assert mod._int(False, 2) == 2
    assert mod._int("bad", 3) == 3
    assert mod._registry_total_levels(tmp_path / "missing.yaml") is None
    assert mod._activate_next_roadmap(tmp_path, next_info={"available": False}) == (False, "")
    assert mod._guidance_generation_levers("not-a-list") == []

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

"""Tests for Exp 4840 `.445` archive / `.446` activation record.

Spec refs: REQ-CAPSTONE-4840, SCENARIO-CAPSTONE-4840,
SCENARIO-CAPSTONE-4840-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4840-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4840_archive_445_activate_446 as mod


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
        stdout="132 passed in 7.0s",
        stderr="",
    )


def _red_poison_smart_subset(_root: Path) -> mod.CommandResult:
    return mod.CommandResult(
        command=["pytest", "smart-subset"],
        exit_code=1,
        stdout="1 failed, 131 passed in 7.1s",
        stderr="test_expected_exploration_prior_followup still expects another reweighting lever",
    )


def _a1_4831() -> JsonDict:
    return {
        "experiment": "experiment_4831_amortized_incontext_exploration_prior_live",
        "experiment_id": 4831,
        "honest_verdict": "complete_amortized_prior_no_first_win_lift_l1_wall_survives",
        "baseline_first_win_rate": 0.04,
        "first_win_rate_with_prior": 0.0,
        "first_win_rate_no_prior_ablation": 0.0,
        "first_win_delta_ci95": {"confidence": 0.95, "low": 0.0, "high": 0.0, "n_boot": 1000},
        "go_explore_archive_alive": {
            "alive": True,
            "observations": 2,
            "stored_cells": 2,
            "prefixes_injected": 1,
            "actions_injected": 1,
            "verifier_is_oracle": False,
        },
        "prior_changed_proposals": True,
        "imitation_control_heldout_games": {
            "heldout_not_in_distillation_set": True,
            "lift_holds": False,
            "heldout_games": ["bp35"],
            "distillation_games": ["cd82", "cn04"],
        },
        "live_path_reachable": True,
        "inference_substrate": "live_llm_inference",
    }


def _b1_4835() -> JsonDict:
    return {
        "experiment": "experiment_4835_silent_bug_audit",
        "experiment_id": 4835,
        "honest_verdict": "complete_arc_null_silent_bug_audit_3_nulls_0_reopen",
        "nulls_audited": 3,
        "trusted_nulls": [
            "experiment_4831_amortized_incontext_exploration_prior_live",
            "experiment_4832_levelup_attempt",
            "experiment_4834_heldout_first_win_readiness",
        ],
        "silent_bugs_found": [],
        "a1_archive_alive_and_prior_exercised": True,
        "a1_control_check": {
            "archive_alive": True,
            "prior_changed": True,
            "proposal_order_changed": True,
            "proposal_changes": 1,
            "heldout_not_in_distillation_set": True,
            "imitation_control_confirmed": True,
            "imitation_lift_holds": False,
            "first_win_rate_with_prior": 0.0,
            "first_win_rate_no_prior_ablation": 0.0,
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _capstone_4839() -> JsonDict:
    return {
        "experiment": "experiment_4839_capstone_v445",
        "experiment_id": 4839,
        "capstone_ready": True,
        "honest_verdict": (
            "complete_a1_genuine_null_l1_wall_survives_exploration_prior_closed_capstone_ready"
        ),
        "reproducible_total_levels": 65,
        "a1_amortized_prior_verdict": {
            "source": "A1",
            "experiment_id": 4831,
            "verdict": "genuine_null_l1_wall_survives_exploration_prior_closed",
            "upstream_honest_verdict": "complete_amortized_prior_no_first_win_lift_l1_wall_survives",
            "archive_alive": True,
            "archive_alive_confirmed_by_b1": True,
            "prior_changed_proposals": True,
            "prior_exercised_confirmed_by_b1": True,
            "imitation_control_confirmed": True,
            "heldout_not_in_distillation_set": True,
            "imitation_lift_holds": False,
            "baseline_first_win_rate": 0.04,
            "first_win_rate_with_prior": 0.0,
            "first_win_rate_no_prior_ablation": 0.0,
            "first_win_delta_ci95": {
                "confidence": 0.95,
                "low": 0.0,
                "high": 0.0,
                "n_boot": 1000,
            },
            "first_win_ci_excludes_zero": False,
            "lift_over_baseline": False,
            "wall_moves": False,
            "genuine_null": True,
            "exploration_prior_class_closed": True,
            "dead_archive_non_test": False,
            "live_path_reachable": True,
            "reason": "archive_alive_prior_exercised_no_heldout_lift",
            "direction_next": "perception_representation_frontier",
            "silent_bugs_found": [],
        },
        "readiness": {
            "a1_verdict": "genuine_null_l1_wall_survives_exploration_prior_closed",
            "exploration_prior_class_closed": True,
            "heldout_decision": "flat_baseline_first_win_null",
            "l1_wall_survives": True,
            "ready_for_operator_submit": False,
            "reason": "a1_genuine_null_frontier_moves_to_perception_representation",
            "v446_frontier": "perception/representation",
            "wall_moves": False,
        },
        "heldout_readiness": {
            "heldout_first_win_rate": 0.04,
            "first_win_baseline": 0.04,
            "heldout_first_win_delta_vs_baseline": 0.0,
            "heldout_first_win_delta_vs_prior_best": 0.0,
            "decision": "flat_baseline_first_win_null",
        },
        "silent_bug_audit": {
            "a1_archive_alive_and_prior_exercised": True,
            "silent_bugs_found": [],
            "silent_bugs_found_count": 0,
            "a1_control_check": _b1_4835()["a1_control_check"],
        },
        "sota_handoff": {
            "decision": "perception_representation_handoff",
            "v446_frontier": "perception/representation",
            "l1_wall_context": {
                "root_cause": "perception/representation",
                "exploration_strategy_class_retired": True,
                "nulled_lever_count_approx": 15,
                "planner_constraint": (
                    "Do not spend .446 on another exploration reweighting run; require a "
                    "representation that can make a novel winning prefix enter the pool."
                ),
            },
        },
    }


def _write_repo_fixture(
    root: Path,
    *,
    active_milestone: str = "2026.06.446",
    next_present: bool = False,
    registry_total: int = 65,
    capstone_present: bool = True,
    a1_present: bool = True,
    b1_present: bool = True,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# test\n", encoding="utf-8")
    active_text = (
        "# Pre-staged .446 roadmap\n"
        f"milestone: {active_milestone}\n"
        "theme: exploration-prior class CLOSED; perception/representation frontier\n"
        "tasks:\n"
        "  - id: exp4840-phase0\n"
        "    deliverable: results/experiment_4840_archive_445_activate_446.json\n"
    )
    (root / "research-roadmap.yaml").write_text(active_text, encoding="utf-8")
    if next_present:
        (root / "research-roadmap-next.yaml").write_text(
            active_text.replace(active_milestone, "2026.06.446", 1),
            encoding="utf-8",
        )
    registry = root / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        "schema_version: 1\n"
        "updated: '2026-06-27'\n"
        f"reproducible_total_levels: {registry_total}\n",
        encoding="utf-8",
    )
    spec = root / "openspec" / "capabilities" / "capstone" / "spec.md"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text("REQ-CAPSTONE-4840\n", encoding="utf-8")
    if capstone_present:
        _write_json(root / "results" / "experiment_4839_capstone_v445.json", _capstone_4839())
    if a1_present:
        _write_json(
            root / "results" / "experiment_4831_amortized_incontext_exploration_prior_live.json",
            _a1_4831(),
        )
    if b1_present:
        _write_json(root / "results" / "experiment_4835_silent_bug_audit.json", _b1_4835())


def _artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4840_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4840: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4840_restores_next_and_records_exploration_class_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4840: active `.446` records the true `.445` close-state."""

    _write_repo_fixture(tmp_path, next_present=False)
    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=2.0,
        now_s=2.3,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(artifact)
    assert (tmp_path / "research-roadmap-next.yaml").exists()
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"][
        "restored_from_active_roadmap"
    ] is True
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"][
        "literal_precondition_passed"
    ] is True
    assert artifact["honest_verdict"] == (
        "complete_445_archived_446_activated_already_active_exploration_prior_closed"
    )
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.445",
        "activated_milestone": "2026.06.446",
        "active_milestone_confirmed": True,
        "activation_state": "already_activated_by_conductor",
        "archive_state": "archive_noop_or_already_recorded",
    }
    assert artifact["exploration_prior_class_closed"] is True
    assert artifact["energy_program_concluded"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["reproducible_total_levels"] == 65
    assert artifact["poison_test_resolved"] == {
        "resolved": True,
        "current_gate_passed": True,
        "poison_tests": [],
        "action": "no_poison_observed_current_gate_green",
    }

    close = artifact["close_state_445"]
    assert close["capstone_honest_verdict"] == (
        "complete_a1_genuine_null_l1_wall_survives_exploration_prior_closed_capstone_ready"
    )
    assert close["reproducible_total_levels"] == 65
    assert close["energy_program_concluded"] is True
    assert close["exploration_prior_class_closed"] is True
    a1 = close["a1_amortized_prior_verdict"]
    assert a1["verdict"] == "genuine_null_l1_wall_survives_exploration_prior_closed"
    assert a1["archive_alive"] is True
    assert a1["prior_exercised_confirmed_by_b1"] is True
    assert a1["first_win_rate_with_prior"] == 0.0
    assert a1["first_win_rate_no_prior_ablation"] == 0.0
    assert a1["baseline_first_win_rate"] == 0.04
    assert a1["lift_over_baseline"] is False
    assert a1["first_win_delta_ci95"] == {
        "confidence": 0.95,
        "low": 0.0,
        "high": 0.0,
        "n_boot": 1000,
    }

    frontier = artifact["v446_frontier"]
    assert frontier["root_cause"] == "perception/representation"
    assert frontier["headline_build"] == "generic_object_identity_perception_layer_for_goal_grounding"
    assert frontier["planner_must_not_repropose_exploration_strategy_levers"] is True
    assert frontier["planner_must_not_reopen_energy_program"] is True
    assert artifact["leaderboard_submission"] is False
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4840_can_activate_literal_next_roadmap(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4840: present next roadmap activates onto active YAML."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.445", next_present=True)

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=3.0,
        now_s=3.4,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8").startswith(
        "# Pre-staged .446 roadmap\nmilestone: 2026.06.446"
    )
    assert artifact["transition"]["activation_state"] == "activated_from_research_roadmap_next"
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["activation_attempted"] is True
    assert artifact["honest_verdict"] == (
        "complete_445_archived_446_activated_from_next_exploration_prior_closed"
    )


def test_scenario_capstone_4840_blockers_and_poison_signature_are_explicit(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4840-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.445", next_present=False)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=4.0,
        now_s=4.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_next_yaml"
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is None
    assert artifact["exploration_prior_class_closed"] is False
    assert artifact["energy_program_concluded"] is False
    assert artifact["close_state_445"] == {}
    assert artifact["v446_frontier"] == {}

    checks = _artifact(tmp_path / "good")["preconditions_checked"]
    assert mod._first_blocker(checks) is None

    for key, expected in {
        "agents_md": "missing_agents_md",
        "codex_or_opencode_md": "missing_codex_or_opencode_md",
        "capstone_spec": "missing_capstone_spec_req_4840",
        "registry": "arc_solve_registry",
        "capstone_4839": "missing_experiment_4839_capstone_v445",
        "a1_4831": "missing_experiment_4831_amortized_incontext_exploration_prior_live",
        "b1_4835": "missing_experiment_4835_silent_bug_audit",
    }.items():
        bad = copy.deepcopy(checks)
        bad[key]["available"] = False
        if key == "capstone_spec":
            bad[key]["has_req_4840"] = False
        assert mod._first_blocker(bad) == expected

    offline_bad = copy.deepcopy(checks)
    offline_bad["offline_arcade"]["available"] = False
    assert mod._first_blocker(offline_bad) == "offline_arcade"

    registry_bad = copy.deepcopy(checks)
    registry_bad["registry"]["reproducible_total_levels"] = 64
    assert mod._first_blocker(registry_bad) == "arc_solve_registry_total_levels_not_65"

    activation_bad = copy.deepcopy(checks)
    activation_bad["research_roadmap_next_yaml"]["activation_error"] = "permission denied"
    assert mod._first_blocker(activation_bad) == "research_roadmap_activation_error"

    active_bad = copy.deepcopy(checks)
    active_bad["active_research_roadmap_yaml"]["milestone"] = "2026.06.445"
    assert mod._first_blocker(active_bad) == "research_roadmap_446_unavailable"

    bad_smart = mod.build_artifact(
        tmp_path / "good",
        started_s=5.0,
        now_s=5.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_red_poison_smart_subset,
    )
    assert bad_smart["honest_verdict"] == "blocked_smart_subset_pretest_gate"
    assert bad_smart["poison_test_resolved"]["poison_tests"] == [
        {
            "id": "test_expected_exploration_prior_followup",
            "reason": "single-failure smart-subset signature matches a stale transition expectation",
            "action": "blocked_for_fix_or_quarantine_before_tail_continues",
        }
    ]


def test_scenario_capstone_4840_field_principle_validation_fails_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4840-FIELD-PRINCIPLES: schema drift fails loudly."""

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

    bad_principles = copy.deepcopy(valid)
    bad_principles["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad_principles)

    bad_poison = copy.deepcopy(valid)
    bad_poison["poison_test_resolved"]["resolved"] = False
    with pytest.raises(ValueError, match="poison"):
        mod.validate_artifact(bad_poison)

    blocked = mod._blocked_artifact(
        reason="unit_test",
        preconditions_checked=valid["preconditions_checked"],
        poison_test_resolved=valid["poison_test_resolved"],
        duration_s=0.1,
        cited_upstream_artifacts=valid["cited_upstream_artifacts"],
    )
    blocked["close_state_445"] = {"fabricated": True}
    blocked["reproducibility_checksum"] = "sha256:" + mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    wrong_total = copy.deepcopy(valid)
    wrong_total["reproducible_total_levels"] = 64
    with pytest.raises(ValueError, match="registry total"):
        mod.validate_artifact(wrong_total)

    wrong_exploration = copy.deepcopy(valid)
    wrong_exploration["exploration_prior_class_closed"] = False
    with pytest.raises(ValueError, match="exploration-prior"):
        mod.validate_artifact(wrong_exploration)

    wrong_energy = copy.deepcopy(valid)
    wrong_energy["energy_program_concluded"] = False
    with pytest.raises(ValueError, match="energy program"):
        mod.validate_artifact(wrong_energy)

    for field, value in {
        "verdict": "first_win_lift_wall_moves",
        "archive_alive": False,
        "archive_alive_confirmed_by_b1": False,
        "prior_changed_proposals": False,
        "prior_exercised_confirmed_by_b1": False,
        "heldout_not_in_distillation_set": False,
        "imitation_lift_holds": True,
        "first_win_rate_with_prior": 0.08,
        "first_win_rate_no_prior_ablation": 0.04,
        "baseline_first_win_rate": 0.05,
        "lift_over_baseline": True,
        "wall_moves": True,
        "exploration_prior_class_closed": False,
        "dead_archive_non_test": True,
        "live_path_reachable": False,
        "silent_bugs_found": ["bug"],
    }.items():
        wrong = copy.deepcopy(valid)
        wrong["close_state_445"]["a1_amortized_prior_verdict"][field] = value
        with pytest.raises(ValueError, match="amortized-prior"):
            mod.validate_artifact(wrong)

    wrong_ci = copy.deepcopy(valid)
    wrong_ci["close_state_445"]["a1_amortized_prior_verdict"]["first_win_delta_ci95"] = {
        "low": 0.1,
        "high": 0.2,
    }
    with pytest.raises(ValueError, match="amortized-prior"):
        mod.validate_artifact(wrong_ci)

    wrong_frontier = copy.deepcopy(valid)
    wrong_frontier["v446_frontier"]["root_cause"] = "exploration"
    with pytest.raises(ValueError, match="perception/representation"):
        mod.validate_artifact(wrong_frontier)

    bad_checksum_prefix = copy.deepcopy(valid)
    bad_checksum_prefix["reproducibility_checksum"] = "not-a-checksum"
    with pytest.raises(ValueError, match="sha256-prefixed"):
        mod.validate_artifact(bad_checksum_prefix)

    bad_checksum = copy.deepcopy(valid)
    bad_checksum["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad_checksum)

    assert mod._activate_next_roadmap(tmp_path, next_info={"available": False}) == (False, "")

    restore_error_root = tmp_path / "restore_error"
    restore_error_root.mkdir()
    (restore_error_root / "research-roadmap.yaml").mkdir()
    restored, restore_error = mod._restore_next_from_active_if_needed(
        restore_error_root,
        active_info={"available": True, "parses": True, "milestone": "2026.06.446"},
    )
    assert restored is False
    assert restore_error

    activation_error_root = tmp_path / "activation_error"
    activation_error_root.mkdir()
    (activation_error_root / "research-roadmap.yaml").mkdir()
    (activation_error_root / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.06.446\n",
        encoding="utf-8",
    )
    activated, activation_error = mod._activate_next_roadmap(
        activation_error_root,
        next_info={"available": True, "parses": True, "milestone": "2026.06.446"},
    )
    assert activated is False
    assert activation_error

    fallback_capstone = _capstone_4839()
    del fallback_capstone["a1_amortized_prior_verdict"]["first_win_rate_with_prior"]
    del fallback_capstone["a1_amortized_prior_verdict"]["first_win_rate_no_prior_ablation"]
    del fallback_capstone["a1_amortized_prior_verdict"]["baseline_first_win_rate"]
    fallback_a1 = _a1_4831()
    fallback = mod._a1_close_state(fallback_capstone, fallback_a1, _b1_4835())
    assert fallback["first_win_rate_with_prior"] == 0.0
    assert fallback["first_win_rate_no_prior_ablation"] == 0.0
    assert fallback["baseline_first_win_rate"] == 0.04

    def _offline_raises() -> bool:
        raise RuntimeError("offline arcade unavailable")

    offline_root = tmp_path / "offline"
    _write_repo_fixture(offline_root)
    offline_artifact = mod.build_artifact(
        offline_root,
        started_s=6.0,
        now_s=6.1,
        offline_arcade_checker=_offline_raises,
        smart_subset_checker=_green_smart_subset,
    )
    assert offline_artifact["honest_verdict"] == "blocked_offline_arcade"
    assert offline_artifact["preconditions_checked"]["offline_arcade"]["error"] == (
        "offline arcade unavailable"
    )

"""Tests for Exp 4830 `.444` archive / `.445` activation record.

Spec refs: REQ-CAPSTONE-4830, SCENARIO-CAPSTONE-4830,
SCENARIO-CAPSTONE-4830-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4830-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4830_archive_444_activate_445 as mod


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
        stderr="test_expected_energy_s4_followup still expects live energy value",
    )


def _s3_4821() -> JsonDict:
    return {
        "experiment": "experiment_4821_structural_energy_s3_generation_lift",
        "experiment_id": 4821,
        "honest_verdict": "complete_structural_energy_s3_bounded_no_generation_lift",
        "verifier_is_oracle": False,
        "live_path_reachable": True,
        "lambda0_control": {
            "description": "goal_guidance_lambda=0 disables plan_in_model goal_energy guidance",
            "lambda": 0.0,
            "matched_control": True,
        },
        "lambda_guidance": 1.0,
        "n_headroom_games": 24,
        "min_headroom_games": 5,
        "positive_control_passed": True,
        "new_levels_not_in_bare_pool": [],
        "winners_newly_entering_pool_delta": 0.0,
        "winners_newly_entering_pool_delta_ci95": [0.0, 0.0],
        "retire_if_same_verdict": True,
        "solve_provenance": "live_agent_self_discovery",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
    }


def _capstone_4829() -> JsonDict:
    return {
        "experiment": "experiment_4829_capstone_v444",
        "experiment_id": 4829,
        "capstone_ready": True,
        "honest_verdict": "complete_s3_bounded_no_generation_lift_capstone_ready",
        "reproducible_total_levels": 65,
        "s3_structural_energy_verdict": {
            "verdict": "bounded_no_generation_lift",
            "bounded_no_generation_lift": True,
            "generation_win": False,
            "s4_authorized": False,
            "direction_after_s3": "bounded_at_real_offline_discriminator_no_live_value",
            "upstream_honest_verdict": "complete_structural_energy_s3_bounded_no_generation_lift",
            "verifier_is_oracle": False,
            "live_path_reachable": True,
            "matched_lambda0_control": True,
            "lambda0_control": {
                "description": "goal_guidance_lambda=0 disables plan_in_model goal_energy guidance",
                "lambda": 0.0,
                "matched_control": True,
            },
            "controls_verified_by_b1": True,
            "b1_control_snapshot": {
                "controls_verified_by_b1": True,
                "matched_lambda0_control_b1": True,
                "new_levels_not_re_ranking_b1": True,
                "reachable_winner_positive_control_b1": True,
                "s3_guidance_exercised_b1": True,
            },
            "positive_control_passed": True,
            "reachable_winner_positive_control": True,
            "guidance_exercised": True,
            "n_headroom_games": 24,
            "min_headroom_games": 5,
            "headroom_floor_met": True,
            "new_levels_not_re_ranking": True,
            "new_levels_not_in_bare_pool": [],
            "winners_newly_entering_pool_delta": 0.0,
            "winners_newly_entering_pool_delta_ci95": [0.0, 0.0],
            "ci_includes_zero": True,
            "ci_excludes_zero": False,
            "reason": "no_generation_lift_ci_includes_zero",
        },
        "silent_bug_audit": {
            "s3_controls_verified": True,
            "s3_guidance_exercised": True,
            "trusted_nulls": [
                "experiment_4821_structural_energy_s3_generation_lift",
                "experiment_4822_levelup_attempt",
                "experiment_4824_heldout_first_win_readiness",
            ],
        },
        "readiness": {
            "s3_verdict": "bounded_no_generation_lift",
            "s4_authorized": False,
            "structural_energy_direction": "bounded_at_real_offline_discriminator_no_live_value",
            "ready_for_operator_submit": False,
        },
        "heldout_readiness": {
            "heldout_first_win_rate": 0.04,
            "first_win_baseline": 0.04,
            "heldout_first_win_delta_vs_baseline": 0.0,
            "decision": "flat_null_no_readiness_gain",
        },
    }


def _write_repo_fixture(
    root: Path,
    *,
    active_milestone: str = "2026.06.445",
    next_present: bool = False,
    registry_total: int = 65,
    capstone_present: bool = True,
    s3_present: bool = True,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# test\n", encoding="utf-8")
    roadmap_text = (
        f"milestone: {active_milestone}\n"
        "theme: energy program CONCLUDED; refocus L1-FIRST-CONTACT generation wall\n"
        "tasks:\n"
        "  - id: exp4830-phase0\n"
        "    deliverable: results/experiment_4830_archive_444_activate_445.json\n"
    )
    (root / "research-roadmap.yaml").write_text(roadmap_text, encoding="utf-8")
    if next_present:
        (root / "research-roadmap-next.yaml").write_text(
            roadmap_text.replace(active_milestone, "2026.06.445", 1),
            encoding="utf-8",
        )
    registry = root / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        "schema_version: 1\n"
        "updated: '2026-06-26'\n"
        f"reproducible_total_levels: {registry_total}\n",
        encoding="utf-8",
    )
    spec = root / "openspec" / "capabilities" / "capstone" / "spec.md"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text("REQ-CAPSTONE-4830\n", encoding="utf-8")
    if capstone_present:
        _write_json(root / "results" / "experiment_4829_capstone_v444.json", _capstone_4829())
    if s3_present:
        _write_json(root / "results" / "experiment_4821_structural_energy_s3_generation_lift.json", _s3_4821())


def _artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4830_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4830: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4830_restores_next_and_records_energy_concluded(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4830: active `.445` records S3 null and restores next YAML."""

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
        "complete_444_archived_445_activated_already_active_energy_program_concluded"
    )
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.444",
        "activated_milestone": "2026.06.445",
        "active_milestone_confirmed": True,
        "activation_state": "already_activated_by_conductor",
        "archive_state": "archive_noop_or_already_recorded",
    }
    assert artifact["energy_program_concluded"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["reproducible_total_levels"] == 65
    assert artifact["poison_test_resolved"] == {
        "resolved": True,
        "current_gate_passed": True,
        "poison_tests": [],
        "action": "no_poison_observed_current_gate_green",
    }

    close = artifact["close_state_444"]
    assert close["capstone_honest_verdict"] == "complete_s3_bounded_no_generation_lift_capstone_ready"
    assert close["reproducible_total_levels"] == 65
    energy = close["energy_close_state"]
    assert energy["s3_verdict"] == "bounded_no_generation_lift"
    assert energy["s3_honest_verdict"] == "complete_structural_energy_s3_bounded_no_generation_lift"
    assert energy["energy_program_concluded"] is True
    assert energy["s4_moot"] is True
    assert energy["adds_live_arc_value"] is False
    assert energy["winners_newly_entering_pool_delta"] == 0.0
    assert energy["winners_newly_entering_pool_delta_ci95"] == [0.0, 0.0]
    assert energy["lambda0_control"]["lambda"] == 0.0
    assert energy["n_headroom_games"] == 24
    assert energy["min_headroom_games"] == 5
    assert energy["positive_control_passed"] is True
    assert energy["live_path_reachable"] is True
    assert energy["controls_verified_by_b1"] is True

    refocus = artifact["v445_refocus"]
    assert refocus["wall"] == "L1-FIRST-CONTACT"
    assert refocus["generic_first_win_rate"] == 0.04
    assert refocus["generic_first_win_fraction"] == "1/25"
    assert refocus["planner_must_not_repropose_energy_stages"] is True
    assert refocus["headline_task_id"] == "exp4831-a1"
    assert artifact["leaderboard_submission"] is False
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4830_can_activate_literal_next_roadmap(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4830: present next roadmap activates onto active YAML."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.444", next_present=True)

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=3.0,
        now_s=3.4,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8").startswith(
        "milestone: 2026.06.445"
    )
    assert artifact["transition"]["activation_state"] == "activated_from_research_roadmap_next"
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["activation_attempted"] is True
    assert artifact["honest_verdict"] == (
        "complete_444_archived_445_activated_from_next_energy_program_concluded"
    )


def test_scenario_capstone_4830_blockers_and_poison_signature_are_explicit(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4830-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.444", next_present=False)

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
    assert artifact["energy_program_concluded"] is False
    assert artifact["close_state_444"] == {}
    assert artifact["v445_refocus"] == {}

    checks = _artifact(tmp_path / "good")["preconditions_checked"]
    assert mod._first_blocker(checks) is None

    for key, expected in {
        "agents_md": "missing_agents_md",
        "codex_or_opencode_md": "missing_codex_or_opencode_md",
        "capstone_spec": "missing_capstone_spec_req_4830",
        "registry": "arc_solve_registry",
        "capstone_4829": "missing_experiment_4829_capstone_v444",
        "s3_4821": "missing_experiment_4821_structural_energy_s3_generation_lift",
    }.items():
        bad = copy.deepcopy(checks)
        bad[key]["available"] = False
        if key == "capstone_spec":
            bad[key]["has_req_4830"] = False
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
    active_bad["active_research_roadmap_yaml"]["milestone"] = "2026.06.444"
    assert mod._first_blocker(active_bad) == "research_roadmap_445_unavailable"

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
            "id": "test_expected_energy_s4_followup",
            "reason": "single-failure smart-subset signature matches a stale transition expectation",
            "action": "blocked_for_fix_or_quarantine_before_tail_continues",
        }
    ]


def test_scenario_capstone_4830_field_principle_validation_fails_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4830-FIELD-PRINCIPLES: schema drift fails loudly."""

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
    blocked["close_state_444"] = {"fabricated": True}
    blocked["reproducibility_checksum"] = "sha256:" + mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    wrong_total = copy.deepcopy(valid)
    wrong_total["reproducible_total_levels"] = 64
    with pytest.raises(ValueError, match="registry total"):
        mod.validate_artifact(wrong_total)

    wrong_energy = copy.deepcopy(valid)
    wrong_energy["energy_program_concluded"] = False
    with pytest.raises(ValueError, match="energy program"):
        mod.validate_artifact(wrong_energy)

    for field, value in {
        "s3_verdict": "generation_win_s4_authorized",
        "energy_program_concluded": False,
        "s4_moot": False,
        "adds_live_arc_value": True,
        "winners_newly_entering_pool_delta": 0.1,
        "winners_newly_entering_pool_delta_ci95": [0.1, 0.2],
        "n_headroom_games": 4,
        "min_headroom_games": 6,
        "positive_control_passed": False,
        "live_path_reachable": False,
        "controls_verified_by_b1": False,
    }.items():
        wrong = copy.deepcopy(valid)
        wrong["close_state_444"]["energy_close_state"][field] = value
        with pytest.raises(ValueError, match="S3"):
            mod.validate_artifact(wrong)

    wrong_refocus = copy.deepcopy(valid)
    wrong_refocus["v445_refocus"]["wall"] = "S4-energy"
    with pytest.raises(ValueError, match="L1"):
        mod.validate_artifact(wrong_refocus)

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
        active_info={"available": True, "parses": True, "milestone": "2026.06.445"},
    )
    assert restored is False
    assert restore_error

    activation_error_root = tmp_path / "activation_error"
    activation_error_root.mkdir()
    (activation_error_root / "research-roadmap.yaml").mkdir()
    (activation_error_root / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.06.445\n",
        encoding="utf-8",
    )
    activated, activation_error = mod._activate_next_roadmap(
        activation_error_root,
        next_info={"available": True, "parses": True, "milestone": "2026.06.445"},
    )
    assert activated is False
    assert activation_error

    fallback_capstone = _capstone_4829()
    del fallback_capstone["s3_structural_energy_verdict"]["winners_newly_entering_pool_delta"]
    del fallback_capstone["s3_structural_energy_verdict"]["winners_newly_entering_pool_delta_ci95"]
    fallback_energy = mod._energy_close_state(fallback_capstone, _s3_4821())
    assert fallback_energy["winners_newly_entering_pool_delta"] == 0.0
    assert fallback_energy["winners_newly_entering_pool_delta_ci95"] == [0.0, 0.0]

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

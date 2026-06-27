"""Tests for Exp 4860 `.447` archive / `.448` activation record.

Spec refs: REQ-CAPSTONE-4860, SCENARIO-CAPSTONE-4860,
SCENARIO-CAPSTONE-4860-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4860-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4860_archive_447_activate_448 as mod


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
        stdout="142 passed in 8.0s",
        stderr="",
    )


def _red_poison_smart_subset(_root: Path) -> mod.CommandResult:
    return mod.CommandResult(
        command=["pytest", "smart-subset"],
        exit_code=1,
        stdout="1 failed, 141 passed in 8.2s",
        stderr="test_stale_expected_coverage_vocab_lever still expects coverage expansion",
    )


def _per_game_coverage() -> JsonDict:
    return {
        "cd82": {
            "bucket": "NEVER_ENUMERATED",
            "matched_winning_prefix_len": 1,
            "winning_prefix_len": 5,
            "reached_l1_win": False,
        },
        "cn04": {
            "bucket": "NEVER_ENUMERATED",
            "matched_winning_prefix_len": 2,
            "winning_prefix_len": 13,
            "reached_l1_win": False,
        },
        "lp85": {
            "bucket": "COVERED",
            "matched_winning_prefix_len": 0,
            "winning_prefix_len": 5,
            "reached_l1_win": True,
        },
        "ls20": {
            "bucket": "NEVER_ENUMERATED",
            "matched_winning_prefix_len": 2,
            "winning_prefix_len": 13,
            "reached_l1_win": False,
        },
        "m0r0": {
            "bucket": "NEVER_ENUMERATED",
            "matched_winning_prefix_len": 3,
            "winning_prefix_len": 15,
            "reached_l1_win": False,
        },
        "r11l": {
            "bucket": "NEVER_ENUMERATED",
            "matched_winning_prefix_len": 1,
            "winning_prefix_len": 4,
            "reached_l1_win": False,
        },
        "sk48": {
            "bucket": "NEVER_ENUMERATED",
            "matched_winning_prefix_len": 4,
            "winning_prefix_len": 14,
            "reached_l1_win": False,
        },
        "sp80": {
            "bucket": "NEVER_ENUMERATED",
            "matched_winning_prefix_len": 2,
            "winning_prefix_len": 4,
            "reached_l1_win": False,
        },
        "su15": {
            "bucket": "NEVER_ENUMERATED",
            "matched_winning_prefix_len": 1,
            "winning_prefix_len": 7,
            "reached_l1_win": False,
        },
        "wa30": {
            "bucket": "NEVER_ENUMERATED",
            "matched_winning_prefix_len": 3,
            "winning_prefix_len": 33,
            "reached_l1_win": False,
        },
    }


def _a1_4851() -> JsonDict:
    return {
        "experiment": "experiment_4851_generation_coverage_diagnostic",
        "experiment_id": 4851,
        "honest_verdict": "complete_generation_wall_never_enumerated_dominant",
        "dominant_bucket": "NEVER_ENUMERATED",
        "bucket_counts": {"COVERED": 1, "NEVER_ENUMERATED": 9},
        "per_game_coverage": _per_game_coverage(),
        "positive_control_game": "tu93",
        "positive_control_covered": True,
        "positive_control_coverage": {
            "game": "tu93",
            "bucket": "COVERED",
            "matched_winning_prefix_len": 18,
            "winning_prefix_len": 18,
            "reached_l1_win": True,
        },
        "proposer_blind_to_banked_answer": True,
        "n_games_measured": 10,
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _b1_4855() -> JsonDict:
    return {
        "experiment": "experiment_4855_generation_diagnostic_audit",
        "experiment_id": 4855,
        "honest_verdict": "complete_a1_generation_diagnostic_audited",
        "a1_genuinely_diagnostic": True,
        "proposer_blind_confirmed": True,
        "positive_control_confirmed": True,
        "buckets_match_claim": True,
        "source_dominant_bucket": "NEVER_ENUMERATED",
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _capstone_4859() -> JsonDict:
    return {
        "experiment": "experiment_4859_capstone_v447",
        "experiment_id": 4859,
        "capstone_ready": True,
        "honest_verdict": "complete_a1_generation_wall_never_enumerated_capstone_ready",
        "reproducible_total_levels": 65,
        "a1_generation_wall_verdict": {
            "source": "A1",
            "experiment_id": 4851,
            "b1_experiment_id": 4855,
            "upstream_honest_verdict": "complete_generation_wall_never_enumerated_dominant",
            "b1_honest_verdict": "complete_a1_generation_diagnostic_audited",
            "verdict": "generation_wall_never_enumerated",
            "dominant_bucket": "NEVER_ENUMERATED",
            "bucket_counts": {"COVERED": 1, "NEVER_ENUMERATED": 9},
            "n_games_measured": 10,
            "b1_trusted": True,
            "per_game_coverage": _per_game_coverage(),
            "positive_control_game": "tu93",
            "positive_control_covered": True,
            "positive_control_coverage": _a1_4851()["positive_control_coverage"],
            "trust_checks": {
                "a1_genuinely_diagnostic": True,
                "proposer_blind_confirmed": True,
                "positive_control_confirmed_by_b1": True,
                "positive_control_covered": True,
                "buckets_match_claim": True,
            },
        },
        "scored_lever_state": {
            "level_up_banked": False,
            "heldout_first_win_rate": 0.04,
            "live_agent_ran": False,
            "submission_package_ready": True,
        },
    }


def _write_repo_fixture(
    root: Path,
    *,
    active_milestone: str = "2026.06.448",
    next_present: bool = True,
    registry_total: int = 65,
    capstone_present: bool = True,
    a1_present: bool = True,
    b1_present: bool = True,
    roadmap_v448_present: bool = True,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# test\n", encoding="utf-8")
    active_text = f"milestone: {active_milestone}\ntasks: []\n"
    next_text = (
        "milestone: 2026.06.448\n"
        "theme: wall is GUIDANCE/assembly, not coverage\n"
        "tasks:\n"
        "  - id: exp4860-phase0\n"
        f"    deliverable: {mod.RESULT_RELATIVE_PATH}\n"
    )
    (root / "research-roadmap.yaml").write_text(active_text, encoding="utf-8")
    if next_present:
        (root / "research-roadmap-next.yaml").write_text(next_text, encoding="utf-8")
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
    spec.write_text("REQ-CAPSTONE-4860\n", encoding="utf-8")
    roadmap_v448 = root / "openspec" / "change-proposals" / "research-roadmap-v448.md"
    roadmap_v448.parent.mkdir(parents=True, exist_ok=True)
    if roadmap_v448_present:
        roadmap_v448.write_text(
            "NEVER_ENUMERATED dominant. "
            "complete: macro_horizon_collapse_empirical_null_guidance_not_depth. "
            "complete: click_heatmap_generator_premise_falsified_guidance_not_coverage. "
            "Energy CONCLUDED; exploration-prior CLOSED; perception-from-grid null.\n",
            encoding="utf-8",
        )
    if capstone_present:
        _write_json(root / "results" / "experiment_4859_capstone_v447.json", _capstone_4859())
    if a1_present:
        _write_json(
            root / "results" / "experiment_4851_generation_coverage_diagnostic.json",
            _a1_4851(),
        )
    if b1_present:
        _write_json(
            root / "results" / "experiment_4855_generation_diagnostic_audit.json",
            _b1_4855(),
        )


def _artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.2,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4860_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4860: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4860_activates_and_records_guidance_wall(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4860: `.448` records the true `.447` close-state."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.447", next_present=True)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=2.0,
        now_s=2.4,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(artifact)
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8").startswith(
        "milestone: 2026.06.448"
    )
    assert artifact["honest_verdict"] == (
        "complete_447_archived_448_activated_from_next_guidance_assembly_recorded"
    )
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.447",
        "activated_milestone": "2026.06.448",
        "active_milestone_confirmed": True,
        "activation_state": "activated_from_research_roadmap_next",
        "archive_state": "archive_noop_or_already_recorded",
    }
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["a447_generation_wall_never_enumerated"] is True
    assert artifact["wall_is_guidance_not_coverage"] is True
    assert artifact["energy_program_concluded"] is True
    assert artifact["exploration_prior_class_closed"] is True
    assert artifact["reproducible_total_levels"] == 65
    assert artifact["leaderboard_submission"] is False
    assert artifact["poison_test_resolved"] == {
        "resolved": True,
        "current_gate_passed": True,
        "poison_tests": [],
        "action": "no_poison_observed_current_gate_green",
    }

    close = artifact["close_state_447"]
    a1 = close["a1_generation_wall_verdict"]
    assert close["capstone_honest_verdict"] == (
        "complete_a1_generation_wall_never_enumerated_capstone_ready"
    )
    assert a1["dominant_bucket"] == "NEVER_ENUMERATED"
    assert a1["bucket_counts"] == {"COVERED": 1, "NEVER_ENUMERATED": 9}
    assert a1["b1_trusted"] is True
    assert a1["lp85_covered"] is True
    assert a1["tu93_positive_control_covered"] is True
    assert a1["winning_prefix_never_assembled"] is True
    assert a1["never_enumerated_matched_prefix_range"] == [1, 4]
    assert a1["never_enumerated_winning_prefix_len_range"] == [4, 33]
    assert close["retired_coverage_levers"] == [
        "complete: macro_horizon_collapse_empirical_null_guidance_not_depth",
        "complete: click_heatmap_generator_premise_falsified_guidance_not_coverage",
    ]
    assert close["wall_is_guidance_not_coverage"] is True

    frontier = artifact["v448_frontier"]
    assert frontier["root_cause"] == "guidance_assembly_not_coverage"
    assert frontier["planner_must_not_repropose_coverage_vocabulary_levers"] is True
    assert frontier["planner_must_not_reopen_energy_program"] is True
    assert frontier["planner_must_not_repropose_exploration_or_perception_from_grid"] is True
    assert frontier["headline_fork"] == "guidance_gap_vs_world_model_inducer_ceiling"
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4860_blocked_literal_next_still_records_close_state(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4860-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.448", next_present=False)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=3.0,
        now_s=3.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_next_yaml"
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["passed"] is False
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["error_type"] == (
        "FileNotFoundError"
    )
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is None
    assert artifact["a447_generation_wall_never_enumerated"] is True
    assert artifact["wall_is_guidance_not_coverage"] is True
    assert artifact["energy_program_concluded"] is True
    assert artifact["exploration_prior_class_closed"] is True
    assert artifact["reproducible_total_levels"] == 65
    assert artifact["close_state_447"]["a1_generation_wall_verdict"]["b1_trusted"] is True

    checks = _artifact(tmp_path / "good")["preconditions_checked"]
    assert mod._first_blocker(checks) is None
    assert mod._activate_next_roadmap(tmp_path / "good", next_info={}) == (False, "")
    assert mod._bucket_counts({"a": {"bucket": "COVERED"}, "b": {"bucket": "COVERED"}, "c": {}}) == {
        "COVERED": 2
    }

    for key, expected in {
        "agents_md": "missing_agents_md",
        "codex_or_opencode_md": "missing_codex_or_opencode_md",
        "capstone_spec": "missing_capstone_spec_req_4860",
        "registry": "arc_solve_registry",
        "capstone_4859": "missing_experiment_4859_capstone_v447",
        "a1_4851": "missing_experiment_4851_generation_coverage_diagnostic",
        "b1_4855": "missing_experiment_4855_generation_diagnostic_audit",
        "roadmap_v448": "missing_research_roadmap_v448",
    }.items():
        bad = copy.deepcopy(checks)
        bad[key]["available"] = False
        if key == "capstone_spec":
            bad[key]["has_req_4860"] = False
        assert mod._first_blocker(bad) == expected

    offline_bad = copy.deepcopy(checks)
    offline_bad["offline_arcade"]["passed"] = False
    assert mod._first_blocker(offline_bad) == "offline_arcade"

    activation_bad = copy.deepcopy(checks)
    activation_bad["research_roadmap_next_yaml"]["activation_error"] = "permission denied"
    assert mod._first_blocker(activation_bad) == "research_roadmap_activation_error"

    active_bad = copy.deepcopy(checks)
    active_bad["active_research_roadmap_yaml"]["milestone"] = "2026.06.447"
    assert mod._first_blocker(active_bad) == "research_roadmap_448_unavailable"

    registry_bad = copy.deepcopy(checks)
    registry_bad["registry"]["reproducible_total_levels"] = 64
    assert mod._first_blocker(registry_bad) == "arc_solve_registry_total_levels_not_65"

    bad_yaml_root = tmp_path / "bad-yaml"
    _write_repo_fixture(bad_yaml_root)
    (bad_yaml_root / "research-roadmap-next.yaml").write_text("milestone: [\n", encoding="utf-8")
    assert mod._precondition_next_yaml(bad_yaml_root)["error_type"] == "YAMLError"

    wrong_milestone_root = tmp_path / "wrong-milestone"
    _write_repo_fixture(wrong_milestone_root)
    (wrong_milestone_root / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.06.449\n",
        encoding="utf-8",
    )
    wrong_next = mod._precondition_next_yaml(wrong_milestone_root)
    assert wrong_next["error_type"] == "MilestoneMismatch"
    assert wrong_next["exit_code"] == 1

    blocked = mod._blocked_artifact(
        reason="unit_test",
        preconditions_checked=checks,
        poison_test_resolved={"resolved": False},
        duration_s=0.1,
        cited_upstream_artifacts=[],
        close_state_447={},
        v448_frontier={},
        reproducible_total_levels=None,
    )
    assert blocked["honest_verdict"] == "blocked_unit_test"
    assert mod.validate_artifact(blocked) is None


def test_scenario_capstone_4860_poison_signature_and_validation_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4860-FIELD-PRINCIPLES: schema drift fails loudly."""

    valid = _artifact(tmp_path / "valid")

    bad_smart = mod.build_artifact(
        tmp_path / "valid",
        started_s=4.0,
        now_s=4.2,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_red_poison_smart_subset,
    )
    assert bad_smart["honest_verdict"] == "blocked_smart_subset_pretest_gate"
    assert bad_smart["poison_test_resolved"]["poison_tests"] == [
        {
            "id": "test_stale_expected_coverage_vocab_lever",
            "reason": "single-failure smart-subset signature may be a stale transition expectation",
            "action": "blocked_for_fix_or_quarantine_before_tail_continues",
        }
    ]

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

    wrong_total = copy.deepcopy(valid)
    wrong_total["reproducible_total_levels"] = 64
    with pytest.raises(ValueError, match="registry total"):
        mod.validate_artifact(wrong_total)

    for field in (
        "a447_generation_wall_never_enumerated",
        "wall_is_guidance_not_coverage",
        "energy_program_concluded",
        "exploration_prior_class_closed",
    ):
        wrong = copy.deepcopy(valid)
        wrong[field] = False
        with pytest.raises(ValueError, match=field):
            mod.validate_artifact(wrong)

    wrong_a1 = copy.deepcopy(valid)
    wrong_a1["close_state_447"]["a1_generation_wall_verdict"]["dominant_bucket"] = "COVERED"
    with pytest.raises(ValueError, match="generation wall"):
        mod.validate_artifact(wrong_a1)

    wrong_frontier = copy.deepcopy(valid)
    wrong_frontier["v448_frontier"]["planner_must_not_repropose_coverage_vocabulary_levers"] = False
    with pytest.raises(ValueError, match="v448 frontier"):
        mod.validate_artifact(wrong_frontier)

"""Tests for Exp 4522 `.417` archive / `.418` activation.

Spec refs: REQ-CAPSTONE-4522, SCENARIO-CAPSTONE-4522,
SCENARIO-CAPSTONE-4522-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4522_archive_417_activate_418 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _capstone() -> JsonDict:
    return {
        "honest_verdict": (
            "complete: v417_none_clean_equal_solve_rate_median_actions_7760_vs_7760_heldout_0.143"
        ),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "median_actions_baseline": 7760.0,
        "median_actions_best_lever": {
            "lever": "none_clean_equal_solve_rate",
            "median_actions": 7760.0,
            "reason": "no_clean_lever_beat_7760_at_equal_or_better_solve_rate",
        },
        "per_lever_scorecard": [
            {
                "lever": "A1_prune",
                "status": "no_clean_equal_solve_win",
                "median_actions": 7766.0,
                "equal_or_better_solve_rate": False,
            },
            {
                "lever": "A2_imitation",
                "status": "no_clean_equal_solve_win",
                "median_actions": 7733.0,
                "equal_or_better_solve_rate": False,
            },
            {
                "lever": "A3_adaptive_budget",
                "status": "excluded_flagged_adversarial",
                "median_actions": None,
            },
            {
                "lever": "A4_lazy_best_first",
                "status": "no_clean_equal_solve_win",
                "selected_value_weight": 0.0,
                "core_solves_preserved": False,
            },
        ],
        "flagged_artifacts_excluded": [
            {"artifact_key": "A3_adaptive_budget", "reason": "flagged_adversarial"},
            {"artifact_key": "A6_integration", "reason": "flagged_adversarial"},
        ],
        "level_up_context": {
            "level_up_banked": True,
            "target_game": "m0r0",
            "reproduced_levels": 2,
            "offline_reproduced": True,
        },
        "action_efficiency_decision": {
            "baseline_median_actions": 7760.0,
            "beats_7760_at_equal_solve_rate": False,
            "winning_lever": None,
        },
        "integrated_scorecard": {
            "lever": "A6_integration",
            "status": "excluded_flagged_adversarial",
        },
        "variant_transfer_context": {
            "status": "scoreboard_context",
            "heldout_solve_rate": 0.1428571429,
            "variant_transfer_rate": 0.28,
        },
        "submission_readiness_decision": {
            "decision": "ready_for_operator_submit",
            "submitted_to_leaderboard": False,
        },
        "reproducible_total_levels": 47,
        "cited_upstream_artifacts": [
            {"artifact_key": "A6_integration", "path": "results/experiment_4516_integration_8game_gate.json"}
        ],
    }


def _write_repo_fixture(root: Path) -> None:
    (root / "research-roadmap.yaml").write_text(
        "milestone: 2026.06.418\n"
        "tasks:\n"
        "  - id: exp4522-phase0\n"
        "    deliverable: results/experiment_4522_archive_417_activate_418.json\n",
        encoding="utf-8",
    )
    (root / "research-complete.yaml").write_text(
        "milestones:\n"
        "- id: 2026.06.417\n"
        "  finding: prior roadmap archived by conductor\n",
        encoding="utf-8",
    )
    registry = root / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        "schema_version: 1\n"
        "updated: '2026-06-20'\n"
        "reproducible_total_levels: 48\n"
        "reproducible_total_games: 24\n"
        "provisional_total_levels: 1\n",
        encoding="utf-8",
    )
    _write_json(root / "results" / "experiment_4521_capstone_v417.json", _capstone())


def _green_smart_subset(_root: Path) -> mod.CommandResult:
    return mod.CommandResult(
        command=["pytest", "smart-subset"],
        exit_code=0,
        stdout="green",
        stderr="",
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


def test_req_capstone_4522_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4522: OpenSpec declares the transition artifact contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4522" in spec
    assert "SCENARIO-CAPSTONE-4522" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "reproducible_total_levels" in spec
    assert "A3 adaptive false-win flagged" in spec


def test_scenario_capstone_4522_records_true_close_state_when_already_activated(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4522: already-activated `.418` still writes the close-state."""

    _write_repo_fixture(tmp_path)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    out_path = tmp_path / mod.OUTPUT_REL_PATH
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.417",
        "activated_milestone": "2026.06.418",
        "active_milestone_confirmed": True,
        "activation_state": "already_active_roadmap_next_consumed",
        "archive_state": "research_complete_contains_2026.06.417",
    }
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["available"] is False
    assert artifact["preconditions_checked"]["active_research_roadmap_yaml"]["milestone"] == "2026.06.418"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["exit_code"] == 0

    close = artifact["close_state_417"]
    assert close["reproducible_total_levels"] == 48
    assert close["score_lever_scorecard"]["A1_prune"]["decision"] == "null_solve_rate_guard_failed"
    assert close["score_lever_scorecard"]["A2_prior"]["decision"] == "null_solve_rate_guard_failed"
    assert close["score_lever_scorecard"]["A3_adaptive"]["false_win_flags"] == [
        "lever_inert",
        "commit_count_0",
        "metric_mismatch",
    ]
    assert close["score_lever_scorecard"]["A4_value_weight"]["selected_value_weight"] == 0.0
    assert close["score_lever_scorecard"]["A5_m0r0_L2"]["banked"] is True
    assert close["score_lever_scorecard"]["A6_integration"]["nav_tax"] == {
        "reset_replays": 1546,
        "forward_walk_hits": 6,
    }
    assert close["net_417"]["solve_capability_grew"] is True
    assert close["net_417"]["action_efficiency_moved"] is False
    assert close["net_417"]["submitted_config"] == "unchanged"
    assert close["net_417"]["gate_baseline_median_actions"] == 7760.0
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4522_blocks_without_fabricating_missing_capstone(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4522: missing required close-state input blocks honestly."""

    _write_repo_fixture(tmp_path)
    (tmp_path / "results" / "experiment_4521_capstone_v417.json").unlink()

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=2.0,
        now_s=2.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_missing_experiment_4521_capstone_v417"
    assert artifact["preconditions_checked"]["capstone_4521"]["available"] is False
    assert artifact["close_state_417"] == {}
    assert artifact["transition"]["active_milestone_confirmed"] is False
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4522_records_next_roadmap_activation_state(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4522: an extant next roadmap is recorded as activation input."""

    _write_repo_fixture(tmp_path)
    (tmp_path / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.06.418\ntasks: []\n",
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


def test_scenario_capstone_4522_precondition_blockers_are_classified(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4522: each required precondition has an honest blocked reason."""

    artifact = _artifact(tmp_path)
    preconditions = artifact["preconditions_checked"]

    active_bad = copy.deepcopy(preconditions)
    active_bad["active_research_roadmap_yaml"]["milestone"] = "2026.06.417"
    active_bad["research_roadmap_next_yaml"]["available"] = False
    active_bad["research_roadmap_next_yaml"]["parses"] = False
    assert mod._first_blocker(active_bad) == "research_roadmap_418_unavailable"

    offline_bad = copy.deepcopy(preconditions)
    offline_bad["offline_arcade"]["available"] = False
    assert mod._first_blocker(offline_bad) == "offline_arcade"

    smart_bad = copy.deepcopy(preconditions)
    smart_bad["smart_subset_pretest_gate"]["passed"] = False
    assert mod._first_blocker(smart_bad) == "smart_subset_pretest_gate"

    registry_bad = copy.deepcopy(preconditions)
    registry_bad["registry"]["available"] = False
    assert mod._first_blocker(registry_bad) == "arc_solve_registry"


def test_scenario_capstone_4522_parse_helpers_are_defensive(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4522: malformed inputs are detected instead of fabricated."""

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
    assert mod._registry_total_levels(list_yaml) is None

    list_json = tmp_path / "list.json"
    list_json.write_text("[1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod._read_json(list_json)

    close = mod._close_state_417({"per_lever_scorecard": []}, 48)
    assert close["score_lever_scorecard"]["A1_prune"]["source_status"] is None


def test_scenario_capstone_4522_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4522-FIELD-PRINCIPLES: schema drift fails loudly."""

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
    blocked["close_state_417"] = {"fabricated": True}
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    inactive = copy.deepcopy(valid)
    inactive["transition"]["active_milestone_confirmed"] = False
    with pytest.raises(ValueError, match="active .418"):
        mod.validate_artifact(inactive)

    no_levels = copy.deepcopy(valid)
    del no_levels["close_state_417"]["reproducible_total_levels"]
    with pytest.raises(ValueError, match="reproducible_total_levels"):
        mod.validate_artifact(no_levels)

    wrong_net = copy.deepcopy(valid)
    wrong_net["close_state_417"]["net_417"]["action_efficiency_moved"] = True
    with pytest.raises(ValueError, match="capability growth"):
        mod.validate_artifact(wrong_net)

    bad_checksum_prefix = copy.deepcopy(valid)
    bad_checksum_prefix["reproducibility_checksum"] = "not-a-checksum"
    with pytest.raises(ValueError, match="sha256-prefixed"):
        mod.validate_artifact(bad_checksum_prefix)

    bad_checksum_value = copy.deepcopy(valid)
    bad_checksum_value["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad_checksum_value)

"""Tests for Exp 4748 `.436` archive / `.437` activation record.

Spec refs: REQ-CAPSTONE-4748, SCENARIO-CAPSTONE-4748,
SCENARIO-CAPSTONE-4748-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4748-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4748_archive_436_activate_437 as mod


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
        stdout="123 passed, 1 warning",
        stderr="",
    )


def _red_poison_smart_subset(_root: Path) -> mod.CommandResult:
    return mod.CommandResult(
        command=["pytest", "smart-subset"],
        exit_code=1,
        stdout="1 failed, 123 passed, 1 warning",
        stderr="test_expected_stale_honest_verdict failed against now-correct complete verdict",
    )


def _capstone_4747() -> JsonDict:
    return {
        "honest_verdict": "complete: no_bridge_crossed_capability_unchanged",
        "bridge_crossed_for_solve": False,
        "reproducible_total_levels": 64,
        "reproducible_total_levels_delta": 0,
        "a1_goal_energy_result": {
            "arms_non_degenerate": True,
            "banked": False,
            "baseline_first_win": 0.0,
            "beat_baseline_by_0_05": False,
            "crossed": False,
            "deepened_to_l2": False,
            "generated": False,
            "goal_energy_first_win": 0.0,
            "goal_energy_vs_baseline_delta": 0.0,
            "included_in_headline": True,
            "offline_reproduced": False,
            "reason": "goal_energy_real_non_degenerate_zero_lift_null",
            "reproduced_levels": 1,
            "solve_provenance": "live_agent_self_discovery",
        },
        "a2_energy_qd_result": {
            "arms_non_degenerate": True,
            "banked": False,
            "crossed": False,
            "deepened_to_l2": False,
            "energy_qd_first_win": 0.0,
            "energy_qd_vs_naive_delta": 0.0,
            "generated": False,
            "generated_winner_where_naive_missed": False,
            "included_in_headline": True,
            "naive_search_first_win": 0.0,
            "novel_candidates_generated": 8,
            "offline_reproduced": False,
            "reason": "energy_qd_real_non_degenerate_zero_lift_null",
            "reproduced_levels": 1,
            "solve_provenance": "live_agent_self_discovery",
            "target_game": "energy_qd",
        },
        "a3_banked_level": {
            "banked": False,
            "crossed": False,
            "generated": False,
            "new_levels_banked": 0,
            "reason": "no_clean_registry_bank",
            "reproduced_levels": 2,
            "reproducible_total_levels_after": 64,
            "reproducible_total_levels_before": 64,
            "solve_provenance": "development_proxy",
            "target_game": "re86",
        },
        "headline_decision": {
            "bridge_crossed_for_solve": False,
            "capability_delta": 0,
            "a1_arms_non_degenerate": True,
            "a1_beat_baseline_by_0_05": False,
            "a1_deepened_to_l2": False,
            "a2_generated_winner_where_naive_missed": False,
            "a2_deepened_to_l2": False,
            "a3_banked_64_to_65": False,
        },
        "next_milestone_fallback": {
            "strongest_open_lever": "A2_energy_qd_generation",
            "deferred_reopens": ["P1_go_explore", "P4_subgoal", "A2_active_probe"],
        },
    }


def _write_repo_fixture(
    root: Path,
    *,
    active_milestone: str = "2026.06.437",
    next_present: bool = False,
    registry_total: int = 64,
    capstone_present: bool = True,
    conductor_text: str = "| 2026-06-25 22:48 UTC | Milestone 2026.06.437 activated | OK | 12 tasks queued |\n",
) -> None:
    (root / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# test\n", encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        f"milestone: {active_milestone}\n"
        "tasks:\n"
        "  - id: exp4748-phase0\n"
        "    deliverable: results/experiment_4748_archive_436_activate_437.json\n",
        encoding="utf-8",
    )
    if next_present:
        (root / "research-roadmap-next.yaml").write_text(
            "milestone: 2026.06.437\n"
            "tasks:\n"
            "  - id: exp4748-phase0\n"
            "    deliverable: results/experiment_4748_archive_436_activate_437.json\n",
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
    spec.write_text("REQ-CAPSTONE-4748\n", encoding="utf-8")
    complete = root / "research-complete.yaml"
    complete.write_text("- id: 2026.06.436\n  finding: archived by conductor\n", encoding="utf-8")
    log = root / "ops" / "conductor-log.md"
    log.write_text(conductor_text, encoding="utf-8")
    if capstone_present:
        _write_json(root / "results" / "experiment_4747_capstone_v436.json", _capstone_4747())


def _artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4748_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4748: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4748" in spec
    assert "SCENARIO-CAPSTONE-4748" in spec
    assert "SCENARIO-CAPSTONE-4748-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CAPSTONE-4748-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "A1 goal-energy candidate generation as a non-degenerate zero-lift null" in spec
    assert "ProductWorldModel structured action-effect engine" in spec
    assert "structural-alignment detector" in spec
    for field, provenance in mod.FIELD_PROVENANCE.items():
        assert field in spec
        assert provenance["principle"] in spec


def test_scenario_capstone_4748_records_true_close_state_when_already_activated(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4748: active `.437` allows a complete record without next YAML."""

    _write_repo_fixture(tmp_path)
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
    assert artifact["honest_verdict"] == "complete: archive_436_activate_437_true_close_state_recorded"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.436",
        "activated_milestone": "2026.06.437",
        "active_milestone_confirmed": True,
        "activation_state": "already_activated_by_conductor",
        "archive_state": "research_complete_contains_2026.06.436",
    }
    next_check = artifact["preconditions_checked"]["research_roadmap_next_yaml"]
    assert next_check["available"] is False
    assert next_check["literal_precondition_passed"] is False
    assert next_check["accepted_missing_because_already_active"] is True
    assert artifact["preconditions_checked"]["active_research_roadmap_yaml"]["milestone"] == "2026.06.437"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is True

    poison = artifact["poison_pretest_resolved"]
    assert poison == {
        "resolved": True,
        "current_gate_passed": True,
        "historical_signature_observed": False,
        "historical_signature": "",
        "poison_tests": [],
        "action": "no_poison_observed_current_gate_green",
    }

    close = artifact["close_state_436"]
    assert close["source_capstone_honest_verdict"] == "complete: no_bridge_crossed_capability_unchanged"
    assert close["bridge_crossed_for_solve"] is False
    assert close["reproducible_total_levels"] == 64
    assert close["reproducible_total_levels_delta"] == 0
    assert close["a1_guidance_class_generation"]["reason"] == "goal_energy_real_non_degenerate_zero_lift_null"
    assert close["a1_guidance_class_generation"]["arms_non_degenerate"] is True
    assert close["a1_guidance_class_generation"]["beat_baseline_by_0_05"] is False
    assert close["a2_guidance_class_generation"]["reason"] == "energy_qd_real_non_degenerate_zero_lift_null"
    assert close["a2_guidance_class_generation"]["novel_candidates_generated"] == 8
    assert close["a2_guidance_class_generation"]["generated_winner_where_naive_missed"] is False
    assert close["a3_level_up_guarantee"]["new_levels_banked"] == 0
    assert close["net_436"] == {
        "bridge_crossed_for_solve": False,
        "capability_grew": False,
        "guidance_class_generation_validly_tested": True,
        "registry_total_after": 64,
    }

    assert artifact["v437_pivot"] == {
        "headline": "FIX induction-quality wall",
        "a1_structured_engine": {
            "action": "wire existing ProductWorldModel programmatic experts as the action-effect engine",
            "replaces": "0.12-accurate free-form codex engine",
            "existing_scaffold": "python/carnot/agentic/arc_executable_world_model.py:ProductWorldModel",
        },
        "a2_structural_alignment_detector": {
            "action": "fix perception-grounded structural-alignment detector segmentation and pairing",
            "resolves": "exp4712 over-segmentation: goal_count=42 aligned_piece_count=0",
            "existing_scaffold": "python/carnot/agentic/arc_value_learner.py structural alignment pipeline",
        },
        "retired_retries": ["pure prompt-engineering retry", "retired CNN driver"],
    }
    assert artifact["leaderboard_submission"] is False
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4748_can_activate_literal_next_roadmap(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4748: present next roadmap is activated onto the active YAML."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.436", next_present=True)

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=3.0,
        now_s=3.4,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8").startswith(
        "milestone: 2026.06.437"
    )
    assert artifact["transition"]["activation_state"] == "activated_from_research_roadmap_next"
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["activation_attempted"] is True
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"][
        "accepted_missing_because_already_active"
    ] is False


def test_scenario_capstone_4748_blockers_and_poison_signature_are_explicit(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4748-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.436", next_present=False)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=4.0,
        now_s=4.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_437_unavailable"
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is None
    assert artifact["poison_pretest_resolved"]["resolved"] is False
    assert artifact["close_state_436"] == {}
    assert artifact["v437_pivot"] == {}
    assert (tmp_path / mod.OUTPUT_REL_PATH).exists()

    checks = _artifact(tmp_path)["preconditions_checked"]
    assert mod._first_blocker(checks) is None

    for key, expected in {
        "agents_md": "missing_agents_md",
        "codex_or_opencode_md": "missing_codex_or_opencode_md",
        "capstone_spec": "missing_capstone_spec_req_4748",
        "registry": "arc_solve_registry",
        "capstone_4747": "missing_experiment_4747_capstone_v436",
        "conductor_log": "missing_conductor_log",
    }.items():
        bad = copy.deepcopy(checks)
        bad[key]["available"] = False
        if key == "capstone_spec":
            bad[key]["has_req_4748"] = False
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
        started_s=5.0,
        now_s=5.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_red_poison_smart_subset,
    )
    assert bad_smart["honest_verdict"] == "blocked_smart_subset_pretest_gate"
    assert bad_smart["poison_pretest_resolved"]["poison_tests"] == [
        {
            "id": "test_expected_stale_honest_verdict",
            "reason": "single-failure smart-subset signature matches a stale honest-verdict expectation",
            "action": "blocked_for_fix_or_quarantine_before_tail_continues",
        }
    ]


def test_scenario_capstone_4748_field_principle_validation_fails_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4748-FIELD-PRINCIPLES: schema drift fails loudly."""

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
    blocked["close_state_436"] = {"fabricated": True}
    blocked["reproducibility_checksum"] = "sha256:" + mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    inactive = copy.deepcopy(valid)
    inactive["transition"]["active_milestone_confirmed"] = False
    with pytest.raises(ValueError, match="active .437"):
        mod.validate_artifact(inactive)

    wrong_bridge = copy.deepcopy(valid)
    wrong_bridge["close_state_436"]["bridge_crossed_for_solve"] = True
    with pytest.raises(ValueError, match="bridge"):
        mod.validate_artifact(wrong_bridge)

    wrong_levels = copy.deepcopy(valid)
    wrong_levels["close_state_436"]["reproducible_total_levels"] = 65
    with pytest.raises(ValueError, match="registry"):
        mod.validate_artifact(wrong_levels)

    wrong_a1 = copy.deepcopy(valid)
    wrong_a1["close_state_436"]["a1_guidance_class_generation"]["arms_non_degenerate"] = False
    with pytest.raises(ValueError, match="A1"):
        mod.validate_artifact(wrong_a1)

    wrong_a2 = copy.deepcopy(valid)
    wrong_a2["close_state_436"]["a2_guidance_class_generation"][
        "generated_winner_where_naive_missed"
    ] = True
    with pytest.raises(ValueError, match="A2"):
        mod.validate_artifact(wrong_a2)

    wrong_pivot = copy.deepcopy(valid)
    wrong_pivot["v437_pivot"]["headline"] = "retry prompt engineering"
    with pytest.raises(ValueError, match="v437 pivot"):
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
    assert mod._json_object(tmp_path / "missing.json") == {}
    assert (
        mod._poison_signature(
            mod._transition_log_scope(
                "old row: 1 failed, 91 passed\nMilestone 2026.06.436 activated\nall green"
            )
        )
        == ""
    )
    assert (
        mod._poison_signature(
            mod._transition_log_scope("Milestone 2026.06.436 activated\n1 failed, 86 passed")
        )
        == "1 failed, 86 passed"
    )

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
    assert mod._json_object(list_json) == {}

"""Tests for Exp 4760 `.437` archive / `.438` activation record.

Spec refs: REQ-CAPSTONE-4760, SCENARIO-CAPSTONE-4760,
SCENARIO-CAPSTONE-4760-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4760-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4760_archive_437_activate_438 as mod


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
        stdout="87 passed, 1 warning in 9.05s",
        stderr="",
    )


def _red_poison_smart_subset(_root: Path) -> mod.CommandResult:
    return mod.CommandResult(
        command=["pytest", "smart-subset"],
        exit_code=1,
        stdout="1 failed, 91 passed, 1 warning in 6.55s",
        stderr="test_expected_stale_honest_verdict failed against now-correct honest verdict",
    )


def _capstone_4759() -> JsonDict:
    return {
        "honest_verdict": "success: real_bank_landed_sk48_L2_capstone_complete",
        "bridge_crossed_for_solve": False,
        "reproducible_total_levels": 65,
        "induction_quality_decision": {
            "a1": {
                "present": True,
                "decision": "skipped_flagged_adversarial",
                "beat_0_12_freeform_baseline": None,
                "banked_l2": False,
            },
            "a2": {
                "present": True,
                "decision": "detector_fixed_no_satisfiable_goal_no_bank",
                "goal_predicate_satisfiable": False,
                "l2_plan_reaches_goal": False,
                "offline_reproduced": False,
                "reproduced_levels": 1,
                "banked_l2": False,
            },
            "cleared_induction_quality_wall": False,
        },
        "scorecard": {
            "A1": {"present": True, "decision": "skipped_flagged_adversarial"},
            "A2": {"present": True, "decision": "detector_fixed_no_satisfiable_goal_no_bank"},
            "A3": {
                "present": True,
                "decision": "real_bank_landed",
                "target_game": "sk48",
                "reached_level": 2,
                "new_levels_banked": 1,
                "offline_reproduced": True,
            },
            "A4": {"present": True, "decision": "skipped_flagged_adversarial"},
        },
        "skipped_artifacts": [
            {"experiment_id": 4749, "source": "A1", "reason": "flagged_adversarial"},
            {"experiment_id": 4752, "source": "A4", "reason": "flagged_adversarial"},
        ],
        "submission_package_ready": False,
        "paper_ready": True,
    }


def _write_repo_fixture(
    root: Path,
    *,
    active_milestone: str = "2026.06.438",
    next_present: bool = False,
    registry_total: int = 65,
    capstone_present: bool = True,
    conductor_text: str = "| 2026-06-26 03:40 UTC | Milestone 2026.06.438 activated | OK | 10 tasks queued |\n",
) -> None:
    (root / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# test\n", encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        f"milestone: {active_milestone}\n"
        "tasks:\n"
        "  - id: exp4760-phase0\n"
        "    deliverable: results/experiment_4760_archive_437_activate_438.json\n",
        encoding="utf-8",
    )
    if next_present:
        (root / "research-roadmap-next.yaml").write_text(
            "milestone: 2026.06.438\n"
            "tasks:\n"
            "  - id: exp4760-phase0\n"
            "    deliverable: results/experiment_4760_archive_437_activate_438.json\n",
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
    spec.write_text("REQ-CAPSTONE-4760\n", encoding="utf-8")
    log = root / "ops" / "conductor-log.md"
    log.write_text(conductor_text, encoding="utf-8")
    if capstone_present:
        _write_json(root / "results" / "experiment_4759_capstone_v437.json", _capstone_4759())
    _write_json(
        root / "results" / "experiment_4749_structured_engine_vs_freeform.json",
        {
            "experiment_id": 4749,
            "honest_verdict": "complete_structured_engine_no_improvement_null",
            "structured_engine_non_degenerate": False,
            "structured_heldout_accuracy": 0.5,
            "freeform_heldout_accuracy": 0.0,
            "flagged_adversarial": True,
        },
    )
    _write_json(
        root / "results" / "experiment_4750_structural_alignment_detector_fix.json",
        {
            "experiment_id": 4750,
            "honest_verdict": "complete_detector_fixed_but_no_bank_no_reachable_plan",
            "detector_goal_count": 2,
            "detector_piece_count": 2,
            "detector_raw_goal_count": 42,
            "goal_predicate_satisfiable": False,
            "l2_plan_reaches_goal": False,
            "offline_reproduced": False,
            "reproduced_levels": 1,
        },
    )
    _write_json(
        root / "results" / "experiment_4751_levelup_selfplay.json",
        {
            "experiment_id": 4751,
            "honest_verdict": "success: sk48_L2_offline_reproduced",
            "target_game": "sk48",
            "new_levels_banked": 1,
            "reached_level": 2,
            "offline_reproduced": True,
            "reproducible_total_levels": 65,
        },
    )
    _write_json(
        root / "results" / "experiment_4752_held_out_first_win_readiness.json",
        {
            "experiment_id": 4752,
            "honest_verdict": "complete: held_out_first_win_flat_no_leaderboard_change",
            "first_win_rate_integrated": 0.04,
            "submission_package_ready": False,
            "flagged_adversarial": True,
        },
    )


def _artifact(root: Path) -> JsonDict:
    _write_repo_fixture(root)
    return mod.build_artifact(
        root,
        started_s=1.0,
        now_s=1.25,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )


def test_req_capstone_4760_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4760: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4760" in spec
    assert "SCENARIO-CAPSTONE-4760" in spec
    assert "SCENARIO-CAPSTONE-4760-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CAPSTONE-4760-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, provenance in mod.FIELD_PROVENANCE.items():
        assert field in spec
        assert provenance["principle"] in spec


def test_scenario_capstone_4760_records_true_close_state_when_already_activated(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4760: active `.438` allows a complete record without next YAML."""

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
    assert artifact["honest_verdict"] == "complete_437_archived_438_activated_already_active_true_close_state_recorded"
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.437",
        "activated_milestone": "2026.06.438",
        "active_milestone_confirmed": True,
        "activation_state": "already_activated_by_conductor",
        "archive_state": "archive_noop_or_already_recorded",
    }
    assert artifact["reproducible_total_levels"] == 65
    next_check = artifact["preconditions_checked"]["research_roadmap_next_yaml"]
    assert next_check["available"] is False
    assert next_check["literal_precondition_passed"] is False
    assert next_check["accepted_missing_because_already_active"] is True
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is True

    poison = artifact["poison_test_resolved"]
    assert poison == {
        "resolved": True,
        "current_gate_passed": True,
        "historical_signature_observed": False,
        "historical_signature": "",
        "poison_tests": [],
        "action": "no_poison_observed_current_gate_green",
    }

    close = artifact["close_state_437"]
    assert close["source_capstone_honest_verdict"] == "success: real_bank_landed_sk48_L2_capstone_complete"
    assert close["bridge_crossed_for_solve"] is False
    assert close["reproducible_total_levels"] == 65
    assert close["a1_structured_engine"] == {
        "decision": "skipped_flagged_adversarial",
        "artifact_flagged_adversarial": True,
        "structured_engine_non_degenerate": False,
        "structured_heldout_accuracy": 0.5,
        "freeform_heldout_accuracy": 0.0,
        "forward_claim_status": "quarantined_not_forward_claim",
    }
    assert close["a2_detector_fix"]["goal_predicate_satisfiable"] is False
    assert close["a2_detector_fix"]["l2_plan_reaches_goal"] is False
    assert close["a2_detector_fix"]["offline_reproduced"] is False
    assert close["a2_detector_fix"]["modest_result"] == "detector_fixed_no_satisfiable_goal_no_bank"
    assert close["a3_levelup"]["target_game"] == "sk48"
    assert close["a3_levelup"]["new_levels_banked"] == 1
    assert close["flagged_tasks"] == [
        {"experiment_id": 4749, "source": "A1", "reason": "flagged_adversarial"},
        {"experiment_id": 4752, "source": "A4", "reason": "flagged_adversarial"},
    ]
    assert artifact["v438_pivot"]["headline"] == "oracle-distinct structural energy S0 core-bet probe"
    assert artifact["leaderboard_submission"] is False
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4760_can_activate_literal_next_roadmap(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4760: present next roadmap is activated onto the active YAML."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.437", next_present=True)

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=3.0,
        now_s=3.4,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8").startswith(
        "milestone: 2026.06.438"
    )
    assert artifact["transition"]["activation_state"] == "activated_from_research_roadmap_next"
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["activation_attempted"] is True
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"][
        "accepted_missing_because_already_active"
    ] is False


def test_scenario_capstone_4760_blockers_and_poison_signature_are_explicit(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4760-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.437", next_present=False)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=4.0,
        now_s=4.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_438_unavailable"
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is None
    assert artifact["poison_test_resolved"]["resolved"] is False
    assert artifact["close_state_437"] == {}
    assert artifact["v438_pivot"] == {}
    assert (tmp_path / mod.OUTPUT_REL_PATH).exists()

    checks = _artifact(tmp_path)["preconditions_checked"]
    assert mod._first_blocker(checks) is None

    for key, expected in {
        "agents_md": "missing_agents_md",
        "codex_or_opencode_md": "missing_codex_or_opencode_md",
        "capstone_spec": "missing_capstone_spec_req_4760",
        "registry": "arc_solve_registry",
        "capstone_4759": "missing_experiment_4759_capstone_v437",
        "conductor_log": "missing_conductor_log",
    }.items():
        bad = copy.deepcopy(checks)
        bad[key]["available"] = False
        if key == "capstone_spec":
            bad[key]["has_req_4760"] = False
        assert mod._first_blocker(bad) == expected

    offline_bad = copy.deepcopy(checks)
    offline_bad["offline_arcade"]["available"] = False
    assert mod._first_blocker(offline_bad) == "offline_arcade"

    def _offline_raises() -> bool:
        raise RuntimeError("offline arcade unavailable")

    offline_artifact = mod.build_artifact(
        tmp_path,
        started_s=4.2,
        now_s=4.3,
        offline_arcade_checker=_offline_raises,
        smart_subset_checker=_green_smart_subset,
    )
    assert offline_artifact["honest_verdict"] == "blocked_offline_arcade"
    assert offline_artifact["preconditions_checked"]["offline_arcade"]["error"] == "offline arcade unavailable"

    smart_bad = copy.deepcopy(checks)
    smart_bad["smart_subset_pretest_gate"]["passed"] = False
    assert mod._first_blocker(smart_bad) == "smart_subset_pretest_gate"

    registry_bad = copy.deepcopy(checks)
    registry_bad["registry"]["reproducible_total_levels"] = 64
    assert mod._first_blocker(registry_bad) == "arc_solve_registry_total_levels_not_65"

    bad_smart = mod.build_artifact(
        tmp_path,
        started_s=5.0,
        now_s=5.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_red_poison_smart_subset,
    )
    assert bad_smart["honest_verdict"] == "blocked_smart_subset_pretest_gate"
    assert bad_smart["poison_test_resolved"]["poison_tests"] == [
        {
            "id": "test_expected_stale_honest_verdict",
            "reason": "single-failure smart-subset signature matches a stale honest-verdict expectation",
            "action": "blocked_for_fix_or_quarantine_before_tail_continues",
        }
    ]


def test_scenario_capstone_4760_field_principle_validation_fails_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4760-FIELD-PRINCIPLES: schema drift fails loudly."""

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
    blocked["close_state_437"] = {"fabricated": True}
    blocked["reproducibility_checksum"] = "sha256:" + mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    inactive = copy.deepcopy(valid)
    inactive["transition"]["active_milestone_confirmed"] = False
    with pytest.raises(ValueError, match="active .438"):
        mod.validate_artifact(inactive)

    wrong_total = copy.deepcopy(valid)
    wrong_total["reproducible_total_levels"] = 64
    with pytest.raises(ValueError, match="registry total"):
        mod.validate_artifact(wrong_total)

    wrong_close_total = copy.deepcopy(valid)
    wrong_close_total["close_state_437"]["reproducible_total_levels"] = 64
    with pytest.raises(ValueError, match="registry total"):
        mod.validate_artifact(wrong_close_total)

    wrong_a1 = copy.deepcopy(valid)
    wrong_a1["close_state_437"]["a1_structured_engine"]["forward_claim_status"] = "trusted"
    with pytest.raises(ValueError, match="A1"):
        mod.validate_artifact(wrong_a1)

    wrong_a2 = copy.deepcopy(valid)
    wrong_a2["close_state_437"]["a2_detector_fix"]["goal_predicate_satisfiable"] = True
    with pytest.raises(ValueError, match="A2"):
        mod.validate_artifact(wrong_a2)

    wrong_a3 = copy.deepcopy(valid)
    wrong_a3["close_state_437"]["a3_levelup"]["new_levels_banked"] = 0
    with pytest.raises(ValueError, match="A3"):
        mod.validate_artifact(wrong_a3)

    wrong_pivot = copy.deepcopy(valid)
    wrong_pivot["v438_pivot"]["headline"] = "retry prompt engineering"
    with pytest.raises(ValueError, match="v438 pivot"):
        mod.validate_artifact(wrong_pivot)

    bad_checksum_prefix = copy.deepcopy(valid)
    bad_checksum_prefix["reproducibility_checksum"] = "not-a-checksum"
    with pytest.raises(ValueError, match="sha256-prefixed"):
        mod.validate_artifact(bad_checksum_prefix)

    bad_checksum = copy.deepcopy(valid)
    bad_checksum["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad_checksum)

    activation_root = tmp_path / "activation_error"
    activation_root.mkdir()
    (activation_root / "research-roadmap.yaml").mkdir()
    (activation_root / "research-roadmap-next.yaml").write_text("milestone: 2026.06.438\n", encoding="utf-8")
    activated, activation_error = mod._activate_next_roadmap(
        activation_root,
        next_info={"available": True, "parses": True, "milestone": "2026.06.438"},
    )
    assert activated is False
    assert activation_error

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
                "old row: 1 failed, 91 passed\nMilestone 2026.06.437 activated\nall green"
            )
        )
        == ""
    )
    assert (
        mod._poison_signature(
            mod._transition_log_scope("Milestone 2026.06.437 activated\n1 failed, 91 passed")
        )
        == "1 failed, 91 passed"
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

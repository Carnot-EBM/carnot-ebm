"""Tests for Exp 4820 `.443` archive / `.444` activation record.

Spec refs: REQ-CAPSTONE-4820, SCENARIO-CAPSTONE-4820,
SCENARIO-CAPSTONE-4820-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4820-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4820_archive_443_activate_444 as mod


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
        stdout="129 passed in 7.0s",
        stderr="",
    )


def _red_poison_smart_subset(_root: Path) -> mod.CommandResult:
    return mod.CommandResult(
        command=["pytest", "smart-subset"],
        exit_code=1,
        stdout="1 failed, 128 passed in 7.1s",
        stderr="test_expected_old_443_s2v3_pause still expects paused selection",
    )


def _s2v3_4811() -> JsonDict:
    return {
        "experiment": "experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate",
        "experiment_id": 4811,
        "honest_verdict": "complete_structural_energy_s2v3_bounded_corpus_wide",
        "DEGENERATE_CANDIDATE_POOL": False,
        "verifier_is_oracle": False,
        "live_path_reachable": True,
        "energy_selected_offpath_cell_recall": 0.30662955385318924,
        "accuracy_gate_selected_offpath_cell_recall": 0.216080109217671,
        "energy_minus_accuracy_delta": 0.09054944463551816,
        "energy_minus_accuracy_delta_ci95": [-0.06276362669828736, 0.26410130774644547],
        "n_available_games": 25,
        "n_games_attempted": 25,
        "n_effective_games": 23,
        "required_effective_games": 15,
        "positive_control_passed": True,
        "false_negative_risk_checked": True,
        "candidates_genuinely_induced": True,
        "s3_authorized": False,
    }


def _capstone_4819() -> JsonDict:
    return {
        "experiment": "experiment_4819_capstone_v443",
        "experiment_id": 4819,
        "capstone_ready": True,
        "honest_verdict": "complete_s2v3_genuine_corpus_wide_bounded_null_pivot_to_s3_generation",
        "reproducible_total_levels": 65,
        "s2v3_structural_energy_verdict": {
            "verdict": "genuine_corpus_wide_bounded_null",
            "reason": "corpus_wide_diverse_pool_ci_includes_zero",
            "degenerate_candidate_pool_flagged": False,
            "reported_energy_minus_accuracy_delta": 0.090549444636,
            "reported_energy_minus_accuracy_delta_ci95": [
                -0.06276362669828736,
                0.26410130774644547,
            ],
            "n_available_games": 25,
            "n_games_attempted": 25,
            "n_effective_games": 23,
            "required_effective_games": 15,
            "positive_control_passed": True,
            "verifier_is_oracle": False,
            "live_path_reachable": True,
            "s3_authorized": False,
            "upstream_honest_verdict": "complete_structural_energy_s2v3_bounded_corpus_wide",
        },
        "readiness": {
            "s2v3_verdict": "genuine_corpus_wide_bounded_null",
            "s3_authorized": False,
            "pivot_energy_to_s3_generation": True,
        },
        "sota_handoff": {
            "decision": "sota_handoff_mapped",
            "flagged_for_v444_candidates": [
                "bolt_cold_cfg_value_tree_generator_for_s3",
                "bes_energy_fitness_pool_inserter",
            ],
            "s3_context": {"roadmap_target": ".444", "s3_generation_allowed": True},
        },
        "submission_package_state": {"submission_package_ready": True, "operator_only": True},
    }


def _write_repo_fixture(
    root: Path,
    *,
    active_milestone: str = "2026.06.444",
    next_present: bool = False,
    registry_total: int = 65,
    capstone_present: bool = True,
    s2v3_present: bool = True,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# test\n", encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(
        f"milestone: {active_milestone}\n"
        "tasks:\n"
        "  - id: exp4820-phase0\n"
        "    deliverable: results/experiment_4820_archive_443_activate_444.json\n",
        encoding="utf-8",
    )
    if next_present:
        (root / "research-roadmap-next.yaml").write_text(
            "milestone: 2026.06.444\n"
            "tasks:\n"
            "  - id: exp4820-phase0\n"
            "    deliverable: results/experiment_4820_archive_443_activate_444.json\n",
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
    spec.write_text("REQ-CAPSTONE-4820\n", encoding="utf-8")
    if capstone_present:
        _write_json(root / "results" / "experiment_4819_capstone_v443.json", _capstone_4819())
    if s2v3_present:
        _write_json(
            root / "results" / "experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate.json",
            _s2v3_4811(),
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


def test_req_capstone_4820_spec_anchor_declares_transition_contract() -> None:
    """REQ-CAPSTONE-4820: OpenSpec declares the transition contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4820_records_s2v3_settled_and_kv260_note(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4820: active `.444` records S2-v3 settlement."""

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
    assert artifact["honest_verdict"] == (
        "complete_443_archived_444_activated_already_active_"
        "selection_settled_bounded_pivot_to_s3"
    )
    assert artifact["transition"] == {
        "archived_milestone": "2026.06.443",
        "activated_milestone": "2026.06.444",
        "active_milestone_confirmed": True,
        "activation_state": "already_activated_by_conductor",
        "archive_state": "archive_noop_or_already_recorded",
    }
    assert artifact["s2v3_selection_settled"] is True
    assert artifact["kv260_offline_noted"] is True
    assert artifact["reproducible_total_levels"] == 65
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"][
        "accepted_missing_because_already_active"
    ] is True
    assert artifact["poison_test_resolved"] == {
        "resolved": True,
        "current_gate_passed": True,
        "poison_tests": [],
        "action": "no_poison_observed_current_gate_green",
    }

    close = artifact["close_state_443"]
    assert close["capstone_honest_verdict"] == (
        "complete_s2v3_genuine_corpus_wide_bounded_null_pivot_to_s3_generation"
    )
    assert close["reproducible_total_levels"] == 65
    assert close["sota_handoff"]["s3_context"]["roadmap_target"] == ".444"

    s2v3 = close["s2v3_record"]
    assert s2v3["selection_status"] == "selection_settled_bounded_pivot_to_s3"
    assert s2v3["verdict"] == "bounded_corpus_wide"
    assert s2v3["n_available_games"] == 25
    assert s2v3["n_games_attempted"] == 25
    assert s2v3["n_effective_games"] == 23
    assert s2v3["required_effective_games"] == 15
    assert s2v3["coverage_floor_met"] is True
    assert s2v3["positive_control_passed"] is True
    assert s2v3["degenerate_candidate_pool_fired"] is False
    assert s2v3["reported_energy_minus_accuracy_delta"] == pytest.approx(0.09054944463551816)
    assert s2v3["reported_energy_minus_accuracy_delta_ci95"] == [
        -0.06276362669828736,
        0.26410130774644547,
    ]
    assert s2v3["ci_includes_zero"] is True
    assert s2v3["point_estimate_flipped_from_s2v2"] is True
    assert s2v3["previous_s2v2_delta_at_n5"] == pytest.approx(-0.15765776352537078)
    assert s2v3["energy_direction"] == "roughly_neutral_at_engine_selection"
    assert s2v3["pivot"] == "S3_generation_lift"

    kv = close["kv260_offline_note"]
    assert kv["experiment_id"] == 4817
    assert kv["failed_attempts"] == 3
    assert kv["failure_mode"] == "no_file_changes"
    assert kv["board_offline"] is True
    assert kv["board_address"] == "192.168.51.98"
    assert kv["blocked_artifact_was_written"] is False
    assert kv["v444_c_task_corrected_to_write_blocked_artifact"] is True
    assert artifact["v444_pivot"]["task_id"] == "exp4821-a1"
    assert artifact["v444_pivot"]["direction"] == "S3_generation_lift"
    assert artifact["leaderboard_submission"] is False
    assert mod.validate_artifact(artifact) is None


def test_scenario_capstone_4820_can_activate_literal_next_roadmap(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4820: present next roadmap activates onto active YAML."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.443", next_present=True)

    artifact = mod.run(
        root=tmp_path,
        write=False,
        started_s=3.0,
        now_s=3.4,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8").startswith(
        "milestone: 2026.06.444"
    )
    assert artifact["transition"]["activation_state"] == "activated_from_research_roadmap_next"
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml"]["activation_attempted"] is True
    assert artifact["honest_verdict"] == (
        "complete_443_archived_444_activated_from_next_"
        "selection_settled_bounded_pivot_to_s3"
    )


def test_scenario_capstone_4820_blockers_and_poison_signature_are_explicit(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4820-BLOCKED-PRECONDITION: blocked paths do not fabricate."""

    _write_repo_fixture(tmp_path, active_milestone="2026.06.443", next_present=False)

    artifact = mod.run(
        root=tmp_path,
        write=True,
        started_s=4.0,
        now_s=4.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )

    assert artifact["honest_verdict"] == "blocked_research_roadmap_444_unavailable"
    assert artifact["transition"]["activation_state"] == "blocked_missing_or_failed_precondition"
    assert artifact["preconditions_checked"]["smart_subset_pretest_gate"]["passed"] is None
    assert artifact["s2v3_selection_settled"] is False
    assert artifact["kv260_offline_noted"] is False
    assert artifact["close_state_443"] == {}
    assert artifact["v444_pivot"] == {}

    checks = _artifact(tmp_path / "good")["preconditions_checked"]
    assert mod._first_blocker(checks) is None

    for key, expected in {
        "agents_md": "missing_agents_md",
        "codex_or_opencode_md": "missing_codex_or_opencode_md",
        "capstone_spec": "missing_capstone_spec_req_4820",
        "registry": "arc_solve_registry",
        "capstone_4819": "missing_experiment_4819_capstone_v443",
        "s2v3_4811": "missing_experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate",
    }.items():
        bad = copy.deepcopy(checks)
        bad[key]["available"] = False
        if key == "capstone_spec":
            bad[key]["has_req_4820"] = False
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
            "id": "test_expected_old_443_s2v3_pause",
            "reason": "single-failure smart-subset signature matches a stale transition expectation",
            "action": "blocked_for_fix_or_quarantine_before_tail_continues",
        }
    ]

    other_smart_failure = copy.deepcopy(checks)
    other_smart_failure["smart_subset_pretest_gate"]["passed"] = False
    other_smart_failure["smart_subset_pretest_gate"]["stdout_tail"] = "2 failed, 127 passed"
    other_smart_failure["smart_subset_pretest_gate"]["stderr_tail"] = ""
    assert mod._poison_test_resolution(other_smart_failure)["poison_tests"] == []


def test_scenario_capstone_4820_field_principle_validation_fails_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4820-FIELD-PRINCIPLES: schema drift fails loudly."""

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
    blocked["close_state_443"] = {"fabricated": True}
    blocked["reproducibility_checksum"] = "sha256:" + mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(blocked)

    wrong_total = copy.deepcopy(valid)
    wrong_total["reproducible_total_levels"] = 64
    with pytest.raises(ValueError, match="registry total"):
        mod.validate_artifact(wrong_total)

    wrong_top_level_s2v3 = copy.deepcopy(valid)
    wrong_top_level_s2v3["s2v3_selection_settled"] = False
    with pytest.raises(ValueError, match="S2-v3"):
        mod.validate_artifact(wrong_top_level_s2v3)

    wrong_top_level_kv = copy.deepcopy(valid)
    wrong_top_level_kv["kv260_offline_noted"] = False
    with pytest.raises(ValueError, match="KV260"):
        mod.validate_artifact(wrong_top_level_kv)

    for field, value in {
        "selection_status": "paused",
        "verdict": "win",
        "n_effective_games": 22,
        "required_effective_games": 16,
        "coverage_floor_met": False,
        "degenerate_candidate_pool_fired": True,
        "point_estimate_flipped_from_s2v2": False,
        "pivot": "paused",
    }.items():
        wrong = copy.deepcopy(valid)
        wrong["close_state_443"]["s2v3_record"][field] = value
        with pytest.raises(ValueError, match="S2-v3"):
            mod.validate_artifact(wrong)

    for field, value in {
        "failed_attempts": 2,
        "failure_mode": "ssh_timeout",
        "board_offline": False,
        "blocked_artifact_was_written": True,
        "v444_c_task_corrected_to_write_blocked_artifact": False,
    }.items():
        wrong = copy.deepcopy(valid)
        wrong["close_state_443"]["kv260_offline_note"][field] = value
        with pytest.raises(ValueError, match="KV260"):
            mod.validate_artifact(wrong)

    wrong_pivot = copy.deepcopy(valid)
    wrong_pivot["v444_pivot"]["direction"] = "S2_selection"
    with pytest.raises(ValueError, match="v444 pivot"):
        mod.validate_artifact(wrong_pivot)

    bad_checksum_prefix = copy.deepcopy(valid)
    bad_checksum_prefix["reproducibility_checksum"] = "not-a-checksum"
    with pytest.raises(ValueError, match="sha256-prefixed"):
        mod.validate_artifact(bad_checksum_prefix)

    bad_checksum = copy.deepcopy(valid)
    bad_checksum["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad_checksum)

    assert mod._activate_next_roadmap(tmp_path, next_info={"available": False}) == (False, "")

    activation_root = tmp_path / "activation_error"
    activation_root.mkdir()
    (activation_root / "research-roadmap.yaml").mkdir()
    (activation_root / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.06.444\n",
        encoding="utf-8",
    )
    activated, activation_error = mod._activate_next_roadmap(
        activation_root,
        next_info={"available": True, "parses": True, "milestone": "2026.06.444"},
    )
    assert activated is False
    assert activation_error

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

    missing_s2_root = tmp_path / "missing_s2v3"
    _write_repo_fixture(missing_s2_root, s2v3_present=False)
    missing_s2_artifact = mod.build_artifact(
        missing_s2_root,
        started_s=7.0,
        now_s=7.1,
        offline_arcade_checker=lambda: True,
        smart_subset_checker=_green_smart_subset,
    )
    assert missing_s2_artifact["honest_verdict"] == (
        "blocked_missing_experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate"
    )

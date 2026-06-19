"""Tests for Exp 4420 `.408` archive / `.409` activation.

Spec refs: REQ-REPORT-4420, SCENARIO-REPORT-4420,
SCENARIO-REPORT-4420-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from carnot.reporting import archive_408_activate_409_4420 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="42 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.407\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.408\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-19'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4423-capstone-v408\n"
        "    result: OK (conductor)\n"
    )
    return head + block * duplicates


def _manifest_text() -> str:
    return (
        "retired_extras:\n"
        "- id: circular_arc_solve_not_oracle_distinct_moat\n"
        "  verifier_is_oracle: true\n"
    )


def _capstone(**overrides: object) -> dict:
    payload = {
        "honest_verdict": (
            "complete: v408_config_rule_grounded_no_new_levels_localizer_closed_"
            "sovereign_gap4_holds_vocab_false_detection_false_arc_levels_34_publication_ready"
        ),
        "verifier_is_oracle": False,
        "arc_config_rule_state": "grounded_config_rules_no_new_reproducible_levels",
        "localizer_program_state": "closed_position_bound_text_and_hidden",
        "sovereign_verifier_state": "sovereign_gap4_local_gate_holds_execution_grounded",
        "config_rule_vocabulary_transfers": False,
        "detection_calibrated_multi_domain": False,
        "reproducible_total_levels": 34,
        "arc_config_rule": {
            "new_levels_reproduced_from_artifacts": 0,
            "grounded_win_rules_count": 1,
            "grounded_win_rules": [
                {
                    "game": "ka59",
                    "tier": 2,
                    "predicate": "editable_count_4_equals_reference_count_4_32",
                    "fires_on_win": True,
                    "false_positive_rate": 0.0,
                    "literal_hardcode": False,
                }
            ],
            "agent2world_adaptive_e3": {
                "new_levels_reproduced": 0,
                "per_target_scorecard": [
                    {"game": "ar25", "offline_reproduced": False},
                    {"game": "tn36", "offline_reproduced": False},
                    {"game": "lp85", "offline_reproduced": False},
                ],
            },
        },
        "arc_reproducible_progress": {
            "prior_reproducible_total_levels": 34,
            "prior_reproducible_total_games": 17,
            "reproducible_total_levels": 34,
            "reproducible_total_games": 17,
            "new_levels_since_prior": 0,
            "new_games_since_prior": 0,
            "status": "loaded",
        },
        "publication_gate": {"paper_ready": True, "unmet_gates": []},
        "availability_report": {
            "flagged_artifacts_excluded": [],
            "missing_upstream_artifacts": [],
            "axes": {
                "arc_config_rule": {"verdict": "grounded", "flagged_artifacts": []},
                "localizer_program": {
                    "verdict": "closed_position_bound_text_and_hidden",
                    "flagged_artifacts": [],
                },
                "sovereign_verifier": {
                    "verdict": "sovereign_gap4_local_gate_holds_execution_grounded",
                    "flagged_artifacts": [],
                },
                "vocabulary": {"verdict": False, "flagged_artifacts": []},
                "detection": {"verdict": False, "flagged_artifacts": []},
            },
        },
        "capstone_live_adversarial_recheck": {
            "status": "clean",
            "flags": [],
            "circular_moat_overclaim": False,
        },
    }
    payload.update(overrides)
    return payload


def _config_rule(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete_config_rule_partial",
        "verifier_is_oracle": True,
        "new_levels_reproduced": 0,
        "reproducible_total_levels": 34,
        "config_win_rules_grounded": [
            {
                "game": "ka59",
                "tier": 2,
                "predicate": "editable_count_4_equals_reference_count_4_32",
                "fires_on_win": True,
                "false_positive_rate": 0.0,
                "literal_hardcode": False,
            }
        ],
        "per_target_scorecard": [
            {
                "game": "ka59",
                "prior_best_level": 1,
                "new_reproduced_level": 1,
                "offline_reproduced": False,
                "grounding_tier": 2,
                "search_blocker": "no_registered_next_level_config_adapter",
                "win_rule_predicate": "editable_count_4_equals_reference_count_4_32",
            }
        ],
        "flagged_adversarial": None,
    }
    payload.update(overrides)
    return payload


def _agent2world(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete_e3_adaptive_partial",
        "verifier_is_oracle": True,
        "new_levels_reproduced": 0,
        "reproducible_total_levels": 34,
        "per_target_scorecard": [
            {
                "game": "ar25",
                "prior_best_level": 1,
                "new_reproduced_level": 1,
                "offline_reproduced": False,
                "adaptive_tests_passed": 1,
                "adaptive_tests_total": 2,
                "residual_failing_behavior": "ar25_l2_hidden_undo_stack_state_not_visible_in_rollout",
            },
            {
                "game": "tn36",
                "prior_best_level": 7,
                "new_reproduced_level": 7,
                "offline_reproduced": False,
                "adaptive_tests_passed": 1,
                "adaptive_tests_total": 2,
                "residual_failing_behavior": "tn36_l8_palette_population_or_later_program_state_still_wrong",
            },
            {
                "game": "lp85",
                "prior_best_level": 5,
                "new_reproduced_level": 5,
                "offline_reproduced": False,
                "adaptive_tests_passed": 1,
                "adaptive_tests_total": 2,
                "residual_failing_behavior": "lp85_l6_button_permutation_search_reproduction_still_wrong",
            },
        ],
        "flagged_adversarial": None,
    }
    payload.update(overrides)
    return payload


def _hidden_state(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: clean_powered_null_position_only_not_beaten",
        "verifier_is_oracle": False,
        "hidden_state_localizer_has_nonposition_signal": False,
        "position_only_baseline_f1": 1.0,
        "localization_f1_comparison": {
            "n_traces": 1000,
            "position_only_baseline_f1": 1.0,
            "hidden_state_probe_f1": 1.0,
            "delta_vs_position_only": 0.0,
            "delta_ci95": [0.0, 0.0],
        },
        "flagged_adversarial": None,
    }
    payload.update(overrides)
    return payload


def _gap4(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: sovereign_gap4_local_gate_holds_flat_cov_0.2333_fires_0_lost_0",
        "verifier_is_oracle": True,
        "sovereign_gap4_gate_holds": True,
        "pass2_vs_vote": {
            "gated_pass2": 0.4516,
            "vote_pass2": 0.4516,
            "delta": 0.0,
            "delta_ci95": [0.0, 0.0],
            "graded_gate_fires": 0,
            "pass2_vote_wins_lost": 0,
        },
        "local_generator_coverage": 0.2333,
        "k_consistency_details": {
            "unique_tasks": 30,
            "demo_perfect_unique_tasks": 7,
            "k_consistent_entries": 0,
            "lost": 0,
        },
        "flagged_adversarial": None,
    }
    payload.update(overrides)
    return payload


def _vocabulary(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "blocked_local_model_unavailable",
        "verifier_is_oracle": False,
        "config_rule_vocabulary_transfers": False,
        "grounding_rate_lift": None,
        "flagged_adversarial": None,
    }
    payload.update(overrides)
    return payload


def _steerconf(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: clean_null_steered_confidence_does_not_rescue_code_detector",
        "verifier_is_oracle": False,
        "detection_calibrated_multi_domain": False,
        "flagged_adversarial": None,
    }
    payload.update(overrides)
    return payload


def _make_root(tmp_path: Path, *, duplicates: int = 1) -> Path:
    (tmp_path / "ops").mkdir(parents=True)
    (tmp_path / "results").mkdir(parents=True)
    (tmp_path / "research-complete.yaml").write_text(
        _research_complete_text(duplicates=duplicates), encoding="utf-8"
    )
    (tmp_path / "ops/exclusion_manifest.yaml").write_text(_manifest_text(), encoding="utf-8")
    (tmp_path / "research-roadmap.yaml").write_text(
        "milestone: 2026.06.409\n", encoding="utf-8"
    )
    _write_json(tmp_path / "results/experiment_4423_capstone_v408.json", _capstone())
    _write_json(tmp_path / "results/experiment_4414_config_rule_induction_solve.json", _config_rule())
    _write_json(
        tmp_path / "results/experiment_4415_agent2world_adaptive_e3_repair.json",
        _agent2world(),
    )
    _write_json(
        tmp_path / "results/experiment_4416_hidden_state_localizer_falsification_audit.json",
        _hidden_state(),
    )
    _write_json(
        tmp_path / "results/experiment_4417_gap4_local_generator_sovereign_arm.json",
        _gap4(),
    )
    _write_json(
        tmp_path / "results/experiment_4418_config_rule_vocabulary_transfer.json",
        _vocabulary(),
    )
    _write_json(
        tmp_path / "results/experiment_4419_steerconf_code_detection_calibration_repair.json",
        _steerconf(),
    )
    return tmp_path


def _sources() -> dict:
    return {
        "4423": _capstone(),
        "4414": _config_rule(),
        "4415": _agent2world(),
        "4416": _hidden_state(),
        "4417": _gap4(),
        "4418": _vocabulary(),
        "4419": _steerconf(),
    }


def test_run_archives_v408_and_records_true_close_state(tmp_path: Path) -> None:
    # REQ-REPORT-4420 / SCENARIO-REPORT-4420
    root = _make_root(tmp_path, duplicates=2)

    output_path = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    artifact = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith(("complete:", "complete_", "success:", "success_"))
    assert artifact["archived_milestone"] == "2026.06.408"
    assert artifact["activated_milestone"] == "2026.06.409"
    assert artifact["active_milestone_confirmed"] == "2026.06.409"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert artifact["field_principles"]["honest_verdict"] == mod.HONEST_VERDICT_PRINCIPLE
    assert artifact["trm_training_ran"] is False
    assert artifact["leaderboard_submission"] is False

    history = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(history) == 1
    assert "activation_recorded: exp4420-archive-408-activate-409" in history
    assert "config-rule grounded; no new reproducible levels" in history

    close = artifact["v408_close_state"]
    assert close["capstone_honest_verdict"].startswith("complete:")
    assert close["arc_config_rule_state"] == "grounded_config_rules_no_new_reproducible_levels"
    assert close["config_rule_new_levels_reproduced"] == 0
    assert close["grounded_win_rules_count"] == 1
    assert close["grounded_win_rules"][0]["game"] == "ka59"
    assert close["agent2world_outcome"] == "honest_partial_zero_new_levels"
    assert close["agent2world_new_levels_reproduced"] == 0
    assert close["agent2world_residual_games"] == ["ar25", "tn36", "lp85"]
    assert close["reproducible_total_levels"] == 34
    assert close["reproducible_total_games"] == 17
    assert close["new_levels_since_prior"] == 0
    assert close["localizer_program_state"] == "closed_position_bound_text_and_hidden"
    assert close["hidden_state_localizer_has_nonposition_signal"] is False
    assert close["hidden_state_delta_vs_position_only"] == 0.0
    assert close["sovereign_verifier_state"] == "sovereign_gap4_local_gate_holds_execution_grounded"
    assert close["sovereign_gap4_gate_holds"] is True
    assert close["gap4_pass2_vote_wins_lost"] == 0
    assert close["local_generator_coverage"] == 0.2333
    assert close["config_rule_vocabulary_transfers"] is False
    assert close["vocabulary_outcome"] == "blocked_local_model_unavailable_or_false"
    assert close["detection_calibrated_multi_domain"] is False
    assert close["steerconf_outcome"] == "clean_null_code_detector_not_rescued"
    assert close["paper_ready"] is True
    assert close["verifier_is_oracle_respected"] is True
    assert close["circular_execution_grounded_solves_not_moat"] is True
    assert artifact["flagged_artifacts_excluded"] == []


def test_flagged_upstream_artifacts_are_not_used_as_truth(tmp_path: Path) -> None:
    # REQ-REPORT-4420: flagged_adversarial sources are skipped.
    root = _make_root(tmp_path)
    _write_json(
        root / "results/experiment_4414_config_rule_induction_solve.json",
        _config_rule(
            flagged_adversarial=True,
            new_levels_reproduced=99,
            config_win_rules_grounded=[{"game": "fake", "tier": 99}],
        ),
    )

    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5).read_text(
            encoding="utf-8"
        )
    )

    assert artifact["flagged_artifacts_excluded"] == ["4414"]
    close = artifact["v408_close_state"]
    assert close["config_rule_new_levels_reproduced"] == 0
    assert close["grounded_win_rules"][0]["game"] == "ka59"


def test_run_blocks_when_pretest_red_without_editing_history(tmp_path: Path) -> None:
    # SCENARIO-REPORT-4420-BLOCKED-PRECONDITION
    root = _make_root(tmp_path)
    before = (root / "research-complete.yaml").read_text(encoding="utf-8")

    artifact = json.loads(
        mod.run(root, pretest_result=RED, started_s=1000.0, now_s=1000.5).read_text(
            encoding="utf-8"
        )
    )

    assert artifact["honest_verdict"] == "blocked_smart_subset_pretest_not_green"
    assert artifact["preconditions_checked"]["smart_subset_pretest"]["green"] is False
    assert artifact["v408_close_state"] == {}
    assert (root / "research-complete.yaml").read_text(encoding="utf-8") == before


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda root: (root / "research-complete.yaml").unlink(), "blocked_research_complete_yaml_missing"),
        (
            lambda root: (root / "research-complete.yaml").write_text("milestones: [", encoding="utf-8"),
            "blocked_research_complete_yaml_poison",
        ),
        (lambda root: (root / "ops/exclusion_manifest.yaml").unlink(), "blocked_exclusion_manifest_missing"),
        (
            lambda root: (root / "ops/exclusion_manifest.yaml").write_text("retired_extras: [", encoding="utf-8"),
            "blocked_exclusion_manifest_yaml_poison",
        ),
        (
            lambda root: (root / "research-roadmap.yaml").write_text("milestone: 2026.06.408\n", encoding="utf-8"),
            "blocked_v409_not_active",
        ),
        (
            lambda root: (root / "results/experiment_4423_capstone_v408.json").unlink(),
            "blocked_v408_capstone_missing",
        ),
    ],
)
def test_run_blocks_each_precondition_failure(tmp_path: Path, mutate: object, reason: str) -> None:
    # SCENARIO-REPORT-4420-BLOCKED-PRECONDITION
    root = _make_root(tmp_path)
    mutate(root)

    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5).read_text(
            encoding="utf-8"
        )
    )

    assert artifact["honest_verdict"] == reason
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["v408_close_state"] == {}


def test_run_blocks_if_research_complete_edit_would_not_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # SCENARIO-REPORT-4420-BLOCKED-PRECONDITION
    root = _make_root(tmp_path)

    def poisoned(_: str, close_state: dict) -> tuple[str, int, str]:
        return "milestones: [", 0, "updated"

    monkeypatch.setattr(mod, "dedupe_or_update_record", poisoned)
    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5).read_text(
            encoding="utf-8"
        )
    )

    assert artifact["honest_verdict"] == "blocked_research_complete_edit_invalid"
    assert artifact["v408_close_state"] == {}


def test_validate_artifact_rejects_false_success() -> None:
    # REQ-REPORT-4420
    payload = mod.build_complete_artifact(
        v408_close_state=mod.build_v408_close_state(_sources()),
        preconditions_checked={"smart_subset_pretest": {"green": True}},
        duration_s=0.5,
        active_roadmap_path="research-roadmap.yaml",
        research_complete_record_action="updated",
        research_complete_duplicates_removed=0,
        cited_upstream_artifacts=[],
        flagged_artifacts_excluded=[],
    )
    payload = copy.deepcopy(payload)
    payload["v408_close_state"]["reproducible_total_levels"] = 35

    with pytest.raises(ValueError, match="ARC total levels"):
        mod.validate_artifact(payload)


def test_record_helpers_cover_append_update_and_unchanged_paths() -> None:
    # REQ-REPORT-4420
    close = mod.build_v408_close_state(_sources())

    appended, removed, action = mod.dedupe_or_update_record("milestones:\n", close)
    assert action == "appended"
    assert removed == 0
    assert mod.archive_record_count(appended) == 1

    unchanged, removed_again, action_again = mod.dedupe_or_update_record(appended, close)
    assert action_again == "unchanged"
    assert removed_again == 0
    assert unchanged == appended
    assert mod._ci95("not-a-ci", [1.0, 2.0]) == [1.0, 2.0]
    assert mod._insert_before_tasks(["- id: 2026.06.408"], "  finding: x")[-1] == "  finding: x"

    updated, removed_update, action_update = mod.dedupe_or_update_record(
        (
            "milestones:\n"
            "- id: 2026.06.408\n"
            "  title: old\n"
            "  activation_recorded: stale\n"
        ),
        close,
    )
    assert action_update == "updated"
    assert removed_update == 0
    assert "finding:" in updated
    assert "activation_recorded: exp4420-archive-408-activate-409" in updated


def test_run_blocks_if_written_research_complete_turns_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # SCENARIO-REPORT-4420-BLOCKED-PRECONDITION
    root = _make_root(tmp_path)
    calls = {"n": 0}
    original = mod.yaml_parses

    def fails_after_write(text: str) -> bool:
        calls["n"] += 1
        if calls["n"] >= 4 and "config-rule grounded" in text:
            return False
        return original(text)

    monkeypatch.setattr(mod, "yaml_parses", fails_after_write)
    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5).read_text(
            encoding="utf-8"
        )
    )

    assert artifact["honest_verdict"] == "blocked_research_complete_yaml_poison_after_edit"
    assert artifact["v408_close_state"] == {}


def test_module_main_and_results_runner_delegate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-REPORT-4420
    root = _make_root(tmp_path)
    monkeypatch.setattr(mod, "run", lambda _: root / "module-main-sentinel.json")
    assert mod.main(root) == 0

    script_path = Path(__file__).parents[2] / "results/experiment_4420_archive_408_activate_409.py"
    script_repo_root = script_path.parents[1]
    removed = {str(script_repo_root), str(script_repo_root / "python")}
    monkeypatch.setattr(sys, "path", [item for item in sys.path if item not in removed])
    spec = importlib.util.spec_from_file_location("exp4420_runner", script_path)
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "run", lambda _: root / "runner-sentinel.json")

    assert runner.main(root) == 0

"""Tests for Exp 4413 `.407` archive / `.408` activation.

Spec refs: REQ-REPORT-4413, SCENARIO-REPORT-4413,
SCENARIO-REPORT-4413-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from carnot.reporting import archive_v407_activate_v408_4413 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="81 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.406\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.407\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-18'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4412-capstone-v407\n"
        "    result: OK (conductor)\n"
    )
    return head + block * duplicates


def _manifest_text() -> str:
    return (
        "retired_extras:\n"
        "- id: first_error_text_localizer_retired_exp4403_v407\n"
        "  experiment_ids: [exp4403]\n"
        "  retire_if_same_verdict: true\n"
    )


def _registry_text() -> str:
    return (
        "schema_version: 1\n"
        "updated: '2026-06-18'\n"
        "reproducible_total_levels: 34\n"
        "reproducible_total_games: 17\n"
    )


def _capstone(**overrides: object) -> dict:
    payload = {
        "honest_verdict": (
            "complete: v407_localizer_position_bound_retired_compounds_false_"
            "calibrated_false_arc_levels_34_publication_ready"
        ),
        "localizer_state": "position_bound_retired",
        "localizer_compounds": False,
        "detection_calibrated_multi_domain": False,
        "paper_ready": True,
        "reproducible_total_levels": 34,
        "publication_gate": {"paper_ready": True, "unmet_gates": []},
        "arc_reproducible_progress": {
            "reproducible_total_levels": 34,
            "reproducible_total_games": 17,
            "new_levels_since_prior": 0,
            "new_games_since_prior": 0,
        },
        "localizer": {
            "status": "position_bound_retired",
            "real_intervention": {
                "status": "position_only_tied",
                "localizer_genuinely_beats_position_only": False,
                "beats_position_only_baseline": False,
                "position_only_baseline_f1": 1.0,
                "template_family_holdout_drop": 0.0,
                "localization_f1_by_domain": {
                    "FoVer": {
                        "position_only_baseline": 1.0,
                        "real_intervention_localizer": 1.0,
                        "delta_vs_position_only": 0.0,
                        "delta_ci95": [0.0, 0.0],
                        "template_family_holdout_drop": 0.0,
                        "n_error_traces": 947,
                    },
                    "GAP-4 ARC": {
                        "position_only_baseline": 0.788462,
                        "real_intervention_localizer": 0.807692,
                        "delta_vs_position_only": 0.019231,
                        "delta_ci95": [-0.134615, 0.173077],
                        "n_error_traces": 52,
                    },
                },
            },
        },
        "self_learning": {
            "localizer_compounds": False,
            "positive_control_passed": False,
            "compounding_delta_ci95": [0.0, 0.0],
            "active_vs_random_learning_curve": [
                {"corpus_size": 51, "f1_active": 1.0, "f1_random": 1.0},
                {"corpus_size": 512, "f1_active": 1.0, "f1_random": 1.0},
            ],
        },
        "calibration": {
            "detection_calibrated_multi_domain": False,
            "domains_at_chance": ["code_humaneval"],
            "detection_by_domain": [
                {
                    "domain": "code_humaneval",
                    "n": 539,
                    "detection_auroc": 0.577374,
                    "auroc_ci95": [0.461255, 0.692756],
                    "claim_scope": "proper_pool_n>=300",
                },
            ],
        },
        "arc_e3_outcomes": {
            "deeper_high_headroom": {
                "new_levels_reproduced": 0,
                "per_target_scorecard": [
                    {"game": "lp85", "lookahead_fidelity": 0.833333, "mechanic_unit_test_pass_rate": 1.0},
                    {"game": "tu93", "lookahead_fidelity": 0.8, "mechanic_unit_test_pass_rate": 1.0},
                    {"game": "tn36", "lookahead_fidelity": 0.875, "mechanic_unit_test_pass_rate": 1.0},
                ],
            },
            "blocked_mechanics": {
                "new_levels_reproduced": 0,
                "per_game_scorecard": [
                    {"game": "ar25", "lookahead_fidelity": 0.733333, "register_unit_test_pass_rate": 1.0},
                    {"game": "ka59", "lookahead_fidelity": 0.112281, "register_unit_test_pass_rate": 1.0},
                    {"game": "ft09", "lookahead_fidelity": 0.347518, "register_unit_test_pass_rate": 1.0},
                ],
            },
        },
    }
    payload.update(overrides)
    return payload


def _localizer() -> dict:
    return {
        "honest_verdict": "complete: clean_powered_null_position_only_not_beaten",
        "localizer_genuinely_beats_position_only": False,
        "beats_position_only_baseline": False,
        "position_only_baseline_f1": 1.0,
        "template_family_holdout_drop": 0.0,
        "localization_f1_by_domain": {
            "FoVer": {
                "position_only_baseline": 1.0,
                "real_intervention_localizer": 1.0,
                "delta_vs_position_only": 0.0,
                "delta_ci95": [0.0, 0.0],
                "template_family_holdout_drop": 0.0,
                "n_error_traces": 947,
            },
            "GAP-4 ARC": {
                "position_only_baseline": 0.788462,
                "real_intervention_localizer": 0.807692,
                "delta_vs_position_only": 0.019231,
                "delta_ci95": [-0.134615, 0.173077],
                "n_error_traces": 52,
            },
        },
    }


def _compounds() -> dict:
    return {
        "honest_verdict": "complete: clean_null_position_bound_or_saturated",
        "localizer_compounds": False,
        "positive_control_passed": False,
        "compounding_delta_ci95": [0.0, 0.0],
        "active_vs_random_learning_curve": [
            {"corpus_size": 51, "f1_active": 1.0, "f1_random": 1.0, "position_only_floor": 1.0},
            {"corpus_size": 512, "f1_active": 1.0, "f1_random": 1.0, "position_only_floor": 1.0},
        ],
    }


def _calibration() -> dict:
    return {
        "honest_verdict": "complete: calibrated_multi_domain_contract_false_deconfounded",
        "detection_calibrated_multi_domain": False,
        "domains_at_chance": ["code_humaneval"],
        "positive_control_passed": True,
        "detection_by_domain": [
            {
                "domain": "code_humaneval",
                "n": 539,
                "detection_auroc": 0.577374,
                "auroc_ci95": [0.461255, 0.692756],
                "claim_scope": "proper_pool_n>=300",
            },
            {"domain": "fover", "n": 8829, "detection_auroc": 0.918304},
        ],
    }


def _arc_deeper() -> dict:
    return {
        "honest_verdict": "complete_e3_deeper_partial",
        "new_levels_reproduced": 0,
        "reproducible_total_levels": 34,
        "per_target_scorecard": [
            {"game": "lp85", "lookahead_fidelity": 0.833333, "mechanic_unit_test_pass_rate": 1.0},
            {"game": "tn36", "lookahead_fidelity": 0.875, "mechanic_unit_test_pass_rate": 1.0},
        ],
    }


def _arc_tails() -> dict:
    return {
        "honest_verdict": "complete_e3_ar25_ka59_ft09_partial",
        "new_levels_reproduced": 0,
        "reproducible_total_levels": 34,
        "per_game_scorecard": [
            {"game": "ar25", "lookahead_fidelity": 0.733333, "register_unit_test_pass_rate": 1.0},
            {"game": "ka59", "lookahead_fidelity": 0.112281, "register_unit_test_pass_rate": 1.0},
            {"game": "ft09", "lookahead_fidelity": 0.347518, "register_unit_test_pass_rate": 1.0},
        ],
    }


def _sota() -> dict:
    return {
        "honest_verdict": "complete: sota_ingestion_v408_mapped",
        "flagged_for_v408": "agent2world_adaptive_e3_mechanic_repair_v408",
        "methods_mapped": [
            {"arxiv_id_or_url": "2512.22336"},
            {"arxiv_id_or_url": "2605.25931"},
            {"arxiv_id_or_url": "2605.13772"},
            {"arxiv_id_or_url": "2503.02863"},
            {"arxiv_id_or_url": "2508.02298"},
        ],
    }


def _make_root(tmp_path: Path, *, duplicates: int = 1) -> Path:
    (tmp_path / "ops").mkdir(parents=True)
    (tmp_path / "openspec/change-proposals").mkdir(parents=True)
    (tmp_path / "results").mkdir(parents=True)
    (tmp_path / "research-complete.yaml").write_text(
        _research_complete_text(duplicates=duplicates), encoding="utf-8"
    )
    (tmp_path / "ops/exclusion_manifest.yaml").write_text(_manifest_text(), encoding="utf-8")
    (tmp_path / "ops/arc_solve_registry.yaml").write_text(_registry_text(), encoding="utf-8")
    (tmp_path / "research-roadmap.yaml").write_text(
        "milestone: 2026.06.408\nmilestone_doc: openspec/change-proposals/research-roadmap-v408.md\n",
        encoding="utf-8",
    )
    (tmp_path / "openspec/change-proposals/research-roadmap-v408.md").write_text(
        "# Research Roadmap v408\n\nPIVOT-to-verifier-grounded-config-rule-induction.\n",
        encoding="utf-8",
    )
    _write_json(tmp_path / "results/experiment_4412_capstone_v407.json", _capstone())
    _write_json(tmp_path / "results/experiment_4403_real_intervention_localizer_deconfound.json", _localizer())
    _write_json(tmp_path / "results/experiment_4407_active_learning_self_learning_compounds.json", _compounds())
    _write_json(tmp_path / "results/experiment_4408_cross_domain_detection_calibration_repair.json", _calibration())
    _write_json(tmp_path / "results/experiment_4405_e3_deeper_mechanic_unit_tests.json", _arc_deeper())
    _write_json(tmp_path / "results/experiment_4406_e3_blocked_mechanic_tails_unit_tests.json", _arc_tails())
    _write_json(tmp_path / "results/experiment_4409_sota_ingestion_v408.json", _sota())
    return tmp_path


def _sources() -> dict:
    return {
        "4412": _capstone(),
        "4403": _localizer(),
        "4407": _compounds(),
        "4408": _calibration(),
        "4405": _arc_deeper(),
        "4406": _arc_tails(),
        "4409": _sota(),
        "arc_solve_registry": {"reproducible_total_levels": 34, "reproducible_total_games": 17},
    }


def test_run_archives_v407_and_records_true_close_state(tmp_path: Path) -> None:
    # REQ-REPORT-4413 / SCENARIO-REPORT-4413
    root = _make_root(tmp_path, duplicates=2)

    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    artifact = json.loads(out.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith(("complete:", "success:", "passed:", "shipped:"))
    assert artifact["archived_milestone"] == "2026.06.407"
    assert artifact["activated_milestone"] == "2026.06.408"
    assert artifact["active_milestone_confirmed"] == "2026.06.408"
    assert artifact["pretest_suite_green"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["research_complete_duplicates_removed"] == 1
    history = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(history) == 1
    assert "position-bound" in history

    close = artifact["v407_close_state"]
    assert close["localizer_axis_state"] == "RETIRED_POSITION_BOUND"
    assert close["localizer_state"] == "position_bound_retired"
    assert close["localizer_genuinely_beats_position_only"] is False
    assert close["beats_position_only_baseline"] is False
    assert close["position_only_baseline_f1"] == 1.0
    assert close["fover_real_intervention_localizer_f1"] == 1.0
    assert close["fover_delta_vs_position_only"] == 0.0
    assert close["template_family_holdout_drop"] == 0.0
    assert close["retire_if_same_verdict_fired"] is True
    assert close["localizer_compounds"] is False
    assert close["self_learning_positive_control_passed"] is False
    assert close["self_learning_f1_active_first"] == 1.0
    assert close["self_learning_f1_active_last"] == 1.0
    assert close["detection_calibrated_multi_domain"] is False
    assert close["code_humaneval_detection_auroc"] == 0.577374
    assert close["code_humaneval_at_chance"] is True
    assert close["arc_reproducible_total_levels"] == 34
    assert close["arc_reproducible_total_games"] == 17
    assert close["arc_new_levels_since_prior"] == 0
    assert close["arc_new_levels_reproduced_exp4405"] == 0
    assert close["arc_new_levels_reproduced_exp4406"] == 0
    assert close["arc_tail_fidelity_ar25"] == 0.733333
    assert close["per_mechanic_unit_tests_passed_but_reproduction_not_proven"] is True
    assert close["flagged_for_v408"] == "agent2world_adaptive_e3_mechanic_repair_v408"
    assert close["v408_method_map_arxiv_ids"] == [
        "2512.22336",
        "2605.25931",
        "2605.13772",
        "2503.02863",
        "2508.02298",
    ]
    assert close["paper_ready"] is True
    assert close["v408_frame"] == mod.V408_FRAME
    assert artifact["field_principles"]["v407_close_state"] == mod.FIELD_PRINCIPLES["v407_close_state"]


def test_run_blocks_when_pretest_red_without_editing_history(tmp_path: Path) -> None:
    # SCENARIO-REPORT-4413-BLOCKED-PRECONDITION
    root = _make_root(tmp_path)
    before = (root / "research-complete.yaml").read_text(encoding="utf-8")

    artifact = json.loads(
        mod.run(root, pretest_result=RED, started_s=1000.0, now_s=1000.5).read_text(
            encoding="utf-8"
        )
    )

    assert artifact["honest_verdict"] == "blocked_smart_subset_pretest_not_green"
    assert artifact["preconditions_checked"]["smart_subset_pretest"]["green"] is False
    assert artifact["v407_close_state"] == {}
    assert (root / "research-complete.yaml").read_text(encoding="utf-8") == before


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda root: (root / "research-complete.yaml").unlink(), "blocked_research_complete_yaml_missing"),
        (
            lambda root: (root / "research-complete.yaml").write_text(
                "milestones: [", encoding="utf-8"
            ),
            "blocked_research_complete_yaml_poison",
        ),
        (lambda root: (root / "ops/exclusion_manifest.yaml").unlink(), "blocked_exclusion_manifest_missing"),
        (
            lambda root: (root / "ops/exclusion_manifest.yaml").write_text(
                "retired_extras: [", encoding="utf-8"
            ),
            "blocked_exclusion_manifest_yaml_poison",
        ),
        (
            lambda root: (root / "research-roadmap.yaml").write_text(
                "milestone: 2026.06.407\n", encoding="utf-8"
            ),
            "blocked_v408_not_active",
        ),
        (
            lambda root: (root / "results/experiment_4403_real_intervention_localizer_deconfound.json").unlink(),
            "blocked_real_intervention_localizer_missing",
        ),
    ],
)
def test_run_blocks_each_precondition_failure(tmp_path: Path, mutate: object, reason: str) -> None:
    # SCENARIO-REPORT-4413-BLOCKED-PRECONDITION
    root = _make_root(tmp_path)
    mutate(root)

    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5).read_text(
            encoding="utf-8"
        )
    )

    assert artifact["honest_verdict"] == reason
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["v407_close_state"] == {}


def test_run_blocks_if_research_complete_edit_would_not_parse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # SCENARIO-REPORT-4413-BLOCKED-PRECONDITION
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
    assert artifact["v407_close_state"] == {}


def test_validate_artifact_rejects_false_success() -> None:
    # REQ-REPORT-4413
    payload = mod.build_complete_artifact(
        v407_close_state=mod.build_v407_close_state(_sources()),
        preconditions_checked={"smart_subset_pretest": {"green": True}},
        duration_s=0.5,
        active_roadmap_path="research-roadmap.yaml",
        research_complete_record_action="updated",
        research_complete_duplicates_removed=0,
        cited_upstream_artifacts=[],
    )
    payload["v407_close_state"] = copy.deepcopy(payload["v407_close_state"])
    payload["v407_close_state"]["localizer_state"] = "not_retired"

    with pytest.raises(ValueError, match="localizer state"):
        mod.validate_artifact(payload)


def test_record_helpers_cover_append_and_unchanged_paths() -> None:
    # REQ-REPORT-4413
    close = mod.build_v407_close_state(_sources())

    appended, removed, action = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.406\n  finding: old\n", close
    )
    assert action == "appended"
    assert removed == 0
    assert mod.archive_record_count(appended) == 1

    unchanged, removed_again, action_again = mod.dedupe_or_update_record(appended, close)
    assert action_again == "unchanged"
    assert removed_again == 0
    assert unchanged == appended
    assert mod._insert_before_tasks(["- id: 2026.06.407"], "  finding: x")[-1] == "  finding: x"
    assert mod._ci95("not-a-ci", [1.0, 2.0]) == [1.0, 2.0]
    assert mod._domain([], "missing") == {}
    assert mod._card([], "missing") == {}


def test_run_updates_existing_activation_line_and_missing_finding(tmp_path: Path) -> None:
    # REQ-REPORT-4413
    root = _make_root(tmp_path)
    (root / "research-complete.yaml").write_text(
        (
            "milestones:\n"
            "- id: 2026.06.407\n"
            "  title: old\n"
            "  completed: '2026-06-18'\n"
            "  activation_recorded: stale\n"
            "  tasks:\n"
            "  - id: exp4412-capstone-v407\n"
            "    result: OK\n"
        ),
        encoding="utf-8",
    )

    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5).read_text(
            encoding="utf-8"
        )
    )
    text = (root / "research-complete.yaml").read_text(encoding="utf-8")

    assert artifact["research_complete_record_action"] == "updated"
    assert "activation_recorded: exp4413-archive-v407-activate-v408" in text
    assert "finding:" in text


def test_run_blocks_if_written_research_complete_turns_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # SCENARIO-REPORT-4413-BLOCKED-PRECONDITION
    root = _make_root(tmp_path)
    calls = {"n": 0}
    original = mod.yaml_parses

    def fails_after_write(text: str) -> bool:
        calls["n"] += 1
        if calls["n"] >= 4 and "position-bound" in text:
            return False
        return original(text)

    monkeypatch.setattr(mod, "yaml_parses", fails_after_write)
    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5).read_text(
            encoding="utf-8"
        )
    )

    assert artifact["honest_verdict"] == "blocked_research_complete_yaml_poison_after_edit"
    assert artifact["v407_close_state"] == {}


def test_module_main_delegates_to_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-REPORT-4413
    root = _make_root(tmp_path)
    monkeypatch.setattr(mod, "run", lambda _: root / "module-main-sentinel.json")

    assert mod.main(root) == 0


def test_script_runner_delegates_to_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-REPORT-4413
    root = _make_root(tmp_path)
    script_path = Path(__file__).parents[2] / "results/experiment_4413_archive_v407_activate_v408.py"
    script_repo_root = script_path.parents[1]
    removed = {str(script_repo_root), str(script_repo_root / "python")}
    monkeypatch.setattr(sys, "path", [item for item in sys.path if item not in removed])
    spec = importlib.util.spec_from_file_location("exp4413_runner", script_path)
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "run", lambda _: root / "sentinel.json")

    assert runner.main(root) == 0

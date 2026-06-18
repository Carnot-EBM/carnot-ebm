"""Tests for Exp 4402 `.406` archive / `.407` activation.

Spec refs: REQ-REPORT-4402, SCENARIO-REPORT-4402,
SCENARIO-REPORT-4402-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_v406_activate_v407_4402 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="88 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.405\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.406\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-18'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4401-capstone-v406\n"
        "    result: OK (conductor)\n"
    )
    return head + block * duplicates


def _manifest_text() -> str:
    return (
        "retired_extras:\n"
        "- id: cross_domain_selection_retired_exp4314_v399\n"
        "  experiment_ids: [exp4305, exp4314]\n"
        "  operator_reopen_required: true\n"
        "  retire_if_same_verdict: true\n"
        "- id: cross_game_value_transfer_retired_exp4342_v401\n"
        "  experiment_ids: [exp4318, exp4331, exp4342]\n"
        "  operator_reopen_required: true\n"
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
            "complete: v406_localizer_localizes_but_not_genuine_compounds_false_"
            "calibrated_false_arc_levels_34_publication_ready"
        ),
        "localizer_state": "localizes_but_not_genuine",
        "localizer_compounds": False,
        "detection_calibrated_multi_domain": False,
        "paper_ready": True,
        "publication_gate": {"paper_ready": True, "unmet_gates": []},
        "reproducible_total_levels": 34,
        "localizer": {
            "measurement": {
                "localizer_beats_ensemble_baseline": True,
                "localization_f1_by_domain": {
                    "FoVer": {
                        "synthetic_trained_localizer": 1.0,
                        "ensemble_baseline_0096": 0.096,
                        "delta": 0.904,
                        "delta_ci95": [0.904, 0.904],
                        "n_error_traces": 114,
                    },
                    "GAP-4 ARC": {
                        "synthetic_trained_localizer": 0.692308,
                        "ensemble_baseline_0096": 0.096,
                        "delta": 0.596308,
                        "delta_ci95": [0.461692, 0.711692],
                        "n_error_traces": 52,
                    },
                },
            },
            "skeptic_validation": {
                "localizer_win_is_genuine": False,
                "held_out_real_localization_delta_ci95": [0.904, 0.904],
            },
            "status": "localizes_but_not_genuine",
        },
        "self_learning": {
            "localizer_compounds": False,
            "compounding_delta_ci95": [0.0, 0.0],
            "learning_curve": [
                {"train_corpus_size": 566, "held_out_localization_f1": 1.0},
                {"train_corpus_size": 5661, "held_out_localization_f1": 1.0},
            ],
        },
        "calibration": {
            "detection_calibrated_multi_domain": False,
            "detection_by_domain": [
                {
                    "domain": "code_humaneval",
                    "n": 100,
                    "detection_auroc": 0.9808,
                    "claim_scope": "underpowered_n=100; report_n_only_scope_claim",
                    "base_rate": 0.75,
                    "ece_lodo_calibrated": 0.06406,
                    "ece_uncalibrated": 0.1205,
                },
                {"domain": "gap4_arc", "n": 28443, "detection_auroc": 0.963317},
            ],
        },
        "arc_reproducible_progress": {
            "reproducible_total_levels": 34,
            "reproducible_total_games": 17,
            "new_levels_since_prior": 0,
            "new_games_since_prior": 0,
        },
        "arc_e3_outcomes": {
            "deeper_high_headroom": {
                "new_levels_reproduced": 0,
                "per_target_scorecard": [
                    {"game": "lp85", "lookahead_fidelity": 0.833333},
                    {"game": "tu93", "lookahead_fidelity": 0.8},
                    {"game": "tn36", "lookahead_fidelity": 0.875},
                    {"game": "tr87", "lookahead_fidelity": 0.857143},
                ],
            },
            "blocked_mechanics": {
                "new_levels_reproduced": 0,
                "per_game_scorecard": [
                    {"game": "ar25", "lookahead_fidelity": 0.733333},
                    {"game": "ka59", "lookahead_fidelity": 0.112281},
                    {"game": "ft09", "lookahead_fidelity": 0.347518},
                ],
            },
        },
    }
    payload.update(overrides)
    return payload


def _localizer_measurement() -> dict:
    return {
        "honest_verdict": "success: synthetic_process_localizer_beats_ensemble_baseline",
        "localizer_beats_ensemble_baseline": True,
        "model_specs": {"exp405_ensemble_baseline_first_error_f1": 0.096},
    }


def _skeptic() -> dict:
    return {
        "honest_verdict": "complete: a1_win_quarantined_as_artifact_confounded",
        "localizer_win_is_genuine": False,
        "beats_position_only_baseline": False,
        "template_ablation_drop": 0.0,
        "held_out_real_localization_delta_ci95": [0.904, 0.904],
        "diagnostics": {
            "position_only_baseline": {
                "position_only_f1": 1.0,
                "a1_f1": 1.0,
                "beats_position_only_baseline": False,
            },
            "template_ablation": {
                "a1_f1": 1.0,
                "template_ablated_f1": 1.0,
                "drop": 0.0,
            },
        },
    }


def _compounds() -> dict:
    return {
        "honest_verdict": "complete: clean_saturated_null_localizer",
        "localizer_compounds": False,
        "compounding_delta_ci95": [0.0, 0.0],
        "positive_control_passed": True,
        "learning_curve": [
            {"train_corpus_size": 566, "held_out_localization_f1": 1.0},
            {"train_corpus_size": 1415, "held_out_localization_f1": 1.0},
            {"train_corpus_size": 2830, "held_out_localization_f1": 1.0},
            {"train_corpus_size": 5661, "held_out_localization_f1": 1.0},
        ],
    }


def _calibration() -> dict:
    return {
        "honest_verdict": "complete: calibrated_multi_domain_contract_false",
        "detection_calibrated_multi_domain": False,
        "positive_control_passed": True,
        "detection_by_domain": [
            {
                "domain": "code_humaneval",
                "n": 100,
                "detection_auroc": 0.9808,
                "claim_scope": "underpowered_n=100; report_n_only_scope_claim",
                "base_rate": 0.75,
                "ece_lodo_calibrated": 0.06406,
                "ece_uncalibrated": 0.1205,
            },
            {"domain": "fover", "n": 8829, "detection_auroc": 0.918304},
            {"domain": "gap4_arc", "n": 28443, "detection_auroc": 0.963317},
            {"domain": "gsm8k", "n": 1600, "detection_auroc": 0.990196},
        ],
        "model_specs": {
            "cached_corpora": {
                "code_humaneval": {
                    "n": 100,
                    "claim_scope": "underpowered_n=100; report_n_only_scope_claim",
                    "base_rate": 0.75,
                }
            }
        },
    }


def _sota() -> dict:
    return {
        "honest_verdict": "complete: sota_ingestion_v407_mapped",
        "flagged_for_v407": "intervention_active_real_first_error_deconfounding_v407",
        "methods_mapped": [
            {"arxiv_id_or_url": "2601.14209"},
            {"arxiv_id_or_url": "2603.25412"},
            {"arxiv_id_or_url": "2504.10559"},
            {"arxiv_id_or_url": "2602.07842"},
            {"arxiv_id_or_url": "2606.16070"},
        ],
    }


def _arc_deeper() -> dict:
    return {
        "honest_verdict": "complete_e3_deeper_partial",
        "new_levels_reproduced": 0,
        "per_target_scorecard": [
            {"game": "lp85", "lookahead_fidelity": 0.833333, "fidelity_gate_passed": False},
            {"game": "tu93", "lookahead_fidelity": 0.8, "fidelity_gate_passed": False},
            {"game": "tn36", "lookahead_fidelity": 0.875, "fidelity_gate_passed": False},
            {"game": "tr87", "lookahead_fidelity": 0.857143, "fidelity_gate_passed": False},
        ],
    }


def _arc_tails() -> dict:
    return {
        "honest_verdict": "complete_e3_ar25_ka59_ft09_partial",
        "new_levels_reproduced": 0,
        "per_game_scorecard": [
            {"game": "ar25", "lookahead_fidelity": 0.733333, "fidelity_gate_passed": False},
            {"game": "ka59", "lookahead_fidelity": 0.112281, "fidelity_gate_passed": False},
            {"game": "ft09", "lookahead_fidelity": 0.347518, "fidelity_gate_passed": False},
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
        "milestone: 2026.06.407\nmilestone_doc: openspec/change-proposals/research-roadmap-v407.md\n",
        encoding="utf-8",
    )
    (tmp_path / "openspec/change-proposals/research-roadmap-v407.md").write_text(
        "# Research Roadmap v407\n\nREAL intervention data and per-mechanic executable unit tests.\n",
        encoding="utf-8",
    )
    _write_json(tmp_path / "results/experiment_4401_capstone_v406.json", _capstone())
    _write_json(
        tmp_path / "results/experiment_4392_verifiable_process_data_localizer.json",
        _localizer_measurement(),
    )
    _write_json(tmp_path / "results/experiment_4393_localizer_skeptic_proof.json", _skeptic())
    _write_json(
        tmp_path / "results/experiment_4396_localizer_self_learning_compounds.json",
        _compounds(),
    )
    _write_json(
        tmp_path / "results/experiment_4397_cross_domain_detection_calibration.json",
        _calibration(),
    )
    _write_json(tmp_path / "results/experiment_4398_sota_ingestion_v407.json", _sota())
    _write_json(tmp_path / "results/experiment_4394_e3_deeper_fidelity_gate.json", _arc_deeper())
    _write_json(
        tmp_path / "results/experiment_4395_e3_blocked_mechanic_tails_ar25_ka59_ft09.json",
        _arc_tails(),
    )
    _write_json(
        tmp_path / "results/experiment_4374_diffusiongemma_scorer_repair_or_retire.json",
        {"honest_verdict": "retired_in_generation_conversion_unmeasurable"},
    )
    _write_json(
        tmp_path / "results/experiment_4370_llm_generated_action_cost_heuristics.json",
        {"honest_verdict": "complete: llm_heuristic_clean_null_linear_settled"},
    )
    return tmp_path


def test_run_archives_v406_and_records_true_close_state(tmp_path: Path) -> None:
    # REQ-REPORT-4402 / SCENARIO-REPORT-4402
    root = _make_root(tmp_path, duplicates=2)

    out = mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5)
    artifact = json.loads(out.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith(("complete:", "success:", "passed:", "shipped:"))
    assert artifact["archived_milestone"] == "2026.06.406"
    assert artifact["activated_milestone"] == "2026.06.407"
    assert artifact["active_milestone_confirmed"] == "2026.06.407"
    assert artifact["pretest_suite_green"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["research_complete_duplicates_removed"] == 1
    assert mod.archive_record_count((root / "research-complete.yaml").read_text(encoding="utf-8")) == 1
    assert "PURE POSITION BIAS" in (root / "research-complete.yaml").read_text(encoding="utf-8")

    close = artifact["v406_close_state"]
    assert close["localizer_axis_state"] == "LOCALIZES_BUT_NOT_GENUINE_SYNTHETIC_POSITION_BIAS"
    assert close["synthetic_localizer_quarantined"] is True
    assert close["localizer_win_is_genuine"] is False
    assert close["beats_position_only_baseline"] is False
    assert close["template_ablation_drop"] == 0.0
    assert close["localizer_compounds"] is False
    assert close["self_learning_axis_state"] == "SATURATED_NULL"
    assert close["self_learning_f1_first"] == 1.0
    assert close["self_learning_f1_last"] == 1.0
    assert close["detection_calibrated_multi_domain"] is False
    assert close["code_humaneval_n"] == 100
    assert close["arc_reproducible_total_levels"] == 34
    assert close["arc_reproducible_total_games"] == 17
    assert close["lookahead_fidelity_min_headline"] == 0.733333
    assert close["lookahead_fidelity_max_headline"] == 0.875
    assert close["flagged_for_v407"] == "intervention_active_real_first_error_deconfounding_v407"
    assert close["cross_game_value_transfer_axis_state"] == "RETIRED_EXP4318_EXP4331_EXP4342"
    assert close["cross_domain_selection_axis_state"] == "RETIRED_EXP4314_DOMAIN_BOUND"
    assert close["in_generation_diffusiongemma_axis_state"] == "RETIRED_EXP4374_FOURTH_BLOCK"
    assert close["llm_heuristic_efficiency_axis_state"] == "SETTLED_EXP4370_CLEAN_NULL"
    assert close["paper_ready"] is True
    assert close["v407_frame"] == mod.V407_FRAME
    assert artifact["field_principles"]["v406_close_state"] == mod.FIELD_PRINCIPLES["v406_close_state"]


def test_run_blocks_when_pretest_red_without_editing_history(tmp_path: Path) -> None:
    # SCENARIO-REPORT-4402-BLOCKED-PRECONDITION
    root = _make_root(tmp_path)
    before = (root / "research-complete.yaml").read_text(encoding="utf-8")

    artifact = json.loads(
        mod.run(root, pretest_result=RED, started_s=1000.0, now_s=1000.5).read_text(
            encoding="utf-8"
        )
    )

    assert artifact["honest_verdict"] == "blocked_smart_subset_pretest_not_green"
    assert artifact["preconditions_checked"]["smart_subset_pretest"]["green"] is False
    assert artifact["v406_close_state"] == {}
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
                "milestone: 2026.06.406\n", encoding="utf-8"
            ),
            "blocked_v407_not_active",
        ),
        (
            lambda root: (root / "results/experiment_4393_localizer_skeptic_proof.json").unlink(),
            "blocked_localizer_skeptic_proof_missing",
        ),
    ],
)
def test_run_blocks_each_precondition_failure(
    tmp_path: Path, mutate: object, reason: str
) -> None:
    # SCENARIO-REPORT-4402-BLOCKED-PRECONDITION
    root = _make_root(tmp_path)
    mutate(root)

    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5).read_text(
            encoding="utf-8"
        )
    )

    assert artifact["honest_verdict"] == reason
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["v406_close_state"] == {}


def test_run_blocks_if_research_complete_edit_would_not_parse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # SCENARIO-REPORT-4402-BLOCKED-PRECONDITION
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
    assert artifact["v406_close_state"] == {}


def test_run_blocks_if_written_research_complete_turns_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # SCENARIO-REPORT-4402-BLOCKED-PRECONDITION
    root = _make_root(tmp_path)
    calls = {"n": 0}
    original = mod.yaml_parses

    def fails_after_write(text: str) -> bool:
        calls["n"] += 1
        if calls["n"] >= 4 and "PURE POSITION BIAS" in text:
            return False
        return original(text)

    monkeypatch.setattr(mod, "yaml_parses", fails_after_write)
    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=1000.0, now_s=1000.5).read_text(
            encoding="utf-8"
        )
    )

    assert artifact["honest_verdict"] == "blocked_research_complete_yaml_poison_after_edit"
    assert artifact["v406_close_state"] == {}


def test_validate_artifact_rejects_false_success() -> None:
    # REQ-REPORT-4402
    payload = mod.build_complete_artifact(
        v406_close_state=mod.build_v406_close_state(
            {
                "4401": _capstone(),
                "4392": _localizer_measurement(),
                "4393": _skeptic(),
                "4396": _compounds(),
                "4397": _calibration(),
                "4398": _sota(),
                "4394": _arc_deeper(),
                "4395": _arc_tails(),
                "arc_solve_registry": {
                    "reproducible_total_levels": 34,
                    "reproducible_total_games": 17,
                },
                "exclusion_manifest": {},
            }
        ),
        preconditions_checked={"smart_subset_pretest": {"green": True}},
        duration_s=0.5,
        active_roadmap_path="research-roadmap.yaml",
        research_complete_record_action="updated",
        research_complete_duplicates_removed=0,
        cited_upstream_artifacts=[],
    )
    payload["v406_close_state"] = copy.deepcopy(payload["v406_close_state"])
    payload["v406_close_state"]["template_ablation_drop"] = 0.5

    with pytest.raises(ValueError, match="template ablation"):
        mod.validate_artifact(payload)


def test_record_helpers_cover_append_unchanged_and_missing_task_paths() -> None:
    # REQ-REPORT-4402
    close = mod.build_v406_close_state(
        {
            "4401": _capstone(),
            "4392": _localizer_measurement(),
            "4393": _skeptic(),
            "4396": _compounds(),
            "4397": _calibration(),
            "4398": _sota(),
            "4394": _arc_deeper(),
            "4395": _arc_tails(),
            "arc_solve_registry": {
                "reproducible_total_levels": 34,
                "reproducible_total_games": 17,
            },
            "exclusion_manifest": {},
        }
    )

    appended, removed, action = mod.dedupe_or_update_record(
        "milestones:\n- id: 2026.06.405\n  finding: old\n", close
    )
    assert action == "appended"
    assert removed == 0
    assert mod.archive_record_count(appended) == 1

    unchanged, removed_again, action_again = mod.dedupe_or_update_record(appended, close)
    assert action_again == "unchanged"
    assert removed_again == 0
    assert unchanged == appended

    inserted = mod._insert_before_tasks(["- id: 2026.06.406"], "  finding: x")
    assert inserted[-1] == "  finding: x"
    assert mod._ci95("not-a-ci", [1.0, 2.0]) == [1.0, 2.0]


def test_run_updates_existing_activation_line_and_missing_finding(tmp_path: Path) -> None:
    # REQ-REPORT-4402
    root = _make_root(tmp_path)
    (root / "research-complete.yaml").write_text(
        (
            "milestones:\n"
            "- id: 2026.06.406\n"
            "  title: old\n"
            "  completed: '2026-06-18'\n"
            "  activation_recorded: stale\n"
            "  tasks:\n"
            "  - id: exp4401-capstone-v406\n"
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
    assert "activation_recorded: exp4402-archive-v406-activate-v407" in text
    assert "finding:" in text


def test_module_main_delegates_to_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-REPORT-4402
    root = _make_root(tmp_path)
    monkeypatch.setattr(mod, "run", lambda _: root / "module-main-sentinel.json")

    assert mod.main(root) == 0


def test_script_runner_delegates_to_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # REQ-REPORT-4402
    root = _make_root(tmp_path)
    script_path = Path(__file__).parents[2] / "results/experiment_4402_archive_v406_activate_v407.py"
    spec = importlib.util.spec_from_file_location("exp4402_runner", script_path)
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "run", lambda _: root / "sentinel.json")

    assert runner.main(root) == 0

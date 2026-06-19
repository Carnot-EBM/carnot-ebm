"""Tests for Exp 4466 `.412` archive / `.413` activation.

Spec refs: REQ-REPORT-4466, SCENARIO-REPORT-4466,
SCENARIO-REPORT-4466-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_412_activate_413_4466 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="17 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.411\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.412\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-19'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4465-capstone-412\n"
        "    result: OK (conductor)\n"
    )
    return head + block * duplicates


def _registry_text() -> str:
    return (
        "schema_version: 1\n"
        "updated: '2026-06-19'\n"
        "reproducible_total_levels: 39\n"
        "reproducible_total_games: 20\n"
        "latest_hygiene_4449:\n"
        "  reproducible_total_levels: 39\n"
        "  reproducible_total_games: 20\n"
        "latest_hygiene_4461:\n"
        "  reproducible_total_levels: 39\n"
        "  reproducible_total_games: 20\n"
        "  filled_gap_ids:\n"
        "  - GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER\n"
        "  open_gap_ids:\n"
        "  - GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT\n"
        "  - GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER\n"
        "  - GAP-4458-SB26-COLOR-MATCH-SLOT-SEQUENCE\n"
        "latest_submission_package_4460:\n"
        "  submission_package_ready: true\n"
        "  total_reproduced_levels_in_package: 39\n"
        "  submitted_to_leaderboard: false\n"
    )


def _capstone(**overrides: object) -> dict:
    payload = {
        "honest_verdict": (
            "complete: v412_generic_solver_partial_loo_v3_6_levels_39_games_20_"
            "submission_ready_publication_ready"
        ),
        "generic_solver_gap_state": "partial",
        "generic_loo_solve_count_v3": 6,
        "reproducible_total_levels": 39,
        "reproducible_total_games": 20,
        "flagged_artifacts_excluded": [],
        "dc22_bank": {
            "honest_verdict": "blocked_baseline_pytest_coverage",
            "dc22_grounded": False,
            "gap_closed": False,
            "reproduced_levels": 0,
            "missing_verifier_gaps": ["precondition_baseline_pytest_failed_coverage_gate"],
            "verifier_is_oracle": False,
        },
        "glyph_rewrite": {
            "closed_gap_ids": [
                "GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER"
            ],
            "gap_closed": True,
            "tr87_resolved_generically": True,
            "tr87_generic_level_reproduced": 1,
            "verifier_is_oracle": True,
        },
        "loo_v3": {
            "generic_loo_solve_count_v2_baseline": 5,
            "generic_loo_solve_count_v3": 6,
            "loo_gate_passed": True,
            "missing_verifier_gaps": [
                {
                    "game": "sc25",
                    "residual_delta": "missing_cast_grid_spell_shrink_tank_exit_verifier",
                }
            ],
            "verifier_is_oracle": True,
        },
        "cast_grid": {
            "state": "missing_or_excluded",
            "sc25_gap_closed": False,
            "sc25_level_banked": False,
        },
        "first_contact_new_game": {
            "target_game": "sb26",
            "banked_new_game": False,
            "missing_verifier_gaps": [
                {
                    "game": "sb26",
                    "gap_id": "GAP-4458-SB26-MISSING-COLOR-MATCH-SLOT-SEQUENCE-VERIFIER",
                    "residual_delta": "missing_color_match_slot_sequence_verifier",
                }
            ],
            "verifier_is_oracle": True,
        },
        "submission_package": {
            "submission_package_ready": True,
            "total_reproduced_levels_in_package": 39,
            "submitted_to_leaderboard": False,
            "verifier_is_oracle": True,
        },
        "publication_gate": {
            "paper_ready": True,
            "unmet_gates": [],
            "gates": {"G1": {"pass": True}, "G2": {"pass": True}, "G3": {"pass": True}, "G4": {"pass": True}},
        },
        "paper_ready": True,
        "next_backlog": {
            "open_gap_ids": [
                "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT",
                "GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER",
                "GAP-4458-SB26-COLOR-MATCH-SLOT-SEQUENCE",
            ],
            "missing_primitives": [
                "config_rule_verifier_grounding",
                "cast_grid_spell_shrink_tank_exit_verifier",
                "color_match_slot_sequence_verifier",
            ],
        },
        "submission_package_ready": True,
        "submitted_to_leaderboard": False,
        "verifier_is_oracle": False,
    }
    payload.update(overrides)
    return payload


def _make_root(tmp_path: Path, *, duplicates: int = 1, with_next: bool = True) -> Path:
    (tmp_path / "ops").mkdir(parents=True)
    (tmp_path / "results").mkdir(parents=True)
    (tmp_path / "research-complete.yaml").write_text(
        _research_complete_text(duplicates=duplicates), encoding="utf-8"
    )
    (tmp_path / "ops/exclusion_manifest.yaml").write_text(
        "retired_extras:\n- id: circular_arc_solve_not_moat\n", encoding="utf-8"
    )
    (tmp_path / "ops/arc_solve_registry.yaml").write_text(_registry_text(), encoding="utf-8")
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.06.413\n", encoding="utf-8")
    if with_next:
        (tmp_path / "research-roadmap-next.yaml").write_text(
            "milestone: 2026.06.413\ntasks: []\n", encoding="utf-8"
        )
    _write_json(tmp_path / "results/experiment_4465_capstone_v412.json", _capstone())
    return tmp_path


def test_req_report_4466_spec_anchor_declares_required_contract() -> None:
    """REQ-REPORT-4466: OpenSpec declares the archive contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-4466" in spec
    assert "SCENARIO-REPORT-4466" in spec
    assert "results/experiment_4466_archive_412_activate_413.json" in spec
    assert "research-roadmap-next.yaml" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["reproducible_total_levels"] in spec
    assert mod.FIELD_PRINCIPLES["open_gap_ids"] in spec
    assert mod.FIELD_PRINCIPLES["prior_milestone_churn_note"] in spec
    assert mod.FIELD_PRINCIPLES["inference_substrate"] in spec


def test_run_archives_v412_and_records_true_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4466: complete path records .412 truth."""

    root = _make_root(tmp_path, duplicates=2)
    output = mod.run(root, pretest_result=GREEN, started_s=10.0, now_s=10.5)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith(mod.TERMINAL_PREFIXES)
    assert artifact["archived_milestone"] == "2026.06.412"
    assert artifact["activated_milestone"] == "2026.06.413"
    assert artifact["active_milestone_confirmed"] == "2026.06.413"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["research_roadmap_next_yaml_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["trm_training_ran"] is False
    assert artifact["leaderboard_submission"] is False
    assert artifact["no_3090_inference"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["reproducible_total_levels"] == 39
    assert artifact["reproducible_total_games"] == 20
    assert artifact["open_gap_ids"] == [
        "GAP-4423-DC22",
        "GAP-4432-LOO-SC25",
        "GAP-4458-SB26",
    ]
    assert "ZERO new reproduced levels" in artifact["prior_milestone_churn_note"]

    close = artifact["v412_close_state"]
    assert close["generic_solver_gap_state"] == "partial"
    assert close["generic_loo_solve_count_v2_baseline"] == 5
    assert close["generic_loo_solve_count_v3"] == 6
    assert close["reproducible_total_levels"] == 39
    assert close["reproducible_total_games"] == 20
    assert close["reproducible_levels_delta_vs_v411"] == 0
    assert close["zero_new_reproduced_levels"] is True
    assert close["closed_gap_ids"] == ["GAP-4432-LOO-TR87"]
    assert close["submission_package_ready"] is True
    assert close["submission_package_levels"] == 39
    assert close["submitted_to_leaderboard"] is False
    assert close["paper_ready"] is True
    assert close["fover_auc"] == 0.9131
    assert close["publication_gate"] == "G1-G4 FROZEN"
    assert close["verifier_is_oracle_honored"] is True
    assert close["execution_grounded_arc_solve_not_moat_headline"] is True

    gaps = {gap["gap_id"]: gap for gap in artifact["open_gap_failure_modes"]}
    assert gaps["GAP-4423-DC22"]["failure_mode"] == "blocked_baseline_pytest_coverage"
    assert gaps["GAP-4423-DC22"]["dc22_cegis_solve_ran"] is False
    assert gaps["GAP-4432-LOO-SC25"]["failure_mode"] == "no_artifact_produced"
    assert gaps["GAP-4432-LOO-SC25"]["provisional_l2_l5_banked"] is False
    assert gaps["GAP-4458-SB26"]["failure_mode"] == "missing_color_match_slot_sequence_verifier"

    history = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(history) == 1
    assert "activation_recorded: exp4466-archive-412-activate-413" in history
    assert "generic_solver_gap_state=partial; generic_loo_solve_count_v3=6" in history
    assert "ZERO new reproduced levels" in history


def test_missing_roadmap_next_blocks_before_history_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4466-BLOCKED-PRECONDITION: missing YAML blocks."""

    root = _make_root(tmp_path, with_next=False)
    before = (root / "research-complete.yaml").read_text(encoding="utf-8")
    output = mod.run(root, pretest_result=GREEN, started_s=1.0, now_s=1.1)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"] == "blocked_yaml_parse"
    assert artifact["research_roadmap_next_yaml_parses"] is False
    assert artifact["pretest_suite_green"] is False
    assert artifact["v412_close_state"] == {}
    assert (root / "research-complete.yaml").read_text(encoding="utf-8") == before


def test_smart_subset_red_blocks_without_fabricating_success(tmp_path: Path) -> None:
    """REQ-REPORT-4466: red smart-subset gate blocks complete artifact."""

    root = _make_root(tmp_path)
    output = mod.run(root, pretest_result=RED, started_s=1.0, now_s=1.1)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"] == "blocked_smart_subset_pretest_not_green"
    assert artifact["pretest_suite_green"] is False
    assert artifact["open_gap_ids"] == []


def test_flagged_artifacts_are_reported_but_not_promoted(tmp_path: Path) -> None:
    """REQ-REPORT-4466: flagged upstream evidence is skipped."""

    root = _make_root(tmp_path)
    capstone = _capstone(
        flagged_artifacts_excluded=[
            {
                "experiment_id": 9999,
                "reason": "flagged_adversarial",
                "path": "results/experiment_9999_bad.json",
            }
        ]
    )
    _write_json(root / "results/experiment_4465_capstone_v412.json", capstone)

    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=2.0, now_s=2.5).read_text(
            encoding="utf-8"
        )
    )

    assert artifact["flagged_artifacts_skipped"] == [9999]
    assert artifact["v412_close_state"]["flagged_artifacts_skipped"] == [9999]
    assert artifact["reproducible_total_levels"] == 39


def test_validate_artifact_rejects_hidden_leaderboard_submission(tmp_path: Path) -> None:
    """REQ-REPORT-4466: validation rejects fields that hide transition risk."""

    root = _make_root(tmp_path)
    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=3.0, now_s=3.5).read_text(
            encoding="utf-8"
        )
    )
    altered = copy.deepcopy(artifact)
    altered["leaderboard_submission"] = True

    with pytest.raises(ValueError, match="training and leaderboard submission"):
        mod.validate_artifact(altered)


def test_script_entrypoint_exists() -> None:
    """SCENARIO-REPORT-4466: the requested script entrypoint exists."""

    script = Path("python/carnot/experiment_4466_archive_412_activate_413.py")

    assert script.exists()
    assert "archive_412_activate_413_4466" in script.read_text(encoding="utf-8")

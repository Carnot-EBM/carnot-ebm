"""Tests for Exp 4442 `.410` archive / `.411` activation.

Spec refs: REQ-REPORT-4442, SCENARIO-REPORT-4442,
SCENARIO-REPORT-4442-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_410_activate_411_4442 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="44 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.409\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.410\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-19'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4441-capstone-410\n"
        "    result: OK (conductor)\n"
    )
    return head + block * duplicates


def _registry_text() -> str:
    return (
        "schema_version: 1\n"
        "updated: '2026-06-19'\n"
        "reproducible_total_levels: 37\n"
        "reproducible_total_games: 18\n"
        "games:\n"
        "- game: tu93\n"
        "  reproducibility: reproduced\n"
        "  levels_reproduced: 5\n"
        "- game: g50t\n"
        "  reproducibility: unsolved\n"
        "  levels_reproduced: 0\n"
    )


def _residuals() -> list[dict]:
    return [
        {"game": "tr87", "residual_delta": "missing_glyph_rewrite_rule_verifier_without_tr87_adapter"},
        {"game": "sc25", "residual_delta": "missing_cast_grid_spell_shrink_tank_exit_verifier"},
        {"game": "ka59", "residual_delta": "missing_push_block_world_model_and_dynamic_selection"},
        {"game": "ar25", "residual_delta": "missing_reflection_world_model_and_object_motion_plan"},
        {"game": "ft09", "residual_delta": "missing_local_constraint_color_cycle_verifier"},
    ]


def _capstone(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: v410_generic_solver_partial_loo_2_residuals_5_levels_37_publication_ready",
        "generic_solver_gap_state": "partial",
        "generic_loo_solve_count": 2,
        "reproducible_total_levels": 37,
        "verifier_is_oracle": False,
        "flagged_artifacts_excluded": [
            {
                "artifact_key": "4433_win_induction",
                "experiment_id": 4433,
                "path": "results/experiment_4433_example_conditioned_win_induction.json",
                "reason": "flagged_adversarial",
                "live_critical_flags": [{"kind": "DURATION_TOO_SHORT"}],
                "stamped_flagged_adversarial": True,
            }
        ],
        "headline_question_answers": {
            "exp4432": {
                "generic_loo_solve_count": 2,
                "residual_delta_count": 5,
                "residual_deltas": _residuals(),
            },
            "exp4433": {
                "few_shot_examples_demonstrably_helped": False,
                "held_out_level_banked": False,
                "state": "excluded_flagged_adversarial",
            },
            "exp4434": {
                "accuracy_delta": 0.285714,
                "helped_vs_cold_control": True,
                "state": "examples_helped_no_reproduced_level",
            },
            "exp4435": {
                "routed_solve_banked": False,
                "state": "contract_fixed_no_routed_solve",
                "verdict_contract_fixed": True,
            },
            "exp4436": {
                "no_regression": True,
                "primitives_consolidated_count": 5,
                "state": "consolidated_no_regression",
            },
        },
        "action_model": {
            "accuracy_delta": 0.285714,
            "helped_vs_cold_control": True,
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "state": "examples_helped_no_reproduced_level",
            "world_model_accuracy_cold": 0.714286,
            "world_model_accuracy_with_examples": 1.0,
        },
        "first_contact": {
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "routed_solve_banked": False,
            "state": "contract_fixed_no_routed_solve",
            "target_game": "dc22",
            "verdict_contract_fixed": True,
        },
        "primitives": {
            "count": 5,
            "deepened_game": "tu93",
            "new_levels_reproduced": 1,
            "no_regression": True,
            "state": "consolidated_no_regression",
        },
        "residual_deltas": _residuals(),
    }
    payload.update(overrides)
    return payload


def _make_root(tmp_path: Path, *, duplicates: int = 1) -> Path:
    (tmp_path / "ops").mkdir(parents=True)
    (tmp_path / "results").mkdir(parents=True)
    (tmp_path / "research-complete.yaml").write_text(
        _research_complete_text(duplicates=duplicates), encoding="utf-8"
    )
    (tmp_path / "ops/exclusion_manifest.yaml").write_text(
        "retired_extras:\n- id: circular_arc_solve_not_moat\n", encoding="utf-8"
    )
    (tmp_path / "ops/arc_solve_registry.yaml").write_text(_registry_text(), encoding="utf-8")
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.06.411\n", encoding="utf-8")
    (tmp_path / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.06.411\ntasks: []\n", encoding="utf-8"
    )
    _write_json(tmp_path / "results/experiment_4441_capstone_v410.json", _capstone())
    return tmp_path


def test_req_report_4442_spec_anchor_declares_required_contract() -> None:
    """REQ-REPORT-4442: OpenSpec declares the archive contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-4442" in spec
    assert "SCENARIO-REPORT-4442" in spec
    assert "results/experiment_4442_archive_410_activate_411.json" in spec
    assert "research-roadmap-next.yaml" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["reproducible_total_levels"] in spec
    assert mod.FIELD_PRINCIPLES["inference_substrate"] in spec


def test_run_archives_v410_and_records_true_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4442: complete path records .410 truth."""

    root = _make_root(tmp_path, duplicates=2)
    output = mod.run(root, pretest_result=GREEN, started_s=10.0, now_s=10.5)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith(mod.TERMINAL_PREFIXES)
    assert artifact["archived_milestone"] == "2026.06.410"
    assert artifact["activated_milestone"] == "2026.06.411"
    assert artifact["active_milestone_confirmed"] == "2026.06.411"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["research_roadmap_next_yaml_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["trm_training_ran"] is False
    assert artifact["leaderboard_submission"] is False
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["reproducible_total_levels"] == 37
    assert artifact["reproducible_total_games"] == 18

    close = artifact["v410_close_state"]
    assert close["generic_solver_gap_state"] == "partial"
    assert close["generic_loo_solve_count"] == 2
    assert close["generic_loo_target_count"] == 7
    assert [row["game"] for row in close["residual_deltas"]] == [
        "tr87",
        "sc25",
        "ka59",
        "ar25",
        "ft09",
    ]
    assert close["action_model"]["accuracy_delta"] == 0.285714
    assert close["action_model"]["helped_vs_cold_control"] is True
    assert close["action_model"]["offline_reproduced"] is False
    assert close["first_contact"]["verdict_contract_fixed"] is True
    assert close["first_contact"]["routed_solve_banked"] is False
    assert close["primitives"]["deepened_game"] == "tu93"
    assert close["primitives"]["new_levels_reproduced"] == 1
    assert close["primitives"]["count"] == 5
    assert close["primitives"]["no_regression"] is True
    assert close["g50t_l1_quarantine"]["offline_reproduced"] is True
    assert close["g50t_l1_quarantine"]["trusted_for_aggregation"] is False
    assert close["g50t_l1_quarantine"]["substrate_declaration_false_positive"] is True
    assert close["flagged_artifacts_skipped"] == [4433]
    assert close["verifier_is_oracle_honored"] is True

    history = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(history) == 1
    assert "activation_recorded: exp4442-archive-410-activate-411" in history
    assert "generic_solver_gap_state=partial; generic_loo_solve_count=2/7" in history


def test_flagged_win_induction_cannot_be_banked_from_capstone_claim(tmp_path: Path) -> None:
    """REQ-REPORT-4442: flagged Exp 4433 evidence stays quarantined."""

    root = _make_root(tmp_path)
    capstone = _capstone()
    altered = copy.deepcopy(capstone["headline_question_answers"])
    altered["exp4433"] = {
        "few_shot_examples_demonstrably_helped": True,
        "held_out_level_banked": True,
        "state": "claimed_win_but_flagged",
    }
    _write_json(
        root / "results/experiment_4441_capstone_v410.json",
        _capstone(headline_question_answers=altered),
    )

    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=1.0, now_s=1.25).read_text(
            encoding="utf-8"
        )
    )

    quarantine = artifact["v410_close_state"]["g50t_l1_quarantine"]
    assert quarantine["held_out_level_banked"] is False
    assert quarantine["trusted_for_aggregation"] is False
    assert quarantine["reason"] == "flagged_adversarial_DURATION_TOO_SHORT_inference_substrate_none"


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda root: (root / "research-roadmap-next.yaml").unlink(),
            "blocked_yaml_parse",
        ),
        (
            lambda root: (root / "ops/exclusion_manifest.yaml").write_text(
                "retired_extras: [", encoding="utf-8"
            ),
            "blocked_yaml_parse",
        ),
        (
            lambda root: (root / "research-roadmap.yaml").write_text(
                "milestone: 2026.06.410\n", encoding="utf-8"
            ),
            "blocked_v411_not_active",
        ),
        (
            lambda root: (root / "results/experiment_4441_capstone_v410.json").unlink(),
            "blocked_v410_capstone_missing",
        ),
        (
            lambda root: (root / "ops/arc_solve_registry.yaml").write_text(
                "reproducible_total_levels: [", encoding="utf-8"
            ),
            "blocked_arc_solve_registry_yaml_parse",
        ),
    ],
)
def test_run_blocks_precondition_failures(tmp_path: Path, mutate: object, reason: str) -> None:
    """SCENARIO-REPORT-4442-BLOCKED-PRECONDITION: failures do not archive."""

    root = _make_root(tmp_path)
    mutate(root)
    history_before = (root / "research-complete.yaml").read_text(encoding="utf-8")

    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=2.0, now_s=2.2).read_text(
            encoding="utf-8"
        )
    )

    assert artifact["honest_verdict"] == reason
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["v410_close_state"] == {}
    assert (root / "research-complete.yaml").read_text(encoding="utf-8") == history_before


def test_red_smart_subset_blocks_without_history_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4442-BLOCKED-PRECONDITION: red pre-test gate blocks."""

    root = _make_root(tmp_path)
    history_before = (root / "research-complete.yaml").read_text(encoding="utf-8")

    artifact = json.loads(
        mod.run(root, pretest_result=RED, started_s=3.0, now_s=3.5).read_text(
            encoding="utf-8"
        )
    )

    assert artifact["honest_verdict"] == "blocked_smart_subset_pretest_not_green"
    assert artifact["pretest_suite_green"] is False
    assert (root / "research-complete.yaml").read_text(encoding="utf-8") == history_before


def test_no_existing_record_appends_canonical_archive(tmp_path: Path) -> None:
    """REQ-REPORT-4442: missing .410 history row is appended canonically."""

    root = _make_root(tmp_path, duplicates=0)
    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=4.0, now_s=4.3).read_text(
            encoding="utf-8"
        )
    )

    history = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert artifact["research_complete_record_action"] == "appended"
    assert mod.archive_record_count(history) == 1
    assert "Archive .410 and activate .411" in history


def test_helpers_cover_fallback_archive_paths() -> None:
    """REQ-REPORT-4442: helper fallbacks stay deterministic."""

    assert mod._int(True, 7) == 7
    assert mod._int("bad", 3) == 3
    assert mod._float(True, 1.5) == 1.5
    assert mod._float("bad", 2.5) == 2.5
    assert mod._has_duration_too_short({"flagged_artifacts_excluded": []}) is False
    assert mod._insert_before_tasks(["- id: x"], "  finding: y") == [
        "- id: x",
        "  finding: y",
    ]

    close = {
        "residual_deltas": _residuals(),
        "generic_solver_gap_state": "partial",
        "generic_loo_solve_count": 2,
        "reproducible_total_levels": 37,
    }
    text = (
        "milestones:\n"
        "- id: 2026.06.410\n"
        "  activation_recorded: old\n"
        "  tasks:\n"
        "  - id: exp4441-capstone-410\n"
    )
    new_text, removed, action = mod.dedupe_or_update_record(text, close)

    assert removed == 0
    assert action == "updated"
    assert "activation_recorded: exp4442-archive-410-activate-411" in new_text
    assert "generic_solver_gap_state=partial; generic_loo_solve_count=2/7" in new_text


def test_invalid_history_edit_blocks_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-4442-BLOCKED-PRECONDITION: invalid edits fail closed."""

    root = _make_root(tmp_path)
    history_before = (root / "research-complete.yaml").read_text(encoding="utf-8")
    monkeypatch.setattr(
        mod,
        "dedupe_or_update_record",
        lambda *_args: ("milestones: [", 0, "updated"),
    )

    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=6.0, now_s=6.2).read_text(
            encoding="utf-8"
        )
    )

    assert artifact["honest_verdict"] == "blocked_research_complete_edit_invalid"
    assert (root / "research-complete.yaml").read_text(encoding="utf-8") == history_before


def test_validation_rejects_malformed_payloads(tmp_path: Path) -> None:
    """REQ-REPORT-4442: schema helper rejects invalid terminal artifacts."""

    root = _make_root(tmp_path)
    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=5.0, now_s=5.3).read_text(
            encoding="utf-8"
        )
    )

    for mutate, message in [
        (lambda p: p.pop("honest_verdict"), "missing required artifact field"),
        (lambda p: p.__setitem__("honest_verdict", "partial: nope"), "honest_verdict"),
        (lambda p: p.__setitem__("field_principles", {}), "field principle"),
        (lambda p: p.__setitem__("verifier_is_oracle", False), "verifier_is_oracle"),
        (lambda p: p.__setitem__("trm_training_ran", True), "training"),
        (lambda p: p.__setitem__("reproducibility_checksum", "bad"), "SHA-256"),
    ]:
        bad = copy.deepcopy(artifact)
        mutate(bad)
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(bad)


def test_run_uses_internal_smart_subset_and_entrypoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-4442: required runner delegates through the module entrypoint."""

    root = _make_root(tmp_path)
    monkeypatch.setattr(mod, "run_smart_subset", lambda _: GREEN)
    assert mod.main(root) == 0
    assert (root / mod.OUTPUT_REL_PATH).exists()

    import carnot.experiment_4442_archive_410_activate_411 as entrypoint

    called: list[Path] = []

    def fake_run(path: Path) -> Path:
        called.append(path)
        return path / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main(root) == 0
    assert called == [root]

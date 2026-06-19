"""Tests for Exp 4454 `.411` archive / `.412` activation.

Spec refs: REQ-REPORT-4454, SCENARIO-REPORT-4454,
SCENARIO-REPORT-4454-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot.reporting import archive_411_activate_412_4454 as mod


GREEN = mod.CommandResult(command=["pytest"], exit_code=0, stdout="12 passed", stderr="")
RED = mod.CommandResult(command=["pytest"], exit_code=1, stdout="FAILED smart subset", stderr="")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _research_complete_text(*, duplicates: int = 1) -> str:
    head = (
        "# Carnot Research - Completed Experiments\n"
        "milestones:\n"
        "- id: 2026.06.410\n"
        "  finding: prior milestone\n"
    )
    block = (
        "- id: 2026.06.411\n"
        "  title: old conductor row\n"
        "  completed: '2026-06-19'\n"
        "  finding: See conductor log for per-experiment results.\n"
        "  tasks:\n"
        "  - id: exp4453-capstone-411\n"
        "    result: OK (conductor)\n"
    )
    return head + block * duplicates


def _registry_text() -> str:
    return (
        "schema_version: 1\n"
        "updated: '2026-06-19'\n"
        "reproducible_total_levels: 39\n"
        "reproducible_total_games: 20\n"
        "games:\n"
        "- game: sc25\n"
        "  reproducibility: reproduced\n"
        "  levels_reproduced: 1\n"
        "  levels_live_recorded: 5\n"
        "- game: g50t\n"
        "  reproducibility: reproduced\n"
        "  levels_reproduced: 1\n"
        "- game: vc33\n"
        "  reproducibility: reproduced\n"
        "  levels_reproduced: 1\n"
    )


def _capstone(**overrides: object) -> dict:
    payload = {
        "honest_verdict": "complete: v411_generic_solver_partial_loo_v2_5_levels_39_games_20_publication_ready",
        "generic_solver_gap_state": "partial",
        "generic_loo_solve_count_v2": 5,
        "reproducible_total_levels": 39,
        "reproducible_total_games": 20,
        "verifier_is_oracle": False,
        "flagged_artifacts_excluded": [],
        "g50t_bank": {
            "g50t_l1_cleanly_banked": True,
            "plus_one_game_level": True,
            "reproduced_levels": 1,
            "reproducible_total_games": 19,
            "reproducible_total_levels": 38,
            "state": "g50t_l1_banked",
            "target_game": "g50t",
            "verifier_is_oracle": True,
        },
        "config_rule": {
            "dc22_banked": False,
            "dc22_state": "not_grounded",
            "ft09_resolved_generically": True,
            "missing_verifier_gaps": [
                {
                    "game": "dc22",
                    "gap_id": "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT",
                    "residual_delta": "missing_config_rule_verifier_grounding",
                    "status": "open",
                }
            ],
            "state": "ft09_closed_dc22_open",
            "verifier_is_oracle": True,
        },
        "object_motion": {
            "closed_ar25_ka59": True,
            "residuals_closed_generically": ["ar25", "ka59"],
            "state": "ar25_ka59_closed_accuracy_lift",
            "world_model_accuracy_cold": 0.25,
            "world_model_accuracy_with_examples": 1.0,
            "verifier_is_oracle": True,
        },
        "first_contact": {
            "target_game": "vc33",
            "routed_to": "s5i5",
            "routed_generic_first_contact_banked": True,
            "reproduced_levels": 1,
            "state": "routed_generic_first_contact_banked",
            "verifier_is_oracle": True,
        },
        "library": {
            "library_coverage": 1.0,
            "library_generalizes": True,
            "primitives_documented_count": 18,
            "retrieval_precision_at_1": 1.0,
            "state": "documented_library_generalizes",
            "verifier_is_oracle": True,
        },
        "loo_v2": {
            "generic_loo_solve_count_v1_baseline": 2,
            "generic_loo_solve_count_v2": 5,
            "loo_gate_passed": True,
            "residual_deltas": [
                {
                    "game": "tr87",
                    "residual_delta": "missing_glyph_rewrite_rule_verifier_without_tr87_adapter",
                    "status": "open",
                },
                {
                    "game": "sc25",
                    "residual_delta": "missing_cast_grid_spell_shrink_tank_exit_verifier",
                    "status": "open",
                },
            ],
            "state": "v2_rises_above_baseline",
            "v2_rose_above_baseline": True,
            "verifier_is_oracle": True,
        },
        "next_backlog": {
            "open_gap_ids": [
                "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT",
                "GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER",
                "GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER",
            ],
            "missing_primitives": [
                "cast_grid_spell_shrink_tank_exit_verifier",
                "config_rule_verifier_grounding",
                "glyph_rewrite_rule_verifier_without_tr87_adapter",
            ],
        },
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
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.06.412\n", encoding="utf-8")
    (tmp_path / "research-roadmap-next.yaml").write_text(
        "milestone: 2026.06.412\ntasks: []\n", encoding="utf-8"
    )
    _write_json(tmp_path / "results/experiment_4453_capstone_v411.json", _capstone())
    return tmp_path


def test_req_report_4454_spec_anchor_declares_required_contract() -> None:
    """REQ-REPORT-4454: OpenSpec declares the archive contract."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-4454" in spec
    assert "SCENARIO-REPORT-4454" in spec
    assert "results/experiment_4454_archive_411_activate_412.json" in spec
    assert "research-roadmap-next.yaml" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["reproducible_total_levels"] in spec
    assert mod.FIELD_PRINCIPLES["open_gap_ids"] in spec
    assert mod.FIELD_PRINCIPLES["inference_substrate"] in spec


def test_run_archives_v411_and_records_true_close_state(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4454: complete path records .411 truth."""

    root = _make_root(tmp_path, duplicates=2)
    output = mod.run(root, pretest_result=GREEN, started_s=10.0, now_s=10.5)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith(mod.TERMINAL_PREFIXES)
    assert artifact["archived_milestone"] == "2026.06.411"
    assert artifact["activated_milestone"] == "2026.06.412"
    assert artifact["active_milestone_confirmed"] == "2026.06.412"
    assert artifact["research_complete_yaml_parses"] is True
    assert artifact["exclusion_manifest_parses"] is True
    assert artifact["research_roadmap_next_yaml_parses"] is True
    assert artifact["pretest_suite_green"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["trm_training_ran"] is False
    assert artifact["leaderboard_submission"] is False
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["reproducible_total_levels"] == 39
    assert artifact["reproducible_total_games"] == 20
    assert artifact["open_gap_ids"] == [
        "GAP-4423-DC22",
        "GAP-4432-LOO-TR87",
        "GAP-4432-LOO-SC25",
    ]

    close = artifact["v411_close_state"]
    assert close["generic_solver_gap_state"] == "partial"
    assert close["generic_loo_solve_count"] == 5
    assert close["generic_loo_solve_count_v1_baseline"] == 2
    assert close["generic_loo_target_count"] == 7
    assert close["g50t_l1"]["cleanly_banked"] is True
    assert close["g50t_l1"]["plus_one_game_level"] is True
    assert close["config_rule"]["ft09_resolved_generically"] is True
    assert close["config_rule"]["dc22_state"] == "not_grounded"
    assert close["object_motion"]["residuals_closed_generically"] == ["ar25", "ka59"]
    assert close["object_motion"]["world_model_accuracy_cold"] == 0.25
    assert close["object_motion"]["world_model_accuracy_with_examples"] == 1.0
    assert close["first_contact"]["target_game"] == "vc33"
    assert close["first_contact"]["routed_to"] == "s5i5"
    assert close["library"]["library_coverage"] == 1.0
    assert close["loo_v2"]["generic_loo_solve_count_v2"] == 5
    assert close["loo_v2"]["v2_rose_above_baseline"] is True
    assert close["sc25_provisional_live_recorded_levels"] == 4
    assert close["execution_grounded_arc_solve_not_moat_headline"] is True

    history = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert mod.archive_record_count(history) == 1
    assert "activation_recorded: exp4454-archive-411-activate-412" in history
    assert "generic_solver_gap_state=partial; generic_loo_solve_count=5/7" in history
    assert "GAP-4432-LOO-SC25" in history


def test_flagged_adversarial_rows_are_skipped_but_reported(tmp_path: Path) -> None:
    """REQ-REPORT-4454: flagged upstream rows are not treated as trusted evidence."""

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
    _write_json(root / "results/experiment_4453_capstone_v411.json", capstone)

    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=1.0, now_s=1.2).read_text(
            encoding="utf-8"
        )
    )

    assert artifact["v411_close_state"]["flagged_artifacts_skipped"] == [9999]
    assert artifact["flagged_artifacts_skipped"] == [9999]


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
                "milestone: 2026.06.411\n", encoding="utf-8"
            ),
            "blocked_v412_not_active",
        ),
        (
            lambda root: (root / "results/experiment_4453_capstone_v411.json").unlink(),
            "blocked_v411_capstone_missing",
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
    """SCENARIO-REPORT-4454-BLOCKED-PRECONDITION: failures do not archive."""

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
    assert artifact["v411_close_state"] == {}
    assert artifact["open_gap_ids"] == []
    assert (root / "research-complete.yaml").read_text(encoding="utf-8") == history_before


def test_red_smart_subset_blocks_without_history_edit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4454-BLOCKED-PRECONDITION: red pre-test gate blocks."""

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
    """REQ-REPORT-4454: missing .411 history row is appended canonically."""

    root = _make_root(tmp_path, duplicates=0)
    artifact = json.loads(
        mod.run(root, pretest_result=GREEN, started_s=4.0, now_s=4.3).read_text(
            encoding="utf-8"
        )
    )

    history = (root / "research-complete.yaml").read_text(encoding="utf-8")
    assert artifact["research_complete_record_action"] == "appended"
    assert mod.archive_record_count(history) == 1
    assert "Archive .411 and activate .412" in history


def test_helpers_cover_fallback_archive_paths() -> None:
    """REQ-REPORT-4454: helper fallbacks stay deterministic."""

    assert mod._int(True, 7) == 7
    assert mod._int("bad", 3) == 3
    assert mod._float(True, 1.5) == 1.5
    assert mod._float("bad", 2.5) == 2.5
    assert mod._bool("bad", True) is True
    assert mod.normalize_gap_ids(["unknown", "GAP-4423-DC22-extra"]) == ["GAP-4423-DC22"]
    assert mod._insert_before_tasks(["- id: x"], "  finding: y") == [
        "- id: x",
        "  finding: y",
    ]

    close = {
        "generic_solver_gap_state": "partial",
        "generic_loo_solve_count": 5,
        "generic_loo_target_count": 7,
        "generic_loo_solve_count_v1_baseline": 2,
        "reproducible_total_levels": 39,
        "reproducible_total_games": 20,
        "open_gap_ids": ["GAP-4423-DC22", "GAP-4432-LOO-TR87", "GAP-4432-LOO-SC25"],
    }
    text = (
        "milestones:\n"
        "- id: 2026.06.411\n"
        "  activation_recorded: old\n"
        "  tasks:\n"
        "  - id: exp4453-capstone-411\n"
    )
    new_text, removed, action = mod.dedupe_or_update_record(text, close)

    assert removed == 0
    assert action == "updated"
    assert "activation_recorded: exp4454-archive-411-activate-412" in new_text
    assert "generic_solver_gap_state=partial; generic_loo_solve_count=5/7" in new_text


def test_invalid_history_edit_blocks_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-4454-BLOCKED-PRECONDITION: invalid edits fail closed."""

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


def test_validation_rejects_malformed_complete_payloads(tmp_path: Path) -> None:
    """REQ-REPORT-4454: schema helper rejects invalid terminal artifacts."""

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
        (lambda p: p.__setitem__("open_gap_ids", []), "open_gap_ids"),
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
    """REQ-REPORT-4454: required runner delegates through the module entrypoint."""

    root = _make_root(tmp_path)
    monkeypatch.setattr(mod, "run_smart_subset", lambda _: GREEN)
    assert mod.main(root) == 0
    assert (root / mod.OUTPUT_REL_PATH).exists()

    import carnot.experiment_4454_archive_411_activate_412 as entrypoint

    called: list[Path] = []

    def fake_run(path: Path) -> Path:
        called.append(path)
        return path / mod.OUTPUT_REL_PATH

    monkeypatch.setattr(entrypoint, "run", fake_run)
    assert entrypoint.main(root) == 0
    assert called == [root]

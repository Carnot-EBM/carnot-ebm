"""Tests for Exp 4453 .411 milestone capstone.

Spec refs: REQ-CAPSTONE-4453, SCENARIO-CAPSTONE-4453.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v411_4453 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_support_files(root: Path) -> None:
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "publication_gate.py").write_text("# fixture\n", encoding="utf-8")


def _clean_live_flags(_: Path) -> list[dict[str, str]]:
    return []


def _summarize_zero(_: Path, __: Path) -> int:
    return 0


def _publication_gate(paper_ready: bool = True) -> JsonDict:
    return {
        "paper_ready": paper_ready,
        "gates": {
            "G1": {"pass": True, "detail": "fixture", "source": "exp2850"},
            "G2": {"pass": paper_ready, "detail": "fixture"},
            "G3": {"pass": True, "detail": "fixture", "hits": []},
            "G4": {"pass": True, "detail": "fixture", "source": "exp2850"},
        },
        "unmet_gates": [] if paper_ready else ["G2"],
    }


def _fixture_payloads(*, g50t_flagged: bool = False) -> dict[str, JsonDict]:
    return {
        "4443_g50t_bank": {
            "experiment": "experiment_4443_bank_g50t_example_conditioned_win",
            "honest_verdict": "success: example_conditioned_g50t_L1_banked_with_correct_substrate",
            "target_game": "g50t",
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "reproducible_total_games": 19,
            "reproducible_total_levels": 38,
            "flagged_adversarial": g50t_flagged,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "a" * 64,
        },
        "4444_config_rule": {
            "experiment": "experiment_4444_generic_config_rule_verifier_operator",
            "honest_verdict": "complete: ft09_generic_resolved_dc22_not_grounded_gap_logged",
            "ft09_resolved_generically": True,
            "dc22_state": "not_grounded",
            "dc22_reproduction_result": {"game": "dc22", "reproduced": False, "reached_level": 0},
            "missing_verifier_gaps": [
                {
                    "game": "dc22",
                    "gap_id": "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT",
                    "residual_delta": "missing_config_rule_verifier_grounding",
                    "status": "open",
                }
            ],
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "no_regression": True,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "b" * 64,
        },
        "4445_object_motion": {
            "experiment": "experiment_4445_generic_object_motion_world_model_operator",
            "honest_verdict": "success: ar25_ka59_object_motion_generic_L1_offline_reproduced",
            "residuals_closed_generically": ["ar25", "ka59"],
            "world_model_accuracy_cold": 0.25,
            "world_model_accuracy_with_examples": 1.0,
            "accuracy_margin": 0.75,
            "missing_verifier_gaps": [],
            "offline_reproduced": True,
            "reproduced_levels": 2,
            "no_regression": True,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "c" * 64,
        },
        "4446_first_contact": {
            "experiment": "experiment_4446_drive_generic_first_contact_bank",
            "honest_verdict": "success: generic_first_contact_vc33_L1_offline_reproduced",
            "target_game": "vc33",
            "routed_to": "s5i5",
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "missing_verifier_gaps": [],
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "d" * 64,
        },
        "4447_library": {
            "experiment": "experiment_4447_lilo_documented_primitive_library",
            "honest_verdict": "success: documented_primitive_library_retrieval_gate_passed",
            "library_coverage": 1.0,
            "retrieval_precision_at_1": 1.0,
            "primitives_documented": [{"game": "g50t"}, {"game": "ft09"}],
            "no_regression": True,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "e" * 64,
        },
        "4448_loo_v2": {
            "experiment": "experiment_4448_loo_generic_solve_benchmark_v2",
            "honest_verdict": "success: generic_loo_solve_count_v2_5_of_7_beats_v1_2",
            "generic_loo_solve_count_v1_baseline": 2,
            "generic_loo_solve_count_v2": 5,
            "loo_gate_passed": True,
            "offline_reproduced": True,
            "missing_verifier_gaps": [
                {
                    "game": "tr87",
                    "residual_delta": "missing_glyph_rewrite_rule_verifier_without_tr87_adapter",
                    "retrieved_operator": "config_rule_grounding",
                },
                {
                    "game": "sc25",
                    "residual_delta": "missing_cast_grid_spell_shrink_tank_exit_verifier",
                    "retrieved_operator": "active_data_collection",
                },
            ],
            "closed_residuals_by_new_operator": [
                {"game": "ka59", "closed_by_operator": "object_motion_world_model"},
                {"game": "ar25", "closed_by_operator": "object_motion_world_model"},
                {"game": "ft09", "closed_by_operator": "config_rule_verifier"},
            ],
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "f" * 64,
        },
        "4449_hygiene": {
            "experiment": "experiment_4449_registry_gaps_hygiene",
            "honest_verdict": "complete: registry_gaps_hygiene_4449_guard_passed",
            "reproducible_total_games": 20,
            "reproducible_total_levels": 39,
            "regression_guard_passed": True,
            "availability_report": {"available_artifact_keys": []},
            "registry_reconciliation": {
                "open_gap_ids": [
                    "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT",
                    "GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER",
                    "GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER",
                ],
                "filled_gap_ids": ["GAP-4423-G50T-UNSELECTABLE-FIRST-CONTACT"],
            },
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "1" * 64,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)


def test_req_capstone_4453_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4453: OpenSpec declares the .411 capstone contract first."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4453" in spec
    assert "SCENARIO-CAPSTONE-4453" in spec
    assert "results/experiment_4453_capstone_v411.json" in spec
    assert "publication_gate.py --json" in spec
    assert "aggregate-available-report-gaps" in spec
    assert "CIRCULAR_MOAT_OVERCLAIM" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_capstone_4453_current_artifacts_answer_headline_questions() -> None:
    """SCENARIO-CAPSTONE-4453: current .411 artifacts produce the honest scorecard."""

    artifact = mod.build_artifact(
        Path.cwd(),
        started_s=1.0,
        now_s=1.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["generic_solver_gap_state"] == "partial"
    assert artifact["generic_loo_solve_count_v2"] == 5
    assert artifact["reproducible_total_levels"] == 39
    assert artifact["reproducible_total_games"] == 20
    assert artifact["headline_question_answers"]["exp4443"]["g50t_l1_cleanly_banked"] is True
    assert artifact["headline_question_answers"]["exp4443"]["plus_one_game_level"] is True
    assert artifact["headline_question_answers"]["exp4444"]["ft09_resolved_generically"] is True
    assert artifact["headline_question_answers"]["exp4444"]["dc22_banked"] is False
    assert artifact["headline_question_answers"]["exp4445"]["closed_ar25_ka59"] is True
    assert artifact["headline_question_answers"]["exp4445"]["accuracy_lift"] == pytest.approx(0.75)
    assert artifact["headline_question_answers"]["exp4446"]["routed_generic_first_contact_banked"] is True
    assert artifact["headline_question_answers"]["exp4447"]["library_coverage"] == pytest.approx(1.0)
    assert artifact["headline_question_answers"]["exp4448"]["v2_rose_above_baseline"] is True
    assert artifact["headline_question_answers"]["exp4448"]["generic_loo_solve_count_v2"] == 5
    assert artifact["publication_gate"]["paper_ready"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verifier_is_oracle_honored"] is True
    assert artifact["next_backlog"]["missing_primitives"] == [
        "cast_grid_spell_shrink_tank_exit_verifier",
        "config_rule_verifier_grounding",
        "glyph_rewrite_rule_verifier_without_tr87_adapter",
    ]
    assert len(artifact["next_backlog"]["residual_deltas"]) == 3
    assert "gated_on" not in artifact
    assert {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]} == {
        4443,
        4444,
        4445,
        4446,
        4447,
        4448,
        4449,
    }


def test_req_capstone_4453_missing_and_flagged_inputs_are_per_axis_gaps(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4453: missing/flagged upstreams do not erase unrelated axes."""

    _write_support_files(tmp_path)
    payloads = _fixture_payloads(g50t_flagged=True)
    payloads.pop("4447_library")
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=3.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(False),
    )

    mod.validate_artifact(artifact)
    assert artifact["generic_loo_solve_count_v2"] == 5
    assert artifact["headline_question_answers"]["exp4443"]["g50t_l1_cleanly_banked"] is False
    assert artifact["headline_question_answers"]["exp4447"]["state"] == "missing_or_excluded"
    assert {"axis": "library", "artifact_key": "4447_library", "experiment_id": 4447} in (
        artifact["per_axis_gaps"]
    )
    assert artifact["availability_report"]["axes"]["g50t_bank"]["flagged_artifacts"] == [
        {
            "axis": "g50t_bank",
            "artifact_key": "4443_g50t_bank",
            "experiment_id": 4443,
            "reason": "flagged_adversarial",
        }
    ]
    flagged_cite = next(
        row for row in artifact["cited_upstream_artifacts"] if row["experiment_id"] == 4443
    )
    assert flagged_cite["fields_imported"] == []
    assert artifact["publication_gate"]["paper_ready"] is False
    assert artifact["honest_verdict"].endswith("publication_not_ready")


def test_req_capstone_4453_branch_helpers_keep_alternate_states_honest(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4453: alternate helper branches stay explicit and bounded."""

    assert mod.g50t_bank_read(None, False)["state"] == "missing_or_excluded"
    assert mod.g50t_bank_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert (
        mod.g50t_bank_read(
            {"target_game": "g50t", "offline_reproduced": False, "reproduced_levels": 0},
            False,
        )["state"]
        == "not_banked"
    )
    assert mod.config_rule_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.config_rule_read(None, False)["state"] == "missing_or_excluded"
    assert (
        mod.config_rule_read(
            {
                "ft09_resolved_generically": True,
                "offline_reproduced": True,
                "reproduced_levels": 1,
                "dc22_reproduction_result": {"reproduced": True, "reached_level": 1},
            },
            False,
        )["state"]
        == "ft09_and_dc22_closed"
    )
    assert mod.config_rule_read({}, False)["state"] == "config_rule_open"
    assert mod.object_motion_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.object_motion_read(None, False)["state"] == "missing_or_excluded"
    assert mod.first_contact_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.first_contact_read(None, False)["state"] == "missing_or_excluded"
    assert mod.library_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.library_read(None, False)["state"] == "missing_or_excluded"
    assert mod.loo_v2_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.loo_v2_read(None, False)["state"] == "missing_or_excluded"
    assert mod.hygiene_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.hygiene_read(None, False)["state"] == "missing_or_excluded"
    assert mod._missing_primitive_name("missing_example") == "example"  # noqa: SLF001
    assert mod._residual_rows({"missing_verifier_gaps": [object(), {"residual_delta": "none"}]}) == []  # noqa: SLF001
    gate, check, gaps = mod._publication_gate_or_gap(tmp_path, lambda _: _publication_gate(True))  # noqa: SLF001
    assert gate["paper_ready"] is False
    assert check["exists"] is False
    assert gaps == [
        {"axis": "publication_gate", "artifact_key": "publication_gate", "reason": "unrunnable"}
    ]
    backlog = mod.build_next_backlog(
        config_rule={
            "residual_deltas": [
                object(),
                {"game": "x", "residual_delta": ""},
                {"game": "x", "residual_delta": "missing_a"},
                {"game": "x", "residual_delta": "missing_a"},
            ]
        },
        loo_v2={"residual_deltas": [{"game": "y", "residual_delta": "missing_b"}]},
        hygiene={},
    )
    assert backlog["missing_primitives"] == ["a", "b"]
    assert backlog["open_gap_ids"] == []
    assert (
        mod.decide_generic_solver_gap_state(
            {"generic_loo_solve_count_v2": 0, "v2_rose_above_baseline": False},
            {"residual_deltas": []},
            {"g50t_l1_cleanly_banked": False},
            {"routed_generic_first_contact_banked": False},
            {"closed_ar25_ka59": False},
        )
        == "total-gap"
    )
    assert (
        mod.decide_generic_solver_gap_state(
            {"generic_loo_solve_count_v2": 7, "v2_rose_above_baseline": True},
            {"residual_deltas": []},
            {"g50t_l1_cleanly_banked": True},
            {"routed_generic_first_contact_banked": True},
            {"closed_ar25_ka59": True},
        )
        == "closing"
    )


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"honest_verdict": "draft"}, "honest_verdict"),
        ({"generic_solver_gap_state": "almost"}, "generic_solver_gap_state"),
        ({"generic_loo_solve_count_v2": True}, "generic_loo_solve_count_v2"),
        ({"reproducible_total_levels": "39"}, "reproducible_total_levels"),
        ({"reproducible_total_games": "20"}, "reproducible_total_games"),
        ({"next_backlog": []}, "next_backlog"),
        ({"publication_gate": []}, "publication_gate"),
        ({"verifier_is_oracle": True}, "verifier_is_oracle"),
        ({"inference_substrate": "training"}, "inference_substrate"),
        ({"reproducibility_checksum": "bad"}, "reproducibility_checksum"),
        ({"field_principles": {"honest_verdict": "loose"}}, "field_principles"),
    ],
)
def test_validate_artifact_rejects_schema_violations(
    tmp_path: Path,
    patch: JsonDict,
    message: str,
) -> None:
    """REQ-CAPSTONE-4453: schema violations fail closed."""

    _write_support_files(tmp_path)
    _write_default_artifacts(tmp_path, _fixture_payloads())
    artifact = mod.build_artifact(
        tmp_path,
        started_s=4.0,
        now_s=5.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )
    artifact.update(patch)

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(artifact)


def test_scenario_capstone_4453_write_artifact_records_clean_live_recheck(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4453: written capstone carries live adversarial re-check."""

    _write_support_files(tmp_path)
    _write_default_artifacts(tmp_path, _fixture_payloads())

    output = mod.write_artifact(
        tmp_path,
        started_s=6.0,
        now_s=7.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
        capstone_live_flag_runner=_clean_live_flags,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["capstone_live_adversarial_recheck"] == {
        "status": "clean",
        "flags": [],
        "circular_moat_overclaim": False,
    }
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert artifact["verifier_is_oracle"] is False

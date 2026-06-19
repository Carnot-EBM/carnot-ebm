"""Tests for Exp 4441 .410 milestone capstone.

Spec refs: REQ-CAPSTONE-4441, SCENARIO-CAPSTONE-4441.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v410_4441 as mod


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


def _fixture_payloads(
    *,
    win_flagged: bool = True,
    first_contact_solved: bool = False,
    residuals: list[str] | None = None,
    loo_count: int = 2,
    action_helped: bool = True,
    primitives_count: int = 5,
    total_levels: int = 37,
) -> dict[str, JsonDict]:
    residual_rows = [
        {
            "game": f"g{i}",
            "routed_to": "donor",
            "residual_delta": residual,
            "solved_without_own_recipe": False,
        }
        for i, residual in enumerate(
            residuals
            if residuals is not None
            else [
                "missing_glyph_rewrite_rule_verifier_without_tr87_adapter",
                "missing_cast_grid_spell_shrink_tank_exit_verifier",
            ],
            start=1,
        )
    ]
    solved_rows = [
        {
            "game": f"solved{i}",
            "routed_to": "donor",
            "residual_delta": "none",
            "solved_without_own_recipe": True,
        }
        for i in range(max(0, loo_count))
    ]
    primitive_rows = [
        {"operator": f"primitive_{i}", "derived_from_games": ["solved1"]}
        for i in range(primitives_count)
    ]
    with_examples = 1.0 if action_helped else 0.5
    cold = 0.714286
    return {
        "4432_loo_generic": {
            "experiment": "experiment_4432_loo_generic_solve_benchmark",
            "honest_verdict": "complete: generic_loo_solve_count_fixture",
            "generic_loo_solve_count": loo_count,
            "per_game": solved_rows + residual_rows,
            "missing_verifier_gaps": [
                {
                    "game": row["game"],
                    "routed_to": row["routed_to"],
                    "residual_delta": row["residual_delta"],
                    "attempt_mode": "fixture",
                }
                for row in residual_rows
            ],
            "offline_reproduced": loo_count > 0,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "a" * 64,
        },
        "4433_win_induction": {
            "experiment": "experiment_4433_example_conditioned_win_induction",
            "honest_verdict": "success: example_conditioned_fixture",
            "target_game": "g50t",
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "few_shot_examples_used": [{"game": "ka59"}, {"game": "s5i5"}, {"game": "tr87"}],
            "flagged_adversarial": win_flagged,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "b" * 64,
        },
        "4434_action_model": {
            "experiment": "experiment_4434_example_conditioned_action_model",
            "honest_verdict": "success: example_conditioning_improved_world_model_accuracy",
            "target_game": "cn04",
            "world_model_accuracy_cold": cold,
            "world_model_accuracy_with_examples": with_examples,
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "missing_verifier_gaps": [],
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "c" * 64,
        },
        "4435_first_contact": {
            "experiment": "experiment_4435_generic_first_contact_fixed",
            "honest_verdict": (
                "success: generic_first_contact_routed_solve_banked"
                if first_contact_solved
                else "complete: generic_first_contact_dc22_routed_no_new_level_gap_logged"
            ),
            "target_game": "dc22",
            "verdict_contract_fixed": True,
            "offline_reproduced": first_contact_solved,
            "reproduced_levels": 1 if first_contact_solved else 0,
            "missing_verifier_gaps": []
            if first_contact_solved
            else [
                {
                    "gap_id": "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT",
                    "game": "dc22",
                    "failure_mode": "needs_per_game_RE",
                    "missing_discriminator": "selectable verifier",
                    "candidate_design": "adapt config-rule predicate grounding",
                }
            ],
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "d" * 64,
        },
        "4436_primitives": {
            "experiment": "experiment_4436_deepen_plus_primitive_consolidation",
            "honest_verdict": "success: tu93_L5_deepened_primitives_consolidated",
            "deepened_game": "tu93",
            "offline_reproduced": True,
            "reproduced_levels": 5,
            "new_levels_reproduced": 1,
            "no_regression": True,
            "primitives_consolidated": primitive_rows,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "e" * 64,
        },
        "4438_hygiene": {
            "experiment": "experiment_4438_registry_gaps_hygiene",
            "honest_verdict": "complete: registry_gaps_hygiene_fixture",
            "reproducible_total_levels": total_levels,
            "reproducible_total_games": 18,
            "regression_guard_passed": True,
            "availability_report": {"available_artifact_keys": []},
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "f" * 64,
        },
        "4440_sota_ingestion": {
            "experiment": "experiment_4440_sota_ingestion_410",
            "honest_verdict": "complete: sota_ingestion_410_mapped",
            "flagged_for_v411": (
                "LILO-style documented library induction over the ARC solver/example corpus "
                "(arXiv:2310.19791)"
            ),
            "v410_outcome_conditioning": {"loo_generic_solve_count_2_of_7": loo_count >= 2},
            "inference_substrate": "cpu_reliable_channel_sota_ingestion_no_training",
            "reproducibility_checksum": "sha256:" + "1" * 64,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)


def test_req_capstone_4441_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4441: OpenSpec declares the .410 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4441" in spec
    assert "SCENARIO-CAPSTONE-4441" in spec
    assert "results/experiment_4441_capstone_v410.json" in spec
    assert "publication_gate.py --json" in spec
    assert "aggregate-available-report-gaps" in spec
    assert "CIRCULAR_MOAT_OVERCLAIM" in spec
    assert "NO gated_on" not in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_capstone_4441_current_artifacts_answer_headline_questions() -> None:
    """SCENARIO-CAPSTONE-4441: current .410 artifacts produce the honest scorecard."""

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
    assert artifact["generic_loo_solve_count"] == 2
    assert len(artifact["residual_deltas"]) == 5
    assert artifact["headline_question_answers"]["exp4432"]["generic_loo_solve_count"] == 2
    assert artifact["headline_question_answers"]["exp4432"]["residual_delta_count"] == 5
    assert artifact["win_induction_state"] == "excluded_flagged_adversarial"
    assert artifact["headline_question_answers"]["exp4433"]["held_out_level_banked"] is False
    assert artifact["headline_question_answers"]["exp4433"]["few_shot_examples_demonstrably_helped"] is False
    assert artifact["action_model_state"] == "examples_helped_no_reproduced_level"
    assert artifact["headline_question_answers"]["exp4434"]["helped_vs_cold_control"] is True
    assert artifact["headline_question_answers"]["exp4434"]["accuracy_delta"] == pytest.approx(
        0.285714,
    )
    assert artifact["first_contact_state"] == "contract_fixed_no_routed_solve"
    assert artifact["headline_question_answers"]["exp4435"]["verdict_contract_fixed"] is True
    assert artifact["headline_question_answers"]["exp4435"]["routed_solve_banked"] is False
    assert artifact["primitives_consolidated_count"] == 5
    assert artifact["headline_question_answers"]["exp4436"]["no_regression"] is True
    assert artifact["reproducible_total_levels"] == 37
    assert artifact["next_backlog"]["residual_deltas"] == artifact["residual_deltas"]
    assert any(
        row.get("gap_id") == "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT"
        for row in artifact["next_backlog"]["missing_gaps"]
    )
    assert any(
        row.get("gap_id") == "EXP4433-FLAGGED-ADVERSARIAL-RERUN"
        for row in artifact["next_backlog"]["missing_gaps"]
    )
    assert artifact["publication_gate"]["paper_ready"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verifier_is_oracle_honored"] is True
    assert "gated_on" not in artifact
    assert {row["experiment_id"] for row in artifact["flagged_artifacts_excluded"]} == {4433}
    assert 4433 not in {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]}
    assert artifact["capstone_live_adversarial_recheck"] == {"status": "not_run_until_write"}


def test_scenario_capstone_4441_clean_positive_fixture_can_be_closing(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4441: clean examples plus no residuals become closing."""

    _write_support_files(tmp_path)
    _write_default_artifacts(
        tmp_path,
        _fixture_payloads(
            win_flagged=False,
            first_contact_solved=True,
            residuals=[],
            loo_count=3,
            action_helped=True,
            primitives_count=6,
            total_levels=39,
        ),
    )

    artifact = mod.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=3.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["generic_solver_gap_state"] == "closing"
    assert artifact["win_induction_state"] == "held_out_level_banked_examples_helped"
    assert artifact["headline_question_answers"]["exp4433"]["held_out_level_banked"] is True
    assert artifact["headline_question_answers"]["exp4433"]["few_shot_examples_demonstrably_helped"] is True
    assert artifact["first_contact_state"] == "contract_fixed_routed_solve_banked"
    assert artifact["primitives_consolidated_count"] == 6
    assert artifact["next_backlog"]["residual_deltas"] == []
    assert artifact["honest_verdict"].endswith("publication_ready")


def test_req_capstone_4441_missing_and_flagged_inputs_are_per_axis_gaps(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4441: missing/flagged upstreams do not erase other axes."""

    _write_support_files(tmp_path)
    payloads = _fixture_payloads(win_flagged=True, total_levels=37)
    payloads.pop("4434_action_model")
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=4.0,
        now_s=5.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(False),
    )

    mod.validate_artifact(artifact)
    assert artifact["generic_loo_solve_count"] == 2
    assert artifact["action_model_state"] == "missing_or_excluded"
    assert {"axis": "action_model", "artifact_key": "4434_action_model", "experiment_id": 4434} in (
        artifact["per_axis_gaps"]
    )
    assert artifact["availability_report"]["axes"]["win_induction"]["flagged_artifacts"] == [
        {
            "axis": "win_induction",
            "artifact_key": "4433_win_induction",
            "experiment_id": 4433,
            "reason": "flagged_adversarial",
        }
    ]
    assert artifact["publication_gate"]["paper_ready"] is False
    assert artifact["honest_verdict"].endswith("publication_not_ready")


def test_req_capstone_4441_branch_helpers_keep_alternate_states_honest(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4441: alternate branches stay explicit and bounded."""

    assert mod._residual_rows({"note": "no rows"}) == []  # noqa: SLF001
    assert mod._residual_rows(  # noqa: SLF001
        {
            "per_game": [
                object(),
                {"game": "x", "residual_delta": ""},
                {"game": "x", "residual_delta": "none"},
                {"game": "x", "residual_delta": "missing_a", "routed_to": "y"},
            ],
            "missing_verifier_gaps": [
                {"game": "x", "residual_delta": "missing_a", "routed_to": "z"},
                {"game": "z", "residual_delta": "missing_b"},
            ],
        }
    ) == [
        {
            "game": "x",
            "routed_to": "y",
            "residual_delta": "missing_a",
            "source_artifact": "results/experiment_4432_loo_generic_solve_benchmark.json",
        },
        {
            "game": "z",
            "routed_to": "",
            "residual_delta": "missing_b",
            "source_artifact": "results/experiment_4432_loo_generic_solve_benchmark.json",
        },
    ]
    assert mod.loo_generic_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.loo_generic_read(None, False)["state"] == "missing_or_excluded"
    assert mod.win_induction_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.win_induction_read(None, False)["state"] == "missing_or_excluded"
    assert (
        mod.win_induction_read(
            {
                "offline_reproduced": True,
                "reproduced_levels": 1,
                "few_shot_examples_used": [{"game": "ka59"}],
            },
            False,
        )["state"]
        == "held_out_level_banked_examples_not_demonstrated"
    )
    assert (
        mod.win_induction_read(
            {"offline_reproduced": False, "reproduced_levels": 0, "few_shot_examples_used": []},
            False,
        )["state"]
        == "no_held_out_bank"
    )
    assert mod.action_model_read(None, False)["state"] == "missing_or_excluded"
    assert (
        mod.action_model_read(
            {
                "world_model_accuracy_cold": 0.7,
                "world_model_accuracy_with_examples": 0.6,
                "offline_reproduced": False,
                "reproduced_levels": 0,
            },
            False,
        )["state"]
        == "examples_no_help"
    )
    assert (
        mod.action_model_read(
            {
                "world_model_accuracy_cold": 0.7,
                "world_model_accuracy_with_examples": 0.8,
                "offline_reproduced": True,
                "reproduced_levels": 1,
            },
            False,
        )["state"]
        == "examples_helped_and_reproduced_level"
    )
    assert mod.first_contact_read(None, False)["state"] == "missing_or_excluded"
    assert (
        mod.first_contact_read(
            {"verdict_contract_fixed": False, "offline_reproduced": False, "reproduced_levels": 0},
            False,
        )["state"]
        == "contract_not_fixed"
    )
    assert mod.primitives_read(None, False)["state"] == "missing_or_excluded"
    assert (
        mod.primitives_read({"no_regression": False, "primitives_consolidated": [{"operator": "x"}]}, False)[
            "state"
        ]
        == "regression_or_empty"
    )
    assert mod.hygiene_read(None, False)["state"] == "missing_or_excluded"
    assert mod.sota_ingestion_read(None, False)["state"] == "missing_or_excluded"
    gate, check, gaps = mod._publication_gate_or_gap(tmp_path, lambda _: _publication_gate(True))  # noqa: SLF001
    assert gate["paper_ready"] is False
    assert check["exists"] is False
    assert gaps == [
        {"axis": "publication_gate", "artifact_key": "publication_gate", "reason": "unrunnable"}
    ]
    assert mod._first_contact_missing_gaps(  # noqa: SLF001
        {"missing_verifier_gaps": [object(), {"failure_mode": "x"}], "target_game": "fallback"}
    ) == [
        {
            "gap_id": "EXP4435-FIRST-CONTACT-GAP",
            "game": "fallback",
            "failure_mode": "x",
            "missing_discriminator": "",
            "candidate_design": "",
            "source_artifact": "results/experiment_4435_generic_first_contact_fixed.json",
        }
    ]
    backlog = mod.build_next_backlog(
        loo={"residual_deltas": []},
        win_induction={"state": "clean"},
        action_model={"missing_verifier_gaps": [object(), {"gap_id": "A"}]},
        first_contact={"missing_verifier_gaps": []},
    )
    assert backlog["missing_gaps"] == [
        {
            "gap_id": "A",
            "source_artifact": "results/experiment_4434_example_conditioned_action_model.json",
        }
    ]
    assert (
        mod._cited_upstream_artifacts(  # noqa: SLF001
            [
                {"skipped": True, "fields_imported": ["x"]},
                {"skipped": False, "fields_imported": []},
            ]
        )
        == []
    )
    assert (
        mod.decide_generic_solver_gap_state(
            {"generic_loo_solve_count": 0, "residual_deltas": []},
            {"helped_vs_cold_control": False},
            {"routed_solve_banked": False},
            {"no_regression": False, "count": 0},
        )
        == "total-gap"
    )


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"honest_verdict": "draft"}, "honest_verdict"),
        ({"generic_solver_gap_state": "almost"}, "generic_solver_gap_state"),
        ({"reproducible_total_levels": "37"}, "reproducible_total_levels"),
        ({"next_backlog": []}, "next_backlog"),
        ({"generic_loo_solve_count": True}, "generic_loo_solve_count"),
        ({"residual_deltas": {}}, "residual_deltas"),
        ({"primitives_consolidated_count": "5"}, "primitives_consolidated_count"),
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
    """REQ-CAPSTONE-4441: schema violations fail closed."""

    _write_support_files(tmp_path)
    _write_default_artifacts(tmp_path, _fixture_payloads())
    artifact = mod.build_artifact(
        tmp_path,
        started_s=8.0,
        now_s=9.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )
    artifact.update(patch)

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(artifact)


def test_scenario_capstone_4441_write_artifact_records_clean_live_recheck(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4441: written capstone carries live adversarial re-check."""

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

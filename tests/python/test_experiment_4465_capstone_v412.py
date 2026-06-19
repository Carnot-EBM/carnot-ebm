"""Tests for Exp 4465 .412 milestone capstone.

Spec refs: REQ-CAPSTONE-4465, SCENARIO-CAPSTONE-4465.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v412_4465 as mod


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


def _fixture_payloads(*, glyph_flagged: bool = False) -> dict[str, JsonDict]:
    return {
        "4455_dc22": {
            "experiment": "experiment_4455_solve_dc22_cegis_config_rule",
            "honest_verdict": "blocked_baseline_pytest_coverage",
            "target_game": "dc22",
            "dc22_grounded": False,
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "reproducible_total_levels": 39,
            "missing_verifier_gaps": ["precondition_baseline_pytest_failed_coverage_gate"],
            "flagged_adversarial": False,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "a" * 64,
        },
        "4456_glyph_rewrite": {
            "experiment": "experiment_4456_generic_glyph_rewrite_operator",
            "honest_verdict": "success: tr87_generic_glyph_rewrite_L1_offline_reproduced",
            "target_game": "tr87",
            "tr87_resolved_generically": True,
            "tr87_generic_level_reproduced": 1,
            "offline_reproduced": True,
            "closed_gap_ids": [
                "GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER"
            ],
            "missing_verifier_gaps": [],
            "no_regression": True,
            "flagged_adversarial": glyph_flagged,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "b" * 64,
        },
        "4458_first_contact": {
            "experiment": "experiment_4458_first_contact_new_game",
            "honest_verdict": "complete: generic_first_contact_sb26_routed_no_new_level",
            "target_game": "sb26",
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "reproducible_total_levels": 39,
            "missing_verifier_gaps": [
                {
                    "game": "sb26",
                    "gap_id": "GAP-4458-SB26-MISSING-COLOR-MATCH-SLOT-SEQUENCE-VERIFIER",
                    "residual_delta": "missing_color_match_slot_sequence_verifier",
                    "status": "open",
                }
            ],
            "flagged_adversarial": False,
            "verifier_is_oracle": True,
            "submitted_to_leaderboard": False,
            "reproducibility_checksum": "sha256:" + "c" * 64,
        },
        "4459_loo_v3": {
            "experiment": "experiment_4459_loo_generic_solve_benchmark_v3",
            "honest_verdict": "success: generic_loo_solve_count_v3_6_of_7_beats_v2_5",
            "generic_loo_solve_count_v2_baseline": 5,
            "generic_loo_solve_count_v3": 6,
            "loo_gate_passed": True,
            "offline_reproduced": True,
            "missing_verifier_gaps": [
                {
                    "attempt_mode": "v3_412_operator_remeasurement",
                    "game": "sc25",
                    "residual_delta": "missing_cast_grid_spell_shrink_tank_exit_verifier",
                    "retrieved_operator": "active_data_collection",
                }
            ],
            "per_game": [
                {
                    "game": "tr87",
                    "solved_without_own_recipe": True,
                    "closed_by_operator": "glyph_rewrite_rule_verifier",
                    "residual_delta": "none",
                },
                {
                    "game": "sc25",
                    "solved_without_own_recipe": False,
                    "closed_by_operator": "none",
                    "residual_delta": "missing_cast_grid_spell_shrink_tank_exit_verifier",
                },
            ],
            "flagged_adversarial": False,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "d" * 64,
        },
        "4460_submission": {
            "experiment": "experiment_4460_submission_package_prep",
            "honest_verdict": "success: submission_package_ready_39_levels_beats_13_quarantined_0",
            "submission_package_ready": True,
            "total_reproduced_levels_in_package": 39,
            "prior_submitted_baseline_levels": 13,
            "beats_prior_baseline": True,
            "submitted_to_leaderboard": False,
            "flagged_adversarial": False,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "e" * 64,
        },
        "4461_hygiene": {
            "experiment": "experiment_4461_registry_gaps_hygiene",
            "honest_verdict": "complete: registry_gaps_hygiene_4461_guard_passed",
            "reproducible_total_games": 20,
            "reproducible_total_levels": 39,
            "regression_guard_passed": True,
            "open_gap_ids": [
                "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT",
                "GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER",
                "GAP-4458-SB26-COLOR-MATCH-SLOT-SEQUENCE",
            ],
            "registry_reconciliation": {
                "filled_gap_ids": [
                    "GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER"
                ],
                "open_gap_ids": [
                    "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT",
                    "GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER",
                    "GAP-4458-SB26-COLOR-MATCH-SLOT-SEQUENCE",
                ],
            },
            "availability_report": {"available_artifact_keys": []},
            "flagged_adversarial": False,
            "verifier_is_oracle": False,
            "submitted_to_leaderboard": False,
            "reproducibility_checksum": "sha256:" + "f" * 64,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)


def test_req_capstone_4465_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4465: OpenSpec declares the .412 capstone contract first."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4465" in spec
    assert "SCENARIO-CAPSTONE-4465" in spec
    assert "results/experiment_4465_capstone_v412.json" in spec
    assert "publication_gate.py --json" in spec
    assert "aggregate-available-report-gaps" in spec
    assert "CIRCULAR_MOAT_OVERCLAIM" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_capstone_4465_current_artifacts_answer_headline_questions() -> None:
    """SCENARIO-CAPSTONE-4465: current .412 artifacts produce the honest scorecard."""

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
    assert artifact["generic_loo_solve_count_v3"] == 6
    assert artifact["reproducible_total_levels"] == 39
    assert artifact["reproducible_total_games"] == 20
    assert artifact["submission_package_ready"] is True
    assert artifact["submission_readiness_decision"] == "ready_beats_prior_baseline"
    assert artifact["headline_question_answers"]["exp4455"]["dc22_l1_cleanly_banked"] is False
    assert artifact["headline_question_answers"]["exp4455"]["gap_closed"] is False
    assert artifact["headline_question_answers"]["exp4456"]["tr87_resolved_generically"] is True
    assert artifact["headline_question_answers"]["exp4456"]["gap_closed"] is True
    assert artifact["headline_question_answers"]["exp4457"]["state"] == "missing_or_excluded"
    assert artifact["headline_question_answers"]["exp4457"]["sc25_gap_closed"] is False
    assert artifact["headline_question_answers"]["exp4458"]["banked_new_game"] is False
    assert artifact["headline_question_answers"]["exp4459"]["v3_rose_above_baseline"] is True
    assert artifact["headline_question_answers"]["exp4460"]["submission_package_ready"] is True
    assert artifact["publication_gate"]["paper_ready"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verifier_is_oracle_honored"] is True
    assert set(artifact["next_backlog"]["open_gap_ids"]) == {
        "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT",
        "GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER",
        "GAP-4458-SB26-COLOR-MATCH-SLOT-SEQUENCE",
    }
    assert {
        "cast_grid_spell_shrink_tank_exit_verifier",
        "color_match_slot_sequence_verifier",
        "config_rule_verifier_grounding",
    }.issubset(set(artifact["next_backlog"]["missing_primitives"]))
    assert {"axis": "cast_grid", "artifact_key": "4457_cast_grid", "experiment_id": 4457} in (
        artifact["per_axis_gaps"]
    )
    assert "gated_on" not in artifact
    assert artifact["submitted_to_leaderboard"] is False
    assert {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]} == {
        4455,
        4456,
        4458,
        4459,
        4460,
        4461,
    }


def test_req_capstone_4465_missing_and_flagged_inputs_are_per_axis_gaps(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4465: missing/flagged upstreams do not erase unrelated axes."""

    _write_support_files(tmp_path)
    payloads = _fixture_payloads(glyph_flagged=True)
    payloads.pop("4460_submission")
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
    assert artifact["generic_loo_solve_count_v3"] == 6
    assert artifact["headline_question_answers"]["exp4456"]["state"] == "excluded_flagged_adversarial"
    assert artifact["headline_question_answers"]["exp4460"]["state"] == "missing_or_excluded"
    assert {"axis": "submission_package", "artifact_key": "4460_submission", "experiment_id": 4460} in (
        artifact["per_axis_gaps"]
    )
    assert artifact["availability_report"]["axes"]["glyph_rewrite"]["flagged_artifacts"] == [
        {
            "axis": "glyph_rewrite",
            "artifact_key": "4456_glyph_rewrite",
            "experiment_id": 4456,
            "reason": "flagged_adversarial",
        }
    ]
    flagged_cite = next(
        row for row in artifact["cited_upstream_artifacts"] if row["experiment_id"] == 4456
    )
    assert flagged_cite["fields_imported"] == []
    assert artifact["publication_gate"]["paper_ready"] is False
    assert artifact["honest_verdict"].endswith("publication_not_ready")


def test_req_capstone_4465_branch_helpers_keep_alternate_states_honest() -> None:
    """REQ-CAPSTONE-4465: alternate helper branches stay explicit and bounded."""

    assert mod.dc22_read(None, False)["state"] == "missing_or_excluded"
    assert mod.dc22_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert (
        mod.dc22_read(
            {
                "target_game": "dc22",
                "dc22_grounded": True,
                "offline_reproduced": True,
                "reproduced_levels": 1,
            },
            False,
        )["state"]
        == "dc22_grounded_l1_banked"
    )
    assert mod.glyph_rewrite_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.glyph_rewrite_read(None, False)["state"] == "missing_or_excluded"
    assert mod.cast_grid_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.cast_grid_read(None, False)["state"] == "missing_or_excluded"
    assert (
        mod.cast_grid_read(
            {
                "target_game": "sc25",
                "sc25_resolved_generically": True,
                "provisional_promoted_to_reproduced": True,
                "offline_reproduced": True,
                "reproduced_levels": 1,
            },
            False,
        )["state"]
        == "sc25_closed_and_banked"
    )
    assert mod.first_contact_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.first_contact_read(None, False)["state"] == "missing_or_excluded"
    assert mod.loo_v3_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.loo_v3_read(None, False)["state"] == "missing_or_excluded"
    assert mod.submission_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.submission_read(None, False)["state"] == "missing_or_excluded"
    assert mod.hygiene_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.hygiene_read(None, False)["state"] == "missing_or_excluded"
    assert mod.hygiene_read(
        {
            "reproducible_total_levels": 1,
            "reproducible_total_games": 1,
            "registry_reconciliation": {"open_gap_ids": ["GAP-X"]},
        },
        False,
    )["open_gap_ids"] == ["GAP-X"]
    assert mod._publication_gate_or_gap(Path("/tmp/definitely_missing_capstone_root"), lambda _: {})[2] == [  # noqa: SLF001
        {"axis": "publication_gate", "artifact_key": "publication_gate", "reason": "unrunnable"}
    ]
    backlog = mod.build_next_backlog(
        dc22={"residual_deltas": [object(), {"game": "x", "residual_delta": ""}]},
        cast_grid={"residual_deltas": [{"game": "x", "residual_delta": "missing_y"}]},
        first_contact={"residual_deltas": [{"game": "x", "residual_delta": "missing_y"}]},
        loo_v3={"closed_residuals_by_412_operator": []},
        hygiene={"open_gap_ids": []},
    )
    assert backlog["missing_primitives"] == ["y"]
    assert mod.decide_generic_solver_gap_state(
        dc22={"dc22_l1_cleanly_banked": True},
        glyph_rewrite={"gap_closed": True},
        cast_grid={"sc25_gap_closed": True},
        first_contact={"banked_new_game": True},
        loo_v3={"v3_rose_above_baseline": True},
        next_backlog={"residual_deltas": []},
    ) == "closing"
    assert mod.decide_generic_solver_gap_state(
        dc22={},
        glyph_rewrite={},
        cast_grid={},
        first_contact={},
        loo_v3={},
        next_backlog={"residual_deltas": []},
    ) == "total-gap"
    assert mod._missing_primitive_name("missing_example") == "example"  # noqa: SLF001
    assert mod._residual_rows({"missing_verifier_gaps": [object(), {"residual_delta": "none"}]}) == []  # noqa: SLF001


def test_req_capstone_4465_write_path_and_validation_guards(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4465: write path validates and records capstone re-checks."""

    _write_support_files(tmp_path)
    _write_default_artifacts(tmp_path, _fixture_payloads())

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        started_s=4.0,
        now_s=5.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
        capstone_live_flag_runner=lambda _: [
            {"kind": "CIRCULAR_MOAT_OVERCLAIM", "severity": "critical"}
        ],
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["capstone_live_adversarial_recheck"] == {
        "circular_moat_overclaim": True,
        "flags": [{"kind": "CIRCULAR_MOAT_OVERCLAIM", "severity": "critical"}],
        "status": "critical_flags",
    }

    valid = mod.build_artifact(
        tmp_path,
        started_s=6.0,
        now_s=7.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )
    mutations = [
        ("honest_verdict", "blocked", "honest_verdict"),
        ("generic_solver_gap_state", "unknown", "generic_solver_gap_state"),
        ("generic_loo_solve_count_v3", True, "generic_loo_solve_count_v3"),
        ("submission_package_ready", "yes", "submission_package_ready"),
        ("next_backlog", [], "next_backlog"),
        ("publication_gate", [], "publication_gate"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
        ("inference_substrate", "other", "inference_substrate"),
        ("reproducibility_checksum", "sha256:not-a-real-sha", "reproducibility_checksum"),
        ("field_principles", {}, "field_principles"),
    ]
    for field, value, message in mutations:
        invalid = json.loads(json.dumps(valid))
        invalid[field] = value
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(invalid)

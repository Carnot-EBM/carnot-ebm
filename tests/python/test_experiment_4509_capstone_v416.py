"""Tests for Exp 4509 .416 capstone aggregation.

Spec refs: REQ-CAPSTONE-4509, SCENARIO-CAPSTONE-4509.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v416_4509 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_live_flags(_: Path) -> list[dict[str, str]]:
    return []


def _summarize_zero(_: Path, __: Path) -> int:
    return 0


def _fixture_payloads(*, value_flagged: bool = True, a3_flagged: bool = False) -> dict[str, JsonDict]:
    return {
        "4500_value_weight": {
            "honest_verdict": "complete: value_weight_remeasure_null_keep_0_1_of_7",
            "flagged_adversarial": value_flagged,
            "selected_value_weight": 0.0,
            "submitted_value_weight_after": 0.0,
            "action_budget": 400,
            "eval_budget_median_wall_s": 390.0,
            "selection": {
                "beats_control": False,
                "control_solve_rate": 1 / 7,
                "selected_solve_rate": 1 / 7,
                "selected_value_weight": 0.0,
                "should_raise_submitted_value_weight": False,
                "within_wall_budget": True,
            },
            "per_weight": [
                {
                    "value_weight": 0.0,
                    "heldout_solve_rate": 1 / 7,
                    "median_actions_to_first_levelup": 20,
                    "median_per_game_wall_seconds": 1.3,
                    "solved_games": 1,
                    "attempted_games": 7,
                },
                {
                    "value_weight": 1.0,
                    "heldout_solve_rate": 1 / 7,
                    "median_actions_to_first_levelup": 20,
                    "median_per_game_wall_seconds": 1.0,
                    "solved_games": 1,
                    "attempted_games": 7,
                },
            ],
            "preconditions_checked": {"offline_arcade_import": True},
        },
        "4501_frame_change": {
            "honest_verdict": "complete: frame_change_predictor_rerun_staged_corpus_shortfall_null_guard",
            "heldout_median_actions_before": 1.0,
            "heldout_median_actions_after": 1.0,
            "implied_efficiency_delta": 0.0,
            "solve_rate_before": 1.0,
            "solve_rate_after": 1.0,
            "solve_rate_dropped": False,
            "positive_control": {"actions_reduced": True},
            "preconditions_checked": {"training_shards_present": True},
        },
        "4502_energy_ranking": {
            "honest_verdict": "complete: energy_augmented_ranking_honest_null",
            "predictor_only_median_actions": 1.0,
            "energy_augmented_median_actions": 1.0,
            "efficiency_delta_vs_predictor_only": 0.0,
            "predictor_only_solve_rate": 1.0,
            "energy_augmented_solve_rate": 1.0,
            "solve_rate_delta_vs_predictor_only": 0.0,
            "solve_rate_dropped": False,
            "energy_term_added_value": False,
            "ranking_formula": "P(frame_change)*(-delta_E)",
            "gate_artifact_summary": {"loo_gate_passed": True, "v3_loo_auroc": 0.674},
            "flagged_adversarial": a3_flagged,
            "preconditions_checked": {"energy_gate_passed": True},
        },
        "4503_hud_l2": {
            "honest_verdict": "complete: ka59_hud_register_deepen_l2_honest_residual",
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "reproduction_gate": {
                "game": "ka59",
                "claimed_level": 2,
                "reached_level": 1,
                "reproduced": False,
            },
            "residual_blockers": ["ka59_l2_not_reproduced"],
            "preconditions_checked": {"offline_arcade_import_smoke": True},
        },
        "4504_adapter_l2": {
            "honest_verdict": "success: cd82_adapter_deepen_l2_offline_reproduced",
            "target_game": "cd82",
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "reproduction_gate": {
                "game": "cd82",
                "claimed_level": 2,
                "reached_level": 2,
                "reproduced": True,
            },
            "residual_blockers": [],
            "preconditions_checked": {"offline_arcade_import_smoke": True},
        },
        "4505_scoreboard": {
            "honest_verdict": (
                "complete: submitted_agent_scoreboard_refresh_generic_1_of_7_"
                "variant_7_of_25_value_weight_0"
            ),
            "a1_value_weight_verdict": {
                "state": "keep_zero_value_weight",
                "selected_value_weight": 0.0,
                "submitted_value_weight_after": 0.0,
                "source_flagged_adversarial": True,
            },
            "headline_metrics": {
                "submitted_default_heldout_generic_attempted": 7,
                "submitted_default_heldout_generic_solve_rate": 1 / 7,
                "submitted_default_heldout_generic_solved": 1,
                "variant_transfer_attempted": 25,
                "variant_transfer_rate": 0.28,
                "variant_transfer_solved": 7,
            },
            "scoreboard_row": {
                "heldout_generic_measurement": {
                    "source_artifact": "results/experiment_4500_value_weight_remeasure.json",
                    "source_flagged_adversarial": True,
                    "median_actions_to_first_levelup": 20,
                },
                "variant_transfer_measurement": {
                    "source_artifact": "results/experiment_4499_capstone_v415.json",
                    "rate": 0.28,
                },
            },
            "leaderboard_submission": False,
            "preconditions_checked": {"parity_test_target": "tests/python/test_arc_submitted_agent_parity.py"},
        },
        "4506_lazy_value": {
            "honest_verdict": "complete: lazy_value_eval_speedup_232.69x_quality_preserved_80_of_80",
            "speedup_factor": 232.69,
            "routing_quality_preserved": True,
            "routing_quality_match_rate": 1.0,
            "value_head_call_reduction_factor": 320.0,
            "preconditions_checked": {"cached_candidate_frontiers_built": True},
        },
        "4507_hardware": {
            "honest_verdict": "complete: hardware_continuity_audit_4507",
            "per_board_reachability": {"kv260": True, "gatemate": False, "polarfire": True},
            "preconditions_checked": [{"resource": "kv260_ssh", "available": True}],
        },
        "4508_sota": {
            "honest_verdict": "complete: arc_affordance_sota_416_mapped_for_v417",
            "strongest_for_v417": "flagged_for_v417: affordance-pruned frame-change predictor",
            "source_ids": ["2006.15085", "2008.09241", "2602.03201"],
            "preconditions_checked": {"leaderboard_submission": False},
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)


def test_req_capstone_4509_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4509: OpenSpec declares the .416 capstone contract first."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4509" in spec
    assert "SCENARIO-CAPSTONE-4509" in spec
    assert "results/experiment_4509_capstone_v416.json" in spec
    assert "scripts/summarize_artifact.py" in spec
    assert "flagged_adversarial:true" in spec
    assert "reproducible_total_levels" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_capstone_4509_current_artifacts_answer_headline_questions() -> None:
    """SCENARIO-CAPSTONE-4509: current .416 artifacts produce the honest scorecard."""

    artifact = mod.build_artifact(
        Path.cwd(),
        started_s=10.0,
        now_s=11.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["a1_value_weight_verdict"]["state"] == "excluded_flagged_adversarial"
    assert artifact["a1_value_weight_verdict"]["positive_weight_beats_zero_within_budget"] is False
    assert artifact["a1_value_weight_verdict"]["submitted_value_weight_after"] == 0.0
    assert artifact["a2_frame_change_predictor_efficiency_delta"]["efficiency_delta"] == 0.0
    assert artifact["a2_frame_change_predictor_efficiency_delta"]["efficiency_win"] is False
    assert artifact["a2_frame_change_predictor_efficiency_delta"]["solve_rate_dropped"] is False
    assert artifact["a3_energy_augmented_ranking"]["solve_rate_delta"] == 0.0
    assert artifact["a3_energy_augmented_ranking"]["efficiency_delta"] == 0.0
    assert artifact["a3_energy_augmented_ranking"]["energy_term_added_value"] is False
    assert artifact["a4_a5_l2_banked"] is True
    assert artifact["a4_a5_l2_details"]["adapter_l2"]["l2_banked"] is True
    assert artifact["a4_a5_l2_details"]["hud_register"]["l2_banked"] is False
    assert artifact["submitted_agent_heldout_solve_rate"] == pytest.approx(1 / 7)
    assert artifact["submitted_agent_scoreboard"]["heldout_source_flagged_adversarial"] is True
    assert artifact["variant_transfer_rate"] == pytest.approx(0.28)
    assert artifact["submitted_agent_scoreboard"]["variant_transfer_solved"] == 7
    assert artifact["verifier_is_oracle"] is False
    assert all(isinstance(claim["verifier_is_oracle"], bool) for claim in artifact["verifier_claims"])
    assert any(row["experiment_id"] == 4500 for row in artifact["flagged_artifacts_skipped"])
    skipped_4500 = next(row for row in artifact["cited_upstream_artifacts"] if row["experiment_id"] == 4500)
    assert skipped_4500["fields_imported"] == []
    assert artifact["submitted_to_leaderboard"] is False
    assert "gated_on" not in artifact
    assert "reproducible_total_levels" not in artifact


def test_req_capstone_4509_missing_and_flagged_inputs_are_recorded(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4509: missing/flagged upstreams do not fabricate clean metrics."""

    payloads = _fixture_payloads(a3_flagged=True)
    payloads.pop("4505_scoreboard")
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=20.0,
        now_s=21.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
    )

    mod.validate_artifact(artifact)
    assert artifact["a3_energy_augmented_ranking"]["state"] == "excluded_flagged_adversarial"
    assert artifact["a3_energy_augmented_ranking"]["energy_augmented_solve_rate"] == 0.0
    assert artifact["submitted_agent_heldout_solve_rate"] == 0.0
    assert artifact["variant_transfer_rate"] == 0.0
    assert {"axis": "submitted_agent_scoreboard", "artifact_key": "4505_scoreboard", "experiment_id": 4505} in (
        artifact["per_axis_gaps"]
    )
    flagged = next(row for row in artifact["cited_upstream_artifacts"] if row["experiment_id"] == 4502)
    assert flagged["fields_imported"] == []
    assert any(row["experiment_id"] == 4502 for row in artifact["flagged_artifacts_skipped"])


def test_req_capstone_4509_helper_branches_and_validation_guards(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4509: helper branches and schema validation fail closed."""

    assert mod.a1_read(None, True, None, False)["state"] == "excluded_flagged_adversarial"
    assert mod.a1_read(None, False, None, False)["state"] == "missing_or_excluded"
    assert mod.a1_read(
        {
            "selection": {
                "beats_control": True,
                "within_wall_budget": True,
                "selected_value_weight": 1.0,
            },
            "submitted_value_weight_after": 1.0,
        },
        False,
        None,
        False,
    )["positive_weight_beats_zero_within_budget"] is True
    assert mod.a1_read(
        {
            "selected_value_weight": 2.0,
            "selection": {"beats_control": True, "within_wall_budget": True},
        },
        False,
        None,
        False,
    )["selected_value_weight"] == 2.0
    assert mod.a2_read(None, False)["state"] == "missing_or_excluded"
    assert mod.a2_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.a2_read(
        {
            "heldout_median_actions_before": 10,
            "heldout_median_actions_after": 4,
            "implied_efficiency_delta": 0.2,
            "solve_rate_dropped": False,
        },
        False,
    )["efficiency_win"] is True
    assert mod.a3_read(None, False)["state"] == "missing_or_excluded"
    assert mod.a3_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.a3_read(
        {
            "efficiency_delta_vs_predictor_only": 0.1,
            "solve_rate_delta_vs_predictor_only": 0.0,
            "energy_term_added_value": True,
            "solve_rate_dropped": False,
        },
        False,
    )["efficiency_win"] is True
    assert mod.a4_a5_l2_read(None, None, False, False)["state"] == "missing_or_excluded"
    assert mod.a4_a5_l2_read(None, None, True, True)["state"] == "excluded_flagged_adversarial"
    assert mod.a4_a5_l2_read(
        None,
        {"reproduction_gate": {"claimed_level": 2, "reached_level": 2, "reproduced": True}},
        False,
        False,
    )["any_l2_banked"] is True
    assert mod.a4_a5_l2_read(
        {"reproduction_gate": {"claimed_level": 2, "reached_level": 2, "reproduced": True}},
        None,
        True,
        False,
    )["hud_register"]["state"] == "excluded_flagged_adversarial"
    assert mod.scoreboard_read(None, False)["state"] == "missing_or_excluded"
    assert mod.scoreboard_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.scoreboard_read(
        {
            "a1_value_weight_verdict": {"source_flagged_adversarial": True},
            "headline_metrics": {"submitted_default_heldout_generic_solve_rate": 0.25},
        },
        False,
    )["heldout_source_flagged_adversarial"] is True
    assert mod.operational_context_read(None, False)["state"] == "missing_or_excluded"
    assert mod.operational_context_read(None, True)["state"] == "excluded_flagged_adversarial"

    _write_default_artifacts(tmp_path, _fixture_payloads())
    valid = mod.build_artifact(
        tmp_path,
        started_s=30.0,
        now_s=31.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
    )
    invalid_cases = [
        ("__delete__honest_verdict", None, "missing required field"),
        ("honest_verdict", "blocked", "honest_verdict"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("preconditions_checked", [], "preconditions_checked"),
        ("a1_value_weight_verdict", [], "a1_value_weight_verdict"),
        ("a2_frame_change_predictor_efficiency_delta", [], "a2_frame_change_predictor_efficiency_delta"),
        ("a3_energy_augmented_ranking", [], "a3_energy_augmented_ranking"),
        ("a4_a5_l2_banked", "yes", "a4_a5_l2_banked"),
        ("submitted_agent_heldout_solve_rate", True, "submitted_agent_heldout_solve_rate"),
        ("variant_transfer_rate", True, "variant_transfer_rate"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
        ("verifier_claims", {}, "verifier_claims"),
        ("flagged_artifacts_skipped", {}, "flagged_artifacts_skipped"),
        ("cited_upstream_artifacts", {}, "cited_upstream_artifacts"),
        ("random_seed", 1, "random_seed"),
        ("reproducibility_checksum", "bad", "reproducibility_checksum"),
        ("field_principles", {}, "field_principles"),
        ("upstream_provenance", {}, "upstream_provenance"),
        ("upstream_provenance", [1], "upstream provenance row"),
        ("upstream_provenance", [{"sha256": "bad"}], "invalid sha256"),
        (
            "upstream_provenance",
            [{"sha256": "1" * 64, "skipped": True, "fields_imported": ["x"]}],
            "skipped upstreams",
        ),
        ("verifier_claims", [{"claim": "x"}], "verifier_claims"),
        ("__gated_on__", True, "gated_on"),
        ("__reproducible_total_levels__", 47, "reproducible_total_levels"),
        ("__checksum_mismatch__", True, "reproducibility_checksum"),
    ]
    for field, value, message in invalid_cases:
        invalid = json.loads(json.dumps(valid))
        if field == "__delete__honest_verdict":
            invalid.pop("honest_verdict")
        elif field == "__gated_on__":
            invalid["gated_on"] = value
        elif field == "__reproducible_total_levels__":
            invalid["reproducible_total_levels"] = value
        elif field == "__checksum_mismatch__":
            invalid["reproducibility_checksum"] = "sha256:" + "1" * 64
        else:
            invalid[field] = value
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(invalid)


def test_req_capstone_4509_write_path_records_capstone_recheck(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4509: writer validates and records the live capstone re-check."""

    _write_default_artifacts(tmp_path, _fixture_payloads())

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        started_s=40.0,
        now_s=41.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
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


def test_req_capstone_4509_unparseable_and_live_critical_inputs_are_excluded(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4509: unsafe upstreams import no headline fields."""

    bad_path = tmp_path / mod.DEFAULT_UPSTREAMS["4501_frame_change"].path
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("{not-json", encoding="utf-8")
    raw, provenance, exclusions = mod._read_inputs(  # noqa: SLF001
        tmp_path,
        _clean_live_flags,
        _summarize_zero,
    )
    assert raw["4501_frame_change"] is None
    assert provenance[0]["parse_error"].startswith("JSONDecodeError")
    assert provenance[0]["fields_imported"] == []
    assert exclusions[0]["reason"] == "unparsable_or_non_object"

    payloads = {"4502_energy_ranking": _fixture_payloads(value_flagged=False)["4502_energy_ranking"]}
    _write_default_artifacts(tmp_path, payloads)
    raw, provenance, exclusions = mod._read_inputs(  # noqa: SLF001
        tmp_path,
        lambda _: [{"kind": "TEST_CRITICAL", "severity": "critical"}],
        _summarize_zero,
    )
    energy_row = next(row for row in provenance if row["artifact_key"] == "4502_energy_ranking")
    assert raw["4502_energy_ranking"]["flagged_adversarial"] is True
    assert energy_row["live_critical"] is True
    assert energy_row["fields_imported"] == []
    assert any(row["reason"] == "live_critical_adversarial" for row in exclusions)

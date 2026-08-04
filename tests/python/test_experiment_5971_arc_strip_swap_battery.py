"""Tests for Exp5971 ARC strip-swap battery.

Spec refs: REQ-ARC-CPTB-5971,
SCENARIO-ARC-CPTB-5971-GATE-REPLAY-MATRIX-SEAL,
SCENARIO-ARC-CPTB-5971-LIVE-PATH-CELL-HEALTH,
SCENARIO-ARC-CPTB-5971-GAME-UNIT-FORCED-VERDICT-REFUSAL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.agentic import arc_strip_swap_battery as mod


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/agentic-harness/spec.md"


def _fixture_rows(*, transformed_anchor_live: bool = False) -> list[dict]:
    rows: list[dict] = []
    games = ["g1", "g2", "g3"]
    seeds = [101, 102]
    for game in games:
        for seed in seeds:
            for condition in mod.CONDITIONS:
                for arm in mod.ARMS:
                    levels = 0
                    if condition == "original":
                        if game == "g1" and arm in {"SHIP", "HUDO"}:
                            levels = 1
                        if game == "g2" and arm == "CTRL":
                            levels = 1
                    elif transformed_anchor_live:
                        if game == "g1" and arm == "SHIP":
                            levels = 1
                        if game == "g3" and arm == "FRONT":
                            levels = 1
                    rows.append(
                        {
                            "cell_id": f"{game}|{arm}|{seed}|{condition}",
                            "game": game,
                            "seed": seed,
                            "condition": condition,
                            "arm": arm,
                            "terminal_state": "completed",
                            "completed": True,
                            "missing": False,
                            "errored": False,
                            "generator_invalid": False,
                            "ran": True,
                            "levels": levels,
                            "progress": float(levels),
                            "actions": 7 + levels,
                            "actions_to_first_levelup": 4 if levels else None,
                            "elapsed_s": 0.01,
                            "error": None,
                            "transform_selected_condition_id": (
                                None if condition == "original" else "C5_strip_swap_rows_bottom_t2"
                            ),
                            "hud_predicate_changed": condition == "strip_swap",
                            "hud_mask_resolved_before": True,
                            "hud_mask_resolved_after": condition == "original",
                            "frontier_predicate_dose": 0.0,
                            "policy_decisions": [{"kind": "RESET"}, {"kind": "ACTION1"}],
                            "observations": [{"level": 0}],
                            "health": {"valid_action_count": 1, "step_ok_count": 1},
                        }
                    )
    return rows


def test_req_arc_cptb_5971_spec_declares_battery_contract() -> None:
    """REQ-ARC-CPTB-5971: OpenSpec freezes gate replay, matrix, fields, and safeguards."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-CPTB-5971") :]

    for marker in (
        "SCENARIO-ARC-CPTB-5971-GATE-REPLAY-MATRIX-SEAL",
        "SCENARIO-ARC-CPTB-5971-LIVE-PATH-CELL-HEALTH",
        "SCENARIO-ARC-CPTB-5971-GAME-UNIT-FORCED-VERDICT-REFUSAL",
        mod.RESULT_RELATIVE_PATH,
        "make_carnot_agent",
        "E3AgentPolicy",
        "strip_swap_sentinel_ready_score == 1.0",
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section

    for field, principle in mod.REQUIRED_FIELD_PROVENANCE.items():
        assert f"`{field}`" in section
        assert principle["principle"] in section


def test_scenario_arc_cptb_5971_gate_replay_and_matrix_seal() -> None:
    """SCENARIO-ARC-CPTB-5971-GATE-REPLAY-MATRIX-SEAL: preconditions seal all planned cells."""

    gate = mod.replay_exp5970_gate(REPO)
    arms = mod.load_preregistered_arms(REPO)
    seal = mod.build_matrix_seal(REPO, arms=arms, action_budget=3, wall_time_s=5.0)

    assert gate["ready"] is True
    assert gate["strip_swap_sentinel_ready_score"] == 1.0
    assert gate["artifact_sha256"]
    assert list(arms) == list(mod.ARMS)
    assert seal["sealed_before_outcomes"] is True
    assert seal["n_games"] == 25
    assert seal["n_arms"] == 4
    assert seal["n_seeds"] == 5
    assert seal["n_conditions"] == 2
    assert seal["n_cells_expected"] == 25 * 4 * 5 * 2
    assert seal["action_budget"] == 3


def test_scenario_arc_cptb_5971_live_path_cell_health_smoke() -> None:
    """SCENARIO-ARC-CPTB-5971-LIVE-PATH-CELL-HEALTH: a bounded cell uses the submitted path."""

    row = mod.run_live_cell(
        REPO,
        game="tn36",
        arm="SHIP",
        seed=20260726,
        condition="strip_swap",
        arm_kwargs=mod.load_preregistered_arms(REPO)["SHIP"]["kwargs"],
        action_budget=1,
        wall_time_s=10.0,
    )

    assert row["game"] == "tn36"
    assert row["arm"] == "SHIP"
    assert row["condition"] == "strip_swap"
    assert row["live_path"] == "make_carnot_agent/E3AgentPolicy.choose_action"
    assert row["terminal_state"] in {"completed", "errored"}
    assert row["health"]["source_bfs_adapter_prior_game_hidden_state_access_count"] == 0
    assert row["transform_selected_condition_id"] in {
        "C4_strip_swap_rows_top_t2",
        "C5_strip_swap_rows_bottom_t2",
        "C6_strip_swap_cols_left_t2",
        "C7_strip_swap_cols_right_t2",
    }


def test_scenario_arc_cptb_5971_game_unit_forced_verdict_refusal() -> None:
    """SCENARIO-ARC-CPTB-5971-GAME-UNIT-FORCED-VERDICT-REFUSAL: destroyed anchors force null."""

    analysis = mod.analyze_rows(_fixture_rows(transformed_anchor_live=False), expected_cells=48)

    hud = analysis["anchor_survival_and_discriminating_game_support"]["hud_given_frontier_on"]
    stats = analysis["game_unit_sign_jackknife_intervals_and_p_floors"]["hud_given_frontier_on"]

    assert hud["original_anchor_won_by_matched_arm"] is True
    assert hud["transformed_anchor_retains_valid_support"] is False
    assert hud["interpretable"] is False
    assert stats["strip_swap"]["exact_one_sided_sign_test"]["n_independent_games"] <= 2
    assert analysis["convention_dependence_decision"]["status"] == "complete_null"
    assert "anchor support" in analysis["convention_dependence_decision"]["reason"]
    assert analysis["overall_hud_value_not_identified_receipt"]["flag_flip_recommended"] is False


def test_req_arc_cptb_5971_artifact_schema_validation_and_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-CPTB-5971: artifact fields, immutability receipts, and checksum are enforced."""

    rows = _fixture_rows(transformed_anchor_live=True)
    monkeypatch.setattr(mod, "run_frozen_matrix", lambda *args, **kwargs: rows)

    artifact = mod.build_artifact(
        root=REPO,
        result_output_path=tmp_path / "experiment_5971.json",
        action_budget=3,
        wall_time_s=5.0,
        test_exit_codes={"focused_unit": 0},
    )

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["status"] in {"complete_null", "complete_underpowered", "complete_positive"}
    assert artifact["preconditions_checked"]["checked"] is True
    assert artifact["gate_replay_receipt"]["ready"] is True
    assert artifact["expected_completed_missing_errored_and_generator_invalid_cells"]["expected"] == 1000
    assert artifact["live_agent_path_and_disabled_escape_hatches"]["normal_path"].endswith(
        "E3AgentPolicy.choose_action"
    )
    assert artifact["verifier_is_oracle"] is False
    assert artifact["no_solve_credit_receipt"]["solve_credit_claimed"] is False
    mod.validate_artifact(artifact)

    with pytest.raises(ValueError, match="missing required fields"):
        bad = dict(artifact)
        del bad["status"]
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact({**artifact, "reproducibility_checksum": "sha256:bad"})
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact({**artifact, "verifier_is_oracle": True})
    with pytest.raises(ValueError, match="solve credit"):
        bad = json.loads(json.dumps(artifact))
        bad["no_solve_credit_receipt"]["solve_credit_claimed"] = True
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="flag flip"):
        bad = json.loads(json.dumps(artifact))
        bad["overall_hud_value_not_identified_receipt"]["flag_flip_recommended"] = True
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="registry"):
        bad = json.loads(json.dumps(artifact))
        bad["shipped_flag_and_registry_immutability"]["registry_unchanged"] = False
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="honest_verdict"):
        bad = json.loads(json.dumps(artifact))
        bad["honest_verdict"] = "ready: bad"
        bad["reproducibility_checksum"] = mod.artifact_checksum(bad)
        mod.validate_artifact(bad)


def test_req_arc_cptb_5971_writer_round_trips_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-CPTB-5971: writer emits the validated artifact returned by the builder."""

    payload = {field: None for field in mod.REQUIRED_ARTIFACT_FIELDS}
    payload.update(
        {
            "status": "complete_null",
            "honest_verdict": "complete_null: fixture",
            "reproducibility_checksum": "sha256:fixture",
        }
    )
    monkeypatch.setattr(mod, "build_artifact", lambda **kwargs: payload)

    out = tmp_path / "experiment_5971.json"
    written = mod.write_artifact(root=REPO, result_output_path=out, test_exit_codes={"unit": 0})

    assert written is payload
    assert json.loads(out.read_text(encoding="utf-8"))["honest_verdict"] == "complete_null: fixture"

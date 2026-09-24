"""REQ-ARC-WMTE-10013 tests for the planner-arm measurement harness."""

from __future__ import annotations

import json

import numpy as np

from carnot import experiment_10012_gate_usefulness as exp12
from carnot import experiment_10013_planner_dedup_tiebreak as exp
from carnot.agentic import arc_executable_world_model as e3


def _row(
    pair: str,
    arm: str,
    *,
    game: str = "su15",
    cohort: str = "stall_window",
    control: bool = False,
    family: str = "THINK",
    win: bool = False,
    actions: int = 0,
    calls: int = 0,
    wall: float = 0.0,
) -> dict[str, object]:
    return {
        "pair_id": pair,
        "game": game,
        "arm": arm,
        "provenance_cohort": cohort,
        "is_control": control,
        "engine_family": family,
        "real_level_up": win,
        "real_actions_used": actions,
        "planner_engine_calls": calls,
        "planner_wall_s": wall,
    }


def test_three_arms_are_pre_registered_and_live_budget_is_unchanged() -> None:
    """SCENARIO-ARC-WMTE-10013-MEASUREMENT-AND-GUARDS freezes three arms."""
    assert [arm.name for arm in exp.PLANNER_ARMS] == [
        "OFF",
        "HUD_DEDUP",
        "HUD_DEDUP+TIEBREAK",
    ]
    assert all(arm.max_nodes == exp12.LIVE_SCORED.max_nodes == 20_000 for arm in exp.PLANNER_ARMS)
    assert all(arm.max_depth == exp12.LIVE_SCORED.max_depth == 80 for arm in exp.PLANNER_ARMS)
    assert exp.PLANNER_ARMS[0].environment == {}
    assert exp.PLANNER_ARMS[1].environment == {"CARNOT_ARC_PLAN_HUD_DEDUP": "1"}
    assert exp.PLANNER_ARMS[2].environment["CARNOT_ARC_PLAN_GOAL_TIEBREAK"] == "novelty"


def test_arm_environment_restores_prior_values(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-10013-FLAGS-OFF-IDENTITY keeps arms isolated."""
    monkeypatch.setenv("CARNOT_ARC_PLAN_HUD_DEDUP", "prior")
    monkeypatch.delenv("CARNOT_ARC_PLAN_GOAL_TIEBREAK", raising=False)
    with exp.planner_arm_environment(exp.PLANNER_ARMS[2]):
        assert exp.os.environ["CARNOT_ARC_PLAN_HUD_DEDUP"] == "1"
        assert exp.os.environ["CARNOT_ARC_PLAN_GOAL_TIEBREAK"] == "novelty"
    assert exp.os.environ["CARNOT_ARC_PLAN_HUD_DEDUP"] == "prior"
    assert "CARNOT_ARC_PLAN_GOAL_TIEBREAK" not in exp.os.environ


def test_measurement_mask_uses_production_detector_and_swallow_guard(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-10013-HUD-DEDUP-SAFETY reuses production safety."""
    frame_mask = np.array([[False, False], [True, True]])
    monkeypatch.setattr(exp.agent, "_compute_hud_mask_from_frame", lambda *_a, **_k: frame_mask)
    monkeypatch.setattr(e3, "logical_hud_mask", lambda mask, _cell: mask)
    before = np.zeros((2, 2), dtype=np.int16)
    after = before.copy()
    after[0, :] = 4
    transitions = [e3.Transition(before, 1, None, after, 0, 0)]
    mask, record = exp.resolve_measurement_mask(object(), 1, transitions)
    assert np.array_equal(mask, frame_mask)
    assert record["status"] == "applied"
    assert record["swallow_check"]["reason"] == "ok"

    swallowing_after = before.copy()
    swallowing_after[1, 0] = 9
    refused, refusal = exp.resolve_measurement_mask(
        object(),
        1,
        [e3.Transition(before, 1, None, swallowing_after, 0, 0)],
    )
    assert refused is None
    assert refusal["status"] == "refused"


def test_reducer_reports_primary_named_checks_and_regressions() -> None:
    """SCENARIO-ARC-WMTE-10013-MEASUREMENT-AND-GUARDS reduces primary controls."""
    rows = []
    for arm in ("OFF", "HUD_DEDUP", "HUD_DEDUP+TIEBREAK"):
        rows.extend(
            [
                _row(
                    "dc22-expert",
                    arm,
                    game="dc22",
                    control=True,
                    family="EXPERT",
                    win=arm != "OFF",
                    calls=19_000,
                ),
                _row(
                    "wa30-expert",
                    arm,
                    game="wa30",
                    control=True,
                    family="EXPERT",
                    win=arm.endswith("TIEBREAK"),
                    calls=20_000,
                ),
                _row(
                    "sb26-expert",
                    arm,
                    game="sb26",
                    control=True,
                    family="EXPERT",
                    win=arm != "OFF",
                    calls=12_000,
                ),
            ]
        )
    rows.extend(
        [
            _row("candidate-regresses", "OFF", win=True, actions=2),
            _row("candidate-regresses", "HUD_DEDUP", win=False, actions=5),
            _row("candidate-regresses", "HUD_DEDUP+TIEBREAK", win=False, actions=6),
        ]
    )
    reduced = exp.reduce_rows(rows)
    assert reduced["primary"]["HUD_DEDUP"]["expert_win_count"] == 2
    assert reduced["primary"]["HUD_DEDUP+TIEBREAK"]["named_checks"]["wa30"]["level_up"] is True
    assert {row["arm"] for row in reduced["guard_1_regressions"]} == {
        "HUD_DEDUP",
        "HUD_DEDUP+TIEBREAK",
    }


def test_harm_and_cost_guards_keep_appendix_separate() -> None:
    """SCENARIO-ARC-WMTE-10013-MEASUREMENT-AND-GUARDS separates harm and cost."""
    rows = [
        _row("main-good", "OFF", win=True, actions=2, wall=0.2),
        _row("main-bad", "OFF", win=False, actions=7, wall=0.4),
        _row(
            "appendix-bad",
            "OFF",
            cohort="h2h_replay_counterfactual",
            win=False,
            actions=11,
            wall=0.6,
        ),
        _row("expert", "OFF", control=True, family="EXPERT", win=False, actions=13, wall=0.8),
    ]
    reduced = exp.reduce_rows(rows)
    main = reduced["guard_2_harm"]["main_candidates"]["OFF"]
    appendix = reduced["guard_2_harm"]["h2h_replay_counterfactual"]["OFF"]
    assert main == {"pair_count": 2, "useful_count": 1, "wasted_real_actions": 7}
    assert appendix == {"pair_count": 1, "useful_count": 0, "wasted_real_actions": 11}
    assert reduced["guard_3_cost"]["OFF"]["planner_call_count"] == 4
    assert reduced["guard_3_cost"]["OFF"]["mean_wall_s_per_call"] == 0.5


def test_artifact_has_required_provenance_and_reproducible_checksum(tmp_path) -> None:
    """SCENARIO-ARC-WMTE-10013-ARTIFACT retains terminal provenance."""
    rows = [_row("pair", "OFF", calls=4, wall=0.25)]
    output = tmp_path / "experiment_10013.json"
    artifact = exp.build_artifact(
        rows,
        preconditions_checked=[{"resource": "fixture", "available": True}],
        duration_s=1.25,
    )
    exp.atomic_json(output, artifact)
    loaded = json.loads(output.read_text())
    assert loaded["honest_verdict"].startswith("complete_")
    assert loaded["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert loaded["solve_provenance"] == "development_proxy"
    assert loaded["random_seed"] == 10013
    checksum = loaded.pop("reproducibility_checksum")
    assert checksum == exp.canonical_checksum(loaded)
    assert loaded["per_pair_arm_rows"] == rows


def test_blocked_artifact_keeps_terminal_fields(tmp_path) -> None:
    """SCENARIO-ARC-WMTE-10013-ARTIFACT reports failed preconditions honestly."""
    output = tmp_path / "blocked.json"
    artifact = exp.build_artifact(
        [],
        preconditions_checked=[{"resource": "fixture", "available": False}],
        duration_s=0.5,
    )
    exp.atomic_json(output, artifact)
    loaded = json.loads(output.read_text())
    assert loaded["honest_verdict"] == "blocked_precondition_failed"
    assert loaded["per_pair_arm_rows"] == []
    assert loaded["preconditions_checked"][0]["available"] is False


def test_harness_imports_experiment_10012_live_scored_contract() -> None:
    """SCENARIO-ARC-WMTE-10013-MEASUREMENT-AND-GUARDS reuses LIVE_SCORED."""
    assert exp.SOURCE_EXPERIMENT_ID == exp12.EXPERIMENT_ID
    assert exp.LIVE_EXECUTION_ARM is exp12.LIVE_SCORED
    assert exp.INFERENCE_SUBSTRATE == exp12.INFERENCE_SUBSTRATE
    assert exp.SOLVE_PROVENANCE == exp12.SOLVE_PROVENANCE

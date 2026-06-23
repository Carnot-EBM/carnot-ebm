"""Tests for Exp 4634 live action-efficiency metric helper.

Spec refs: REQ-ARC-WMTE-4634, SCENARIO-ARC-WMTE-4634.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _a2_artifact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "experiment": "synthetic_a2_action_effect_predictor",
        "live_measurement": {
            "per_level_efficiency": rows,
            "first_win_rate_predictor": 0.5,
        },
    }


def _row(
    *,
    game: str = "aa00",
    level: int = 0,
    baseline_actions: float,
    agent_actions: float,
    solved: bool = True,
) -> dict[str, Any]:
    return {
        "game": game,
        "level": level,
        "baseline_actions": baseline_actions,
        "agent_actions": agent_actions,
        "solved": solved,
    }


def test_req_arc_wmte_4634_spec_declares_live_action_efficiency_contract() -> None:
    """REQ-ARC-WMTE-4634: OpenSpec declares the helper and required fields."""

    from carnot import experiment_4634_live_action_efficiency_metric as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4634" in spec
    assert "SCENARIO-ARC-WMTE-4634" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4634_agent_equal_baseline_reports_one() -> None:
    """SCENARIO-ARC-WMTE-4634: agent_actions==baseline_actions reports efficiency 1.0."""

    from carnot import experiment_4634_live_action_efficiency_metric as mod

    metric = mod.compute_live_action_efficiency(
        _a2_artifact([_row(baseline_actions=12, agent_actions=12)])
    )

    assert metric["live_action_efficiency"] == pytest.approx(1.0)
    assert metric["solved_level_count"] == 1
    assert metric["per_level_efficiency"] == [
        {
            "game": "aa00",
            "level": 0,
            "agent_actions": 12.0,
            "baseline_actions": 12.0,
            "efficiency": 1.0,
            "source": "live_measurement.per_level_efficiency",
        }
    ]
    assert metric["null_delta_methodology_note"] is None


def test_scenario_arc_wmte_4634_agent_twice_baseline_reports_quarter() -> None:
    """SCENARIO-ARC-WMTE-4634: agent_actions==2*baseline_actions reports 0.25."""

    from carnot import experiment_4634_live_action_efficiency_metric as mod

    metric = mod.compute_live_action_efficiency(
        _a2_artifact([_row(baseline_actions=10, agent_actions=20)])
    )

    assert metric["live_action_efficiency"] == pytest.approx(0.25)
    assert metric["per_level_efficiency"][0]["efficiency"] == pytest.approx(0.25)
    assert metric["per_level_efficiency"][0]["agent_actions"] == pytest.approx(20.0)
    assert metric["per_level_efficiency"][0]["baseline_actions"] == pytest.approx(10.0)


def test_scenario_arc_wmte_4634_zero_solved_levels_reports_zero_with_null_note() -> None:
    """REQ-ARC-WMTE-4634: zero live-solved levels reports 0.0 with a null note."""

    from carnot import experiment_4634_live_action_efficiency_metric as mod

    metric = mod.compute_live_action_efficiency(
        _a2_artifact([_row(baseline_actions=10, agent_actions=20, solved=False)])
    )
    artifact = mod.build_artifact(
        preconditions_checked={"ok": True, "offline_arcade": True},
        a2_artifact=_a2_artifact([]),
        registry={"reproducible_total_levels": 55},
        offline_to_live_artifact={"offline_to_live_transfer_ratio": 0.0},
        live_submittable_artifact={"live_submittable_level_count": 54},
        integration_artifact={},
        duration_s=0.001,
        tests_added={"passed": True, "test_file": __file__},
    )

    assert metric["live_action_efficiency"] == pytest.approx(0.0)
    assert metric["per_level_efficiency"] == []
    assert "undefined" in metric["null_delta_methodology_note"]
    assert artifact["live_action_efficiency"] == pytest.approx(0.0)
    assert "undefined" in artifact["null_delta_methodology_note"]
    assert artifact["coheadline_block"]["reported_side_by_side"] == [
        "live_action_efficiency",
        "reproducible_total_levels",
        "live_submittable_level_count",
        "first_win_rate",
        "offline_to_live_transfer_ratio",
    ]
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4634_multi_level_mean_is_correct() -> None:
    """SCENARIO-ARC-WMTE-4634: multi-level mean averages per-level score terms."""

    from carnot import experiment_4634_live_action_efficiency_metric as mod

    metric = mod.compute_live_action_efficiency(
        _a2_artifact(
            [
                _row(game="aa00", level=0, baseline_actions=10, agent_actions=10),
                _row(game="bb00", level=0, baseline_actions=10, agent_actions=20),
                _row(game="cc00", level=1, baseline_actions=10, agent_actions=5),
            ]
        )
    )
    coheadline = mod.build_coheadline_block(
        a2_artifact={
            **_a2_artifact(
                [
                    _row(game="aa00", level=0, baseline_actions=10, agent_actions=10),
                    _row(game="bb00", level=0, baseline_actions=10, agent_actions=20),
                    _row(game="cc00", level=1, baseline_actions=10, agent_actions=5),
                ]
            ),
            "first_win_rate_delta": 0.25,
        },
        registry={"reproducible_total_levels": "55"},
        offline_to_live_artifact={"offline_to_live_transfer_ratio": 0.426},
        live_submittable_artifact={"live_submittable_level_count": "54"},
        integration_artifact={"action_efficiency_integrated": {"efficiency_score_term": 1.0}},
    )

    assert metric["per_level_efficiency"][0]["efficiency"] == pytest.approx(1.0)
    assert metric["per_level_efficiency"][1]["efficiency"] == pytest.approx(0.25)
    assert metric["per_level_efficiency"][2]["efficiency"] == pytest.approx(1.0)
    assert metric["live_action_efficiency"] == pytest.approx(0.75)
    assert coheadline["live_action_efficiency"] == pytest.approx(0.75)
    assert coheadline["reproducible_total_levels"] == 55
    assert coheadline["live_submittable_level_count"] == 54
    assert coheadline["first_win_rate"] == pytest.approx(0.5)
    assert coheadline["offline_to_live_transfer_ratio"] == pytest.approx(0.426)


def test_req_arc_wmte_4634_defensive_parsing_paths_are_deterministic() -> None:
    """REQ-ARC-WMTE-4634: malformed rows are skipped and legacy A2 rows still parse."""

    from carnot import experiment_4634_live_action_efficiency_metric as mod

    skipped = mod.compute_live_action_efficiency(
        {
            "live_measurement": {
                "per_level_efficiency": [
                    {"game": "aa00", "baseline_actions": 4, "live_solved": False},
                    {"game": "bb00"},
                    {"game": "cc00", "agent_actions": 4},
                ]
            }
        }
    )
    legacy = mod.compute_live_action_efficiency(
        {
            "live_measurement": {
                "per_game": {
                    "aa00": {
                        "solve_rate_predictor": 1.0,
                        "median_actions_to_first_levelup_predictor": 2.0,
                    }
                }
            }
        }
    )
    coheadline = mod.build_coheadline_block(
        a2_artifact={
            "per_level_efficiency": [_row(baseline_actions=3, agent_actions=3)],
            "aggregate_metrics": {"first_win_rate_predictor": 0.125},
        },
        registry={"reproducible_total_levels": 1},
        offline_to_live_artifact={},
    )

    assert mod._as_float(True, 7.5) == pytest.approx(7.5)
    assert mod._as_int(False, 8) == 8
    assert mod._as_int("not-an-int", 9) == 9
    assert skipped["live_action_efficiency"] == pytest.approx(0.0)
    assert skipped["per_level_efficiency"] == []
    assert legacy["live_action_efficiency"] == pytest.approx(0.25)
    assert legacy["per_level_efficiency"][0]["source"] == "live_measurement.per_game"
    assert coheadline["first_win_rate"] == pytest.approx(0.125)

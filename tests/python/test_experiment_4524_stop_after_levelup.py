"""Tests for Exp 4524 stop-after-levelup action-efficiency measurement.

Spec refs: REQ-ARC-FCP-4524, SCENARIO-ARC-FCP-4524.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_4524_stop_after_levelup as exp4524
from carnot.agentic.arc_competition_agent import E3AgentPolicy, SUBMITTED_AGENT_CONFIG


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _measurement(
    *,
    actions: dict[str, int],
    levels: dict[str, int] | None = None,
) -> dict[str, object]:
    level_map = levels or {game: 1 for game in actions}
    rows = []
    for game in exp4524.GATE_GAMES:
        solved = game in actions
        best_level = int(level_map.get(game, 0))
        total_actions = int(actions.get(game, 8000))
        rows.append(
            {
                "game": game,
                "solved": solved,
                "levels": best_level,
                "best_level": best_level,
                "actions": total_actions,
                "actions_to_reach_levels": (
                    {
                        str(level): min(total_actions, level * 10)
                        for level in range(1, best_level + 1)
                    }
                    if solved
                    else {}
                ),
            }
        )
    return exp4524.summarize_rows(rows, target_levels=exp4524.STOP_AT_SCORED_TARGET_LEVELS)


def test_req_arc_fcp_4524_spec_declares_required_fields() -> None:
    """REQ-ARC-FCP-4524: OpenSpec anchors the stop-after-levelup artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4524" in spec
    assert "SCENARIO-ARC-FCP-4524" in spec
    assert exp4524.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4524.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_fcp_4524_success_requires_core_and_level_depth(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-4524: lower total actions wins only with CORE and depth preserved."""

    control = _measurement(actions={"lp85": 7792, "m0r0": 7789, "sp80": 7724, "vc33": 7731})
    stopped = _measurement(actions={"lp85": 20, "m0r0": 4210, "sp80": 7218, "vc33": 1758})
    artifact = exp4524.build_artifact(
        preconditions_checked={"offline_arcade_import": True},
        control_measurement=control,
        stop_measurement=stopped,
        positive_control={"passed": True, "control_median": 7760.0, "improved_median": 6760.0},
        random_seed=4524,
        duration_s=0.25,
    )

    assert (
        artifact["honest_verdict"] == "success: stop_after_levelup_core_actions_2984_below_control"
    )
    assert artifact["core_solves_preserved"] is True
    assert artifact["levels_per_game_preserved"]["passed"] is True
    assert artifact["median_actions_on_core_control"] == 7760.0
    assert artifact["median_actions_on_core_best"] == 2984.0
    assert artifact["action_field_used"] == "actions"
    assert exp4524.artifact_schema_errors(artifact) == []

    out = tmp_path / exp4524.RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True)
    exp4524.write_artifact(artifact, tmp_path)
    assert json.loads(out.read_text(encoding="utf-8")) == artifact

    shed_depth = _measurement(
        actions={"lp85": 20, "m0r0": 4210, "sp80": 7218, "vc33": 1758},
        levels={"lp85": 1, "m0r0": 1, "sp80": 1, "vc33": 1},
    )
    deeper_control = _measurement(
        actions={"lp85": 7792, "m0r0": 7789, "sp80": 7724, "vc33": 7731},
        levels={"lp85": 2, "m0r0": 1, "sp80": 1, "vc33": 1},
    )
    null_artifact = exp4524.build_artifact(
        preconditions_checked={"offline_arcade_import": True},
        control_measurement=deeper_control,
        stop_measurement=shed_depth,
        positive_control={"passed": True, "control_median": 7760.0, "improved_median": 6760.0},
        random_seed=4524,
        duration_s=0.25,
    )

    assert (
        null_artifact["honest_verdict"]
        == "complete: stop_after_levelup_drops_level_depth_honest_null"
    )
    assert null_artifact["core_solves_preserved"] is True
    assert null_artifact["levels_per_game_preserved"]["passed"] is False
    assert exp4524.artifact_schema_errors(null_artifact) == []


def test_req_arc_fcp_4524_metric_mismatch_guard_rejects_mixed_action_fields() -> None:
    """REQ-ARC-FCP-4524: the harness refuses A3-style mixed action metrics."""

    control = _measurement(actions={"lp85": 7792, "m0r0": 7789, "sp80": 7724, "vc33": 7731})
    stopped = _measurement(actions={"lp85": 20, "m0r0": 4210, "sp80": 7218, "vc33": 1758})
    stopped["action_metric"] = {
        "field": "actions_to_first_levelup",
        "definition": "invalid mixed metric",
    }

    assert exp4524.action_metric_compatibility_error(control, stopped) == (
        "action metric mismatch control=actions treatment=actions_to_first_levelup"
    )


def test_req_arc_fcp_4524_submitted_target_matches_scored_stop_policy() -> None:
    """REQ-ARC-FCP-4524: the submitted explorer's target_levels is consistently wired
    end to end (SUBMITTED_AGENT_CONFIG -> the constructed policy's explorer).

    NOTE (2026-07-12): exp4524's own historical measurement (see
    `results/experiment_4524_stop_after_levelup.json`, honest_verdict
    `success: stop_after_levelup_core_actions_2825_below_control`) genuinely
    validated `STOP_AT_SCORED_TARGET_LEVELS = 1` as the accepted policy AT THE
    TIME. Commit `0fad75f38` ("PHASE A1 ... raise target_levels") later
    superseded that recommendation with a newer, deliberate scoring-lever
    decision (`SUBMITTED_TARGET_LEVELS` is now 3) -- a legitimate policy
    evolution, not drift, and exp4524's constant is correctly left unchanged
    as the historical record of what THAT experiment measured and
    recommended. This test therefore no longer asserts
    `SUBMITTED_AGENT_CONFIG["target_levels"] == exp4524.STOP_AT_SCORED_TARGET_LEVELS`
    (which would incorrectly imply the two are still supposed to match); it
    checks the still-meaningful invariant that the shipped config value is
    what actually reaches the constructed policy's explorer.
    """

    policy = E3AgentPolicy("lp85", proposer=None)

    assert policy.explorer.target_levels == SUBMITTED_AGENT_CONFIG["target_levels"]

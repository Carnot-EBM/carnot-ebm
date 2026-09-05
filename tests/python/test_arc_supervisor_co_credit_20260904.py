"""Spec: REQ-ARC-WMTE-7013, SCENARIO-ARC-WMTE-7013-A, SCENARIO-ARC-WMTE-7013-B,
SCENARIO-ARC-WMTE-7013-C, SCENARIO-ARC-WMTE-7013-D.

Redirect outcomes record how many redirects shared one level-up's credit.

INCIDENT 2026-09-03 (r11l tools A/B). Three redirects at actions 120, 240 and 360 were all
credited by the single level-up at 885 -- in BOTH arms of the A/B, including the arm where
the tool rung never fired. `helped` could not express that the credit was shared, so per-arm
follow rates read as effects. The receipt now carries the share; the refinement report shows
the strict counts next to the pooled one. The frozen decision rules are unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from carnot.agentic.arc_supervisor_refinement import (
    empty_ledger,
    evaluate,
    ingest_files,
    render_report,
)
from carnot.agentic.arc_trajectory_supervisor import (
    ARM_ALLOW_REINDUCTION,
    ARM_DROP_GOAL_BIAS,
    ARM_FORCE_DIVERSITY,
    ARM_ORDER,
    TrajectorySnapshot,
    TrajectorySupervisor,
)

# Constructing E3AgentPolicy loads the explorer stack (+549MB measured on a fresh worker),
# which the conftest memory watchdog reads as a leak. Same marker the sibling policy tests use.
pytestmark = pytest.mark.memory_watchdog_skip


def _snap(
    level: int = 0,
    goal_bias_installed: bool = False,
    induced: bool = False,
    induction_attempts: int = 0,
    new_transitions: int = 0,
    diversity_active: bool = False,
) -> TrajectorySnapshot:
    return TrajectorySnapshot(
        level=level,
        goal_bias_installed=goal_bias_installed,
        induced=induced,
        induction_attempts=induction_attempts,
        new_transitions_since_induction=new_transitions,
        diversity_active=diversity_active,
    )


def _fire_three_arms(sup: TrajectorySupervisor) -> None:
    """window=1: every stagnant action completes a window, so each observe fires the next arm."""
    assert sup.observe(_snap(goal_bias_installed=True)) is not None  # drop_goal_bias
    assert (
        sup.observe(_snap(induced=True, new_transitions=500, induction_attempts=0)) is not None
    )  # allow_reinduction
    assert sup.observe(_snap()) is not None  # force_exploration_diversity


# --- SCENARIO-A: a shared credit is counted -----------------------------------------------


def test_one_levelup_stamps_every_pending_redirect_with_the_share_size() -> None:
    """SCENARIO-ARC-WMTE-7013-A: three pending redirects, one level-up: each reads 3."""
    sup = TrajectorySupervisor(window=1)
    _fire_three_arms(sup)
    receipt = sup.receipt()
    assert [r["co_credited_count"] for r in receipt["redirects"]] == [None, None, None]

    sup.observe(_snap(level=1))

    receipt = sup.receipt()
    assert [r["co_credited_count"] for r in receipt["redirects"]] == [3, 3, 3]
    assert all(r["resolved_by_levelup"] for r in receipt["redirects"])
    for arm in (ARM_DROP_GOAL_BIAS, ARM_ALLOW_REINDUCTION, ARM_FORCE_DIVERSITY):
        assert receipt["arm_outcomes"][arm] == {"fired": 1, "helped": 1}  # unchanged shape
        assert receipt["arm_credit"][arm] == {"helped_sole": 0, "helped_share": 0.3333}


def test_a_sole_credit_reads_one() -> None:
    """SCENARIO-ARC-WMTE-7013-A: one pending redirect, one level-up: sole credit, full share."""
    sup = TrajectorySupervisor(window=1)
    assert sup.observe(_snap(goal_bias_installed=True)) is not None
    sup.observe(_snap(level=1))
    receipt = sup.receipt()
    (row,) = receipt["redirects"]
    assert row["co_credited_count"] == 1
    assert receipt["arm_credit"][ARM_DROP_GOAL_BIAS] == {"helped_sole": 1, "helped_share": 1.0}
    for arm in ARM_ORDER:
        if arm != ARM_DROP_GOAL_BIAS:
            assert receipt["arm_credit"][arm] == {"helped_sole": 0, "helped_share": 0.0}


def test_an_earlier_credit_is_frozen_by_a_later_levelup() -> None:
    """SCENARIO-ARC-WMTE-7013-A: a second level-up credits only the redirects pending since the
    first; the first row's share size does not change."""
    sup = TrajectorySupervisor(window=1)
    assert sup.observe(_snap(goal_bias_installed=True)) is not None
    sup.observe(_snap(level=1))
    assert sup.observe(_snap(level=1, goal_bias_installed=True)) is not None
    assert sup.observe(_snap(level=1)) is not None  # force diversity, pending too
    sup.observe(_snap(level=2))
    counts = [r["co_credited_count"] for r in sup.receipt()["redirects"]]
    assert counts == [1, 2, 2]


# --- SCENARIO-B: the shadow receipt renames the split too ---------------------------------


def test_the_shadow_receipt_marks_the_split_as_counterfactual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-7013-B: in shadow mode `arm_credit` becomes `would_have_arm_credit`,
    so a reader cannot ingest a counterfactual split as a real one."""
    from carnot.agentic import arc_competition_agent as agent
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    monkeypatch.delenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR", raising=False)
    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW", "1")
    monkeypatch.setattr(agent, "_level_of", lambda frame: int(frame.levels_completed))
    policy = E3AgentPolicy("lp85", proposer=object(), target_levels=2, value_head=None)
    policy.explorer = SimpleNamespace(goal_bias=object(), _hybrid_diversity=False)
    policy._maybe_supervise_trajectory(SimpleNamespace(levels_completed=0))
    policy._maybe_supervise_trajectory(SimpleNamespace(levels_completed=1))

    receipt = policy.trajectory_supervisor_diagnostics()
    assert receipt["mode"] == "shadow"
    assert "arm_credit" not in receipt
    assert receipt["would_have_arm_credit"][ARM_DROP_GOAL_BIAS] == {
        "helped_sole": 1,
        "helped_share": 1.0,
    }
    assert receipt["would_have_redirects"][0]["co_credited_count"] == 1


# --- SCENARIO-C / D: the refinement report shows the strict counts ------------------------


def _row(seed: int, redirects: list[dict]) -> dict:
    return {
        "game": "tu93",
        "seed": seed,
        "arm": "S",
        "levels": 1,
        "trajectory_supervisor": {
            "enabled": True,
            "mode": "applied",
            "window": 120,
            "actions_observed": 399,
            "arms_used": [],
            "arm_outcomes": {},
            "stagnations_unredirected": 0,
            "redirects": redirects,
        },
    }


def _redirect(arm: str, resolved: bool, k: int | None) -> dict:
    return {
        "arm": arm,
        "action_index": 100,
        "level": 0,
        "diagnosis": "t",
        "resolved_by_levelup": resolved,
        "actions_to_levelup": 50 if resolved else None,
        "co_credited_count": k,
    }


def test_the_refinement_report_carries_sole_and_share_next_to_helped(tmp_path) -> None:
    """SCENARIO-ARC-WMTE-7013-C: per-arm `helped_sole`, `helped_share` and a Wilson interval
    on the sole count; rows without the field count in neither."""
    import json

    rows = [
        _row(
            1, [_redirect(ARM_DROP_GOAL_BIAS, True, 3), _redirect(ARM_ALLOW_REINDUCTION, True, 3)]
        ),
        _row(2, [_redirect(ARM_DROP_GOAL_BIAS, True, 1)]),
        _row(3, [_redirect(ARM_DROP_GOAL_BIAS, True, None)]),  # a row written before the field
        _row(4, [_redirect(ARM_DROP_GOAL_BIAS, False, None)]),
    ]
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    ledger = empty_ledger()
    ingest_files(ledger, [path], NOW := "2026-09-04T00:00:00+00:00")
    rec = evaluate(ledger, NOW)
    by_arm = {row["arm"]: row for row in rec["per_arm"]}
    drop = by_arm[ARM_DROP_GOAL_BIAS]
    assert (drop["fired"], drop["helped"]) == (4, 3)
    assert drop["helped_sole"] == 1
    assert drop["helped_share"] == round(1 / 3 + 1.0, 4)
    assert drop["helped_with_known_split"] == 2
    assert 0.0 < drop["sole_wilson_lower"] < drop["sole_wilson_upper"] < 1.0
    assert drop["sole_wilson_upper"] <= drop["wilson_upper"]
    assert by_arm[ARM_ALLOW_REINDUCTION]["helped_sole"] == 0
    assert by_arm[ARM_ALLOW_REINDUCTION]["helped_share"] == 0.3333
    report = render_report(rec)
    assert "sole=1 share=1.3333" in report


def test_the_frozen_rules_still_key_on_pooled_helped(tmp_path) -> None:
    """SCENARIO-ARC-WMTE-7013-D: the retire rule reads `helped`, not the strict count -- an arm
    with ten shared credits is NOT a retire candidate. Changing the rule is a spec act."""
    import json

    rows = [_row(i, [_redirect(ARM_DROP_GOAL_BIAS, True, 3)]) for i in range(10)]
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    ledger = empty_ledger()
    ingest_files(ledger, [path], "2026-09-04T00:00:00+00:00")
    rec = evaluate(ledger, "2026-09-04T00:00:00+00:00")
    drop = next(r for r in rec["per_arm"] if r["arm"] == ARM_DROP_GOAL_BIAS)
    assert drop["helped"] == 10 and drop["helped_sole"] == 0
    assert not any(r["kind"] == "retire_candidate" for r in rec["recommendations"])

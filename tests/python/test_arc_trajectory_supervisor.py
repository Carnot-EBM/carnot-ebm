"""REQ-ARC-WMTE-6600: trajectory supervisor for the live agent (default OFF).

The supervisor is the weak-generator adaptation of AVO's supervisor
(arXiv 2603.24517): mechanical stagnation detection over trajectory
statistics, plus a closed decision table of strategy redirects applied only
through existing seams. See
docs/research-notes/avo-adaptation-for-local-generator-2026-08-21.md.
"""

from __future__ import annotations

from types import SimpleNamespace

from carnot.agentic import arc_competition_agent as agent
from carnot.agentic.arc_competition_agent import E3AgentPolicy
from carnot.agentic.arc_trajectory_supervisor import (
    ARM_ALLOW_REINDUCTION,
    ARM_DROP_GOAL_BIAS,
    ARM_FORCE_DIVERSITY,
    Redirect,
    TrajectorySnapshot,
    TrajectorySupervisor,
)


def _snap(**overrides) -> TrajectorySnapshot:
    base = {
        "level": 0,
        "goal_bias_installed": False,
        "induced": False,
        "induction_attempts": 0,
        "new_transitions_since_induction": 0,
        "diversity_active": False,
    }
    base.update(overrides)
    return TrajectorySnapshot(**base)


def _redirect(arm: str) -> Redirect:
    return Redirect(arm=arm, action_index=1, level=0, diagnosis="test")


# --- SCENARIO-ARC-WMTE-6600-1 (default off) ---


def test_scenario_6600_1_default_off_no_supervisor(monkeypatch):
    """SCENARIO-ARC-WMTE-6600-1: env unset -> no supervisor, disabled diagnostics."""

    monkeypatch.delenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR", raising=False)
    policy = E3AgentPolicy("lp85", proposer=object(), target_levels=2, value_head=None)

    assert policy._trajectory_supervisor is None
    assert policy.trajectory_supervisor_diagnostics() == {"enabled": False}


def test_scenario_6600_1_off_observe_path_is_inert(monkeypatch):
    """SCENARIO-ARC-WMTE-6600-1: with the supervisor off the routed-path hook
    is a no-op even when a frame is present."""

    monkeypatch.delenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR", raising=False)
    monkeypatch.setattr(agent, "_level_of", lambda frame: int(frame.levels_completed))
    policy = E3AgentPolicy("lp85", proposer=object(), target_levels=2, value_head=None)
    policy.induced = True

    policy._maybe_supervise_trajectory(SimpleNamespace(levels_completed=0))

    assert policy.induced is True  # nothing observed, nothing redirected


# --- SCENARIO-ARC-WMTE-6600-2 (window, progress resets) ---


def test_scenario_6600_2_no_fire_before_window():
    """SCENARIO-ARC-WMTE-6600-2: fewer stagnant observations than the window
    produce no redirect; the window-th one fires."""

    sup = TrajectorySupervisor(window=5)
    for _ in range(4):
        assert sup.observe(_snap(goal_bias_installed=True)) is None
    fired = sup.observe(_snap(goal_bias_installed=True))

    assert fired is not None
    assert fired.arm == ARM_DROP_GOAL_BIAS


def test_scenario_6600_2_levelup_resets_counter():
    """SCENARIO-ARC-WMTE-6600-2: a level-up observation resets stagnation."""

    sup = TrajectorySupervisor(window=3)
    assert sup.observe(_snap(level=0, goal_bias_installed=True)) is None
    assert sup.observe(_snap(level=0, goal_bias_installed=True)) is None
    # Level-up: the counter restarts, so two more stagnant actions stay quiet.
    assert sup.observe(_snap(level=1, goal_bias_installed=True)) is None
    assert sup.observe(_snap(level=1, goal_bias_installed=True)) is None
    assert sup.observe(_snap(level=1, goal_bias_installed=True)) is None
    fired = sup.observe(_snap(level=1, goal_bias_installed=True))

    assert fired is not None and fired.level == 1


# --- SCENARIO-ARC-WMTE-6600-3 (arm order, one per episode, once per level) ---


def test_scenario_6600_3_arm_order_and_bounds():
    """SCENARIO-ARC-WMTE-6600-3: fixed order, one arm per stagnation episode,
    each arm at most once per level, then silence."""

    sup = TrajectorySupervisor(window=2)
    eligible_all = {
        "goal_bias_installed": True,
        "induced": True,
        "new_transitions_since_induction": 250,
        "induction_attempts": 1,
        "diversity_active": False,
    }
    fired = []
    for _ in range(12):  # 6 windows' worth of stagnation, every arm eligible
        redirect = sup.observe(_snap(**eligible_all))
        if redirect is not None:
            fired.append(redirect.arm)

    assert fired == [ARM_DROP_GOAL_BIAS, ARM_ALLOW_REINDUCTION, ARM_FORCE_DIVERSITY]


def test_scenario_6600_3_skips_ineligible_arms():
    """SCENARIO-ARC-WMTE-6600-3: with no bias installed and no latch, the first
    stagnation fires the diversity arm directly."""

    sup = TrajectorySupervisor(window=2)
    sup.observe(_snap())
    fired = sup.observe(_snap())

    assert fired is not None
    assert fired.arm == ARM_FORCE_DIVERSITY


def test_scenario_6600_3_nothing_eligible_restarts_window():
    """SCENARIO-ARC-WMTE-6600-3: when nothing is eligible the window restarts
    instead of re-evaluating every action."""

    sup = TrajectorySupervisor(window=2)
    busy = _snap(diversity_active=True)  # only arm left is ineligible
    assert sup.observe(busy) is None
    assert sup.observe(busy) is None  # window reached, nothing eligible
    # A bias installed one action later must wait a FULL fresh window.
    assert sup.observe(_snap(goal_bias_installed=True, diversity_active=True)) is None
    fired = sup.observe(_snap(goal_bias_installed=True, diversity_active=True))

    assert fired is not None and fired.arm == ARM_DROP_GOAL_BIAS


# --- SCENARIO-ARC-WMTE-6600-4 (reinduction bounds) ---


def test_scenario_6600_4_reinduction_evidence_floor():
    """SCENARIO-ARC-WMTE-6600-4: below 200 new transitions the reinduction arm
    is not eligible; the ladder falls through to diversity."""

    sup = TrajectorySupervisor(window=2)
    starved = _snap(induced=True, new_transitions_since_induction=199)
    sup.observe(starved)
    fired = sup.observe(starved)

    assert fired is not None
    assert fired.arm == ARM_FORCE_DIVERSITY


def test_scenario_6600_4_reinduction_attempt_cap():
    """SCENARIO-ARC-WMTE-6600-4: at 3 attempts the reinduction arm is not
    eligible even with plentiful evidence."""

    sup = TrajectorySupervisor(window=2)
    capped = _snap(induced=True, new_transitions_since_induction=500, induction_attempts=3)
    sup.observe(capped)
    fired = sup.observe(capped)

    assert fired is not None
    assert fired.arm == ARM_FORCE_DIVERSITY


# --- SCENARIO-ARC-WMTE-6600-5 (application seams) ---


def test_scenario_6600_5_drop_goal_bias_seam(monkeypatch):
    """SCENARIO-ARC-WMTE-6600-5: the bias drop goes through set_goal_bias(None)."""

    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR", "1")
    policy = E3AgentPolicy("lp85", proposer=object(), target_levels=2, value_head=None)
    policy.explorer.set_goal_bias(lambda grid: 0.0, label="induced-goal")
    assert policy.explorer.goal_bias is not None

    policy._apply_trajectory_redirect(_redirect(ARM_DROP_GOAL_BIAS))

    assert policy.explorer.goal_bias is None


def test_scenario_6600_5_allow_reinduction_seam(monkeypatch):
    """SCENARIO-ARC-WMTE-6600-5: the reinduction arm resets the latch only."""

    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR", "1")
    policy = E3AgentPolicy("lp85", proposer=object(), target_levels=2, value_head=None)
    policy.induced = True

    policy._apply_trajectory_redirect(_redirect(ARM_ALLOW_REINDUCTION))

    assert policy.induced is False


def test_scenario_6600_5_force_diversity_seam(monkeypatch):
    """SCENARIO-ARC-WMTE-6600-5: the diversity arm enables the hybrid draw and
    pushes the stall counter past the threshold."""

    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR", "1")
    policy = E3AgentPolicy("lp85", proposer=object(), target_levels=2, value_head=None)
    assert policy.explorer._hybrid_diversity is False

    policy._apply_trajectory_redirect(_redirect(ARM_FORCE_DIVERSITY))

    assert policy.explorer._hybrid_diversity is True
    assert policy.explorer._steps_since_progress > policy.explorer._stall_threshold


# --- SCENARIO-ARC-WMTE-6600-6 (receipts) ---


def test_scenario_6600_6_receipts():
    """SCENARIO-ARC-WMTE-6600-6: the redirect ledger carries action index,
    level, arm, diagnosis, plus the used-arm set and window."""

    sup = TrajectorySupervisor(window=2)
    sup.observe(_snap(level=3, goal_bias_installed=True))
    sup.observe(_snap(level=3, goal_bias_installed=True))
    receipt = sup.receipt()

    assert receipt["enabled"] is True
    assert receipt["window"] == 2
    assert receipt["actions_observed"] == 2
    assert receipt["arms_used"] == [ARM_DROP_GOAL_BIAS]
    (row,) = receipt["redirects"]
    assert row["arm"] == ARM_DROP_GOAL_BIAS
    assert row["action_index"] == 2
    assert row["level"] == 3
    assert row["diagnosis"]


# --- SCENARIO-ARC-WMTE-6600-7 (level-up resets arms) ---


def test_scenario_6600_7_levelup_resets_arms():
    """SCENARIO-ARC-WMTE-6600-7: after a level-up the same arm can fire again."""

    sup = TrajectorySupervisor(window=2)
    sup.observe(_snap(level=0, goal_bias_installed=True))
    first = sup.observe(_snap(level=0, goal_bias_installed=True))
    assert first is not None and first.arm == ARM_DROP_GOAL_BIAS
    # Level-up, then a fresh stagnant window on the new level.
    assert sup.observe(_snap(level=1, goal_bias_installed=True)) is None
    sup.observe(_snap(level=1, goal_bias_installed=True))
    second = sup.observe(_snap(level=1, goal_bias_installed=True))

    assert second is not None and second.arm == ARM_DROP_GOAL_BIAS


# --- REQ-ARC-WMTE-6600 wiring (rules 1, 2, 7) ---


def test_req_6600_enabled_constructs_with_window(monkeypatch):
    """REQ-ARC-WMTE-6600 rules 1+2: the env flag constructs the supervisor and
    the window env var sizes it."""

    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR", "1")
    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW", "7")
    policy = E3AgentPolicy("lp85", proposer=object(), target_levels=2, value_head=None)

    assert isinstance(policy._trajectory_supervisor, TrajectorySupervisor)
    assert policy._trajectory_supervisor.window == 7
    assert policy.trajectory_supervisor_diagnostics()["enabled"] is True


def test_req_6600_snapshot_reflects_policy_state(monkeypatch):
    """REQ-ARC-WMTE-6600 rule 1: the per-action snapshot carries the policy's
    real induction and explorer state, and a redirect is applied."""

    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR", "1")
    monkeypatch.setattr(agent, "_level_of", lambda frame: int(frame.levels_completed))
    policy = E3AgentPolicy("lp85", proposer=object(), target_levels=2, value_head=None)
    policy.induced = True
    policy._induction_attempt_count = 2
    policy.transitions = [object()] * 5
    policy._transitions_at_last_induction_attempt = 1

    seen: list[TrajectorySnapshot] = []

    class _Stub:
        def observe(self, snapshot):
            seen.append(snapshot)
            return _redirect(ARM_ALLOW_REINDUCTION)

    policy._trajectory_supervisor = _Stub()
    policy._maybe_supervise_trajectory(SimpleNamespace(levels_completed=4))

    (snapshot,) = seen
    assert snapshot.level == 4
    assert snapshot.induced is True
    assert snapshot.induction_attempts == 2
    assert snapshot.new_transitions_since_induction == 4
    assert snapshot.diversity_active is False
    assert policy.induced is False  # the returned redirect was applied


def test_req_6600_routed_path_consults_supervisor(monkeypatch):
    """REQ-ARC-WMTE-6600 rule 1: `_next_move_routed` calls the supervisor hook
    exactly once per action."""

    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR", "1")
    policy = E3AgentPolicy("lp85", proposer=object(), target_levels=2, value_head=None)
    calls: list[object] = []
    policy._maybe_supervise_trajectory = lambda latest: calls.append(latest)

    move = policy._next_move_routed([], None)

    assert len(calls) == 1
    assert move == ("RESET", None)  # explorer bootstrap; the routed path ran

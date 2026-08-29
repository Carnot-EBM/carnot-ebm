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


def test_scenario_6600_1_default_is_shadow_nothing_applies(monkeypatch):
    """SCENARIO-ARC-WMTE-6600-1 as superseded by REQ-ARC-WMTE-6660 rule 1:
    the INTENT — env unset applies nothing to the scored path — is
    unchanged; the mechanics moved from no-supervisor to a shadow instance
    whose receipt says so unambiguously."""

    monkeypatch.delenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR", raising=False)
    policy = E3AgentPolicy("lp85", proposer=object(), target_levels=2, value_head=None)

    assert policy._trajectory_supervisor is not None
    assert policy._trajectory_supervisor_applies is False
    d = policy.trajectory_supervisor_diagnostics()
    assert d["enabled"] is False
    assert d["mode"] == "shadow"


def test_scenario_6600_1_shadow_observe_path_applies_nothing(monkeypatch):
    """SCENARIO-ARC-WMTE-6600-1 / 6660: with the env unset the routed-path
    hook OBSERVES but never mutates the policy, even when a frame is
    present."""

    monkeypatch.delenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR", raising=False)
    monkeypatch.setattr(agent, "_level_of", lambda frame: int(frame.levels_completed))
    policy = E3AgentPolicy("lp85", proposer=object(), target_levels=2, value_head=None)
    policy.induced = True

    policy._maybe_supervise_trajectory(SimpleNamespace(levels_completed=0))

    assert policy.induced is True  # observed in shadow, nothing redirected
    assert policy.trajectory_supervisor_diagnostics()["actions_observed"] == 1


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


# --- REQ-ARC-WMTE-6640 (redirect outcomes make the ledger measurable) ---


def test_scenario_6640_1_outcome_fields_present_not_absent():
    """SCENARIO-ARC-WMTE-6640-1: a redirect no level-up has followed carries
    `resolved_by_levelup: False` and `actions_to_levelup: None` — present,
    not absent. No end-of-run finalize step exists to add them later."""

    sup = TrajectorySupervisor(window=2)
    sup.observe(_snap(goal_bias_installed=True))
    assert sup.observe(_snap(goal_bias_installed=True)) is not None

    (row,) = sup.receipt()["redirects"]
    assert row["resolved_by_levelup"] is False
    assert row["actions_to_levelup"] is None


def test_scenario_6640_2_levelup_credits_open_redirects():
    """SCENARIO-ARC-WMTE-6640-2: a level-up credits the redirect fired since
    the last progress event, with the action distance to the level-up."""

    sup = TrajectorySupervisor(window=2)
    sup.observe(_snap(goal_bias_installed=True))
    fired = sup.observe(_snap(goal_bias_installed=True))  # redirect at action 2
    assert fired is not None
    sup.observe(_snap(goal_bias_installed=True))  # action 3, still stagnant
    sup.observe(_snap(level=1))  # action 4, level-up

    (row,) = sup.receipt()["redirects"]
    assert row["resolved_by_levelup"] is True
    assert row["actions_to_levelup"] == 2  # actions 3 and 4


def test_scenario_6640_2_earlier_resolution_stays_frozen():
    """SCENARIO-ARC-WMTE-6640-2: a later level-up must not re-credit a
    redirect an earlier level-up already resolved. The two gaps here differ
    (2 vs 1), so a wrongful re-credit would change the first number."""

    sup = TrajectorySupervisor(window=2)
    sup.observe(_snap(goal_bias_installed=True))
    assert sup.observe(_snap(goal_bias_installed=True)) is not None  # A at action 2
    sup.observe(_snap(goal_bias_installed=True))  # action 3
    sup.observe(_snap(level=1))  # action 4: A credited with 2
    sup.observe(_snap(level=1, goal_bias_installed=True))  # action 5
    assert sup.observe(_snap(level=1, goal_bias_installed=True)) is not None  # B at 6
    sup.observe(_snap(level=2))  # action 7: B credited with 1

    rows = sup.receipt()["redirects"]
    assert [r["actions_to_levelup"] for r in rows] == [2, 1]
    assert all(r["resolved_by_levelup"] for r in rows)


def test_scenario_6640_3_arm_outcomes_aggregate_with_zeros():
    """SCENARIO-ARC-WMTE-6640-3: every arm appears in `arm_outcomes`, zeros
    included. One credited drop_goal_bias firing reads fired=1 helped=1;
    the arms that never fired read 0/0 rather than being absent."""

    sup = TrajectorySupervisor(window=2)
    sup.observe(_snap(goal_bias_installed=True))
    assert sup.observe(_snap(goal_bias_installed=True)) is not None
    sup.observe(_snap(level=1))  # credit it

    outcomes = sup.receipt()["arm_outcomes"]
    assert outcomes[ARM_DROP_GOAL_BIAS] == {"fired": 1, "helped": 1}
    assert outcomes[ARM_ALLOW_REINDUCTION] == {"fired": 0, "helped": 0}
    assert outcomes[ARM_FORCE_DIVERSITY] == {"fired": 0, "helped": 0}


def test_scenario_6640_3_unhelped_firing_counts_fired_only():
    """SCENARIO-ARC-WMTE-6640-3: a firing never followed by progress counts
    in `fired` and not in `helped`."""

    sup = TrajectorySupervisor(window=2)
    sup.observe(_snap(goal_bias_installed=True))
    assert sup.observe(_snap(goal_bias_installed=True)) is not None

    outcomes = sup.receipt()["arm_outcomes"]
    assert outcomes[ARM_DROP_GOAL_BIAS] == {"fired": 1, "helped": 0}


def test_scenario_6640_4_exhausted_stagnation_counted():
    """SCENARIO-ARC-WMTE-6640-4: a window that fires no arm increments
    `stagnations_unredirected`; a window that fires one does not."""

    sup = TrajectorySupervisor(window=2)
    busy = _snap(diversity_active=True)  # the only eligible-by-state arm is off
    sup.observe(busy)
    sup.observe(busy)  # window reached, nothing eligible
    assert sup.receipt()["stagnations_unredirected"] == 1

    fires = TrajectorySupervisor(window=2)
    fires.observe(_snap(goal_bias_installed=True))
    assert fires.observe(_snap(goal_bias_installed=True)) is not None
    assert fires.receipt()["stagnations_unredirected"] == 0


def test_scenario_6640_6_default_off_no_ledger_keys_leak(monkeypatch):
    """SCENARIO-ARC-WMTE-6640-6 as superseded by REQ-ARC-WMTE-6660: the
    INTENT — applied-ledger keys must never appear on a run where nothing
    was applied — is preserved by the shadow key rename. With the env unset
    the receipt says enabled False, and `redirects` / `arm_outcomes` (the
    keys REAL outcomes live under) do not exist."""

    monkeypatch.delenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR", raising=False)
    policy = E3AgentPolicy("lp85", proposer=object(), target_levels=2, value_head=None)

    d = policy.trajectory_supervisor_diagnostics()
    assert d["enabled"] is False
    assert "redirects" not in d
    assert "arm_outcomes" not in d


# --- REQ-ARC-WMTE-6660 (shadow mode: every run's counterfactual is readable) ---


def _shadow_policy(monkeypatch, window: str = "2"):
    monkeypatch.delenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR", raising=False)
    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW", window)
    monkeypatch.setattr(agent, "_level_of", lambda frame: int(frame.levels_completed))
    return E3AgentPolicy("lp85", proposer=object(), target_levels=2, value_head=None)


def test_scenario_6660_shadow_records_without_applying(monkeypatch):
    """SCENARIO-ARC-WMTE-6660-SHADOW-RECORDS-WITHOUT-APPLYING: a stagnant
    window in shadow mode lands in would_have_redirects while the explorer
    and the induction latch stay untouched."""

    policy = _shadow_policy(monkeypatch)
    policy._next_move_routed([], None)  # bootstrap constructs the explorer
    policy.induced = True
    explorer = policy.explorer
    diversity_before = explorer._hybrid_diversity
    goal_bias_before = getattr(explorer, "goal_bias", None)

    frame = SimpleNamespace(levels_completed=0)
    policy._maybe_supervise_trajectory(frame)
    policy._maybe_supervise_trajectory(frame)  # window=2 -> would-have fire

    # NOTHING applied, whichever arm was chosen: bias identity, diversity
    # flag and the induction latch are all untouched.
    assert explorer._hybrid_diversity is diversity_before
    assert getattr(explorer, "goal_bias", None) is goal_bias_before
    assert policy.induced is True
    d = policy.trajectory_supervisor_diagnostics()
    (row,) = d["would_have_redirects"]
    assert row["arm"] in (ARM_DROP_GOAL_BIAS, ARM_ALLOW_REINDUCTION, ARM_FORCE_DIVERSITY)
    assert row["levelup_followed_without_redirect"] is False
    assert row["actions_to_levelup_without_redirect"] is None


def test_scenario_6660_applied_mode_unchanged(monkeypatch):
    """SCENARIO-ARC-WMTE-6660-APPLIED-MODE-UNCHANGED: env=1 keeps the
    REQ-6600/6640 shape and behavior, plus mode: applied."""

    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR", "1")
    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW", "2")
    monkeypatch.setattr(agent, "_level_of", lambda frame: int(frame.levels_completed))
    policy = E3AgentPolicy("lp85", proposer=object(), target_levels=2, value_head=None)
    policy._next_move_routed([], None)

    policy.induced = True
    frame = SimpleNamespace(levels_completed=0)
    policy._maybe_supervise_trajectory(frame)
    policy._maybe_supervise_trajectory(frame)

    d = policy.trajectory_supervisor_diagnostics()
    assert d["mode"] == "applied"
    assert d["enabled"] is True
    (row,) = d["redirects"]
    # The redirect APPLIED — asserted by the effect of whichever arm fired.
    if row["arm"] == ARM_DROP_GOAL_BIAS:
        assert policy.explorer.goal_bias is None
    elif row["arm"] == ARM_ALLOW_REINDUCTION:
        assert policy.induced is False
    else:
        assert policy.explorer._hybrid_diversity is True


def test_scenario_6660_keys_disambiguate(monkeypatch):
    """SCENARIO-ARC-WMTE-6660-KEYS-DISAMBIGUATE: a reader summing `redirects`
    can never ingest a shadow counterfactual, by construction."""

    shadow = _shadow_policy(monkeypatch)
    d_shadow = shadow.trajectory_supervisor_diagnostics()
    assert "redirects" not in d_shadow
    assert "arm_outcomes" not in d_shadow
    assert d_shadow["mode"] == "shadow" and d_shadow["enabled"] is False

    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR", "1")
    applied = E3AgentPolicy("lp85", proposer=object(), target_levels=2, value_head=None)
    d_applied = applied.trajectory_supervisor_diagnostics()
    assert not any(k.startswith("would_have") for k in d_applied)
    assert d_applied["mode"] == "applied"


def test_scenario_6660_raising_supervisor_cannot_break_the_run(monkeypatch):
    """SCENARIO-ARC-WMTE-6660-RAISING-SUPERVISOR-CANNOT-BREAK-THE-RUN:
    fail-open for the scored path, visible in the receipt."""

    policy = _shadow_policy(monkeypatch)

    class _Poisoned:
        window = 2

        def observe(self, snapshot):
            raise RuntimeError("diagnostics must never cost the run")

        def receipt(self):
            return {
                "enabled": True,
                "window": 2,
                "actions_observed": 0,
                "arms_used": [],
                "redirects": [],
                "arm_outcomes": {},
                "stagnations_unredirected": 0,
            }

    policy._trajectory_supervisor = _Poisoned()
    frame = SimpleNamespace(levels_completed=0)
    policy._maybe_supervise_trajectory(frame)  # must not raise
    policy._maybe_supervise_trajectory(frame)

    assert policy.trajectory_supervisor_diagnostics()["observe_errors"] == 2


# --- The fourth rung: tool-loop re-induction (REQ-ARC-WMTE-6760, 2026-08-29) ------------------
# Default OFF. The selfparse transport's gate passed at ceiling on 2026-08-28, so the loop RUNS;
# whether it induces BETTER is what the resumed holdout-equalized A/B measures, and that has not
# reported. The arm exists so the supervisor can be finetuned against it through the outcome
# ledger, without an unmeasured lever changing live scored behaviour first.

import os as _os  # noqa: E402

from carnot.agentic.arc_trajectory_supervisor import (  # noqa: E402
    ARM_ALLOW_REINDUCTION,
    ARM_TOOL_LOOP_REINDUCTION,
    tool_loop_arm_enabled,
)


def _stagnate(sup, snapshot, n):
    """Run n actions with no level change; return the last redirect seen."""
    last = None
    for _ in range(n):
        got = sup.observe(snapshot)
        if got is not None:
            last = got
    return last


def test_the_tool_arm_is_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("CARNOT_ARC_SUPERVISOR_TOOL_ARM", raising=False)
    assert tool_loop_arm_enabled() is False


def test_only_the_exact_string_one_arms_it(monkeypatch) -> None:
    """A stray truthy value must not switch a live-path strategy change on by accident."""
    monkeypatch.setenv("CARNOT_ARC_SUPERVISOR_TOOL_ARM", "yes")
    assert tool_loop_arm_enabled() is False
    monkeypatch.setenv("CARNOT_ARC_SUPERVISOR_TOOL_ARM", "1")
    assert tool_loop_arm_enabled() is True


def test_the_tool_arm_never_fires_while_disabled(monkeypatch) -> None:
    """The whole point of default-off: an unmeasured lever changes nothing live."""
    monkeypatch.delenv("CARNOT_ARC_SUPERVISOR_TOOL_ARM", raising=False)
    sup = TrajectorySupervisor(window=3, reinduction_evidence_floor=0)
    snap = TrajectorySnapshot(
        level=0,
        goal_bias_installed=False,
        induced=True,
        induction_attempts=1,
        new_transitions_since_induction=500,
        diversity_active=True,
    )
    fired = {r["arm"] for r in _fired_arms(sup, snap, 40)}
    assert ARM_TOOL_LOOP_REINDUCTION not in fired


def _fired_arms(sup, snap, n):
    for _ in range(n):
        sup.observe(snap)
    return sup.receipt()["redirects"]


def test_the_tool_arm_waits_for_plain_reinduction_to_be_spent(monkeypatch) -> None:
    """It is the ESCALATION rung: paying for a multi-turn loop is only honest once the
    single-shot re-draw has already been spent on this level and stagnation continued."""
    monkeypatch.setenv("CARNOT_ARC_SUPERVISOR_TOOL_ARM", "1")
    sup = TrajectorySupervisor(window=3, reinduction_evidence_floor=0)
    # induced=False makes ARM_ALLOW_REINDUCTION ineligible, so it is never spent.
    snap = TrajectorySnapshot(
        level=0,
        goal_bias_installed=False,
        induced=False,
        induction_attempts=0,
        new_transitions_since_induction=0,
        diversity_active=True,
    )
    fired = {r["arm"] for r in _fired_arms(sup, snap, 40)}
    assert ARM_TOOL_LOOP_REINDUCTION not in fired


def test_the_tool_arm_fires_after_reinduction_and_is_recorded(monkeypatch) -> None:
    monkeypatch.setenv("CARNOT_ARC_SUPERVISOR_TOOL_ARM", "1")
    sup = TrajectorySupervisor(window=3, reinduction_evidence_floor=0)
    snap = TrajectorySnapshot(
        level=0,
        goal_bias_installed=False,
        induced=True,
        induction_attempts=1,
        new_transitions_since_induction=500,
        diversity_active=True,
    )
    fired = [r["arm"] for r in _fired_arms(sup, snap, 40)]
    assert ARM_ALLOW_REINDUCTION in fired
    assert ARM_TOOL_LOOP_REINDUCTION in fired
    assert fired.index(ARM_ALLOW_REINDUCTION) < fired.index(ARM_TOOL_LOOP_REINDUCTION)


def test_the_ledger_reports_the_new_arm_so_it_can_be_finetuned(monkeypatch) -> None:
    """An arm with no outcome row cannot be retired or promoted -- which is the whole
    purpose of adding it before the evidence lands."""
    monkeypatch.setenv("CARNOT_ARC_SUPERVISOR_TOOL_ARM", "1")
    sup = TrajectorySupervisor(window=3, reinduction_evidence_floor=0)
    assert ARM_TOOL_LOOP_REINDUCTION in sup.receipt()["arm_outcomes"]

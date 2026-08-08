"""Spec: REQ-ARC-WMTE-6229.

Regression tests for the bounded re-induction gap.

docs/research-notes/live-agent-adversarial-review-2026-08-08.md, "Gaps" section, major
finding 2:

  "LLM induction fires at most once per level on the scored path: `self.induced` latches
  and resets only at a level boundary or under the env-gated active probe, so after one
  refused induction (the measured modal outcome) the remaining ~1900 actions accumulate
  exactly the transition evidence a second attempt would need while the LLM tier is
  structurally unreachable. ... Fix: bounded re-induction on renewed stall (reset the latch
  after N new transitions, capped K attempts), or at minimum record 'tier latched off' in
  the liveness witness."

THE FIX. Both halves of the review's suggested fix, shipped together:

  1. ALWAYS-ON diagnostic: `generator_liveness_witness()` now reports
     `induction_tier_latched_off`, `induction_attempt_count`,
     `transitions_since_last_induction_attempt`, and `bounded_reinduction_enabled` --
     the minimum fix, zero behaviour change.
  2. GATED-OFF-BY-DEFAULT mechanism: `_should_enter_induction` gains a third branch
     (`"renewed_stall_reinduction"`) that resets the latch once `_REINDUCTION_TRANSITION_
     THRESHOLD` (200) new transitions have accumulated since the last attempt, capped at
     `_REINDUCTION_MAX_ATTEMPTS` (3) attempts per level. Gated behind
     `CARNOT_ARC_BOUNDED_REINDUCTION=1` because its effect on win rate is unmeasured (the
     400->2000 action-budget raise that makes the wasted tail large was validated on an
     LLM-off sweep) -- shipping it as a default would repeat the exact mistake Gaps
     finding 1 (the dead early-stop-grace flag) already made once.
"""

from __future__ import annotations

from carnot.agentic import arc_competition_agent as agent
from carnot.agentic.arc_competition_agent import E3AgentPolicy


def _policy(**overrides) -> E3AgentPolicy:
    p = E3AgentPolicy("lp85", proposer=object(), target_levels=3, value_head=None)
    p.transitions = list(range(overrides.pop("n_transitions", 0)))
    p._episode_transition_start = 0
    p._transitions_at_last_induction_attempt = overrides.pop("transitions_at_last_attempt", 0)
    p._induction_attempt_count = overrides.pop("attempt_count", 0)
    p.induced = overrides.pop("induced", True)
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


class TestGatedOffByDefault:
    def test_renewed_stall_never_fires_when_env_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("CARNOT_ARC_BOUNDED_REINDUCTION", raising=False)
        p = _policy(n_transitions=500, transitions_at_last_attempt=0, attempt_count=0)

        should_induce, reason = p._should_enter_induction(stalled=True, won=False)

        assert should_induce is False
        assert reason is None

    def test_renewed_stall_never_fires_when_env_explicitly_off(self, monkeypatch) -> None:
        monkeypatch.setenv("CARNOT_ARC_BOUNDED_REINDUCTION", "0")
        p = _policy(n_transitions=500, transitions_at_last_attempt=0, attempt_count=0)

        should_induce, reason = p._should_enter_induction(stalled=True, won=False)

        assert should_induce is False


class TestGatedOnBehaviour:
    def test_fires_once_enough_new_transitions_have_accumulated(self, monkeypatch) -> None:
        monkeypatch.setenv("CARNOT_ARC_BOUNDED_REINDUCTION", "1")
        p = _policy(n_transitions=200, transitions_at_last_attempt=0, attempt_count=0)

        should_induce, reason = p._should_enter_induction(stalled=True, won=False)

        assert should_induce is True
        assert reason == "renewed_stall_reinduction"

    def test_does_not_fire_below_the_new_transition_threshold(self, monkeypatch) -> None:
        monkeypatch.setenv("CARNOT_ARC_BOUNDED_REINDUCTION", "1")
        p = _policy(n_transitions=199, transitions_at_last_attempt=0, attempt_count=0)

        should_induce, reason = p._should_enter_induction(stalled=True, won=False)

        assert should_induce is False
        assert reason is None

    def test_threshold_is_measured_since_the_LAST_attempt_not_since_the_level_start(
        self, monkeypatch
    ) -> None:
        """200 total transitions but only 50 since the last attempt must not fire -- the
        whole point is fresh evidence, not a re-ask on the same prompt."""
        monkeypatch.setenv("CARNOT_ARC_BOUNDED_REINDUCTION", "1")
        p = _policy(n_transitions=200, transitions_at_last_attempt=150, attempt_count=0)

        should_induce, reason = p._should_enter_induction(stalled=True, won=False)

        assert should_induce is False

    def test_does_not_fire_once_the_attempt_cap_is_reached(self, monkeypatch) -> None:
        monkeypatch.setenv("CARNOT_ARC_BOUNDED_REINDUCTION", "1")
        p = _policy(
            n_transitions=1000,
            transitions_at_last_attempt=0,
            attempt_count=agent._REINDUCTION_MAX_ATTEMPTS,
        )

        should_induce, reason = p._should_enter_induction(stalled=True, won=False)

        assert should_induce is False, (
            "a pathologically-stalled game must not spend its whole budget re-asking a "
            "generator that keeps refusing"
        )

    def test_does_not_fire_when_not_stalled_or_when_won(self, monkeypatch) -> None:
        monkeypatch.setenv("CARNOT_ARC_BOUNDED_REINDUCTION", "1")
        p1 = _policy(n_transitions=1000, transitions_at_last_attempt=0, attempt_count=0)
        assert p1._should_enter_induction(stalled=False, won=False) == (False, None)

        p2 = _policy(n_transitions=1000, transitions_at_last_attempt=0, attempt_count=0)
        assert p2._should_enter_induction(stalled=True, won=True) == (False, None)

    def test_does_not_fire_when_not_currently_latched(self, monkeypatch) -> None:
        """If self.induced is already False, the ordinary "stall" branch already handles
        it -- this lever exists ONLY for the latched case."""
        monkeypatch.setenv("CARNOT_ARC_BOUNDED_REINDUCTION", "1")
        p = _policy(
            n_transitions=1000, transitions_at_last_attempt=0, attempt_count=0, induced=False
        )

        should_induce, reason = p._should_enter_induction(stalled=True, won=False)

        assert reason == "stall", "the ordinary stall path must win when not yet latched"


class TestLivenessWitnessDiagnostic:
    def test_reports_latched_state_and_wasted_transitions(self, monkeypatch) -> None:
        monkeypatch.delenv("CARNOT_ARC_DISABLE_INDUCTION", raising=False)
        monkeypatch.delenv("CARNOT_ARC_BOUNDED_REINDUCTION", raising=False)
        p = _policy(n_transitions=350, transitions_at_last_attempt=100, attempt_count=1)
        p.induction_attempts = []

        row = p.generator_liveness_witness()

        assert row["induction_tier_latched_off"] is True
        assert row["induction_attempt_count"] == 1
        assert row["transitions_since_last_induction_attempt"] == 250
        assert row["bounded_reinduction_enabled"] is False

    def test_reports_bounded_reinduction_enabled_when_the_env_flag_is_set(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("CARNOT_ARC_BOUNDED_REINDUCTION", "1")
        p = _policy(n_transitions=0, transitions_at_last_attempt=0, attempt_count=0)
        p.induction_attempts = []

        row = p.generator_liveness_witness()

        assert row["bounded_reinduction_enabled"] is True

    def test_reports_not_latched_when_induced_is_false(self, monkeypatch) -> None:
        p = _policy(n_transitions=10, transitions_at_last_attempt=0, attempt_count=0, induced=False)
        p.induction_attempts = []

        row = p.generator_liveness_witness()

        assert row["induction_tier_latched_off"] is False


class TestPerLevelReset:
    def test_level_boundary_resets_attempt_count_and_transition_snapshot(self, monkeypatch) -> None:
        from types import SimpleNamespace

        monkeypatch.setattr(agent, "_level_of", lambda frame: int(frame.levels_completed))
        p = E3AgentPolicy("lp85", proposer=object(), target_levels=3, value_head=None)
        p.transitions = [object(), object(), object()]
        p._induction_attempt_count = 3
        p._transitions_at_last_induction_attempt = 1

        p._observe_level_boundary(SimpleNamespace(levels_completed=0), frames_seen=1)
        p._observe_level_boundary(SimpleNamespace(levels_completed=1), frames_seen=2)

        assert p._induction_attempt_count == 0, (
            "the K-attempts cap is PER LEVEL -- a level-up must reset it, or a game with "
            "many short levels would exhaust the whole-episode cap on level 1"
        )
        assert p._transitions_at_last_induction_attempt == len(p.transitions)


class TestCallSiteWiringBySourceInspection:
    """The decision logic (`_should_enter_induction`) is unit-tested thoroughly above; these
    confirm the CALLER actually acts on a `renewed_stall_reinduction` verdict, by checking the
    reset and the counter-update sit at the right place in the real source -- a full
    `next_move()` integration test would need to mock frames/explorer/proposer for no
    additional coverage of the risk that matters here (a one-line reset, not new decision
    logic)."""

    def test_renewed_stall_reinduction_resets_the_latch_at_its_call_site(self) -> None:
        import inspect

        src = inspect.getsource(E3AgentPolicy._next_move_routed)
        idx = src.index('reason == "renewed_stall_reinduction"')
        # The reset must be the next statement, allowing room for the explanatory comment
        # above the actual `self.induced = False` line.
        tail = src[idx : idx + 500]
        assert "self.induced = False" in tail, (
            "a renewed_stall_reinduction verdict must reset the latch, or the "
            "'phase == \"induce\" and not self.induced' gate silently no-ops past it"
        )

    def test_induction_firing_updates_the_attempt_count_and_transition_snapshot(self) -> None:
        import inspect

        src = inspect.getsource(E3AgentPolicy._next_move_routed)
        idx = src.index("self.induced = True")
        tail = src[idx : idx + 250]
        assert "self._induction_attempt_count += 1" in tail
        assert "self._transitions_at_last_induction_attempt = len(self.transitions)" in tail

"""Tests for the LIVE-agent wiring of the REQ-CAPSTONE-4582 feature router.

Why this file exists: `arc_solve_learning.extract_early_play_signature` /
`classify_early_play_mechanic` / `route_feature_approach` were built and unit-tested
(`test_experiment_4582_feature_router_transfer.py`), and even imported by
`arc_competition_agent.py`, but the scored `E3AgentPolicy` never actually fed them live
transitions -- `_feature_router_payload` always returned `enabled: False,
reason: "early_play_signature_not_supplied"` in practice. These tests pin the fix: a
purely behavioral (no game-identity, no win-check, no LLM) mechanic classification built
from `E3AgentPolicy`'s OWN collected transitions, fired once per game, that only changes
live search behavior above a confidence floor (kept conservative given
`results/experiment_4582_feature_router_transfer.json`'s honest null on the closely
related full-solver-swap version of this idea).

Spec refs: REQ-CAPSTONE-4582, SCENARIO-CAPSTONE-4582.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from carnot.agentic import arc_competition_agent as agent
from carnot.agentic.arc_competition_agent import E3AgentPolicy


@dataclass
class _FakeTransition:
    grid: Any
    action: int
    data: Any
    next_grid: Any
    level_before: int = 0
    level_after: int = 0


def _click_transitions(n: int) -> list[_FakeTransition]:
    """n click transitions, each changing one cell -- classifies as click_graph."""

    out = []
    for i in range(n):
        before = ((0, 0), (0, 0))
        after = ((i % 9 + 1, 0), (0, 0))
        out.append(_FakeTransition(before, 6, {"x": i, "y": 0}, after))
    return out


def test_req_capstone_4582_early_play_rows_reads_transition_fields() -> None:
    """_early_play_rows converts Transition-shaped objects using only the fields it needs,
    never a game-identity lookup."""

    transitions = _click_transitions(3)
    rows = agent._early_play_rows(transitions, k=8)

    assert len(rows) == 3
    assert rows[0]["action_id"] == 6
    assert rows[0]["data"] == {"x": 0, "y": 0}
    assert rows[0]["before"] == ((0, 0), (0, 0))
    assert rows[0]["after"] == ((1, 0), (0, 0))


def test_req_capstone_4582_early_play_rows_caps_at_k() -> None:
    transitions = _click_transitions(20)
    rows = agent._early_play_rows(transitions, k=8)
    assert len(rows) == 8


def test_scenario_capstone_4582_route_from_transitions_fires_once_after_k() -> None:
    """SCENARIO-CAPSTONE-4582: the live check is a no-op before k transitions, fires exactly
    once at k, and is idempotent afterward -- it never re-fires or re-queries every step."""

    policy = E3AgentPolicy("livewiretest", proposer=None, value_head=lambda _f: 0.0)
    assert policy.feature_router is None
    assert policy._feature_router_checked is False

    # Below the k=8 threshold: no-op every time.
    for i in range(agent._FEATURE_ROUTER_EARLY_PLAY_K - 1):
        policy.transitions.append(_click_transitions(1)[0])
        policy._maybe_route_from_transitions()
        assert policy._feature_router_checked is False
        assert policy.feature_router is None

    # Crossing k: fires exactly once.
    policy.transitions.append(_click_transitions(1)[0])
    policy._maybe_route_from_transitions()
    assert policy._feature_router_checked is True
    first_result = policy.feature_router
    assert isinstance(first_result, dict)
    assert first_result.get("enabled") is True
    assert first_result.get("mechanic_class") == "click_graph"

    # Idempotent: calling again (e.g. next action) does not re-query or change the result.
    policy.transitions.append(_click_transitions(1)[0])
    policy._maybe_route_from_transitions()
    assert policy.feature_router is first_result


def test_req_capstone_4582_never_uses_game_identity_or_win_check() -> None:
    """The classification is purely behavioral: an UNKNOWN, made-up game id (guaranteed not
    in the public survey / registry) still produces a real classification, because
    extract_early_play_signature/classify_early_play_mechanic never consult game identity or
    frame.levels_completed."""

    policy = E3AgentPolicy(
        "totally-unseen-hidden-game-id-9999", proposer=None, value_head=lambda _f: 0.0
    )
    for t in _click_transitions(agent._FEATURE_ROUTER_EARLY_PLAY_K):
        policy.transitions.append(t)
    policy._maybe_route_from_transitions()

    assert policy.feature_router is not None
    assert policy.feature_router["mechanic_class"] == "click_graph"
    assert policy.feature_router["signature"]["llm_used"] is False


def test_scenario_capstone_4582_low_confidence_does_not_change_strategy(monkeypatch) -> None:
    """SCENARIO-CAPSTONE-4582: a classified-but-low-confidence route is stored for
    observability but must NOT change live search behavior (the confidence gate)."""

    def _fake_recommend(game_id, *, mechanic=None, early_play_signature=None):
        return {
            "strategy": {"name": "graph_explore", "uses_goal_distance_heuristic": True},
            "feature_router": {
                "enabled": True,
                "mechanic_class": "avatar_navigation",
                "approach": "goal_distance_astar",
                "confidence": agent._FEATURE_ROUTER_MIN_CONFIDENCE - 0.01,
            },
        }

    monkeypatch.setattr(agent, "_recommend_live_approach", _fake_recommend)
    policy = E3AgentPolicy("lowconftest", proposer=None, value_head=lambda _f: 0.0)
    original_route = dict(policy.strategy_route)
    for t in _click_transitions(agent._FEATURE_ROUTER_EARLY_PLAY_K):
        policy.transitions.append(t)

    policy._maybe_route_from_transitions()

    assert policy.feature_router is not None  # observability still populated
    assert policy.strategy_route == original_route  # but behavior unchanged


def test_scenario_capstone_4582_high_confidence_avatar_navigation_biases_goal_distance(
    monkeypatch,
) -> None:
    """SCENARIO-CAPSTONE-4582: a high-confidence avatar_navigation classification (whose
    routed approach is goal_distance_astar) DOES flip uses_goal_distance_heuristic and
    re-derives the explore budget -- the one concrete live prioritization effect."""

    def _fake_recommend(game_id, *, mechanic=None, early_play_signature=None):
        return {
            "strategy": {"name": "graph_explore", "uses_goal_distance_heuristic": False},
            "feature_router": {
                "enabled": True,
                "mechanic_class": "avatar_navigation",
                "approach": "goal_distance_astar",
                "confidence": 0.9,
            },
        }

    monkeypatch.setattr(agent, "_recommend_live_approach", _fake_recommend)
    policy = E3AgentPolicy("highconftest", proposer=None, value_head=lambda _f: 0.0)
    policy.strategy_route = {"name": "graph_explore", "uses_goal_distance_heuristic": False}
    for t in _click_transitions(agent._FEATURE_ROUTER_EARLY_PLAY_K):
        policy.transitions.append(t)

    policy._maybe_route_from_transitions()

    assert policy.strategy_route.get("uses_goal_distance_heuristic") is True
    assert policy.strategy_route.get("feature_router_mechanic_class") == "avatar_navigation"
    assert policy.strategy_route.get("feature_router_approach") == "goal_distance_astar"


def test_req_capstone_4582_recommend_live_approach_threads_early_play_signature(
    monkeypatch,
) -> None:
    """_recommend_live_approach forwards early_play_signature through to
    arc_solve_learning.recommend_approach (not silently dropped)."""

    captured: dict[str, Any] = {}

    def _fake_recommend_approach(game_id, *, mechanic=None, early_play_signature=None):
        captured["early_play_signature"] = early_play_signature
        return {"strategy": {"name": "graph_explore"}}

    monkeypatch.setattr(agent.arc_solve_learning, "recommend_approach", _fake_recommend_approach)
    rows = [{"action_id": 6, "data": {"x": 1, "y": 1}, "before": (), "after": ()}]

    agent._recommend_live_approach("anygame", early_play_signature=rows)

    assert captured["early_play_signature"] == rows


# ---------------------------------------------------------------------------------------------
# RunLocalMechanicLedger (scoped: docs/research-notes/arc-agi3-run-local-cross-game-adaptation-
# scope-2026-07-12.md; operator ruling recorded there clears the "is this memorization" question).
# ---------------------------------------------------------------------------------------------


def test_scenario_capstone_4582_run_local_disabled_by_default(monkeypatch) -> None:
    """SCENARIO-CAPSTONE-4582-RUN-LOCAL: the kill switch defaults OFF -- no env var means
    zero effect, regardless of what's already in the ledger."""

    monkeypatch.delenv("CARNOT_ARC_RUN_LOCAL_ADAPTATION", raising=False)
    assert agent._run_local_adaptation_enabled() is False

    agent._RUN_LOCAL_LEDGER.update("avatar_navigation", 999999, "goal_distance_astar", 1.0)
    assert agent._run_local_confidence_bonus("avatar_navigation", "goal_distance_astar") == 0.0


def test_scenario_capstone_4582_run_local_opt_in_flag(monkeypatch) -> None:
    monkeypatch.delenv("CARNOT_ARC_RUN_LOCAL_ADAPTATION", raising=False)
    assert agent._run_local_adaptation_enabled() is False
    monkeypatch.setenv("CARNOT_ARC_RUN_LOCAL_ADAPTATION", "1")
    assert agent._run_local_adaptation_enabled() is True


def test_req_capstone_4582_run_local_ledger_thread_safety() -> None:
    """REQ-CAPSTONE-4582-RUN-LOCAL: Swarm.main() runs every game's agent concurrently on its
    own thread within one process (confirmed against the real scored submission path in the
    scope doc) -- the ledger MUST NOT lose updates under real concurrent writes."""

    import threading

    ledger = agent.RunLocalMechanicLedger()
    n_threads = 50
    writes_per_thread = 20

    def worker(tid: int) -> None:
        for i in range(writes_per_thread):
            ledger.update("avatar_navigation", tid, "goal_distance_astar", float(i))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Each thread writes under its OWN stable key -- exactly n_threads distinct entries expected;
    # any lost update would show up as a lower count.
    assert ledger.sample_count("avatar_navigation") == n_threads


def test_req_capstone_4582_run_local_ledger_overwrites_same_game_key() -> None:
    """A single game's repeated updates (as its own play progresses) overwrite its own prior
    entry rather than accumulating duplicates -- sample_count tracks DISTINCT games, not calls."""

    ledger = agent.RunLocalMechanicLedger()
    for score in (0.0, 0.0, 1.0):
        ledger.update("avatar_navigation", 42, "goal_distance_astar", score)

    assert ledger.sample_count("avatar_navigation") == 1
    assert ledger.mean_outcome("avatar_navigation", "goal_distance_astar") == 1.0


def test_req_capstone_4582_run_local_ledger_empty_and_unknown_mechanic_class() -> None:
    ledger = agent.RunLocalMechanicLedger()
    assert ledger.sample_count("avatar_navigation") == 0
    assert ledger.mean_outcome("avatar_navigation", "goal_distance_astar") is None

    # unknown / empty mechanic_class or approach must be silently ignored, not recorded.
    ledger.update("unknown", 1, "goal_distance_astar", 1.0)
    ledger.update("", 2, "goal_distance_astar", 1.0)
    ledger.update("avatar_navigation", 3, "", 1.0)
    assert ledger.sample_count("unknown") == 0
    assert ledger.sample_count("avatar_navigation") == 0


def test_req_capstone_4582_run_local_bonus_requires_min_samples(monkeypatch) -> None:
    """REQ-CAPSTONE-4582-RUN-LOCAL: the bonus is exactly 0.0 (a WEAK signal, deliberately
    inert) until _RUN_LOCAL_MIN_SAMPLES distinct games of this mechanic class have
    contributed, even when adaptation is enabled."""

    monkeypatch.setenv("CARNOT_ARC_RUN_LOCAL_ADAPTATION", "1")
    ledger = agent.RunLocalMechanicLedger()
    monkeypatch.setattr(agent, "_RUN_LOCAL_LEDGER", ledger)

    for i in range(agent._RUN_LOCAL_MIN_SAMPLES - 1):
        ledger.update("avatar_navigation", i, "goal_distance_astar", 1.0)
        assert agent._run_local_confidence_bonus("avatar_navigation", "goal_distance_astar") == 0.0

    # Crossing the threshold: bonus becomes non-zero (positive outcomes -> positive bonus).
    ledger.update("avatar_navigation", agent._RUN_LOCAL_MIN_SAMPLES, "goal_distance_astar", 1.0)
    bonus = agent._run_local_confidence_bonus("avatar_navigation", "goal_distance_astar")
    assert bonus > 0.0
    assert bonus <= agent._RUN_LOCAL_MAX_CONFIDENCE_BONUS


def test_req_capstone_4582_run_local_bonus_bounded_and_disabled_games_contribute_nothing(
    monkeypatch,
) -> None:
    """A mechanic class with only NEGATIVE (0.0) recorded outcomes contributes a 0.0 bonus
    (never negative -- the bonus only ever helps clear the gate, never actively penalizes),
    and the bonus never exceeds _RUN_LOCAL_MAX_CONFIDENCE_BONUS even with a perfect record."""

    monkeypatch.setenv("CARNOT_ARC_RUN_LOCAL_ADAPTATION", "1")
    ledger = agent.RunLocalMechanicLedger()
    monkeypatch.setattr(agent, "_RUN_LOCAL_LEDGER", ledger)

    for i in range(agent._RUN_LOCAL_MIN_SAMPLES + 2):
        ledger.update("config_toggle", i, "diversity_graph_explore", 0.0)
    assert agent._run_local_confidence_bonus("config_toggle", "diversity_graph_explore") == 0.0

    for i in range(agent._RUN_LOCAL_MIN_SAMPLES + 2):
        ledger.update("keyboard_graph", i, "systematic_bfs", 1.0)
    bonus = agent._run_local_confidence_bonus("keyboard_graph", "systematic_bfs")
    assert bonus == agent._RUN_LOCAL_MAX_CONFIDENCE_BONUS


def test_req_capstone_4582_run_local_outcome_proxy_detects_level_progress() -> None:
    no_progress = [
        _FakeTransition((), 1, None, (), level_before=3, level_after=3) for _ in range(4)
    ]
    assert agent._run_local_outcome_proxy(no_progress) == 0.0

    with_progress = no_progress + [_FakeTransition((), 1, None, (), level_before=3, level_after=4)]
    assert agent._run_local_outcome_proxy(with_progress) == 1.0


def test_scenario_capstone_4582_route_from_transitions_records_to_ledger_when_enabled(
    monkeypatch,
) -> None:
    """SCENARIO-CAPSTONE-4582-RUN-LOCAL: _maybe_route_from_transitions contributes to the
    shared ledger when the opt-in flag is set, using this game's own id() as a stable key --
    and does NOT touch the ledger at all when the flag is unset (true kill-switch semantics)."""

    def _fake_recommend(game_id, *, mechanic=None, early_play_signature=None):
        return {
            "strategy": {"name": "graph_explore", "uses_goal_distance_heuristic": False},
            "feature_router": {
                "enabled": True,
                "mechanic_class": "avatar_navigation",
                "approach": "goal_distance_astar",
                "confidence": 0.1,
            },
        }

    monkeypatch.setattr(agent, "_recommend_live_approach", _fake_recommend)
    ledger = agent.RunLocalMechanicLedger()
    monkeypatch.setattr(agent, "_RUN_LOCAL_LEDGER", ledger)

    # Flag OFF (default): no ledger interaction at all.
    monkeypatch.delenv("CARNOT_ARC_RUN_LOCAL_ADAPTATION", raising=False)
    policy_off = E3AgentPolicy("offtest", proposer=None, value_head=lambda _f: 0.0)
    for t in _click_transitions(agent._FEATURE_ROUTER_EARLY_PLAY_K):
        policy_off.transitions.append(t)
    policy_off._maybe_route_from_transitions()
    assert ledger.sample_count("avatar_navigation") == 0

    # Flag ON: this game's choice is recorded under its own id().
    monkeypatch.setenv("CARNOT_ARC_RUN_LOCAL_ADAPTATION", "1")
    policy_on = E3AgentPolicy("ontest", proposer=None, value_head=lambda _f: 0.0)
    for t in _click_transitions(agent._FEATURE_ROUTER_EARLY_PLAY_K):
        policy_on.transitions.append(t)
    policy_on._maybe_route_from_transitions()
    assert ledger.sample_count("avatar_navigation") == 1

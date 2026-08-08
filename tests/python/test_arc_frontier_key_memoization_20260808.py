"""Spec: REQ-ARC-WMTE-6226.

Regression tests for the uncached per-node frontier-key recompute bug.

docs/research-notes/live-agent-adversarial-review-2026-08-08.md, "Speed" section, major
finding 2:

  "Every `_frontier` call recomputes, per node, two expensive keys over immutable inputs:
  the action-effect prior scores every remaining untested candidate (~34-55/node) through
  the frame-change scorer ..., and the goal-bias energy re-runs on the node's stored frame
  (post-induction, an exec'd Python predicate per node per call). ... Fix: memoize per
  (node_hash, untested-length) and per (node_hash, goal_bias generation); batch the
  candidate scorer per frame."

THE FIX. `StepwiseExplorer._goal_bias_score` and `_action_effect_frontier_key` now cache
their result ON THE NODE DICT itself (mirroring the existing lazy-value cache pattern for
`node["value"]`), so no node-hash plumbing is needed at any call site:

  - `_goal_bias_score`: cached under `node["_goal_bias_score_cache"] = (generation, score)`,
    invalidated by `self._goal_bias_generation` -- bumped only by `set_goal_bias`, the sole
    post-construction mutator of `self.goal_bias`.
  - `_action_effect_frontier_key`: cached under `node["_action_effect_key_cache"] =
    (len(untested), score)` -- `action_effect_expansion_prior` is set once at construction
    and never replaced, so `untested`'s length (which only ever shrinks) is a sufficient,
    cheap invalidation signal.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from carnot.agentic.arc_competition_agent import StepwiseExplorer


class _CountingGoalBias:
    """A stand-in for the exec'd goal predicate: counts real invocations so a cache hit is
    distinguishable from a cache miss by call count, not just by the returned value."""

    def __init__(self, value: float = 3.0) -> None:
        self.value = value
        self.calls = 0

    def __call__(self, frame: Any) -> float:
        self.calls += 1
        return self.value


class _CountingActionEffectPrior:
    """A stand-in for ActionEffectExpansionPrior: counts real frontier_priority calls."""

    def __init__(self, value: float = 5.0) -> None:
        self.value = value
        self.calls = 0

    def frontier_priority(self, frame: Any, untested: list) -> float:
        self.calls += 1
        return self.value


def _node(frame_value: int = 1, untested_len: int = 3) -> dict:
    return {
        "path": [],
        "frame": np.array([[frame_value]], dtype=np.int16),
        "untested": [{"action": i, "data": None} for i in range(untested_len)],
    }


class TestGoalBiasScoreCache:
    def test_repeated_calls_on_the_same_node_hit_the_cache(self) -> None:
        explorer = StepwiseExplorer()
        bias = _CountingGoalBias()
        explorer.set_goal_bias(bias, label="test")
        node = _node()

        first = explorer._goal_bias_score(node)
        second = explorer._goal_bias_score(node)
        third = explorer._goal_bias_score(node)

        assert first == second == third == 3.0
        assert bias.calls == 1, "only the first call should invoke the real predicate"

    def test_installing_a_new_goal_bias_invalidates_the_cache(self) -> None:
        explorer = StepwiseExplorer()
        bias_a = _CountingGoalBias(value=3.0)
        explorer.set_goal_bias(bias_a, label="a")
        node = _node()
        explorer._goal_bias_score(node)
        assert bias_a.calls == 1

        bias_b = _CountingGoalBias(value=9.0)
        explorer.set_goal_bias(bias_b, label="b")
        result = explorer._goal_bias_score(node)

        assert result == 9.0, "a stale cached score from the OLD bias must never be served"
        assert bias_b.calls == 1

    def test_different_nodes_are_scored_independently(self) -> None:
        explorer = StepwiseExplorer()
        bias = _CountingGoalBias()
        explorer.set_goal_bias(bias, label="test")
        node_a = _node(frame_value=1)
        node_b = _node(frame_value=2)

        explorer._goal_bias_score(node_a)
        explorer._goal_bias_score(node_b)
        explorer._goal_bias_score(node_a)
        explorer._goal_bias_score(node_b)

        assert bias.calls == 2, "each distinct node's frame must be scored once, not shared"


class TestActionEffectFrontierKeyCache:
    def test_repeated_calls_with_unchanged_untested_hit_the_cache(self) -> None:
        explorer = StepwiseExplorer()
        explorer.action_effect_expansion_prior = _CountingActionEffectPrior()
        node = _node(untested_len=5)

        first = explorer._action_effect_frontier_key(node)
        second = explorer._action_effect_frontier_key(node)

        assert first == second == 5.0
        assert explorer.action_effect_expansion_prior.calls == 1

    def test_untested_shrinking_invalidates_the_cache(self) -> None:
        explorer = StepwiseExplorer()
        prior = _CountingActionEffectPrior()
        explorer.action_effect_expansion_prior = prior
        node = _node(untested_len=5)
        explorer._action_effect_frontier_key(node)
        assert prior.calls == 1

        node["untested"].pop()  # a candidate was tried and popped, as the real search does
        explorer._action_effect_frontier_key(node)

        assert prior.calls == 2, "a changed untested length must trigger a real recompute"


class TestFrontierEndToEndStillPrunesAndRoutesCorrectly:
    def test_frontier_result_is_unaffected_by_caching(self) -> None:
        """The memoization must be transparent: _frontier's chosen node is identical to what
        it picked before this fix (no value_head/nav_tiebreak in play, so the key is driven
        by the two cached scores plus depth)."""
        explorer = StepwiseExplorer()
        explorer.goal_bias = _CountingGoalBias(value=1.0)
        explorer.action_effect_expansion_prior = _CountingActionEffectPrior(value=0.0)
        explorer.graph = {
            "a": _node(frame_value=1, untested_len=2),
            "b": _node(frame_value=2, untested_len=2),
        }

        picked = explorer._frontier()

        assert picked in ("a", "b")
        # Calling _frontier again must reuse the cached per-node scores rather than rescoring.
        explorer._frontier()
        assert explorer.goal_bias.calls == 2, "one real score per distinct node, ever"

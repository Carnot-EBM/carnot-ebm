"""Regression tests for the discriminative-frontier-prune never-empty-floor bug.

Spec: REQ-ARC-WMTE-6225.

docs/research-notes/live-agent-adversarial-review-2026-08-08.md, "Correctness" section,
major finding 3:

  "The online-discriminator frontier prune has no never-empty floor, and `explored_out` is
  a one-way latch. A logistic discriminator fittable from as few as 6 samples can
  transiently score every open node below the 0.12 threshold; `_frontier` then returns
  None, `explored_out` latches True (assigned only at init and there), and the game ends
  with most of the 2000-action budget unspent, reading as a legitimate "fully explored"
  null. The codebase's own hazard pruner documents and guards exactly this class
  ("NEVER-EMPTY GUARD, load-bearing"); this older prune path lacks it. Fix: retain the
  single highest-scoring node when the prune would empty the frontier, and count the event
  in diagnostics."

THE FIX. `StepwiseExplorer._frontier` now tracks the highest-`on_path_proba` node the
discriminative prune would otherwise drop. If pruning empties `eligible` entirely, that one
node is un-pruned and used instead of returning None -- mirroring the existing hazard-pruner
guard in this same file. The rescue increments a new `_disc_frontier_never_empty_rescues`
counter, surfaced in `online_discriminator_diagnostics()["never_empty_rescues"]`.

These tests reuse the fixture pattern from
test_experiment_4477_per_game_online_discriminative.py::
test_scenario_phase4_4477_stepwise_collects_negatives_and_prunes_frontier -- fit a real
DiscriminativeVerifier via two `_ingest()` calls (an alive frame, then a GAME_OVER frame),
then hand-build `explorer.graph` to exercise each scenario.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from carnot.agentic.arc_competition_agent import StepwiseExplorer


class _Frame:
    def __init__(self, value: int, *, state: str = "", actions: list[int] | None = None) -> None:
        self.frame = np.array([[value]], dtype=np.int16)
        self.state = state
        self.available_actions = actions if actions is not None else [1]
        self.levels_completed = 0


def _feature(frame: Any) -> list[float]:
    return [float(np.asarray(frame.frame)[0, 0])]


def _fitted_explorer() -> StepwiseExplorer:
    """A StepwiseExplorer with a real, fitted online_discriminator -- same recipe as the
    exp4477 test this file's fixtures are borrowed from."""
    explorer = StepwiseExplorer(
        online_discriminative=True,
        discriminative_featurizer=_feature,
        discriminative_min_positives=1,
        discriminative_min_negatives=1,
        discriminative_fit_iters=300,
        discriminative_prune_threshold=0.55,
    )
    explorer._ingest(_Frame(0))
    explorer.awaiting = {"origin": explorer.cur, "action": 1, "data": None}
    explorer._ingest(_Frame(10, state="GAME_OVER"))
    assert explorer.online_discriminator is not None, "sanity: the discriminator must be fitted"
    return explorer


def _bad_node(action_id: int) -> dict:
    """A node whose discriminative_features score BELOW the 0.55 threshold (see the exp4477
    fixture: features near 10.0 score low, features near 0.0 score high)."""
    return {
        "path": [{"action": action_id, "data": None}],
        "untested": [{"action": action_id + 1, "data": None}],
        "value": 0.0,
        "discriminative_features": [10.0],
    }


class TestAllNodesPrunedIsRescuedNotEmpty:
    def test_frontier_returns_the_rescued_node_instead_of_none(self) -> None:
        explorer = _fitted_explorer()
        explorer.graph = {"bad1": _bad_node(1), "bad2": _bad_node(3)}

        result = explorer._frontier()

        assert result in ("bad1", "bad2"), (
            "every node scored below threshold -- the never-empty guard must rescue one of "
            "them rather than returning None"
        )
        assert explorer.graph[result]["on_path_proba"] < 0.55, (
            "sanity: the rescued node really was below threshold, confirming the guard (not "
            "a fresh above-threshold score) is what let it through"
        )
        assert explorer.graph[result]["discriminative_pruned"] is False, (
            "the rescued node must be marked un-pruned -- it is now part of the eligible set"
        )

    def test_rescue_is_counted_in_diagnostics(self) -> None:
        explorer = _fitted_explorer()
        explorer.graph = {"bad1": _bad_node(1), "bad2": _bad_node(3)}
        before = explorer.online_discriminator_diagnostics()["never_empty_rescues"]

        explorer._frontier()

        after = explorer.online_discriminator_diagnostics()["never_empty_rescues"]
        assert after == before + 1

    def test_rescue_picks_the_highest_scoring_pruned_node(self) -> None:
        explorer = _fitted_explorer()
        # Two below-threshold nodes with DIFFERENT features -- 8.0 scores less confidently
        # low than 10.0 (closer to the 0.0 positive exemplar), so it must be the one rescued.
        explorer.graph = {
            "worse": {**_bad_node(1), "discriminative_features": [10.0]},
            "less_bad": {**_bad_node(3), "discriminative_features": [8.0]},
        }

        result = explorer._frontier()

        assert result == "less_bad", (
            "the guard must retain the SINGLE HIGHEST-scoring pruned node, not an arbitrary "
            "one -- picking the least-confidently-excluded candidate is the whole point of "
            "treating this as a rescue rather than a random fallback"
        )


class TestMixedFrontierIsUnaffected:
    def test_frontier_prunes_normally_when_a_good_node_survives(self) -> None:
        """The ordinary case (an existing exp4477 scenario) must be unchanged: when at least
        one node clears the threshold, the guard never fires and the bad node is genuinely
        pruned, not rescued."""
        explorer = _fitted_explorer()
        explorer.graph = {
            "bad": _bad_node(1),
            "good": {
                "path": [{"action": 3, "data": None}],
                "untested": [{"action": 4, "data": None}],
                "value": 0.0,
                "discriminative_features": [0.0],
            },
        }

        result = explorer._frontier()

        assert result == "good"
        assert explorer.graph["bad"]["discriminative_pruned"] is True
        assert explorer.online_discriminator_diagnostics()["never_empty_rescues"] == 0
        assert explorer.online_discriminator_diagnostics()["frontier_pruned"] >= 1


class TestGenuineExhaustionStillReturnsNone:
    def test_no_open_tier_nodes_at_all_returns_none_and_does_not_rescue(self) -> None:
        """A node with an EMPTY `untested` list never reaches the discriminative-prune check
        at all (`_node_has_open_tier` is False, so the loop `continue`s at the top) -- this is
        real exhaustion, not a discriminator artifact, and the guard must not manufacture a
        candidate for it."""
        explorer = _fitted_explorer()
        explorer.graph = {
            "done1": {**_bad_node(1), "untested": []},
            "done2": {**_bad_node(3), "untested": []},
        }

        result = explorer._frontier()

        assert result is None
        assert explorer.online_discriminator_diagnostics()["never_empty_rescues"] == 0

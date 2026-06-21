"""Live test-time-learning (TTT) world model for the ARC-AGI-3 competition agent.

OPERATOR REFRAME (2026-06-21): the competition scores the agent on HIDDEN games it never sees the
rules for -- only a frame->action INTERFACE, rate-limited to 10 REAL steps/sec (~288k steps / 8h over
~110 games). INTERNAL compute (model fit, simulated rollouts) is FREE. So the winning move is to LEARN
each game's transition model from a few rate-limited real probes, then PLAN for free inside the learned
model and COMMIT only the short winning path -- the score min(human/agent_actions, 1)^2 rewards few
committed real actions, and reaching more games raises the index-weighted mean.

WHY LEARN, NOT INDUCE: the frozen Qwen3.5-9B is too weak to INDUCE a coherent world model (exp4557 scored
0.0 held-out accuracy -- it failed its own single training transition). This module LEARNS the model from
played transitions using the EXISTING zero-LLM rule learner ``arc_world_model_dsl.ObjectDeltaModel`` --
which already fits ARC dynamics (translate / object-translate / recolor / click-recolor rules, greedily
composed; sub-second pure-numpy fit) from <100 transitions. That learner was ALREADY being fit every step
inside ``E3AgentPolicy`` (``_fit_dsl_model``) but used only for a diagnostic consistency-energy -- never as
the planning engine (the planner still calls the failing ``e3.load_engine``). This module packages the
learned model as the planning engine.

LAYERED ENGINE (all expose the ``engine(grid, action, data) -> grid`` contract that
``WorldModelVerifier`` and ``plan_in_model`` already expect, so it drops into the existing trust-gate +
BFS planner with NO new planner):

  L0  EXACT TABLE  -- (full grid bytes, action_key) -> observed next_grid. Zero training, O(1)/step,
                      reproduces every visited transition EXACTLY (so a corrected divergence is predicted
                      exactly on replan). Keyed on full ``grid.tobytes()`` (NOT lossy ascii) to avoid the
                      tens-digit colour-aliasing risk.
  L1  GENERALIZER  -- ObjectDeltaModel, for states never seen at L0.

ORACLE-DISTINCTNESS: this is a LEARNED dynamics model, not the executable oracle that defines correctness,
so any value claim carries ``verifier_is_oracle: False`` per the CLAUDE.md Circularity discipline.

STANDALONE: depends only on stable APIs (``ObjectDeltaModel``, ``WorldModelVerifier``, ``_action_key``-
equivalent). It does NOT edit the conductor-active core; the conductor wires it into
``E3AgentPolicy._induce_and_plan`` later (replacing ``e3.load_engine``), gated by the same
``WorldModelVerifier(...).score(engine).accuracy >= 0.5`` trust check and the same ``plan_in_model`` BFS.
Validate offline FIRST via ``scripts/arc_ttt_validate.py`` (the learning-curve dual-run gate).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from carnot.agentic.arc_world_model_dsl import ObjectDeltaModel


def action_key(action_id: int, data: Any) -> tuple:
    """Canonical action key: ``(6, x, y)`` for a click with coords, else ``(action_id,)``. Mirrors
    ``arc_competition_agent._action_key`` exactly so L0/L1 keying matches the live agent."""
    if int(action_id) == 6 and isinstance(data, dict) and "x" in data and "y" in data:
        return (6, int(data["x"]), int(data["y"]))
    return (int(action_id),)


class LiveTTTWorldModel:
    """A per-game world model learned from played transitions, exposing the engine + win predicate the
    existing ``plan_in_model`` BFS and ``WorldModelVerifier`` consume."""

    def __init__(self, game: str = "?", *, refit_every: int = 8, min_transitions: int = 16) -> None:
        self.game = game
        self.refit_every = int(refit_every)
        self.min_transitions = int(min_transitions)
        self._l0: dict[tuple, np.ndarray] = {}        # (grid.tobytes(), akey) -> next_grid (exact backbone)
        self._dsl_transitions: list[tuple] = []       # (s, akey, s2) feed for ObjectDeltaModel.fit
        self._win_states: set[bytes] = set()          # next_grid bytes observed at a level-up
        self._l1: Optional[ObjectDeltaModel] = None
        self._last_refit = -1
        self._refits = 0

    # --- learning from play (FREE compute; NOT rate-limited) ----------------------------------------
    def observe(self, grid: Any, action: int, data: Any, next_grid: Any,
                level_before: int = 0, level_after: int = 0) -> None:
        """Record one played transition into L0 + the L1 fit buffer; mark a win-state on a level-up."""
        g = np.asarray(grid)
        ng = np.asarray(next_grid)
        akey = action_key(action, data)
        self._l0[(g.tobytes(), akey)] = ng.copy()
        self._dsl_transitions.append((g, akey, ng))
        if int(level_after) > int(level_before):
            self._win_states.add(ng.tobytes())

    def observe_transition(self, t: Any) -> None:
        """Convenience for the ``Transition`` dataclass (grid, action, data, next_grid, level_*)."""
        self.observe(t.grid, t.action, t.data, t.next_grid,
                     getattr(t, "level_before", 0), getattr(t, "level_after", 0))

    def maybe_refit(self, step_idx: int) -> bool:
        """Refit L1 if ``refit_every`` steps elapsed and enough transitions accumulated. FREE compute."""
        if (step_idx - self._last_refit) < self.refit_every:
            return False
        if len(self._dsl_transitions) < self.min_transitions:
            return False
        self._l1 = ObjectDeltaModel(self.game).fit(self._dsl_transitions)
        self._last_refit = step_idx
        self._refits += 1
        return True

    def fit_now(self) -> "LiveTTTWorldModel":
        """Force an immediate L1 fit on all accumulated transitions (offline-harness convenience)."""
        if self._dsl_transitions:
            self._l1 = ObjectDeltaModel(self.game).fit(self._dsl_transitions)
            self._refits += 1
        return self

    # --- the engine seam: engine(grid, action, data) -> grid ----------------------------------------
    def engine(self, grid: np.ndarray, action: int, data: Optional[dict]) -> np.ndarray:
        """The pure transition model ``WorldModelVerifier`` / ``plan_in_model`` call. L0 exact hit ->
        the stored next_grid; miss -> L1 ObjectDeltaModel; no model yet -> identity (planning then
        simply finds no win, never crashes)."""
        g = np.asarray(grid)
        hit = self._l0.get((g.tobytes(), action_key(action, data)))
        if hit is not None:
            return hit
        if self._l1 is not None:
            return self._l1.predict(g, action_key(action, data))
        return g

    def is_level_complete(self, grid: np.ndarray) -> bool:
        """True if ``grid`` matches an observed win-state (a grid reached at a level-up). The BFS in
        ``plan_in_model`` halts here. (L2 value-epsilon goal is a follow-up; this is the MVP predicate.)"""
        return np.asarray(grid).tobytes() in self._win_states

    # --- trust gate (oracle-distinct held-out verification) -----------------------------------------
    def trust(self, held_out: list) -> float:
        """Held-out transition accuracy of the LEARNED engine -- the input to the existing
        ``WorldModelVerifier`` 0.5 trust gate. ``held_out`` is a list of ``Transition``. The model
        EARNS trust only by predicting transitions it was not fit to (the same bar the LLM engine failed
        at 0.0). Returns accuracy in [0, 1]."""
        from carnot.agentic.arc_executable_world_model import WorldModelVerifier
        if not held_out:
            return 0.0
        return float(WorldModelVerifier(held_out).score(self.engine).accuracy)

    # --- diagnostics --------------------------------------------------------------------------------
    def ttt_diagnostics(self) -> dict:
        return {
            "game": self.game,
            "l0_table_size": len(self._l0),
            "n_transitions": len(self._dsl_transitions),
            "l1_refits": self._refits,
            "n_win_states": len(self._win_states),
            "verifier_is_oracle": False,
        }

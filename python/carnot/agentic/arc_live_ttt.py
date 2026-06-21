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

N_COLORS = 16  # ARC grids use colours 0-15


class CNNDynamics:
    """A small fully-convolutional, ACTION-CONDITIONED grid->grid dynamics model, learned ONLINE from
    played transitions on CPU in seconds. Drop-in L1 replacement for ObjectDeltaModel (same fit/predict
    contract) for games whose mechanics the fixed rule class cannot express (the rule learner scored 0%
    on state-CHANGING transitions in arc_ttt_validate). A CNN can fit ARBITRARY LOCAL change rules
    (gravity, growth, click-spread, recolour-by-neighbour) from the ~120 transitions a probe gathers.

    Encoding (per transition): input = [16 colour one-hot] + [1 click-location mask at (y,x)] +
    [7 keyboard/action one-hot broadcast] = 24 channels at the grid's native H x W (fully-convolutional,
    so any grid size works). Target = the next grid's per-cell colour index. Trained with per-cell
    cross-entropy; predict = argmax per cell. Oracle-DISTINCT (a learned net, not the executable oracle).
    torch is imported LAZILY so the rule-only path needs no torch."""

    def __init__(self, game: str = "?", *, epochs: int = 400, lr: float = 5e-3, hidden: int = 48) -> None:
        self.game = game
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.hidden = int(hidden)
        self._net: Any = None
        self._shape: Optional[tuple] = None  # (H, W) the net was trained at

    @staticmethod
    def _click_xy(akey: tuple) -> Optional[tuple]:
        return (int(akey[1]), int(akey[2])) if len(akey) == 3 and int(akey[0]) == 6 else None

    def _encode(self, grid: np.ndarray, akey: tuple, torch: Any) -> Any:
        g = np.clip(np.asarray(grid).astype(np.int64), 0, N_COLORS - 1)
        h, w = g.shape
        chans = torch.zeros((N_COLORS + 1 + 7, h, w), dtype=torch.float32)
        # colour one-hot
        gt = torch.from_numpy(g)
        chans[:N_COLORS].scatter_(0, gt.unsqueeze(0), 1.0)
        xy = self._click_xy(akey)
        if xy is not None:
            x, y = xy
            if 0 <= y < h and 0 <= x < w:
                chans[N_COLORS, y, x] = 1.0           # click-location mask
        aid = int(akey[0])
        if 1 <= aid <= 7:
            chans[N_COLORS + aid] = 1.0               # action one-hot broadcast (index N_COLORS+1 .. +7)
        return chans

    def _build_net(self, torch: Any) -> Any:
        nn = torch.nn
        c_in = N_COLORS + 1 + 7
        return nn.Sequential(
            nn.Conv2d(c_in, self.hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(self.hidden, self.hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(self.hidden, N_COLORS, 3, padding=1),  # -> per-cell colour logits
        )

    def fit(self, transitions, *, epochs: Optional[int] = None, batch_size: int = 256,
            warm_state: Any = None) -> "CNNDynamics":
        """transitions: iterable of (s_grid, akey, s2_grid). Mini-batched CPU training. If warm_state is a
        net state_dict (from a pretrained PRIOR), the net is initialised from it before fine-tuning -- the
        cross-game mechanic prior the per-game live learner adapts from (fewer real probes to converge)."""
        import torch

        items = [(np.asarray(s), tuple(a), np.asarray(s2)) for s, a, s2 in transitions]
        items = [t for t in items if t[0].ndim == 2 and t[0].shape == t[2].shape]
        if not items:
            return self
        # all public ARC-AGI-3 grids are 64x64 -> a single net trains across games; keep the dominant shape
        from collections import Counter
        self._shape = Counter(t[0].shape for t in items).most_common(1)[0][0]
        items = [t for t in items if t[0].shape == self._shape]
        torch.manual_seed(0)
        net = self._build_net(torch)
        if warm_state is not None:
            net.load_state_dict(warm_state)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        n = len(items)
        net.train()
        for _ in range(int(epochs if epochs is not None else self.epochs)):
            perm = torch.randperm(n).tolist()
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                # encode PER BATCH (lazy) -- pre-stacking the whole corpus is ~5GB at 12k x 24x64x64.
                xb = torch.stack([self._encode(items[j][0], items[j][1], torch) for j in idx])
                yb = torch.stack([torch.from_numpy(np.clip(items[j][2].astype(np.int64), 0, N_COLORS - 1))
                                  for j in idx])
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                opt.step()
        net.eval()
        self._net = net
        return self

    def get_state(self) -> Any:
        """The net's state_dict (for saving a pretrained prior), or None if untrained."""
        return None if self._net is None else {k: v.clone() for k, v in self._net.state_dict().items()}

    def predict(self, s_grid, akey: tuple) -> np.ndarray:
        if self._net is None or np.asarray(s_grid).shape != self._shape:
            return np.asarray(s_grid)  # untrained / wrong shape -> identity (never crash the planner)
        import torch

        with torch.no_grad():
            logits = self._net(self._encode(s_grid, tuple(akey), torch).unsqueeze(0))  # [1,16,H,W]
            return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.asarray(s_grid).dtype)


def action_key(action_id: int, data: Any) -> tuple:
    """Canonical action key: ``(6, x, y)`` for a click with coords, else ``(action_id,)``. Mirrors
    ``arc_competition_agent._action_key`` exactly so L0/L1 keying matches the live agent."""
    if int(action_id) == 6 and isinstance(data, dict) and "x" in data and "y" in data:
        return (6, int(data["x"]), int(data["y"]))
    return (int(action_id),)


class LiveTTTWorldModel:
    """A per-game world model learned from played transitions, exposing the engine + win predicate the
    existing ``plan_in_model`` BFS and ``WorldModelVerifier`` consume."""

    def __init__(self, game: str = "?", *, refit_every: int = 8, min_transitions: int = 16,
                 dynamics_backend: str = "dsl", prior_state: Any = None) -> None:
        self.game = game
        self.refit_every = int(refit_every)
        self.min_transitions = int(min_transitions)
        # a pretrained cross-game CNN PRIOR state_dict (models/arc_dynamics_prior.pt); when set and the
        # backend is 'cnn', the per-game learner warm-starts from it -> fewer real probes to converge.
        self._prior_state = prior_state
        # 'dsl' = ObjectDeltaModel rule learner (fast, zero-train, but a fixed hypothesis class);
        # 'cnn' = CNNDynamics learned net (fits arbitrary local rules, the make-or-break test for games
        # the rule class can't express). Both expose .fit(transitions)/.predict(grid, akey).
        self.dynamics_backend = dynamics_backend
        self._l0: dict[tuple, np.ndarray] = {}        # (grid.tobytes(), akey) -> next_grid (exact backbone)
        self._dsl_transitions: list[tuple] = []       # (s, akey, s2) feed for the L1 learner
        self._win_states: set[bytes] = set()          # next_grid bytes observed at a level-up
        self._l1: Any = None
        self._last_refit = -1
        self._refits = 0

    def _new_l1(self) -> Any:
        return CNNDynamics(self.game) if self.dynamics_backend == "cnn" else ObjectDeltaModel(self.game)

    def _fit_l1(self, transitions):
        l1 = self._new_l1()
        if self.dynamics_backend == "cnn" and self._prior_state is not None:
            return l1.fit(transitions, warm_state=self._prior_state)
        return l1.fit(transitions)

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
        self._l1 = self._fit_l1(self._dsl_transitions)
        self._last_refit = step_idx
        self._refits += 1
        return True

    def fit_now(self) -> "LiveTTTWorldModel":
        """Force an immediate L1 fit on all accumulated transitions (offline-harness convenience)."""
        if self._dsl_transitions:
            self._l1 = self._fit_l1(self._dsl_transitions)
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

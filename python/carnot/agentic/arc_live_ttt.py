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

    _OBJ_CHANNELS = 4  # foreground mask + object-centroid-offset (dy, dx) + object-size

    def __init__(
        self,
        game: str = "?",
        *,
        epochs: int = 400,
        lr: float = 5e-3,
        hidden: int = 48,
        change_weight: float = 40.0,
        object_features: bool = False,
    ) -> None:
        self.game = game
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.hidden = int(hidden)
        # OBJECT-DELTA encoding (2026-06-21): augment the per-pixel input with connected-component object
        # structure so the local CNN sees whole OBJECTS, not just colours -- its per-cell residual then
        # moves/recolours objects COHERENTLY instead of independently (the per-pixel CNN's receptive field
        # can't span an object, capping cell-recall). Channels: foreground mask + centroid-offset dy/dx +
        # log-size. Needs its own prior (the input-channel count differs from the plain cnn backend).
        self.object_features = bool(object_features)
        # per-cell loss weight on CHANGED cells (target != KEEP). The KEEP class is ~4000:1 dominant at
        # 64x64, so unweighted cross-entropy collapses the net to "predict KEEP everywhere" = identity =
        # 0% on changing transitions (verified). Upweighting the change forces the net to model it.
        self.change_weight = float(change_weight)
        self._net: Any = None
        self._shape: Optional[tuple] = None  # (H, W) the net was trained at
        self._device = (
            "cpu"  # set to cuda at fit time when available (matches the 16GB CUDA eval GPU)
        )

    @staticmethod
    def _click_xy(akey: tuple) -> Optional[tuple]:
        return (int(akey[1]), int(akey[2])) if len(akey) == 3 and int(akey[0]) == 6 else None

    def _object_channels(self, grid: np.ndarray) -> np.ndarray:
        """[4, H, W] object-structure features: foreground mask + per-cell object-centroid offset (dy, dx,
        normalised) + log-size. Background = the most common colour; objects = connected components of the
        rest. Gives the CNN object-shape context its receptive field can't span."""
        from scipy import ndimage

        g = np.asarray(grid)
        h, w = g.shape
        vals, counts = np.unique(g, return_counts=True)
        bg = int(vals[counts.argmax()])
        fg = g != bg
        labels, n = ndimage.label(fg)
        out = np.zeros((self._OBJ_CHANNELS, h, w), dtype=np.float32)
        out[0] = fg.astype(np.float32)
        if n > 0:
            ys, xs = np.mgrid[0:h, 0:w]
            for oid in range(1, n + 1):
                m = labels == oid
                out[1][m] = (ys[m] - ys[m].mean()) / max(1, h)  # centroid offset dy
                out[2][m] = (xs[m] - xs[m].mean()) / max(1, w)  # centroid offset dx
                out[3][m] = np.log1p(int(m.sum())) / np.log1p(h * w)  # log object size
        return out

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
                chans[N_COLORS, y, x] = 1.0  # click-location mask
        aid = int(akey[0])
        if 1 <= aid <= 7:
            chans[N_COLORS + aid] = 1.0  # action one-hot broadcast (index N_COLORS+1 .. +7)
        if self.object_features:
            chans = torch.cat([chans, torch.from_numpy(self._object_channels(grid))], dim=0)
        return chans

    def _build_net(self, torch: Any) -> Any:
        nn = torch.nn
        c_in = N_COLORS + 1 + 7 + (self._OBJ_CHANNELS if self.object_features else 0)
        # RESIDUAL/KEEP head (2026-06-21): per cell, predict 1 + N_COLORS classes -- class 0 = KEEP (copy
        # input), classes 1..16 = SET-to-colour-(0..15). So the ~4095 unchanged cells of a 64x64 grid learn
        # the trivial KEEP majority class and are correct BY CONSTRUCTION at predict time; the net's
        # capacity + the loss focus only on the sparse CHANGE. (Full-grid colour prediction read 0/5 because
        # exact-match over 4096 absolute colours is unwinnable -- it wasted capacity re-copying the bg.)
        return nn.Sequential(
            nn.Conv2d(c_in, self.hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden, self.hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                self.hidden, 1 + N_COLORS, 3, padding=1
            ),  # -> per-cell {KEEP, set-colour-0..15}
        )

    @staticmethod
    def _residual_target(s: np.ndarray, s2: np.ndarray):
        """Per-cell class: 0 = KEEP (next == input), else colour+1 (1..16). Defined vs the INPUT grid."""
        s = np.asarray(s)
        s2 = np.clip(np.asarray(s2).astype(np.int64), 0, N_COLORS - 1)
        return np.where(s2 == s, 0, s2 + 1)

    def fit(
        self,
        transitions,
        *,
        epochs: Optional[int] = None,
        batch_size: int = 256,
        warm_state: Any = None,
    ) -> "CNNDynamics":
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
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        net = self._build_net(torch).to(self._device)
        if warm_state is not None:
            net.load_state_dict(warm_state)  # copies CPU prior weights onto the net's device
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        n = len(items)
        net.train()
        for _ in range(int(epochs if epochs is not None else self.epochs)):
            perm = torch.randperm(n).tolist()
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                # encode PER BATCH (lazy) -- pre-stacking the whole corpus is ~5GB at 12k x 24x64x64.
                xb = torch.stack([self._encode(items[j][0], items[j][1], torch) for j in idx]).to(
                    self._device
                )
                yb = torch.stack(
                    [torch.from_numpy(self._residual_target(items[j][0], items[j][2])) for j in idx]
                ).to(self._device)  # residual/KEEP target (0=keep,1..16=set)
                opt.zero_grad()
                ce = loss_fn(net(xb), yb)  # [B, H, W] per-cell loss
                w = torch.where(yb > 0, self.change_weight, 1.0)  # upweight CHANGED cells
                loss = (ce * w).sum() / w.sum()
                loss.backward()
                opt.step()
        net.eval()
        self._net = net
        return self

    def get_state(self) -> Any:
        """The net's state_dict (for saving a pretrained prior), or None if untrained."""
        return (
            None
            if self._net is None
            else {k: v.detach().cpu().clone() for k, v in self._net.state_dict().items()}
        )

    def predict(self, s_grid, akey: tuple) -> np.ndarray:
        if self._net is None or np.asarray(s_grid).shape != self._shape:
            return np.asarray(
                s_grid
            )  # untrained / wrong shape -> identity (never crash the planner)
        import torch

        s = np.asarray(s_grid)
        with torch.no_grad():
            xb = self._encode(s, tuple(akey), torch).unsqueeze(0).to(self._device)
            cls = self._net(xb).argmax(dim=1).squeeze(0).cpu().numpy()  # 0=KEEP, 1..16=set colour
        # apply the residual: KEEP -> copy input; else -> colour (class-1). Unchanged cells are correct
        # by construction; only the cells the net flags as changed deviate from the input.
        return np.where(cls == 0, s, (cls - 1)).astype(s.dtype)


def action_key(action_id: int, data: Any) -> tuple:
    """Canonical action key: ``(6, x, y)`` for a click with coords, else ``(action_id,)``. Mirrors
    ``arc_competition_agent._action_key`` exactly so L0/L1 keying matches the live agent."""
    if int(action_id) == 6 and isinstance(data, dict) and "x" in data and "y" in data:
        return (6, int(data["x"]), int(data["y"]))
    return (int(action_id),)


class LiveTTTWorldModel:
    """A per-game world model learned from played transitions, exposing the engine + win predicate the
    existing ``plan_in_model`` BFS and ``WorldModelVerifier`` consume."""

    def __init__(
        self,
        game: str = "?",
        *,
        refit_every: int = 8,
        min_transitions: int = 16,
        dynamics_backend: str = "dsl",
        prior_state: Any = None,
        cnn_epochs: int = 40,
    ) -> None:
        self.game = game
        self.refit_every = int(refit_every)
        self.min_transitions = int(min_transitions)
        # live per-game CNN fine-tuning epochs -- LIGHT (the agent inducts on stalls; the offline prior
        # pretrain uses its own higher epoch count). Warm-starting from the prior means few epochs suffice.
        self.cnn_epochs = int(cnn_epochs)
        # a pretrained cross-game CNN PRIOR state_dict (models/arc_dynamics_prior.pt); when set and the
        # backend is 'cnn', the per-game learner warm-starts from it -> fewer real probes to converge.
        self._prior_state = prior_state
        # 'dsl' = ObjectDeltaModel rule learner (fast, zero-train, but a fixed hypothesis class);
        # 'cnn' = CNNDynamics learned net (fits arbitrary local rules, the make-or-break test for games
        # the rule class can't express). Both expose .fit(transitions)/.predict(grid, akey).
        self.dynamics_backend = dynamics_backend
        self._l0: dict[
            tuple, np.ndarray
        ] = {}  # (grid.tobytes(), akey) -> next_grid (exact backbone)
        self._dsl_transitions: list[tuple] = []  # (s, akey, s2) feed for the L1 learner
        self._win_states: set[bytes] = set()  # next_grid bytes observed at a level-up
        self._l1: Any = None
        self._last_refit = -1
        self._refits = 0

    def _new_l1(self) -> Any:
        if self.dynamics_backend in ("cnn", "cnn_obj"):
            return CNNDynamics(
                self.game,
                epochs=self.cnn_epochs,
                object_features=(self.dynamics_backend == "cnn_obj"),
            )
        return ObjectDeltaModel(self.game)

    def _fit_l1(self, transitions):
        l1 = self._new_l1()
        if self.dynamics_backend == "cnn" and self._prior_state is not None:
            return l1.fit(transitions, warm_state=self._prior_state)
        return l1.fit(transitions)

    # --- learning from play (FREE compute; NOT rate-limited) ----------------------------------------
    def observe(
        self,
        grid: Any,
        action: int,
        data: Any,
        next_grid: Any,
        level_before: int = 0,
        level_after: int = 0,
    ) -> None:
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
        self.observe(
            t.grid,
            t.action,
            t.data,
            t.next_grid,
            getattr(t, "level_before", 0),
            getattr(t, "level_after", 0),
        )

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
        at 0.0). Returns EXACT-FULL-GRID-match accuracy in [0, 1].

        CAVEAT (2026-06-21, the LOO gate probe): exact-full-grid match is ~0 for a 64x64 CNN dynamics
        model that is ~55% changed-cell-accurate -- getting EVERY one of hundreds of changed cells right
        across a full grid essentially never happens. So this metric gates the TTT path OUT on unseen games
        (warm 0/5 on the LOO probe) even though the dynamics genuinely transferred. Use
        ``trust_cell_recall`` as the granularity-matched alternative; see ``gated_engine_from_transitions``
        ``trust_metric``."""
        from carnot.agentic.arc_executable_world_model import WorldModelVerifier

        if not held_out:
            return 0.0
        return float(WorldModelVerifier(held_out).score(self.engine).accuracy)

    def trust_cell_recall(self, held_out: list) -> float:
        """Mean CHANGED-CELL recall of the learned engine on held-out transitions -- the GRADED trust
        metric matched to the CNN dynamics model's granularity. ``trust`` (exact-full-grid match) reads ~0
        for a model that is genuinely 55%-cell-accurate, so it gates the path out; cell-recall reflects the
        real prediction quality the prior actually improved (0.314 -> 0.5485 cross-game). Recall is computed
        ONLY on the changed cells (next != input -- the hard part the dynamics must model); noop transitions
        are correct by construction via the KEEP head and excluded. Returns mean recall in [0, 1].

        Honest limitation: a model passing cell-recall 0.5 still mispredicts ~half the changed cells, so a
        plan BFS'd through it can diverge from reality -- the live agent's execute-and-halt-on-divergence
        loop (``plan_and_execute``) is the safety net, and whether plan-through-an-imperfect-model actually
        SOLVES is a separate, unproven question from whether the gate fires."""
        chg = [
            t for t in held_out if not np.array_equal(np.asarray(t.grid), np.asarray(t.next_grid))
        ]
        if not chg:
            return 0.0
        recalls = []
        for t in chg:
            s = np.asarray(t.grid)
            s2 = np.asarray(t.next_grid)
            pred = np.asarray(self.engine(s, t.action, t.data))
            if pred.shape != s2.shape:
                recalls.append(0.0)
                continue
            m = s != s2
            recalls.append(float((pred[m] == s2[m]).mean()))
        return float(np.mean(recalls)) if recalls else 0.0

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


class _ResidualLiveTTTWorldModel:
    """ReDRAW-style target residual over a frozen base ``LiveTTTWorldModel``.

    The base model is fit on prior level-N transitions and then left untouched.
    The residual model is fit only on level-N+1 target transitions. At prediction
    time, exact target observations and non-identity residual predictions take
    precedence; otherwise the frozen base supplies the dynamics.
    """

    def __init__(self, base: LiveTTTWorldModel, residual: LiveTTTWorldModel) -> None:
        self.base = base
        self.residual = residual

    def engine(self, grid: np.ndarray, action: int, data: Optional[dict]) -> np.ndarray:
        g = np.asarray(grid)
        akey = action_key(action, data)
        hit = self.residual._l0.get((g.tobytes(), akey))
        if hit is not None:
            return hit
        if self.residual._l1 is not None:
            pred = np.asarray(self.residual._l1.predict(g, akey))
            if pred.shape == g.shape and not np.array_equal(pred, g):
                return pred
        return np.asarray(self.base.engine(g, action, data))

    def is_level_complete(self, grid: np.ndarray) -> bool:
        return self.residual.is_level_complete(grid)

    def trust(self, held_out: list) -> float:
        from carnot.agentic.arc_executable_world_model import WorldModelVerifier

        if not held_out:
            return 0.0
        return float(WorldModelVerifier(held_out).score(self.engine).accuracy)

    def trust_cell_recall(self, held_out: list) -> float:
        chg = [
            t for t in held_out if not np.array_equal(np.asarray(t.grid), np.asarray(t.next_grid))
        ]
        if not chg:
            return 0.0
        recalls = []
        for t in chg:
            s = np.asarray(t.grid)
            s2 = np.asarray(t.next_grid)
            pred = np.asarray(self.engine(s, t.action, t.data))
            if pred.shape != s2.shape:
                recalls.append(0.0)
                continue
            m = s != s2
            recalls.append(float((pred[m] == s2[m]).mean()))
        return float(np.mean(recalls)) if recalls else 0.0

    def ttt_diagnostics(self) -> dict:
        base = self.base.ttt_diagnostics()
        residual = self.residual.ttt_diagnostics()
        return {
            "game": base.get("game"),
            "l0_table_size": residual.get("l0_table_size", 0),
            "n_transitions": residual.get("n_transitions", 0),
            "base_transition_count": base.get("n_transitions", 0),
            "base_l0_table_size": base.get("l0_table_size", 0),
            "l1_refits": residual.get("l1_refits", 0),
            "base_l1_refits": base.get("l1_refits", 0),
            "n_win_states": residual.get("n_win_states", 0),
            "verifier_is_oracle": False,
            "warm_start": True,
            "residual_adapter": "redraw_frozen_base_plus_target_residual",
        }


def _load_prior(prior_path: str) -> Any:
    """Load the pretrained cross-game CNN prior state_dict (models/arc_dynamics_prior.pt), or None."""
    try:
        import torch
        from pathlib import Path

        p = Path(prior_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[3] / prior_path
        return torch.load(p) if p.exists() else None
    except Exception:
        return None


def gated_engine_from_transitions(
    game: str,
    transitions: list,
    *,
    prior_path: str = "models/arc_dynamics_prior.pt",
    trust_threshold: float = 0.5,
    holdout_frac: float = 0.25,
    dynamics_backend: str = "cnn",
    trust_metric: str = "cell_recall",
    prior_transitions: Optional[list] = None,
):
    """Build a per-game world-model engine LEARNED from the played transitions, WARM-STARTED from the
    cross-game prior (models/arc_dynamics_prior.pt, the one that transfers 5/5), and GATED by held-out
    trust. Returns ``(engine, is_level_complete, diag)``.

    engine/is_level_complete are None UNLESS the learned model reproduces a held-out split of the played
    transitions at >= trust_threshold changed-cell recall by default. REQ-ARC-FCP-4715 flips this cheap
    floor from exact-grid accuracy to ``cell_recall`` because the online CNN can be useful at changed-cell
    granularity even when exact full-grid accuracy is near zero. Explicit ``trust_metric="exact"`` callers
    still get the old bar. This is the execution-grounded, zero-LLM alternative to e3.load_engine: the
    conductor's live agent tries it FIRST and falls through to the LLM induction if the gate fails. diag
    carries the prior-loaded flag + both held-out metrics for telemetry. (Oracle-distinct learned dynamics;
    verifier_is_oracle False.)

    ``prior_transitions`` is the Exp5157 cross-level extension: when provided,
    fit a frozen level-N base model on that prior evidence and fit only a
    level-N+1 residual model from ``transitions``. Existing callers do not pass
    this argument, so the live cold-slice behavior remains unchanged.
    """
    target = list(transitions or [])
    prior = list(prior_transitions or [])
    warm_start = bool(prior)
    diag: dict = {
        "backend": dynamics_backend,
        "n_transitions": len(target),
        "warm_start": warm_start,
        "prior_transition_count": len(prior),
    }
    if len(target) < 8 and not prior:
        diag["skip"] = "too_few_transitions"
        return None, None, diag
    if prior and len(target) + len(prior) < 8:
        diag["skip"] = "too_few_transitions"
        return None, None, diag
    prior_state = _load_prior(prior_path) if dynamics_backend in ("cnn", "cnn_obj") else None
    diag["prior_loaded"] = prior_state is not None
    if float(holdout_frac) <= 0.0 or not target:
        train, held = target, []
    else:
        k = max(2, int(len(target) * holdout_frac))
        train, held = target[:-k], target[-k:]

    if warm_start:
        base = LiveTTTWorldModel(
            game,
            dynamics_backend=dynamics_backend,
            prior_state=prior_state,
        )
        for t in prior:
            base.observe_transition(t)
        base.fit_now()
        residual = LiveTTTWorldModel(game, dynamics_backend=dynamics_backend)
        for t in train:
            residual.observe_transition(t)
        residual.fit_now()
        ttt = _ResidualLiveTTTWorldModel(base, residual)
        diag.update(ttt.ttt_diagnostics())
    else:
        ttt = LiveTTTWorldModel(
            game,
            dynamics_backend=dynamics_backend,
            prior_state=prior_state,
        )
        for t in train:
            ttt.observe_transition(t)
        ttt.fit_now()
    acc = ttt.trust(held) if held else 1.0  # exact-full-grid match (the original, strict bar)
    cell = ttt.trust_cell_recall(held) if held else 1.0  # changed-cell recall (granularity-matched)
    gate_value = cell if trust_metric == "cell_recall" else acc
    diag.update(
        heldout_accuracy=round(acc, 4),
        heldout_cell_recall=round(cell, 4),
        trust_threshold=trust_threshold,
        trust_metric=trust_metric,
    )
    if gate_value < trust_threshold:
        diag["gate"] = "FAIL"
        return None, None, diag
    for t in held:  # gate passed -> refit on ALL transitions for the final engine
        if warm_start:
            ttt.residual.observe_transition(t)
        else:
            ttt.observe_transition(t)
    if warm_start:
        ttt.residual.fit_now()
        diag.update(ttt.ttt_diagnostics())
    else:
        ttt.fit_now()
    diag["gate"] = "PASS"
    return ttt.engine, ttt.is_level_complete, diag

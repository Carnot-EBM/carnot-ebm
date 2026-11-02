"""Online action-effect scorer: a drop-in for frame_change_scorer that learns ONLINE per game.

Spec refs: REQ-ARC-OAE-4710, SCENARIO-ARC-OAE-4710.

WHY THIS EXISTS
---------------
The shipped E3AgentPolicy uses a FROZEN cross-game scorer (LiveActionEffectScorer loaded from
the exp4629 checkpoint plus a PersistentAEM trained on the transition corpus). That scorer works
reasonably on the 25 public games, but for HIDDEN games it starts from a prior that may be
completely wrong for the new game's mechanics.

This module implements "StochasticGoose-leader": an action-effect CNN trained ONLINE per game
episode, with FREE SUPERVISION -- we observe every (before_frame, action, after_frame) triple
that the explorer actually executes, and use whether the frame CHANGED as the binary label.  No
hand-labels, no held-out corpus: the agent's own exploration provides the signal.

The key insight: frame-change detection is self-supervised (we CAN compare before vs after), but
the direction of change is NOT -- we cannot label "was this change beneficial" from frames alone
without a win signal (the level-up event). So we train on what we can observe freely and rely on
the MEMORY (PersistentAEM) for higher-level signal.

ARCHITECTURE
------------
OnlineActionEffectScorer wraps:
  - memory   : a frozen PersistentAEM (cross-game prior; NOT updated online)
  - cnn_scorer: a FrameChangeScorer around a trainable SmallFrameChangeCNN

Score blend: memory * 1.0 + cnn * 0.05 (same weights as LiveActionEffectScorer).

Online loop:
  1. observe_transition(before, action, data, after) -- buffer a FrameActionEffectExample
  2. Every `fit_every` observations, run one SGD step on the buffered examples
  3. MANDATORY: clear cnn_scorer._cache after every fit (stale predictions from before the
     weight update would otherwise persist and silently mislead the explorer)
  4. reset() on level-up restores the initial prior snapshot, clears Adam state, and starts
     the next level from the cross-game prior rather than stale level-specific weights.

PARITY SAFETY
-------------
The guarded hooks in arc_competition_agent.py check `hasattr(fcs, "observe_transition")` before
calling observe_transition, and `getattr(fcs, "propose_enabled", False)` before proposing.
LiveActionEffectScorer has neither attribute, so the frozen path is byte-identical to before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from carnot.agentic.arc_frame_change_predictor import (
    DEFAULT_FRAME_SIZE,
    DEFAULT_NUM_COLORS,
    FrameActionEffectExample,
    FrameChangeScorer,
    SmallFrameChangeCNN,
    _effect_loss_for_batch,
    frame_state_key,
    load_live_action_effect_scorer,
    load_live_frame_change_cnn_scorer,
    load_cached_transition_effect_rows,
)
from carnot.agentic.arc_agi3_world_model import grid_of


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class OnlineActionEffectScorer:
    """Drop-in replacement for frame_change_scorer that adapts online per game episode.

    REQ-ARC-OAE-4710: this scorer has the same .candidate_score(frame, candidate) interface as
    LiveActionEffectScorer, PLUS online methods (.observe_transition, .propose_coords, .reset).
    The guarded hooks in StepwiseExplorer call these methods iff they exist, so the frozen scorer
    is a perfect no-op parity.

    WHY memory is frozen: the PersistentAEM encodes cross-game statistics from ~14k transition
    rows collected across all 25 public games. Updating it online per-episode would conflate
    game-specific observations with the cross-game prior, corrupting the base that works on known
    games. Instead, only the CNN adapts -- it is small enough to train on-device in <10ms/step.

    WHY cnn_weight=0.05: the CNN starts random (or from the exp4629 warm checkpoint) and may
    be noisy early in an episode. A low weight ensures the PersistentAEM memory dominates until
    the CNN has seen enough transitions to be reliable. This matches the frozen scorer's blend.

    WHY fit_every=5: five transitions give the CNN at least one sample of each directional action
    (there are 5 terminal actions) before the first gradient step, reducing early oscillation.
    Too frequent = high variance gradients on single-sample batches. Too infrequent = slow adapt.
    """

    memory: Any | None
    cnn_scorer: FrameChangeScorer
    memory_weight: float = 1.0
    cnn_weight: float = 0.05
    train_enabled: bool = True
    propose_enabled: bool = False
    lr: float = 1e-4
    fit_every: int = 5
    max_batch: int = 32
    max_buffer: int = 200_000
    propose_k: int = 6

    # Online state -- initialized in __post_init__ so the dataclass stays declarative.
    _optimizer: Any = field(default=None, init=False, repr=False)
    _buffer: list[FrameActionEffectExample] = field(default_factory=list, init=False, repr=False)
    _seen: set[tuple[str, int, Any, Any]] = field(default_factory=set, init=False, repr=False)
    _seen_order: list[tuple[str, int, Any, Any]] = field(default_factory=list, init=False, repr=False)
    _obs_since_fit: int = field(default=0, init=False, repr=False)
    _observed: int = field(default=0, init=False, repr=False)
    _fits: int = field(default=0, init=False, repr=False)
    _errors: int = field(default=0, init=False, repr=False)
    _resets_to_prior: int = field(default=0, init=False, repr=False)
    _reset_levels: list[int] = field(default_factory=list, init=False, repr=False)
    _initial_state: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _last_gradient_norm: float = field(default=0.0, init=False, repr=False)
    _max_gradient_norm: float = field(default=0.0, init=False, repr=False)
    _positive_grad_steps: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Snapshot the starting CNN prior and build the Adam optimizer."""
        self._initial_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.cnn_scorer.model.state_dict().items()
        }
        self._rebuild_optimizer()

    def _rebuild_optimizer(self) -> None:
        """Recreate Adam so level resets do not carry stale moment estimates."""
        if self.train_enabled:
            # WHY set_to_none=True in zero_grad later: releases gradient tensors immediately,
            # reducing peak memory by ~1x the parameter count during the step.
            self._optimizer = torch.optim.Adam(
                self.cnn_scorer.model.parameters(), lr=float(self.lr)
            )
        else:
            self._optimizer = None

    def candidate_score(self, frame: Any, candidate: Any) -> float:
        """REQ-ARC-OAE-4710: blended score matching LiveActionEffectScorer's formula.

        Each component is wrapped in its own try/except so a transient CNN failure does NOT
        silence the memory score. Errors are counted for diagnostics -- we want to know if
        the CNN is failing, but we NEVER want a scorer crash to kill the explorer step.

        DICT-CANDIDATE NORMALIZATION (2026-06-25 false-negative fix): the explorer scores
        candidates from TWO shapes -- ArcAction objects (from rich_action_candidates, with
        ``.action_id``/``.data``) AND plain dict rows ``{"action": .., "data": ..}`` (the
        frontier's ``untested`` lists, _candidates:761). ``FrameChangeScorer.candidate_score``
        reads ``getattr(candidate, "action_id")``, which raises ``AttributeError`` on a dict --
        so the CNN term was silently DISCARDED on ~20/25 games (the shipped LiveActionEffectScorer
        swallows the same AttributeError with a bare except, which is why the bug was invisible).
        That made the trained CNN's output a no-op on most games -> the online-vs-frozen first-win
        null was a FALSE NEGATIVE. We normalize dict candidates to an action-like shim before the
        CNN call so the trained CNN actually contributes. PersistentAEM.candidate_score already
        tolerates dicts, so the memory term is left as-is.
        """
        score = 0.0
        if self.memory is not None:
            try:
                score += float(self.memory_weight) * float(self.memory.candidate_score(candidate))
            except Exception:
                self._errors += 1
        try:
            score += float(self.cnn_weight) * float(
                self.cnn_scorer.candidate_score(frame, _as_action_like(candidate))
            )
        except Exception:
            self._errors += 1
        return float(score)

    def observe_transition(
        self,
        before_frame: Any,
        action_id: int,
        data: Any,
        after_frame: Any,
    ) -> None:
        """REQ-ARC-OAE-4710: observe one (before, action, after) triple and buffer it for fitting.

        WHY frame_delta = binary (0/1): the CNN's click_heatmap and directional heads output
        sigmoid probabilities. They should predict P(frame changed | action), so a binary label
        (changed=1, unchanged=0) is the correct target for BCE loss. We don't need the MAGNITUDE
        of change -- just whether anything moved.

        WHY dedup by (state_key, action_id, x, y): replaying the same action from the same state
        multiple times (e.g. the explorer tries the same click during a loop) adds no new
        information and inflates the batch with correlated samples. Dedup is a micro-regularizer.
        """
        try:
            state_key = frame_state_key(before_frame)
            x = data.get("x") if isinstance(data, dict) else None
            y = data.get("y") if isinstance(data, dict) else None
            dedup_key = (state_key, int(action_id), x, y)
            if dedup_key in self._seen:
                return
            self._seen.add(dedup_key)
            self._seen_order.append(dedup_key)
            if len(self._seen_order) > int(self.max_buffer):
                self._seen.discard(self._seen_order.pop(0))

            # Compute whether the frame changed -- the free self-supervised label.
            try:
                before_grid = np.asarray(grid_of(before_frame))
                after_grid = np.asarray(grid_of(after_frame))
                if before_grid.shape != after_grid.shape:
                    frame_delta = 1.0
                else:
                    frame_delta = 1.0 if np.any(before_grid != after_grid) else 0.0
            except Exception:
                frame_delta = 1.0  # assume changed on error

            example = FrameActionEffectExample(
                frame=before_frame,
                action_id=int(action_id),
                frame_delta=float(frame_delta),
                level_progress=0.0,  # not available at observe time
                state_key=state_key,
                x=x,
                y=y,
            )
            if len(self._buffer) >= int(self.max_buffer):
                self._buffer.pop(0)
            self._buffer.append(example)
            self._observed += 1
            self._obs_since_fit += 1
            self._maybe_fit()
        except Exception:
            self._errors += 1

    def _maybe_fit(self) -> None:
        """Run one online gradient step if we have accumulated enough new observations.

        WHY always call model.eval() and clear cache at the end -- even on no-op: ensures the
        model is in the correct eval mode for inference and the cache does not serve stale
        predictions from before the weight update. If _maybe_fit does nothing (e.g. train
        disabled, or not enough obs), the cache and eval state are still guaranteed consistent.
        """
        if not self.train_enabled or self._obs_since_fit < self.fit_every or not self._buffer:
            return
        batch = self._buffer[-self.max_batch :]
        self.cnn_scorer.model.train()
        try:
            loss = _effect_loss_for_batch(
                self.cnn_scorer.model,
                batch,
                num_colors=DEFAULT_NUM_COLORS,
                size=DEFAULT_FRAME_SIZE,
                device=torch.device("cpu"),
            )
            if loss is not None:
                assert self._optimizer is not None
                self._optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_sq = 0.0
                for parameter in self.cnn_scorer.model.parameters():
                    if parameter.grad is None:
                        continue
                    grad_sq += float(parameter.grad.detach().pow(2).sum().item())
                grad_norm = float(grad_sq**0.5)
                self._last_gradient_norm = grad_norm
                self._max_gradient_norm = max(float(self._max_gradient_norm), grad_norm)
                if grad_norm > 0.0:
                    self._positive_grad_steps += 1
                self._optimizer.step()
                self._fits += 1
        except Exception:
            self._errors += 1
        finally:
            # MANDATORY: always reset eval mode and clear the prediction cache AFTER every
            # fit attempt. Stale-cache risk (#4): if a fit changes the weights but the cache
            # retains pre-fit predictions keyed by frame hash, the explorer will continue
            # using the OLD scores for states it has already seen. Clearing here forces a
            # fresh forward pass on next score request.
            self.cnn_scorer.model.eval()
            self.cnn_scorer._cache.clear()
        self._obs_since_fit = 0

    def propose_coords(self, frame: Any, k: int | None = None) -> list[tuple[int, int]]:
        """REQ-ARC-OAE-4710: return top-k (x,y) grid coordinates with high click-heatmap score.

        WHY propose_enabled=False by default: the CNN starts random and its click heatmap may
        point at arbitrary cells early in an episode. Proposing bad candidates wastes explorer
        budget. Enable only after verifying the CNN has been trained enough to be directional.

        The mapping from 64x64 heatmap coords back to grid coords is the inverse of what
        FrameChangeScorer.candidate_score does for ACTION6: for a grid of shape (h, w),
        heatmap cell (hy, hx) maps to grid (y, x) via round(hy/(size-1)*(h-1)).
        """
        if k is None:
            k = self.propose_k
        results: list[tuple[int, int]] = []
        try:
            tensor = _frame_to_tensor_safe(frame)
            if tensor is None:
                return results
            self.cnn_scorer.model.eval()
            with torch.no_grad():
                click_heatmap, _directional = self.cnn_scorer.model(tensor.unsqueeze(0))
            heatmap = click_heatmap[0, 0].cpu()  # shape (size, size)
            size = int(heatmap.shape[-1])
            grid = grid_of(frame)
            h, w = grid.shape
            # Flatten, argsort descending, convert to (x,y) grid coords -- dedup grid cells.
            flat = heatmap.flatten()
            top_idx = flat.argsort(descending=True)
            seen_cells: set[tuple[int, int]] = set()
            for idx in top_idx:
                if len(results) >= int(k):
                    break
                hy = int(idx.item()) // size
                hx = int(idx.item()) % size
                # Inverse of FrameChangeScorer._predict mapping (grid -> heatmap):
                #   heatmap_y = round(grid_y / max(1, h-1) * (size-1))
                # so grid_y = round(heatmap_y / max(1, size-1) * (h-1))
                gy = round(hy / max(1, size - 1) * max(1, h - 1))
                gx = round(hx / max(1, size - 1) * max(1, w - 1))
                gy = max(0, min(h - 1, gy))
                gx = max(0, min(w - 1, gx))
                cell = (int(gx), int(gy))
                if cell not in seen_cells:
                    seen_cells.add(cell)
                    results.append(cell)
        except Exception:
            self._errors += 1
        return results

    def reset(self, *, level: int | None = None, reset_to_prior: bool = True) -> None:
        """Reset per-level state and, by default, restore the initial CNN prior.

        REQ-ARC-FCP-4715: on level-up the goal-free driver starts the next level from the
        cross-game prior snapshot, not from weights overfit to the previous level. The optimizer
        is rebuilt too, because Adam moments are level-specific state. Pass
        ``reset_to_prior=False`` only for diagnostic ablations that deliberately preserve online
        weights across levels.
        """
        self._buffer.clear()
        self._seen.clear()
        self._seen_order.clear()
        self._obs_since_fit = 0
        if reset_to_prior and self._initial_state:
            state = {name: tensor.clone() for name, tensor in self._initial_state.items()}
            self.cnn_scorer.model.load_state_dict(state)
            self._rebuild_optimizer()
            self._resets_to_prior += 1
            if level is not None:
                self._reset_levels.append(int(level))
        self.cnn_scorer._cache.clear()

    def diagnostics(self) -> dict[str, Any]:
        """REQ-ARC-OAE-4710: self-report for experiment artifacts."""
        return {
            "observed": int(self._observed),
            "fits": int(self._fits),
            "online_train_steps_executed": int(self._fits),
            "train_steps_with_positive_grad_norm": int(self._positive_grad_steps),
            "last_gradient_norm": float(self._last_gradient_norm),
            "max_gradient_norm": float(self._max_gradient_norm),
            "errors": int(self._errors),
            "train_enabled": bool(self.train_enabled),
            "propose_enabled": bool(self.propose_enabled),
            "buffer_size": int(len(self._buffer)),
            "max_buffer": int(self.max_buffer),
            "resets_to_prior": int(self._resets_to_prior),
            "reset_levels": list(self._reset_levels),
        }


def _as_action_like(candidate: Any) -> Any:
    """Return a candidate exposing ``.action_id`` / ``.data`` for FrameChangeScorer.

    The CNN scorer reads ``getattr(candidate, "action_id")``; the frontier hands it plain dict
    rows ``{"action": .., "data": ..}``. Wrap a dict in a SimpleNamespace so the CNN term works
    on BOTH shapes (the 2026-06-25 false-negative fix). Non-dicts (ArcAction) pass through.
    """
    if isinstance(candidate, dict):
        from types import SimpleNamespace

        aid = candidate.get("action_id", candidate.get("action"))
        try:
            aid = int(aid)
        except (TypeError, ValueError):
            return candidate
        return SimpleNamespace(action_id=aid, data=candidate.get("data"))
    return candidate


def _frame_to_tensor_safe(frame: Any) -> "torch.Tensor | None":
    """Convert a frame to a CHW tensor, returning None on any error."""
    try:
        from carnot.agentic.arc_frame_change_predictor import frame_to_tensor

        return frame_to_tensor(frame, num_colors=DEFAULT_NUM_COLORS, size=DEFAULT_FRAME_SIZE)
    except Exception:
        return None


_MEMORY_CACHE: dict[str, Any] = {}


def _load_cached_memory(root: Path) -> Any | None:
    """Load (once, then cache by repo root) the frozen PersistentAEM cross-game prior.

    The memory term is identical for every (game, variant) attempt within a sweep, so reloading
    the transition corpus per attempt is pure waste. Cached by str(root) so a different repo path
    (e.g. a worktree) gets its own entry. Returns None on any failure (the scorer degrades to
    CNN-only, exactly as the pre-cache code did).
    """
    key = str(Path(root).resolve())
    if key in _MEMORY_CACHE:
        return _MEMORY_CACHE[key]
    memory = None
    try:
        from carnot.agentic.arc_solver_kit import PersistentAEM

        rows = load_cached_transition_effect_rows(root)
        if rows:
            memory = PersistentAEM.from_effect_rows(rows)
    except Exception:
        memory = None
    _MEMORY_CACHE[key] = memory
    return memory


def build_online_scorer(arm: str, root: Path) -> Any:
    """REQ-ARC-OAE-4710: factory function that returns the correct scorer for a given arm.

    WHY a factory per arm (not per game): the arm determines the ARCHITECTURE of the scorer
    (frozen vs online-scratch vs online-warm). Within an arm, each game attempt should get
    a FRESH scorer (via OnlineActionEffectScorer.reset()) so the per-game online learning does
    not bleed across games.

    Arms:
      "frozen"             -- the exact LiveActionEffectScorer the shipped agent uses; zero drift
      "online-scratch"     -- fresh random CNN + frozen memory; tests whether online learning from
                             scratch can overcome initialization noise
      "online-warm"        -- exp4629 warm checkpoint + frozen memory + online training; tests
                             whether a warm start + online adaptation beats frozen
      "online-warm-propose"-- same as online-warm but with propose_enabled=True; tests the full
                             StochasticGoose-leader loop including coordinate proposals

    WHY frozen arm returns LiveActionEffectScorer directly (not wrapped in OnlineActionEffectScorer):
    The guarded hooks in arc_competition_agent.py use `hasattr(fcs, "observe_transition")` and
    `getattr(fcs, "propose_enabled", False)` to decide whether to call the online methods.
    Returning the original LiveActionEffectScorer ensures the frozen arm is byte-identical to
    the shipped behavior -- no observe calls, no proposes, no cache clears.
    """
    if arm == "frozen":
        # The SHIPPED scorer verbatim (raw LiveActionEffectScorer) -- the positive control that
        # reproduces the committed exp4605 0.04 baseline. Its CNN term is silently discarded on
        # ~20/25 games by the dict-candidate AttributeError (the bug this module's wrapper fixes);
        # keeping it raw here is deliberate so we can measure what FIXING the bug changes.
        scorer = load_live_action_effect_scorer(root)
        return scorer

    if arm == "frozen-fixed":
        # CONTROL for the false-negative fix: the SAME warm CNN as online-warm, wrapped so the
        # dict-candidate normalization makes the CNN term actually CONTRIBUTE (not discarded), but
        # train_enabled=False so the CNN is NOT updated online. Comparing frozen (CNN discarded) ->
        # frozen-fixed (CNN contributes, untrained) -> online-warm (CNN contributes, trained)
        # cleanly decomposes "fixing the bug" from "online training".
        memory = _load_cached_memory(root)
        cnn = load_live_frame_change_cnn_scorer(root)
        if cnn is None:
            cnn = FrameChangeScorer(SmallFrameChangeCNN(num_colors=DEFAULT_NUM_COLORS, hidden_channels=8))
        return OnlineActionEffectScorer(
            memory=memory, cnn_scorer=cnn, train_enabled=False, propose_enabled=False
        )

    # For all online arms: load the memory (frozen cross-game prior). The PersistentAEM is the
    # SAME static cross-game object for every (game, variant) attempt, so building it once and
    # caching it (keyed by repo root) avoids re-reading the ~14k-row transition corpus 25x+ per
    # arm. Only the CNN differs per scorer; the memory is shared & frozen.
    memory = _load_cached_memory(root)

    if arm == "online-scratch":
        # Fresh random CNN -- no checkpoint, hidden_channels=8 to match exp4629's architecture.
        # WHY hidden_channels=8: if we want to compare online-scratch vs online-warm fairly, the
        # model architectures must be identical. The exp4629 checkpoint used hidden_channels=8.
        cnn = FrameChangeScorer(
            SmallFrameChangeCNN(num_colors=DEFAULT_NUM_COLORS, hidden_channels=8)
        )
        return OnlineActionEffectScorer(
            memory=memory,
            cnn_scorer=cnn,
            train_enabled=True,
            propose_enabled=False,
        )

    if arm in ("online-warm", "online-warm-propose"):
        # Load the exp4629 warm checkpoint. If the checkpoint is missing, fall back to random
        # (same behavior as online-scratch) rather than crashing.
        cnn = load_live_frame_change_cnn_scorer(root)
        if cnn is None:
            # Checkpoint missing: graceful degradation to random init, same architecture.
            cnn = FrameChangeScorer(
                SmallFrameChangeCNN(num_colors=DEFAULT_NUM_COLORS, hidden_channels=8)
            )
        return OnlineActionEffectScorer(
            memory=memory,
            cnn_scorer=cnn,
            train_enabled=True,
            propose_enabled=(arm == "online-warm-propose"),
        )

    raise ValueError(
        f"Unknown arm {arm!r}. "
        "Valid arms: 'frozen', 'online-scratch', 'online-warm', 'online-warm-propose'."
    )

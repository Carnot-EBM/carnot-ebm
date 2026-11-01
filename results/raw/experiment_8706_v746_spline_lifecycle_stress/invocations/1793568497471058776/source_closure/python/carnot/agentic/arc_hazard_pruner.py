"""Online hazard move-pruner -- the WIRED-IN, efficiency-only salvage of the hazard-aware world model.

WHY THIS EXISTS (2026-06-22 operator directive). The hazard-aware nav world model
(``arc_nav_world_model``: induce a nav model from transitions, detect a charging-enemy from the agent's
own death transitions, route safe detours) was developed in the OUTER LOOP but was never reachable from
the LIVE agent's solve path -- it lived only in ``scripts/experiments/*``. The live agent already
deep-solves tu93 to L3 via plain verifier-routed best-first search (``arc_solver_kit.OfflineSolver`` over
a ``GameAdapter``), so the hazard model bought NO new capability. The one thing it CAN add is EFFICIENCY:
the blind search wastes expansions walking into chargers and backtracking. This module turns the hazard
model into a move-pruner the live ``OfflineSolver`` consumes, so the SAME induce->detect->learn process
now runs on the live path and is measured by a states-expanded reduction (a north-star efficiency axis),
not by re-solving an already-solved level.

KEY TRANSFER PROPERTY (the honesty fix). The original ``omni`` lethal rung was VALIDATED by an exhaustive
position-keyed real-env BFS over tu93 L3's ground-truth (state, action, died) labels -- a thing a
hidden-game agent CANNOT run under an action budget. Here the rung (``toward`` vs ``omni``) is instead
selected by IN-SAMPLE OBSERVED-TRANSITION FIT: which rung's ``is_lethal`` best classifies the deaths/safes
the agent ACTUALLY OBSERVED while playing. That selection needs only the agent's own transitions (NO offline
BFS) -- THAT is what makes it transfer to a game it has never seen. The trust/specificity numbers are scored
on those same observed transitions (IN-SAMPLE, hence optimistic: specificity bounds false-positives only on
moves already seen, not on the unobserved grids ``should_prune`` later fires on), so we treat them as a
CONSERVATIVE GATE rather than a guarantee, and only prune when the chosen rung clears a high specificity
bar. The real correctness backstop is empirical -- a false-positive prune that broke a solve would surface
as a failed reproduction gate; measured on tu93 L3 the solve is preserved (reproduced, zero safe-move
prunes). When unsure the gate keeps the model unfitted and the search simply runs as before.

The pruner NO-OPS when no charging-enemy is detected (no avatar-removal deaths observed), so it is safe to
enable for ANY game -- a non-nav / non-hazard game never fits a hazard model and the pruner never prunes.

verifier_is_oracle: false -- this is a LEARNED hazard predictor fit from observed transitions, not the
executable oracle that defines correctness.
"""

from __future__ import annotations

import json
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from carnot.agentic.arc_nav_world_model import HazardAwareNavWorldModel, InducedNavWorldModel


CLICK_ACTION_ID = 6  # ARC-AGI-3's pointer action; its label carries {"data": {"x": .., "y": ..}}


def _default_action_of_label(label: Any) -> Optional[int]:
    """Decode a keyboard-nav action label ('{"action": N}' or a bare int/dict) to its action int.
    Returns None for a label with no action id at all (e.g. a bare '{"x": 1, "y": 2}' coordinate).

    CAVEAT, MEASURED 2026-07-26 -- this does NOT filter clicks. This docstring previously claimed it
    "returns None for non-nav labels (click/paint games) so the pruner cleanly no-ops on them". That
    is true only for a coordinate-ONLY label. The label shape both live consumers actually pass is
    ``{"action": 6, "data": {"x": .., "y": ..}}`` (see ``StepwiseExplorer._ingest`` and
    ``arc_solver_kit.OfflineSolver``'s ``_call_action_labels``), which decodes to the int 6 and is
    therefore buffered as if it were a keyboard-nav action. Consequence: on a click-heavy game the
    transition buffer that ``InducedNavWorldModel.fit`` sees is dominated by pointer rows, and the
    fitter is asked to learn a spatial DISPLACEMENT for action 6. That can only degrade the avatar /
    displacement fit the death predicate depends on. The prune side is unaffected (action 6 is absent
    from any fitted displacement map, so ``is_lethal`` returns False for it), so this is a
    fit-pollution defect, not a false-prune defect.

    The behaviour is left UNCHANGED here on purpose: this function is shared with the offline dev
    twin, whose one published measurement (``results/arc_hazard_prune_ab_tu93.json``) must stay
    bit-reproducible. ``HazardMovePruner(nav_actions_only=True)`` is the opt-in filter, and it is
    what the scored-path coercion below uses.
    """
    d: Any = label
    if isinstance(label, str):
        try:
            d = json.loads(label)
        except (ValueError, TypeError):
            return None
    if isinstance(d, dict) and "action" in d:
        d = d["action"]
    try:
        return int(d)
    except (ValueError, TypeError):
        return None


def _label_is_click(label: Any) -> bool:
    """True when a label denotes the POINTER action rather than a keyboard-nav move.

    Two independent signals, because the live label shapes are not uniform: the action id is 6, or
    the label carries pointer coordinates (``data.x``/``data.y``, or a bare ``{"x":..,"y":..}``).
    Either alone is sufficient -- a row that has coordinates is a click regardless of how its action
    id is spelled, and action 6 is a click even if its coordinates are elsewhere.
    """
    d: Any = label
    if isinstance(label, str):
        try:
            d = json.loads(label)
        except (ValueError, TypeError):
            return False
    if not isinstance(d, dict):
        return False
    if "x" in d or "y" in d:
        return True
    data = d.get("data")
    if isinstance(data, dict) and ("x" in data or "y" in data):
        return True
    try:
        return int(d.get("action")) == CLICK_ACTION_ID
    except (TypeError, ValueError):
        return False


class HazardMovePruner:
    """Accumulates the search's own (frame_before, action, frame_after) transitions, fits a hazard-aware
    nav model from them, and predicts which candidate moves remove the avatar so the search can skip them.

    Lifecycle the consumer (OfflineSolver) drives per expanded edge:
      * ``observe(frame_before, label, frame_after, leveled_up)`` -- after applying an action.
      * ``should_prune(frame, label) -> bool`` -- BEFORE applying, to skip a predicted-lethal move.
    """

    def __init__(
        self,
        grid_of: Callable[[Any], np.ndarray],
        *,
        action_of_label: Optional[Callable[[Any], Optional[int]]] = None,
        min_deaths: int = 3,
        refit_every: int = 50,
        min_trust: float = 0.9,
        min_specificity: float = 0.98,
        nav_actions_only: bool = False,
    ) -> None:
        self._grid_of = grid_of
        self._action_of = action_of_label or _default_action_of_label
        # DEFAULT FALSE so the offline dev twin's published tu93 A/B stays bit-reproducible; the
        # scored-path coercion opts IN. See `_default_action_of_label`'s CAVEAT for the measured
        # defect this filters: a live click label is `{"action": 6, "data": {...}}`, which decodes
        # to the int 6 and would otherwise be buffered as a keyboard-nav transition and pollute the
        # nav displacement fit the death predicate depends on.
        self.nav_actions_only = bool(nav_actions_only)
        self.clicks_skipped = 0
        self.min_deaths = min_deaths  # need this many observed deaths before trusting a fit
        self.refit_every = refit_every  # re-fit cadence (transitions) as more data arrives
        self.min_trust = min_trust  # balanced-accuracy bar to enable pruning
        self.min_specificity = (
            min_specificity  # FP bar -- a safe move wrongly pruned could break the solve
        )
        self._trans: List[Tuple[np.ndarray, int, np.ndarray, bool]] = []
        self._model: Optional[HazardAwareNavWorldModel] = None
        self.lethal_mode: Optional[str] = None
        self.trust: float = 0.0
        self.specificity: float = 0.0
        self.n_deaths: int = 0
        self._since_fit: int = 0
        self.pruned: int = 0
        self.observed: int = 0

    def _g2d(self, frame: Any) -> Optional[np.ndarray]:
        try:
            g = np.asarray(self._grid_of(frame))
        except Exception:
            return None
        if g.ndim == 1:  # some stepped frames flatten -> reshape square
            s = int(round(g.size**0.5))
            if s * s == g.size:
                g = g.reshape(s, s)
        return g if g.ndim == 2 else None

    def observe(
        self, frame_before: Any, label: Any, frame_after: Any, leveled_up: bool = False
    ) -> None:
        if self.nav_actions_only and _label_is_click(label):
            self.clicks_skipped += 1
            return
        a = self._action_of(label)
        if a is None:
            return
        g0, g1 = self._g2d(frame_before), self._g2d(frame_after)
        if g0 is None or g1 is None or g0.shape != g1.shape:
            return
        self._trans.append((g0, a, g1, bool(leveled_up)))
        self.observed += 1
        self._since_fit += 1
        if self._since_fit >= self.refit_every:
            self._fit()
            self._since_fit = 0

    def _death_labels(self, base: InducedNavWorldModel) -> List[bool]:
        """A transition is a DEATH when avatar cells present in g0 vanish in g1 (avatar removed), and it
        is NOT a level-up (the avatar also changes at a win). The avatar colours come from the nav fit."""
        av = list(base.avatar_colors)
        out: List[bool] = []
        for g0, _a, g1, lv in self._trans:
            had = bool(np.isin(g0, av).any())
            gone = not bool(np.isin(g1, av).any())
            out.append(had and gone and not lv)
        return out

    def _fit(self) -> None:
        if len(self._trans) < self.min_deaths + 5:
            return
        tr = [(g0, a, g1, 0, 1 if lv else 0) for (g0, a, g1, lv) in self._trans]
        try:
            base = InducedNavWorldModel.fit(tr)
        except Exception:
            return
        died = self._death_labels(base)
        self.n_deaths = sum(died)
        if self.n_deaths < self.min_deaths:
            return
        best: Optional[Tuple[float, float, str, HazardAwareNavWorldModel]] = None
        for mode in ("toward", "omni"):
            try:
                m = HazardAwareNavWorldModel.fit(tr, goal_color=base.goal_color, lethal_mode=mode)
            except Exception:
                continue
            if not m.hazard_colors:
                continue
            # IN-SAMPLE observed-transition trust: how well does this rung's is_lethal classify the
            # OBSERVED died/safe labels? (no offline ground-truth BFS -- only what the agent saw; scored on
            # the same transitions, so optimistic -- a conservative GATE, not a guarantee). Balanced
            # accuracy + specificity, so a rung that over-predicts death (would break the solve) is rejected.
            tp = fp = tn = fn = 0
            for (g0, a, _g1, _lv), d in zip(self._trans, died):
                pred = bool(m.is_lethal(g0, a))
                if d and pred:
                    tp += 1
                elif d and not pred:
                    fn += 1
                elif (not d) and pred:
                    fp += 1
                else:
                    tn += 1
            sens = tp / (tp + fn) if (tp + fn) else 0.0
            spec = tn / (tn + fp) if (tn + fp) else 1.0
            bal = 0.5 * (sens + spec)
            if best is None or bal > best[0]:
                best = (bal, spec, mode, m)
        if best and best[0] >= self.min_trust and best[1] >= self.min_specificity:
            self.trust, self.specificity, self.lethal_mode, self._model = best

    def should_prune(self, frame: Any, label: Any) -> bool:
        if self._model is None:
            return False
        if self.nav_actions_only and _label_is_click(label):
            return False
        a = self._action_of(label)
        if a is None:
            return False
        g = self._g2d(frame)
        if g is None:
            return False
        try:
            lethal = bool(self._model.is_lethal(g, a))
        except Exception:
            return False
        if lethal:
            self.pruned += 1
        return lethal

    def stats(self) -> dict:
        return {
            "observed": self.observed,
            "pruned": self.pruned,
            "n_deaths": self.n_deaths,
            "lethal_mode": self.lethal_mode,
            "trust": round(self.trust, 4),
            "specificity": round(self.specificity, 4),
            "model_fitted": self._model is not None,
            "transitions_buffered": len(self._trans),
            "nav_actions_only": self.nav_actions_only,
            "clicks_skipped": self.clicks_skipped,
            "min_deaths": self.min_deaths,
            "min_trust": self.min_trust,
            "min_specificity": self.min_specificity,
            "verifier_is_oracle": False,
        }


def coerce_hazard_move_pruner(value: Any) -> Optional[HazardMovePruner]:
    """``StepwiseExplorer``/``E3AgentPolicy``/``CarnotAgentPolicy`` constructor coercion, a verbatim
    mirror of ``arc_inert_click_pruner.coerce_inert_click_pruner``'s shape:
    ``None``/``False`` -> no pruner; an already-constructed instance -> passthrough;
    ``True`` -> build the default instance with the standard live ``grid_of``.

    WHY A COERCION HELPER RATHER THAN A DIRECT CONSTRUCTOR CALL. The scored agent's flags are
    booleans resolved through ``_fd_gate`` (explicit kwarg > env override > ``SUBMITTED_*``
    default), but a test or an A/B arm sometimes needs to inject a PRE-BUILT pruner (e.g. one with
    ``refit_every`` widened, or a stub that always prunes, to exercise the never-empty guard).
    Accepting both shapes in one place is what lets the flag ladder stay purely boolean while the
    injection path stays available -- exactly the pattern every sibling optional component in
    ``arc_competition_agent`` already uses.
    """

    if value is None or value is False:
        return None
    if isinstance(value, HazardMovePruner):
        return value
    if value is True:
        from carnot.agentic.arc_agi3_world_model import grid_of

        # nav_actions_only=True: the scored path's own click labels are
        # `{"action": 6, "data": {"x": .., "y": ..}}`, which `_default_action_of_label` decodes to
        # the int 6 and would otherwise buffer as a keyboard-nav transition. This is a DELIBERATE
        # divergence from the offline dev twin's default construction (which keeps the unfiltered
        # behaviour so its published tu93 A/B stays bit-reproducible); `stats()["nav_actions_only"]`
        # and `["clicks_skipped"]` make it visible in any artifact row.
        return HazardMovePruner(grid_of, nav_actions_only=True)
    return None

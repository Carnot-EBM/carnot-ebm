"""Online inert-click-signature move-pruner -- Reki's dead-signature mechanism, gated by
HazardMovePruner's trust+specificity discipline instead of Reki's own greedy K=2 threshold.

WHY THIS EXISTS (``ops/known-issues.md`` 2026-07-11 task 9, "New 2026-07-11, cheap, reuses
an existing code shape"). Reki's public segmentation write-up (the source read for
``REQ-ARC-FCP-5591``'s ``object_hash``/``blob_topology`` work) also describes a click-pruning
heuristic: track the structural signature of a clicked component -- ``(color, size, is_rect,
twin_count)`` -- and once a click on that signature has been observed twice with no effect on
the frame, suppress future clicks on components sharing that signature. The audit that
surfaced this (``docs/research-notes/arc-perception-grounding-audit-2026-07-13.md``) flagged
Reki's K=2 threshold as over-aggressive: two observations is a thin evidence floor, and a
strict "first effective observation ever = permanently sacred, otherwise K=2 kills it" rule
has no tolerance for a signature that is MOSTLY inert but occasionally does something (a
false suppress there would silently narrow the live agent's action space on a component that
actually mattered sometimes).

THE MECHANISM (mirrors ``arc_hazard_pruner.HazardMovePruner`` -- the live-path move-pruner
precedent, NOT ``arc_relational_mask_pruner.RelationalMaskMovePruner``'s binary
sacred-if-ever-touched rule, deliberately). Per structural signature, this pruner
accumulates observed (obs, inert, leveled) counts from the search's OWN clicks -- no offline
ground truth, so it transfers to a game it has never seen, exactly like both sibling
pruners. It differs from Reki's rule in two ways:

  * EVIDENCE FLOOR raised from 2 to ``min_observations`` (default 4, matching
    ``RelationalMaskMovePruner``'s own explicitly-not-K=2 default) before any signature is
    even considered for pruning.
  * SPECIFICITY THRESHOLD (``min_specificity``, default 0.9) replaces literal-zero-tolerance:
    a signature is pruned once its OBSERVED inert fraction clears the bar, not the instant a
    single effective click is seen. This tolerates a noisy or occasionally-effective
    signature without either permanently vetoing it (RelationalMaskMovePruner's rule) or
    suppressing it after two lucky-inert samples (Reki's rule).

A signature that has EVER produced a real level-up is PERMANENTLY sacred (never pruned,
regardless of specificity) -- this one rule is intentionally kept as a hard binary veto,
mirroring both sibling pruners, because a level-up is categorically too valuable to risk on
a statistical threshold.

Only click actions (action id 6, decoded ``{"x": ..., "y": ...}``) are covered; keyboard/nav
labels are left untouched (``should_prune`` returns ``False`` for them; ``observe`` no-ops).
The clicked component's signature is computed via ``connected_color_blobs``/``blob_at_click``
(``arc_color_blob_salience``, REQ-ARC-FCP-5591/5595) -- the same full-frame blob-decomposition
primitives that back ``object_hash``/``blob_topology``.

``verifier_is_oracle: False`` -- a learned inert-click predictor fit from observed
transitions, never the executable oracle that defines correctness. Consumer (OfflineSolver)
lifecycle per expanded edge, IDENTICAL protocol to both sibling pruners so it composes
through the existing ``arc_relational_mask_pruner.CompositeMovePruner``:
``should_prune(frame, label) -> bool`` BEFORE applying, then
``observe(frame_before, label, frame_after, leveled_up)`` after.

A SEPARATE ``rank_candidates(frame, rows) -> rows`` method implements the identical gating
logic in the filter-protocol shape ``StepwiseExplorer._candidates``
(``arc_competition_agent.py``) already composes with (matching
``program_synthesis_filter``/``goal_candidate_guidance``'s ``rank_candidates`` contract) --
DROPPING click rows whose signature is confidently inert.

WIRED 2026-07-13 (task 9 follow-on). ``coerce_inert_click_pruner`` (below) plugs an
``InertClickSigPruner`` into ``StepwiseExplorer`` (constructor param + a ``rank_candidates``
call inside ``_candidates``, alongside ``program_synthesis_filter``/``goal_candidate_guidance``)
and into ``E3AgentPolicy`` (same param, threaded through). The pruner also gets a real
``observe()`` call from ``_ingest``'s existing per-transition OBSERVE hook (the same site that
feeds ``dense_curiosity``/``controllable_novelty_policy``/``object_centric_proposal_policy``),
so it accumulates evidence from the search's OWN live clicks, exactly like its sibling
``HazardMovePruner``. Gated OFF by default
(``SUBMITTED_INERT_CLICK_PRUNER_ENABLED = False`` in ``arc_competition_agent.py``), matching
every other freshly-wired-but-unvalidated component in that file (``program_synthesis_filter``,
``object_centric_proposal``, ``amortized_first_contact_prior``, etc.) -- per the
``solve_rate_dropped`` guardrail (``docs/research-notes/trm-generator-hidden-game-plan-2026-07-04.md``
Stage 4), flipping the default to ``True`` for the SCORED agent needs its own matched-budget
offline A/B (states/actions-expanded reduction, zero regression in reproduced levels) before
being enabled, mirroring how ``HazardMovePruner``'s own tu93 A/B was measured before trust.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional, Sequence

import numpy as np

from carnot.agentic.arc_color_blob_salience import (
    ColorBlob,
    _cached_blobs_and_counts,
    blob_at_click,
)


def _default_action_of_label(label: Any) -> Optional[dict]:
    """Decode a label to {'action': int, 'data': dict|None}. Returns None for undecodable
    labels so the pruner cleanly no-ops on them (it never prunes what it cannot key) --
    identical decoding contract to ``arc_relational_mask_pruner``'s default."""

    d: Any = label
    if isinstance(label, str):
        try:
            d = json.loads(label)
        except (ValueError, TypeError):
            return None
    if isinstance(d, dict) and "action" in d:
        try:
            return {"action": int(d["action"]), "data": d.get("data")}
        except (ValueError, TypeError):
            return None
    try:
        return {"action": int(d), "data": None}
    except (ValueError, TypeError):
        return None


def _is_rect(blob: ColorBlob) -> bool:
    """A blob is a solid rectangle when its pixel count fills its own bounding box exactly
    -- cheap, no extra scan beyond the fields ``connected_color_blobs`` already computes."""

    return int(blob.pixel_count) == int(blob.height) * int(blob.width)


def click_signature(blob: ColorBlob, blobs: Sequence[ColorBlob]) -> tuple[int, int, bool, int]:
    """Reki's structural signature: ``(color, size, is_rect, twin_count)``. ``twin_count`` is
    the number of OTHER blobs in the same frame sharing this blob's ``(color, pixel_count,
    is_rect)`` triple -- decorative components that repeat (a row of identical dots, a grid
    of identical corner marks) share a signature even when scattered across different frame
    positions, so evidence about one informs the others."""

    is_rect = _is_rect(blob)
    twins = sum(
        1
        for other in blobs
        if other is not blob
        and int(other.color) == int(blob.color)
        and int(other.pixel_count) == int(blob.pixel_count)
        and _is_rect(other) == is_rect
    )
    return (int(blob.color), int(blob.pixel_count), bool(is_rect), int(twins))


class InertClickSigPruner:
    """Prunes clicks on structurally-inert component signatures. See module docstring for
    the full trust+specificity gating discipline this borrows from ``HazardMovePruner``."""

    verifier_is_oracle = False

    def __init__(
        self,
        grid_of: Callable[[Any], np.ndarray],
        *,
        action_of_label: Optional[Callable[[Any], Optional[dict]]] = None,
        min_observations: int = 4,
        min_specificity: float = 0.9,
    ) -> None:
        self._grid_of = grid_of
        self._action_of = action_of_label or _default_action_of_label
        self.min_observations = max(1, int(min_observations))
        self.min_specificity = float(min_specificity)
        # per-signature tally: sig -> {"obs": n, "inert": n_no_change_and_not_leveled, "leveled": n_levelups}
        self._tally: dict[tuple[int, int, bool, int], dict[str, int]] = {}
        self.pruned = 0
        self.observed = 0

    def _g2d(self, frame: Any) -> Optional[np.ndarray]:
        try:
            g = np.asarray(self._grid_of(frame))
        except Exception:
            return None
        if g.ndim == 3 and g.shape[0] == 1:
            g = g[0]
        return g if g.ndim == 2 else None

    def _decode_click(self, label: Any) -> Optional[tuple[int, int]]:
        decoded = self._action_of(label)
        if decoded is None or int(decoded.get("action", -1)) != 6:
            return None
        data = decoded.get("data") or {}
        if "x" not in data or "y" not in data:
            return None
        try:
            return int(data["x"]), int(data["y"])
        except (ValueError, TypeError):
            return None

    def _frame_blobs(self, grid: np.ndarray) -> list[ColorBlob]:
        """Full-frame blob decomposition, routed through the shared per-frame LRU cache
        (``_cached_blobs_and_counts``, arc_color_blob_salience.py) instead of a raw
        ``connected_color_blobs`` call. Behavior-preserving: the cache stores exactly
        ``connected_color_blobs(grid, min_pixels=1, max_component_fraction=1.0)``'s output.
        The int16 normalization matches the cache key ``ColorBlobSaliencePrior.score`` uses
        (it decomposes with the same ``min_pixels=1``/``max_component_fraction=1.0``), so a
        co-active blob prior and this pruner share one decomposition per frame rather than
        each recomputing the O(grid-cells) flood-fill (REQ-ARC-FCP-5699 cache discipline)."""

        g = np.asarray(grid)
        if g.dtype != np.int16:
            g = g.astype(np.int16, copy=False)
        blobs, _counts = _cached_blobs_and_counts(g, min_pixels=1, max_component_fraction=1.0)
        return blobs

    def _signature_for_click(
        self, grid: np.ndarray, x: int, y: int
    ) -> Optional[tuple[int, int, bool, int]]:
        blobs = self._frame_blobs(grid)
        blob = blob_at_click(blobs, x, y)
        if blob is None:
            return None
        return click_signature(blob, blobs)

    def observe(
        self, frame_before: Any, label: Any, frame_after: Any, leveled_up: bool = False
    ) -> None:
        click = self._decode_click(label)
        if click is None:
            return
        g0, g1 = self._g2d(frame_before), self._g2d(frame_after)
        if g0 is None or g1 is None:
            return
        sig = self._signature_for_click(g0, *click)
        if sig is None:
            return
        changed = g0.shape != g1.shape or bool((g0 != g1).any())
        t = self._tally.setdefault(sig, {"obs": 0, "inert": 0, "leveled": 0})
        t["obs"] += 1
        if leveled_up:
            t["leveled"] += 1
        elif not changed:
            t["inert"] += 1
        self.observed += 1

    def _should_prune_signature(self, sig: tuple[int, int, bool, int]) -> bool:
        t = self._tally.get(sig)
        if t is None or t["obs"] < self.min_observations:
            return False  # unproven signature -> never prune
        if t["leveled"] > 0:
            return False  # this signature has completed a level before -> SACRED, never prune
        specificity = t["inert"] / t["obs"] if t["obs"] else 0.0
        return bool(specificity >= self.min_specificity)

    def should_prune(self, frame: Any, label: Any) -> bool:
        click = self._decode_click(label)
        if click is None:
            return False
        g = self._g2d(frame)
        if g is None:
            return False
        sig = self._signature_for_click(g, *click)
        if sig is None:
            return False
        prune = self._should_prune_signature(sig)
        if prune:
            self.pruned += 1
        return prune

    def rank_candidates(self, frame: Any, rows: Sequence[dict]) -> list[dict]:
        """``StepwiseExplorer._candidates``-compatible filter: drop click rows whose
        signature is confidently inert, in the same ``rank_candidates(frame, rows) -> rows``
        shape ``program_synthesis_filter``/``goal_candidate_guidance`` already use. Not yet
        wired into the live composition chain -- see module docstring."""

        g = self._g2d(frame)
        if g is None:
            return list(rows)
        blobs = self._frame_blobs(g)
        kept: list[dict] = []
        for row in rows:
            if int(row.get("action", -1)) != 6:
                kept.append(row)
                continue
            data = row.get("data") or {}
            if "x" not in data or "y" not in data:
                kept.append(row)
                continue
            try:
                x, y = int(data["x"]), int(data["y"])
            except (ValueError, TypeError):
                kept.append(row)
                continue
            blob = blob_at_click(blobs, x, y)
            if blob is None:
                kept.append(row)
                continue
            sig = click_signature(blob, blobs)
            if self._should_prune_signature(sig):
                self.pruned += 1
                continue
            kept.append(row)
        return kept

    def stats(self) -> dict:
        return {
            "observed": self.observed,
            "pruned": self.pruned,
            "signatures_tracked": len(self._tally),
            "pruned_signatures": sum(1 for sig in self._tally if self._should_prune_signature(sig)),
            "min_observations": self.min_observations,
            "min_specificity": self.min_specificity,
            "verifier_is_oracle": False,
        }


def coerce_inert_click_pruner(value: Any) -> Optional[InertClickSigPruner]:
    """``StepwiseExplorer``/``E3AgentPolicy`` constructor coercion, matching
    ``coerce_program_synthesis_filter``/``coerce_amortized_first_contact_prior``'s shape:
    ``None``/``False`` -> no pruner; an already-constructed instance -> passthrough;
    ``True`` -> build the default instance with the standard live ``grid_of``."""

    if value is None or value is False:
        return None
    if isinstance(value, InertClickSigPruner):
        return value
    if value is True:
        from carnot.agentic.arc_agi3_world_model import grid_of

        return InertClickSigPruner(grid_of)
    return None

"""Object-identity-history-aware ARC action salience for live exploration.

Spec refs: REQ-ARC-FCP-5591-2, SCENARIO-ARC-FCP-5591-2.

Wraps ``ColorBlobSaliencePrior`` with a per-object-hash change-history bonus:
click candidates on objects (identified by ``object_hash`` -- REQ-ARC-FCP-5591,
translation-invariant, tracks an object's identity across frames) that have
previously been observed to change the frame when clicked are preferred over
untested or previously-inert objects.

This is the deliberately-deferred live-consuming mechanism ``ops/known-issues.md``
2026-07-11 task 10's DONE note named as its own suggested next step
("preferring an object whose hash was seen to change in a prior frame") but
explicitly did not build, per the Phase Prototype + Empirical Validation
discipline's "distinct, separately-scoped design step."

MECHANISM (mirrors ``arc_inert_click_pruner.InertClickSigPruner`` -- same
per-key evidence-floor discipline, INVERTED polarity, DIFFERENT key). Per
``object_hash``, this prior accumulates observed (obs, changed) counts from
the search's OWN clicks -- no offline ground truth, so it transfers to a game
it has never seen, exactly like InertClickSigPruner. Where InertClickSigPruner
PRUNES a structural signature once its inert fraction clears a specificity
bar, this prior BOOSTS an object-identity hash's score proportionally to its
observed change rate, once it clears ``min_observations`` (an untested or
under-observed object gets zero bonus, not a penalty -- unlike pruning,
under-evidence here should never suppress a candidate that might still be
worth trying).

``verifier_is_oracle: False`` -- a learned track-record bonus fit from
observed transitions, never the executable oracle that defines correctness.

WIRING (deliberately simpler than InertClickSigPruner's). ``action_prior`` is
already a generic, externally-composable slot on both ``StepwiseExplorer``
and ``E3AgentPolicy`` (see ``arc_geometric_salience.GeometricSaliencePrior``,
the existing precedent for wrapping ``ColorBlobSaliencePrior`` this way).
``_ingest``'s existing per-transition OBSERVE hook already calls
``action_prior.observe_transition(before, action_id, data, after)`` and
``action_prior.reset(reset_to_prior=...)`` generically whenever those methods
exist (``hasattr``-gated); ``_candidates``/``rich_action_candidates`` already
calls ``action_prior.score(frame, candidate)`` generically. Wrapping
``action_prior`` with this class therefore needs NO new hook sites in
``arc_competition_agent.py`` -- only a new ``object_history_salience``
constructor param on ``E3AgentPolicy`` that wraps whatever ``action_prior``
already resolved to, gated OFF by default
(``SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED = False``) pending its own
matched-budget offline A/B, per the ``solve_rate_dropped`` guardrail.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from carnot.agentic.arc_color_blob_salience import (
    ColorBlobSaliencePrior,
    _as_grid,
    _cached_blobs_and_counts,
    blob_at_click,
    connected_color_blobs,
    object_hash,
)


@dataclass
class ObjectHistorySaliencePrior:
    """Live action prior that boosts click candidates on objects with an
    observed track record of changing the frame, keyed on translation-invariant
    ``object_hash`` identity rather than raw position."""

    base_prior: ColorBlobSaliencePrior = field(default_factory=ColorBlobSaliencePrior)
    change_bonus_weight: float = 10.0
    min_observations: int = 3
    enabled: bool = True
    source: str = "object_hash_change_history_blob_salience"
    _tally: dict[str, dict[str, int]] = field(default_factory=dict, init=False)

    verifier_is_oracle = False

    @property
    def tracked_hash_count(self) -> int:
        return len(self._tally)

    def for_path(self, _path: list[Mapping[str, Any]]) -> ObjectHistorySaliencePrior:
        """Return the live prior for a graph path without cloning tally memory."""

        return self

    def observe_transition(
        self,
        before: Any,
        action_id: int,
        data: Mapping[str, Any] | None,
        after: Any,
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        """Update the per-hash tally from the agent's own observed click transition."""

        if int(action_id or 0) != 6:
            return
        payload = data or {}
        if "x" not in payload or "y" not in payload:
            return
        try:
            before_grid = _as_grid(before)
            after_grid = _as_grid(after)
        except Exception:
            return
        try:
            x, y = int(payload["x"]), int(payload["y"])
        except (TypeError, ValueError):
            return
        blobs = connected_color_blobs(before_grid, min_pixels=1, max_component_fraction=1.0)
        blob = blob_at_click(blobs, x, y)
        if blob is None:
            return
        h = object_hash(blob)
        changed = before_grid.shape != after_grid.shape or bool((before_grid != after_grid).any())
        tally = self._tally.setdefault(h, {"obs": 0, "changed": 0})
        tally["obs"] += 1
        if changed:
            tally["changed"] += 1

    def reset(self, *_args: Any, reset_to_prior: bool = False, **_kwargs: Any) -> None:
        """Clear the tally when the live policy intentionally resets levels -- a new
        level's visual context makes prior hashes' change history stale."""

        if reset_to_prior:
            self._tally.clear()

    def _change_rate(self, h: str) -> float:
        tally = self._tally.get(h)
        if tally is None or tally["obs"] < self.min_observations:
            return 0.0
        return tally["changed"] / float(tally["obs"])

    def score(self, frame: Any, candidate: Any) -> float:
        """Score one live candidate, with higher values tried earlier."""

        base = float(self.base_prior.score(frame, candidate))
        if not self.enabled or self._candidate_action_id(candidate) != 6:
            return base
        data = self._candidate_data(candidate)
        if "x" not in data or "y" not in data:
            return base
        try:
            grid = _as_grid(frame)
            x, y = int(data["x"]), int(data["y"])
        except Exception:
            return base
        # Route the per-candidate blob decomposition through the SAME module-level
        # per-frame cache ColorBlobSaliencePrior.score() uses (REQ-ARC-FCP-5699 item-2,
        # 2026-07-16). ``self.base_prior.score(frame, candidate)`` above already warmed
        # that cache for this exact (grid, min_pixels=1, max_component_fraction=1.0) key
        # -- these params match ColorBlobSaliencePrior's default cached call
        # (large_flat_deprioritization=True -> max_component_fraction=1.0, min_pixels=1) --
        # so this is a cache HIT, not the raw per-candidate flood-fill that the 8176-calls-
        # for-500-actions profiling eliminated everywhere else. Identical blobs, no re-run.
        blobs, _ = _cached_blobs_and_counts(grid, min_pixels=1, max_component_fraction=1.0)
        blob = blob_at_click(blobs, x, y)
        if blob is None:
            return base
        h = object_hash(blob)
        return float(base + float(self.change_bonus_weight) * self._change_rate(h))

    def diagnostics(self) -> dict[str, Any]:
        return self.as_dict()

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "base_prior": self.base_prior.as_dict(),
            "enabled": bool(self.enabled),
            "change_bonus_weight": float(self.change_bonus_weight),
            "min_observations": int(self.min_observations),
            "tracked_hash_count": int(self.tracked_hash_count),
            "verifier_is_oracle": False,
        }

    @staticmethod
    def _candidate_action_id(candidate: Any) -> int:
        value = (
            candidate.get("action", candidate.get("action_id", 0))
            if isinstance(candidate, Mapping)
            else getattr(candidate, "action_id", 0)
        )
        return int(value or 0)

    @staticmethod
    def _candidate_data(candidate: Any) -> Mapping[str, Any]:
        data = (
            candidate.get("data")
            if isinstance(candidate, Mapping)
            else getattr(candidate, "data", None)
        )
        return data if isinstance(data, Mapping) else {}


def coerce_object_history_salience_prior(value: Any, *, base_prior: Any | None = None) -> Any:
    """``E3AgentPolicy`` action_prior-wrapping coercion: ``None``/``False`` ->
    ``base_prior`` unchanged (no wrapping, a true no-op); an already-constructed
    ``ObjectHistorySaliencePrior`` -> passthrough (ignores ``base_prior``); ``True`` ->
    wrap ``base_prior`` (or a fresh ``ColorBlobSaliencePrior`` if ``base_prior`` is
    ``None``)."""

    if value is None or value is False:
        return base_prior
    if isinstance(value, ObjectHistorySaliencePrior):
        return value
    if value is True:
        return ObjectHistorySaliencePrior(base_prior=base_prior or ColorBlobSaliencePrior())
    return base_prior

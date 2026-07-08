"""Live trajectory-derived frontier prefixes for ARC exploration.

Spec refs: REQ-ARC-FCP-5410, SCENARIO-ARC-FCP-5410.

The live ARC agent needs candidate generation that is still honest when the
game is unseen. This helper only learns from transitions the agent just caused:
it records action -> visible frame-change evidence, combines that support with
the existing color-blob salience prior, and emits short action prefixes only
when the support and uncertainty gate agree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from carnot.agentic.arc_color_blob_salience import ColorBlobSaliencePrior


ActionStep = dict[str, Any]


def _grid(frame: Any) -> np.ndarray:
    value = frame.frame if hasattr(frame, "frame") else frame
    arr = np.asarray(value)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        return np.zeros((1, 1), dtype=np.int16)
    return arr.astype(np.int16, copy=False)


def _candidate_action(candidate: Any) -> int:
    if isinstance(candidate, Mapping):
        return int(candidate.get("action", candidate.get("action_id", 0)) or 0)
    return int(getattr(candidate, "action", getattr(candidate, "action_id", 0)) or 0)


def _candidate_data(candidate: Any) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get("data")
    return getattr(candidate, "data", None)


def _step(action: int, data: Any) -> ActionStep:
    return {"action": int(action), "data": dict(data) if isinstance(data, Mapping) else data}


def _signature(action: int, data: Any) -> tuple[Any, ...]:
    if isinstance(data, Mapping):
        return (int(action), tuple(sorted((str(key), data[key]) for key in data)))
    return (int(action), data)


def _candidate_signature(candidate: Any) -> tuple[Any, ...]:
    return _signature(_candidate_action(candidate), _candidate_data(candidate))


@dataclass(frozen=True)
class FrontierGateRow:
    """Auditable support row for one inferred live action dynamic."""

    action: int
    data: Any
    support_count: int
    effect_count: int
    effect_rate: float
    uncertainty: float
    accepted: bool
    salience_route: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": int(self.action),
            "data": dict(self.data) if isinstance(self.data, Mapping) else self.data,
            "support_count": int(self.support_count),
            "effect_count": int(self.effect_count),
            "effect_rate": float(self.effect_rate),
            "uncertainty": float(self.uncertainty),
            "accepted": bool(self.accepted),
            "salience_route": str(self.salience_route),
        }


class LiveTrajectoryFrontierGenerator:
    """Blob-salience action prior plus uncertainty-gated prefix generator.

    The object intentionally satisfies two live-path protocols: as an
    ``action_prior`` it supplies blob-ordered click points and observes realized
    transitions; as a ``qd_generator`` it can return a short prefix through the
    existing ``StepwiseExplorer`` sequence-injection hook.
    """

    verifier_is_oracle = False

    def __init__(
        self,
        *,
        base_prior: ColorBlobSaliencePrior | None = None,
        min_support: int = 2,
        max_uncertainty: float = 0.4,
        min_effect_rate: float = 0.5,
        min_changed_cells: int = 1,
    ) -> None:
        self.base_prior = base_prior or ColorBlobSaliencePrior()
        self.min_support = max(1, int(min_support))
        self.max_uncertainty = max(0.0, min(1.0, float(max_uncertainty)))
        self.min_effect_rate = max(0.0, min(1.0, float(min_effect_rate)))
        self.min_changed_cells = max(1, int(min_changed_cells))
        self._stats: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._frontier_expansions: list[dict[str, Any]] = []
        self._uncertainty_rejections = 0
        self._salience_routes_used: list[str] = []

    def click_points(self, frame: Any, *, max_points: int | None = None) -> list[tuple[int, int]]:
        return self.base_prior.click_points(frame, max_points=max_points)

    def tier_rows(self, frame: Any) -> list[dict[str, Any]]:
        return self.base_prior.tier_rows(frame)

    def action_tier_rows(self, frame: Any, candidates: Sequence[Any]) -> list[dict[str, Any]]:
        return self.base_prior.action_tier_rows(frame, candidates)

    def score(self, frame: Any, candidate: Any) -> float:
        base = float(self.base_prior.score(frame, candidate))
        row = self._gate_row(_candidate_action(candidate), _candidate_data(candidate), frame)
        return base + (100.0 * (1.0 - row.uncertainty) if row.accepted else 0.0)

    def observe_transition(self, before: Any, action: int, data: Any, after: Any) -> None:
        before_grid = _grid(before)
        after_grid = _grid(after)
        changed = 0
        if before_grid.shape == after_grid.shape:
            changed = int(np.count_nonzero(before_grid != after_grid))
        route = self._salience_route(before, int(action), data)
        key = _signature(int(action), data)
        row = self._stats.setdefault(
            key,
            {
                "action": int(action),
                "data": dict(data) if isinstance(data, Mapping) else data,
                "support_count": 0,
                "effect_count": 0,
                "salience_route": route,
            },
        )
        row["support_count"] = int(row["support_count"]) + 1
        if changed >= self.min_changed_cells:
            row["effect_count"] = int(row["effect_count"]) + 1
        row["last_changed_cells"] = int(changed)
        row["salience_route"] = route

    def best_sequence(
        self,
        frame: Any,
        candidates: Sequence[Any],
        *,
        goal_energy: Any | None = None,
        action_effect_scorer: Any | None = None,
        min_len: int = 2,
    ) -> tuple[ActionStep, ...]:
        del goal_energy, action_effect_scorer
        rows = [self._gate_row(_candidate_action(c), _candidate_data(c), frame) for c in candidates]
        accepted = [row for row in rows if row.accepted]
        if not accepted:
            self._uncertainty_rejections += 1
            return tuple()
        accepted.sort(key=lambda row: (row.uncertainty, -row.effect_rate, row.salience_route))
        first = accepted[0]
        sequence = [_step(first.action, first.data)]
        seen = {_signature(first.action, first.data)}
        for candidate in candidates:
            sig = _candidate_signature(candidate)
            if sig in seen:
                continue
            sequence.append(_step(_candidate_action(candidate), _candidate_data(candidate)))
            seen.add(sig)
            if len(sequence) >= int(min_len):
                break
        while len(sequence) < int(min_len):
            sequence.append(dict(sequence[0]))
        route = first.salience_route
        if route not in self._salience_routes_used:
            self._salience_routes_used.append(route)
        expansion = {
            "prefix": [dict(step) for step in sequence],
            "support_count": int(first.support_count),
            "uncertainty": float(first.uncertainty),
            "salience_route": route,
            "accepted": True,
        }
        self._frontier_expansions.append(expansion)
        return tuple(dict(step) for step in sequence)

    def diagnostics(self) -> dict[str, Any]:
        observations = [
            self._gate_row(int(row["action"]), row.get("data"), None).as_dict()
            for row in self._stats.values()
        ]
        observations.sort(key=lambda row: (row["action"], repr(row["data"])))
        return {
            "enabled": True,
            "source": "live_trajectory_frontier_blob_salience_uncertainty_gate",
            "frontier_expansion_count": int(len(self._frontier_expansions)),
            "frontier_expansions": [dict(row) for row in self._frontier_expansions],
            "salience_routes_used": list(self._salience_routes_used),
            "uncertainty_rejections": int(self._uncertainty_rejections),
            "verifier_observations": observations,
            "observed_transition_count": int(
                sum(int(row["support_count"]) for row in self._stats.values())
            ),
            "verifier_is_oracle": False,
        }

    def as_dict(self) -> dict[str, Any]:
        base = self.base_prior.as_dict()
        base.update(
            {
                "source": "live_trajectory_frontier_blob_salience_uncertainty_gate",
                "trajectory_frontier_generation_enabled": True,
                "uncertainty_gate_enabled": True,
                "min_support": int(self.min_support),
                "max_uncertainty": float(self.max_uncertainty),
            }
        )
        return base

    def _salience_route(self, frame: Any, action: int, data: Any) -> str:
        if int(action) != 6:
            return "keyboard_action"
        try:
            rows = self.base_prior.action_tier_rows(
                frame,
                [{"action": int(action), "data": dict(data) if isinstance(data, Mapping) else data}],
            )
        except Exception:
            rows = []
        row = rows[0] if rows else {}
        tier = row.get("tier")
        if tier is None:
            return "blob_tier_unknown"
        if row.get("button_like"):
            return f"blob_tier_{int(tier)}_button_like"
        if row.get("status_bar"):
            return f"blob_tier_{int(tier)}_status"
        if row.get("large_flat"):
            return f"blob_tier_{int(tier)}_large_flat"
        return f"blob_tier_{int(tier)}_color"

    def _gate_row(self, action: int, data: Any, frame: Any | None) -> FrontierGateRow:
        key = _signature(int(action), data)
        stat = self._stats.get(key)
        if stat is None:
            route = "unobserved"
            if frame is not None:
                route = self._salience_route(frame, int(action), data)
            return FrontierGateRow(
                action=int(action),
                data=dict(data) if isinstance(data, Mapping) else data,
                support_count=0,
                effect_count=0,
                effect_rate=0.0,
                uncertainty=1.0,
                accepted=False,
                salience_route=route,
            )
        support = int(stat["support_count"])
        effect = int(stat["effect_count"])
        effect_rate = float(effect) / float(max(1, support))
        uncertainty = min(1.0, (1.0 / float(support + 1)) + ((1.0 - effect_rate) * 0.5))
        accepted = bool(
            support >= self.min_support
            and effect_rate >= self.min_effect_rate
            and uncertainty <= self.max_uncertainty
        )
        return FrontierGateRow(
            action=int(action),
            data=stat.get("data"),
            support_count=support,
            effect_count=effect,
            effect_rate=effect_rate,
            uncertainty=uncertainty,
            accepted=accepted,
            salience_route=str(stat.get("salience_route") or "observed_unknown"),
        )

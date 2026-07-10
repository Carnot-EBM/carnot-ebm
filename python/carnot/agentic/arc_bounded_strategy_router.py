"""Bounded ARC action-strategy routing for the live candidate hook.

The submitted ARC live path already accepts candidate-router objects through
``rank(frame, candidates, previous_frame=...)``. This helper keeps that exact
interface while adding a small portfolio of deterministic strategies and a
pre-selection repeated-coordinate guard. The guard matters because repeated
coordinates must be removed before the live path chooses the next action, not
afterward in a metric summary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_BOUNDED_ACTION_STRATEGIES: tuple[dict[str, Any], ...] = (
    {
        "name": "salience_first",
        "score_field": "salience_score",
        "bound": 1,
        "live_path_hook": "candidate_router.rank",
        "principle": "try visually salient live candidates first without spending the whole budget on one coordinate.",
    },
    {
        "name": "action_effect_memory",
        "score_field": "effect_score",
        "bound": 1,
        "live_path_hook": "candidate_router.rank",
        "principle": "prefer candidates with observed or cached action-effect evidence.",
    },
    {
        "name": "verifier_router_candidate_ranking",
        "score_field": "verifier_score",
        "bound": 1,
        "live_path_hook": "candidate_router.rank",
        "principle": "rank compatible candidates by verifier-style confidence with stable fallback order.",
    },
    {
        "name": "conservative_reset_reinduction",
        "score_field": "reset_score",
        "bound": 1,
        "live_path_hook": "candidate_router.rank",
        "principle": "reserve a small budget for reset or reinduction-safe candidates after local evidence stalls.",
    },
)


def _candidate_data(candidate: Any) -> Mapping[str, Any]:
    data = candidate.get("data") if isinstance(candidate, Mapping) else getattr(candidate, "data", None)
    return data if isinstance(data, Mapping) else {}


def _candidate_action(candidate: Any) -> int:
    if isinstance(candidate, Mapping):
        value = candidate.get("action", candidate.get("action_id", 0))
    else:
        value = getattr(candidate, "action", getattr(candidate, "action_id", 0))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _candidate_coordinate(candidate: Any) -> tuple[int, int] | None:
    data = _candidate_data(candidate)
    if "x" in data and "y" in data:
        try:
            return int(data["x"]), int(data["y"])
        except (TypeError, ValueError):
            return None
    if isinstance(candidate, Mapping) and "x" in candidate and "y" in candidate:
        try:
            return int(candidate["x"]), int(candidate["y"])
        except (TypeError, ValueError):
            return None
    return None


def _candidate_score(candidate: Any, field: str) -> float:
    if isinstance(candidate, Mapping):
        value = candidate.get(field, candidate.get("score", 0.0))
    else:
        value = getattr(candidate, field, getattr(candidate, "score", 0.0))
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _candidate_signature(candidate: Any) -> str:
    coord = _candidate_coordinate(candidate)
    if coord is not None:
        return f"A{_candidate_action(candidate)}@{coord[0]},{coord[1]}"
    data = _candidate_data(candidate)
    if data:
        payload = ",".join(f"{key}={data[key]}" for key in sorted(data))
        return f"A{_candidate_action(candidate)}@{payload}"
    return f"A{_candidate_action(candidate)}"


class BoundedStrategyCandidateRouter:
    """Live-compatible candidate router for a small strategy portfolio."""

    def __init__(
        self,
        *,
        strategies: Sequence[Mapping[str, Any]] | None = None,
        max_candidates: int = 8,
        per_strategy_limit: int = 1,
        suppress_repeated_coordinates: bool = True,
        avoid_coordinates: set[tuple[int, int]] | frozenset[tuple[int, int]] = frozenset(),
    ) -> None:
        self.strategy_portfolio = [dict(row) for row in (strategies or DEFAULT_BOUNDED_ACTION_STRATEGIES)]
        self.max_candidates = max(1, int(max_candidates))
        self.per_strategy_limit = max(1, int(per_strategy_limit))
        self.suppress_repeated_coordinates = bool(suppress_repeated_coordinates)
        self.avoid_coordinates = {tuple(map(int, point)) for point in avoid_coordinates}
        self.last_diagnostics: dict[str, Any] = {}

    def portfolio_descriptors(self) -> list[dict[str, Any]]:
        """Return auditable strategy descriptors with normalized bounds."""

        descriptors: list[dict[str, Any]] = []
        for strategy in self.strategy_portfolio:
            row = dict(strategy)
            row["bound"] = int(max(1, int(row.get("bound", self.per_strategy_limit))))
            descriptors.append(row)
        return descriptors

    def rank(
        self,
        frame: Any,
        candidates: Sequence[Any],
        *,
        previous_frame: Any | None = None,
    ) -> list[Any]:
        """Rank candidates through bounded strategies for the live candidate hook."""

        del frame, previous_frame
        raw = self._rank_core(candidates, suppress=False)
        ranked = self._rank_core(candidates, suppress=self.suppress_repeated_coordinates)
        raw_signatures = [_candidate_signature(candidate) for candidate in raw["ranked"]]
        ranked_signatures = [_candidate_signature(candidate) for candidate in ranked["ranked"]]
        diagnostics = dict(ranked["diagnostics"])
        diagnostics["selection_changed_by_suppression"] = bool(
            self.suppress_repeated_coordinates and raw_signatures != ranked_signatures
        )
        diagnostics["unsuppressed_signatures"] = raw_signatures
        diagnostics["selected_signatures"] = ranked_signatures
        diagnostics["strategy_portfolio"] = self.portfolio_descriptors()
        self.last_diagnostics = diagnostics
        return list(ranked["ranked"])

    def _rank_core(self, candidates: Sequence[Any], *, suppress: bool) -> dict[str, Any]:
        ordered_candidates = list(candidates)
        selected: list[Any] = []
        selected_ids: set[int] = set()
        seen_coordinates: set[tuple[int, int]] = set()
        suppressed_count = 0
        strategies_used: list[str] = []

        def _add_candidate(candidate: Any, strategy_name: str) -> bool:
            nonlocal suppressed_count
            if id(candidate) in selected_ids:
                return False
            coord = _candidate_coordinate(candidate)
            if suppress and coord is not None:
                if coord in self.avoid_coordinates or coord in seen_coordinates:
                    suppressed_count += 1
                    return False
            selected.append(candidate)
            selected_ids.add(id(candidate))
            if coord is not None:
                seen_coordinates.add(coord)
            if strategy_name and strategy_name not in strategies_used:
                strategies_used.append(strategy_name)
            return True

        for strategy in self.portfolio_descriptors():
            name = str(strategy.get("name") or "unnamed_strategy")
            field = str(strategy.get("score_field") or "score")
            limit = int(max(1, min(self.per_strategy_limit, int(strategy.get("bound", 1)))))
            strategy_order = sorted(
                ordered_candidates,
                key=lambda candidate: (
                    -_candidate_score(candidate, field),
                    _candidate_signature(candidate),
                ),
            )
            picked = 0
            for candidate in strategy_order:
                if _add_candidate(candidate, name):
                    picked += 1
                if picked >= limit or len(selected) >= self.max_candidates:
                    break
            if len(selected) >= self.max_candidates:
                break

        if len(selected) < self.max_candidates:
            score_fields = [
                str(strategy.get("score_field") or "score")
                for strategy in (self.strategy_portfolio or DEFAULT_BOUNDED_ACTION_STRATEGIES)
            ]
            fallback_order = sorted(
                ordered_candidates,
                key=lambda candidate: (
                    -max(_candidate_score(candidate, field) for field in score_fields),
                    _candidate_signature(candidate),
                ),
            )
            for candidate in fallback_order:
                _add_candidate(candidate, "fallback_fill")
                if len(selected) >= self.max_candidates:
                    break

        return {
            "ranked": selected[: self.max_candidates],
            "diagnostics": {
                "suppression_enabled": bool(suppress),
                "suppressed_coordinate_count": int(suppressed_count),
                "selected_strategy_count": int(len([s for s in strategies_used if s != "fallback_fill"])),
                "strategies_used": strategies_used,
                "avoid_coordinates": [
                    {"x": x, "y": y} for x, y in sorted(self.avoid_coordinates)
                ],
            },
        }

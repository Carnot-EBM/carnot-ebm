"""Amortized first-contact exploration priors for the live ARC explorer.

Spec refs: REQ-ARC-WMTE-4701, SCENARIO-ARC-WMTE-4701-LIVE-WIRING.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def _action_id(row: Any) -> int:
    if isinstance(row, Mapping):
        return int(row.get("action", row.get("action_id", 0)) or 0)
    return int(getattr(row, "action", getattr(row, "action_id", 0)) or 0)


def _action_data(row: Any) -> Any:
    if isinstance(row, Mapping):
        return row.get("data")
    return getattr(row, "data", None)


def first_contact_family(row: Any) -> str:
    """Reusable action-family key: action type only, never a game id or raw coordinate."""

    action = _action_id(row)
    if action == 6:
        return "click"
    return f"action:{action}"


def _trace_weight(trace: Mapping[str, Any]) -> float:
    outcome = str(trace.get("outcome") or trace.get("status") or "").lower()
    if "success" in outcome or trace.get("reached_level"):
        return 1.0
    if "near" in outcome or "miss" in outcome:
        return 0.5
    return 0.25


@dataclass(frozen=True)
class PriorScore:
    family: str
    score: float


class AmortizedFirstContactPrior:
    """Frequency fallback for cross-game first-contact action traces."""

    def __init__(
        self,
        step_family_scores: Mapping[int, Mapping[str, float]] | None = None,
        *,
        trace_count: int = 0,
        max_depth: int = 3,
    ) -> None:
        self.step_family_scores = {
            int(depth): {str(key): float(value) for key, value in scores.items()}
            for depth, scores in (step_family_scores or {}).items()
        }
        self.trace_count = int(trace_count)
        self.max_depth = max(1, int(max_depth))
        self._rank_calls = 0

    @classmethod
    def from_traces(
        cls,
        traces: Sequence[Mapping[str, Any]],
        *,
        max_depth: int = 3,
    ) -> "AmortizedFirstContactPrior":
        scores: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        trace_count = 0
        for trace in traces:
            steps = trace.get("steps") or trace.get("actions") or trace.get("solution") or []
            if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
                continue
            if not steps:
                continue
            trace_count += 1
            weight = _trace_weight(trace)
            for depth, step in enumerate(list(steps)[: max(1, int(max_depth))]):
                scores[int(depth)][first_contact_family(step)] += weight
        return cls(scores, trace_count=trace_count, max_depth=max_depth)

    def score_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        path: Sequence[Mapping[str, Any]] | None = None,
    ) -> PriorScore:
        depth = min(len(path or []), self.max_depth - 1)
        family = first_contact_family(candidate)
        return PriorScore(
            family=family, score=float(self.step_family_scores.get(depth, {}).get(family, 0.0))
        )

    def rank_candidates(
        self,
        _frame: Any,
        candidates: Sequence[Mapping[str, Any]],
        *,
        path: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        self._rank_calls += 1
        ranked = []
        for index, candidate in enumerate(candidates):
            row = dict(candidate)
            score = self.score_candidate(row, path=path)
            row["amortized_prior_family"] = score.family
            row["amortized_prior_score"] = round(float(score.score), 6)
            ranked.append((index, row))
        ranked.sort(
            key=lambda item: (
                -float(item[1].get("amortized_prior_score") or 0.0),
                item[0],
            )
        )
        return [row for _index, row in ranked]

    def diagnostics(self) -> dict[str, Any]:
        keys = sorted(
            {
                f"step{depth}:{family}"
                for depth, scores in self.step_family_scores.items()
                for family in scores
            }
        )
        return {
            "enabled": bool(self.step_family_scores),
            "distillation_mode": "frequency_prior",
            "trace_count": int(self.trace_count),
            "max_depth": int(self.max_depth),
            "rank_calls": int(self._rank_calls),
            "learned_family_keys": keys,
            "verifier_is_oracle": False,
        }


def coerce_amortized_first_contact_prior(value: Any) -> AmortizedFirstContactPrior | None:
    if value is None or value is False:
        return None
    if isinstance(value, AmortizedFirstContactPrior):
        return value
    if isinstance(value, Mapping):
        traces = value.get("traces") or value.get("trace_rows") or []
        return AmortizedFirstContactPrior.from_traces(
            list(traces),
            max_depth=int(value.get("max_depth", 3)),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return AmortizedFirstContactPrior.from_traces(list(value))
    if value is True:
        return AmortizedFirstContactPrior.from_traces([])
    return None


def traces_from_solutions(
    solutions: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    exclude_game: str | None = None,
    max_steps: int = 3,
) -> list[dict[str, Any]]:
    traces = []
    excluded = str(exclude_game or "")
    for game, steps in sorted(solutions.items()):
        if excluded and str(game).split("-", 1)[0] == excluded.split("-", 1)[0]:
            continue
        prefix = [dict(step) for step in list(steps)[: max(1, int(max_steps))]]
        if prefix:
            traces.append({"game_id": str(game), "outcome": "success", "steps": prefix})
    return traces

"""Amortized first-contact exploration priors for the live ARC explorer.

Spec refs: REQ-ARC-WMTE-4701, SCENARIO-ARC-WMTE-4701-LIVE-WIRING,
REQ-ARC-WMTE-4831, SCENARIO-ARC-WMTE-4831-IN-CONTEXT-PRIOR.
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


def _path_context(
    path: Sequence[Mapping[str, Any]] | None,
    *,
    max_context: int,
) -> tuple[str, ...]:
    families = [
        first_contact_family(step)
        for step in list(path or [])
        if isinstance(step, Mapping) and step.get("action") is not None
    ]
    return tuple(families[-max(0, int(max_context)) :])


def _context_label(context: Sequence[str]) -> str:
    return "ROOT" if not context else ">".join(str(item) for item in context)


class AmortizedInContextExplorationPrior:
    """REQ-ARC-WMTE-4831: prefix-conditioned prior over reusable action families.

    The learned keys are action-family prefixes such as ``action:2`` or
    ``action:2>click``. Game IDs and raw click coordinates are intentionally not
    part of the feature space, so the prior can bias fresh-game exploration
    without memorizing a public-game label.
    """

    def __init__(
        self,
        context_family_scores: Mapping[Sequence[str], Mapping[str, float]] | None = None,
        *,
        trace_count: int = 0,
        max_context: int = 3,
    ) -> None:
        self.context_family_scores: dict[tuple[str, ...], dict[str, float]] = {
            tuple(str(part) for part in context): {
                str(family): float(score) for family, score in scores.items()
            }
            for context, scores in (context_family_scores or {}).items()
        }
        self.trace_count = int(trace_count)
        self.max_context = max(1, int(max_context))
        self._rank_calls = 0
        self._context_hits = 0
        self._context_misses = 0
        self._proposal_changes = 0
        self._contexts_used: dict[str, int] = defaultdict(int)

    @classmethod
    def from_traces(
        cls,
        traces: Sequence[Mapping[str, Any]],
        *,
        max_context: int = 3,
        max_depth: int | None = None,
    ) -> "AmortizedInContextExplorationPrior":
        scores: dict[tuple[str, ...], dict[str, float]] = defaultdict(lambda: defaultdict(float))
        trace_count = 0
        for trace in traces:
            steps = trace.get("steps") or trace.get("actions") or trace.get("solution") or []
            if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
                continue
            steps = [step for step in steps if isinstance(step, Mapping)]
            if not steps:
                continue
            trace_count += 1
            families = [first_contact_family(step) for step in steps]
            limit = min(len(families), int(max_depth or len(families)))
            weight = _trace_weight(trace)
            for depth, family in enumerate(families[:limit]):
                context = tuple(families[max(0, depth - int(max_context)) : depth])
                scores[context][family] += weight
        return cls(scores, trace_count=trace_count, max_context=max_context)

    def _scores_for_path(
        self,
        path: Sequence[Mapping[str, Any]] | None,
    ) -> tuple[dict[str, float], tuple[str, ...], bool]:
        context = _path_context(path, max_context=self.max_context)
        if context in self.context_family_scores:
            return self.context_family_scores[context], context, True
        for offset in range(1, len(context)):
            suffix = context[offset:]
            if suffix in self.context_family_scores:
                return self.context_family_scores[suffix], suffix, True
        root = ()
        if root in self.context_family_scores:
            return self.context_family_scores[root], root, False
        return {}, context, False

    def score_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        path: Sequence[Mapping[str, Any]] | None = None,
    ) -> PriorScore:
        scores, _context, _hit = self._scores_for_path(path)
        family = first_contact_family(candidate)
        return PriorScore(family=family, score=float(scores.get(family, 0.0)))

    def rank_candidates(
        self,
        _frame: Any,
        candidates: Sequence[Mapping[str, Any]],
        *,
        path: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        self._rank_calls += 1
        scores, context, hit = self._scores_for_path(path)
        if hit:
            self._context_hits += 1
        else:
            self._context_misses += 1
        context_name = _context_label(context)
        self._contexts_used[context_name] += 1
        ranked = []
        original_keys = []
        for index, candidate in enumerate(candidates):
            row = dict(candidate)
            family = first_contact_family(row)
            row["amortized_prior_family"] = family
            row["amortized_prior_context"] = context_name
            row["amortized_prior_score"] = round(float(scores.get(family, 0.0)), 6)
            ranked.append((index, row))
            original_keys.append((row.get("action"), repr(row.get("data"))))
        ranked.sort(
            key=lambda item: (
                -float(item[1].get("amortized_prior_score") or 0.0),
                item[0],
            )
        )
        ordered = [row for _index, row in ranked]
        ordered_keys = [(row.get("action"), repr(row.get("data"))) for row in ordered]
        if ordered_keys != original_keys:
            self._proposal_changes += 1
        return ordered

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.context_family_scores),
            "distillation_mode": "in_context_exploration_prior",
            "trace_count": int(self.trace_count),
            "max_context": int(self.max_context),
            "rank_calls": int(self._rank_calls),
            "context_hits": int(self._context_hits),
            "context_misses": int(self._context_misses),
            "proposal_changes": int(self._proposal_changes),
            "learned_context_keys": sorted(
                _context_label(context) for context in self.context_family_scores
            ),
            "contexts_used": dict(sorted(self._contexts_used.items())),
            "game_id_features_used": False,
            "verifier_is_oracle": False,
        }


def coerce_amortized_first_contact_prior(value: Any) -> AmortizedFirstContactPrior | None:
    if value is None or value is False:
        return None
    if isinstance(value, (AmortizedFirstContactPrior, AmortizedInContextExplorationPrior)):
        return value
    if isinstance(value, Mapping):
        mode = str(value.get("mode") or value.get("distillation_mode") or "").lower()
        traces = value.get("traces") or value.get("trace_rows") or []
        if mode in {
            "in_context",
            "in_context_exploration_prior",
            "amortized_in_context",
        }:
            return AmortizedInContextExplorationPrior.from_traces(
                list(traces),
                max_context=int(value.get("max_context", value.get("max_depth", 3))),
                max_depth=value.get("max_depth"),
            )
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

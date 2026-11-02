"""ACT-style adaptive per-step budget gate for ARC explorer candidates.

Spec refs: REQ-ARC-FCP-4513, SCENARIO-ARC-FCP-4513.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class AdaptiveBudgetDecision:
    enabled: bool
    ambiguity_score: float | None
    threshold: float | None
    budget: int
    normal_width: int
    committed_single_candidate: bool
    components: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "ambiguity_score": self.ambiguity_score,
            "threshold": self.threshold,
            "budget": int(self.budget),
            "normal_width": int(self.normal_width),
            "committed_single_candidate": bool(self.committed_single_candidate),
            "components": dict(self.components),
        }


def _finite_floats(values: Sequence[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(number):
            out.append(number)
    return out


def _candidate_values(frame: Any, candidates: Sequence[Any], value_head: Any | None) -> list[float] | None:
    if value_head is None:
        return None
    if hasattr(value_head, "candidate_values"):
        try:
            return _finite_floats(value_head.candidate_values(frame, list(candidates)))
        except Exception:
            return None
    if hasattr(value_head, "candidate_value"):
        try:
            return _finite_floats([value_head.candidate_value(frame, candidate) for candidate in candidates])
        except Exception:
            return None
    if isinstance(value_head, Callable):
        try:
            return _finite_floats([value_head(frame, candidate) for candidate in candidates])
        except TypeError:
            return None
        except Exception:
            return None
    return None


def value_head_margin(frame: Any, candidates: Sequence[Any], value_head: Any | None) -> float | None:
    """REQ-ARC-FCP-4513: top-1/top-2 candidate margin from an existing value signal."""

    values = _candidate_values(frame, candidates, value_head)
    if values is None or len(values) != len(candidates) or not values:
        return None
    if len(values) == 1:
        return 1.0
    best, second = sorted(values)[:2]
    return abs(float(second) - float(best))


def _candidate_change_score(frame: Any, candidate: Any, scorer: Any | None) -> float | None:
    if scorer is None:
        return None
    try:
        if hasattr(scorer, "candidate_score"):
            return float(scorer.candidate_score(frame, candidate))
        if isinstance(scorer, Callable):
            return float(scorer(frame, candidate))
    except Exception:
        return None
    return None


def predicted_noop_fraction(
    frame: Any,
    candidates: Sequence[Any],
    scorer: Any | None,
    *,
    change_threshold: float = 0.5,
) -> float | None:
    """REQ-ARC-FCP-4513: fraction of candidates predicted to be no-ops."""

    if scorer is None or not candidates:
        return None
    scores: list[float] = []
    for candidate in candidates:
        score = _candidate_change_score(frame, candidate, scorer)
        if score is not None and isfinite(score):
            scores.append(float(score))
    if len(scores) != len(candidates):
        return None
    noop_count = sum(1 for score in scores if score < float(change_threshold))
    return float(noop_count / max(1, len(scores)))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _ambiguity_from_components(
    *,
    margin: float | None,
    noop_fraction: float | None,
    frame_is_novel: bool,
) -> tuple[float, dict[str, Any]]:
    margin_uncertainty = 0.5 if margin is None else 1.0 / (1.0 + max(0.0, float(margin)))
    noop_uncertainty = 0.5 if noop_fraction is None else 1.0 - _clamp01(float(noop_fraction))
    novelty_uncertainty = 1.0 if frame_is_novel else 0.0
    score = (margin_uncertainty + noop_uncertainty + novelty_uncertainty) / 3.0
    return _clamp01(score), {
        "value_head_margin": None if margin is None else float(margin),
        "value_margin_uncertainty": _clamp01(margin_uncertainty),
        "predicted_noop_fraction": None if noop_fraction is None else _clamp01(float(noop_fraction)),
        "noop_fraction_uncertainty": _clamp01(noop_uncertainty),
        "frame_novelty": bool(frame_is_novel),
        "frame_novelty_uncertainty": novelty_uncertainty,
    }


def adaptive_budget_decision(
    frame: Any,
    candidates: Sequence[Any],
    *,
    threshold: float | None,
    value_head: Any | None = None,
    frame_change_scorer: Any | None = None,
    frame_is_novel: bool,
    change_threshold: float = 0.5,
) -> AdaptiveBudgetDecision:
    """SCENARIO-ARC-FCP-4513: decide whether this frame gets budget 1 or normal width."""

    normal_width = int(len(candidates))
    if threshold is None or normal_width <= 1:
        return AdaptiveBudgetDecision(
            enabled=threshold is not None,
            ambiguity_score=None,
            threshold=None if threshold is None else float(threshold),
            budget=normal_width,
            normal_width=normal_width,
            committed_single_candidate=False,
            components={
                "value_head_margin": None,
                "predicted_noop_fraction": None,
                "frame_novelty": bool(frame_is_novel),
            },
        )
    margin = value_head_margin(frame, candidates, value_head)
    noop_fraction = predicted_noop_fraction(
        frame,
        candidates,
        frame_change_scorer,
        change_threshold=float(change_threshold),
    )
    ambiguity, components = _ambiguity_from_components(
        margin=margin,
        noop_fraction=noop_fraction,
        frame_is_novel=bool(frame_is_novel),
    )
    commit = bool(ambiguity < float(threshold))
    return AdaptiveBudgetDecision(
        enabled=True,
        ambiguity_score=float(ambiguity),
        threshold=float(threshold),
        budget=1 if commit else normal_width,
        normal_width=normal_width,
        committed_single_candidate=commit,
        components=components,
    )


def apply_adaptive_budget(
    frame: Any,
    candidates: Sequence[Any],
    *,
    threshold: float | None,
    value_head: Any | None = None,
    frame_change_scorer: Any | None = None,
    frame_is_novel: bool,
    change_threshold: float = 0.5,
) -> tuple[list[Any], AdaptiveBudgetDecision]:
    """REQ-ARC-FCP-4513: retain one candidate for easy frames, else normal width."""

    rows = list(candidates)
    decision = adaptive_budget_decision(
        frame,
        rows,
        threshold=threshold,
        value_head=value_head,
        frame_change_scorer=frame_change_scorer,
        frame_is_novel=frame_is_novel,
        change_threshold=change_threshold,
    )
    if decision.committed_single_candidate and rows:
        return [rows[0]], decision
    return rows, decision

"""Controllable novelty proposal scoring for the live ARC explorer.

Spec refs: REQ-ARC-WMTE-4688, SCENARIO-ARC-WMTE-4688.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import math
import random
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ControllableNoveltyConfig:
    """REQ-ARC-WMTE-4688: opt-in intrinsic proposal bonus configuration."""

    enabled: bool = False
    bonus_weight: float = 1.0
    temperature: float = 1.0
    controllability_gate: bool = True
    raw_frame_novelty: bool = False
    effect_score_threshold: float = 0.10
    min_changed_fraction: float = 1e-6
    episodic_k: int = 3
    episodic_weight: float = 1.0
    lifelong_weight: float = 0.35
    action_effect_weight: float = 1.0
    rnd_dim: int = 8
    rnd_lr: float = 0.35
    max_episodic_embeddings: int = 512
    random_seed: int = 4688


@dataclass(frozen=True)
class NoveltyScore:
    """One candidate's oracle-distinct intrinsic proposal score."""

    episodic: float
    lifelong: float
    action_effect: float
    total: float
    controllable: bool


@dataclass(frozen=True)
class NoveltyObservation:
    """One observed action-effect embedding stored in the novelty memories."""

    action: int
    embedding: tuple[float, ...]
    controllable: bool
    changed_fraction: float
    effect_score: float
    score: NoveltyScore


def _as_grid(frame: Any) -> np.ndarray:
    from carnot.agentic.arc_agi3_world_model import grid_of

    grid = np.asarray(grid_of(frame), dtype=float)
    if grid.ndim == 1:
        side = int(round(grid.size**0.5))
        grid = grid.reshape(side, side) if side * side == grid.size else grid.reshape(1, -1)
    return grid


def _action_id(candidate: Any) -> int:
    if isinstance(candidate, Mapping):
        return int(candidate.get("action", candidate.get("action_id", 0)) or 0)
    return int(getattr(candidate, "action", getattr(candidate, "action_id", 0)) or 0)


def _action_data(candidate: Any) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get("data")
    return getattr(candidate, "data", None)


def _candidate_row(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, Mapping):
        return {"action": _action_id(candidate), "data": candidate.get("data")}
    return {"action": _action_id(candidate), "data": getattr(candidate, "data", None)}


def _click_features(frame: Any, candidate: Any) -> list[float]:
    data = _action_data(candidate) or {}
    if _action_id(candidate) != 6 or not isinstance(data, Mapping):
        return [0.0, 0.0, 0.0]
    grid = _as_grid(frame)
    h, w = grid.shape
    x = int(data.get("x", 0))
    y = int(data.get("y", 0))
    inside = 1.0 if 0 <= y < h and 0 <= x < w else 0.0
    return [
        float(x / max(1, w - 1)),
        float(y / max(1, h - 1)),
        inside,
    ]


def _action_features(action: int) -> list[float]:
    out = [0.0] * 7
    if 0 <= int(action) < len(out):
        out[int(action)] = 1.0
    return out


def _changed_fraction(before: Any, after: Any) -> float:
    left = _as_grid(before)
    right = _as_grid(after)
    if left.shape != right.shape:
        return 1.0
    return float(np.count_nonzero(left != right) / max(1, left.size))


def _stable_unit_values(key: str, width: int) -> np.ndarray:
    values = []
    counter = 0
    while len(values) < width:
        digest = hashlib.sha256(f"{key}:{counter}".encode("utf-8")).digest()
        values.extend(byte / 255.0 for byte in digest)
        counter += 1
    return np.asarray(values[:width], dtype=float)


def _l2(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.size != b.size:
        n = min(a.size, b.size)
        a = a[:n]
        b = b[:n]
    return float(np.linalg.norm(a - b) / math.sqrt(max(1, a.size)))


class ControllableNoveltyProposalPolicy:
    """REQ-ARC-WMTE-4688: episodic kNN + RND-style novelty over controllable effects."""

    verifier_is_oracle = False

    def __init__(
        self,
        config: ControllableNoveltyConfig | Mapping[str, Any] | None = None,
        *,
        action_effect_scorer: Any | None = None,
    ) -> None:
        if config is None:
            config = ControllableNoveltyConfig(enabled=True)
        if isinstance(config, Mapping):
            config = ControllableNoveltyConfig(**dict(config))
        self.config = config
        self.action_effect_scorer = action_effect_scorer
        self._episodic: list[tuple[float, ...]] = []
        self._rnd_mean: np.ndarray | None = None
        self._candidate_scores = 0
        self._observed_effects = 0
        self._gate_rejected = 0
        self._rnd_updates = 0
        self._rng = random.Random(int(config.random_seed))

    def _effect_score(self, frame: Any, candidate: Any) -> float | None:
        scorer = self.action_effect_scorer
        if scorer is None or not hasattr(scorer, "candidate_score"):
            return None
        try:
            return float(scorer.candidate_score(frame, _candidate_row(candidate)))
        except Exception:
            return None

    def _proposal_embedding(self, frame: Any, candidate: Any) -> tuple[float, ...]:
        effect = self._effect_score(frame, candidate)
        effect_value = 0.0 if effect is None else max(0.0, float(effect))
        grid = _as_grid(frame)
        nonzero = float(np.count_nonzero(grid)) / max(1, grid.size)
        color_mean = float(np.mean(grid)) / 15.0 if grid.size else 0.0
        action = _action_id(candidate)
        return tuple(
            [
                *_action_features(action),
                *_click_features(frame, candidate),
                effect_value,
                nonzero,
                color_mean,
            ]
        )

    def _transition_embedding(
        self,
        before: Any,
        after: Any,
        action: Mapping[str, Any],
    ) -> tuple[float, ...]:
        if self.config.raw_frame_novelty and not self.config.controllability_gate:
            from carnot.agentic.arc_value_learner import cross_game_features_v2

            frame_features = [float(value) for value in cross_game_features_v2(after)]
            return tuple([*_action_features(_action_id(action)), *frame_features])

        from carnot.agentic.arc_value_learner import (
            cross_game_feature_slices_v3,
            cross_game_features_v3,
        )

        features = cross_game_features_v3(
            after,
            previous_frame=before,
            action_id=_action_id(action),
        )
        slices = cross_game_feature_slices_v3()
        frame_start, frame_stop = slices["frame_delta"]
        action_start, action_stop = slices["action_conditioned"]
        return tuple(float(value) for value in features[frame_start:frame_stop]) + tuple(
            float(value) for value in features[action_start:action_stop]
        )

    def _target(self, embedding: Sequence[float]) -> np.ndarray:
        key = ",".join(f"{float(value):.3f}" for value in embedding)
        return _stable_unit_values(f"{self.config.random_seed}:{key}", self.config.rnd_dim)

    def _episodic_score(self, embedding: Sequence[float]) -> float:
        if not self._episodic:
            return 1.0
        distances = sorted(_l2(embedding, row) for row in self._episodic)
        k = max(1, min(int(self.config.episodic_k), len(distances)))
        return float(min(1.0, sum(distances[:k]) / k))

    def _lifelong_score(self, embedding: Sequence[float]) -> float:
        target = self._target(embedding)
        if self._rnd_mean is None:
            return float(np.mean(np.square(target)))
        return float(np.mean(np.square(target - self._rnd_mean)))

    def _passes_gate(
        self,
        *,
        effect_score: float | None,
        changed_fraction: float | None = None,
    ) -> bool:
        if not self.config.controllability_gate:
            return True
        return self._controllable_effect(
            effect_score=effect_score,
            changed_fraction=changed_fraction,
        )

    def _controllable_effect(
        self,
        *,
        effect_score: float | None,
        changed_fraction: float | None = None,
    ) -> bool:
        if effect_score is not None:
            return float(effect_score) >= float(self.config.effect_score_threshold)
        return bool(changed_fraction is not None and changed_fraction >= self.config.min_changed_fraction)

    def score_embedding(
        self,
        embedding: Sequence[float],
        *,
        action_effect_score: float | None,
        changed_fraction: float | None = None,
    ) -> NoveltyScore:
        allowed = self._passes_gate(
            effect_score=action_effect_score,
            changed_fraction=changed_fraction,
        )
        controllable = self._controllable_effect(
            effect_score=action_effect_score,
            changed_fraction=changed_fraction,
        )
        if not allowed:
            return NoveltyScore(0.0, 0.0, 0.0, 0.0, False)
        episodic = self._episodic_score(embedding)
        lifelong = self._lifelong_score(embedding)
        action_effect = max(0.0, float(action_effect_score or 0.0))
        temp = max(1e-6, float(self.config.temperature))
        total = (
            float(self.config.bonus_weight)
            * (
                float(self.config.episodic_weight) * episodic
                + float(self.config.lifelong_weight) * lifelong
                + float(self.config.action_effect_weight) * action_effect
            )
            / temp
        )
        return NoveltyScore(
            episodic=float(episodic),
            lifelong=float(lifelong),
            action_effect=float(action_effect),
            total=float(total),
            controllable=bool(controllable),
        )

    def score_candidate(self, frame: Any, candidate: Mapping[str, Any]) -> NoveltyScore:
        self._candidate_scores += 1
        effect = self._effect_score(frame, candidate)
        return self.score_embedding(
            self._proposal_embedding(frame, candidate),
            action_effect_score=effect,
        )

    def rank_candidates(
        self,
        frame: Any,
        candidates: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if not self.config.enabled:
            return [dict(row) for row in candidates]
        rows = []
        for index, candidate in enumerate(candidates):
            score = self.score_candidate(frame, candidate)
            row = dict(candidate)
            row["controllable_novelty_bonus"] = float(score.total)
            row["controllable_novelty_components"] = {
                "episodic": float(score.episodic),
                "lifelong": float(score.lifelong),
                "action_effect": float(score.action_effect),
                "controllable": bool(score.controllable),
                "temperature": float(self.config.temperature),
            }
            rows.append((index, row))
        rows.sort(key=lambda item: (-float(item[1]["controllable_novelty_bonus"]), item[0]))
        return [row for _index, row in rows]

    def record_transition(
        self,
        before: Any,
        after: Any,
        action: Mapping[str, Any],
    ) -> NoveltyObservation | None:
        effect = self._effect_score(before, action)
        changed = _changed_fraction(before, after)
        embedding = self._transition_embedding(before, after, action)
        score = self.score_embedding(
            embedding,
            action_effect_score=effect,
            changed_fraction=changed,
        )
        if not score.controllable and self.config.controllability_gate:
            self._gate_rejected += 1
            return None
        self._observed_effects += 1
        self._episodic.append(tuple(float(value) for value in embedding))
        if len(self._episodic) > int(self.config.max_episodic_embeddings):
            drop = self._rng.randrange(len(self._episodic))
            del self._episodic[drop]
        target = self._target(embedding)
        if self._rnd_mean is None:
            self._rnd_mean = np.zeros_like(target)
        lr = max(0.0, min(1.0, float(self.config.rnd_lr)))
        self._rnd_mean = (1.0 - lr) * self._rnd_mean + lr * target
        self._rnd_updates += 1
        return NoveltyObservation(
            action=_action_id(action),
            embedding=tuple(float(value) for value in embedding),
            controllable=bool(score.controllable),
            changed_fraction=float(changed),
            effect_score=float(effect or 0.0),
            score=score,
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.enabled),
            "controllability_gate_on": bool(self.config.controllability_gate),
            "raw_frame_novelty": bool(self.config.raw_frame_novelty),
            "temperature": float(self.config.temperature),
            "bonus_weight": float(self.config.bonus_weight),
            "candidate_scores": int(self._candidate_scores),
            "observed_effects": int(self._observed_effects),
            "episodic_embeddings": int(len(self._episodic)),
            "rnd_updates": int(self._rnd_updates),
            "controllability_gate_rejected": int(self._gate_rejected),
            "verifier_is_oracle": False,
        }


def coerce_controllable_novelty_policy(
    value: bool | ControllableNoveltyConfig | ControllableNoveltyProposalPolicy | Mapping[str, Any] | None,
    *,
    action_effect_scorer: Any | None = None,
) -> ControllableNoveltyProposalPolicy | None:
    """REQ-ARC-WMTE-4688: normalize live StepwiseExplorer novelty configuration."""

    if isinstance(value, ControllableNoveltyProposalPolicy):
        if value.action_effect_scorer is None and action_effect_scorer is not None:
            value.action_effect_scorer = action_effect_scorer
        return value if value.config.enabled else None
    if value is None or value is False:
        return None
    if value is True:
        config = ControllableNoveltyConfig(enabled=True)
    elif isinstance(value, ControllableNoveltyConfig):
        config = value
    elif isinstance(value, Mapping):
        config = ControllableNoveltyConfig(**dict(value))
    else:
        return None
    if not config.enabled:
        return None
    return ControllableNoveltyProposalPolicy(config, action_effect_scorer=action_effect_scorer)

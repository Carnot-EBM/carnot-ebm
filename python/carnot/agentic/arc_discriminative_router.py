"""Cross-game discriminative verifier router for ARC candidate actions.

Spec refs: REQ-CAPSTONE-4556, SCENARIO-CAPSTONE-4556.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.agentic.arc_value_learner import (
    CrossGameFrameContextV3,
    DiscriminativeVerifier,
    cross_game_feature_slices_v3,
    cross_game_features_v3,
    cross_game_frame_context_v3,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_RELATIVE_PATH = Path("models/arc_discriminative_verifier_v3.json")


def _action_id(action: Any) -> int:
    return int(getattr(action, "action_id", getattr(action, "action", 0)) or 0)


def candidate_action_key(action: Any) -> tuple[Any, ...]:
    data = getattr(action, "data", None)
    aid = _action_id(action)
    if aid == 6 and isinstance(data, dict):
        return (6, int(data.get("x", 0)), int(data.get("y", 0)))
    return (aid,)


def _frame_digest(frame: Any) -> str:
    try:
        from carnot.agentic.arc_agi3_world_model import grid_of

        grid = grid_of(frame)
        return hashlib.sha256(grid.tobytes() + repr(tuple(grid.shape)).encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha256(repr(frame).encode("utf-8", errors="replace")).hexdigest()


class CrossGameDiscriminativeCandidateRouter:
    """REQ-CAPSTONE-4556: rank candidates by learned v3 discrimination scores."""

    verifier_is_oracle = False

    def __init__(
        self,
        verifier: Any,
        *,
        prune_threshold: float | None = None,
        min_candidates: int = 1,
    ) -> None:
        self.verifier = verifier
        self.prune_threshold = prune_threshold
        self.min_candidates = max(1, int(min_candidates))

    def score(
        self,
        frame: Any,
        action: Any,
        *,
        previous_frame: Any | None = None,
        frame_context: CrossGameFrameContextV3 | None = None,
    ) -> float:
        try:
            features = cross_game_features_v3(
                frame,
                previous_frame=previous_frame,
                action_id=_action_id(action),
                goal_frame=None,
                frame_context=frame_context,
            )
            return float(self.verifier.proba_features(features))
        except Exception:
            return 0.5

    def rank(
        self,
        frame: Any,
        candidates: Sequence[Any],
        *,
        previous_frame: Any | None = None,
    ) -> list[Any]:
        # Computed once per rank() call and reused across every candidate -- see
        # CrossGameFrameContextV3's docstring for the O(candidates x components^2) incident this
        # fixes (score() previously recomputed the frame-level features from scratch per
        # candidate even though only the cheap action_id slice actually varies).
        frame_context = cross_game_frame_context_v3(frame, previous_frame, goal_frame=None)
        scored = [
            (
                self.score(
                    frame, action, previous_frame=previous_frame, frame_context=frame_context
                ),
                index,
                action,
            )
            for index, action in enumerate(candidates)
        ]
        if self.prune_threshold is not None and len(scored) > self.min_candidates:
            kept = [item for item in scored if item[0] >= float(self.prune_threshold)]
            if len(kept) < self.min_candidates:
                kept = sorted(scored, key=lambda item: (-item[0], item[1]))[: self.min_candidates]
            scored = kept
        return [
            action
            for _score, _index, action in sorted(scored, key=lambda item: (-item[0], item[1]))
        ]


class RandomCandidateRouter:
    """Deterministic random-router positive control for REQ-CAPSTONE-4556."""

    verifier_is_oracle = False

    def __init__(self, seed: int = 4556) -> None:
        self.seed = int(seed)

    def _score(self, frame: Any, action: Any) -> float:
        payload = json.dumps(
            {
                "seed": self.seed,
                "frame": _frame_digest(frame),
                "action": candidate_action_key(action),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return int(hashlib.sha256(payload).hexdigest()[:16], 16) / float(16**16 - 1)

    def rank(self, frame: Any, candidates: Sequence[Any], **_: Any) -> list[Any]:
        scored = [
            (self._score(frame, action), index, action) for index, action in enumerate(candidates)
        ]
        return [
            action
            for _score, _index, action in sorted(scored, key=lambda item: (-item[0], item[1]))
        ]


class CrossGameDiscriminativeExpansionPriority:
    """REQ-CAPSTONE-4569: score frontier nodes for verifier-guided expansion.

    The graph and stepwise solvers expect lower energies to expand first, while
    the discriminative verifier emits higher P(on winning path). This adapter
    converts the learned probability into a non-oracle energy without inspecting
    executable win checks.
    """

    verifier_is_oracle = False

    def __init__(
        self,
        verifier: Any,
        *,
        featurize: Any = cross_game_features_v3,
        neutral_proba: float = 0.5,
    ) -> None:
        self.verifier = verifier
        self.featurize = featurize
        self.neutral_proba = float(neutral_proba)

    def proba(self, frame: Any) -> float:
        try:
            features = self.featurize(frame)
            return float(self.verifier.proba_features(features))
        except Exception:
            return self.neutral_proba

    def __call__(self, frame: Any) -> float:
        return 1.0 - self.proba(frame)


class RandomExpansionPriority:
    """Deterministic random frontier-priority positive control for REQ-CAPSTONE-4569."""

    verifier_is_oracle = False

    def __init__(self, seed: int = 4569) -> None:
        self.seed = int(seed)

    def __call__(self, frame: Any) -> float:
        payload = json.dumps(
            {
                "seed": self.seed,
                "frame": _frame_digest(frame),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return int(hashlib.sha256(payload).hexdigest()[:16], 16) / float(16**16 - 1)


def load_cross_game_discriminative_router(
    *,
    root: Path | str = REPO_ROOT,
    checkpoint: Path | str | None = None,
    prune_threshold: float | None = None,
) -> CrossGameDiscriminativeCandidateRouter | None:
    """Load the Exp 4545 v3 checkpoint, returning None when unavailable."""

    path = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT_RELATIVE_PATH
    if not path.is_absolute():
        path = Path(root) / path
    try:
        verifier = DiscriminativeVerifier.load(path, cross_game_features_v3)
    except Exception:
        return None
    return CrossGameDiscriminativeCandidateRouter(
        verifier,
        prune_threshold=prune_threshold,
    )


def load_cross_game_discriminative_expansion_priority(
    *,
    root: Path | str = REPO_ROOT,
    checkpoint: Path | str | None = None,
) -> CrossGameDiscriminativeExpansionPriority | None:
    """Load the Exp 4545 v3 checkpoint as a frontier-expansion priority."""

    path = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT_RELATIVE_PATH
    if not path.is_absolute():
        path = Path(root) / path
    try:
        verifier = DiscriminativeVerifier.load(path, cross_game_features_v3)
    except Exception:
        return None
    return CrossGameDiscriminativeExpansionPriority(verifier)


def dominant_feature_family_from_weights(weights: Sequence[float]) -> str:
    """Return the v3 feature slice with the largest absolute logistic mass."""

    slices = cross_game_feature_slices_v3()
    width = max(stop for _start, stop in slices.values())
    values = [float(value) for value in list(weights)[:width]]
    masses = {
        name: sum(abs(value) for value in values[start:stop])
        for name, (start, stop) in slices.items()
    }
    return max(masses, key=lambda name: (masses[name], name))


def dominant_feature_family_from_checkpoint(path: Path | str) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return dominant_feature_family_from_weights(payload.get("weights", []))


def checkpoint_sha256(path: Path | str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()

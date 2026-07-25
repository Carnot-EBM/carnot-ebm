"""Cross-game discriminative verifier router for ARC candidate actions.

Spec refs: REQ-CAPSTONE-4556, SCENARIO-CAPSTONE-4556,
REQ-ARC-FCP-5904, SCENARIO-ARC-FCP-5904.

REQ-ARC-FCP-5904 adds ``OnlineClickTargetRouter``, a PURELY ADDITIVE coordinate-aware
click ranker that repairs the coordinate-blindness of the incumbent router documented in
``arc_click_target_features``'s module docstring. It is DEFAULT OFF (see
``SUBMITTED_ONLINE_CLICK_TARGET_ROUTER_ENABLED`` below), so live behaviour is unchanged
until an operator flips it after a live A/B. Nothing about
``CrossGameDiscriminativeCandidateRouter``, ``_action_id``, ``candidate_action_key`` or
``cross_game_features_v3`` changes -- their shared 79-feature contract is load-bearing for
``models/arc_discriminative_verifier_v3.json`` (79 weights + bias, with no shape validation
on load), and appending features to that vector would raise inside ``proba_features`` where
``score``'s blanket ``except Exception: return 0.5`` would convert the error into a SILENT
all-0.5 constant router -- i.e. it would invisibly re-create the exact bug being fixed.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot.agentic.arc_click_target_features import (
    CLICK_TARGET_FEATURE_DIM,
    ClickEpisodeState,
    ClickTargetFrameContext,
    OnlineClickTargetDiscriminator,
    click_coordinates,
    click_target_features,
    click_target_frame_context,
    click_target_object_identity,
    settled_grid_of,
)
from carnot.agentic.arc_value_learner import (
    CrossGameFrameContextV3,
    DiscriminativeVerifier,
    cross_game_feature_slices_v3,
    cross_game_features_v3,
    cross_game_frame_context_v3,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_RELATIVE_PATH = Path("models/arc_discriminative_verifier_v3.json")

# REQ-ARC-FCP-5904: the online coordinate-aware click router's master switch.
#
# The flag lives HERE rather than in ``arc_competition_agent.py`` on purpose: with it False,
# no edit to the live agent file is required and live parity is exact. Flip it only after a
# LIVE A/B on banked levels -- offline AUROC licenses NOTHING here. Precedent: exp4545's
# 0.725-AUROC discriminator REGRESSED live search, which is why the agent's
# ``SUBMITTED_VALUE_WEIGHT`` is pinned at 1e-12.
SUBMITTED_ONLINE_CLICK_TARGET_ROUTER_ENABLED = False


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


def _episode_key(frame: Any) -> tuple[str, str]:
    """Identify the EPISODE a frame belongs to, so online state cannot leak across games.

    ``FrameDataRaw`` exposes ``.game_id`` (e.g. ``'lp85-305b61c3'``) and ``.guid`` (the
    per-episode uuid); both were verified present on real offline frames. Keying online state
    on this pair is mandatory rather than cosmetic: ``scripts/arc_leaderboard_eval.py`` caches
    ONE router instance for an entire multi-game sweep, so counters/weights stored bare on
    ``self`` would become de-facto cross-game transfer -- the direction retired by
    ``ops/exclusion_manifest.yaml`` id ``cross_game_value_transfer_retired_exp4342_v401``
    (``operator_reopen_required: true``). Test doubles lacking these attributes collapse into
    one ``('unknown', 'unknown')`` bucket.
    """

    game_id = getattr(frame, "game_id", None)
    guid = getattr(frame, "guid", None)
    return (
        str(game_id) if game_id else "unknown",
        str(guid) if guid else "unknown",
    )


class OnlineClickTargetRouter:
    """REQ-ARC-FCP-5904: rank click candidates by an ONLINE, WITHIN-EPISODE progress model.

    Purely additive wrapper. Behaviour contract, in order of importance:

    1. **Default OFF.** ``enabled`` defaults to the module constant
       ``SUBMITTED_ONLINE_CLICK_TARGET_ROUTER_ENABLED`` (False). While off, ``score`` and
       ``rank`` delegate straight to the wrapped ``base`` router (or return the input order
       when there is none) and cost nothing -- no blob segmentation, no featurization.
    2. **Cold start is a no-op.** While the online head has not met its sample gate,
       ``discriminator.proba`` returns exactly ``0.5``, so the additive contribution is
       exactly ``0.0`` and ``score`` returns the base score bit-for-bit. An episode that
       never reaches the gate is indistinguishable from today.
    3. **Per-frame work happens once per ``rank()``.** Both the incumbent
       ``CrossGameFrameContextV3`` and the new ``ClickTargetFrameContext`` are built once and
       reused across candidates. ``score`` remains self-sufficient (it builds/reuses a
       content-cached context when handed none) so a caller scoring candidates one at a time
       still gets identical numbers.
    4. **Only clicks are affected.** A keyboard candidate has no coordinates, contributes
       exactly ``0.0``, and keeps its v3-governed placement.

    THE OBSERVATION HOOK IS NOT YET CALLED IN THE LIVE PATH -- an honest limitation. The call
    site that would feed real in-episode outcomes lives in ``arc_competition_agent.py``, which
    was out of scope for the change that shipped this class. In-repo, ``observe_click_outcome``
    is exercised only by the offline experiment and the tests. The follow-up is a one-line
    agent-side call next to the existing ``observe_transition`` hooks.

    ALSO HONEST ABOUT REACH: ``arc_graph_explore.rich_action_candidates`` applies this router
    and then RE-SORTS with ``rank_arc_actions(scorer=GroundTruthValidatedFrameChangeScorer)``.
    That scorer returns 0.0 until it validates (>=1 agreement, 0 contradictions, permanent).
    Measured over 12 real transitions per game: ``tn36`` and ``r11l`` validate and then
    produce 18-23 distinct click scores, demoting this router to a TIEBREAK; ``bp35``,
    ``lp85`` and ``su15`` are permanently contradicted (all zeros), so this router owns the
    click order there. Any live A/B must therefore stratify on
    ``frame_diff_ground_truth_validated``; a gate that assumes full control everywhere is
    unpassable by construction (the exp5835 defect class).
    """

    verifier_is_oracle = False

    def __init__(
        self,
        base: Any | None = None,
        *,
        enabled: bool | None = None,
        weight: float = 0.25,
        discriminator: OnlineClickTargetDiscriminator | None = None,
        max_episodes: int = 2,
    ) -> None:
        self.base = base
        self.enabled = (
            bool(SUBMITTED_ONLINE_CLICK_TARGET_ROUTER_ENABLED) if enabled is None else bool(enabled)
        )
        self.weight = float(weight)
        self.max_episodes = max(1, int(max_episodes))
        # An explicitly-supplied discriminator is a TEST/EXPERIMENT INJECTION so the caller can
        # hold a handle on the head. It is bound to the FIRST episode only; every subsequent
        # episode gets a fresh head.
        #
        # That single-use restriction is a deliberate leak fix, not an inconvenience. Honouring
        # an injected head for every episode would make one fitted head follow the router
        # across games -- exactly the cross-game value transfer retired by
        # ``ops/exclusion_manifest.yaml`` id ``cross_game_value_transfer_retired_exp4342_v401``
        # -- and it would do so through the very shape that makes such leakage likely:
        # ``scripts/arc_leaderboard_eval.py`` caches ONE router for a whole multi-game sweep.
        # Consuming the injection on first use means no wiring can turn this into a
        # cross-game head, whatever the caller intended.
        self._pending_injected_discriminator = discriminator
        self._episodes: "OrderedDict[tuple[str, str], tuple[OnlineClickTargetDiscriminator, ClickEpisodeState]]" = OrderedDict()

    # ------------------------------------------------------------------ episode state

    def _episode(self, frame: Any) -> tuple[OnlineClickTargetDiscriminator, ClickEpisodeState]:
        key = _episode_key(frame)
        existing = self._episodes.get(key)
        if existing is not None:
            self._episodes.move_to_end(key)
            return existing
        injected = self._pending_injected_discriminator
        self._pending_injected_discriminator = None  # consumed: first episode only
        entry = (
            injected if injected is not None else OnlineClickTargetDiscriminator(dim=CLICK_TARGET_FEATURE_DIM),
            ClickEpisodeState(),
        )
        self._episodes[key] = entry
        if len(self._episodes) > self.max_episodes:
            self._episodes.popitem(last=False)
        return entry

    def reset_episode(self) -> None:
        """Forget all online state. Never persisted to disk; nothing survives a process."""

        self._episodes.clear()

    def discriminator_for(self, frame: Any) -> OnlineClickTargetDiscriminator:
        return self._episode(frame)[0]

    def episode_count(self) -> int:
        return len(self._episodes)

    # ------------------------------------------------------------------ scoring

    def _base_score(
        self,
        frame: Any,
        action: Any,
        *,
        previous_frame: Any | None,
        frame_context: CrossGameFrameContextV3 | None,
    ) -> float:
        if self.base is None:
            return 0.5
        try:
            return float(
                self.base.score(
                    frame,
                    action,
                    previous_frame=previous_frame,
                    frame_context=frame_context,
                )
            )
        except TypeError:
            return float(self.base.score(frame, action))
        except Exception:
            return 0.5

    def click_delta(
        self,
        frame: Any,
        action: Any,
        *,
        click_context: ClickTargetFrameContext | None = None,
    ) -> float:
        """The additive coordinate-aware contribution: ``weight * (P(progress) - 0.5)``.

        Exactly ``0.0`` when disabled, when the candidate is not a click, or when the online
        head has not met its sample gate. That is what makes cold start a no-op.
        """

        if not self.enabled:
            return 0.0
        coords = click_coordinates(action)
        if coords is None:
            return 0.0
        try:
            context = (
                click_context if click_context is not None else click_target_frame_context(frame)
            )
            discriminator, episode_state = self._episode(frame)
            features = click_target_features(
                context, coords[0], coords[1], episode_state=episode_state
            )
            return self.weight * (discriminator.proba(features) - 0.5)
        except Exception:
            return 0.0

    def score(
        self,
        frame: Any,
        action: Any,
        *,
        previous_frame: Any | None = None,
        frame_context: CrossGameFrameContextV3 | None = None,
        click_context: ClickTargetFrameContext | None = None,
    ) -> float:
        base = self._base_score(
            frame, action, previous_frame=previous_frame, frame_context=frame_context
        )
        return base + self.click_delta(frame, action, click_context=click_context)

    def rank(
        self,
        frame: Any,
        candidates: Sequence[Any],
        *,
        previous_frame: Any | None = None,
    ) -> list[Any]:
        items = list(candidates)
        if not self.enabled:
            # Byte-identical passthrough: the wrapped router's own ordering, or the input
            # order when this router is used standalone.
            if self.base is None:
                return items
            return list(self.base.rank(frame, items, previous_frame=previous_frame))

        discriminator, _state = self._episode(frame)
        if not discriminator.fitted:
            # Cold start: no ordering opinion at all. Delegating (rather than scoring with a
            # constant 0.5) keeps the un-fitted path free of any blob segmentation cost.
            if self.base is None:
                return items
            return list(self.base.rank(frame, items, previous_frame=previous_frame))

        frame_context: CrossGameFrameContextV3 | None = None
        if self.base is not None:
            try:
                frame_context = cross_game_frame_context_v3(frame, previous_frame, goal_frame=None)
            except Exception:
                frame_context = None
        try:
            click_context = click_target_frame_context(frame)
        except Exception:
            return (
                items
                if self.base is None
                else list(self.base.rank(frame, items, previous_frame=previous_frame))
            )

        scored = [
            (
                self.score(
                    frame,
                    action,
                    previous_frame=previous_frame,
                    frame_context=frame_context,
                    click_context=click_context,
                ),
                index,
                action,
            )
            for index, action in enumerate(items)
        ]
        return [
            action
            for _score, _index, action in sorted(scored, key=lambda item: (-item[0], item[1]))
        ]

    # ------------------------------------------------------------------ observation

    def observe_click_outcome(
        self,
        frame_before: Any,
        action: Any,
        frame_after: Any,
        *,
        leveled_up: bool = False,
    ) -> bool:
        """Feed one of the agent's OWN executed clicks back into the online head.

        The label is causally downstream of the click by construction: it is read from the
        observed post-click frame (did the settled grid change, or did the level advance).
        This is deliberately NOT the human-replay corpus's ``level_progress``, which is a pure
        function of the step index and therefore learnable with zero perception (exp5835).

        Returns True when an observation was recorded.
        """

        if not self.enabled:
            return False
        coords = click_coordinates(action)
        if coords is None:
            return False
        try:
            context = click_target_frame_context(frame_before)
            discriminator, episode_state = self._episode(frame_before)
            features = click_target_features(
                context, coords[0], coords[1], episode_state=episode_state
            )
            changed = _grids_differ(frame_before, frame_after)
            label = 1.0 if (changed or leveled_up) else 0.0
            discriminator.observe(features, label, leveled_up=bool(leveled_up))
            episode_state.observe_click(
                coords[0],
                coords[1],
                click_target_object_identity(context, coords[0], coords[1]),
            )
            discriminator.maybe_fit()
            return True
        except Exception:
            return False

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "weight": self.weight,
            "episodes": len(self._episodes),
            "per_episode": {
                f"{game_id}/{guid}": discriminator.stats()
                for (game_id, guid), (discriminator, _state) in self._episodes.items()
            },
        }


def _grids_differ(frame_a: Any, frame_b: Any) -> bool:
    """True when two frames' settled grids differ.

    Copies before comparing. ``env.step()`` returns distinct frame OBJECTS whose underlying
    grid data is a SHARED, mutated-in-place buffer (documented at
    ``arc_solver_kit.py:5315-5325``), so a bare reference to an earlier frame silently
    reflects the CURRENT env state and every comparison would read "unchanged".
    """

    a = np.array(settled_grid_of(frame_a), copy=True)
    b = np.array(settled_grid_of(frame_b), copy=True)
    if a.shape != b.shape:
        return True
    return bool(np.any(a != b))


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


def load_online_click_target_router(
    *,
    root: Path | str = REPO_ROOT,
    checkpoint: Path | str | None = None,
    prune_threshold: float | None = None,
    enabled: bool | None = None,
    weight: float = 0.25,
) -> OnlineClickTargetRouter:
    """Wrap the incumbent v3 router in the coordinate-aware online click router.

    Returns a router even when the v3 checkpoint is unavailable (``base=None``), because the
    online head needs no checkpoint -- it fits from the episode's own transitions. Note that
    with the default ``enabled=None`` this resolves to
    ``SUBMITTED_ONLINE_CLICK_TARGET_ROUTER_ENABLED`` (False), so the returned router is a pure
    passthrough to the v3 router until an operator enables it.
    """

    base = load_cross_game_discriminative_router(
        root=root, checkpoint=checkpoint, prune_threshold=prune_threshold
    )
    return OnlineClickTargetRouter(base, enabled=enabled, weight=weight)


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

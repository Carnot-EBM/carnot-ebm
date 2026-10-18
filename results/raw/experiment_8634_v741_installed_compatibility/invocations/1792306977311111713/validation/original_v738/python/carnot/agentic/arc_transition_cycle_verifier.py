"""Generic ARC forward/inverse transition-cycle verifier.

Spec refs: REQ-ARC-WMTE-5619,
SCENARIO-ARC-WMTE-5619-CYCLE-ADMISSION,
SCENARIO-ARC-WMTE-5619-CORRUPTION-REJECTION.

The verifier is intentionally modest: it learns sparse action-effect signatures from the
agent's own observed transitions and admits downstream update receipts only when three checks
agree. Successor plausibility asks whether the observed effect shape has appeared for the
candidate action. Inverse recovery asks whether the effect signature predicts the executed
action. Forward replay applies the learned sparse effect around the action's click coordinate
and compares the predicted successor to the observed successor.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ObservedTransition:
    """One transition caused by the agent, never by a game-source oracle."""

    game: str
    episode: str
    step: int
    state: Any
    action: int
    data: Any
    successor: Any
    condition: str = "valid"
    provenance: str = "agent_owned_runtime_observation"


@dataclass(frozen=True)
class TransitionCycleDecision:
    """Admission result for one candidate world-model update."""

    condition: str
    admitted: bool
    rejected: bool
    abstained: bool
    successor_plausible: bool
    inverse_action: int | None
    inverse_action_matches: bool
    forward_replay_error: float | None
    reason: str
    update_receipt: JsonDict | None = None


@dataclass(frozen=True)
class _EffectFeatures:
    effect_core: tuple[Any, ...]
    action_effect: tuple[Any, ...]
    click_offsets: tuple[tuple[int, int, int, int], ...]
    spatial_offsets: tuple[tuple[int, int, int, int], ...]
    changed_count: int
    state_shape: tuple[int, int]


def _grid(value: Any) -> np.ndarray:
    if hasattr(value, "frame"):
        value = getattr(value, "frame")
    return np.asarray(value, dtype=np.int16)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True)


def _hash_payload(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _click_xy(data: Any) -> tuple[int, int] | None:
    if not isinstance(data, Mapping):
        return None
    if "x" not in data or "y" not in data:
        return None
    return int(data["x"]), int(data["y"])


def _extract_features(state: Any, action: int, data: Any, successor: Any) -> _EffectFeatures:
    before = _grid(state)
    after = _grid(successor)
    if before.shape != after.shape or before.ndim != 2:
        return _EffectFeatures(
            effect_core=("malformed", tuple(before.shape), tuple(after.shape)),
            action_effect=(int(action), "malformed", tuple(before.shape), tuple(after.shape)),
            click_offsets=(),
            spatial_offsets=(),
            changed_count=0,
            state_shape=(0, 0),
        )
    changed = np.argwhere(before != after)
    pairs = Counter((int(before[r, c]), int(after[r, c])) for r, c in changed)
    pair_key = tuple(sorted((src, dst, count) for (src, dst), count in pairs.items()))
    if len(changed):
        y0, x0 = changed.min(axis=0)
        y1, x1 = changed.max(axis=0)
        bbox = (int(y1 - y0 + 1), int(x1 - x0 + 1))
    else:
        bbox = (0, 0)
    click = _click_xy(data)
    offsets: list[tuple[int, int, int, int]] = []
    spatial_offsets: list[tuple[int, int, int, int]] = []
    if click is not None:
        x, y = click
        for r, c in changed:
            offsets.append((int(r - y), int(c - x), int(before[r, c]), int(after[r, c])))
    if len(changed):
        y0, x0 = changed.min(axis=0)
        for r, c in changed:
            spatial_offsets.append((int(r - y0), int(c - x0), int(before[r, c]), int(after[r, c])))
    effect_core = (int(len(changed)), bbox, pair_key)
    action_effect = (int(action), effect_core, tuple(sorted(offsets)))
    return _EffectFeatures(
        effect_core=effect_core,
        action_effect=action_effect,
        click_offsets=tuple(sorted(offsets)),
        spatial_offsets=tuple(sorted(spatial_offsets)),
        changed_count=int(len(changed)),
        state_shape=(int(before.shape[0]), int(before.shape[1])),
    )


def _normalise_transition(row: Any) -> ObservedTransition:
    if isinstance(row, ObservedTransition):
        return row
    state = getattr(row, "grid", getattr(row, "state", None))
    successor = getattr(row, "next_grid", getattr(row, "successor", None))
    return ObservedTransition(
        game=str(getattr(row, "game", "unknown")),
        episode=str(getattr(row, "episode", "episode-0")),
        step=int(getattr(row, "step", 0)),
        state=state,
        action=int(getattr(row, "action")),
        data=getattr(row, "data", None),
        successor=successor,
        condition=str(getattr(row, "condition", "valid")),
        provenance=str(getattr(row, "provenance", "agent_owned_runtime_observation")),
    )


class TransitionCycleVerifier:
    """Session-local verifier for admitting generic action-effect update receipts."""

    def __init__(self, *, min_support: int = 2, forward_error_threshold: float = 0.0) -> None:
        self.min_support = max(1, int(min_support))
        self.forward_error_threshold = max(0.0, float(forward_error_threshold))
        self._action_effect_counts: dict[int, Counter[tuple[Any, ...]]] = defaultdict(Counter)
        self._inverse_counts: dict[tuple[Any, ...], Counter[int]] = defaultdict(Counter)
        self._click_rules: dict[int, Counter[tuple[tuple[int, int, int, int], ...]]] = defaultdict(
            Counter
        )
        self._spatial_rules: dict[int, Counter[tuple[tuple[int, int, int, int], ...]]] = (
            defaultdict(Counter)
        )
        self._observed = 0
        self._evaluated = 0
        self._admitted = 0
        self._rejected = 0
        self._abstained = 0
        self._unsafe_accepts = 0
        self._receipts: list[JsonDict] = []

    def fit(self, rows: Sequence[Any]) -> "TransitionCycleVerifier":
        for row in rows:
            self.update(row)
        return self

    def update(self, row: Any) -> None:
        transition = _normalise_transition(row)
        features = _extract_features(
            transition.state,
            transition.action,
            transition.data,
            transition.successor,
        )
        self._observed += 1
        self._action_effect_counts[int(transition.action)][features.action_effect] += 1
        self._inverse_counts[features.effect_core][int(transition.action)] += 1
        if _click_xy(transition.data) is not None and features.click_offsets:
            self._click_rules[int(transition.action)][features.click_offsets] += 1
        if features.spatial_offsets:
            self._spatial_rules[int(transition.action)][features.spatial_offsets] += 1

    def observe_transition(
        self, before: Any, action: int, data: Any, after: Any
    ) -> TransitionCycleDecision:
        transition = ObservedTransition(
            game="live",
            episode="live",
            step=self._observed,
            state=before,
            action=int(action),
            data=data,
            successor=after,
        )
        decision = self.evaluate(transition)
        self.update(transition)
        return decision

    def evaluate(self, row: Any) -> TransitionCycleDecision:
        transition = _normalise_transition(row)
        features = _extract_features(
            transition.state,
            transition.action,
            transition.data,
            transition.successor,
        )
        inverse_action = self._inverse_action(features.effect_core)
        action_counter = self._action_effect_counts.get(int(transition.action), Counter())
        successor_plausible = action_counter.get(features.action_effect, 0) >= self.min_support
        forward_error = self._forward_replay_error(transition)
        inverse_matches = inverse_action == int(transition.action)
        forward_ok = forward_error is not None and forward_error <= self.forward_error_threshold
        abstained = inverse_action is None or forward_error is None
        admitted = bool(successor_plausible and inverse_matches and forward_ok and not abstained)
        reason = (
            "cycle_consistent"
            if admitted
            else self._reason(successor_plausible, inverse_action, inverse_matches, forward_error)
        )
        receipt = (
            self._receipt(transition, features, inverse_action, forward_error) if admitted else None
        )
        self._record_decision(transition.condition, admitted, abstained, receipt)
        return TransitionCycleDecision(
            condition=transition.condition,
            admitted=admitted,
            rejected=not admitted and not abstained,
            abstained=abstained,
            successor_plausible=successor_plausible,
            inverse_action=inverse_action,
            inverse_action_matches=inverse_matches,
            forward_replay_error=forward_error,
            reason=reason,
            update_receipt=receipt,
        )

    def _inverse_action(self, effect_core: tuple[Any, ...]) -> int | None:
        counts = self._inverse_counts.get(effect_core)
        if not counts:
            return None
        ordered = counts.most_common(2)
        if ordered[0][1] < self.min_support:
            return None
        if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
            return None
        return int(ordered[0][0])

    def _forward_replay_error(self, transition: ObservedTransition) -> float | None:
        before = _grid(transition.state)
        after = _grid(transition.successor)
        if before.shape != after.shape or before.ndim != 2:
            return 1.0
        spatial_error = self._spatial_replay_error(transition, before, after)
        click = _click_xy(transition.data)
        if click is None:
            return spatial_error
        rules = self._click_rules.get(int(transition.action))
        if not rules:
            return spatial_error
        offsets, support = rules.most_common(1)[0]
        if support < self.min_support:
            return spatial_error
        pred = before.copy()
        x, y = click
        for dy, dx, before_value, after_value in offsets:
            r = y + int(dy)
            c = x + int(dx)
            if r < 0 or c < 0 or r >= pred.shape[0] or c >= pred.shape[1]:
                return 1.0
            if int(pred[r, c]) != int(before_value):
                return spatial_error if spatial_error is not None else 1.0
            pred[r, c] = int(after_value)
        return float(np.mean(pred != after))

    def _spatial_replay_error(
        self,
        transition: ObservedTransition,
        before: np.ndarray,
        after: np.ndarray,
    ) -> float | None:
        rules = self._spatial_rules.get(int(transition.action))
        if not rules:
            return None
        offsets, support = rules.most_common(1)[0]
        if support < self.min_support:
            return None
        max_dy = max(offset[0] for offset in offsets)
        max_dx = max(offset[1] for offset in offsets)
        for y0 in range(0, before.shape[0] - max_dy):
            for x0 in range(0, before.shape[1] - max_dx):
                pred = before.copy()
                matched = True
                for dy, dx, before_value, after_value in offsets:
                    r = y0 + int(dy)
                    c = x0 + int(dx)
                    if int(pred[r, c]) != int(before_value):
                        matched = False
                        break
                    pred[r, c] = int(after_value)
                if matched:
                    return float(np.mean(pred != after))
        return 1.0

    @staticmethod
    def _reason(
        successor_plausible: bool,
        inverse_action: int | None,
        inverse_matches: bool,
        forward_error: float | None,
    ) -> str:
        if inverse_action is None:
            return "inverse_abstain_no_supported_signature"
        if not successor_plausible:
            return "successor_implausible_for_action"
        if not inverse_matches:
            return "inverse_action_mismatch"
        if forward_error is None:
            return "forward_abstain_no_replay_rule"
        return "forward_replay_mismatch"

    def _receipt(
        self,
        transition: ObservedTransition,
        features: _EffectFeatures,
        inverse_action: int | None,
        forward_error: float | None,
    ) -> JsonDict:
        payload = {
            "game": transition.game,
            "episode": transition.episode,
            "step": int(transition.step),
            "action": int(transition.action),
            "condition": transition.condition,
            "effect_core": repr(features.effect_core),
            "inverse_action": inverse_action,
            "forward_replay_error": forward_error,
        }
        return {
            **payload,
            "receipt_id": "cycle:" + _hash_payload(payload).split(":", 1)[1][:16],
            "immutable": True,
            "provenance": transition.provenance,
        }

    def _record_decision(
        self,
        condition: str,
        admitted: bool,
        abstained: bool,
        receipt: JsonDict | None,
    ) -> None:
        self._evaluated += 1
        if admitted:
            self._admitted += 1
            if condition != "valid":
                self._unsafe_accepts += 1
            if receipt is not None:
                self._receipts.append(receipt)
            return
        if abstained:
            self._abstained += 1
        else:
            self._rejected += 1

    def diagnostics(self) -> JsonDict:
        return {
            "observed_transition_count": int(self._observed),
            "evaluated_transition_count": int(self._evaluated),
            "admitted_update_count": int(self._admitted),
            "rejected_transition_count": int(self._rejected),
            "abstained_transition_count": int(self._abstained),
            "unsafe_transition_accept_count": int(self._unsafe_accepts),
            "immutable_update_receipts": list(self._receipts),
        }


def make_corrupted_transition(
    row: ObservedTransition,
    condition: str,
    *,
    replacement_action: int | None = None,
    replacement_successor: Any | None = None,
) -> ObservedTransition:
    """Build deterministic corruption controls without game-specific knowledge."""

    if condition == "permuted_action":
        return replace(
            row,
            action=int(replacement_action if replacement_action is not None else row.action + 1),
            data=None,
            condition=condition,
        )
    if condition == "mismatched_successor":
        return replace(row, successor=_grid(replacement_successor).copy(), condition=condition)
    if condition == "noop_substitution":
        return replace(row, successor=_grid(row.state).copy(), condition=condition)
    if condition == "wrong_object_change":
        before = _grid(row.state)
        after = before.copy()
        click = _click_xy(row.data)
        if click is not None:
            x, y = click
            r = min(after.shape[0] - 1, int(y) + 1)
            c = min(after.shape[1] - 1, int(x) + 1)
            after[r, c] = 5 if int(after[r, c]) != 5 else 4
        else:
            after = np.roll(_grid(row.successor), 1, axis=0)
        return replace(row, successor=after, condition=condition)
    raise ValueError(f"unknown corruption condition: {condition}")

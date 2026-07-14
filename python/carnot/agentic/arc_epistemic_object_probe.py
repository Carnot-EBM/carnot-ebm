"""Generic epistemic object-hypothesis probes for ARC live observations.

Spec refs: REQ-ARC-WMTE-5630,
SCENARIO-ARC-WMTE-5630-INFORMATIVE-PROBE-POSITIVE,
SCENARIO-ARC-WMTE-5630-NEGATIVE-AND-UNSAFE-REJECTION.

The planner is intentionally a development proxy, not a solver. It uses the same
evidence the live policy owns at runtime -- rendered grids, action ids, optional
click coordinates, and observed successors -- to keep competing object/effect
hypotheses alive long enough to ask one bounded, executable question.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np

from carnot.agentic.arc_color_blob_salience import (
    ColorBlob,
    blob_at_click,
    blob_topology,
    connected_color_blobs,
    object_hash,
)


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ObjectProbeObservation:
    """One agent-owned ARC transition row."""

    trace_id: str
    step: int
    state: Any
    action: int
    data: Any
    successor: Any
    level_before: int = 0
    level_after: int = 0
    provenance: str = "agent_owned_runtime_observation"


@dataclass(frozen=True)
class LiveProbeAction:
    """Action shape accepted by the live policy interface."""

    action: int
    data: Any = None

    def sort_key(self) -> tuple[int, str]:
        return int(self.action), _stable_json(self.data)

    def as_plan_step(self) -> JsonDict:
        return {"action": int(self.action), "data": self.data}


@dataclass(frozen=True)
class ObjectEffectHypothesis:
    """A generic object/effect explanation for observed transitions."""

    name: str
    mode: str
    action: int
    target_hash: str
    effect_signature: JsonDict
    support: int = 1
    accepted: bool = True
    rejection_reason: str = ""


@dataclass(frozen=True)
class ProbeScore:
    """Auditable score for one bounded executable probe."""

    action: LiveProbeAction
    executable: float
    expected_disagreement_reduction: float
    bounded_environment_cost: float
    score: float
    prediction_buckets: tuple[JsonDict, ...]

    def as_dict(self) -> JsonDict:
        return {
            "action": self.action.as_plan_step(),
            "executable": round(float(self.executable), 8),
            "expected_disagreement_reduction": round(
                float(self.expected_disagreement_reduction), 8
            ),
            "bounded_environment_cost": round(float(self.bounded_environment_cost), 8),
            "score": round(float(self.score), 8),
            "prediction_buckets": [dict(row) for row in self.prediction_buckets],
        }


@dataclass
class TraceObjectHypothesisModel:
    """Competing hypotheses and normalized weights for one trace."""

    trace_id: str
    hypotheses: list[ObjectEffectHypothesis]
    weights: dict[str, float]
    observation_count: int

    @property
    def is_non_degenerate(self) -> bool:
        positive = [value for value in self.weights.values() if float(value) > 0.0]
        return len(positive) >= 2 and not any(float(value) >= 0.999 for value in positive)

    @property
    def hypothesis_names(self) -> list[str]:
        return [hypothesis.name for hypothesis in self.hypotheses]

    def copy_with_weights(self, weights: Mapping[str, float]) -> "TraceObjectHypothesisModel":
        return TraceObjectHypothesisModel(
            trace_id=self.trace_id,
            hypotheses=list(self.hypotheses),
            weights=_normalise_weights(weights, self.hypothesis_names),
            observation_count=int(self.observation_count),
        )


@dataclass(frozen=True)
class _TransitionInfo:
    before: np.ndarray
    after: np.ndarray
    action: int
    data: Any
    clicked_blob: ColorBlob | None
    changed_blob: ColorBlob | None
    clicked_hash: str
    changed_hash: str
    effect_signature: JsonDict
    object_hashes: frozenset[str]


class EpistemicObjectProbePlanner:
    """REQ-ARC-WMTE-5630: bounded object-hypothesis causal-probe planner."""

    def __init__(
        self,
        *,
        max_depth: int = 1,
        cost_weight: float = 0.01,
        random_seed: int = 5630,
    ) -> None:
        self.max_depth = max(1, int(max_depth))
        self.cost_weight = max(0.0, float(cost_weight))
        self._rng = random.Random(int(random_seed))

    def build_trace_model(
        self,
        trace_id: str,
        observations: Sequence[ObjectProbeObservation],
    ) -> TraceObjectHypothesisModel:
        infos = [_transition_info(row) for row in observations]
        infos = [info for info in infos if info is not None]
        if not infos:
            return TraceObjectHypothesisModel(str(trace_id), [], {}, len(observations))
        info = infos[0]
        hypotheses = [
            ObjectEffectHypothesis(
                name="clicked_object_effect",
                mode="clicked_object",
                action=int(info.action),
                target_hash=info.clicked_hash,
                effect_signature=dict(info.effect_signature),
                support=len(infos),
            ),
            ObjectEffectHypothesis(
                name="observed_object_anchor_effect",
                mode="observed_object_anchor",
                action=int(info.action),
                target_hash=info.changed_hash,
                effect_signature=dict(info.effect_signature),
                support=len(infos),
            ),
        ]
        hypotheses = [hypothesis for hypothesis in hypotheses if _hypothesis_has_evidence(hypothesis)]
        weights = _normalise_weights({hypothesis.name: 1.0 for hypothesis in hypotheses}, [])
        return TraceObjectHypothesisModel(
            trace_id=str(trace_id),
            hypotheses=hypotheses,
            weights=weights,
            observation_count=len(observations),
        )

    def score_probes(
        self,
        model: TraceObjectHypothesisModel,
        current_grid: Any,
        legal_actions: Sequence[LiveProbeAction],
    ) -> list[ProbeScore]:
        if not model.is_non_degenerate:
            return []
        grid = _as_grid(current_grid)
        scores = [
            self._score_one_probe(model, grid, action)
            for action in _dedupe_actions(legal_actions)
        ]
        return sorted(
            scores,
            key=lambda row: (
                -float(row.score),
                -float(row.expected_disagreement_reduction),
                row.action.sort_key(),
            ),
        )

    def choose_probe(
        self,
        model: TraceObjectHypothesisModel,
        current_grid: Any,
        legal_actions: Sequence[LiveProbeAction],
    ) -> ProbeScore | None:
        scores = self.score_probes(model, current_grid, legal_actions)
        if not scores or scores[0].expected_disagreement_reduction <= 0.0:
            return None
        return scores[0]

    def observe_probe_result(
        self,
        model: TraceObjectHypothesisModel,
        observation: ObjectProbeObservation,
    ) -> JsonDict:
        before = _entropy(model.weights.values())
        observed_info = _transition_info(observation)
        updated: dict[str, float] = {}
        matched: list[str] = []
        for hypothesis in model.hypotheses:
            prior = float(model.weights.get(hypothesis.name, 0.0))
            if _hypothesis_matches_observation(hypothesis, observed_info):
                likelihood = 0.995
                matched.append(hypothesis.name)
            else:
                likelihood = 0.005
            updated[hypothesis.name] = prior * likelihood
        model.weights = _normalise_weights(updated, model.hypothesis_names)
        after = _entropy(model.weights.values())
        return {
            "posterior_entropy_before": round(float(before), 8),
            "posterior_entropy_after": round(float(after), 8),
            "posterior_entropy_reduction": round(max(0.0, before - after), 8),
            "matched_hypotheses": matched,
            "posterior": {name: round(float(value), 8) for name, value in model.weights.items()},
        }

    def compare_controls(
        self,
        model: TraceObjectHypothesisModel,
        current_grid: Any,
        legal_actions: Sequence[LiveProbeAction],
        *,
        observed: ObjectProbeObservation,
    ) -> JsonDict:
        scores = self.score_probes(model, current_grid, legal_actions)
        if not scores:
            return {
                "informative_entropy_reduction": 0.0,
                "random_control_entropy_reduction": 0.0,
                "informative_control_delta": 0.0,
                "uninformative_control_delta": 0.0,
                "live_interface_replay_rate": 0.0,
            }
        random_expected = sum(
            float(score.expected_disagreement_reduction) for score in scores
        ) / float(len(scores))
        chosen = scores[0]
        update = self.observe_probe_result(model, observed)
        informative = float(update["posterior_entropy_reduction"])
        uninformative_best = max(
            (
                float(score.expected_disagreement_reduction)
                for score in scores
                if score.action.action != observed.action
            ),
            default=0.0,
        )
        return {
            "informative_entropy_reduction": round(informative, 8),
            "random_control_entropy_reduction": round(float(random_expected), 8),
            "informative_control_delta": round(informative - float(random_expected), 8),
            "uninformative_control_delta": round(uninformative_best - float(random_expected), 8),
            "live_interface_replay_rate": float(
                chosen.action.sort_key()
                in {action.sort_key() for action in _dedupe_actions(legal_actions)}
            ),
        }

    def reject_unsafe_models(
        self,
        model: TraceObjectHypothesisModel,
        hypotheses: Sequence[ObjectEffectHypothesis],
        current_grid: Any,
    ) -> int:
        grid = _as_grid(current_grid)
        observed_hashes = _object_hashes(grid)
        unsafe_accept_count = 0
        names = set(model.hypothesis_names)
        for hypothesis in hypotheses:
            if _hypothesis_is_safe(hypothesis, observed_hashes, names):
                unsafe_accept_count += 1
        return int(unsafe_accept_count)

    def predict_successor(
        self,
        hypothesis: ObjectEffectHypothesis,
        grid: Any,
        action: LiveProbeAction,
    ) -> np.ndarray:
        before = _as_grid(grid)
        after = before.copy()
        if int(action.action) != int(hypothesis.action) or not hypothesis.accepted:
            return after
        blobs = connected_color_blobs(before, min_pixels=1, max_component_fraction=0.45)
        target: ColorBlob | None
        if hypothesis.mode == "clicked_object":
            xy = _click_xy(action.data)
            target = blob_at_click(blobs, xy[0], xy[1]) if xy is not None else None
            return _apply_effect(after, target, hypothesis.effect_signature, ignore_source=True)
        if hypothesis.mode == "observed_object_anchor":
            target = next(
                (blob for blob in blobs if object_hash(blob) == hypothesis.target_hash),
                None,
            )
            return _apply_effect(after, target, hypothesis.effect_signature, ignore_source=False)
        return after

    def _score_one_probe(
        self,
        model: TraceObjectHypothesisModel,
        grid: np.ndarray,
        action: LiveProbeAction,
    ) -> ProbeScore:
        buckets = _prediction_buckets(
            model,
            [
                (
                    hypothesis.name,
                    self.predict_successor(hypothesis, grid, action),
                )
                for hypothesis in model.hypotheses
            ],
        )
        before = _entropy(model.weights.values())
        expected_after = 0.0
        for bucket in buckets:
            probability = float(bucket["probability"])
            if probability > 0.0:
                conditional = [
                    float(model.weights[name]) / probability for name in bucket["hypotheses"]
                ]
                expected_after += probability * _entropy(conditional)
        disagreement = max(0.0, before - expected_after)
        cost = _bounded_cost(action, self.max_depth)
        score = 1.0 * (disagreement - self.cost_weight * cost)
        return ProbeScore(
            action=action,
            executable=1.0,
            expected_disagreement_reduction=round(float(disagreement), 12),
            bounded_environment_cost=round(float(cost), 12),
            score=round(float(score), 12),
            prediction_buckets=tuple(buckets),
        )


def make_corrupted_effect_hypothesis(hypothesis: ObjectEffectHypothesis) -> ObjectEffectHypothesis:
    """Build an adversarial effect candidate with an impossible color."""

    corrupted = dict(hypothesis.effect_signature)
    corrupted["destination_color"] = -999
    return replace(hypothesis, effect_signature=corrupted)


def make_hallucinated_object_hypothesis(target_hash: str) -> ObjectEffectHypothesis:
    """Build an adversarial object candidate whose target is not in the frame."""

    return ObjectEffectHypothesis(
        name="hallucinated_object_effect",
        mode="observed_object_anchor",
        action=6,
        target_hash=str(target_hash),
        effect_signature={"source_color": 1, "destination_color": 2, "changed_count": 1},
    )


def _transition_info(row: ObjectProbeObservation) -> _TransitionInfo | None:
    before = _as_grid(row.state)
    after = _as_grid(row.successor)
    if before.shape != after.shape or before.size == 0 or not np.any(before != after):
        return None
    blobs = connected_color_blobs(before, min_pixels=1, max_component_fraction=0.45)
    xy = _click_xy(row.data)
    clicked = blob_at_click(blobs, xy[0], xy[1]) if xy is not None else None
    changed_cells = set(zip(*np.where(before != after), strict=False))
    changed = [blob for blob in blobs if blob.cells & changed_cells]
    changed_blob = max(changed, key=lambda blob: len(blob.cells & changed_cells), default=None)
    if clicked is None or changed_blob is None:
        return None
    topo = blob_topology(before)
    object_hashes = frozenset(str(value) for value in topo["object_hashes"].values())
    signature = _effect_signature(before, after, changed_cells)
    return _TransitionInfo(
        before=before,
        after=after,
        action=int(row.action),
        data=row.data,
        clicked_blob=clicked,
        changed_blob=changed_blob,
        clicked_hash=object_hash(clicked),
        changed_hash=object_hash(changed_blob),
        effect_signature=signature,
        object_hashes=object_hashes,
    )


def _effect_signature(
    before: np.ndarray,
    after: np.ndarray,
    changed_cells: set[tuple[int, int]],
) -> JsonDict:
    pairs: dict[tuple[int, int], int] = defaultdict(int)
    for y, x in changed_cells:
        pairs[(int(before[y, x]), int(after[y, x]))] += 1
    (source, dest), count = max(pairs.items(), key=lambda item: (item[1], item[0]))
    return {
        "source_color": int(source),
        "destination_color": int(dest),
        "changed_count": int(len(changed_cells)),
        "dominant_pair_count": int(count),
    }


def _apply_effect(
    grid: np.ndarray,
    blob: ColorBlob | None,
    signature: Mapping[str, Any],
    *,
    ignore_source: bool,
) -> np.ndarray:
    if blob is None:
        return grid
    dest = int(signature.get("destination_color", -1))
    source = int(signature.get("source_color", -1))
    if dest < 0:
        return grid
    for y, x in blob.cells:
        if ignore_source or int(grid[y, x]) == source:
            grid[y, x] = dest
    return grid


def _hypothesis_has_evidence(hypothesis: ObjectEffectHypothesis) -> bool:
    return bool(hypothesis.target_hash) and int(hypothesis.effect_signature["destination_color"]) >= 0


def _hypothesis_is_safe(
    hypothesis: ObjectEffectHypothesis,
    observed_hashes: frozenset[str],
    model_names: set[str],
) -> bool:
    if hypothesis.name not in model_names and hypothesis.mode != "clicked_object":
        return False
    if not hypothesis.accepted:
        return False
    try:
        dest = int(hypothesis.effect_signature["destination_color"])
        source = int(hypothesis.effect_signature["source_color"])
    except Exception:
        return False
    if dest < 0 or source < 0:
        return False
    if hypothesis.mode == "observed_object_anchor" and hypothesis.target_hash not in observed_hashes:
        return False
    return True


def _hypothesis_matches_observation(
    hypothesis: ObjectEffectHypothesis,
    observed: _TransitionInfo | None,
) -> bool:
    if observed is None:
        return False
    if hypothesis.mode == "clicked_object":
        return observed.clicked_hash == observed.changed_hash
    if hypothesis.mode == "observed_object_anchor":
        return observed.changed_hash == hypothesis.target_hash
    return False


def _prediction_buckets(
    model: TraceObjectHypothesisModel,
    predictions: Sequence[tuple[str, np.ndarray]],
) -> list[JsonDict]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name, prediction in predictions:
        grouped[_grid_signature(prediction)].append(name)
    buckets: list[JsonDict] = []
    for signature, names in grouped.items():
        probability = sum(float(model.weights.get(name, 0.0)) for name in names)
        buckets.append(
            {
                "signature": signature[:160],
                "probability": round(float(probability), 8),
                "hypotheses": sorted(names),
                "n_hypotheses": len(names),
            }
        )
    return sorted(buckets, key=lambda row: (-float(row["probability"]), row["signature"]))


def _dedupe_actions(actions: Sequence[LiveProbeAction]) -> list[LiveProbeAction]:
    out: list[LiveProbeAction] = []
    seen: set[tuple[int, str]] = set()
    for action in actions:
        key = action.sort_key()
        if key in seen:
            continue
        seen.add(key)
        out.append(action)
    return out


def _object_hashes(grid: np.ndarray) -> frozenset[str]:
    return frozenset(
        object_hash(blob)
        for blob in connected_color_blobs(grid, min_pixels=1, max_component_fraction=0.45)
    )


def _bounded_cost(action: LiveProbeAction, max_depth: int) -> float:
    base = 1.0 if int(action.action) == 6 else 0.25
    return float(base) / float(max(1, int(max_depth)))


def _normalise_weights(weights: Mapping[str, float], names: Sequence[str]) -> dict[str, float]:
    keys = list(names) if names else list(weights)
    total = sum(max(0.0, float(weights.get(name, 0.0))) for name in keys)
    if total <= 0.0 and keys:
        uniform = 1.0 / float(len(keys))
        return {name: uniform for name in keys}
    if total <= 0.0:
        return {}
    return {
        name: max(0.0, float(weights.get(name, 0.0))) / total
        for name in keys
    }


def _entropy(values: Sequence[float] | Any) -> float:
    total = 0.0
    for value in values:
        probability = float(value)
        if probability > 0.0:
            total -= probability * math.log(probability)
    return float(total)


def _click_xy(data: Any) -> tuple[int, int] | None:
    if not isinstance(data, Mapping) or "x" not in data or "y" not in data:
        return None
    try:
        return int(data["x"]), int(data["y"])
    except Exception:
        return None


def _as_grid(value: Any) -> np.ndarray:
    arr = np.asarray(value.frame if hasattr(value, "frame") else value)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        return np.zeros((0, 0), dtype=np.int16)
    return arr.astype(np.int16, copy=False)


def _grid_signature(grid: Any) -> str:
    arr = _as_grid(grid)
    payload = {"shape": list(arr.shape), "values": arr.astype(int, copy=False).tolist()}
    return _stable_json(payload)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True)


def stable_checksum(value: Any) -> str:
    """Stable content hash for artifacts and trace receipts."""

    return "sha256:" + hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()

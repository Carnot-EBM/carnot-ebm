"""Active-probe posterior controller for ARC executable world models.

Spec refs: REQ-ARC-WMTE-4727,
SCENARIO-ARC-WMTE-4727-ACTIVE-PROBE-SPLITS-POSTERIOR.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from carnot.agentic.arc_executable_world_model import (
    Transition,
    predict_hypothesis_transition,
)
from carnot.agentic.arc_world_model_trust_energy import WorldModelCandidate


Predictor = Callable[[WorldModelCandidate, np.ndarray, int, Any], np.ndarray]


@dataclass(frozen=True)
class ProbeAction:
    """One candidate live action the controller may use as a discriminator."""

    action: int
    data: Any = None

    def as_plan_step(self) -> dict[str, Any]:
        return {"action": int(self.action), "data": self.data}

    def sort_key(self) -> tuple[int, str]:
        return int(self.action), json.dumps(self.data, sort_keys=True, default=str)


@dataclass(frozen=True)
class PredictionRow:
    hypothesis_name: str
    probability: float
    signature: str
    prediction: np.ndarray | None = None
    error: str = ""


@dataclass(frozen=True)
class ProbeScore:
    action: ProbeAction
    expected_information_gain: float
    energy_score: float
    prediction_buckets: tuple[dict[str, Any], ...]
    verifier_is_oracle: bool = False


@dataclass(frozen=True)
class PosteriorUpdate:
    action: ProbeAction
    posterior_entropy_before: float
    posterior_entropy_after: float
    posterior_entropy_reduction: float
    matched_hypotheses: tuple[str, ...]


@dataclass
class HypothesisPosterior:
    """Small categorical posterior over executable goal/dynamics hypotheses."""

    candidates: dict[str, WorldModelCandidate]
    probabilities: dict[str, float]

    def __post_init__(self) -> None:
        self._normalise()

    def _normalise(self) -> None:
        total = sum(max(0.0, float(p)) for p in self.probabilities.values())
        if total <= 0.0:
            uniform = 1.0 / max(1, len(self.candidates))
            self.probabilities = {name: uniform for name in self.candidates}
            return
        self.probabilities = {
            name: max(0.0, float(self.probabilities.get(name, 0.0))) / total
            for name in self.candidates
        }

    def entropy(self) -> float:
        return _entropy(self.probabilities.values())

    def probability(self, name: str) -> float:
        return float(self.probabilities.get(name, 0.0))

    def best_candidate(self) -> WorldModelCandidate | None:
        if not self.candidates:
            return None
        name = max(
            self.candidates,
            key=lambda key: (self.probability(key), key),
        )
        return self.candidates[name]

    def concentrated(self, threshold: float) -> bool:
        if not self.probabilities:
            return False
        return max(self.probabilities.values()) >= float(threshold)

    def as_dict(self) -> dict[str, float]:
        return {name: round(float(prob), 8) for name, prob in sorted(self.probabilities.items())}


class TransitionSplitEnergyVerifier:
    """Oracle-distinct energy for ranking probes by transition discrimination."""

    verifier_is_oracle = False

    def score_probe(
        self,
        *,
        expected_information_gain: float,
        prediction_buckets: Sequence[Mapping[str, Any]],
    ) -> float:
        return float(expected_information_gain) + 1.0e-9 * float(len(prediction_buckets))


@dataclass
class ActiveProbeController:
    """REQ-ARC-WMTE-4727: posterior-split probe chooser and updater."""

    posterior: HypothesisPosterior
    probe_budget: int = 2
    concentration_threshold: float = 0.9
    energy_verifier: Any = field(default_factory=TransitionSplitEnergyVerifier)
    history: list[dict[str, Any]] = field(default_factory=list)

    def rank_probe_actions(
        self,
        grid: np.ndarray,
        actions: Sequence[ProbeAction],
        *,
        predictor: Predictor = predict_hypothesis_transition,
    ) -> list[ProbeScore]:
        scores = [
            self._score_probe(np.asarray(grid), action, predictor=predictor)
            for action in list(actions)
        ]
        return sorted(
            scores,
            key=lambda score: (
                -float(score.energy_score),
                -float(score.expected_information_gain),
                score.action.sort_key(),
            ),
        )

    def choose_probe(
        self,
        grid: np.ndarray,
        actions: Sequence[ProbeAction],
        *,
        predictor: Predictor = predict_hypothesis_transition,
    ) -> ProbeScore | None:
        if len(self.history) >= int(self.probe_budget):
            return None
        if self.posterior.concentrated(self.concentration_threshold):
            return None
        ranked = self.rank_probe_actions(grid, actions, predictor=predictor)
        return ranked[0] if ranked else None

    def observe_transition(
        self,
        start_grid: np.ndarray,
        action: ProbeAction,
        observed_next_grid: np.ndarray,
        *,
        predictor: Predictor = predict_hypothesis_transition,
        match_likelihood: float = 0.995,
        mismatch_likelihood: float = 0.005,
    ) -> PosteriorUpdate:
        before = self.posterior.entropy()
        observed_sig = _grid_signature(observed_next_grid)
        matched: list[str] = []
        updated: dict[str, float] = {}
        for name, candidate in self.posterior.candidates.items():
            prior = self.posterior.probability(name)
            try:
                pred = predictor(candidate, np.asarray(start_grid), action.action, action.data)
                sig = _grid_signature(pred)
                likelihood = match_likelihood if sig == observed_sig else mismatch_likelihood
                if sig == observed_sig:
                    matched.append(name)
            except Exception:
                likelihood = mismatch_likelihood
            updated[name] = prior * float(likelihood)
        self.posterior.probabilities = updated
        self.posterior._normalise()
        after = self.posterior.entropy()
        update = PosteriorUpdate(
            action=action,
            posterior_entropy_before=before,
            posterior_entropy_after=after,
            posterior_entropy_reduction=max(0.0, before - after),
            matched_hypotheses=tuple(matched),
        )
        self.history.append(
            {
                "action": action.as_plan_step(),
                "posterior_entropy_before": round(float(before), 8),
                "posterior_entropy_after": round(float(after), 8),
                "posterior_entropy_reduction": round(float(update.posterior_entropy_reduction), 8),
                "matched_hypotheses": list(matched),
                "posterior": self.posterior.as_dict(),
            }
        )
        return update

    def diagnostics(self) -> dict[str, Any]:
        reduction = sum(float(row.get("posterior_entropy_reduction") or 0.0) for row in self.history)
        return {
            "hypothesis_posterior_built": bool(self.posterior.candidates),
            "n_hypotheses": int(len(self.posterior.candidates)),
            "posterior": self.posterior.as_dict(),
            "probe_actions_taken": int(len(self.history)),
            "posterior_entropy_reduction": round(float(reduction), 8),
            "concentrated": self.posterior.concentrated(self.concentration_threshold),
            "trace": list(self.history),
            "verifier_is_oracle": False,
        }

    def _score_probe(
        self,
        grid: np.ndarray,
        action: ProbeAction,
        *,
        predictor: Predictor,
    ) -> ProbeScore:
        rows = _prediction_rows(self.posterior, grid, action, predictor)
        buckets = _prediction_buckets(rows)
        before = self.posterior.entropy()
        expected_after = 0.0
        for bucket in buckets:
            bucket_prob = float(bucket["probability"])
            conditional = [
                self.posterior.probability(name) / bucket_prob
                for name in bucket["hypotheses"]
                if bucket_prob > 0.0
            ]
            expected_after += bucket_prob * _entropy(conditional)
        info_gain = max(0.0, before - expected_after)
        verifier = self.energy_verifier
        if hasattr(verifier, "score_probe"):
            energy = float(
                verifier.score_probe(
                    expected_information_gain=info_gain,
                    prediction_buckets=buckets,
                )
            )
            verifier_is_oracle = bool(getattr(verifier, "verifier_is_oracle", False))
        elif callable(verifier):
            energy = float(verifier(info_gain, buckets))
            verifier_is_oracle = False
        else:
            energy = float(info_gain)
            verifier_is_oracle = False
        return ProbeScore(
            action=action,
            expected_information_gain=round(float(info_gain), 12),
            energy_score=round(float(energy), 12),
            prediction_buckets=tuple(buckets),
            verifier_is_oracle=verifier_is_oracle,
        )


def make_hypothesis_posterior(
    candidates: Sequence[WorldModelCandidate],
    *,
    priors: Mapping[str, float] | None = None,
) -> HypothesisPosterior:
    deduped: dict[str, WorldModelCandidate] = {}
    for index, candidate in enumerate(candidates):
        name = str(candidate.name or f"hypothesis_{index}")
        if name in deduped:
            name = f"{name}_{index}"
        deduped[name] = candidate
    weights = {
        name: float(priors[name]) if priors and name in priors else 1.0 for name in deduped
    }
    return HypothesisPosterior(deduped, weights)


def augment_with_transition_baselines(
    candidates: Sequence[WorldModelCandidate],
    transitions: Sequence[Transition],
    *,
    max_candidates: int = 4,
) -> list[WorldModelCandidate]:
    """Add cheap non-oracle hypotheses so a one-model pool can still be probed."""

    out = list(candidates)
    names = {candidate.name for candidate in out}
    if "noop_transition_hypothesis" not in names:
        out.append(WorldModelCandidate("noop_transition_hypothesis", _noop_engine))
    if "click_point_toggle_hypothesis" not in {candidate.name for candidate in out}:
        out.append(WorldModelCandidate("click_point_toggle_hypothesis", _click_point_toggle_engine))
    exact = _exact_action_delta_candidate(transitions)
    if exact is not None and exact.name not in {candidate.name for candidate in out}:
        out.append(exact)
    return out[: max(1, int(max_candidates))]


def probe_actions_from_model_candidates(rows: Sequence[Mapping[str, Any]]) -> list[ProbeAction]:
    actions: list[ProbeAction] = []
    seen: set[tuple[int, str]] = set()
    for row in rows:
        try:
            action = ProbeAction(int(row["action"]), row.get("data"))
        except Exception:
            continue
        key = action.sort_key()
        if key in seen:
            continue
        seen.add(key)
        actions.append(action)
    return actions


def _prediction_rows(
    posterior: HypothesisPosterior,
    grid: np.ndarray,
    action: ProbeAction,
    predictor: Predictor,
) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    for name, candidate in posterior.candidates.items():
        try:
            prediction = np.asarray(predictor(candidate, grid, action.action, action.data))
            signature = _grid_signature(prediction)
            rows.append(
                PredictionRow(
                    hypothesis_name=name,
                    probability=posterior.probability(name),
                    signature=signature,
                    prediction=prediction,
                )
            )
        except Exception as exc:
            rows.append(
                PredictionRow(
                    hypothesis_name=name,
                    probability=posterior.probability(name),
                    signature=f"error:{type(exc).__name__}:{repr(exc)[:80]}",
                    error=repr(exc)[:160],
                )
            )
    return rows


def _prediction_buckets(rows: Sequence[PredictionRow]) -> list[dict[str, Any]]:
    grouped: dict[str, list[PredictionRow]] = defaultdict(list)
    for row in rows:
        grouped[row.signature].append(row)
    buckets: list[dict[str, Any]] = []
    for signature, bucket_rows in grouped.items():
        probability = sum(float(row.probability) for row in bucket_rows)
        buckets.append(
            {
                "signature": signature[:160],
                "probability": round(float(probability), 8),
                "hypotheses": [row.hypothesis_name for row in bucket_rows],
                "n_hypotheses": len(bucket_rows),
            }
        )
    return sorted(buckets, key=lambda row: (-float(row["probability"]), row["signature"]))


def _grid_signature(grid: Any) -> str:
    arr = np.asarray(grid)
    payload = {
        "shape": list(arr.shape),
        "values": arr.astype(int, copy=False).tolist(),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _entropy(values: Sequence[float] | Any) -> float:
    total = 0.0
    for value in values:
        p = float(value)
        if p > 0.0:
            total -= p * math.log(p)
    return float(total)


def _noop_engine(grid: np.ndarray, _action: int, _data: Any = None) -> np.ndarray:
    return np.asarray(grid).copy()


def _click_point_toggle_engine(grid: np.ndarray, action: int, data: Any = None) -> np.ndarray:
    out = np.asarray(grid).copy()
    if int(action) != 6 or not isinstance(data, Mapping):
        return out
    try:
        x = int(data.get("x", 0))
        y = int(data.get("y", 0))
    except Exception:
        return out
    if out.size == 0:
        return out
    row = max(0, min(out.shape[0] - 1, y))
    col = max(0, min(out.shape[1] - 1, x))
    colors = sorted({int(v) for v in out.flatten().tolist()})
    if len(colors) <= 1:
        out[row, col] = int(out[row, col]) + 1
        return out
    current = int(out[row, col])
    try:
        index = colors.index(current)
    except ValueError:
        index = 0
    out[row, col] = colors[(index + 1) % len(colors)]
    return out


def _exact_action_delta_candidate(
    transitions: Sequence[Transition],
) -> WorldModelCandidate | None:
    rows = [t for t in transitions if not np.array_equal(t.grid, t.next_grid)]
    if not rows:
        return None
    by_action: dict[int, Transition] = {}
    for transition in rows:
        by_action.setdefault(int(transition.action), transition)

    def _engine(grid: np.ndarray, action: int, data: Any = None) -> np.ndarray:
        out = np.asarray(grid).copy()
        transition = by_action.get(int(action))
        if transition is None:
            return out
        before = np.asarray(transition.grid)
        after = np.asarray(transition.next_grid)
        if before.shape != out.shape or after.shape != out.shape:
            return out
        changed = before != after
        if data is not None and transition.data is not None and data != transition.data:
            return out
        out[changed] = after[changed]
        return out

    return WorldModelCandidate("observed_action_delta_hypothesis", _engine)

"""Live graded goal-energy compiled from Exp4020's visible-state predicate.

Spec refs: REQ-ARC-WMTE-4640, SCENARIO-ARC-WMTE-4640.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot.agentic.arc_goal_predicate_separation import compile_goal_predicate


DEFAULT_EXP4020_ARTIFACT = Path("results/experiment_4020_goal_induction_separation.json")
GOAL_ENERGY_SOURCE = "exp4020_graded_goal_satisfaction_energy"
RELATIONAL_GOAL_ENERGY_SOURCE = "arc_visible_state_relational_energy_no_llm"
RELATIONAL_GOAL_VARIANCE_FLOOR = 1e-12
SUPPORTED_RELATIONAL_ROUTE_CLASSES = (
    "region_pair_equality",
    "translated_within_frame_target_match",
    "ordered_run_relation",
    "centroid_alignment",
)


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _state_from_visible(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    for attr in ("goal_state", "visible_goal_state", "target_group_state"):
        state = getattr(value, attr, None)
        if isinstance(state, Mapping):
            return state
    return None


def _candidate_field(candidate: Any, key: str, default: Any = None) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get(key, default)
    return getattr(candidate, key, default)


def _candidate_action(candidate: Any) -> int:
    value = _candidate_field(candidate, "action", _candidate_field(candidate, "action_id", 0))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _candidate_data(candidate: Any) -> Any:
    return _candidate_field(candidate, "data")


def _candidate_signature(candidate: Any) -> tuple[Any, str]:
    data = _candidate_data(candidate)
    return (
        _candidate_action(candidate),
        json.dumps(data, sort_keys=True, separators=(",", ":"), default=str),
    )


def _candidate_navigation_energy(candidate: Any, index: int, total: int) -> float:
    for key in (
        "navigation_energy",
        "arc_goal_distance",
        "goal_distance",
        "navigation",
        "heuristic",
        "search_energy",
    ):
        value = _candidate_field(candidate, key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    if total <= 1:
        return 0.0
    return float(index) / float(total - 1)


def _candidate_state_from_row(candidate: Any) -> Any:
    for key in (
        "candidate_state",
        "predicted_candidate_state",
        "next_state",
        "next_frame",
        "goal_state",
        "visible_goal_state",
        "target_group_state",
        "state",
        "frame",
    ):
        value = _candidate_field(candidate, key)
        if value is not None:
            return value
    return None


def _state_hash(value: Any) -> str:
    try:
        import numpy as np

        arr = np.asarray(value.frame if hasattr(value, "frame") else value)
        if arr.ndim >= 1 and arr.dtype != object:
            payload = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
            }
            return hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:16]
    except Exception:
        pass
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def _score_variance(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    mean = sum(float(score) for score in scores) / float(len(scores))
    return sum((float(score) - mean) ** 2 for score in scores) / float(len(scores))


def _grid_from_value(value: Any) -> Any | None:
    try:
        import numpy as np

        raw = None
        if isinstance(value, Mapping):
            for key in ("frame", "grid", "candidate_state", "next_state", "state"):
                if key in value:
                    raw = value.get(key)
                    break
        if raw is None:
            for attr in ("frame", "grid"):
                if hasattr(value, attr):
                    raw = getattr(value, attr)
                    break
        if raw is None:
            return None
        arr = np.asarray(raw)
        return arr if arr.ndim == 2 else None
    except Exception:
        return None


def _relational_receipt_from_value(value: Any) -> Mapping[str, Any] | None:
    for key in ("relational_goal_receipt", "relational_goal", "agent_goal_receipt"):
        if isinstance(value, Mapping):
            receipt = value.get(key)
        else:
            receipt = getattr(value, key, None)
        if isinstance(receipt, Mapping):
            return receipt
    return None


def _bool_mask(value: Any, shape: tuple[int, int]) -> Any | None:
    try:
        import numpy as np

        arr = np.asarray(value, dtype=bool)
        return arr if arr.shape == shape else None
    except Exception:
        return None


def _mask_coords(mask: Any) -> list[tuple[int, int]]:
    try:
        import numpy as np

        return [(int(y), int(x)) for y, x in np.argwhere(mask)]
    except Exception:
        return []


def _dominant_background(arr: Any) -> Any:
    import numpy as np

    vals, counts = np.unique(arr, return_counts=True)
    return vals[int(counts.argmax())]


def _local_centroid(arr: Any, mask: Any) -> tuple[float, float] | None:
    import numpy as np

    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    bg = _dominant_background(arr)
    active = np.argwhere(mask & (arr != bg))
    if active.size == 0:
        return None
    origin = coords.min(axis=0)
    local = active - origin
    return (float(local[:, 0].mean()), float(local[:, 1].mean()))


@dataclass
class GoalEnergyCandidateGuidance:
    """REQ-ARC-WMTE-4737: score predicted candidate states and bias proposal order.

    The guidance is deliberately fail-closed: it only reorders when it has scored
    real predicted candidate states, the scores have non-zero variance, and the
    resulting rank differs from the baseline pool. Otherwise it returns the
    incoming candidate order unchanged and records the degenerate diagnostic.
    """

    goal_energy: Any
    transition_predictor: Any | None = None
    alpha: float = 0.0
    beta: float = 1.0
    lower_is_better: bool = True
    source: str = "goal_energy_candidate_generation_guidance"
    verifier_is_oracle: bool = False
    _candidate_states_scored_total: int = field(default=0, init=False, repr=False)
    _prediction_errors_total: int = field(default=0, init=False, repr=False)
    _scoring_errors_total: int = field(default=0, init=False, repr=False)
    _last_diagnostics: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        total = float(self.alpha) + float(self.beta)
        if abs(total - 1.0) > 1e-9:
            raise ValueError("candidate goal guidance requires alpha + beta == 1")

    def set_goal_energy(self, goal_energy: Any) -> None:
        self.goal_energy = goal_energy

    def _predict_state(self, frame: Any, candidate: Any) -> Any:
        inline_state = _candidate_state_from_row(candidate)
        if inline_state is not None:
            return inline_state
        predictor = self.transition_predictor
        if predictor is None:
            return None
        if hasattr(predictor, "candidate_state"):
            return predictor.candidate_state(frame, candidate)
        if hasattr(predictor, "predict_candidate_state"):
            return predictor.predict_candidate_state(frame, candidate)
        if hasattr(predictor, "predict"):
            from carnot.agentic.arc_agi3_world_model import grid_of

            grid = grid_of(frame)
            action = _candidate_action(candidate)
            data = _candidate_data(candidate)
            akey = (
                (6, int(data["x"]), int(data["y"]))
                if int(action) == 6 and isinstance(data, Mapping)
                else (int(action),)
            )
            return predictor.predict(grid, akey)
        if callable(predictor):
            try:
                return predictor(frame, candidate)
            except TypeError:
                return predictor(frame, _candidate_action(candidate), _candidate_data(candidate))
        return None

    def _score_state(self, state: Any) -> float:
        if self.goal_energy is None:
            return 1.0
        return float(self.goal_energy(state))

    def rank_candidates(self, frame: Any, candidates: Sequence[Any]) -> list[dict[str, Any]]:
        rows = [dict(row) if isinstance(row, Mapping) else {"action": _candidate_action(row), "data": _candidate_data(row)} for row in candidates]
        baseline_signatures = [_candidate_signature(row) for row in rows]
        scored_rows: list[tuple[float, float, int, dict[str, Any]]] = []
        scores: list[float] = []
        state_hashes: list[str] = []
        score_elapsed_s = 0.0
        prediction_errors = 0
        scoring_errors = 0
        total = len(rows)

        for index, row in enumerate(rows):
            try:
                state = self._predict_state(frame, row)
            except Exception:
                prediction_errors += 1
                continue
            if state is None:
                continue
            try:
                started = time.perf_counter()
                raw_goal_score = self._score_state(state)
                score_elapsed_s += max(0.0, time.perf_counter() - started)
            except Exception:
                scoring_errors += 1
                continue
            goal_key = float(raw_goal_score) if self.lower_is_better else -float(raw_goal_score)
            navigation = _candidate_navigation_energy(row, index, total)
            combined = float(self.alpha) * float(navigation) + float(self.beta) * goal_key
            annotated = dict(row)
            annotated["goal_energy_score"] = float(raw_goal_score)
            annotated["goal_energy_navigation"] = float(navigation)
            annotated["combined_goal_energy"] = float(combined)
            annotated["predicted_candidate_state_hash"] = _state_hash(state)
            scored_rows.append((combined, navigation, index, annotated))
            scores.append(float(raw_goal_score))
            state_hashes.append(annotated["predicted_candidate_state_hash"])

        variance = _score_variance(scores)
        real_state_evidence = bool(scored_rows and len(set(state_hashes)) > 1)
        ranked_rows = [row for _combined, _nav, _index, row in sorted(scored_rows, key=lambda item: (item[0], item[2]))]
        if len(ranked_rows) < len(rows):
            scored_indexes = {index for _combined, _nav, index, _row in scored_rows}
            ranked_rows.extend(dict(row) for index, row in enumerate(rows) if index not in scored_indexes)
        ranked_signatures = [_candidate_signature(row) for row in ranked_rows]
        pool_differs = baseline_signatures != ranked_signatures
        arms_non_degenerate = bool(real_state_evidence and variance > 1e-12 and pool_differs)

        self._candidate_states_scored_total += len(scored_rows)
        self._prediction_errors_total += prediction_errors
        self._scoring_errors_total += scoring_errors
        cpu_ms = (
            (score_elapsed_s * 1000.0) / float(len(scored_rows))
            if scored_rows
            else 0.0
        )
        self._last_diagnostics = {
            "enabled": True,
            "source": self.source,
            "verifier_is_oracle": False,
            "candidate_count": int(len(rows)),
            "candidate_states_scored": int(len(scored_rows)),
            "candidate_states_scored_total": int(self._candidate_states_scored_total),
            "prediction_errors": int(prediction_errors),
            "prediction_errors_total": int(self._prediction_errors_total),
            "scoring_errors": int(scoring_errors),
            "scoring_errors_total": int(self._scoring_errors_total),
            "real_candidate_state_evidence": bool(real_state_evidence),
            "goal_energy_score_variance": float(variance),
            "candidate_pool_differs_from_baseline": bool(pool_differs if variance > 1e-12 else False),
            "arms_non_degenerate": bool(arms_non_degenerate),
            "cpu_scoring_ms_per_candidate": float(cpu_ms),
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
        }
        if not arms_non_degenerate:
            return [dict(row) for row in rows]
        return ranked_rows

    def diagnostics(self) -> dict[str, Any]:
        if self._last_diagnostics:
            return dict(self._last_diagnostics)
        return {
            "enabled": True,
            "source": self.source,
            "verifier_is_oracle": False,
            "candidate_count": 0,
            "candidate_states_scored": 0,
            "candidate_states_scored_total": int(self._candidate_states_scored_total),
            "prediction_errors": 0,
            "prediction_errors_total": int(self._prediction_errors_total),
            "scoring_errors": 0,
            "scoring_errors_total": int(self._scoring_errors_total),
            "real_candidate_state_evidence": False,
            "goal_energy_score_variance": 0.0,
            "candidate_pool_differs_from_baseline": False,
            "arms_non_degenerate": False,
            "cpu_scoring_ms_per_candidate": 0.0,
            "score_min": None,
            "score_max": None,
        }


@dataclass(frozen=True)
class GoalSatisfactionEnergy:
    """Fraction-unsatisfied goal energy with a terminal Exp4020 predicate gate."""

    predicate: Callable[[dict[str, Any]], bool]
    predicate_code: str
    source: str = GOAL_ENERGY_SOURCE

    @classmethod
    def from_predicate_code(cls, code: str) -> "GoalSatisfactionEnergy":
        return cls(predicate=compile_goal_predicate(str(code)), predicate_code=str(code))

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> "GoalSatisfactionEnergy":
        code = str(artifact.get("goal_predicate_code") or "")
        if not code:
            raise ValueError("exp4020 artifact missing goal_predicate_code")
        return cls.from_predicate_code(code)

    @classmethod
    def from_artifact_path(cls, path: Path | str) -> "GoalSatisfactionEnergy":
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_artifact(artifact)

    def visible_state(self, value: Any) -> Mapping[str, Any] | None:
        return _state_from_visible(value)

    def predicate_fires(self, value: Any) -> bool:
        state = self.visible_state(value)
        if state is None:
            return False
        try:
            return bool(self.predicate(dict(state)))
        except Exception:
            return False

    def __call__(self, value: Any) -> float:
        state = self.visible_state(value)
        if state is None:
            return 1.0
        if self.predicate_fires(state):
            return 0.0
        total = _as_float(state.get("total_targets"))
        satisfied = _as_float(state.get("satisfied_targets"))
        unsatisfied = _as_float(state.get("unsatisfied_targets"))
        if total <= 0.0 and satisfied + unsatisfied > 0.0:
            total = satisfied + unsatisfied
        if total <= 0.0:
            return 1.0
        if satisfied > 0.0:
            return max(0.0, min(1.0, 1.0 - satisfied / total))
        return max(0.0, min(1.0, unsatisfied / total))


@dataclass
class RelationalGoalEnergy:
    """REQ-ARC-WMTE-5711: route relational placement/spatial receipts into live energy.

    The class deliberately reads only the state it is handed: a 2-D visible grid
    plus an optional agent-owned `relational_goal_receipt`. When that receipt is
    absent or corrupt, it returns a constant no-bias score unless a legacy
    `GoalSatisfactionEnergy` fallback can score the older target-fraction state.
    """

    fallback_goal_energy: Any | None = None
    variance_floor: float = RELATIONAL_GOAL_VARIANCE_FLOOR
    source: str = RELATIONAL_GOAL_ENERGY_SOURCE
    _call_count: int = field(default=0, init=False, repr=False)
    _routed_call_count: int = field(default=0, init=False, repr=False)
    _fallback_count: int = field(default=0, init=False, repr=False)
    _fallback_reasons: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _route_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _last_diagnostics: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def _record_fallback(self, reason: str, value: Any) -> float:
        self._fallback_count += 1
        self._fallback_reasons[reason] = self._fallback_reasons.get(reason, 0) + 1
        used_legacy = False
        score = 0.0
        if self.fallback_goal_energy is not None:
            try:
                score = float(self.fallback_goal_energy(value))
                used_legacy = True
            except Exception:
                score = 0.0
        self._last_diagnostics = {
            "enabled": True,
            "source": self.source,
            "variance_floor": float(self.variance_floor),
            "call_count": int(self._call_count),
            "routed_call_count": int(self._routed_call_count),
            "fallback_count": int(self._fallback_count),
            "fallback_reasons": dict(self._fallback_reasons),
            "route_counts": dict(self._route_counts),
            "last_routed": False,
            "last_route_class": None,
            "last_fallback_reason": (
                "legacy_goal_satisfaction" if used_legacy and _state_from_visible(value) is not None else reason
            ),
            "last_score": float(score),
        }
        return float(score)

    def _score_region_pair(self, arr: Any, receipt: Mapping[str, Any]) -> tuple[bool, float, str]:
        src = _bool_mask(receipt.get("source_mask"), tuple(arr.shape))
        tgt = _bool_mask(receipt.get("target_mask"), tuple(arr.shape))
        if src is None or tgt is None:
            return False, 0.0, "corrupt_receipt"
        if int(src.sum()) <= 0 or int(tgt.sum()) <= 0 or int(src.sum()) != int(tgt.sum()):
            return False, 0.0, "missing_target"
        left = arr[src]
        right = arr[tgt]
        return True, float((left != right).sum()), ""

    def _score_translated(self, arr: Any, receipt: Mapping[str, Any]) -> tuple[bool, float, str]:
        try:
            dy, dx = receipt.get("offset", (None, None))
            dy, dx = int(dy), int(dx)
        except Exception:
            return False, 0.0, "corrupt_receipt"
        src = _bool_mask(receipt.get("source_mask", receipt.get("mask")), tuple(arr.shape))
        if src is None or int(src.sum()) <= 0:
            return False, 0.0, "corrupt_receipt"
        mismatches = 0
        compared = 0
        h, w = arr.shape
        for y, x in _mask_coords(src):
            ty, tx = y + dy, x + dx
            if ty < 0 or ty >= h or tx < 0 or tx >= w:
                continue
            compared += 1
            if arr[y, x] != arr[ty, tx]:
                mismatches += 1
        if compared == 0:
            return False, 0.0, "missing_target"
        return True, float(mismatches), ""

    def _score_ordered_run(self, arr: Any, receipt: Mapping[str, Any]) -> tuple[bool, float, str]:
        mask = _bool_mask(receipt.get("run_mask"), tuple(arr.shape))
        if mask is None or int(mask.sum()) < 2:
            return False, 0.0, "corrupt_receipt"
        coords = _mask_coords(mask)
        values = [arr[y, x] for y, x in coords]
        order = str(receipt.get("order", "ascending"))
        violations = 0
        for left, right in zip(values, values[1:]):
            if order == "descending":
                violations += int(left < right)
            else:
                violations += int(left > right)
        return True, float(violations), ""

    def _score_centroid(self, arr: Any, receipt: Mapping[str, Any]) -> tuple[bool, float, str]:
        src = _bool_mask(receipt.get("source_mask"), tuple(arr.shape))
        tgt = _bool_mask(receipt.get("target_mask"), tuple(arr.shape))
        if src is None or tgt is None:
            return False, 0.0, "corrupt_receipt"
        src_c = _local_centroid(arr, src)
        tgt_c = _local_centroid(arr, tgt)
        if src_c is None or tgt_c is None:
            return False, 0.0, "missing_target"
        dy = float(src_c[0] - tgt_c[0])
        dx = float(src_c[1] - tgt_c[1])
        return True, float((dy * dy + dx * dx) ** 0.5), ""

    def _score_relational(self, value: Any) -> tuple[bool, float, str, str | None]:
        arr = _grid_from_value(value)
        receipt = _relational_receipt_from_value(value)
        if receipt is None:
            return False, 0.0, "missing_relational_receipt", None
        if arr is None:
            return False, 0.0, "missing_visible_grid", None
        route_class = str(receipt.get("route_class") or receipt.get("mechanic_class") or "")
        if route_class not in SUPPORTED_RELATIONAL_ROUTE_CLASSES:
            return False, 0.0, "unsupported_route_class", route_class or None
        if route_class == "region_pair_equality":
            ok, score, reason = self._score_region_pair(arr, receipt)
        elif route_class == "translated_within_frame_target_match":
            ok, score, reason = self._score_translated(arr, receipt)
        elif route_class == "ordered_run_relation":
            ok, score, reason = self._score_ordered_run(arr, receipt)
        else:
            ok, score, reason = self._score_centroid(arr, receipt)
        return ok, float(score), reason, route_class

    def __call__(self, value: Any) -> float:
        self._call_count += 1
        ok, score, reason, route_class = self._score_relational(value)
        if not ok:
            return self._record_fallback(reason, value)
        self._routed_call_count += 1
        assert route_class is not None
        self._route_counts[route_class] = self._route_counts.get(route_class, 0) + 1
        self._last_diagnostics = {
            "enabled": True,
            "source": self.source,
            "variance_floor": float(self.variance_floor),
            "call_count": int(self._call_count),
            "routed_call_count": int(self._routed_call_count),
            "fallback_count": int(self._fallback_count),
            "fallback_reasons": dict(self._fallback_reasons),
            "route_counts": dict(self._route_counts),
            "last_routed": True,
            "last_route_class": route_class,
            "last_fallback_reason": None,
            "last_score": float(score),
        }
        return float(score)

    def predicate_fires(self, value: Any) -> bool:
        ok, score, _reason, _route_class = self._score_relational(value)
        if ok:
            return float(score) == 0.0
        predicate = getattr(self.fallback_goal_energy, "predicate_fires", None)
        return bool(callable(predicate) and predicate(value))

    def diagnostics(self) -> dict[str, Any]:
        if self._last_diagnostics:
            return dict(self._last_diagnostics)
        return {
            "enabled": True,
            "source": self.source,
            "variance_floor": float(self.variance_floor),
            "call_count": int(self._call_count),
            "routed_call_count": int(self._routed_call_count),
            "fallback_count": int(self._fallback_count),
            "fallback_reasons": dict(self._fallback_reasons),
            "route_counts": dict(self._route_counts),
            "last_routed": False,
            "last_route_class": None,
            "last_fallback_reason": None,
            "last_score": None,
        }


@dataclass(frozen=True)
class UniformGoalEnergy:
    """Deterministic uniform/random energy used only as the ablation control."""

    seed: int = 4640

    def __call__(self, value: Any) -> float:
        state = _state_from_visible(value)
        payload = state if state is not None else repr(value)
        encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        digest = hashlib.sha256(f"{int(self.seed)}:{encoded}".encode()).hexdigest()
        return int(digest[:12], 16) / float(16**12 - 1)

    def predicate_fires(self, _value: Any) -> bool:
        return False


def make_uniform_goal_energy(seed: int = 4640) -> UniformGoalEnergy:
    return UniformGoalEnergy(seed=int(seed))


@dataclass(frozen=True)
class GoalEnergyHeuristic:
    """Convex combination of navigation energy and graded goal-satisfaction energy."""

    navigation_energy: Callable[[Any], float] | None
    goal_energy: Callable[[Any], float]
    alpha: float = 0.9
    beta: float = 0.1

    def __post_init__(self) -> None:
        total = float(self.alpha) + float(self.beta)
        if abs(total - 1.0) > 1e-9:
            raise ValueError("goal energy heuristic requires alpha + beta == 1")

    def navigation_component(self, value: Any) -> float:
        if self.navigation_energy is None:
            return 0.0
        return float(self.navigation_energy(value))

    def goal_component(self, value: Any) -> float:
        return float(self.goal_energy(value))

    def __call__(self, value: Any) -> float:
        return float(self.alpha) * self.navigation_component(value) + float(
            self.beta
        ) * self.goal_component(value)

    def predicate_fires(self, value: Any) -> bool:
        predicate = getattr(self.goal_energy, "predicate_fires", None)
        return bool(callable(predicate) and predicate(value))

    def components(self, value: Any) -> dict[str, float]:
        return {
            "navigation": self.navigation_component(value),
            "goal_energy": self.goal_component(value),
            "combined": self(value),
        }


def make_goal_energy_heuristic(
    *,
    navigation_energy: Callable[[Any], float] | None,
    goal_energy: Callable[[Any], float],
    alpha: float = 0.9,
    beta: float = 0.1,
) -> GoalEnergyHeuristic:
    return GoalEnergyHeuristic(
        navigation_energy=navigation_energy,
        goal_energy=goal_energy,
        alpha=float(alpha),
        beta=float(beta),
    )


def load_exp4020_goal_energy(root: Path | str | None = None) -> GoalSatisfactionEnergy | None:
    base = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    path = base / DEFAULT_EXP4020_ARTIFACT
    try:
        return GoalSatisfactionEnergy.from_artifact_path(path)
    except Exception:
        return None


def load_relational_goal_energy(
    root: Path | str | None = None,
) -> RelationalGoalEnergy | None:
    """REQ-ARC-WMTE-5711: submitted live loader for relational-plus-legacy goal energy."""

    return RelationalGoalEnergy(fallback_goal_energy=load_exp4020_goal_energy(root))

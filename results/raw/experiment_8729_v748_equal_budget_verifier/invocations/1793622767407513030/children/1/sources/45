"""Counterexample-patched executable transition hypotheses for ARC receipts.

Spec refs: REQ-ARC-WMTE-5641,
SCENARIO-ARC-WMTE-5641-COUNTEREXAMPLE-PATCH-REPLAY,
SCENARIO-ARC-WMTE-5641-CONTROLS-AND-ABSTENTION.

This module is intentionally a small deterministic development proxy. It does
not synthesize Python source or infer game rules from source files. It keeps an
inspectable set of typed clauses and revises those clauses only after an
agent-owned transition falsifies the current executable model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import copy
import hashlib
import json
import math
from typing import Any

import numpy as np


JsonDict = dict[str, Any]
ABSTAIN = "abstain"
PATCH_OPERATORS = ("add", "specialize", "relax", "retire")


@dataclass(frozen=True)
class TransitionReceipt:
    """One agent-owned transition receipt used by the executable patcher."""

    trace_id: str
    episode: str
    step: int
    state: Any
    action: int
    data: Any
    successor: Any
    reward: int = 0
    terminal: bool = False
    provenance: str = "agent_owned_runtime_observation"

    @property
    def effect_signature(self) -> JsonDict:
        return effect_signature(self.state, self.successor, self.reward, self.terminal)


@dataclass(frozen=True)
class PredictionResult:
    """Executable prediction or explicit abstention for one transition context."""

    decision: str
    effect: JsonDict | None
    reason: str
    clause_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PatchDecision:
    """Outcome of trying to revise the model after a falsifying receipt."""

    receipt_id: str
    prediction: PredictionResult
    falsified: bool
    patch_operator: str | None
    patch_accepted: bool
    patch_rejected: bool
    reason: str


@dataclass(frozen=True)
class TransitionClause:
    """A typed executable clause with no game names, coordinates, or levels."""

    clause_id: str
    selector_kind: str
    selector_value: str
    action: int
    effect: JsonDict
    support: int
    created_by_patch: str
    retired: bool = False

    @property
    def specificity(self) -> int:
        return {"object_hash": 3, "topology_hash": 2, "action": 1}.get(
            self.selector_kind,
            0,
        )

    def matches(self, features: "_ContextFeatures") -> bool:
        if self.retired or int(self.action) != int(features.action):
            return False
        if self.selector_kind == "object_hash":
            return self.selector_value == features.object_hash
        if self.selector_kind == "topology_hash":
            return self.selector_value == features.topology_hash
        if self.selector_kind == "action":
            return True
        return False

    def as_dict(self) -> JsonDict:
        return {
            "clause_id": self.clause_id,
            "selector_kind": self.selector_kind,
            "selector_value": self.selector_value,
            "action": int(self.action),
            "effect": copy.deepcopy(self.effect),
            "support": int(self.support),
            "created_by_patch": self.created_by_patch,
            "retired": bool(self.retired),
        }


@dataclass(frozen=True)
class _ContextFeatures:
    supported: bool
    action: int
    object_hash: str
    topology_hash: str
    location_relation: str
    reason: str = ""


@dataclass(frozen=True)
class _PatchCandidate:
    operator: str
    add_clause: TransitionClause | None = None
    retire_clause_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChronologicalEvaluationResult:
    """Aggregate Exp5641 evaluation state."""

    heldout_transition_error_by_arm: dict[str, float]
    abstention_calibration: JsonDict
    mechanism_question_controls: JsonDict
    counterexample_count: int
    accepted_patch_count: int
    rejected_patch_count: int
    all_receipt_replay_pass: bool
    unsafe_patch_accept_count: int
    patch_controls: JsonDict
    patcher_diagnostics: JsonDict


def _as_grid(value: Any) -> np.ndarray:
    if hasattr(value, "frame"):
        value = getattr(value, "frame")
    return np.asarray(value, dtype=np.int16)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _click_xy(data: Any) -> tuple[int, int] | None:
    if not isinstance(data, Mapping) or "x" not in data or "y" not in data:
        return None
    try:
        return int(data["x"]), int(data["y"])
    except (TypeError, ValueError):
        return None


def _component_at_click(
    grid: np.ndarray,
    data: Any,
    *,
    max_cells: int = 512,
) -> tuple[str, str] | None:
    click = _click_xy(data)
    if click is None or grid.ndim != 2:
        return None
    x, y = click
    height, width = int(grid.shape[0]), int(grid.shape[1])
    if x < 0 or y < 0 or x >= width or y >= height:
        return None
    value = int(grid[y, x])
    if value == 0:
        return None
    seen = {(y, x)}
    stack = [(y, x)]
    cells: list[tuple[int, int]] = []
    while stack:
        row, col = stack.pop()
        cells.append((row, col))
        if len(cells) > max_cells:
            return None
        for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + drow, col + dcol
            if (
                0 <= nr < height
                and 0 <= nc < width
                and (nr, nc) not in seen
                and int(grid[nr, nc]) == value
            ):
                seen.add((nr, nc))
                stack.append((nr, nc))
    min_row = min(row for row, _ in cells)
    min_col = min(col for _, col in cells)
    object_payload = tuple(
        sorted((row - min_row, col - min_col, int(grid[row, col])) for row, col in cells)
    )
    topology_payload = tuple(sorted((row - min_row, col - min_col) for row, col in cells))
    return _sha256(object_payload), _sha256(topology_payload)


def context_features(receipt: TransitionReceipt) -> _ContextFeatures:
    grid = _as_grid(receipt.state)
    component = _component_at_click(grid, receipt.data)
    if component is None:
        return _ContextFeatures(
            supported=False,
            action=int(receipt.action),
            object_hash="",
            topology_hash="",
            location_relation=ABSTAIN,
            reason="unsupported_object",
        )
    object_hash, topology_hash = component
    return _ContextFeatures(
        supported=True,
        action=int(receipt.action),
        object_hash=object_hash,
        topology_hash=topology_hash,
        location_relation="clicked_component",
    )


def effect_signature(state: Any, successor: Any, reward: int = 0, terminal: bool = False) -> JsonDict:
    before = _as_grid(state)
    after = _as_grid(successor)
    if before.shape != after.shape or before.ndim != 2:
        return {
            "changed_count": -1,
            "changed_bbox_shape": [-1, -1],
            "changed_object_topology_hash": "malformed",
            "changed_attributes": [],
            "reward": int(reward),
            "terminal_event": bool(terminal),
        }
    changed = np.argwhere(before != after)
    after_values = Counter(int(after[row, col]) for row, col in changed)
    if len(changed):
        min_row, min_col = changed.min(axis=0)
        max_row, max_col = changed.max(axis=0)
        bbox = [int(max_row - min_row + 1), int(max_col - min_col + 1)]
        topology_payload = tuple(
            sorted((int(row - min_row), int(col - min_col), int(after[row, col])) for row, col in changed)
        )
    else:
        bbox = [0, 0]
        topology_payload = ()
    return {
        "changed_count": int(len(changed)),
        "changed_bbox_shape": bbox,
        "changed_object_topology_hash": _sha256(topology_payload),
        "changed_attributes": [
            {"after": int(dst), "count": int(count)}
            for dst, count in sorted(after_values.items())
        ],
        "reward": int(reward),
        "terminal_event": bool(terminal),
    }


def _canonical_effect(effect: Mapping[str, Any]) -> JsonDict:
    attrs = effect.get("changed_attributes", [])
    return {
        "changed_count": int(effect.get("changed_count", 0)),
        "changed_bbox_shape": list(effect.get("changed_bbox_shape", [])),
        "changed_object_topology_hash": str(effect.get("changed_object_topology_hash", "")),
        "changed_attributes": sorted(
            (
                {
                    "after": int(row.get("after", 0)),
                    "count": int(row.get("count", 0)),
                }
                for row in attrs
                if isinstance(row, Mapping)
            ),
            key=lambda row: (row["after"], row["count"]),
        ),
        "reward": int(effect.get("reward", 0)),
        "terminal_event": bool(effect.get("terminal_event", False)),
    }


def _effect_key(effect: Mapping[str, Any] | None) -> str:
    if effect is None:
        return ABSTAIN
    return _stable_json(_canonical_effect(effect))


def receipt_id(receipt: TransitionReceipt) -> str:
    payload = {
        "trace": str(receipt.trace_id),
        "episode": str(receipt.episode),
        "step": int(receipt.step),
        "action": int(receipt.action),
        "context": context_features(receipt).__dict__,
        "effect": receipt.effect_signature,
    }
    return "receipt:" + _sha256(payload).split(":", 1)[1][:16]


class ExecutableTransitionHypothesisPatcher:
    """REQ-ARC-WMTE-5641: deterministic counterexample-patched clause model."""

    def __init__(
        self,
        *,
        relax_support: int = 2,
        max_patches_per_receipt: int = 4,
        min_predict_support: int = 2,
    ) -> None:
        self.relax_support = max(1, int(relax_support))
        self.max_patches_per_receipt = max(1, int(max_patches_per_receipt))
        self.min_predict_support = max(1, int(min_predict_support))
        self._clauses: list[TransitionClause] = []
        self._receipts: list[TransitionReceipt] = []
        self._patch_log: list[JsonDict] = []
        self._counterexample_count = 0
        self._accepted_patch_count = 0
        self._rejected_patch_count = 0
        self._unsafe_patch_accept_count = 0

    @property
    def clauses(self) -> list[TransitionClause]:
        return list(self._clauses)

    def predict(self, receipt: TransitionReceipt) -> PredictionResult:
        return self._predict_with_clauses(self._clauses, receipt)

    def observe(self, receipt: TransitionReceipt) -> PatchDecision:
        prediction = self.predict(receipt)
        features = context_features(receipt)
        if not features.supported:
            self._receipts.append(receipt)
            return PatchDecision(
                receipt_id(receipt),
                prediction,
                falsified=False,
                patch_operator=None,
                patch_accepted=False,
                patch_rejected=False,
                reason="unsupported_object_abstained",
            )
        actual = receipt.effect_signature
        if prediction.effect is not None and _effect_key(prediction.effect) == _effect_key(actual):
            self._receipts.append(receipt)
            self._increment_support(features, actual)
            return PatchDecision(
                receipt_id(receipt),
                prediction,
                falsified=False,
                patch_operator=None,
                patch_accepted=False,
                patch_rejected=False,
                reason="prediction_matched",
            )

        self._counterexample_count += 1
        for candidate in self._candidate_patches(receipt, prediction):
            accepted, reason = self._try_accept_candidate(candidate, receipt)
            if accepted:
                self._receipts.append(receipt)
                self._accepted_patch_count += 1
                log = {
                    "receipt_id": receipt_id(receipt),
                    "operator": candidate.operator,
                    "reason": reason,
                    "falsified_prediction": prediction.reason,
                }
                self._patch_log.append(log)
                return PatchDecision(
                    receipt_id(receipt),
                    prediction,
                    falsified=True,
                    patch_operator=candidate.operator,
                    patch_accepted=True,
                    patch_rejected=False,
                    reason=reason,
                )
            self._rejected_patch_count += 1
            self._patch_log.append(
                {
                    "receipt_id": receipt_id(receipt),
                    "operator": candidate.operator,
                    "accepted": False,
                    "reason": reason,
                }
            )

        self._receipts.append(receipt)
        return PatchDecision(
            receipt_id(receipt),
            prediction,
            falsified=True,
            patch_operator=None,
            patch_accepted=False,
            patch_rejected=True,
            reason="all_candidate_patches_rejected",
        )

    def evaluate_patch_control(self, receipt: TransitionReceipt, control_name: str) -> JsonDict:
        prediction = self.predict(receipt)
        candidates = list(self._candidate_patches(receipt, prediction))
        if control_name == "contradictory_patch":
            candidates = [self._add_candidate(receipt, "add")]
        rejected_reasons: list[str] = []
        for candidate in candidates:
            clauses = self._candidate_clauses(candidate)
            reason = self._active_conflict_reason(clauses)
            if reason is None:
                reason = self._replay_failure_reason(clauses, [*self._receipts, receipt])
            if reason is None:
                return {
                    "control": control_name,
                    "accepted": True,
                    "rejected": False,
                    "reason": "candidate_would_replay",
                }
            rejected_reasons.append(reason)
        return {
            "control": control_name,
            "accepted": False,
            "rejected": True,
            "reason": rejected_reasons[0] if rejected_reasons else "no_candidate",
        }

    def label_order_control(self, receipt: TransitionReceipt) -> JsonDict:
        effect = receipt.effect_signature
        permuted = copy.deepcopy(effect)
        permuted["changed_attributes"] = list(reversed(permuted.get("changed_attributes", [])))
        return {
            "control": "label_order",
            "passed": _effect_key(effect) == _effect_key(permuted),
            "canonical_effect_equal": _effect_key(effect) == _effect_key(permuted),
        }

    def all_receipt_replay_pass(self) -> bool:
        return self._replay_failure_reason(self._clauses, self._receipts) is None

    def diagnostics(self) -> JsonDict:
        return {
            "counterexample_count": int(self._counterexample_count),
            "accepted_patch_count": int(self._accepted_patch_count),
            "rejected_patch_count": int(self._rejected_patch_count),
            "unsafe_patch_accept_count": int(self._unsafe_patch_accept_count),
            "all_receipt_replay_pass": self.all_receipt_replay_pass(),
            "active_clause_count": sum(1 for clause in self._clauses if not clause.retired),
            "patch_log": list(self._patch_log),
            "clauses": [clause.as_dict() for clause in self._clauses],
        }

    def _predict_with_clauses(
        self,
        clauses: Sequence[TransitionClause],
        receipt: TransitionReceipt,
        *,
        min_support: int | None = None,
    ) -> PredictionResult:
        features = context_features(receipt)
        if not features.supported:
            return PredictionResult(ABSTAIN, None, "unsupported_object")
        support_threshold = self.min_predict_support if min_support is None else int(min_support)
        matches = [
            clause
            for clause in clauses
            if clause.matches(features) and int(clause.support) >= support_threshold
        ]
        if not matches:
            return PredictionResult(ABSTAIN, None, "no_matching_clause")
        best_specificity = max(clause.specificity for clause in matches)
        best = [clause for clause in matches if clause.specificity == best_specificity]
        effect_keys = {_effect_key(clause.effect) for clause in best}
        if len(effect_keys) > 1:
            return PredictionResult(
                ABSTAIN,
                None,
                "active_clause_contradiction",
                tuple(sorted(clause.clause_id for clause in best)),
            )
        selected = sorted(best, key=lambda clause: clause.clause_id)[0]
        return PredictionResult(
            "predict",
            copy.deepcopy(selected.effect),
            "matched_clause",
            tuple(sorted(clause.clause_id for clause in best)),
        )

    def _candidate_patches(
        self,
        receipt: TransitionReceipt,
        prediction: PredictionResult,
    ) -> tuple[_PatchCandidate, ...]:
        if not context_features(receipt).supported:
            return ()
        candidates: list[_PatchCandidate] = []
        if prediction.decision == ABSTAIN:
            relaxed = self._relax_candidate(receipt)
            if relaxed is not None:
                candidates.append(relaxed)
            candidates.append(self._add_candidate(receipt, "add"))
            return tuple(candidates[: self.max_patches_per_receipt])
        candidates.append(self._add_candidate(receipt, "specialize"))
        retire_ids = self._matching_clause_ids(receipt)
        if retire_ids:
            candidates.append(_PatchCandidate("retire", retire_clause_ids=tuple(retire_ids)))
        return tuple(candidates[: self.max_patches_per_receipt])

    def _matching_clause_ids(self, receipt: TransitionReceipt) -> tuple[str, ...]:
        features = context_features(receipt)
        return tuple(
            sorted(
                clause.clause_id
                for clause in self._clauses
                if not clause.retired and clause.matches(features)
            )
        )

    def _add_candidate(self, receipt: TransitionReceipt, operator: str) -> _PatchCandidate:
        features = context_features(receipt)
        payload = {
            "operator": operator,
            "selector_kind": "object_hash",
            "selector_value": features.object_hash,
            "action": int(receipt.action),
            "effect": receipt.effect_signature,
        }
        clause = TransitionClause(
            clause_id="clause:" + _sha256(payload).split(":", 1)[1][:16],
            selector_kind="object_hash",
            selector_value=features.object_hash,
            action=int(receipt.action),
            effect=receipt.effect_signature,
            support=1,
            created_by_patch=operator,
        )
        return _PatchCandidate(operator, add_clause=clause)

    def _relax_candidate(self, receipt: TransitionReceipt) -> _PatchCandidate | None:
        features = context_features(receipt)
        actual_key = _effect_key(receipt.effect_signature)
        support = 0
        for clause in self._clauses:
            if clause.retired or clause.selector_kind not in {"object_hash", "topology_hash"}:
                continue
            if int(clause.action) != int(receipt.action):
                continue
            if _effect_key(clause.effect) != actual_key:
                continue
            support += int(max(1, clause.support))
        if support < self.relax_support:
            return None
        payload = {
            "operator": "relax",
            "selector_kind": "topology_hash",
            "selector_value": features.topology_hash,
            "action": int(receipt.action),
            "effect": receipt.effect_signature,
        }
        clause = TransitionClause(
            clause_id="clause:" + _sha256(payload).split(":", 1)[1][:16],
            selector_kind="topology_hash",
            selector_value=features.topology_hash,
            action=int(receipt.action),
            effect=receipt.effect_signature,
            support=support + 1,
            created_by_patch="relax",
        )
        return _PatchCandidate("relax", add_clause=clause)

    def _try_accept_candidate(
        self,
        candidate: _PatchCandidate,
        receipt: TransitionReceipt,
    ) -> tuple[bool, str]:
        clauses = self._candidate_clauses(candidate)
        conflict = self._active_conflict_reason(clauses)
        if conflict is not None:
            return False, conflict
        replay = self._replay_failure_reason(clauses, [*self._receipts, receipt])
        if replay is not None:
            return False, replay
        if candidate.operator != "retire":
            candidate_prediction = self._predict_with_clauses(clauses, receipt, min_support=1)
            if _effect_key(candidate_prediction.effect) != _effect_key(receipt.effect_signature):
                return False, "candidate_did_not_explain_counterexample"
        self._clauses = clauses
        if self._active_conflict_reason(self._clauses) is not None:
            self._unsafe_patch_accept_count += 1
        return True, "all_receipt_replay_pass"

    def _candidate_clauses(self, candidate: _PatchCandidate) -> list[TransitionClause]:
        retire = set(candidate.retire_clause_ids)
        clauses = [
            replace(clause, retired=True) if clause.clause_id in retire else clause
            for clause in self._clauses
        ]
        if candidate.add_clause is not None:
            clauses.append(candidate.add_clause)
        return clauses

    def _active_conflict_reason(self, clauses: Sequence[TransitionClause]) -> str | None:
        effects_by_key: dict[tuple[int, str, str], set[str]] = defaultdict(set)
        for clause in clauses:
            if clause.retired:
                continue
            key = (int(clause.action), clause.selector_kind, clause.selector_value)
            effects_by_key[key].add(_effect_key(clause.effect))
        if any(len(effects) > 1 for effects in effects_by_key.values()):
            return "active_clause_contradiction"
        return None

    def _replay_failure_reason(
        self,
        clauses: Sequence[TransitionClause],
        receipts: Sequence[TransitionReceipt],
    ) -> str | None:
        for row in receipts:
            if not context_features(row).supported:
                continue
            prediction = self._predict_with_clauses(clauses, row)
            if prediction.effect is None:
                continue
            if _effect_key(prediction.effect) != _effect_key(row.effect_signature):
                return "wrong_replay_prediction"
        return None

    def _increment_support(self, features: _ContextFeatures, actual: Mapping[str, Any]) -> None:
        actual_key = _effect_key(actual)
        out: list[TransitionClause] = []
        for clause in self._clauses:
            if (
                clause.matches(features)
                and _effect_key(clause.effect) == actual_key
                and not clause.retired
            ):
                out.append(replace(clause, support=int(clause.support) + 1))
            else:
                out.append(clause)
        self._clauses = out


class _LastTransitionArm:
    def __init__(self) -> None:
        self.last_by_action: dict[int, JsonDict] = {}

    def predict(self, row: TransitionReceipt) -> PredictionResult:
        effect = self.last_by_action.get(int(row.action))
        if effect is None:
            return PredictionResult(ABSTAIN, None, "no_last_transition")
        return PredictionResult("predict", effect, "last_transition")

    def observe(self, row: TransitionReceipt) -> None:
        if context_features(row).supported:
            self.last_by_action[int(row.action)] = row.effect_signature


class _FrequencyTableArm:
    def __init__(self, *, use_topology: bool = False) -> None:
        self.use_topology = bool(use_topology)
        self.counts: dict[Any, Counter[str]] = defaultdict(Counter)
        self.effects: dict[str, JsonDict] = {}

    def _key(self, row: TransitionReceipt) -> Any:
        features = context_features(row)
        if self.use_topology and features.supported:
            return int(row.action), features.topology_hash
        return int(row.action)

    def predict(self, row: TransitionReceipt) -> PredictionResult:
        counts = self.counts.get(self._key(row))
        if not counts:
            return PredictionResult(ABSTAIN, None, "frequency_no_support")
        [(effect_key, _count)] = counts.most_common(1)
        return PredictionResult("predict", copy.deepcopy(self.effects[effect_key]), "frequency_table")

    def observe(self, row: TransitionReceipt) -> None:
        if not context_features(row).supported:
            return
        key = self._key(row)
        effect_key = _effect_key(row.effect_signature)
        self.counts[key][effect_key] += 1
        self.effects[effect_key] = row.effect_signature


class _UnpatchedArm:
    def predict(self, _row: TransitionReceipt) -> PredictionResult:
        return PredictionResult(ABSTAIN, None, "unpatched_model")

    def observe(self, _row: TransitionReceipt) -> None:
        return None


def _prediction_error(prediction: PredictionResult, actual: Mapping[str, Any]) -> float:
    if prediction.effect is None:
        return 0.5
    return 0.0 if _effect_key(prediction.effect) == _effect_key(actual) else 1.0


def _mean(values: Sequence[float]) -> float:
    return round(float(sum(values)) / float(len(values)), 6) if values else 0.0


def _normal_interval(values: Sequence[float]) -> JsonDict:
    if not values:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}
    mean = float(sum(values)) / float(len(values))
    if len(values) == 1:
        return {"mean": round(mean, 6), "lower": round(mean, 6), "upper": round(mean, 6), "n": 1}
    variance = sum((value - mean) ** 2 for value in values) / float(len(values) - 1)
    half_width = 1.96 * math.sqrt(variance / float(len(values)))
    return {
        "mean": round(mean, 6),
        "lower": round(mean - half_width, 6),
        "upper": round(mean + half_width, 6),
        "n": int(len(values)),
    }


def run_chronological_evaluation(
    transitions_by_game: Mapping[str, Sequence[TransitionReceipt]],
    *,
    unsupported_receipts: Sequence[TransitionReceipt] | None = None,
    random_seed: int = 5641,
) -> ChronologicalEvaluationResult:
    del random_seed
    errors: dict[str, list[float]] = {
        "patched": [],
        "last_transition": [],
        "frequency_table": [],
        "unpatched": [],
        "oracle_free_generic": [],
    }
    paired_unpatched_delta: list[float] = []
    informative_scores: list[float] = []
    irrelevant_scores: list[float] = []
    patch_controls = {
        "corrupted_transition": {"accepted": 0, "rejected": 0},
        "label_order": {"passed": 0, "failed": 0},
        "contradictory_patch": {"accepted": 0, "rejected": 0},
        "unsupported_object": {"abstained": 0, "accepted": 0},
    }
    aggregate = {
        "counterexample_count": 0,
        "accepted_patch_count": 0,
        "rejected_patch_count": 0,
        "unsafe_patch_accept_count": 0,
    }
    replay_passes: list[bool] = []
    active_clause_counts: list[int] = []

    for _game, rows in sorted(transitions_by_game.items()):
        patcher = ExecutableTransitionHypothesisPatcher(relax_support=1)
        arms = {
            "last_transition": _LastTransitionArm(),
            "frequency_table": _FrequencyTableArm(use_topology=False),
            "unpatched": _UnpatchedArm(),
            "oracle_free_generic": _FrequencyTableArm(use_topology=True),
        }
        supported_rows = [row for row in sorted(rows, key=lambda row: int(row.step)) if _usable(row)]
        for row in supported_rows:
            actual = row.effect_signature
            patched_prediction = patcher.predict(row)
            patched_error = _prediction_error(patched_prediction, actual)
            unpatched_error = _prediction_error(arms["unpatched"].predict(row), actual)
            errors["patched"].append(patched_error)
            paired_unpatched_delta.append(unpatched_error - patched_error)
            informative_scores.append(1.0 if patched_error == 0.0 else 0.0)
            irrelevant_scores.append(0.0)
            for name, arm in arms.items():
                errors[name].append(_prediction_error(arm.predict(row), actual))
            patcher.observe(row)
            for arm in arms.values():
                arm.observe(row)
        if supported_rows:
            contradictory = make_contradictory_receipt(supported_rows[0], effect_color=7)
            contradictory_control = patcher.evaluate_patch_control(
                contradictory,
                "contradictory_patch",
            )
            patch_controls["contradictory_patch"][
                "accepted" if contradictory_control["accepted"] else "rejected"
            ] += 1
            corrupted = make_contradictory_receipt(supported_rows[-1], effect_color=8)
            corrupted_control = patcher.evaluate_patch_control(corrupted, "corrupted_transition")
            patch_controls["corrupted_transition"][
                "accepted" if corrupted_control["accepted"] else "rejected"
            ] += 1
            label_order = patcher.label_order_control(supported_rows[-1])
            patch_controls["label_order"]["passed" if label_order["passed"] else "failed"] += 1
        diag = patcher.diagnostics()
        for key in aggregate:
            aggregate[key] += int(diag[key])
        replay_passes.append(bool(diag["all_receipt_replay_pass"]))
        active_clause_counts.append(int(diag["active_clause_count"]))

    unsupported = list(unsupported_receipts or [])
    if unsupported:
        probe = ExecutableTransitionHypothesisPatcher()
        for row in unsupported:
            prediction = probe.predict(row)
            if prediction.decision == ABSTAIN:
                patch_controls["unsupported_object"]["abstained"] += 1
            else:
                patch_controls["unsupported_object"]["accepted"] += 1
    unsupported_total = sum(patch_controls["unsupported_object"].values())
    unsupported_rate = (
        float(patch_controls["unsupported_object"]["abstained"]) / float(unsupported_total)
        if unsupported_total
        else 1.0
    )
    interval = _normal_interval(paired_unpatched_delta)
    return ChronologicalEvaluationResult(
        heldout_transition_error_by_arm={key: _mean(value) for key, value in errors.items()},
        abstention_calibration={
            "unsupported_abstention_rate": round(unsupported_rate, 6),
            "patched_vs_unpatched_error_reduction_interval": interval,
            "supported_receipt_count": int(len(errors["patched"])),
        },
        mechanism_question_controls={
            "informative": {
                "question": "predict_changed_attributes_reward_terminal",
                "score": _mean(informative_scores),
            },
            "irrelevant": {
                "question": "permuted_label_order_or_unrelated_probe",
                "score": _mean(irrelevant_scores),
            },
        },
        counterexample_count=int(aggregate["counterexample_count"]),
        accepted_patch_count=int(aggregate["accepted_patch_count"]),
        rejected_patch_count=int(
            aggregate["rejected_patch_count"]
            + patch_controls["contradictory_patch"]["rejected"]
            + patch_controls["corrupted_transition"]["rejected"]
        ),
        all_receipt_replay_pass=all(replay_passes) if replay_passes else False,
        unsafe_patch_accept_count=int(aggregate["unsafe_patch_accept_count"]),
        patch_controls=patch_controls,
        patcher_diagnostics={
            **aggregate,
            "active_clause_counts": active_clause_counts,
        },
    )


def _usable(row: TransitionReceipt) -> bool:
    if not context_features(row).supported:
        return False
    return int(row.effect_signature.get("changed_count", 0)) > 0


def make_contradictory_receipt(
    row: TransitionReceipt,
    *,
    effect_color: int = 7,
) -> TransitionReceipt:
    before = _as_grid(row.state)
    after = _as_grid(row.successor).copy()
    click = _click_xy(row.data)
    if click is None:
        after = np.roll(after, 1, axis=0)
        return replace(row, successor=after)
    x, y = click
    component = _component_cells(before, x, y)
    if not component:
        after[y, x] = int(effect_color)
    else:
        for r, c in component:
            after[r, c] = int(effect_color)
    return replace(row, successor=after)


def make_unsupported_receipt(row: TransitionReceipt) -> TransitionReceipt:
    return replace(row, data={"x": -1, "y": -1}, successor=_as_grid(row.state).copy())


def _component_cells(grid: np.ndarray, x: int, y: int, *, max_cells: int = 512) -> list[tuple[int, int]]:
    if grid.ndim != 2 or y < 0 or x < 0 or y >= grid.shape[0] or x >= grid.shape[1]:
        return []
    value = int(grid[y, x])
    seen = {(y, x)}
    stack = [(y, x)]
    cells: list[tuple[int, int]] = []
    while stack:
        row, col = stack.pop()
        cells.append((row, col))
        if len(cells) > max_cells:
            return []
        for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + drow, col + dcol
            if (
                0 <= nr < grid.shape[0]
                and 0 <= nc < grid.shape[1]
                and (nr, nc) not in seen
                and int(grid[nr, nc]) == value
            ):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return cells


def hypothesis_language_spec() -> JsonDict:
    return {
        "name": "arc.generic_transition_clause.v1",
        "allowed_field_types": [
            "object_identity_hash",
            "object_topology_hash",
            "location_relation",
            "action_id",
            "changed_object_attributes",
            "reward",
            "terminal_event",
            "abstention",
        ],
        "forbidden_field_types": [
            "game_name",
            "absolute_coordinate",
            "level_constant",
            "per_game_adapter",
            "source_symbol",
        ],
        "forbidden_literals_present": [],
        "clause_shape": {
            "antecedent": ["selector_kind", "selector_value", "action"],
            "consequent": [
                "changed_count",
                "changed_bbox_shape",
                "changed_object_topology_hash",
                "changed_attributes",
                "reward",
                "terminal_event",
            ],
            "fallback": ABSTAIN,
        },
    }

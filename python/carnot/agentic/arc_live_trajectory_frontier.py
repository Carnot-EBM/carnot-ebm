"""Live trajectory-derived frontier prefixes for ARC exploration.

Spec refs: REQ-ARC-FCP-5410, SCENARIO-ARC-FCP-5410.

The live ARC agent needs candidate generation that is still honest when the
game is unseen. This helper only learns from transitions the agent just caused:
it records action -> visible frame-change evidence, combines that support with
the existing color-blob salience prior, and emits short action prefixes only
when the support and uncertainty gate agree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from carnot.agentic.arc_color_blob_salience import ColorBlobSaliencePrior


ActionStep = dict[str, Any]


def _grid(frame: Any) -> np.ndarray:
    value = frame.frame if hasattr(frame, "frame") else frame
    arr = np.asarray(value)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        return np.zeros((1, 1), dtype=np.int16)
    return arr.astype(np.int16, copy=False)


def _candidate_action(candidate: Any) -> int:
    if isinstance(candidate, Mapping):
        return int(candidate.get("action", candidate.get("action_id", 0)) or 0)
    return int(getattr(candidate, "action", getattr(candidate, "action_id", 0)) or 0)


def _candidate_data(candidate: Any) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get("data")
    return getattr(candidate, "data", None)


def _step(action: int, data: Any) -> ActionStep:
    return {"action": int(action), "data": dict(data) if isinstance(data, Mapping) else data}


def _signature(action: int, data: Any) -> tuple[Any, ...]:
    if isinstance(data, Mapping):
        return (int(action), tuple(sorted((str(key), data[key]) for key in data)))
    return (int(action), data)


def _candidate_signature(candidate: Any) -> tuple[Any, ...]:
    return _signature(_candidate_action(candidate), _candidate_data(candidate))


def _level(frame: Any) -> int:
    for name in ("levels_completed", "level_progress", "level"):
        value = getattr(frame, name, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                return 0
    return 0


def _frame_hash(frame: Any) -> str:
    grid = _grid(frame)
    h = hashlib.sha256()
    h.update(str(tuple(grid.shape)).encode("ascii"))
    h.update(grid.tobytes())
    return h.hexdigest()[:16]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=repr)


def _first_action_allowed(sequence: Sequence[ActionStep], candidates: Sequence[Any]) -> bool:
    if not sequence:
        return False
    first = sequence[0]
    first_sig = _signature(int(first["action"]), first.get("data"))
    return any(_candidate_signature(candidate) == first_sig for candidate in candidates)


@dataclass(frozen=True)
class FrontierGateRow:
    """Auditable support row for one inferred live action dynamic."""

    action: int
    data: Any
    support_count: int
    effect_count: int
    effect_rate: float
    uncertainty: float
    accepted: bool
    salience_route: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": int(self.action),
            "data": dict(self.data) if isinstance(self.data, Mapping) else self.data,
            "support_count": int(self.support_count),
            "effect_count": int(self.effect_count),
            "effect_rate": float(self.effect_rate),
            "uncertainty": float(self.uncertainty),
            "accepted": bool(self.accepted),
            "salience_route": str(self.salience_route),
        }


class LiveTrajectoryFrontierGenerator:
    """Blob-salience action prior plus uncertainty-gated prefix generator.

    The object intentionally satisfies two live-path protocols: as an
    ``action_prior`` it supplies blob-ordered click points and observes realized
    transitions; as a ``qd_generator`` it can return a short prefix through the
    existing ``StepwiseExplorer`` sequence-injection hook.
    """

    verifier_is_oracle = False

    def __init__(
        self,
        *,
        base_prior: ColorBlobSaliencePrior | None = None,
        min_support: int = 2,
        max_uncertainty: float = 0.4,
        min_effect_rate: float = 0.5,
        min_changed_cells: int = 1,
    ) -> None:
        self.base_prior = base_prior or ColorBlobSaliencePrior()
        self.min_support = max(1, int(min_support))
        self.max_uncertainty = max(0.0, min(1.0, float(max_uncertainty)))
        self.min_effect_rate = max(0.0, min(1.0, float(min_effect_rate)))
        self.min_changed_cells = max(1, int(min_changed_cells))
        self._stats: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._frontier_expansions: list[dict[str, Any]] = []
        self._uncertainty_rejections = 0
        self._salience_routes_used: list[str] = []

    def click_points(self, frame: Any, *, max_points: int | None = None) -> list[tuple[int, int]]:
        return self.base_prior.click_points(frame, max_points=max_points)

    def tier_rows(self, frame: Any) -> list[dict[str, Any]]:
        return self.base_prior.tier_rows(frame)

    def action_tier_rows(self, frame: Any, candidates: Sequence[Any]) -> list[dict[str, Any]]:
        return self.base_prior.action_tier_rows(frame, candidates)

    def score(self, frame: Any, candidate: Any) -> float:
        base = float(self.base_prior.score(frame, candidate))
        row = self._gate_row(_candidate_action(candidate), _candidate_data(candidate), frame)
        return base + (100.0 * (1.0 - row.uncertainty) if row.accepted else 0.0)

    def observe_transition(self, before: Any, action: int, data: Any, after: Any) -> None:
        before_grid = _grid(before)
        after_grid = _grid(after)
        changed = 0
        if before_grid.shape == after_grid.shape:
            changed = int(np.count_nonzero(before_grid != after_grid))
        route = self._salience_route(before, int(action), data)
        key = _signature(int(action), data)
        row = self._stats.setdefault(
            key,
            {
                "action": int(action),
                "data": dict(data) if isinstance(data, Mapping) else data,
                "support_count": 0,
                "effect_count": 0,
                "salience_route": route,
            },
        )
        row["support_count"] = int(row["support_count"]) + 1
        if changed >= self.min_changed_cells:
            row["effect_count"] = int(row["effect_count"]) + 1
        row["last_changed_cells"] = int(changed)
        row["salience_route"] = route

    def best_sequence(
        self,
        frame: Any,
        candidates: Sequence[Any],
        *,
        goal_energy: Any | None = None,
        action_effect_scorer: Any | None = None,
        min_len: int = 2,
    ) -> tuple[ActionStep, ...]:
        del goal_energy, action_effect_scorer
        rows = [self._gate_row(_candidate_action(c), _candidate_data(c), frame) for c in candidates]
        accepted = [row for row in rows if row.accepted]
        if not accepted:
            self._uncertainty_rejections += 1
            return tuple()
        accepted.sort(key=lambda row: (row.uncertainty, -row.effect_rate, row.salience_route))
        first = accepted[0]
        sequence = [_step(first.action, first.data)]
        seen = {_signature(first.action, first.data)}
        for candidate in candidates:
            sig = _candidate_signature(candidate)
            if sig in seen:
                continue
            sequence.append(_step(_candidate_action(candidate), _candidate_data(candidate)))
            seen.add(sig)
            if len(sequence) >= int(min_len):
                break
        while len(sequence) < int(min_len):
            sequence.append(dict(sequence[0]))
        route = first.salience_route
        if route not in self._salience_routes_used:
            self._salience_routes_used.append(route)
        expansion = {
            "prefix": [dict(step) for step in sequence],
            "support_count": int(first.support_count),
            "uncertainty": float(first.uncertainty),
            "salience_route": route,
            "accepted": True,
        }
        self._frontier_expansions.append(expansion)
        return tuple(dict(step) for step in sequence)

    def diagnostics(self) -> dict[str, Any]:
        observations = [
            self._gate_row(int(row["action"]), row.get("data"), None).as_dict()
            for row in self._stats.values()
        ]
        observations.sort(key=lambda row: (row["action"], repr(row["data"])))
        return {
            "enabled": True,
            "source": "live_trajectory_frontier_blob_salience_uncertainty_gate",
            "frontier_expansion_count": int(len(self._frontier_expansions)),
            "frontier_expansions": [dict(row) for row in self._frontier_expansions],
            "salience_routes_used": list(self._salience_routes_used),
            "uncertainty_rejections": int(self._uncertainty_rejections),
            "verifier_observations": observations,
            "observed_transition_count": int(
                sum(int(row["support_count"]) for row in self._stats.values())
            ),
            "verifier_is_oracle": False,
        }

    def as_dict(self) -> dict[str, Any]:
        base = self.base_prior.as_dict()
        base.update(
            {
                "source": "live_trajectory_frontier_blob_salience_uncertainty_gate",
                "trajectory_frontier_generation_enabled": True,
                "uncertainty_gate_enabled": True,
                "min_support": int(self.min_support),
                "max_uncertainty": float(self.max_uncertainty),
            }
        )
        return base

    def _salience_route(self, frame: Any, action: int, data: Any) -> str:
        if int(action) != 6:
            return "keyboard_action"
        try:
            rows = self.base_prior.action_tier_rows(
                frame,
                [{"action": int(action), "data": dict(data) if isinstance(data, Mapping) else data}],
            )
        except Exception:
            rows = []
        row = rows[0] if rows else {}
        tier = row.get("tier")
        if tier is None:
            return "blob_tier_unknown"
        if row.get("button_like"):
            return f"blob_tier_{int(tier)}_button_like"
        if row.get("status_bar"):
            return f"blob_tier_{int(tier)}_status"
        if row.get("large_flat"):
            return f"blob_tier_{int(tier)}_large_flat"
        return f"blob_tier_{int(tier)}_color"

    def _gate_row(self, action: int, data: Any, frame: Any | None) -> FrontierGateRow:
        key = _signature(int(action), data)
        stat = self._stats.get(key)
        if stat is None:
            route = "unobserved"
            if frame is not None:
                route = self._salience_route(frame, int(action), data)
            return FrontierGateRow(
                action=int(action),
                data=dict(data) if isinstance(data, Mapping) else data,
                support_count=0,
                effect_count=0,
                effect_rate=0.0,
                uncertainty=1.0,
                accepted=False,
                salience_route=route,
            )
        support = int(stat["support_count"])
        effect = int(stat["effect_count"])
        effect_rate = float(effect) / float(max(1, support))
        uncertainty = min(1.0, (1.0 / float(support + 1)) + ((1.0 - effect_rate) * 0.5))
        accepted = bool(
            support >= self.min_support
            and effect_rate >= self.min_effect_rate
            and uncertainty <= self.max_uncertainty
        )
        return FrontierGateRow(
            action=int(action),
            data=stat.get("data"),
            support_count=support,
            effect_count=effect,
            effect_rate=effect_rate,
            uncertainty=uncertainty,
            accepted=accepted,
            salience_route=str(stat.get("salience_route") or "observed_unknown"),
        )


class LiveCoExLandmarkFrontierGenerator(LiveTrajectoryFrontierGenerator):
    """CoEx-style persistent frontier generator over live-observed transitions.

    Spec refs: REQ-ARC-FCP-5423, SCENARIO-ARC-FCP-5423.
    """

    def __init__(
        self,
        *,
        max_persisted_frontiers: int = 16,
        landmark_min_changed_cells: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_persisted_frontiers = max(1, int(max_persisted_frontiers))
        self.landmark_min_changed_cells = max(1, int(landmark_min_changed_cells))
        self._current_path: list[ActionStep] = []
        self._reset_count = 0
        self._measurement_receipts: list[dict[str, Any]] = []
        self._runtime_observations: list[dict[str, Any]] = []
        self._frontier_transitions: list[dict[str, Any]] = []
        self._landmarks: list[dict[str, Any]] = []
        self._landmark_keys: set[tuple[str, tuple[Any, ...]]] = set()
        self._persistent_frontiers: list[dict[str, Any]] = []
        self._action_clusters: dict[str, dict[str, Any]] = {}
        self._coex_action_sequence_receipts: list[dict[str, Any]] = []

    def record_reset(self, *, level: int | None = None) -> None:
        del level
        self._reset_count += 1
        self._current_path = []

    def reset(self, *, level: int | None = None, reset_to_prior: bool = True) -> None:
        del level, reset_to_prior
        self._current_path = []

    def observe_transition(self, before: Any, action: int, data: Any, after: Any) -> None:
        super().observe_transition(before, action, data, after)
        before_grid = _grid(before)
        after_grid = _grid(after)
        changed = (
            int(np.count_nonzero(before_grid != after_grid))
            if before_grid.shape == after_grid.shape
            else 0
        )
        step = _step(int(action), data)
        before_hash = _frame_hash(before)
        after_hash = _frame_hash(after)
        level_before = _level(before)
        level_after = _level(after)
        receipt = {
            "receipt_id": f"m{len(self._measurement_receipts) + 1:04d}",
            "before_hash": before_hash,
            "after_hash": after_hash,
            "action": int(action),
            "data": dict(data) if isinstance(data, Mapping) else data,
            "changed_cells": int(changed),
            "level_before": int(level_before),
            "level_after": int(level_after),
            "path_len_before": int(len(self._current_path)),
        }
        self._measurement_receipts.append(receipt)
        self._runtime_observations.append(
            {
                "action": int(action),
                "data": receipt["data"],
                "changed_cells": int(changed),
                "level_before": int(level_before),
                "level_after": int(level_after),
                "receipt_id": receipt["receipt_id"],
            }
        )
        self._current_path.append(step)
        cluster_id = self._cluster_id(step)
        cluster = self._action_clusters.setdefault(
            cluster_id,
            {
                "cluster_id": cluster_id,
                "support_count": 0,
                "sequence": [dict(step)],
                "receipt_ids": [],
                "changed_cells": 0,
            },
        )
        cluster["support_count"] = int(cluster["support_count"]) + 1
        cluster["changed_cells"] = int(cluster["changed_cells"]) + int(changed)
        cluster["receipt_ids"].append(receipt["receipt_id"])
        gate = self._gate_row(int(action), data, before)
        transition = {
            "from_hash": before_hash,
            "to_hash": after_hash,
            "action": int(action),
            "data": receipt["data"],
            "changed_cells": int(changed),
            "level_before": int(level_before),
            "level_after": int(level_after),
            "cluster_id": cluster_id,
            "receipt_id": receipt["receipt_id"],
            "accepted": bool(gate.accepted),
        }
        self._frontier_transitions.append(transition)
        if gate.accepted:
            self._persist_frontier(cluster_id, [step], [receipt["receipt_id"]], gate, changed)
        if changed >= self.landmark_min_changed_cells or level_after > level_before:
            self._record_landmark(
                after_hash=after_hash,
                level_after=level_after,
                changed=changed,
                receipt_id=receipt["receipt_id"],
            )

    def best_sequence(
        self,
        frame: Any,
        candidates: Sequence[Any],
        *,
        goal_energy: Any | None = None,
        action_effect_scorer: Any | None = None,
        min_len: int = 2,
    ) -> tuple[ActionStep, ...]:
        del goal_energy, action_effect_scorer
        candidates = list(candidates)
        persisted = [
            row
            for row in self._persistent_frontiers
            if _first_action_allowed(row.get("sequence") or [], candidates)
        ]
        if persisted:
            persisted.sort(key=lambda row: (-float(row.get("score", 0.0)), row["cluster_id"]))
            chosen = persisted[0]
            sequence = [dict(step) for step in chosen["sequence"]]
            first = sequence[0]
            row = self._gate_row(int(first["action"]), first.get("data"), frame)
            cluster_id = str(chosen["cluster_id"])
            receipt_ids = list(chosen.get("receipt_ids") or [])
        else:
            rows = [
                self._gate_row(_candidate_action(candidate), _candidate_data(candidate), frame)
                for candidate in candidates
            ]
            accepted = [row for row in rows if row.accepted]
            if not accepted:
                self._uncertainty_rejections += 1
                return tuple()
            accepted.sort(key=lambda row: (row.uncertainty, -row.effect_rate, row.salience_route))
            row = accepted[0]
            sequence = [_step(row.action, row.data)]
            cluster_id = self._cluster_id(sequence[0])
            receipt_ids = list((self._action_clusters.get(cluster_id) or {}).get("receipt_ids") or [])

        seen = {_signature(int(step["action"]), step.get("data")) for step in sequence}
        for candidate in candidates:
            sig = _candidate_signature(candidate)
            if sig in seen:
                continue
            sequence.append(_step(_candidate_action(candidate), _candidate_data(candidate)))
            seen.add(sig)
            if len(sequence) >= int(min_len):
                break
        while len(sequence) < int(min_len):
            sequence.append(dict(sequence[0]))
        route = row.salience_route
        if route not in self._salience_routes_used:
            self._salience_routes_used.append(route)
        expansion = {
            "prefix": [dict(step) for step in sequence],
            "support_count": int(row.support_count),
            "uncertainty": float(row.uncertainty),
            "salience_route": route,
            "cluster_id": cluster_id,
            "accepted": True,
            "source": "coex_persistent_frontier",
        }
        self._frontier_expansions.append(expansion)
        self._record_sequence_receipt(sequence, cluster_id, receipt_ids)
        return tuple(dict(step) for step in sequence)

    def diagnostics(self) -> dict[str, Any]:
        base = super().diagnostics()
        clusters = sorted(
            (
                {
                    "cluster_id": row["cluster_id"],
                    "support_count": int(row["support_count"]),
                    "sequence": [dict(step) for step in row["sequence"]],
                    "changed_cells": int(row["changed_cells"]),
                }
                for row in self._action_clusters.values()
            ),
            key=lambda row: (-row["support_count"], row["cluster_id"]),
        )
        base.update(
            {
                "source": "live_coex_landmark_frontier",
                "coex_frontier_persistence_enabled": True,
                "hierarchical_landmarks_enabled": True,
                "action_history_clustering_enabled": True,
                "measurement_access_receipts_enabled": True,
                "frontier_expansion_count": int(len(self._frontier_expansions)),
                "reset_count": int(self._reset_count),
                "frontier_transitions": [dict(row) for row in self._frontier_transitions],
                "landmark_count": int(len(self._landmarks)),
                "discovered_landmarks": [dict(row) for row in self._landmarks],
                "action_history_clusters": clusters,
                "measurement_access_receipts": [dict(row) for row in self._measurement_receipts],
                "runtime_observations": [dict(row) for row in self._runtime_observations],
                "action_sequence_receipts": [
                    {
                        **row,
                        "sequence": [dict(step) for step in row["sequence"]],
                        "measurement_receipts": [dict(item) for item in row["measurement_receipts"]],
                    }
                    for row in self._coex_action_sequence_receipts
                ],
            }
        )
        return base

    def as_dict(self) -> dict[str, Any]:
        base = super().as_dict()
        base.update(
            {
                "source": "live_coex_landmark_frontier",
                "coex_frontier_persistence_enabled": True,
                "hierarchical_landmarks_enabled": True,
                "action_history_clustering_enabled": True,
                "measurement_access_receipts_enabled": True,
                "max_persisted_frontiers": int(self.max_persisted_frontiers),
            }
        )
        return base

    def _cluster_id(self, step: Mapping[str, Any]) -> str:
        payload = {"action": int(step["action"]), "data": step.get("data")}
        digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:12]
        return f"cluster:{digest}"

    def _persist_frontier(
        self,
        cluster_id: str,
        sequence: Sequence[ActionStep],
        receipt_ids: Sequence[str],
        gate: FrontierGateRow,
        changed: int,
    ) -> None:
        row = {
            "cluster_id": cluster_id,
            "sequence": [dict(step) for step in sequence],
            "receipt_ids": list(receipt_ids),
            "score": float(gate.effect_rate) + float(changed) * 0.01,
        }
        self._persistent_frontiers = [
            item for item in self._persistent_frontiers if item["cluster_id"] != cluster_id
        ]
        self._persistent_frontiers.append(row)
        self._persistent_frontiers.sort(key=lambda item: (-float(item["score"]), item["cluster_id"]))
        del self._persistent_frontiers[self.max_persisted_frontiers :]

    def _record_landmark(
        self,
        *,
        after_hash: str,
        level_after: int,
        changed: int,
        receipt_id: str,
    ) -> None:
        signature = tuple(_signature(int(step["action"]), step.get("data")) for step in self._current_path)
        key = (after_hash, signature)
        if key in self._landmark_keys:
            return
        self._landmark_keys.add(key)
        self._landmarks.append(
            {
                "frame_hash": after_hash,
                "level_after": int(level_after),
                "reach_sequence": [dict(step) for step in self._current_path],
                "score": float(changed) + float(max(0, level_after)) * 0.1,
                "receipt_id": receipt_id,
            }
        )

    def _record_sequence_receipt(
        self,
        sequence: Sequence[ActionStep],
        cluster_id: str,
        receipt_ids: Sequence[str],
    ) -> None:
        measurements = [
            row for row in self._measurement_receipts if row["receipt_id"] in set(receipt_ids)
        ]
        if not measurements and self._measurement_receipts:
            measurements = [self._measurement_receipts[-1]]
        receipt = {
            "sequence": [dict(step) for step in sequence],
            "cluster_id": cluster_id,
            "measurement_receipts": [dict(row) for row in measurements],
            "replayable": bool(sequence),
        }
        signature = _stable_json(receipt["sequence"])
        if any(_stable_json(row["sequence"]) == signature for row in self._coex_action_sequence_receipts):
            return
        self._coex_action_sequence_receipts.append(receipt)

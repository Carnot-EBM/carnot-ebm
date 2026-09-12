"""Build bounded transition witnesses from the agent's own observation history.

The adapter explains where an executable engine disagrees with observed motion. It
does not infer a game rule or provide replacement code. The model remains responsible
for writing the next engine.

Spec: REQ-ARC-WMTE-7248 and SCENARIO-ARC-WMTE-7248-*.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, Mapping, Sequence

import numpy as np

WITNESS_ENV = "CARNOT_ARC_TRANSITION_WITNESS"
WITNESS_MARKER = "ARC_TRANSITION_WITNESS_V1"
MAX_WITNESS_MISMATCHES = 8


def witness_feedback_enabled() -> bool:
    """Keep the runtime adapter off unless the experiment selects it exactly."""

    return os.environ.get(WITNESS_ENV) == "1"


def canonical_witness_bytes(value: Mapping[str, Any]) -> bytes:
    """Use one stable encoding for request delivery and artifact receipts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _frame_hash(frame: Any) -> str:
    """Bind a witness to the exact pre-action array without storing the full frame."""

    array = np.ascontiguousarray(np.asarray(frame))
    digest = hashlib.sha256()
    digest.update(b"carnot.arc.transition.pre-frame.v1\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.dtype.str.encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def _json_action_data(value: Any) -> Any:
    """Keep normal action data exact and make uncommon values safe for JSON receipts."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_action_data(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_action_data(item) for item in value]
    return str(value)


def _coordinates(mask: np.ndarray) -> list[list[int]]:
    """Return row-major coordinates so the same evidence always has the same bytes."""

    return [[int(row), int(col)] for row, col in np.argwhere(mask).tolist()]


def _base_row(transition: Any, before: np.ndarray) -> dict[str, Any]:
    return {
        "action": {
            "action": int(getattr(transition, "action")),
            "data": _json_action_data(getattr(transition, "data", None)),
        },
        "pre_frame_hash": _frame_hash(before),
        "observed_changed_coordinates": [],
        "predicted_changed_coordinates": [],
    }


def _select_diverse(rows: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Round-robin mismatch types, then use canonical bytes to break every tie."""

    ordered = sorted(rows, key=canonical_witness_bytes)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in ordered:
        groups.setdefault(str(row["typed_mismatch"]), []).append(row)
    selected: list[dict[str, Any]] = []
    while groups and len(selected) < limit:
        for kind in sorted(tuple(groups)):
            selected.append(groups[kind].pop(0))
            if not groups[kind]:
                del groups[kind]
            if len(selected) >= limit:
                break
    return selected


def build_transition_witness(
    transitions: Sequence[Any],
    engine: Callable[[np.ndarray, int, Any], Any],
    *,
    max_mismatches: int = MAX_WITNESS_MISMATCHES,
) -> dict[str, Any]:
    """Compare one engine with agent-owned transitions and return a bounded payload.

    No-change rows still count as observations. An all-no-change corpus is unavailable
    because it cannot show whether an identity prediction missed a real action effect.
    """

    counts = {
        "total": 0,
        "observed_changed": 0,
        "observed_unchanged": 0,
        "correct_unchanged_identity": 0,
        "mismatched": 0,
    }
    mismatches: list[dict[str, Any]] = []
    valid_no_change_observations: list[dict[str, Any]] = []
    for transition in transitions:
        counts["total"] += 1
        before = np.asarray(getattr(transition, "grid"))
        observed = np.asarray(getattr(transition, "next_grid"))
        row = _base_row(transition, before)
        if before.ndim != 2 or observed.ndim != 2 or before.shape != observed.shape:
            counts["observed_changed"] += 1
            counts["mismatched"] += 1
            row["typed_mismatch"] = "observed_shape_change"
            mismatches.append(row)
            continue

        observed_mask = before != observed
        observed_coordinates = _coordinates(observed_mask)
        row["observed_changed_coordinates"] = observed_coordinates
        observed_changed = bool(observed_coordinates)
        counts["observed_changed" if observed_changed else "observed_unchanged"] += 1
        try:
            from carnot.agentic.arc_engine_call_guard import guarded_call

            predicted = np.asarray(
                guarded_call(
                    engine,
                    before.copy(),
                    int(getattr(transition, "action")),
                    getattr(transition, "data", None),
                )
            )
        except Exception as exc:  # noqa: BLE001 - the error is typed evidence for refinement.
            counts["mismatched"] += 1
            row["typed_mismatch"] = "engine_error"
            row["error_kind"] = type(exc).__name__
            mismatches.append(row)
            continue

        if predicted.ndim != 2 or predicted.shape != before.shape:
            counts["mismatched"] += 1
            row["typed_mismatch"] = "malformed_output"
            row["predicted_shape"] = [int(value) for value in predicted.shape]
            mismatches.append(row)
            continue

        predicted_mask = before != predicted
        predicted_coordinates = _coordinates(predicted_mask)
        row["predicted_changed_coordinates"] = predicted_coordinates
        if not observed_changed:
            if not predicted_coordinates:
                counts["correct_unchanged_identity"] += 1
                row["observation_type"] = "valid_no_change_identity"
                valid_no_change_observations.append(row)
                continue
            counts["mismatched"] += 1
            row["typed_mismatch"] = "change_predicted_for_unchanged_transition"
            mismatches.append(row)
            continue
        if np.array_equal(predicted, observed):
            continue

        counts["mismatched"] += 1
        observed_set = {tuple(coord) for coord in observed_coordinates}
        predicted_set = {tuple(coord) for coord in predicted_coordinates}
        if not predicted_set:
            row["typed_mismatch"] = "identity_on_changed_transition"
        elif observed_set == predicted_set:
            row["typed_mismatch"] = "wrong_values_at_observed_coordinates"
        elif observed_set - predicted_set and not predicted_set - observed_set:
            row["typed_mismatch"] = "missing_observed_change_coordinates"
        else:
            row["typed_mismatch"] = "changed_coordinate_set_mismatch"
        mismatches.append(row)

    available = counts["observed_changed"] > 0
    return {
        "schema": "carnot.arc_transition_witness.v1",
        "source": "agent_observation_history",
        "available": available,
        "reason": "available" if available else "no_observed_changed_transitions",
        "readiness_score": int(available and bool(mismatches)),
        "observation_counts": counts,
        "valid_no_change_observations": sorted(
            valid_no_change_observations,
            key=canonical_witness_bytes,
        )[:MAX_WITNESS_MISMATCHES],
        "mismatches": (
            _select_diverse(mismatches, max(0, min(int(max_mismatches), MAX_WITNESS_MISMATCHES)))
            if available
            else []
        ),
    }


def render_transition_witness(payload: Mapping[str, Any]) -> str:
    """Render exact JSON only; do not add a suggested rule or replacement engine."""

    return WITNESS_MARKER + "\n" + canonical_witness_bytes(payload).decode("ascii")

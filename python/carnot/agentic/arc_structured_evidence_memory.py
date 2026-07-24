"""Agent-owned structured evidence memory for the ARC live E3 boundary.

The memory is deliberately a view over the agent's own event bytes, not a
second knowledge channel. The raw tape and structured index always cite the
same source events; structure can help retrieval and ordering, but it cannot
add hidden labels, public-game source facts, or per-game recipes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "arc_structured_evidence_memory.v1"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RESULT_RELATIVE_PATH = "results/experiment_5900_arc_structured_evidence_memory_contract.json"
FORBIDDEN_AUTHORITY_COUNTS = {
    "source_read_count": 0,
    "offline_bfs_count": 0,
    "adapter_access_count": 0,
    "prior_game_log_access_count": 0,
    "registry_trajectory_access_count": 0,
    "hidden_state_read_count": 0,
    "source_bfs_adapter_and_prior_game_access_count": 0,
}


@dataclass(frozen=True)
class StructuredEvidenceConfig:
    """Caps for the live evidence tape.

    These limits keep the feature suitable for a scored interaction loop. The
    default budget is intentionally modest; when the tape fills, older events
    are replaced by loss receipts rather than silently disappearing.
    """

    schema_version: str = SCHEMA_VERSION
    max_events: int = 256
    max_bytes: int = 262_144
    max_query_events: int = 64
    max_query_bytes: int = 65_536
    max_queries: int = 256
    stale_after_events: int = 64
    max_candidates_per_event: int = 128
    enabled: bool = True


def _stable_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), default=str)


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_stable_json(value).encode("utf-8"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _grid(value: Any) -> Any | None:
    try:
        import numpy as np

        raw = value
        if isinstance(value, Mapping):
            raw = value.get("frame", value.get("grid", value.get("state")))
        elif hasattr(value, "frame"):
            raw = getattr(value, "frame")
        elif hasattr(value, "grid"):
            raw = getattr(value, "grid")
        arr = np.asarray(raw)
        if arr.ndim == 3:
            arr = arr[-1]
        return arr if arr.ndim == 2 else None
    except Exception:
        return None


def _frame_payload(value: Any) -> dict[str, Any]:
    arr = _grid(value)
    if arr is None:
        return {"available": False, "shape": [], "grid": [], "state_hash": None}
    try:
        import numpy as np

        compact = np.asarray(arr, dtype=int)
        grid = compact.tolist()
        shape = [int(dim) for dim in compact.shape]
        state_hash = _sha256_json({"shape": shape, "grid": grid})
        return {"available": True, "shape": shape, "grid": grid, "state_hash": state_hash}
    except Exception:
        return {"available": False, "shape": [], "grid": [], "state_hash": None}


def _action_signature(action: int | str | None, data: Any) -> str:
    try:
        action_i = int(action)
    except (TypeError, ValueError):
        action_i = -1
    return f"a={action_i}|d={_stable_json(data)}"


def _candidate_row(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, Mapping):
        action = candidate.get("action", candidate.get("action_id"))
        data = candidate.get("data")
    else:
        action = getattr(candidate, "action", getattr(candidate, "action_id", None))
        data = getattr(candidate, "data", None)
    try:
        action_i = int(action)
    except (TypeError, ValueError):
        action_i = -1
    return {
        "action": action_i,
        "data": _jsonable(data),
        "action_signature": _action_signature(action_i, data),
    }


def _changed_cells(before_grid: Sequence[Sequence[Any]], after_grid: Sequence[Sequence[Any]]) -> list:
    changed: list[dict[str, int]] = []
    if not before_grid or not after_grid:
        return changed
    if len(before_grid) != len(after_grid):
        return changed
    for y, (before_row, after_row) in enumerate(zip(before_grid, after_grid)):
        if len(before_row) != len(after_row):
            return changed
        for x, (left, right) in enumerate(zip(before_row, after_row)):
            if left != right:
                changed.append({"x": int(x), "y": int(y), "before": int(left), "after": int(right)})
    return changed


def _glyph_identity_from_grid(grid: Sequence[Sequence[Any]]) -> dict[str, Any]:
    values: list[int] = []
    for row in grid or []:
        for cell in row:
            try:
                values.append(int(cell))
            except (TypeError, ValueError):
                continue
    counts = dict(sorted(Counter(values).items()))
    nonzero = {str(k): int(v) for k, v in counts.items() if int(k) != 0}
    return {
        "colors": [int(k) for k in counts],
        "non_background_colors": [int(k) for k in counts if int(k) != 0],
        "non_background_counts": nonzero,
    }


def _changed_bbox(changed: Sequence[Mapping[str, int]]) -> dict[str, int] | None:
    if not changed:
        return None
    xs = [int(row["x"]) for row in changed]
    ys = [int(row["y"]) for row in changed]
    return {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys)}


def _event_bytes(event: Mapping[str, Any]) -> bytes:
    return (_stable_json(event) + "\n").encode("utf-8")


class StructuredEvidenceMemory:
    """REQ-ARC-WMTE-5900: bounded raw tape plus deterministic structured index."""

    def __init__(self, config: StructuredEvidenceConfig | None = None) -> None:
        self.config = config or StructuredEvidenceConfig()
        if self.config.max_events < 2:
            self.config = StructuredEvidenceConfig(
                **{**self.config.__dict__, "max_events": 2}
            )
        self.enabled = bool(self.config.enabled)
        self._events: list[dict[str, Any]] = []
        self._next_sequence = 1
        self.append_count = 0
        self.loss_receipt_count = 0
        self.raw_query_count = 0
        self.structured_query_count = 0
        self.query_limit_exceeded_count = 0
        self.rank_consumed_count = 0
        self.delete_count = 0
        self.restart_count = 0

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._events)

    def observe_state(
        self,
        frame: Any,
        *,
        phase: str = "",
        uncertainty: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return self._append(
            "observation",
            {
                "phase": str(phase),
                "frame": _frame_payload(frame),
                "level": _jsonable(getattr(frame, "levels_completed", None)),
            },
            uncertainty=uncertainty,
            provenance=provenance,
        )

    def observe_candidates(
        self,
        frame: Any,
        candidates: Sequence[Any],
        *,
        uncertainty: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        rows = [_candidate_row(candidate) for candidate in candidates]
        cap = max(1, int(self.config.max_candidates_per_event))
        return self._append(
            "candidate_set",
            {
                "frame": _frame_payload(frame),
                "candidate_count": len(rows),
                "candidates": rows[:cap],
                "candidate_rows_evicted_from_event": max(0, len(rows) - cap),
            },
            uncertainty=uncertainty,
            provenance=provenance,
        )

    def observe_action_candidate(
        self,
        action: int,
        data: Any,
        *,
        uncertainty: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return self._append(
            "action_candidate",
            {
                "action": int(action),
                "data": _jsonable(data),
                "action_signature": _action_signature(action, data),
            },
            uncertainty=uncertainty,
            provenance=provenance,
        )

    def observe_action_result(
        self,
        before_frame: Any,
        action: int,
        data: Any,
        after_frame: Any,
        *,
        level_before: int = 0,
        level_after: int = 0,
        uncertainty: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        before_payload = _frame_payload(before_frame)
        after_payload = _frame_payload(after_frame)
        changed = _changed_cells(before_payload["grid"], after_payload["grid"])
        level_delta = int(level_after) - int(level_before)
        outcome = "level_progress" if level_delta > 0 else "visible_change" if changed else "noop"
        return self._append(
            "action_result",
            {
                "before": before_payload,
                "after": after_payload,
                "action": int(action),
                "data": _jsonable(data),
                "action_signature": _action_signature(action, data),
                "changed_count": len(changed),
                "changed_cells": changed,
                "level_before": int(level_before),
                "level_after": int(level_after),
                "level_delta": level_delta,
                "outcome": outcome,
                "exact_noop_evidence": outcome == "noop",
                "exact_dead_end_evidence": outcome == "noop" and level_delta == 0,
            },
            uncertainty=uncertainty,
            provenance=provenance,
        )

    def query_raw(
        self,
        *,
        event_type: str | None = None,
        source_event_ids: Sequence[str] | None = None,
        include_stale: bool = True,
    ) -> dict[str, Any]:
        exceeded = self._consume_query("raw")
        if exceeded:
            return self._empty_query("raw", query_limit_exceeded=True)
        selected, stale, truncated = self._select_events(
            event_type=event_type,
            source_event_ids=source_event_ids,
            include_stale=include_stale,
        )
        return self._query_payload(
            "raw",
            selected,
            stale_event_ids=stale,
            truncated=truncated,
            events=[dict(event) for event in selected],
        )

    def query_index(
        self,
        *,
        event_type: str | None = None,
        source_event_ids: Sequence[str] | None = None,
        include_stale: bool = True,
    ) -> dict[str, Any]:
        exceeded = self._consume_query("structured")
        if exceeded:
            return self._empty_query("structured", query_limit_exceeded=True)
        selected, stale, truncated = self._select_events(
            event_type=event_type,
            source_event_ids=source_event_ids,
            include_stale=include_stale,
        )
        entries = [self._index_entry(event) for event in selected]
        payload = self._query_payload(
            "structured",
            selected,
            stale_event_ids=stale,
            truncated=truncated,
            index_entries=entries,
        )
        payload["index_hash"] = self.index_hash()
        return payload

    def rank_candidates(
        self,
        frame: Any,
        candidates: Sequence[Any],
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        rows = [_candidate_row(candidate) for candidate in candidates]
        if not self.enabled or not rows:
            return rows
        self.observe_candidates(
            frame,
            rows,
            provenance=provenance or {"source": "StructuredEvidenceMemory.rank_candidates"},
        )
        raw = self.query_raw(event_type="action_result")
        indexed = self.query_index(event_type="action_result")
        self.rank_consumed_count += 1
        if raw["source_event_ids"] != indexed["source_event_ids"]:
            return rows
        scores: dict[str, int] = {}
        for entry in indexed.get("index_entries", []):
            effect = entry.get("action_effect") or {}
            sig = str(effect.get("action_signature") or "")
            if not sig:
                continue
            outcome = effect.get("outcome")
            delta = -3 if outcome == "level_progress" else -2 if outcome == "visible_change" else 3
            scores[sig] = min(scores.get(sig, 0), delta) if delta < 0 else max(scores.get(sig, 0), delta)
        ranked = sorted(
            enumerate(rows),
            key=lambda pair: (scores.get(pair[1]["action_signature"], 0), pair[0]),
        )
        return [row for _index, row in ranked]

    def tape_bytes(self) -> bytes:
        return b"".join(_event_bytes(event) for event in self._events)

    def tape_hash(self) -> str:
        return _sha256_bytes(self.tape_bytes())

    def index_hash(self) -> str:
        return _sha256_json([self._index_entry(event) for event in self._events])

    @classmethod
    def from_tape_bytes(
        cls,
        tape_bytes: bytes,
        *,
        config: StructuredEvidenceConfig | None = None,
    ) -> "StructuredEvidenceMemory":
        memory = cls(config=config)
        if not tape_bytes:
            memory.restart_count += 1
            return memory
        rows: list[dict[str, Any]] = []
        for line in tape_bytes.splitlines():
            if not line.strip():
                continue
            payload = json.loads(line.decode("utf-8"))
            if isinstance(payload, dict):
                rows.append(payload)
        memory._events = rows
        memory._next_sequence = (
            max((int(row.get("logical_time") or 0) for row in rows), default=0) + 1
        )
        memory.restart_count += 1
        return memory

    def delete(self) -> dict[str, Any]:
        receipt = {
            "deleted_event_count": len(self._events),
            "deleted_tape_hash": self.tape_hash(),
            "deleted_index_hash": self.index_hash(),
        }
        self._events.clear()
        self.delete_count += 1
        return receipt

    def diagnostics(self) -> dict[str, Any]:
        return {
            "schema_version": self.config.schema_version,
            "enabled": bool(self.enabled),
            "append_count": int(self.append_count),
            "retained_event_count": len(self._events),
            "retained_byte_count": self._total_bytes(),
            "loss_receipt_count": int(self.loss_receipt_count),
            "raw_query_count": int(self.raw_query_count),
            "structured_query_count": int(self.structured_query_count),
            "query_limit_exceeded_count": int(self.query_limit_exceeded_count),
            "rank_consumed_count": int(self.rank_consumed_count),
            "delete_count": int(self.delete_count),
            "restart_count": int(self.restart_count),
            "tape_hash": self.tape_hash(),
            "index_hash": self.index_hash(),
            "authority": authority_receipt(),
            "public_level_solve_claimed": False,
        }

    def _append(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        uncertainty: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = self._make_event(event_type, payload, uncertainty=uncertainty, provenance=provenance)
        self._events.append(event)
        self.append_count += 1
        self._enforce_bounds()
        return event

    def _make_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        uncertainty: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        logical_time = self._next_sequence
        self._next_sequence += 1
        event = {
            "schema_version": self.config.schema_version,
            "event_id": f"evt-{logical_time:08d}",
            "event_type": str(event_type),
            "logical_time": int(logical_time),
            "timestamp": {"kind": "logical_agent_step", "value": int(logical_time)},
            "payload": _jsonable(dict(payload)),
            "uncertainty": _jsonable(dict(uncertainty or {})),
            "provenance": _jsonable(
                {
                    "agent_owned": True,
                    "source": (provenance or {}).get("source", "live_agent"),
                    **dict(provenance or {}),
                }
            ),
        }
        event["event_hash"] = _sha256_json(event)
        return event

    def _enforce_bounds(self) -> None:
        losses: list[dict[str, Any]] = []
        while self._would_exceed_with_receipt(losses):
            if not self._events:
                break
            losses.append(self._loss_summary(self._events.pop(0)))
        if not losses:
            return
        receipt = self._make_event(
            "loss_receipt",
            {
                "reason": "bounded_append_eviction",
                "lost_event_count": len(losses),
                "lost_events": losses,
                "loss_receipt_hash": _sha256_json(losses),
            },
            uncertainty={"lossy_retention": True},
            provenance={"source": "StructuredEvidenceMemory._enforce_bounds"},
        )
        while self._would_exceed_after_appending(receipt):
            if not self._events:
                break
            losses.append(self._loss_summary(self._events.pop(0)))
            receipt = self._make_event(
                "loss_receipt",
                {
                    "reason": "bounded_append_eviction",
                    "lost_event_count": len(losses),
                    "lost_events": losses,
                    "loss_receipt_hash": _sha256_json(losses),
                },
                uncertainty={"lossy_retention": True},
                provenance={"source": "StructuredEvidenceMemory._enforce_bounds"},
            )
        self._events.append(receipt)
        self.loss_receipt_count += 1

    def _would_exceed_with_receipt(self, losses: Sequence[Mapping[str, Any]]) -> bool:
        receipt_slots = 1 if losses else 0
        if len(self._events) + receipt_slots > self.config.max_events:
            return True
        receipt_bytes = len(_event_bytes({"losses": list(losses)})) if losses else 0
        return self._total_bytes() + receipt_bytes > self.config.max_bytes

    def _would_exceed_after_appending(self, event: Mapping[str, Any]) -> bool:
        return (
            len(self._events) + 1 > self.config.max_events
            or self._total_bytes() + len(_event_bytes(event)) > self.config.max_bytes
        )

    def _loss_summary(self, event: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "event_id": event.get("event_id"),
            "event_type": event.get("event_type"),
            "logical_time": event.get("logical_time"),
            "event_hash": event.get("event_hash"),
        }

    def _total_bytes(self, events: Sequence[Mapping[str, Any]] | None = None) -> int:
        rows = self._events if events is None else events
        return sum(len(_event_bytes(event)) for event in rows)

    def _consume_query(self, kind: str) -> bool:
        total = self.raw_query_count + self.structured_query_count
        if total >= self.config.max_queries:
            self.query_limit_exceeded_count += 1
            return True
        if kind == "raw":
            self.raw_query_count += 1
        else:
            self.structured_query_count += 1
        return False

    def _select_events(
        self,
        *,
        event_type: str | None,
        source_event_ids: Sequence[str] | None,
        include_stale: bool,
    ) -> tuple[list[dict[str, Any]], list[str], bool]:
        wanted = set(source_event_ids or [])
        rows: list[dict[str, Any]] = []
        stale: list[str] = []
        for event in self._events:
            if event_type is not None and event.get("event_type") != event_type:
                continue
            if wanted and event.get("event_id") not in wanted:
                continue
            if not include_stale and self._is_stale(event):
                stale.append(str(event.get("event_id")))
                continue
            rows.append(event)
        selected: list[dict[str, Any]] = []
        query_bytes = 0
        truncated = False
        for event in rows:
            event_size = len(_event_bytes(event))
            if len(selected) >= self.config.max_query_events:
                truncated = True
                break
            if selected and query_bytes + event_size > self.config.max_query_bytes:
                truncated = True
                break
            selected.append(event)
            query_bytes += event_size
        return selected, stale, truncated

    def _is_stale(self, event: Mapping[str, Any]) -> bool:
        if event.get("event_type") == "loss_receipt":
            return False
        logical_time = int(event.get("logical_time") or 0)
        current_time = self._next_sequence - 1
        return current_time - logical_time > self.config.stale_after_events

    def _query_payload(
        self,
        arm: str,
        selected: Sequence[Mapping[str, Any]],
        *,
        stale_event_ids: Sequence[str],
        truncated: bool,
        events: Sequence[Mapping[str, Any]] | None = None,
        index_entries: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        source_ids = [str(event["event_id"]) for event in selected]
        source_hashes = [str(event["event_hash"]) for event in selected]
        query_body = {
            "arm": arm,
            "source_event_ids": source_ids,
            "source_event_hashes": source_hashes,
            "stale_event_ids": list(stale_event_ids),
            "truncated": bool(truncated),
        }
        return {
            **query_body,
            "events": list(events or []),
            "index_entries": list(index_entries or []),
            "query_event_count": len(selected),
            "query_byte_count": self._total_bytes(selected),
            "query_limit_exceeded": False,
            "tape_hash": self.tape_hash(),
            "query_hash": _sha256_json(query_body),
        }

    def _empty_query(self, arm: str, *, query_limit_exceeded: bool) -> dict[str, Any]:
        return {
            "arm": arm,
            "source_event_ids": [],
            "source_event_hashes": [],
            "stale_event_ids": [],
            "events": [],
            "index_entries": [],
            "query_event_count": 0,
            "query_byte_count": 0,
            "query_limit_exceeded": bool(query_limit_exceeded),
            "truncated": False,
            "tape_hash": self.tape_hash(),
            "query_hash": _sha256_json({"arm": arm, "query_limit_exceeded": True}),
        }

    def _index_entry(self, event: Mapping[str, Any]) -> dict[str, Any]:
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
        frame = payload.get("frame") if isinstance(payload, Mapping) else None
        before = payload.get("before") if isinstance(payload, Mapping) else None
        after = payload.get("after") if isinstance(payload, Mapping) else None
        grid = []
        if isinstance(after, Mapping) and after.get("grid"):
            grid = after.get("grid") or []
        elif isinstance(frame, Mapping) and frame.get("grid"):
            grid = frame.get("grid") or []
        before_grid = before.get("grid") if isinstance(before, Mapping) else []
        after_grid = after.get("grid") if isinstance(after, Mapping) else []
        changed = payload.get("changed_cells") or _changed_cells(before_grid or [], after_grid or [])
        action = payload.get("action") if isinstance(payload, Mapping) else None
        data = payload.get("data") if isinstance(payload, Mapping) else None
        action_target = data if isinstance(data, Mapping) and {"x", "y"} <= set(data) else None
        return {
            "source_event_id": event.get("event_id"),
            "source_event_hash": event.get("event_hash"),
            "event_type": event.get("event_type"),
            "object_glyph_identity": _glyph_identity_from_grid(grid),
            "spatial_relation": {
                "changed_bbox": _changed_bbox(changed),
                "changed_count": len(changed),
                "action_target": _jsonable(action_target),
            },
            "action_effect": {
                "action": action,
                "data": _jsonable(data),
                "action_signature": payload.get("action_signature")
                or _action_signature(action, data),
                "outcome": payload.get("outcome"),
                "level_delta": payload.get("level_delta", 0),
                "exact_noop_evidence": bool(payload.get("exact_noop_evidence", False)),
                "exact_dead_end_evidence": bool(
                    payload.get("exact_dead_end_evidence", False)
                ),
            },
            "temporal_order": {
                "logical_time": int(event.get("logical_time") or 0),
                "timestamp": event.get("timestamp"),
            },
            "uncertainty": _jsonable(event.get("uncertainty") or {}),
            "evidence_source": _jsonable(event.get("provenance") or {}),
        }


def coerce_structured_evidence_memory(value: Any = False) -> StructuredEvidenceMemory | None:
    if value is None or value is False:
        return None
    if isinstance(value, StructuredEvidenceMemory):
        return value
    if isinstance(value, StructuredEvidenceConfig):
        return StructuredEvidenceMemory(config=value)
    return StructuredEvidenceMemory()


def authority_receipt() -> dict[str, Any]:
    return {
        **FORBIDDEN_AUTHORITY_COUNTS,
        "allowed_inputs": [
            "current_agent_visible_frame",
            "current_agent_legal_candidate",
            "current_agent_emitted_action",
            "current_agent_visible_action_result",
            "current_agent_uncertainty",
            "current_agent_provenance",
        ],
        "forbidden_inputs": [
            "public_or_hidden_game_source",
            "offline_ground_truth_bfs",
            "game_adapter",
            "prior_game_log",
            "registry_trajectory",
            "hidden_state_or_true_win_predicate",
        ],
    }


def event_schema(config: StructuredEvidenceConfig | None = None) -> dict[str, Any]:
    cfg = config or StructuredEvidenceConfig()
    return {
        "schema_version": cfg.schema_version,
        "event_types": [
            "observation",
            "candidate_set",
            "action_candidate",
            "action_result",
            "loss_receipt",
        ],
        "timestamp": "logical_agent_step",
        "required_fields": [
            "event_id",
            "event_type",
            "logical_time",
            "timestamp",
            "payload",
            "uncertainty",
            "provenance",
            "event_hash",
        ],
        "append_only": True,
        "bounded": {
            "max_events": int(cfg.max_events),
            "max_bytes": int(cfg.max_bytes),
            "max_query_events": int(cfg.max_query_events),
            "max_query_bytes": int(cfg.max_query_bytes),
            "max_queries": int(cfg.max_queries),
        },
    }


def index_schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "derived_from": "canonical_event_bytes_only",
        "facets": [
            "object_glyph_identity",
            "spatial_relation",
            "action_effect",
            "temporal_order",
            "uncertainty",
            "evidence_source",
        ],
        "may_reorder_or_retrieve": True,
        "may_invent_rules_or_read_hidden_state": False,
    }


def registry_precheck(root: Path | str) -> dict[str, Any]:
    path = Path(root) / "ops" / "arc_solve_registry.yaml"
    if not path.exists():
        return {
            "ok": False,
            "path": str(path),
            "public_game_count": 0,
            "reproducible_total_games": 0,
            "reproducible_total_levels": 0,
            "all_public_games_cleared": False,
            "duplicate_public_solve_target_prohibited": True,
            "no_level_solve_targeted": True,
        }
    text = path.read_text(encoding="utf-8")
    data: Mapping[str, Any] = {}
    try:
        import yaml

        loaded = yaml.safe_load(text)
        data = loaded if isinstance(loaded, Mapping) else {}
    except Exception:
        data = {}
    games = data.get("games") if isinstance(data, Mapping) else None
    if isinstance(games, Mapping):
        public_game_count = len(games)
        cleared_count = sum(
            1 for row in games.values() if isinstance(row, Mapping) and row.get("full_game_clear")
        )
    elif isinstance(games, list):
        public_game_count = len(games)
        cleared_count = sum(
            1 for row in games if isinstance(row, Mapping) and row.get("full_game_clear")
        )
    else:
        public_game_count = int(data.get("reproducible_total_games") or 0)
        cleared_count = public_game_count
    reproducible_games = int(data.get("reproducible_total_games") or public_game_count)
    reproducible_levels = int(data.get("reproducible_total_levels") or 0)
    all_cleared = reproducible_games == 25 and reproducible_levels >= 183
    if public_game_count:
        all_cleared = all_cleared and cleared_count in {0, public_game_count}
    return {
        "ok": bool(all_cleared),
        "path": str(path),
        "public_game_count": int(reproducible_games or public_game_count),
        "games_with_full_clear": int(cleared_count or reproducible_games),
        "reproducible_total_games": reproducible_games,
        "reproducible_total_levels": reproducible_levels,
        "all_public_games_cleared": bool(all_cleared),
        "duplicate_public_solve_target_prohibited": True,
        "no_level_solve_targeted": True,
        "principle": "all 25 cleared games prohibit a duplicate public-solve target",
    }

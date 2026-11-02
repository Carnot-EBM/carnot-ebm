"""Persistent first-party receipts for ARC tool gaps (REQ-ARC-6859).

The induction loop already detects missing tools and returns errors to the
model. This module preserves that existing exchange and joins it to the next
scored action and outcome. It never selects a tool or an action. The factory is
default-off, so an ordinary submitted run constructs nothing and writes
nothing.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from carnot.paths import results_path


RECEIPT_SCHEMA = "carnot.arc.first_party_tool_gap_receipt.v1"
ENABLE_ENV = "CARNOT_ARC_TOOL_GAP_RECEIPTS"
PATH_ENV = "CARNOT_ARC_TOOL_GAP_RECEIPT_PATH"
PROVENANCE_CLASSES = (
    "fixture",
    "terminal_replay",
    "authentic_live",
    "reconstructed",
    "development_proxy",
)
REQUIRED_HOPS = (
    "gap_detection",
    "request",
    "tool_response",
    "agent_delivery",
    "next_action",
    "exact_outcome",
)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    """Convert receipt payloads to stable JSON without reading game state."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=str)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    action_id = getattr(value, "action_id", None)
    if action_id is not None:
        return {
            "action_id": int(action_id),
            "data": _jsonable(getattr(value, "data", None)),
        }
    return {"type": type(value).__qualname__, "text": str(value)}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    try:
        data = Path(path).read_bytes()
    except OSError:
        return canonical_sha256({"unreadable_source": str(path)})
    return "sha256:" + hashlib.sha256(data).hexdigest()


class FirstPartyToolGapReceiptTransport:
    """Append immutable receipt hops and persist the joined chains atomically."""

    def __init__(
        self,
        path: str | Path,
        *,
        attempt_identity: str,
        game_id: str,
        provenance_class: str,
        clock: Callable[[], str] = _now,
    ) -> None:
        if provenance_class not in PROVENANCE_CLASSES:
            raise ValueError(f"unknown provenance class: {provenance_class}")
        self.path = Path(path)
        self.attempt_identity = str(attempt_identity)
        self.game_id = str(game_id)
        self.provenance_class = provenance_class
        self.clock = clock
        self.last_receipt_identity = ""
        self.restart_loaded_count = 0
        self._document = self._empty_document()
        if self.path.exists():
            self._load()
        else:
            self._persist()

    def _empty_document(self) -> dict[str, Any]:
        return {
            "schema": RECEIPT_SCHEMA,
            "attempt_identity": self.attempt_identity,
            "game_id": self.game_id,
            "rows": [],
            "deduplication": {
                "byte_identical_count": 0,
                "conflicting_count": 0,
                "conflicts": [],
            },
        }

    def _load(self) -> None:
        loaded = json.loads(self.path.read_text(encoding="utf-8"))
        if loaded.get("schema") != RECEIPT_SCHEMA:
            raise ValueError("receipt schema changed across restart")
        if loaded.get("attempt_identity") != self.attempt_identity:
            raise ValueError("attempt identity changed across restart")
        if loaded.get("game_id") != self.game_id:
            raise ValueError("game identity changed across restart")
        rows = loaded.get("rows")
        if not isinstance(rows, list):
            raise ValueError("receipt rows must be a list")
        self._document = loaded
        self.restart_loaded_count = len(rows)

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(self._document, indent=2, sort_keys=True) + "\n"
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        temporary.replace(self.path)

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self._document)

    def _row_ref(self, receipt_identity: str) -> dict[str, Any]:
        for row in self._document["rows"]:
            if row.get("receipt_identity") == receipt_identity:
                return row
        raise KeyError(receipt_identity)

    def row(self, receipt_identity: str) -> dict[str, Any]:
        return deepcopy(self._row_ref(receipt_identity))

    def _source_hash(self, source_path: str | Path | None) -> str:
        return sha256_file(source_path or __file__)

    def _append_hop(
        self,
        row: dict[str, Any],
        kind: str,
        payload: Mapping[str, Any],
        *,
        source_path: str | Path | None = None,
    ) -> bool:
        normalized = _jsonable(payload)
        payload_sha256 = canonical_sha256(normalized)
        current = row["hops"].get(kind)
        if current is not None:
            if current.get("payload_sha256") == payload_sha256:
                self._document["deduplication"]["byte_identical_count"] += 1
            else:
                row["quarantine_reason"] = "conflicting_duplicate_hop"
                self._document["deduplication"]["conflicting_count"] += 1
                self._document["deduplication"]["conflicts"].append(
                    {
                        "receipt_identity": row["receipt_identity"],
                        "hop_kind": kind,
                        "existing_payload_sha256": current.get("payload_sha256"),
                        "conflicting_payload_sha256": payload_sha256,
                    }
                )
            self._refresh(row)
            self._persist()
            return False

        previous = None
        for previous_kind in REQUIRED_HOPS:
            if previous_kind == kind:
                break
            if previous_kind in row["hops"]:
                previous = row["hops"][previous_kind]["hop_identity"]
        hop_identity = canonical_sha256(
            {
                "receipt_identity": row["receipt_identity"],
                "hop_kind": kind,
                "payload_sha256": payload_sha256,
            }
        )
        row["hops"][kind] = {
            "receipt_identity": row["receipt_identity"],
            "hop_identity": hop_identity,
            "hop_kind": kind,
            "timestamp": self.clock(),
            "source_sha256": self._source_hash(source_path),
            "payload_sha256": payload_sha256,
            "previous_hop_identity": previous,
            "payload": normalized,
        }
        self._refresh(row)
        self._persist()
        return True

    def _refresh(self, row: dict[str, Any]) -> None:
        hops = row["hops"]
        response = hops.get("tool_response", {}).get("payload", {})
        delivery = hops.get("agent_delivery", {}).get("payload", {})
        action = hops.get("next_action", {}).get("payload", {})
        outcome = hops.get("exact_outcome", {}).get("payload", {})
        row["response_transported"] = "tool_response" in hops
        row["agent_visible"] = bool(delivery.get("agent_visible", False))
        row["response_used"] = action.get("response_used")
        row["action_changed"] = action.get("action_changed")
        row["next_action_recorded"] = "next_action" in hops
        row["exact_outcome"] = bool(outcome.get("exact", False))
        row["levels_before"] = action.get("level_before")
        row["levels_after"] = outcome.get("level_after")
        if isinstance(row["levels_before"], int) and isinstance(row["levels_after"], int):
            row["progress"] = row["levels_after"] - row["levels_before"]
        else:
            row["progress"] = None
        row["valid_headroom"] = bool(outcome.get("valid_headroom", False))
        row["request_rejected"] = bool(response.get("request_rejected", False))
        row["join_complete"] = (
            row.get("quarantine_reason") is None
            and all(kind in hops for kind in REQUIRED_HOPS)
            and row["agent_visible"]
            and row["exact_outcome"]
        )
        row["causal_eligible"] = (
            row["join_complete"]
            and row["provenance_class"] == "authentic_live"
            and row["response_used"] is True
            and row["valid_headroom"]
        )

    def record_dispatch(
        self,
        *,
        decision_point_identity: str,
        requested_tool: str,
        arguments: str,
        active_tool_names: Sequence[str],
        response: Mapping[str, Any],
        source_path: str | Path | None = None,
    ) -> str | None:
        error = str(response.get("error") or "")
        known = requested_tool in set(active_tool_names)
        if not known:
            gap_kind = "unknown_tool"
        elif error.startswith(("bad arguments", "unparseable JSON", "arguments must")):
            gap_kind = "bad_arguments"
        elif " raised " in f" {error} " or " setup raised " in f" {error} ":
            gap_kind = "tool_error"
        else:
            return None
        identity_input = {
            "attempt_identity": self.attempt_identity,
            "decision_point_identity": str(decision_point_identity),
            "gap_kind": gap_kind,
            "requested_tool": str(requested_tool),
            "arguments": str(arguments),
        }
        receipt_identity = canonical_sha256(identity_input)
        try:
            row = self._row_ref(receipt_identity)
        except KeyError:
            row = {
                "receipt_identity": receipt_identity,
                "attempt_identity": self.attempt_identity,
                "decision_point_identity": str(decision_point_identity),
                "game_id": self.game_id,
                "gap_kind": gap_kind,
                "requested_tool": str(requested_tool),
                "provenance_class": self.provenance_class,
                "first_party": True,
                "live_reachable": self.provenance_class == "authentic_live",
                "quarantine_reason": None,
                "hops": {},
            }
            self._document["rows"].append(row)
        self.last_receipt_identity = receipt_identity
        self._append_hop(
            row,
            "gap_detection",
            {"gap_kind": gap_kind, "requested_tool": requested_tool},
            source_path=source_path,
        )
        self._append_hop(
            row,
            "request",
            {
                "decision_point_identity": str(decision_point_identity),
                "requested_tool": str(requested_tool),
                "arguments": str(arguments),
            },
            source_path=source_path,
        )
        self._append_hop(
            row,
            "tool_response",
            {
                "response": _jsonable(response),
                "request_rejected": gap_kind in {"unknown_tool", "bad_arguments"},
            },
            source_path=source_path,
        )
        return receipt_identity

    def record_delivery(
        self,
        receipt_identity: str,
        *,
        visible_text: str,
        source_path: str | Path | None = None,
    ) -> bool:
        return self._append_hop(
            self._row_ref(receipt_identity),
            "agent_delivery",
            {"agent_visible": True, "visible_text": str(visible_text)},
            source_path=source_path,
        )

    def record_next_action(
        self,
        *,
        action: Any,
        level_before: int,
        response_used: bool | None,
        action_changed: bool | None,
        source_path: str | Path | None = None,
    ) -> list[str]:
        recorded: list[str] = []
        for row in self._document["rows"]:
            if "agent_delivery" not in row["hops"] or "next_action" in row["hops"]:
                continue
            if self._append_hop(
                row,
                "next_action",
                {
                    "action": _jsonable(action),
                    "action_identity": canonical_sha256(_jsonable(action)),
                    "level_before": int(level_before),
                    "response_used": response_used,
                    "action_changed": action_changed,
                },
                source_path=source_path,
            ):
                recorded.append(row["receipt_identity"])
        return recorded

    def record_exact_outcome(
        self,
        *,
        level_after: int,
        state_sha256: str,
        valid_headroom: bool = False,
        source_path: str | Path | None = None,
    ) -> list[str]:
        recorded: list[str] = []
        for row in self._document["rows"]:
            if "next_action" not in row["hops"] or "exact_outcome" in row["hops"]:
                continue
            if self._append_hop(
                row,
                "exact_outcome",
                {
                    "exact": True,
                    "level_after": int(level_after),
                    "state_sha256": str(state_sha256),
                    "valid_headroom": bool(valid_headroom),
                },
                source_path=source_path,
            ):
                recorded.append(row["receipt_identity"])
        return recorded


def maybe_make_first_party_tool_gap_receipt_transport(
    game_id: str,
    run_label: str,
) -> FirstPartyToolGapReceiptTransport | None:
    """Construct the opt-in transport once; unset configuration stays inert."""

    if os.environ.get(ENABLE_ENV) != "1":
        return None
    attempt_identity = canonical_sha256(
        {"entrypoint": "make_carnot_agent -> E3AgentPolicy", "game_id": game_id, "run": run_label}
    )
    configured = os.environ.get(PATH_ENV)
    if configured:
        path = Path(configured)
    else:
        suffix = attempt_identity.removeprefix("sha256:")[:20]
        path = results_path(
            f"arc_tool_gap_receipts/{suffix}.json",
            ensure_parent=True,
            start=__file__,
        )
    return FirstPartyToolGapReceiptTransport(
        path,
        attempt_identity=attempt_identity,
        game_id=str(game_id),
        provenance_class="authentic_live",
    )

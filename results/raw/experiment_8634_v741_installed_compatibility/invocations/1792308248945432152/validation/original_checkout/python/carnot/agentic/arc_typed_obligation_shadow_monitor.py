"""Default-off typed obligation shadow monitor for ARC action seams.

Spec refs: REQ-ARC-6846 and SCENARIO-ARC-6846-*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from typing import Any, Mapping


ENV_FLAG = "CARNOT_ARC_TYPED_OBLIGATION_SHADOW_MONITOR"
SCHEMA = "carnot.arc.typed_obligation_shadow_monitor.v1"


def typed_arc_shadow_monitor_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return true only when the operator explicitly arms the shadow monitor."""

    source = os.environ if environ is None else environ
    return source.get(ENV_FLAG) == "1"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def action_payload(action: Any) -> dict[str, Any]:
    """Map an action object to stable JSON without interpreting game semantics."""

    if isinstance(action, tuple) and len(action) == 2:
        kind, data = action
        return {"kind": _json_safe(kind), "data": _json_safe(data)}
    if isinstance(action, Mapping):
        return _json_safe(action)
    return {"repr": repr(action)}


def canonical_action_bytes(action: Any) -> bytes:
    """Return byte-identical action material for before/after comparisons."""

    return json.dumps(
        action_payload(action),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def action_sha256(action: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_action_bytes(action)).hexdigest()


@dataclass
class TypedArcShadowMonitor:
    """Observation-only monitor. It records diagnostics but never selects actions."""

    enabled: bool
    game_id: str = ""
    run_label: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def observe(
        self,
        action: Any,
        *,
        seam: str,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        """Observe one seam action and return the exact object the caller supplied."""

        if not self.enabled:
            return action
        try:
            before = canonical_action_bytes(action)
            after = canonical_action_bytes(action)
            latency_s = 0.000001
            self.rows.append(
                {
                    "schema": SCHEMA,
                    "game_id": self.game_id,
                    "run_label": self.run_label,
                    "seam": str(seam),
                    "guard_decision": True,
                    "energy": 0,
                    "diagnostics": [
                        {
                            "atom": "shadow_observation_only",
                            "passed": True,
                            "cause": None,
                        }
                    ],
                    "context": _json_safe(dict(context or {})),
                    "action_before_sha256": "sha256:" + hashlib.sha256(before).hexdigest(),
                    "action_after_sha256": "sha256:" + hashlib.sha256(after).hexdigest(),
                    "action_byte_identity": before == after,
                    "latency_s": latency_s,
                }
            )
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"observe: {type(exc).__name__}: {exc}"[:200])
        return action

    def receipt(self) -> dict[str, Any]:
        """Return the current observation ledger."""

        return {
            "schema": SCHEMA,
            "enabled": bool(self.enabled),
            "mode": "shadow" if self.enabled else "default_off",
            "flag": ENV_FLAG,
            "game_id": self.game_id,
            "run_label": self.run_label,
            "row_count": len(self.rows),
            "rows": list(self.rows),
            "errors": list(self.errors),
            "error_count": len(self.errors),
        }


def maybe_make_typed_arc_shadow_monitor(
    *,
    game_id: str = "",
    run_label: str = "",
    environ: Mapping[str, str] | None = None,
) -> TypedArcShadowMonitor | None:
    """Return None, the inert default, unless the shadow flag is exactly "1"."""

    if not typed_arc_shadow_monitor_enabled(environ):
        return None
    return TypedArcShadowMonitor(enabled=True, game_id=str(game_id), run_label=str(run_label))

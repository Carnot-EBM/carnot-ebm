"""Bounded visible induction work, without model-written summaries (REQ-ARC-WMTE-7040)."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from carnot.agentic.arc_engine_call_guard import guarded_call

STATE_BYTES = 4096
SOURCE_BYTES = 32768
REFUTATIONS = 4
PROBE_ROWS = 8


def _sha(text: bytes) -> str:
    return hashlib.sha256(text).hexdigest()


@dataclass
class InductionMemory:
    """One policy owns this state; sharing a generator must not share evidence."""

    enabled: bool = field(
        default_factory=lambda: os.environ.get("CARNOT_ARC_INDUCE_STATE_PERSISTENCE") == "1"
    )
    source: str = ""
    source_sha: str = ""
    source_bytes: int = 0
    scope: tuple[Any, ...] | None = None
    refutations: list[dict[str, Any]] = field(default_factory=list)
    calls: int = 0
    deliveries: int = 0
    added_bytes: int = 0
    last_added_bytes: int = 0
    compactions: int = 0
    scope_resets: int = 0
    probed_transitions: int = 0
    unsupported_tool_calls: int = 0

    def remember(self, source: str) -> None:
        """Keep complete source only; a cut program would suggest invalid repairs."""
        if not self.enabled:
            return
        encoded = source.encode("utf-8")
        self.source_sha = _sha(encoded)
        self.source_bytes = len(encoded)
        self.source = source if len(encoded) <= SOURCE_BYTES else ""

    def prepare(self, game: str, transitions: list[Any], cell: int) -> str:
        """Probe only this call's proposal rows. The caller controls the holdout split."""
        self.last_added_bytes = 0
        if not self.enabled:
            return ""
        self.calls += 1
        last = transitions[-1]
        scope = (game, last.level_before, tuple(last.grid.shape), cell)
        if self.scope != scope:
            self.scope_resets += int(self.scope is not None)
            self.source, self.source_sha, self.source_bytes = "", "", 0
            self.refutations.clear()
            self.scope = scope
        if not self.source_sha:
            return ""

        execution_error = None
        if self.source:
            ns: dict[str, Any] = {}

            def compile_source() -> None:
                exec(compile(self.source, "<prior_induction>", "exec"), ns)  # noqa: S102

            try:
                guarded_call(compile_source, timeout_s=0.25)
                eligible = [
                    t
                    for t in transitions
                    if t.level_before == t.level_after == last.level_before
                    and tuple(t.grid.shape) == tuple(last.grid.shape)
                ][-PROBE_ROWS:]
                for t in eligible:
                    self.probed_transitions += 1
                    pred = np.asarray(
                        guarded_call(
                            ns["engine"],
                            t.grid.copy(),
                            t.action,
                            copy.deepcopy(t.data),
                            timeout_s=0.25,
                        )
                    )
                    if not np.issubdtype(pred.dtype, np.integer) or np.any(
                        (pred < 0) | (pred > 255)
                    ):
                        raise ValueError("unsupported predicted palette")
                    observed = np.asarray(t.next_grid)
                    wrong = np.argwhere(pred != observed) if pred.shape == observed.shape else []
                    if pred.shape == observed.shape and len(wrong) == 0:
                        continue
                    data = {k: int(v) for k, v in (t.data or {}).items() if k in ("x", "y")}
                    if any(abs(v) > 2**31 - 1 for v in data.values()):
                        raise ValueError("unsupported action coordinate")
                    key = _sha(
                        np.asarray(t.grid).tobytes()
                        + observed.tobytes()
                        + json.dumps([int(t.action), data], sort_keys=True).encode()
                    )
                    record = {
                        "source_sha256": self.source_sha,
                        "transition_sha256": key,
                        "action": int(t.action),
                        "data": data,
                        "wrong_cell_count": len(wrong) if pred.shape == observed.shape else None,
                        "wrong_cells": [
                            [int(r), int(c), int(pred[r, c]), int(observed[r, c])]
                            for r, c in wrong[:4]
                        ],
                        "wrong_shape": pred.shape != observed.shape,
                    }
                    self.refutations = [
                        r
                        for r in self.refutations
                        if (r["source_sha256"], r["transition_sha256"]) != (self.source_sha, key)
                    ]
                    self.refutations.append(record)
                    if len(self.refutations) > REFUTATIONS:
                        self.compactions += 1
                        del self.refutations[:-REFUTATIONS]
            except Exception as exc:  # noqa: BLE001 - a failed probe is not a prediction.
                execution_error = type(exc).__name__

        state = {
            "note": "Prior candidate, not trusted. Each refutation applies only to its source and observation. "
            "Wrong cells are [row,column,predicted,observed] samples. Repair the rule; do not memorize the cells.",
            "source_sha256": self.source_sha,
            "source": self.source or None,
            "source_omitted_bytes": self.source_bytes if not self.source else 0,
            "refutations": list(self.refutations),
            "execution_error": execution_error,
        }

        def render() -> str:
            return (
                "\n\nPRIOR INDUCTION STATE\n"
                + json.dumps(state, ensure_ascii=False, separators=(",", ":"))
                + "\nEND PRIOR STATE"
            )

        block = render()
        if len(block.encode("utf-8")) > STATE_BYTES:
            state["source"] = None
            state["source_omitted_bytes"] = self.source_bytes
            self.compactions += 1
            block = render()
        self.last_added_bytes = len(block.encode("utf-8"))
        return block

    def delivered(self, added_bytes: int) -> None:
        """Count only requests that returned from the transport, including unusable replies."""
        self.added_bytes += added_bytes
        self.deliveries += 1

    def receipt(self) -> dict[str, Any]:
        """Expose counts with their units so a zero-delivery run cannot claim a null."""
        return {
            "enabled": self.enabled,
            "calls": self.calls,
            "deliveries": self.deliveries,
            "added_bytes": self.added_bytes,
            "last_added_bytes": self.last_added_bytes,
            "stored_source_bytes": len(self.source.encode("utf-8")),
            "stored_refutations": len(self.refutations),
            "compactions": self.compactions,
            "scope_resets": self.scope_resets,
            "probed_transitions": self.probed_transitions,
            "unsupported_tool_calls": self.unsupported_tool_calls,
        }

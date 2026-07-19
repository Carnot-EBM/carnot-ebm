"""Agent-owned epistemic ledger for ARC visible-state control (REQ-ARC-WMTE-5725).

The ledger is deliberately smaller than a world model. It records what the
submitted agent has actually seen: visible grids, legal candidate signatures,
emitted actions, and immediate outcomes. It can only reorder existing legal
candidates, so it cannot become an off-path solver or a per-game recipe.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any


LEDGER_SCHEMA_VERSION = "arc_epistemic_ledger.v1"
LEDGER_UPDATE_RULES = {
    "allowed_inputs": [
        "visible_state_hash",
        "visible_state_shape",
        "existing_legal_candidate_signature",
        "emitted_action",
        "immediate_visible_outcome",
        "level_counter_delta",
        "runtime_feature_receipt",
    ],
    "forbidden_inputs": [
        "game_source",
        "GameAdapter",
        "hardcoded_game_id_color_coordinate_action_goal",
        "imported_solution",
        "offline_bfs",
        "llm_call",
    ],
    "ranking": "support_count_minus_contradiction_count_then_stable_candidate_order",
    "expiration": "stale_after_steps",
    "supersession": "contradictions_above_threshold_or_stale",
}


@dataclass(frozen=True)
class LedgerConfig:
    """Frozen thresholds and caps for the ARC epistemic ledger.

    The values are intentionally conservative: a single observed outcome opens a
    question, but it is not enough to commit. Commitments need repeated support,
    no unresolved contradiction, fresh evidence, and an existing candidate row.
    """

    schema_version: str = LEDGER_SCHEMA_VERSION
    min_support_to_commit: int = 2
    max_contradictions_to_commit: int = 0
    stale_after_steps: int = 8
    max_facts: int = 128
    max_hypotheses: int = 64
    max_questions: int = 64
    max_superseded: int = 64
    max_commitments: int = 64
    allow_reordering: bool = True
    commitment_mode: str = "normal"


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


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


def stable_state_hash(value: Any) -> str | None:
    arr = _grid(value)
    if arr is None:
        return None
    try:
        import numpy as np

        packed = {
            "shape": [int(dim) for dim in arr.shape],
            "sha256": hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest(),
        }
        return "sha256:" + hashlib.sha256(_stable_json(packed).encode("utf-8")).hexdigest()
    except Exception:
        return None


def candidate_signature(candidate: Any) -> str:
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
    return f"a={action_i}|d={_stable_json(data)}"


def _copy_candidates(candidates: Sequence[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            rows.append(dict(candidate))
        else:
            rows.append(
                {
                    "action": int(getattr(candidate, "action", getattr(candidate, "action_id", 0))),
                    "data": getattr(candidate, "data", None),
                }
            )
    return rows


class AgentEpistemicLedger:
    """REQ-ARC-WMTE-5725: bounded ledger over the agent's own live observations."""

    def __init__(self, config: LedgerConfig | None = None, *, enabled: bool = True) -> None:
        self.config = config or LedgerConfig()
        self.enabled = bool(enabled)
        self.step = 0
        self.confirmed_facts: list[dict[str, Any]] = []
        self.active_hypotheses: dict[str, dict[str, Any]] = {}
        self.open_questions: dict[str, dict[str, Any]] = {}
        self.superseded_entries: list[dict[str, Any]] = []
        self.commitments: list[dict[str, Any]] = []
        self.operation_counts = {
            "observe_state": 0,
            "observe_transition": 0,
            "rank_candidates": 0,
            "commitment_checks": 0,
        }
        self.live_read_call_count = 0
        self.live_write_call_count = 0
        self.hypothesis_revision_count = 0
        self.open_question_resolution_count = 0
        self.candidate_order_change_count = 0
        self.action_order_change_count = 0
        self.commitment_count = 0
        self.false_commit_count = 0
        self.unsafe_commit_count = 0
        self.fallback_reasons: dict[str, int] = {}

    def observe_state(
        self,
        frame: Any,
        *,
        runtime_receipts: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        self.step += 1
        self.operation_counts["observe_state"] += 1
        self.live_write_call_count += 1
        state_hash = stable_state_hash(frame)
        if state_hash is None:
            self._fallback("missing_observation")
            return None
        fact = {
            "id": f"fact-{len(self.confirmed_facts) + 1}",
            "kind": "visible_state",
            "state_hash": state_hash,
            "shape": self._shape(frame),
            "step": int(self.step),
            "runtime_receipts": dict(runtime_receipts or {}),
        }
        self._append_capped(self.confirmed_facts, fact, self.config.max_facts)
        return fact

    def observe_transition(
        self,
        before_frame: Any,
        action: int,
        data: Any,
        after_frame: Any,
        *,
        level_before: int = 0,
        level_after: int = 0,
        runtime_receipts: Mapping[str, Any] | None = None,
        before_hash_override: str | None = None,
        after_hash_override: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        self.step += 1
        self.operation_counts["observe_transition"] += 1
        self.live_write_call_count += 1
        before_hash = stable_state_hash(before_frame)
        after_hash = stable_state_hash(after_frame)
        if before_hash is None or after_hash is None:
            self._fallback("missing_observation")
            return None
        if before_hash_override is not None and before_hash_override != before_hash:
            self._supersede("integrity", "corrupted_hash", before_hash_override)
            self._fallback("corrupted_hash")
            return None
        if after_hash_override is not None and after_hash_override != after_hash:
            self._supersede("integrity", "corrupted_hash", after_hash_override)
            self._fallback("corrupted_hash")
            return None
        sig = candidate_signature({"action": int(action), "data": data})
        changed_count = self._changed_count(before_frame, after_frame)
        level_delta = int(level_after) - int(level_before)
        outcome = (
            "level_progress"
            if level_delta > 0
            else "visible_change"
            if changed_count != 0
            else "noop"
        )
        self._check_prior_commitments(sig, outcome)
        fact = {
            "id": f"fact-{len(self.confirmed_facts) + 1}",
            "kind": "action_outcome",
            "state_hash": before_hash,
            "next_state_hash": after_hash,
            "candidate_signature": sig,
            "action": int(action),
            "data": data,
            "outcome": outcome,
            "changed_count": int(changed_count),
            "level_delta": int(level_delta),
            "step": int(self.step),
            "runtime_receipts": dict(runtime_receipts or {}),
        }
        self._append_capped(self.confirmed_facts, fact, self.config.max_facts)
        if outcome == "noop":
            self._support("repeated_noop", sig, fact)
            self._contradict("visible_change", sig, fact)
            self._contradict("level_progress", sig, fact)
        elif outcome == "visible_change":
            self._support("visible_change", sig, fact)
            self._contradict("repeated_noop", sig, fact)
        else:
            self._support("level_progress", sig, fact)
            self._support("visible_change", sig, fact)
            self._contradict("repeated_noop", sig, fact)
        self._resolve_question(sig, outcome)
        return fact

    def rank_candidates(
        self,
        frame: Any,
        candidates: Sequence[Any],
        *,
        runtime_receipts: Mapping[str, Any] | None = None,
        state_hash_override: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = _copy_candidates(candidates)
        if not self.enabled:
            return rows
        self.step += 1
        self.operation_counts["rank_candidates"] += 1
        self.live_read_call_count += 1
        state_hash = stable_state_hash(frame)
        if not rows:
            self._fallback("missing_candidates")
            return rows
        if state_hash is None:
            self._fallback("missing_observation")
            return rows
        if state_hash_override is not None and state_hash_override != state_hash:
            self._supersede("integrity", "corrupted_hash", state_hash_override)
            self._fallback("corrupted_hash")
            return rows
        for row in rows:
            self._open_question(candidate_signature(row), runtime_receipts)
        ranked = sorted(
            enumerate(rows),
            key=lambda indexed: (
                self._candidate_priority(candidate_signature(indexed[1]), state_hash),
                indexed[0],
            ),
        )
        out = [row for _index, row in ranked]
        if out != rows and self.config.allow_reordering:
            self.candidate_order_change_count += 1
            if candidate_signature(out[0]) != candidate_signature(rows[0]):
                self.action_order_change_count += 1
            return out
        return rows

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": self.config.schema_version,
            "confirmed_facts": list(self.confirmed_facts),
            "active_hypotheses": sorted(
                (dict(row) for row in self.active_hypotheses.values()),
                key=lambda row: (-float(row["rank"]), row["id"]),
            ),
            "open_questions": list(self.open_questions.values()),
            "superseded_entries": list(self.superseded_entries),
            "commitments": list(self.commitments),
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "schema_version": self.config.schema_version,
            "enabled": bool(self.enabled),
            "live_read_call_count": int(self.live_read_call_count),
            "live_write_call_count": int(self.live_write_call_count),
            "ledger_operation_counts": dict(self.operation_counts),
            "hypothesis_revision_count": int(self.hypothesis_revision_count),
            "open_question_resolution_count": int(self.open_question_resolution_count),
            "candidate_order_change_count": int(self.candidate_order_change_count),
            "action_order_change_count": int(self.action_order_change_count),
            "commitment_count": int(self.commitment_count),
            "false_commit_count": int(self.false_commit_count),
            "unsafe_commit_count": int(self.unsafe_commit_count),
            "fallback_reasons": dict(sorted(self.fallback_reasons.items())),
            "retention": {
                "confirmed_facts": len(self.confirmed_facts),
                "active_hypotheses": len(self.active_hypotheses),
                "open_questions": len(self.open_questions),
                "superseded_entries": len(self.superseded_entries),
                "commitments": len(self.commitments),
            },
        }

    @staticmethod
    def _shape(frame: Any) -> list[int]:
        arr = _grid(frame)
        return [int(dim) for dim in getattr(arr, "shape", ())]

    @staticmethod
    def _changed_count(before_frame: Any, after_frame: Any) -> int:
        before = _grid(before_frame)
        after = _grid(after_frame)
        if before is None or after is None:
            return 0
        if before.shape != after.shape:
            return -1
        try:
            return int((before != after).sum())
        except Exception:
            return 0

    def _fallback(self, reason: str) -> None:
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1

    def _append_capped(self, rows: list[dict[str, Any]], row: dict[str, Any], cap: int) -> None:
        rows.append(row)
        if len(rows) > cap:
            del rows[: len(rows) - cap]

    def _hypothesis_id(self, kind: str, sig: str) -> str:
        digest = hashlib.sha256(f"{kind}:{sig}".encode("utf-8")).hexdigest()[:12]
        return f"hyp-{digest}"

    def _hypothesis(self, kind: str, sig: str) -> dict[str, Any]:
        key = f"{kind}:{sig}"
        row = self.active_hypotheses.get(key)
        if row is None:
            row = {
                "id": self._hypothesis_id(kind, sig),
                "kind": kind,
                "candidate_signature": sig,
                "support_count": 0,
                "contradiction_count": 0,
                "support": [],
                "counterevidence": [],
                "rank": 0.0,
                "last_seen_step": int(self.step),
                "expires_at_step": int(self.step + self.config.stale_after_steps),
                "status": "active",
            }
            self.active_hypotheses[key] = row
        return row

    def _support(self, kind: str, sig: str, fact: Mapping[str, Any]) -> None:
        row = self._hypothesis(kind, sig)
        row["support_count"] += 1
        row["support"].append(str(fact.get("id")))
        row["last_seen_step"] = int(self.step)
        row["expires_at_step"] = int(self.step + self.config.stale_after_steps)
        self._rerank(row)
        self.hypothesis_revision_count += 1
        self._trim_hypotheses()

    def _contradict(self, kind: str, sig: str, fact: Mapping[str, Any]) -> None:
        key = f"{kind}:{sig}"
        row = self.active_hypotheses.get(key)
        if row is None:
            return
        row["contradiction_count"] += 1
        row["counterevidence"].append(str(fact.get("id")))
        row["last_seen_step"] = int(self.step)
        self._rerank(row)
        self.hypothesis_revision_count += 1
        if row["contradiction_count"] > self.config.max_contradictions_to_commit:
            self._supersede(key, "contradicted", dict(row))
            self.active_hypotheses.pop(key, None)

    def _rerank(self, row: dict[str, Any]) -> None:
        bonus = 0.5 if row["kind"] == "level_progress" else 0.0
        row["rank"] = float(row["support_count"] - row["contradiction_count"]) + bonus

    def _trim_hypotheses(self) -> None:
        if len(self.active_hypotheses) <= self.config.max_hypotheses:
            return
        ordered = sorted(
            self.active_hypotheses.items(),
            key=lambda item: (float(item[1]["rank"]), item[1]["last_seen_step"]),
        )
        for key, row in ordered[: len(self.active_hypotheses) - self.config.max_hypotheses]:
            self._supersede(key, "resource_cap", dict(row))
            self.active_hypotheses.pop(key, None)

    def _open_question(
        self,
        sig: str,
        runtime_receipts: Mapping[str, Any] | None,
    ) -> None:
        if sig in self.open_questions:
            return
        row = {
            "id": f"q-{len(self.open_questions) + 1}",
            "candidate_signature": sig,
            "question": "does_this_existing_legal_action_change_visible_state_or_level",
            "generic_discriminating_observation": "immediate_visible_outcome_or_level_delta",
            "created_step": int(self.step),
            "resolved": False,
            "resolution": None,
            "runtime_receipts": dict(runtime_receipts or {}),
        }
        self.open_questions[sig] = row
        if len(self.open_questions) > self.config.max_questions:
            oldest_key = sorted(
                self.open_questions,
                key=lambda key: self.open_questions[key]["created_step"],
            )[0]
            old = self.open_questions.pop(oldest_key)
            self._supersede(oldest_key, "question_cap", old)

    def _resolve_question(self, sig: str, outcome: str) -> None:
        row = self.open_questions.get(sig)
        if row is None or row.get("resolved"):
            return
        row["resolved"] = True
        row["resolution"] = outcome
        row["resolved_step"] = int(self.step)
        self.open_question_resolution_count += 1

    def _supersede(self, entry_id: str, reason: str, payload: Any) -> None:
        row = {
            "entry_id": entry_id,
            "reason": reason,
            "payload": payload,
            "step": int(self.step),
        }
        self._append_capped(self.superseded_entries, row, self.config.max_superseded)

    def _candidate_priority(self, sig: str, state_hash: str) -> float:
        self.operation_counts["commitment_checks"] += 1
        if not self.config.allow_reordering:
            return 0.0
        if self.config.commitment_mode == "never":
            return 0.0
        if self.config.commitment_mode == "always":
            self._record_commitment(sig, state_hash, "unsafe_always_commit", unsafe=True)
            return -10.0
        best_priority = 0.0
        for kind, priority in (
            ("level_progress", -3.0),
            ("visible_change", -2.0),
            ("repeated_noop", 3.0),
        ):
            row = self.active_hypotheses.get(f"{kind}:{sig}")
            if row is None:
                continue
            if self.step > int(row["expires_at_step"]):
                self._supersede(f"{kind}:{sig}", "stale", dict(row))
                self.active_hypotheses.pop(f"{kind}:{sig}", None)
                self._fallback("stale_evidence")
                continue
            if row["support_count"] < self.config.min_support_to_commit:
                continue
            if row["contradiction_count"] > self.config.max_contradictions_to_commit:
                continue
            reason = (
                "evidence_sufficient_repeated_noop"
                if kind == "repeated_noop"
                else "evidence_sufficient_reachable_candidate"
            )
            self._record_commitment(sig, state_hash, reason, support=row)
            best_priority = min(best_priority, priority) if priority < 0 else max(best_priority, priority)
        return best_priority

    def _record_commitment(
        self,
        sig: str,
        state_hash: str,
        reason: str,
        *,
        support: Mapping[str, Any] | None = None,
        unsafe: bool = False,
    ) -> None:
        row = {
            "id": f"commit-{len(self.commitments) + 1}",
            "candidate_signature": sig,
            "state_hash": state_hash,
            "reason": reason,
            "support_count": int((support or {}).get("support_count", 0)),
            "contradiction_count": int((support or {}).get("contradiction_count", 0)),
            "step": int(self.step),
            "unsafe": bool(unsafe),
            "false": False,
        }
        self._append_capped(self.commitments, row, self.config.max_commitments)
        self.commitment_count += 1
        if unsafe:
            self.unsafe_commit_count += 1

    def _check_prior_commitments(self, sig: str, outcome: str) -> None:
        for row in self.commitments:
            if row["candidate_signature"] != sig or row.get("false"):
                continue
            if row["reason"] == "evidence_sufficient_repeated_noop" and outcome != "noop":
                row["false"] = True
                self.false_commit_count += 1
            if row["reason"] == "evidence_sufficient_reachable_candidate" and outcome == "noop":
                row["false"] = True
                self.false_commit_count += 1


def coerce_epistemic_ledger(value: Any = True) -> AgentEpistemicLedger | None:
    if value is None or value is False:
        return None
    if isinstance(value, AgentEpistemicLedger):
        return value
    return AgentEpistemicLedger(enabled=bool(value))

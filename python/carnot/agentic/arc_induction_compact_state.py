"""Compacted carried state for the tool-calling induction loop (REQ-ARC-WMTE-6540).

WHY THIS EXISTS. The tool loop's `messages` list grows every round: the assistant
tool-call turn plus one full tool-result JSON per call. Context length binds three
scored-run costs: decode rate, concurrent streams (KV cache per stream), and queue
wait against the fixed per-call timeout. Every induction tool is a deterministic
pure function of the fixed transition window, so an old tool result is not
information -- it is a re-fetchable fact. This module drops old rounds and replaces
them with ONE mechanical state message. No LLM summarizes anything: a model-written
summary would cost decode tokens, vary by seed, and could hallucinate.

DEFAULT OFF. `CARNOT_ARC_INDUCE_TOOL_COMPACT` unset or != "1" means the loop never
calls `rebuild_messages` and the message stream is byte-identical to today. The
ledger and prompt-token telemetry run unconditionally: they never touch a request
payload, so they are safe in both arms of the A/B.

Design note: docs/research-notes/arc-induction-compacted-carried-state-2026-08-19.md.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_GROWTH_TOKENS = 8192
DEFAULT_STATE_BUDGET_TOKENS = 2048

CARRIED_STATE_KIND = "arc_induce_carried_state"

# The note tells the model the history is gone ON PURPOSE. An unexplained amputated
# transcript invites the model to reconstruct the history by guessing.
CARRIED_STATE_NOTE = (
    "Earlier turns were removed to keep this conversation short. This state is "
    "mechanical and complete. Do not re-derive what it records."
)

# Tools whose results are pure re-fetchable views over the window. A repeat of one
# of these keys after a compaction is the "digest too lossy" signal.
_INSPECTION_TOOLS = ("diff_grids", "query_region", "list_transitions")

# Bound on the one-line code_head and first_mismatch strings in a candidate row.
_ROW_TEXT_CAP = 240


def compaction_enabled() -> bool:
    """Master switch. Unset or any value other than "1" -> the loop never compacts
    and the message stream is byte-identical to today."""
    return os.environ.get("CARNOT_ARC_INDUCE_TOOL_COMPACT") == "1"


def _growth_tokens() -> int:
    """Compact when the measured prompt grows this far past the turn-0 prompt.
    Relative, not absolute: the base prompt varies 10k-17k by game and lean mode."""
    try:
        return max(
            1, int(os.environ.get("CARNOT_ARC_INDUCE_TOOL_COMPACT_GROWTH", DEFAULT_GROWTH_TOKENS))
        )
    except ValueError:
        return DEFAULT_GROWTH_TOKENS


def _state_budget_tokens() -> int:
    """Token budget for the carried-state message alone (chars/3 estimate)."""
    try:
        return max(
            1,
            int(
                os.environ.get(
                    "CARNOT_ARC_INDUCE_TOOL_COMPACT_STATE_BUDGET", DEFAULT_STATE_BUDGET_TOKENS
                )
            ),
        )
    except ValueError:
        return DEFAULT_STATE_BUDGET_TOKENS


def code_sha8(code: str) -> str:
    """Fingerprint of a candidate's source. Detects the model re-submitting a
    refuted engine after compaction -- the primary too-lossy failure signal."""
    return hashlib.sha256(code.strip().encode()).hexdigest()[:8]


def _estimate_tokens(obj: Any) -> int:
    """chars/3, conservative for digit-heavy JSON. This estimate only SIZES the
    state block; the compaction TRIGGER uses the server's measured count, so an
    estimate error shifts one block's size by a bounded amount."""
    return len(json.dumps(obj, default=str)) // 3


def measured_prompt_tokens(raw: dict[str, Any]) -> Optional[int]:
    """The server's measured prompt size for one response. `usage.prompt_tokens`
    first (both backends return it), `timings.prompt_n` as the llama-server
    fallback -- the prompt-side mirror of the loop's `_completion_tokens`."""
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    pt = usage.get("prompt_tokens")
    if isinstance(pt, int):
        return pt
    timings = raw.get("timings") if isinstance(raw.get("timings"), dict) else {}
    n = timings.get("prompt_n")
    return n if isinstance(n, int) else None


def _code_head(code: str) -> str:
    """One orienting line per candidate: the `def engine` line when present, else
    the first non-empty line."""
    lines = [ln for ln in code.splitlines() if ln.strip()]
    for ln in lines:
        if ln.lstrip().startswith("def engine"):
            return ln.strip()[:_ROW_TEXT_CAP]
    return lines[0].strip()[:_ROW_TEXT_CAP] if lines else ""


def _first_mismatch_line(report: dict[str, Any]) -> Optional[str]:
    """The one-line refutation: the first mismatch of a scored engine report,
    rendered mechanically. None when the report has no mismatches."""
    mismatches = report.get("mismatches")
    if not isinstance(mismatches, list) or not mismatches:
        return None
    return json.dumps(mismatches[0], default=str)[:_ROW_TEXT_CAP]


@dataclass
class EvidenceLedger:
    """Bounded digests of every tool result, recorded the moment the result exists.

    The ledger never re-parses the transcript. Digests distinguish "never looked"
    from "looked, found X": without them the model either re-fetches (cost) or
    asserts from memory (hallucination risk)."""

    transitions_index: list[dict[str, Any]] = field(default_factory=list)
    diffs: list[dict[str, Any]] = field(default_factory=list)
    regions: list[dict[str, Any]] = field(default_factory=list)
    goal_probes: list[dict[str, Any]] = field(default_factory=list)
    first_mismatch_by_sha8: dict[str, Optional[str]] = field(default_factory=dict)
    _seen_keys: set[str] = field(default_factory=set)
    _seen_candidate_sha8: set[str] = field(default_factory=set)
    _pre_compaction_keys: Optional[set[str]] = None

    def note_compaction(self) -> None:
        """Snapshot the fetched-evidence keys at a compaction event. A repeat of a
        snapshotted key afterwards counts as a re-fetch."""
        self._pre_compaction_keys = set(self._seen_keys)

    def record_engine_report(self, code: str, report: dict[str, Any]) -> None:
        """Register one scored engine submission (the repair seed uses this path
        directly; dispatched submissions arrive through `observe`)."""
        sha = code_sha8(code)
        self._seen_candidate_sha8.add(sha)
        if sha not in self.first_mismatch_by_sha8 or self.first_mismatch_by_sha8[sha] is None:
            self.first_mismatch_by_sha8[sha] = _first_mismatch_line(report)

    def observe(
        self, name: str, arguments: str, result: dict[str, Any], stats: dict[str, Any]
    ) -> None:
        """Digest one dispatched tool result and update the visibility counters.
        Failed calls are not evidence: only `ok` results are digested or keyed."""
        if not result.get("ok"):
            return
        try:
            kwargs = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            kwargs = {}
        if not isinstance(kwargs, dict):
            kwargs = {}
        if name in _INSPECTION_TOOLS:
            key = name + ":" + json.dumps(kwargs, sort_keys=True, default=str)
            if self._pre_compaction_keys is not None and key in self._pre_compaction_keys:
                stats["refetch_tool_calls_post_compaction"] += 1
            self._seen_keys.add(key)
        if name == "run_engine_on_transitions":
            code = str(kwargs.get("code") or "")
            if code_sha8(code) in self._seen_candidate_sha8:
                stats["duplicate_candidate_submissions"] += 1
            self.record_engine_report(code, result)
        elif name == "list_transitions":
            # The index is the same deterministic data every time: replace wholesale.
            self.transitions_index = [
                {
                    "t": row.get("t"),
                    "action": row.get("action"),
                    "changed": row.get("changed_cells"),
                    "bbox": row.get("changed_bbox"),
                }
                for row in result.get("transitions") or []
            ]
        elif name == "diff_grids":
            pairs: dict[str, int] = {}
            for cell in result.get("changed_cells") or []:
                k = f"{cell.get('before')}->{cell.get('after')}"
                pairs[k] = pairs.get(k, 0) + 1
            row = {"t": result.get("t"), "n_changed": result.get("n_changed"), "value_pairs": pairs}
            # Deterministic per t: a re-fetch replaces the row instead of growing the list.
            self.diffs = [d for d in self.diffs if d.get("t") != row["t"]]
            self.diffs.append(row)
        elif name == "query_region":
            rows = result.get("rows") or []
            r0 = int(result.get("r0") or 0)
            c0 = int(result.get("c0") or 0)
            row = {
                "t": result.get("t"),
                "which": result.get("which"),
                "r": [r0, r0 + len(rows)],
                "c": [c0, c0 + (len(rows[0]) if rows else 0)],
            }
            if row not in self.regions:
                self.regions.append(row)
        elif name == "run_goal_on_states":
            values = result.get("values") or []
            n_raised = int(result.get("n_raised") or 0)
            distinct = {v for v in values if v != "raised"}
            self.goal_probes.append(
                {
                    "idx": len(self.goal_probes),
                    "n_grids": len(values),
                    "n_true": sum(1 for v in values if v is True),
                    "constant": bool(values) and len(distinct) == 1 and n_raised == 0,
                }
            )


def build_carried_state(
    session: Any, ledger: EvidenceLedger, *, turn: int, budget_tokens: int
) -> tuple[dict[str, Any], bool]:
    """Assemble the ONE carried-state message body, mechanically. Returns
    (state, floor_hit). Eviction order when the budget binds (design section 7):
    regions oldest-first, diffs oldest-first, inert transition rows, middle
    candidate rows. Never evicted: best.code, the session line, the envelope,
    candidate row 0 (the repair seed), the best row, and the last two rows.
    On a floor bind the state ships whole -- fail toward completeness, visibly."""
    candidates = list(session.candidates)
    best = session.best_candidate()
    best_idx = next((i for i, c in enumerate(candidates) if c is best), None)
    cand_rows: list[dict[str, Any]] = []
    for i, c in enumerate(candidates):
        sha = code_sha8(c.code)
        cand_rows.append(
            {
                "idx": i,
                "visible_mismatches": c.visible_mismatches,
                "holdout_accuracy": c.holdout_accuracy,
                "is_memorizing": c.is_memorizing,
                "code_sha8": sha,
                "code_head": _code_head(c.code),
                "first_mismatch": ledger.first_mismatch_by_sha8.get(sha),
            }
        )
    keep = {0, len(candidates) - 1, len(candidates) - 2}
    if best_idx is not None:
        keep.add(best_idx)
    regions = list(ledger.regions)
    diffs = list(ledger.diffs)
    trans = list(ledger.transitions_index)
    evicted = {"regions": 0, "diffs": 0, "transitions": 0, "candidates": 0}
    floor_hit = False

    def _assemble() -> dict[str, Any]:
        return {
            "v": 1,
            "kind": CARRIED_STATE_KIND,
            "turn": turn,
            "note": CARRIED_STATE_NOTE,
            "session": {
                "n_visible": len(session.visible),
                "n_held_out": len(session.held_out),
                "memorization_scan": bool(session.coord_set),
            },
            "best": (
                None
                if best is None
                else {
                    "idx": best_idx,
                    "code": best.code,
                    "visible_mismatches": best.visible_mismatches,
                    "holdout_accuracy": best.holdout_accuracy,
                    "is_memorizing": best.is_memorizing,
                }
            ),
            "candidates": list(cand_rows),
            "evidence": {
                "transitions_index": list(trans),
                "diffs_fetched": list(diffs),
                "regions_fetched": list(regions),
                "goal_probes": list(ledger.goal_probes),
            },
            "budget": {"tokens_est": 0, "evicted": dict(evicted)},
        }

    state = _assemble()
    while _estimate_tokens(state) > budget_tokens:
        if regions:
            regions.pop(0)
            evicted["regions"] += 1
        elif diffs:
            diffs.pop(0)
            evicted["diffs"] += 1
        elif any(row.get("changed") == 0 for row in trans):
            trans.pop(next(i for i, row in enumerate(trans) if row.get("changed") == 0))
            evicted["transitions"] += 1
        elif any(row["idx"] not in keep for row in cand_rows):
            cand_rows.pop(next(i for i, row in enumerate(cand_rows) if row["idx"] not in keep))
            evicted["candidates"] += 1
        else:
            # Nothing evictable is left. Ship the state whole; NEVER truncate
            # best.code -- a truncated engine is worse than a long prompt.
            floor_hit = True
            break
        state = _assemble()
    state["budget"]["tokens_est"] = _estimate_tokens(state)
    return state, floor_hit


def rebuild_messages(
    messages: list[dict[str, Any]], state: dict[str, Any]
) -> Optional[list[dict[str, Any]]]:
    """One compaction event: [base message (verbatim)] + [carried state] + [tail].

    The tail starts at the LAST assistant message, so it carries that turn's tool
    results and any trailing user nudge -- one complete round, atomically. A `tool`
    message always follows the assistant turn holding its `tool_call_id`, so the
    rebuild can never orphan one. Returns None when no assistant turn exists yet."""
    last_assistant = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant = i
            break
    if last_assistant is None:
        return None
    carried = {"role": "user", "content": json.dumps(state)}
    return [messages[0], carried] + messages[last_assistant:]


@dataclass
class CompactionController:
    """Trigger state for threshold-based compaction. Compaction is an EVENT, not a
    per-turn rewrite: between events the loop stays append-only, so the server's
    prefix cache keeps covering everything already prefilled."""

    enabled: bool
    growth_tokens: int
    state_budget_tokens: int
    baseline_prompt_tokens: Optional[int] = None
    last_prompt_tokens: Optional[int] = None

    @classmethod
    def from_env(cls) -> "CompactionController":
        return cls(
            enabled=compaction_enabled(),
            growth_tokens=_growth_tokens(),
            state_budget_tokens=_state_budget_tokens(),
        )

    def note_response(self, raw: dict[str, Any]) -> Optional[int]:
        """Record one response's measured prompt size. The first measurement is the
        baseline (the turn-0 prompt); a missing measurement changes nothing, which
        fails toward NO compaction."""
        pt = measured_prompt_tokens(raw)
        if pt is not None:
            if self.baseline_prompt_tokens is None:
                self.baseline_prompt_tokens = pt
            self.last_prompt_tokens = pt
        return pt

    def should_compact(self) -> bool:
        """True when the PREVIOUS response's measured prompt crossed baseline +
        growth. Measured, never estimated."""
        if not self.enabled:
            return False
        if self.baseline_prompt_tokens is None or self.last_prompt_tokens is None:
            return False
        return self.last_prompt_tokens >= self.baseline_prompt_tokens + self.growth_tokens

    def note_rebuild(self) -> None:
        """Forget the pre-rebuild measurement. The trigger must not re-fire on a
        stale value; it waits for a fresh post-rebuild measurement."""
        self.last_prompt_tokens = None

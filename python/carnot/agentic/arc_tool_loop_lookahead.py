"""Tool-calling orientation + multi-step lookahead search for ARC-AGI-3 (2026-07-23, operator
follow-up to REQ-ARC-WMTE-5827).

**Researcher summary:**
    The verifier-filtered reactive loop (REQ-ARC-WMTE-5827) tested clean but null on the 3
    worst live/oracle-gap games -- diagnosed (`ops/verifier_gaps.md` GAP-ARC-REACTIVE-FILTER-MYOPIC)
    as MYOPIC: its two filter signals are purely local/single-step, with no direction toward a
    distant goal, and these games need deep (13-33+ action) narrow winning sequences. This module
    adds the two things that diagnosis named as missing, per operator directive: (1) up to 12
    tool-calling/REPL-style turns per decision (inspect frame/history/actions, reason, THEN
    propose), matching Duck Harness's actual per-decision orientation budget rather than a single
    forced-terse completion; (2) real multi-step lookahead by REUSING `arc_solver_kit.OfflineSolver`
    (the project's already-built, already-tested-across-25-games best-first search engine with
    replay-from-reset backtracking) rather than writing a new search algorithm -- this module only
    supplies OfflineSolver's three pluggable hooks (`action_labels`, `apply`, `verifier`) plus a
    `move_pruner` wrapping REQ-ARC-WMTE-5827's dead-end filter.

**Detailed explanation for engineers:**
    The tool API is a small, SAFE, named-function dispatch table (inspect_frame, inspect_cell,
    count_color, inspect_history, list_available_actions) -- NOT arbitrary Python code execution.
    Duck Harness's own mechanism is a full sandboxed Python REPL; a constrained tool API is a
    deliberate, disclosed scope reduction for a 9B model (which this session already found
    struggles with reliable structured output even for a single JSON array -- arbitrary
    free-form code from this model would be materially less reliable, and a real Python sandbox
    adds a security surface this task does not need). The turn loop still gives the model
    multiple ROUNDS of inspection before committing, which is the mechanism's actual claimed
    value (orientation-time compute before commitment), just via a safer tool surface.

    `verifier()` (OfflineSolver's best-first ordering heuristic, LOWER=closer to the win) is fed
    by the LLM's OWN self-reported confidence from its final ACTION turn -- a genuine, if noisy,
    goal-directed signal, which is exactly the missing piece GAP-ARC-REACTIVE-FILTER-MYOPIC named
    (no per-game hand_verifier exists for un-adaptered games, so a hand-authored goal-distance
    function is not available; an LLM-judged one generalizes without per-game authoring). Threaded
    through a small mutable box (`_last_confidence`) rather than OfflineSolver's frame-only
    verifier signature, since apply() and verifier() are called back-to-back on the same child
    frame within one search-loop iteration (see class docstring for the exact ordering this
    relies on).

Spec: openspec/capabilities/arc-world-model-trust-energy/spec.md REQ-ARC-WMTE-5828
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

MAX_TOOL_TURNS = 12
MAX_CANDIDATES = 3
_TOOL_TOKENS = 200

_TOOL_RE = re.compile(r"TOOL:\s*(\w+)\s*(.*)", re.I)
_ACTION_RE = re.compile(r"ACTION:\s*(\[.*\])", re.S)


@dataclass
class Candidate:
    action_id: int
    data: Optional[dict]
    confidence: float


@dataclass
class ToolLoopOutcome:
    """`candidates` is ranked highest-confidence first, length 1..MAX_CANDIDATES -- a RANKED SET,
    not one action, so the caller (the OfflineSolver search wiring) has real branching to
    backtrack into. A single, forced-greedy top choice would make the reused search engine
    degenerate to a straight-line rollout with no actual lookahead (see module docstring)."""

    candidates: list[Candidate]
    turns_used: int
    transcript: list[str] = field(default_factory=list)
    ok: bool = True


def _ascii_grid(grid: np.ndarray) -> str:
    hexd = "0123456789abcdef"
    g = np.clip(np.asarray(grid).astype(int), 0, 15)
    return "\n".join("".join(hexd[v] for v in row) for row in g)


class _ToolDispatch:
    """The safe, named-function tool surface. No arbitrary code execution -- see module
    docstring for why this scope reduction was chosen over a full Python REPL."""

    def __init__(self, grid: np.ndarray, recent: list[str], avail: list[int]) -> None:
        self.grid = grid
        self.recent = recent
        self.avail = avail

    def call(self, name: str, args: str) -> str:
        name = name.lower()
        try:
            if name == "inspect_frame":
                return _ascii_grid(self.grid)
            if name == "inspect_cell":
                parts = [int(x) for x in re.findall(r"-?\d+", args)[:2]]
                if len(parts) != 2:
                    return "ERROR: inspect_cell needs row col"
                row, col = parts
                h, w = self.grid.shape
                if not (0 <= row < h and 0 <= col < w):
                    return f"ERROR: ({row},{col}) out of bounds for {h}x{w} grid"
                return str(int(self.grid[row, col]))
            if name == "count_color":
                m = re.search(r"-?\d+", args)
                if not m:
                    return "ERROR: count_color needs a color id"
                color = int(m.group(0))
                return str(int(np.sum(self.grid == color)))
            if name == "inspect_history":
                if not self.recent:
                    return "(no history yet)"
                return "\n".join(self.recent[-8:])
            if name == "list_available_actions":
                return str(self.avail)
        except Exception as exc:  # noqa: BLE001 - a tool error is a datum returned to the model
            return f"ERROR: {exc!r}"[:200]
        return f"ERROR: unknown tool {name!r}. Valid tools: inspect_frame, inspect_cell, count_color, inspect_history, list_available_actions"


_SYSTEM_FRAME = (
    "/no_think\n"
    "You are inspecting an ARC puzzle to decide your next action. You may call a TOOL to learn "
    "more before deciding, up to {max_turns} times total. Tools:\n"
    "  TOOL: inspect_frame\n"
    "  TOOL: inspect_cell <row> <col>\n"
    "  TOOL: count_color <color>\n"
    "  TOOL: inspect_history\n"
    "  TOOL: list_available_actions\n"
    "Never repeat a tool call you already made this turn sequence -- the transcript below shows "
    "everything you already know; re-asking wastes a turn.\n"
    "Respond with EXACTLY ONE LINE: either a NEW tool call, or your final decision -- up to "
    f"{MAX_CANDIDATES} candidate actions ranked BEST FIRST, most confident first:\n"
    '  ACTION: [{{"a":<action id>,"x":<x>,"y":<y>,"confidence":<0.0-1.0>}}, ...]\n'
    '("x"/"y" only needed for action 6. confidence = how sure you are THIS action moves toward '
    "winning the level -- 0.0 if it's a random guess, 1.0 if you're certain. Give multiple "
    "candidates only if genuinely unsure between them; one candidate is fine if you are sure.)\n"
    "{urgency}"
    "Turns remaining: {remaining}\n\n"
    "TRANSCRIPT SO FAR:\n{transcript}\n\n"
    "Your response (ONE line, TOOL: ... or ACTION: [...]):\n"
)
_URGENCY_THRESHOLD = 3
_URGENCY_NOTE = (
    "URGENT: you have very few turns left. Respond with ACTION now using your best guess -- do "
    "NOT call another tool.\n"
)


def _parse_turn(text: str) -> tuple[str, Any]:
    """Returns ("tool", (name, args)) or ("action", list[dict]) or ("unparseable", None)."""
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    m = _ACTION_RE.search(line) or _ACTION_RE.search(text)
    if m:
        try:
            parsed = json.loads(m.group(1))
            return ("action", parsed) if isinstance(parsed, list) else ("unparseable", None)
        except Exception:
            return "unparseable", None
    m = _TOOL_RE.match(line)
    if m:
        return "tool", (m.group(1), m.group(2).strip())
    return "unparseable", None


def _candidates_from_payload(payload: list) -> list[Candidate]:
    out: list[Candidate] = []
    for item in payload[:MAX_CANDIDATES]:
        if not isinstance(item, dict) or item.get("a") is None:
            continue
        try:
            aid = int(item["a"])
        except Exception:
            continue
        data = None
        if aid == 6 and "x" in item and "y" in item:
            try:
                data = {"x": int(item["x"]), "y": int(item["y"])}
            except Exception:
                data = None
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
        except Exception:
            confidence = 0.5
        out.append(Candidate(action_id=aid, data=data, confidence=confidence))
    return out


def _completion(proposer: Any, prompt: str, *, max_tokens: int) -> tuple[bool, str]:
    """Minimal free-text completion, same rationale as arc_reactive_verifier_filter._raw_llm_completion
    (proposer.generate() is hardcoded for Python code synthesis, unusable for a free-text task)."""
    if not proposer._ensure_server():
        return False, "gpu llama-server failed to start"
    import urllib.request

    try:
        req = urllib.request.Request(
            proposer._url() + "/completion",
            data=json.dumps(
                {
                    "prompt": prompt,
                    "n_predict": int(max_tokens),
                    "temperature": 0.2,
                    "cache_prompt": True,
                    "stop": ["\n"],
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=proposer.timeout) as r:
            response = json.load(r)
        return True, str(response.get("content") or "")
    except Exception as exc:  # noqa: BLE001 - a completion failure is a datum, not a crash
        return False, f"completion request failed: {exc!r}"[:200]


def run_tool_loop(
    grid: np.ndarray,
    recent: list[str],
    avail: list[int],
    proposer: Any,
    *,
    max_turns: int = MAX_TOOL_TURNS,
) -> ToolLoopOutcome:
    """Give the model up to `max_turns` rounds to inspect the puzzle (via the safe tool
    dispatch) before it must commit to one action + a self-assessed confidence. An unparseable
    or tool-calling turn consumes budget but does not end the loop; running out of turns without
    a committed action returns ok=False (the caller decides the fallback)."""
    dispatch = _ToolDispatch(grid, recent, avail)
    transcript_lines: list[str] = []
    seen_tool_calls: set[tuple[str, str]] = set()
    for turn in range(max_turns):
        remaining = max_turns - turn
        prompt = _SYSTEM_FRAME.format(
            max_turns=max_turns,
            remaining=remaining,
            urgency=_URGENCY_NOTE if remaining <= _URGENCY_THRESHOLD else "",
            transcript="\n".join(transcript_lines) or "(nothing yet)",
        )
        ok, text = _completion(proposer, prompt, max_tokens=_TOOL_TOKENS)
        if not ok:
            transcript_lines.append(f"[turn {turn}] completion failed: {text}")
            continue
        kind, payload = _parse_turn(text)
        if kind == "action":
            candidates = _candidates_from_payload(payload)
            if not candidates:
                transcript_lines.append(f"[turn {turn}] ACTION had no valid candidates, ignored")
                continue
            candidates.sort(key=lambda c: c.confidence, reverse=True)
            transcript_lines.append(
                f"[turn {turn}] ACTION committed: "
                + ", ".join(f"a={c.action_id} conf={c.confidence}" for c in candidates)
            )
            return ToolLoopOutcome(
                candidates=candidates,
                turns_used=turn + 1,
                transcript=transcript_lines,
                ok=True,
            )
        if kind == "tool":
            name, args = payload
            call_key = (name.lower(), args)
            if call_key in seen_tool_calls:
                # A dead-cheap, deterministic fix for a real observed failure mode: this model
                # re-asks a question it already has the answer to instead of committing. Rather
                # than burn a real completion telling it so, short-circuit locally.
                transcript_lines.append(
                    f"[turn {turn}] TOOL {name} {args} -> (already asked -- see transcript above)"
                )
                continue
            seen_tool_calls.add(call_key)
            result = dispatch.call(name, args)
            transcript_lines.append(f"[turn {turn}] TOOL {name} {args} -> {result}")
            continue
        transcript_lines.append(f"[turn {turn}] unparseable: {text[:120]!r}")
    return ToolLoopOutcome(
        candidates=[],
        turns_used=max_turns,
        transcript=transcript_lines,
        ok=False,
    )


# ---------------------------------------------------------------------------
# OfflineSolver wiring: real multi-step lookahead by reusing the project's own
# best-first search engine (arc_solver_kit.OfflineSolver), not a new search algorithm.
# ---------------------------------------------------------------------------


class _DeadEndPruner:
    """A move_pruner (OfflineSolver's should_prune/observe protocol) wrapping the exact
    exact-match dead-end check from REQ-ARC-WMTE-5827's reactive filter -- reused, not
    reimplemented. should_prune/observe take (frame, label)/(frame_before, label, frame_after,
    leveled), OfflineSolver's own protocol; frame hashing reuses frame_state_key so the dead-end
    key convention matches arc_reactive_verifier_filter's exactly."""

    def __init__(self) -> None:
        self._dead_end_keys: set[tuple] = set()

    def _key(self, frame: Any, label: str) -> tuple:
        from carnot.agentic.arc_frame_change_predictor import frame_state_key

        from carnot.agentic.arc_reactive_verifier_filter import _dead_end_key

        step = json.loads(label)
        return _dead_end_key(frame_state_key(frame), step.get("action"), step.get("data"))

    def should_prune(self, frame: Any, label: Any) -> bool:
        try:
            return self._key(frame, label) in self._dead_end_keys
        except Exception:
            return False

    def observe(self, frame_before: Any, label: Any, frame_after: Any, leveled_up: bool) -> None:
        if leveled_up:
            return
        try:
            from carnot.agentic.arc_llm_guided_solve import _delta_desc
            from carnot.agentic.arc_executable_world_model import to_logical, detect_cell
            from carnot.agentic.arc_agi3_world_model import grid_of

            cell = detect_cell(grid_of(frame_before))
            before = to_logical(grid_of(frame_before), cell)
            after = to_logical(grid_of(frame_after), cell)
            if _delta_desc(before, after) == "no change":
                self._dead_end_keys.add(self._key(frame_before, label))
        except Exception:
            pass


class ToolLoopLookaheadSession:
    """Supplies OfflineSolver's three hooks, backed by the tool-calling loop above.

    ONE call to `run_tool_loop` per EXPANDED search node (not per candidate) -- the tool loop
    itself returns a ranked candidate SET, which becomes this node's children, so search
    breadth comes from the model's own ranked alternatives rather than from re-running the
    (expensive, up-to-12-turn) tool loop once per candidate.

    `apply()` and `_priority()`/`verifier()` are called back-to-back on the same child frame
    within a single OfflineSolver expansion-loop iteration (see arc_solver_kit.py
    OfflineSolver.solve_level: `f2 = self.apply(...)` then, a few lines later in the SAME
    iteration, `heapq.heappush(heap, (self._priority(env, child_path), ...))` which reads
    `self.last_frame` -- OfflineSolver sets `self.last_frame = f2` itself right after apply()
    returns). No other apply()/verifier() call interleaves between them, so a single mutable
    `_pending_confidence` dict keyed by label is safe without extra synchronization.
    """

    # OfflineSolver's own _replay() re-applies warmup_label on EVERY node visit and every
    # sibling-restoration replay (arc_solver_kit.py:5241) -- if apply() recorded every call into
    # `recent`, the model-facing history would be flooded with repeated fake warmup entries the
    # model never actually chose, not genuine search decisions. A real smoke test found the
    # OBVIOUS fix (compare the label against the warmup ACTION, e.g. action 1) is wrong: if the
    # model's genuine search choice happens to also BE action 1, it gets silently swallowed as
    # "just the warmup" too (found directly -- an early attempt zeroed out `recent` entirely
    # because every real candidate that run happened to be action 1). Fixed by using a SENTINEL
    # label that can never collide with a genuine `_json_action_label()`-encoded candidate
    # (those are always valid JSON starting with "{"; this sentinel deliberately is not).
    WARMUP_LABEL = "__TOOL_LOOP_WARMUP_SENTINEL__"
    _WARMUP_ACTION_ID = 1

    def __init__(self, proposer: Any, *, max_turns: int = MAX_TOOL_TURNS) -> None:
        self.proposer = proposer
        self.max_turns = max_turns
        self.recent: list[str] = []
        self.pruner = _DeadEndPruner()
        self._pending_confidence: dict[str, float] = {}
        self._last_confidence: float = 0.5
        self._tool_loop_calls: int = 0
        self._transcripts: list[list[str]] = []
        self._node_frame: Any = None

    def action_labels(self, env: Any, frame: Any, path: tuple) -> list[str]:
        del env, path
        from carnot.agentic.arc_agi3_world_model import grid_of
        from carnot.agentic.arc_executable_world_model import to_logical, detect_cell
        from carnot.agentic.arc_game_adapters import _json_action_label
        from carnot.agentic.arc_graph_explore import rich_action_candidates

        if frame is None:
            return []
        try:
            cell = detect_cell(grid_of(frame))
            grid = to_logical(grid_of(frame), cell)
        except Exception:
            return []
        avail = list(getattr(frame, "available_actions", []) or range(1, 7))
        outcome = run_tool_loop(grid, self.recent, avail, self.proposer, max_turns=self.max_turns)
        self._tool_loop_calls += 1
        self._transcripts.append(outcome.transcript)
        labels: list[str] = []
        seen_labels: set[str] = set()
        if outcome.ok:
            for cand in outcome.candidates:
                label = _json_action_label(cand.action_id, cand.data)
                self._pending_confidence[label] = cand.confidence
                labels.append(label)
                seen_labels.add(label)
        # A real diagnosed failure mode (found via a direct trace, not assumed): if the tool
        # loop's ONLY candidate turns out to be a no-op (a real risk with a 9B model's often
        # uninformed first guess -- observed directly: a click on empty background), that
        # candidate's resulting state hashes IDENTICAL to the parent and never gets pushed to
        # the search frontier at all (correctly -- it is not a new state). With only one
        # candidate offered, the search then has NOTHING left to explore and dies after one
        # wasted node, regardless of the 12-turn tool-loop budget or the max_nodes search
        # budget -- neither ever gets to matter. Padding with a couple of cheap, STRUCTURED
        # fallback candidates (the same rich_action_candidates() salience ranking the rest of
        # the project's search machinery already uses) guarantees the search always has real
        # alternatives to branch into, even when the tool loop's own judgment is a dead end.
        # Given a LOW confidence (0.1, well below any genuine LLM judgment) so best-first order
        # still tries the tool loop's own choices first when they exist.
        if len(labels) < 2:
            try:
                for cand in rich_action_candidates(frame)[:3]:
                    fallback_label = _json_action_label(cand.action_id, cand.data)
                    if fallback_label in seen_labels:
                        continue
                    self._pending_confidence[fallback_label] = 0.1
                    labels.append(fallback_label)
                    seen_labels.add(fallback_label)
            except Exception:
                pass
        # apply() (below) is called by OfflineSolver with frame=None during search expansion
        # (arc_solver_kit.py:5275,5329 -- only _replay's own top-level call passes the real
        # frame). action_labels() is the only hook that reliably receives the node's real
        # PRE-expansion frame, so it is captured here for apply()'s delta-description bookkeeping.
        self._node_frame = frame
        return labels

    def apply(self, env: Any, label: str, frame: Any) -> Any:
        del (
            frame
        )  # see action_labels()'s comment -- OfflineSolver passes None here; use _node_frame
        from arcengine import GameAction
        from carnot.agentic.arc_agi3_live_adapter import _game_action
        from carnot.agentic.arc_executable_world_model import to_logical, detect_cell
        from carnot.agentic.arc_agi3_world_model import grid_of
        from carnot.agentic.arc_llm_guided_solve import _delta_desc

        if label == self.WARMUP_LABEL:
            # OfflineSolver's own _replay() re-issues this on every node visit and every
            # sibling-restoration replay -- not a genuine search decision. Execute the warmup
            # action directly (no JSON parse -- this sentinel is not JSON) without recording, so
            # the model-facing history and verifier score aren't polluted by internal search
            # bookkeeping. See the WARMUP_LABEL class attribute's comment for why this can't be
            # done by comparing action VALUES instead of using a dedicated sentinel string.
            return env.step(_game_action(GameAction, self._WARMUP_ACTION_ID), data=None)
        step = json.loads(label)
        aid, data = int(step["action"]), step.get("data")
        nf = env.step(_game_action(GameAction, aid), data=data)
        prev_frame = getattr(self, "_node_frame", None)
        self._last_confidence = self._pending_confidence.pop(label, 0.5)
        try:
            if prev_frame is not None and nf is not None:
                cell = detect_cell(grid_of(nf))
                before = to_logical(grid_of(prev_frame), cell)
                after = to_logical(grid_of(nf), cell)
                delta = _delta_desc(before, after)
            else:
                delta = "unknown"
            data_str = f"x={data['x']},y={data['y']}" if data else ""
            self.recent.append(f"{aid} ({data_str}) -> {delta}")
        except Exception:
            pass
        return nf

    def verifier(self, game_obj: Any, frame: Any) -> float:
        del game_obj, frame
        # LOWER = closer to the win (OfflineSolver's own convention); confidence is
        # HIGHER = more likely to help, so invert.
        return 1.0 - self._last_confidence

    def state_key(self, game_obj: Any, frame: Any) -> Any:
        del game_obj
        from carnot.agentic.arc_frame_change_predictor import frame_state_key

        return frame_state_key(frame)

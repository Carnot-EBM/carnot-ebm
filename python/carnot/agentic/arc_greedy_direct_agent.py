"""Duck-Harness-style GREEDY-DIRECT ARC agent (2026-07-23, operator: "match the leaderboard leaders").

WHY. The 2026-07-20 winner audit (docs/research-notes/arc-top-project-search-architecture-audit-
2026-07-20.md) found all three Milestone-1 winners are GREEDY single-commit generators: a capable
27-31B LLM directly picks each action (Duck via a tool-loop, Reki/forge via a 1-4 action micro-plan),
with NO tree/beam/MCTS search and NO induced world model. Carnot already has strictly MORE search than
any winner; our own induce-then-plan architecture already NULLED with a 31B inducer swapped in
(experiment_5722, delta 0.0) -- because we used the big model the WRONG way (as a world-model inducer
feeding our search), not the winners' way (as the direct action decider). The operator directive is to
MATCH the winners: the 27-31B model AS a direct greedy action generator. This module is that agent.

WHAT IT IS. Per decision: render the current frame (compact logical grid) + recent action->effect
history; give the model up to `max_turns` tool-calling turns to inspect (reusing the safe tool surface
from arc_tool_loop_lookahead); then it commits a SHORT SEQUENCE of 1..`max_seq` actions which are
executed DIRECTLY on the real env (irreversible -- no search, no rollback, no induced model), matching
the winners (Reki/forge commit 1-4 actions/orientation to amortize the LLM cost). Re-orient after the
sequence, a level-up, or a game-over. This is the offline_arcade_live_agent_runtime_self_discovery
substrate: the agent DISCOVERS progress from its own play, no GameAdapter, no banked trajectory.

CLICK-COORDINATE CORRECTNESS. The model sees the LOGICAL grid (to_logical: raw 64x64 downsampled by
the detected cell size) and gives click x=col,y=row as indices into THAT grid; we map back to raw
pixels (center of the cell block, clamped to [0,63]) before env.step. The earlier tool-loop
(arc_tool_loop_lookahead) fed the model's coords straight to env.step while showing it the logical
grid -- a latent coordinate mismatch this module deliberately fixes.

COMPLETION PATH. Raw /completion with fence-priming ("ACTION: [") -- smoke-verified for gemma-4-31B on
2026-07-23 (raw-unprimed derails into gemma's <|channel>thought; the chat-template endpoint returned
empty; the primed raw path returns clean parseable JSON). Same pattern that works for Qwen3.5-9B.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from carnot.agentic.arc_tool_loop_lookahead import _ToolDispatch, _ascii_grid, _TOOL_RE

MAX_ORIENT_TURNS = (
    4  # tool-inspection turns before a commit (Duck uses up to 12; a 31B needs fewer)
)
MAX_SEQ = 5  # actions committed per orientation (Reki/forge commit 1-4)
_TOOL_TOKENS = 220
_ACTION_TOKENS = 200
_URGENCY_TURNS = 1


@dataclass
class GreedyDirectResult:
    game: str
    levels_gained: int
    reached_level: int
    actions_taken: int
    orientations: int
    game_over: bool
    wall_s: float
    transcript_sample: list[str] = field(default_factory=list)
    final_notes: str = ""


def _complete(
    proposer: Any, prompt: str, *, max_tokens: int, stop: list[str], seed: Optional[int] = None
) -> tuple[bool, str]:
    """Raw /completion (fence-priming friendly). Returns (ok, text). A failure is a datum, not a crash."""
    if not proposer._ensure_server():
        return False, "gpu llama-server failed to start"
    import urllib.request

    payload = {
        "prompt": prompt,
        "n_predict": int(max_tokens),
        "temperature": 0.3,
        "cache_prompt": True,
        "stop": list(stop),
    }
    if seed is not None:
        payload["seed"] = int(seed)
    try:
        req = urllib.request.Request(
            proposer._url() + "/completion",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=proposer.timeout) as r:
            return True, str(json.load(r).get("content") or "")
    except Exception as exc:  # noqa: BLE001
        return False, f"completion request failed: {exc!r}"[:200]


_ORIENT_FRAME = (
    "You are an agent DISCOVERING how to win an unknown grid game by acting on it. Actions: "
    "1=up 2=down 3=left 4=right 6=click(x,y). You commit actions DIRECTLY to the real game (there is "
    "NO undo), so think before acting. You may call ONE tool per line to inspect first (up to "
    "{max_turns} turns), then commit.\n"
    "  TOOL: inspect_cell <row> <col> | count_color <color> | inspect_history | list_available_actions\n"
    "The frame is a {h}x{w} grid of hex colors (row 0 = top, col 0 = left); click x=COLUMN, y=ROW as "
    "indices into THIS grid:\n{grid}\n"
    "Available actions this frame: {avail}\n"
    "LEARNED NOTES (your evolving understanding of THIS game -- authoritative but revisable):\n{notes}\n"
    "Recent (your last moves -> observed effect):\n{recent}\n"
    "EXPLORE to find what the game responds to: if an action in your history shows 'no change', do "
    "NOT repeat it -- try a DIFFERENT action id or a DIFFERENT location. Vary moves vs clicks.\n"
    "{urgency}Turns left: {remaining}. Notes so far:\n{transcript}\n\n"
    "Respond with EXACTLY ONE LINE: either a NEW tool call, OR commit a SHORT SEQUENCE of 1-{max_seq} "
    "actions to run in order (then you re-inspect). Use STRICT JSON exactly like this:\n"
    '  ACTION: [{{"a":6,"x":<col>,"y":<row>}}]   or   ACTION: [{{"a":4}},{{"a":4}}]   '
    '("x","y" only for action 6)\n'
    "Your line:\n"
)
_URGENCY = "URGENT: few turns left -- commit an ACTION now, do not inspect.\n"

# v2 perception (2026-07-23): the winner audit's #1 "highest-value steal" -- all three Milestone-1
# winners feed OBJECT SEGMENTATION as the primary view (Duck: "segmentation is the primary view; the
# raw grid is not available"). v1 showed the model the raw hex grid and it fixated on clicking a
# coordinate cluster, discovering nothing. This mode segments the frame into discrete objects (reusing
# arc_color_blob_salience) so the model reasons over OBJECTS (click object #N), not raw cells.
_ORIENT_FRAME_OBJ = (
    "You are an agent DISCOVERING how to win an unknown grid game by acting on it. Actions: "
    "1=up 2=down 3=left 4=right 6=click. You act DIRECTLY on the real game (there is NO undo).\n"
    "The frame is a {h}x{w} grid. Its DISCRETE OBJECTS (connected same-color regions; the large "
    "uniform background is omitted) are:\n{objects}\n"
    "Available actions this frame: {avail}\n"
    "LEARNED NOTES (your evolving understanding of THIS game -- authoritative but revisable):\n{notes}\n"
    "Recent (your last moves -> observed effect):\n{recent}\n"
    "EXPLORE to find what the game responds to: if an action in your history shows 'no change', do NOT "
    "repeat it -- try a DIFFERENT object or a move. Vary moves (1-4) vs clicks (6).\n"
    "{urgency}Turns left: {remaining}. Notes so far:\n{transcript}\n\n"
    "Respond with EXACTLY ONE LINE: a NEW tool call, OR commit a SHORT SEQUENCE of 1-{max_seq} actions. "
    "To click an OBJECT, use its id. STRICT JSON:\n"
    '  ACTION: [{{"a":6,"obj":<id>}}]   or   ACTION: [{{"a":4}},{{"a":4}}]   or a raw cell '
    '{{"a":6,"x":<col>,"y":<row>}}\n'
    "Your line:\n"
)


def _render_objects(logical: np.ndarray) -> tuple[str, list[dict]]:
    """Segment the (logical) frame into discrete objects for the winner-style object view. Returns
    (rendered_text, objects) where each object carries its LOGICAL centroid (row, col)."""
    try:
        from carnot.agentic.arc_color_blob_salience import connected_color_blobs

        blobs = connected_color_blobs(logical)  # excludes the dominant background wholesale
    except Exception:
        return "(segmentation unavailable)", []
    objects: list[dict] = []
    for i, b in enumerate(blobs):
        objects.append(
            {
                "id": i,
                "color": int(b.color),
                "row": int(round(b.centroid[0])),
                "col": int(round(b.centroid[1])),
                "size": int(b.pixel_count),
            }
        )
    if not objects:
        return "(no discrete objects segmented -- try moves or a raw-cell click)", objects
    text = "\n".join(
        f"#{o['id']} color={o['color']} at row={o['row']} col={o['col']} size={o['size']}"
        for o in objects
    )
    return text, objects


# v3 persistent reflection memory (2026-07-23): Reki/forge's key mechanism (winner audit) -- an NL
# notes doc the model periodically REWRITES from its accumulated play and that is re-injected into every
# decision. v1/v2 gave the model only thin recent-history; without a durable, self-authored model of the
# game's rules/goal it re-explored blindly. This closes that gap.
_REFLECT_FRAME = (
    "You are learning an unknown grid game by playing it. Below are your CURRENT NOTES and your RECENT "
    "moves with their observed effects. REWRITE your notes to capture what you have learned so far -- "
    "concise, concrete, only what is useful for winning. Do not include reasoning, just the notes.\n\n"
    "CURRENT NOTES:\n{notes}\n\n"
    "RECENT MOVES (action -> effect):\n{recent}\n\n"
    "Rewrite under these headings (one short line each):\n"
    "RULES: what each move/click actually does in THIS game\n"
    "GOAL: your best hypothesis for how to level up / win\n"
    "PROGRESS: what you've achieved or definitively ruled out\n"
    "AVOID: specific actions/objects/locations that do nothing (dead-ends)\n\n"
)
_REFLECT_TOKENS = 300
_NOTES_MAX_CHARS = 1200


def reflect(proposer: Any, notes: str, recent: list[str], *, seed: Optional[int] = None) -> str:
    """Rewrite the persistent game notes from accumulated play (Reki/forge's reflection). Primed with
    'RULES:' so gemma writes the notes directly instead of derailing into its reasoning channel."""
    prompt = _REFLECT_FRAME.format(
        notes=notes or "(none yet)", recent="\n".join(recent[-20:]) or "(none yet)"
    )
    ok, text = _complete(
        proposer, prompt + "RULES:", max_tokens=_REFLECT_TOKENS, stop=["\n\n\n"], seed=seed
    )
    if not ok or not text.strip():
        return notes  # keep prior notes on a failed reflection
    return ("RULES:" + text).strip()[:_NOTES_MAX_CHARS]


def _parse_sequence(
    primed_text: str, *, max_seq: int, avail: list[int], objects: Optional[list[dict]] = None
) -> list[dict]:
    """Parse the model's completion AFTER the 'ACTION: [' prime into a validated action sequence.
    Wraps with the leading '[' the prime supplied; salvages a missing trailing ']'."""
    raw = "[" + primed_text.strip()
    if not raw.rstrip().endswith("]"):
        raw = raw.rstrip().rstrip(",") + "]"
    try:
        items = json.loads(raw)
    except Exception:
        return []
    if not isinstance(items, list):
        return []
    # Loose-format recovery: the model sometimes emits a FLAT number list after the prime, e.g.
    # "6, 18, 41]" -> [6,18,41] (a=6,x=18,y=41) or "4]" -> [4] (a=4). Coerce a purely-numeric top
    # level into a single action dict so a valid-but-unbracketed intent is not wasted as a retry
    # (observed with gemma-4-31B on the first smoke, 2026-07-23).
    if items and all(isinstance(i, (int, float)) and not isinstance(i, bool) for i in items):
        nums = [int(i) for i in items]
        if len(nums) == 1:
            items = [{"a": nums[0]}]
        elif len(nums) >= 3:
            items = [{"a": nums[0], "x": nums[1], "y": nums[2]}]
        else:
            return []
    out: list[dict] = []
    for item in items[:max_seq]:
        if not isinstance(item, dict) or item.get("a") is None:
            continue
        try:
            aid = int(item["a"])
        except Exception:
            continue
        if aid not in avail:
            continue
        step = {"a": aid}
        if aid == 6:
            if "obj" in item and objects is not None:
                # winner-style object reference -> resolve to that object's LOGICAL centroid
                try:
                    oid = int(item["obj"])
                except Exception:
                    continue
                obj = next((o for o in objects if o["id"] == oid), None)
                if obj is None:
                    continue
                step["x"] = int(obj["col"])
                step["y"] = int(obj["row"])
            else:
                try:
                    step["x"] = int(item["x"])
                    step["y"] = int(item["y"])
                except Exception:
                    continue  # a click with no valid coords/obj is unusable
        out.append(step)
    return out


def decide_sequence(
    logical: np.ndarray,
    recent: list[str],
    avail: list[int],
    proposer: Any,
    *,
    max_turns: int = MAX_ORIENT_TURNS,
    max_seq: int = MAX_SEQ,
    seed: Optional[int] = None,
    perception: str = "grid",
    notes: str = "",
) -> tuple[list[dict], list[str]]:
    """One decision: up to `max_turns` tool-inspection turns, then a committed action sequence (in
    LOGICAL grid coords for clicks). Returns (sequence, transcript). `perception`: "grid" (raw hex,
    v1) or "objects" (winner-style segmentation view, v2). `notes`: persistent reflection memory (v3)."""
    h, w = logical.shape if logical.ndim == 2 else (0, 0)
    dispatch = _ToolDispatch(logical, recent, avail)
    transcript: list[str] = []
    seen_tools: set[tuple[str, str]] = set()
    obj_text, objects = _render_objects(logical) if perception == "objects" else ("", None)

    def _prompt(remaining: int) -> str:
        common = dict(
            max_turns=max_turns,
            max_seq=max_seq,
            h=h,
            w=w,
            avail=avail,
            notes=notes or "(none yet -- explore to learn the game)",
            recent="\n".join(recent[-6:]) or "(none yet)",
            urgency=_URGENCY if remaining <= _URGENCY_TURNS else "",
            remaining=remaining,
            transcript="\n".join(transcript) or "(nothing yet)",
        )
        if perception == "objects":
            return _ORIENT_FRAME_OBJ.format(objects=obj_text, **common)
        return _ORIENT_FRAME.format(grid=_ascii_grid(logical), **common)

    for turn in range(max_turns):
        remaining = max_turns - turn
        base = _prompt(remaining)
        # Ask for one line; if it's a tool call (and turns remain) honor it, else commit an action.
        ok, line = _complete(
            proposer,
            base,
            max_tokens=_TOOL_TOKENS,
            stop=["\n"],
            seed=(seed + turn if seed else None),
        )
        m = _TOOL_RE.match(line.strip()) if ok else None
        if m and remaining > 1:
            name, args = m.group(1), m.group(2).strip()
            key = (name.lower(), args)
            if key in seen_tools:
                transcript.append(f"[t{turn}] TOOL {name} {args} -> (already asked)")
                continue
            seen_tools.add(key)
            transcript.append(f"[t{turn}] TOOL {name} {args} -> {dispatch.call(name, args)}")
            continue
        # Commit: re-complete PRIMED with the answer fence.
        ok2, primed = _complete(
            proposer,
            base + "ACTION: [",
            max_tokens=_ACTION_TOKENS,
            stop=["]"],
            seed=(seed + 500 + turn if seed else None),
        )
        seq = _parse_sequence(primed, max_seq=max_seq, avail=avail, objects=objects) if ok2 else []
        if seq:
            transcript.append(f"[t{turn}] COMMIT {seq}")
            return seq, transcript
        transcript.append(f"[t{turn}] unparseable/empty action: {primed[:60]!r}")
    return [], transcript


def run_greedy_direct(
    game: str,
    proposer: Any,
    *,
    action_budget: int = 150,
    max_turns: int = MAX_ORIENT_TURNS,
    max_seq: int = MAX_SEQ,
    seed: Optional[int] = None,
    arcade: Any = None,
    perception: str = "grid",
    reflection_interval: int = 0,
) -> GreedyDirectResult:
    """The Duck-Harness-style greedy-direct loop against the offline arcade (adapter-free, no search).
    `perception`: "grid" (raw hex, v1) or "objects" (winner-style segmentation view, v2).
    `reflection_interval` (v3): if >0, the model rewrites its persistent notes every N actions
    (Reki/forge's reflection memory), re-injected into every decision. 0 disables it."""
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import (
        _game_action,
        _game_over,
        _available_action_ids,
    )
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_executable_world_model import detect_cell, to_logical
    from carnot.agentic.arc_solver_kit import offline_arcade, frame_level
    from carnot.agentic.arc_llm_guided_solve import _delta_desc

    t0 = time.time()
    arc = arcade or offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    start_level = frame_level(frame)
    max_level = start_level
    recent: list[str] = []
    notes = ""
    actions_at_last_reflect = 0
    actions_taken = 0
    orientations = 0
    transcript_sample: list[str] = []

    while actions_taken < action_budget and not _game_over(frame):
        raw = grid_of(frame)
        cell = detect_cell(raw)
        logical = to_logical(raw, cell)
        avail = _available_action_ids(frame) or [1, 2, 3, 4, 6]
        # v3: periodically rewrite the persistent notes from accumulated play (Reki/forge reflection).
        if (
            reflection_interval > 0
            and actions_taken - actions_at_last_reflect >= reflection_interval
        ):
            notes = reflect(proposer, notes, recent, seed=(seed + actions_taken if seed else None))
            actions_at_last_reflect = actions_taken
            if orientations <= 3 or (orientations % 20 == 0):
                transcript_sample.append(f"[reflect@{actions_taken}] {notes[:180]}")
        seq, transcript = decide_sequence(
            logical,
            recent,
            avail,
            proposer,
            max_turns=max_turns,
            max_seq=max_seq,
            seed=seed,
            perception=perception,
            notes=notes,
        )
        orientations += 1
        if orientations <= 3:
            transcript_sample.append(f"orient#{orientations}: " + " | ".join(transcript[-4:]))
        if not seq:
            # fallback: one structured exploratory action so the loop never stalls on a parse failure
            seq = [{"a": avail[0]}]
        for step in seq:
            if actions_taken >= action_budget or _game_over(frame):
                break
            aid = int(step["a"])
            data = None
            if aid == 6 and "x" in step and "y" in step:
                # LOGICAL (col,row) -> RAW pixel center of the cell block, clamped to the frame.
                rx = min(63, max(0, step["x"] * cell + cell // 2))
                ry = min(63, max(0, step["y"] * cell + cell // 2))
                data = {"x": int(rx), "y": int(ry)}
            prev_raw = raw
            frame = env.step(_game_action(GameAction, aid), data=data)
            actions_taken += 1
            new_raw = grid_of(frame)
            try:
                delta = _delta_desc(to_logical(prev_raw, cell), to_logical(new_raw, cell))
            except Exception:
                delta = "?"
            coord = f" ({step.get('x')},{step.get('y')})" if aid == 6 else ""
            recent.append(f"{aid}{coord} -> {delta}")
            lvl = frame_level(frame)
            if lvl > max_level:
                max_level = lvl
                recent.append(f"** LEVEL UP -> L{lvl} **")
                break  # re-orient on progress
            raw = new_raw

    return GreedyDirectResult(
        game=game,
        levels_gained=max_level - start_level,
        reached_level=max_level,
        actions_taken=actions_taken,
        orientations=orientations,
        game_over=_game_over(frame),
        wall_s=round(time.time() - t0, 1),
        transcript_sample=transcript_sample,
        final_notes=notes,
    )

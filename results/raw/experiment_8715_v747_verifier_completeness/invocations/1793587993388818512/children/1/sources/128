"""LLM-as-reasoner gradient for the ARC-AGI-3 hard tail (the SOTA-flagged layer, 2026-06-21).

The hard-tail wins are deep (13-33 action), NARROW, specifically-ordered sequences with no intermediate
reward -- blind exploration (diversity + Go-Explore) is refuted (0/4) because broad coverage cannot thread
a narrow deep needle without a GOAL GRADIENT. This is that gradient: on stall, give the LOCAL Qwen LLM the
current grid (ASCII) + recent action->effect examples + the available actions, and ask it to PROPOSE the
next actions most likely to make progress -- LLM REASONING about what-to-try, grounded by executing the
proposals against the env. This is DISTINCT from LLM world-model INDUCTION (write exact dynamics), which we
showed fails (induced models predict near-identity, gated out): proposing what-to-try needs no exact model,
only plausible spatial reasoning, and every proposal is verified by the env (level-up or not).

Honest scope: uncertain -- Qwen-9B's spatial reasoning on raw ASCII grids is unproven (the induction failure
is a caution). The measurement on ls20 (navigation-to-goal, the most LLM-tractable tail game) is the test of
whether the LLM gradient helps at all. Verifier-grounded: the LLM proposes, the env disposes.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import numpy as np

_HEX = "0123456789abcdef"


def _ascii(grid: np.ndarray) -> str:
    g = np.clip(np.asarray(grid).astype(int), 0, 15)
    return "\n".join("".join(_HEX[v] for v in row) for row in g)


def _delta_desc(prev: np.ndarray, cur: np.ndarray, max_cells: int = 6) -> str:
    a, b = np.asarray(prev), np.asarray(cur)
    if a.shape != b.shape:
        return "grid reshaped"
    ys, xs = np.where(a != b)
    if len(ys) == 0:
        return "no change"
    parts = [
        f"({int(y)},{int(x)}){int(a[y, x])}->{int(b[y, x])}"
        for y, x in list(zip(ys, xs))[:max_cells]
    ]
    more = "" if len(ys) <= max_cells else f" +{len(ys) - max_cells} more"
    return ", ".join(parts) + more


_ACTIONS_RE = re.compile(r"ACTIONS_JSON\s*:\s*(\[.*?\])", re.S)


def _parse_actions(text: str) -> list[dict]:
    m = _ACTIONS_RE.search(text) or re.search(r"(\[\s*\{.*?\}\s*\])", text, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(1))
    except Exception:
        return []
    out = []
    for item in arr if isinstance(arr, list) else []:
        if not isinstance(item, dict):
            continue
        a = item.get("a", item.get("action"))
        if a is None:
            continue
        try:
            aid = int(a)
        except Exception:
            continue
        data = None
        if aid == 6 and "x" in item and "y" in item:
            try:
                data = {"x": int(item["x"]), "y": int(item["y"])}
            except Exception:
                data = None
        out.append({"action": aid, "data": data})
    return out[:8]


def _prompt(grid: np.ndarray, recent: list[str], avail: list[int], propose_n: int) -> str:
    return (
        "/no_think\n"
        "You control an agent in a grid puzzle (colors 0-9,a-f). The game advances a LEVEL when its hidden "
        "objective is met. Reason about the grid structure to make progress.\n\n"
        f"GRID ({grid.shape[0]}x{grid.shape[1]}, rows top->bottom):\n{_ascii(grid)}\n\n"
        f"AVAILABLE ACTIONS: {avail}  (action 6 = CLICK at x,y; actions 1-5 = buttons whose effect you infer "
        "from the recent moves below).\n"
        "RECENT MOVES (action -> cells that changed as (row,col)old->new):\n"
        + ("\n".join(f"  {r}" for r in recent[-8:]) or "  (none yet)")
        + "\n\n"
        f"Propose the NEXT {propose_n} actions most likely to advance the level. Think briefly, then output "
        'EXACTLY one line:\nACTIONS_JSON: [{"a":3},{"a":6,"x":4,"y":2}, ...]\n'
    )


def llm_guided_solve(
    game: str,
    *,
    budget: int = 1500,
    warmup_explore: int = 24,
    propose_n: int = 6,
    max_llm_calls: int = 25,
    seed: int = 0,
    warmup: bool = False,
) -> dict:
    import random
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_agi3_live_adapter import _levels_completed, _game_action
    from carnot.agentic.arc_graph_explore import rich_action_candidates, _warm
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_executable_world_model import (
        ARC_LIVE_GENERATOR_REPO_SUBSTR,
        LocalGGUFProposer,
        detect_cell,
        to_logical,
    )

    rng = random.Random(seed)
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _warm(env, warmup)

    def ok(fr) -> bool:
        try:
            return np.asarray(grid_of(fr)).ndim == 2
        except Exception:
            return False

    if not ok(f):
        return {"game": game, "levels_reached": 0, "error": "degenerate start"}
    cell = detect_cell(grid_of(f))
    start_level = _levels_completed(f)
    best_level = start_level

    # Generator pin read from the canonical constant (2026-07-28 switch to gemma-4-31B-it), never
    # a local literal. mtp=False was already correct here and stays correct: gemma-4-31B has no
    # MTP heads at all, so this call site needed no MTP fix -- unlike the two live-agent sites,
    # which defaulted CARNOT_ARC_MTP to "1" and would have double-loaded the weights.
    proposer = LocalGGUFProposer(
        repo_substr=ARC_LIVE_GENERATOR_REPO_SUBSTR,
        mtp=False,
        kv_quant="q8_0",
        no_think_prefix="",
        max_tokens=512,
    )
    if not proposer._ensure_server():
        return {"game": game, "levels_reached": 0, "error": "llm server failed (no GPU?)"}

    actions = 0
    llm_calls = 0
    first_levelup: Optional[int] = None
    recent: list[str] = []
    gp = to_logical(grid_of(f), cell)

    def alabel(aid, data):
        if data and "x" in data:
            return f"{aid} (x={data['x']},y={data['y']})"
        return f"{aid}"

    def step(aid, data):
        nonlocal f, actions, best_level, first_levelup, gp
        prev = gp
        nf = env.step(_game_action(GameAction, int(aid)), data=data)
        actions += 1
        if nf is None or not ok(nf):
            f = _warm(env, warmup)
            gp = to_logical(grid_of(f), cell) if ok(f) else prev
            recent.append(f"{alabel(aid, data)} -> reset")
            return False
        f = nf
        cur = to_logical(grid_of(nf), cell)
        recent.append(f"{alabel(aid, data)} -> {_delta_desc(prev, cur)}")
        gp = cur
        lvl = _levels_completed(nf)
        if lvl > best_level:
            best_level = lvl
            if first_levelup is None:
                first_levelup = actions
            return True
        return False

    # warmup: gather action->effect examples (salient + a few of each type) so the LLM can infer semantics
    while actions < warmup_explore and actions < budget:
        cands = rich_action_candidates(f) if ok(f) else []
        if not cands:
            f = _warm(env, warmup)
            continue
        c = cands[rng.randrange(min(len(cands), 8))]
        step(int(c.action_id), c.data)

    # LLM-guided loop: propose -> execute -> observe, until win or budget/call cap
    while actions < budget and best_level == start_level and llm_calls < max_llm_calls:
        avail = list(getattr(f, "available_actions", []) or range(1, 7))
        ok_code, text = proposer.generate(
            _prompt(gp, recent, avail, propose_n), required=(), validate=None, tries=1
        )
        llm_calls += 1
        proposed = _parse_actions(text) if ok_code else []
        if not proposed:  # LLM gave nothing usable -> a few salient explores to refresh context
            cands = rich_action_candidates(f) if ok(f) else []
            for _ in range(4):
                if not cands or actions >= budget:
                    break
                c = cands[rng.randrange(min(len(cands), 8))]
                if step(int(c.action_id), c.data):
                    break
            continue
        for mv in proposed:
            if actions >= budget:
                break
            if step(mv["action"], mv["data"]):
                break

    return {
        "game": game,
        "levels_reached": int(best_level - start_level),
        "first_levelup_actions": first_levelup,
        "actions": actions,
        "llm_calls": llm_calls,
        "executor": "llm_as_reasoner_gradient",
    }

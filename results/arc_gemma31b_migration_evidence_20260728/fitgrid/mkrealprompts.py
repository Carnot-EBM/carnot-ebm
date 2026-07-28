"""PHASE 2 prompt corpus: REAL ARC induce prompts from REAL transitions.

Unlike Phase 1's mkprompt.py (which built a synthetic worst-case 64x64 grid through the
production prompt builder), this steps the ACTUAL offline arcade over the ACTUAL public
games, collects transitions through the SAME production primitives the live agent uses
(grid_of -> detect_cell -> to_logical -> Transition), and renders them through the SAME
production induce_prompt().

Everything downstream (every config in the speed grid, every arm of the quality check)
consumes THIS ONE frozen prompt set, so configs are never compared on different inputs.
"""

import hashlib
import json
import sys

import numpy as np

sys.path.insert(0, "/home/ianblenke/github.com/ianblenke/carnot/python")

from carnot.agentic.arc_agi3_world_model import grid_of  # noqa: E402
from carnot.agentic.arc_executable_world_model import (  # noqa: E402
    Transition,
    detect_cell,
    induce_prompt,
)
from carnot.agentic.arc_solver_kit import offline_arcade  # noqa: E402

SEED = 5900
OUT = "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/fitgrid/real_prompts.json"

# Spread of real public games. Chosen for grid-size variety so the corpus spans the
# short-prompt and long-prompt ends of what the induce path really sends.
GAMES = ["ls20", "tu93", "sc25", "lp85", "vc33", "dc22", "ft09", "cn04"]

# The production collector gathers ~25 transitions and induce_prompt shows k=8 of them.
N_TRANS = 25
K = 8


def collect(game: str, rng: np.random.Generator) -> tuple[list[Transition], int] | None:
    """Step the REAL offline env with REAL actions; build REAL Transitions.

    Mirrors the live collector in arc_competition_agent.E3AgentPolicy.next_move:
    logical grid = to_logical(grid_of(frame), cell), level read from the FRAME.
    """
    arc = offline_arcade()
    sc = arc.open_scorecard()
    env = arc.make(game, scorecard_id=sc)
    frame = env.reset()
    # gotcha #4: the first step after reset is consumed in at least sc25 -- warm up.
    try:
        frame = env.step(_action(1))
    except Exception:
        pass

    raw = grid_of(frame)
    cell = detect_cell(raw)
    trans: list[Transition] = []
    prev = raw[::cell, ::cell] if cell > 1 else raw
    prev_level = int(getattr(frame, "levels_completed", 0) or 0)

    # EFFECT-SEEKING collection, mirroring what the live StepwiseExplorer actually does:
    # the agent does not flail at random, it hunts for actions that CHANGE the grid, because
    # inert transitions carry no inductive signal. We keep every changing transition and admit
    # inert ones only up to INERT_CAP, so the corpus has the realistic mix (mostly-informative
    # with some no-ops) rather than a degenerate all-no-op prompt.
    INERT_CAP = 6
    inert = 0
    attempts = 0
    h_px, w_px = raw.shape
    while len(trans) < N_TRANS and attempts < N_TRANS * 40:
        attempts += 1
        # Keyboard/directional actions only (data=None). ACTION6 clicks were tried and
        # REJECTED for this corpus: arcengine's GameAction is an enum SINGLETON, so
        # `ACTION6.set_data(...)` mutates shared global state that leaks into subsequent
        # steps -- it cost two games (lp85 degenerate, cn04 TypeError) while leaving prompt
        # length essentially unchanged. Not worth the contamination for a throughput corpus.
        aid, data = int(rng.integers(1, 6)), None
        try:
            frame = env.step(_action(aid, data))
        except Exception:
            break
        g = grid_of(frame)
        nxt = g[::cell, ::cell] if cell > 1 else g
        if nxt.shape != prev.shape:
            prev = nxt
            continue
        changed = not np.array_equal(prev, nxt)
        if not changed:
            if inert >= INERT_CAP:
                continue
            inert += 1
        lvl = int(getattr(frame, "levels_completed", 0) or 0)
        trans.append(Transition(prev.copy(), aid, data, nxt.copy(), prev_level, lvl))
        prev, prev_level = nxt, lvl

    return (trans, cell) if len(trans) >= K else None


def _action(aid: int, data=None):
    from arcengine.enums import GameAction

    a = getattr(GameAction, f"ACTION{aid}")
    if data is not None:
        a.set_data(data)
    return a


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = []
    for game in GAMES:
        try:
            got = collect(game, rng)
        except Exception as exc:  # a game that will not step is skipped, and SAID so
            print(f"SKIP {game}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if got is None:
            print(f"SKIP {game}: too few usable transitions", file=sys.stderr)
            continue
        trans, cell = got
        prompt = induce_prompt(game, trans, cell=cell, k=K)
        # Persist the transitions so the QUALITY arm can GRADE a generated engine against
        # HELD-OUT transitions (indices K.. are never shown in the prompt). Without this the
        # f16-vs-q8_0 comparison could only eyeball the text, which is not a quality measurement.
        np.savez_compressed(
            f"{OUT.rsplit('/', 1)[0]}/trans_{game}.npz",
            grids=np.stack([t.grid for t in trans]),
            next_grids=np.stack([t.next_grid for t in trans]),
            actions=np.array([t.action for t in trans]),
        )
        h, w = trans[0].grid.shape
        n_changed = sum(int((t.grid != t.next_grid).sum()) for t in trans)
        out.append(
            {
                "game": game,
                "grid_h": int(h),
                "grid_w": int(w),
                "cell": int(cell),
                "n_transitions": len(trans),
                "total_changed_cells": n_changed,
                "chars": len(prompt),
                "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt": prompt,
            }
        )
        print(f"OK {game}: {h}x{w} cell={cell} n={len(trans)} chars={len(prompt)}", file=sys.stderr)

    corpus_sha = hashlib.sha256(
        "".join(p["sha256"] for p in sorted(out, key=lambda d: d["game"])).encode()
    ).hexdigest()
    json.dump(
        {"seed": SEED, "k": K, "n_trans": N_TRANS, "corpus_sha256": corpus_sha, "prompts": out},
        open(OUT, "w"),
    )
    print(json.dumps({"n_prompts": len(out), "corpus_sha256": corpus_sha, "path": OUT}))


if __name__ == "__main__":
    main()

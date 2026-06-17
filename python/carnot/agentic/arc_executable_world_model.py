"""E3 — Carnot Executable-World-Model solver for ARC-AGI-3 (induce -> VERIFY ->
refactor -> plan), after arXiv:2605.05138 "Executable World Models for ARC-AGI-3 in
the Era of Coding Agents" (GPT-5.5 fully solves 15/25 games).

The paper's own thesis IS Carnot's: "LLMs are most reliable when used not as final
authorities, but as PROPOSAL mechanisms inside systems that can check their outputs."
So the LLM (codex / gpt-5.5) PROPOSES an executable Python world model; CARNOT'S
VERIFIER grounds it by checking the model reproduces the game's real offline
transitions, and HALTS planning the instant the model's prediction diverges from the
environment. The verifier is the moat; the LLM is the (swappable) proposer.

Loop:
  1. collect_transitions(game)      -- gather (grid, action, data, next_grid) offline (zero quota)
  2. proposer.induce(...)           -- codex writes results/arc_e3/<game>/world_model.py
                                       with engine(grid, action, data)->grid + is_level_complete(grid)
  3. WorldModelVerifier.score(...)  -- % of transitions the engine reproduces exactly;
                                       returns the failing transitions as mismatch artifacts
  4. proposer.refactor(...)         -- feed the mismatches back; codex fixes/simplifies (MDL proxy)
  5. plan_and_execute(...)          -- plan to is_level_complete INSIDE the verified model,
                                       execute in the real env, halt on any predicted!=observed divergence

This module is representation-careful: ARC-AGI-3 frames are 64x64 pixel renders of a
coarser LOGICAL grid. We detect the logical cell size and run the whole pipeline at
LOGICAL resolution (the paper's "settled ASCII frame"), so the induced model reasons
about game cells, not pixels.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[3]
E3_DIR = REPO / "results" / "arc_e3"

# ---------------------------------------------------------------------------
# logical-resolution helpers (the "settled ASCII frame")
# ---------------------------------------------------------------------------

def detect_cell(grid: np.ndarray) -> int:
    """Largest c in {8,4,2,1} (divisor of 64) for which the grid is EXACTLY constant
    within every c x c block -> the logical cell size. Lossless downsample factor."""
    h, w = grid.shape
    for c in (8, 4, 2):
        if h % c or w % c:
            continue
        blocks = grid.reshape(h // c, c, w // c, c)
        # constant within each block iff min == max over the (c,c) axes
        if np.array_equal(blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))):
            return c
    return 1


def to_logical(grid: np.ndarray, cell: int) -> np.ndarray:
    h, w = grid.shape
    return grid[:: cell, :: cell] if cell > 1 else grid


def to_ascii(logical: np.ndarray) -> str:
    """Compact one-char-per-cell ASCII (single trailing digit of the color)."""
    return "\n".join("".join(str(int(v))[-1] for v in row) for row in logical)


# ---------------------------------------------------------------------------
# transition collection (zero quota — offline sim)
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    grid: np.ndarray            # logical grid BEFORE
    action: int
    data: Optional[dict]
    next_grid: np.ndarray       # logical grid AFTER
    level_before: int
    level_after: int


def collect_transitions(game: str, n: int = 120, warmup: bool = False,
                        seed: int = 0) -> tuple[list[Transition], int]:
    """Explore the offline sim and record logical-resolution transitions. Uses the
    salience-ordered candidate generator so the dataset covers meaningful actions, not
    just raster order. Returns (transitions, cell_size)."""
    import random
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_agi3_live_adapter import _levels_completed, _game_over, _game_action
    from carnot.agentic.arc_graph_explore import rich_action_candidates, _warm
    from carnot.agentic import arc_solver_kit as kit

    rng = random.Random(seed)
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _warm(env, warmup)
    cell = detect_cell(grid_of(f))
    trans: list[Transition] = []
    restarts = 0
    while len(trans) < n and restarts < n:
        cands = rich_action_candidates(f)
        if not cands:
            f = _warm(env, warmup); restarts += 1; continue
        c = cands[rng.randrange(min(len(cands), 8))]   # bias to salient, keep some variety
        g0 = to_logical(grid_of(f), cell)
        l0 = _levels_completed(f)
        nf = env.step(_game_action(GameAction, c.action_id), data=c.data)
        if nf is None:
            f = _warm(env, warmup); restarts += 1; continue
        g1 = to_logical(grid_of(nf), cell)
        l1 = _levels_completed(nf)
        trans.append(Transition(g0, int(c.action_id), c.data, g1, l0, l1))
        if _game_over(nf) or l1 > l0:
            f = _warm(env, warmup); restarts += 1
        else:
            f = nf
    return trans, cell


# ---------------------------------------------------------------------------
# THE CARNOT VERIFIER — grounds the LLM-induced model against reality
# ---------------------------------------------------------------------------

@dataclass
class VerifyResult:
    n: int
    n_correct: int
    accuracy: float
    mismatches: list[dict] = field(default_factory=list)
    error: Optional[str] = None


class WorldModelVerifier:
    """Checks that an induced engine(grid, action, data) -> grid reproduces the real
    recorded transitions. This is the verification that makes the LLM accountable: a
    proposed model only earns trust by predicting transitions it was NOT hand-fit to.
    Returns mismatch artifacts (the failing transitions) for the refactor step."""

    def __init__(self, transitions: list[Transition]) -> None:
        self.transitions = transitions

    def score(self, engine: Callable[[np.ndarray, int, Optional[dict]], np.ndarray],
              max_mismatch: int = 8) -> VerifyResult:
        n_correct, mism = 0, []
        for i, t in enumerate(self.transitions):
            try:
                pred = np.asarray(engine(t.grid.copy(), t.action, t.data))
            except Exception as e:  # a crashing engine fails the transition
                if len(mism) < max_mismatch:
                    mism.append({"i": i, "action": t.action, "error": repr(e)[:160]})
                continue
            if pred.shape == t.next_grid.shape and np.array_equal(pred, t.next_grid):
                n_correct += 1
            elif len(mism) < max_mismatch:
                mism.append({"i": i, "action": t.action, "data": t.data,
                             "before": to_ascii(t.grid), "predicted": to_ascii(pred)
                             if pred.shape == t.next_grid.shape else f"shape{pred.shape}",
                             "observed": to_ascii(t.next_grid)})
        n = len(self.transitions)
        return VerifyResult(n, n_correct, n_correct / max(1, n), mism)


def load_engine(game: str):
    """Import the codex-written world_model.py for a game and return (engine,
    is_level_complete). Re-imports fresh each call so a refactor is picked up."""
    import importlib.util
    p = E3_DIR / game / "world_model.py"
    if not p.exists():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location(f"arc_wm_{game}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return getattr(mod, "engine"), getattr(mod, "is_level_complete", None)


# ---------------------------------------------------------------------------
# the proposer — codex / gpt-5.5 writes the executable model (swappable)
# ---------------------------------------------------------------------------

CODEX_BIN = "codex"


def _codex(prompt: str, timeout: int = 420) -> tuple[bool, str]:
    """Invoke codex/gpt-5.5 (the orchestrate.py form) with the prompt on stdin. The
    agent edits files in the repo directly. Returns (ok, tail_of_output)."""
    cmd = [CODEX_BIN, "exec", "--dangerously-bypass-approvals-and-sandbox",
           "--color", "never", "--cd", str(REPO), "--model", "gpt-5.5", "-"]
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, (p.stdout or "")[-2000:] + (p.stderr or "")[-500:]
    except subprocess.TimeoutExpired:
        return False, f"codex timeout after {timeout}s"


def _transitions_block(trans: list[Transition], k: int = 10) -> str:
    """Render a sample of transitions as ascii (before -> action -> after) for the
    prompt. Prefer grid-CHANGING transitions (they carry the dynamics) but keep a
    couple of no-ops so the model learns which actions do nothing in which states."""
    changed = [t for t in trans if not np.array_equal(t.grid, t.next_grid)]
    noop = [t for t in trans if np.array_equal(t.grid, t.next_grid)]
    sample = changed[: k - 2] + noop[:2]
    out = []
    for t in sample:
        click = f" data={t.data}" if t.data else ""
        out.append(f"--- ACTION{t.action}{click}  (level {t.level_before}->{t.level_after})\n"
                   f"BEFORE:\n{to_ascii(t.grid)}\nAFTER:\n{to_ascii(t.next_grid)}")
    return "\n".join(out)


def induce_prompt(game: str, trans: list[Transition], cell: int) -> str:
    h, w = trans[0].grid.shape
    colors = sorted(set(int(v) for t in trans for v in t.grid.flatten().tolist()))
    return f"""You are inducing an EXECUTABLE WORLD MODEL for the ARC-AGI-3 game '{game}'.

The game state is a {h}x{w} integer grid (logical resolution; colors {colors}). You are
given REAL observed transitions: a BEFORE grid, an action, and the resulting AFTER grid.
Actions are integers 1-7; ACTION6 is a click with data={{'x':px,'y':py}} in PIXEL coords
(pixel = logical*{cell}). Other actions are keyboard/directional with data=None.

Write a Python file at results/arc_e3/{game}/world_model.py with EXACTLY two functions:

    import numpy as np
    def engine(grid, action, data):
        # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
        ...
    def is_level_complete(grid):
        # return True if `grid` is a level-complete / win state, else False.
        ...

Induce the transition RULES from the observed data (movement, gravity, toggling,
pushing, collection, etc.). Prefer SIMPLE GENERAL rules over per-frame special cases.
Use only numpy + stdlib. Do not read files or network. Make engine() pure and
deterministic. Write ONLY that one file.

OBSERVED TRANSITIONS:
{_transitions_block(trans)}
"""


def refactor_prompt(game: str, vr: VerifyResult) -> str:
    mism = json.dumps(vr.mismatches[:5], indent=1)[:4000]
    return f"""The executable world model at results/arc_e3/{game}/world_model.py reproduces
only {vr.n_correct}/{vr.n} ({vr.accuracy:.0%}) of the observed transitions. Below are
failing cases (BEFORE / your PREDICTED / the true OBSERVED next grid). Fix engine() so it
reproduces these too, and REFACTOR toward simpler, more general rules (replace special
cases with shared rules) while keeping the cases it already gets right. Edit only that
file.

MISMATCHES:
{mism}
"""


@dataclass
class CodexProposer:
    timeout: int = 420

    def induce(self, game: str, trans: list[Transition], cell: int) -> tuple[bool, str]:
        (E3_DIR / game).mkdir(parents=True, exist_ok=True)
        return _codex(induce_prompt(game, trans, cell), self.timeout)

    def refactor(self, game: str, vr: VerifyResult) -> tuple[bool, str]:
        return _codex(refactor_prompt(game, vr), self.timeout)


# ---------------------------------------------------------------------------
# plan in the verified model, execute in reality, halt on divergence
# ---------------------------------------------------------------------------

def plan_and_execute(game: str, engine, is_level_complete, *, warmup: bool = False,
                     max_plan: int = 200, max_depth: int = 40) -> dict:
    """BFS to an is_level_complete state INSIDE the induced model, then execute the plan
    in the REAL env step-by-step, halting the instant predicted != observed (the
    verifier-grounded safety the paper emphasizes). Returns an outcome dict."""
    from collections import deque
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_agi3_live_adapter import _levels_completed, _game_action
    from carnot.agentic.arc_graph_explore import rich_action_candidates, _warm, to_logical as _tl  # noqa
    from carnot.agentic import arc_solver_kit as kit

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _warm(env, warmup)
    cell = detect_cell(grid_of(f))
    start = to_logical(grid_of(f), cell)
    start_level = _levels_completed(f)

    # plan inside the model
    seen = {to_ascii(start)}
    frontier = deque([(start, [])])
    plan = None
    expansions = 0
    while frontier and expansions < max_plan and plan is None:
        g, path = frontier.popleft()
        if len(path) >= max_depth:
            continue
        for c in rich_action_candidates(f)[:12]:        # candidate actions at logical state
            try:
                ng = np.asarray(engine(g.copy(), int(c.action_id), c.data))
            except Exception:
                continue
            expansions += 1
            key = to_ascii(ng) if ng.ndim == 2 else None
            if key is None or key in seen:
                continue
            seen.add(key)
            npath = path + [{"action": int(c.action_id), "data": c.data}]
            if is_level_complete is not None:
                try:
                    if bool(is_level_complete(ng)):
                        plan = npath; break
                except Exception:
                    pass
            frontier.append((ng, npath))
    if plan is None:
        return {"game": game, "planned": False, "reason": "no plan to is_level_complete in model"}

    # execute in reality, halting on model/observation divergence
    f = _warm(env, warmup)
    gp = to_logical(grid_of(f), cell)
    for step in plan:
        pred = np.asarray(engine(gp.copy(), step["action"], step["data"]))
        nf = env.step(_game_action(GameAction, step["action"]), data=step["data"])
        if nf is None:
            return {"game": game, "planned": True, "executed": False, "reason": "env returned None"}
        obs = to_logical(grid_of(nf), cell)
        if _levels_completed(nf) > start_level:
            return {"game": game, "planned": True, "executed": True, "level_up": True,
                    "plan_len": len(plan)}
        if pred.shape != obs.shape or not np.array_equal(pred, obs):
            return {"game": game, "planned": True, "executed": False, "divergence_step": step,
                    "reason": "model prediction diverged from observation — halted (verifier-grounded)"}
        gp = obs
    return {"game": game, "planned": True, "executed": True, "level_up": False,
            "reason": "plan executed but no level-up — model goal predicate imperfect"}

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
                ok_shape = pred.shape == t.next_grid.shape
                # COMPACT mismatch (deltas, not full grids — fits a local model's context):
                # what the TRUE action did vs where the engine's prediction was wrong.
                mism.append({"i": i, "action": t.action, "data": t.data,
                             "true_change": _delta(t.grid, t.next_grid),
                             "your_prediction_was_wrong_at": (_delta(pred, t.next_grid)
                                                              if ok_shape else f"wrong shape {pred.shape}")})
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


def _delta(g0: np.ndarray, g1: np.ndarray, cap: int = 80) -> list:
    """Changed cells (row, col, from, to) — a COMPACT transition encoding that fits a
    local model's small context (full 64x64 before/after grids blow it; a few-cell delta
    is tiny and is arguably MORE learnable: it shows exactly what the action changed)."""
    g0 = np.asarray(g0); g1 = np.asarray(g1)
    if g0.shape != g1.shape:
        return []
    diff = np.argwhere(g0 != g1)
    return [(int(r), int(c), int(g0[r, c]), int(g1[r, c])) for r, c in diff[:cap]]


def _transitions_block(trans: list[Transition], k: int = 8) -> str:
    """Compact transition encoding for the induce prompt: ONE full grid (the layout) +
    per-transition DELTAS (changed cells), + the full WIN state if observed. Prefers
    grid-CHANGING transitions; keeps a couple of no-ops. Small enough for a local model's
    context window (the full-grid form overflowed gemma-4-12B at ~67k tokens)."""
    changed = [t for t in trans if not np.array_equal(t.grid, t.next_grid)]
    noop = [t for t in trans if np.array_equal(t.grid, t.next_grid)]
    sample = changed[: k - 2] + noop[:2]
    out = []
    if sample:
        out.append("INITIAL GRID (one full example of the state layout; all grids are this shape):\n"
                   + to_ascii(sample[0].grid))
    for t in sample:
        click = f" data={t.data}" if t.data else ""
        out.append(f"--- ACTION{t.action}{click} (level {t.level_before}->{t.level_after}): "
                   f"changed cells (row,col,from,to) = {_delta(t.grid, t.next_grid)}")
    win = next((t for t in trans if t.level_after > t.level_before), None)
    if win is not None:
        out.append("WIN STATE (full grid of a level-complete state — is_level_complete must return True here):\n"
                   + to_ascii(win.next_grid))
    return "\n".join(out)


def induce_prompt(game: str, trans: list[Transition], cell: int) -> str:
    h, w = trans[0].grid.shape
    colors = sorted(set(int(v) for t in trans for v in t.grid.flatten().tolist()))
    return f"""You are inducing an EXECUTABLE WORLD MODEL for the ARC-AGI-3 game '{game}'.

The game state is a {h}x{w} integer grid (logical resolution; colors {colors}). You are
given REAL observed transitions COMPACTLY: one full INITIAL grid (the layout), then per
transition the action and its DELTA = the list of changed cells (row, col, from_value,
to_value). Apply a transition's delta to the prior grid to get the next grid. A full WIN
STATE grid is shown if a level was completed. Actions are integers 1-7; ACTION6 is a click
with data={{'x':px,'y':py}} in PIXEL coords (pixel = logical*{cell}); others are
keyboard/directional with data=None.

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
    """DEV-ONLY proposer (codex/gpt-5.5). Requires INTERNET, so it is NOT legal in the
    OFFLINE competition eval — use it only to validate the E3 loop during development.
    For the competition, use LocalGGUFProposer (open-weight, offline)."""
    timeout: int = 420
    offline_legal: bool = False

    def induce(self, game: str, trans: list[Transition], cell: int) -> tuple[bool, str]:
        (E3_DIR / game).mkdir(parents=True, exist_ok=True)
        return _codex(induce_prompt(game, trans, cell), self.timeout)

    def refactor(self, game: str, vr: VerifyResult) -> tuple[bool, str]:
        return _codex(refactor_prompt(game, vr), self.timeout)


def _resolve_gguf(repo_substr: str) -> Optional[str]:
    """Find a cached GGUF weight file for an open-weight SOTA model (offline)."""
    import glob
    base = Path.home() / ".cache" / "huggingface" / "hub"
    for d in base.glob(f"models--*{repo_substr}*GGUF"):
        hits = sorted(d.glob("snapshots/*/*.gguf"))
        if hits:
            return str(hits[0])
    return None


# The CUDA-built llama.cpp binary (the venv's llama-cpp-python is CPU-ONLY:
# llama_supports_gpu_offload()==False). SOTA open-weight models MUST run on GPU —
# CPU is excruciatingly slow AND pegs ~20 cores, fighting the conductor (operator
# directive 2026-06-17). This binary offloads all layers to the 2x3090s.
LLAMA_CLI = Path.home() / ".cache" / "llama.cpp-master" / "build" / "bin" / "llama-cli"


@dataclass
class LocalGGUFProposer:
    """OFFLINE-LEGAL, DECENTRALIZED, GPU-ENFORCED proposer (CLAUDE.md decentralization
    rule 1-2 + the always-GPU directive): an OPEN-WEIGHT local model induces the world
    model with NO internet, so it runs inside the competition's offline eval sandbox. The
    induced code quality is GROUNDED by the Carnot WorldModelVerifier regardless of model
    strength — a weaker local model just earns a lower verifier score, honestly.

    GPU-ENFORCED: uses the CUDA-built llama.cpp binary (-ngl 999 = all layers on GPU). If
    the GPU binary is missing it FAILS LOUD rather than silently falling back to CPU (the
    CPU path is excruciatingly slow and a 20-core conductor-fight). NOT TRM and NOT a
    closed model: an open local LLM (a TRM-class trained model is the other local engine)."""
    repo_substr: str = "gemma-4-12B-it"     # lightweight SOTA: fast on GPU for per-game induction
    n_ctx: int = 16384                       # digit-dense grids tokenize ~1 char/token; 8192 overflowed
    max_tokens: int = 2048
    timeout: int = 300
    offline_legal: bool = True

    def _gen_to_file(self, game: str, prompt: str) -> tuple[bool, str]:
        (E3_DIR / game).mkdir(parents=True, exist_ok=True)
        path = _resolve_gguf(self.repo_substr)
        if not path:
            return False, f"no cached GGUF matching {self.repo_substr}"
        if not LLAMA_CLI.exists():            # GPU enforcement: no silent CPU fallback
            return False, (f"GPU llama.cpp binary missing at {LLAMA_CLI}; SOTA models must "
                           "run on GPU (the venv llama-cpp-python is CPU-only)")
        cmd = [str(LLAMA_CLI), "-m", path, "-ngl", "999", "-no-cnv", "--no-display-prompt",
               "-c", str(self.n_ctx), "-n", str(self.max_tokens), "--temp", "0.2", "-p", prompt]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return False, f"local gguf (GPU) timeout after {self.timeout}s"
        code = _extract_python(p.stdout)
        if not code or "def engine" not in code:
            return False, f"local model produced no usable engine() code (tail: {p.stdout[-200:]!r})"
        (E3_DIR / game / "world_model.py").write_text(code)
        return True, "local gguf (GPU) wrote world_model.py"

    def induce(self, game: str, trans: list[Transition], cell: int) -> tuple[bool, str]:
        return self._gen_to_file(game, induce_prompt(game, trans, cell) +
                                 "\n\nReturn ONLY one ```python code block with engine + is_level_complete.\n```python\n")

    def refactor(self, game: str, vr: VerifyResult) -> tuple[bool, str]:
        return self._gen_to_file(game, refactor_prompt(game, vr) +
                                 "\n\nReturn ONLY the corrected ```python file.\n```python\n")


def _extract_python(text: str) -> str:
    """Pull the first python code block (or the whole text if it looks like code)."""
    if "```python" in text:
        text = text.split("```python", 1)[1]
    if "```" in text:
        text = text.split("```", 1)[0]
    return text.strip()


# ---------------------------------------------------------------------------
# plan in the verified model, execute in reality, halt on divergence
# ---------------------------------------------------------------------------

def _model_candidates(grid: np.ndarray) -> list[dict]:
    """Action candidates to try when planning INSIDE the induced model (no env): the 5
    directional/confirm keyboard actions + a click on each detected object (salience-
    ordered). Pure-grid, so it works on the engine's predicted grids."""
    from carnot.agentic.arc_graph_explore import _components_detailed
    cands = [{"action": a, "data": None} for a in (1, 2, 3, 4, 5)]
    comps = _components_detailed(grid)
    if comps:
        from collections import Counter
        cc = Counter(int(v) for v in grid.flatten().tolist())
        comps.sort(key=lambda c: c[2] * (1.0 + 1.0 / (1 + cc.get(c[3], 0))), reverse=True)
        for (cy, cx, _a, _c) in comps[:32]:
            cands.append({"action": 6, "data": {"x": int(cx), "y": int(cy)}})
    return cands


def plan_in_model(engine, is_level_complete, start_grid: np.ndarray, *,
                  max_nodes: int = 20000, max_depth: int = 40) -> Optional[list]:
    """BFS a path to an is_level_complete state ENTIRELY INSIDE the induced model
    (engine is pure: grid,action,data -> grid; no environment). Returns the action
    sequence [{"action","data"}] that the model believes reaches a win, or None. This
    is the harness-friendly planner: the agent computes the plan with zero real actions,
    then executes it in the real env (few real actions = the EFFICIENCY win), halting if
    reality diverges from the model."""
    from collections import deque
    if is_level_complete is None:
        return None
    start = np.asarray(start_grid)
    seen = {to_ascii(start)}
    q = deque([(start, [])])
    nodes = 0
    while q and nodes < max_nodes:
        grid, path = q.popleft()
        if len(path) >= max_depth:
            continue
        for c in _model_candidates(grid):
            try:
                ng = np.asarray(engine(grid.copy(), c["action"], c["data"]))
            except Exception:
                continue
            nodes += 1
            if ng.shape != start.shape:
                continue
            key = to_ascii(ng)
            if key in seen:
                continue
            seen.add(key)
            npath = path + [c]
            try:
                if bool(is_level_complete(ng)):
                    return npath
            except Exception:
                pass
            q.append((ng, npath))
    return None


def plan_and_execute(game: str, engine, is_level_complete, *, warmup: bool = False,
                     max_plan: int = 200, max_depth: int = 40) -> dict:
    """BFS to an is_level_complete state INSIDE the induced model, then execute the plan
    in the REAL env step-by-step, halting the instant predicted != observed (the
    verifier-grounded safety the paper emphasizes). Returns an outcome dict."""
    from collections import deque
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_agi3_live_adapter import _levels_completed, _game_action
    from carnot.agentic.arc_graph_explore import rich_action_candidates, _warm
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

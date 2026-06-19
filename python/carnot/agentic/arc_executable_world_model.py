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
import time
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
    return grid[::cell, ::cell] if cell > 1 else grid


def to_ascii(logical: np.ndarray) -> str:
    """Compact one-char-per-cell ASCII (single trailing digit of the color)."""
    return "\n".join("".join(str(int(v))[-1] for v in row) for row in logical)


# ---------------------------------------------------------------------------
# transition collection (zero quota — offline sim)
# ---------------------------------------------------------------------------


@dataclass
class Transition:
    grid: np.ndarray  # logical grid BEFORE
    action: int
    data: Optional[dict]
    next_grid: np.ndarray  # logical grid AFTER
    level_before: int
    level_after: int


def collect_transitions(
    game: str, n: int = 120, warmup: bool = False, seed: int = 0
) -> tuple[list[Transition], int]:
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
            f = _warm(env, warmup)
            restarts += 1
            continue
        c = cands[rng.randrange(min(len(cands), 8))]  # bias to salient, keep some variety
        g0 = to_logical(grid_of(f), cell)
        l0 = _levels_completed(f)
        nf = env.step(_game_action(GameAction, c.action_id), data=c.data)
        if nf is None:
            f = _warm(env, warmup)
            restarts += 1
            continue
        g1 = to_logical(grid_of(nf), cell)
        l1 = _levels_completed(nf)
        trans.append(Transition(g0, int(c.action_id), c.data, g1, l0, l1))
        if _game_over(nf) or l1 > l0:
            f = _warm(env, warmup)
            restarts += 1
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

    def score(
        self, engine: Callable[[np.ndarray, int, Optional[dict]], np.ndarray], max_mismatch: int = 8
    ) -> VerifyResult:
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
                mism.append(
                    {
                        "i": i,
                        "action": t.action,
                        "data": t.data,
                        "true_change": _delta(t.grid, t.next_grid),
                        "your_prediction_was_wrong_at": (
                            _delta(pred, t.next_grid) if ok_shape else f"wrong shape {pred.shape}"
                        ),
                    }
                )
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
    cmd = [
        CODEX_BIN,
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--color",
        "never",
        "--cd",
        str(REPO),
        "--model",
        "gpt-5.5",
        "-",
    ]
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, (p.stdout or "")[-2000:] + (p.stderr or "")[-500:]
    except subprocess.TimeoutExpired:
        return False, f"codex timeout after {timeout}s"


def _delta(g0: np.ndarray, g1: np.ndarray, cap: int = 80) -> list:
    """Changed cells (row, col, from, to) — a COMPACT transition encoding that fits a
    local model's small context (full 64x64 before/after grids blow it; a few-cell delta
    is tiny and is arguably MORE learnable: it shows exactly what the action changed)."""
    g0 = np.asarray(g0)
    g1 = np.asarray(g1)
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
        out.append(
            "INITIAL GRID (one full example of the state layout; all grids are this shape):\n"
            + to_ascii(sample[0].grid)
        )
    for t in sample:
        click = f" data={t.data}" if t.data else ""
        out.append(
            f"--- ACTION{t.action}{click} (level {t.level_before}->{t.level_after}): "
            f"changed cells (row,col,from,to) = {_delta(t.grid, t.next_grid)}"
        )
    win = next((t for t in trans if t.level_after > t.level_before), None)
    if win is not None:
        out.append(
            "WIN STATE (full grid of a level-complete state — is_level_complete must return True here):\n"
            + to_ascii(win.next_grid)
        )
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


# llama.cpp SERVER, GPU-enforced. PREFER the HIP build (ROCm iGPU gfx1150 / Radeon 890M,
# ~108GB UNIFIED memory) — it does NOT contend with the conductor's CUDA experiments on the
# 2x3090s (operator directive 2026-06-17: iGPU is the outer-loop target, never CPU). Falls
# back to the CUDA build only if the iGPU build is absent. The venv llama-cpp-python is
# CPU-only; llama-cli/llama-completion hang/crash on gemma's chat template — the server's
# /completion does RAW completion (no chat template) and keeps the model loaded across calls.
def _resolve_llama_server() -> Path:
    # Kaggle/live submission: point CARNOT_LLAMA_SERVER at the bundled CUDA llama-server binary
    # (/kaggle/input/<llamacpp-dataset>/llama-server). MTP (--spec-type draft-mtp) lives in
    # libllama-common, which the BINARY links -- the stock llama-cpp-python wheel cannot do native MTP,
    # so the submission bundles this binary + its shared libs, NOT a wheel.
    import os

    env = os.environ.get("CARNOT_LLAMA_SERVER")
    if env:
        return Path(env)
    base = Path.home() / ".cache" / "llama.cpp-master"
    hip = base / "build-hip" / "bin" / "llama-server"  # ROCm iGPU — no conductor contention
    return hip if hip.exists() else base / "build" / "bin" / "llama-server"  # CUDA 3090 fallback


LLAMA_SERVER = _resolve_llama_server()

# Qwen3.5-9B-MTP loads ~11.5GB on a 3090 (weights + MTP self-draft + q8 KV, validated 2026-06-19).
# Require headroom above that so the opt-in 3090 path NEVER binds a card a conductor training job is
# using -- this is the "yield-if-the-conductor-needs-it" guard.
_GENERATOR_CUDA_MIN_FREE_MB = 13000


def _cuda_gpu_free_mb(idx: int) -> int:
    """Free VRAM (MiB) on CUDA GPU `idx` via nvidia-smi; -1 if unavailable. The guard input for the
    opt-in 3090 generator path."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return int(lines[idx]) if 0 <= idx < len(lines) else -1
    except Exception:
        return -1


def _generator_server_and_env() -> tuple[Path, Optional[dict]]:
    """Resolve the llama-server binary + launch env for the generator, evaluated at LAUNCH time so the
    3090 guard sees current GPU state.

    Priority:
      1. CARNOT_LLAMA_SERVER (Kaggle/live bundled CUDA binary) -- unchanged; inherits ambient env.
      2. OPT-IN CARNOT_ARC_GENERATOR_CUDA_GPU=<idx> -> the local CUDA build pinned to that 3090 via
         CUDA_VISIBLE_DEVICES, but ONLY if the card has >=_GENERATOR_CUDA_MIN_FREE_MB free. This is the
         operator-approved (2026-06-19) use of one idle 3090 for generator throughput now that the TRM
         run is retired; the free-memory guard yields to any conductor job already on the card.
      3. Default: the iGPU HIP build (no conductor contention), else the CUDA build.
    Returns (server_path, env_or_None); env=None means inherit the ambient environment (legacy behavior).
    """
    import os

    explicit = os.environ.get("CARNOT_LLAMA_SERVER")
    if explicit:
        return Path(explicit), None
    base = Path.home() / ".cache" / "llama.cpp-master"
    cuda = base / "build" / "bin" / "llama-server"
    hip = base / "build-hip" / "bin" / "llama-server"
    gpu = (os.environ.get("CARNOT_ARC_GENERATOR_CUDA_GPU") or "").strip()
    if gpu and cuda.exists():
        try:
            idx = int(gpu)
        except ValueError:
            idx = -1
        if idx >= 0 and _cuda_gpu_free_mb(idx) >= _GENERATOR_CUDA_MIN_FREE_MB:
            return cuda, dict(os.environ, CUDA_VISIBLE_DEVICES=str(idx))
        # guard tripped (card busy / unavailable / bad idx) -> fall through to the iGPU path,
        # never fight the conductor for the 3090.
    return (hip if hip.exists() else cuda), None


@dataclass
class LocalGGUFProposer:
    """OFFLINE-LEGAL, DECENTRALIZED, GPU-ENFORCED proposer (CLAUDE.md decentralization
    rule 1-2 + the always-GPU directive): an OPEN-WEIGHT local model induces the world
    model with NO internet, so it runs inside the competition's offline eval sandbox. The
    induced code quality is GROUNDED by the Carnot WorldModelVerifier regardless of model
    strength — a weaker local model just earns a lower verifier score, honestly.

    GPU-ENFORCED via a CUDA llama-server (-ngl 999 = all layers on GPU); FAILS LOUD if the
    server can't start — never a silent CPU fallback (the CPU path is excruciatingly slow
    and a 20-core conductor-fight). The model stays loaded across induce/refactor calls.
    NOT TRM and NOT a closed model: an open local LLM (a TRM-class trained model is the
    other local engine)."""

    repo_substr: str = "gemma-4-12B-it"  # lightweight SOTA: fast on GPU for per-game induction
    n_ctx: int = 16384  # digit-dense grids tokenize ~1 char/token; 8192 overflowed
    max_tokens: int = 4096  # a full world-model engine needs >2048 (it truncated mid-code)
    timeout: int = 300
    port: int = 8919
    offline_legal: bool = True
    # Live-submission deploy config (all OPT-IN; defaults preserve legacy behavior). Validated 2026-06-19:
    # Qwen3.5-9B-MTP is the selected ARC live generator (62.5% Layer-B grounding vs DeepSeek-Flash 25%,
    # ~13 tok/s with MTP, 5.9GB Q4 fits 16GB). See docs/research-notes/arc-16gb-model-alternatives-2026-06-18.md.
    mtp: bool = False  # --spec-type draft-mtp (self-draft via the -MTP- GGUF's nextn heads)
    kv_quant: Optional[str] = (
        None  # e.g. "q8_0" -> --cache-type-k/v q8_0 (halves KV, near-lossless)
    )
    no_think_prefix: str = ""  # e.g. "/no_think\n" -> suppress hybrid-thinking CoT (Qwen3)
    model_path: Optional[str] = (
        None  # explicit .gguf path; on Kaggle set to the bundled /kaggle/input/... path
    )
    _proc: Any = None

    def _url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _healthy(self) -> bool:
        import urllib.request

        try:
            with urllib.request.urlopen(self._url() + "/health", timeout=2) as r:
                return b"ok" in r.read()
        except Exception:
            return False

    def _ensure_server(self) -> bool:
        if self._healthy():
            return True  # reuse an already-running server (loaded model)
        path = self.model_path or _resolve_gguf(
            self.repo_substr
        )  # explicit path (Kaggle bundle) else cache
        # Resolve the server + launch env at LAUNCH time so the opt-in 3090 guard sees current GPU state
        # (CARNOT_ARC_GENERATOR_CUDA_GPU=<idx> -> CUDA build pinned to that card iff it has headroom).
        server, launch_env = _generator_server_and_env()
        if not path or not server.exists():
            return False  # GPU enforcement: no CPU fallback
        args = [
            str(server),
            "-m",
            path,
            "-ngl",
            "999",
            "-c",
            str(self.n_ctx),
            "--port",
            str(self.port),
            "--host",
            "127.0.0.1",
        ]
        if self.mtp:  # native llama.cpp MTP speculative decoding (self-draft)
            args += ["--spec-type", "draft-mtp", "--model-draft", path]
        if self.kv_quant:  # 8-bit KV cache doubles usable context, near-lossless
            args += ["--cache-type-k", self.kv_quant, "--cache-type-v", self.kv_quant]
        # env=launch_env: None inherits the ambient env (legacy iGPU path); a dict pins CUDA_VISIBLE_DEVICES.
        self._proc = subprocess.Popen(
            args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=launch_env
        )
        for _ in range(90):  # model load on GPU can take ~10-30s
            if self._healthy():
                return True
            time.sleep(2)
        return False

    def generate(
        self,
        prompt: str,
        required: tuple = ("engine", "is_level_complete"),
        validate=None,
        tries: int = 3,
    ) -> tuple[bool, str]:
        """Generic GPU-server completion: returns (True, code) where `code` contains every
        `def <name>` in `required`, PARSES, and (if `validate` is given) passes the
        runtime check `validate(code) -> bool`. Retries on the iGPU (fast). This is the
        gap-filler entry point: the LLM writes a FOCUSED component (a goal_distance
        heuristic, a state_key, a verifier invariant) — not a full solver. `validate` lets
        the caller reject runtime-buggy code (e.g. a heuristic that returns None)."""
        import ast
        import json as _json
        import urllib.request

        if not self._ensure_server():
            return False, (
                f"GPU llama-server failed for {self.repo_substr}; SOTA models "
                "must run on GPU (no CPU fallback)"
            )
        if self.no_think_prefix:  # suppress hybrid-thinking CoT so the model emits code directly
            prompt = self.no_think_prefix + prompt
        last = ""
        for attempt in range(tries):
            body = _json.dumps(
                {
                    "prompt": prompt,
                    "n_predict": self.max_tokens,
                    "temperature": 0.2 + 0.1 * attempt,
                    "cache_prompt": True,
                }
            ).encode()
            try:
                req = urllib.request.Request(
                    self._url() + "/completion",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    text = _json.load(r).get("content", "")
            except Exception as e:
                return False, f"local gguf (GPU server) failed: {e!r}"[:200]
            code = _extract_python(text)
            if not code or any(f"def {fn}" not in code for fn in required):
                last = f"missing {required} in output"
                continue
            try:
                ast.parse(code)  # never use code that doesn't parse
            except SyntaxError as se:
                last = f"syntax error line {se.lineno}: {se.msg}"
                continue
            if validate is not None:
                try:
                    if not validate(code):
                        last = "failed runtime validation (e.g. returned non-number)"
                        continue
                except Exception as ve:
                    last = f"runtime check raised: {ve!r}"[:120]
                    continue
            return True, code
        return False, f"local model code unusable after {tries} tries ({last})"

    def _gen_to_file(self, game: str, prompt: str) -> tuple[bool, str]:
        (E3_DIR / game).mkdir(parents=True, exist_ok=True)
        ok, code = self.generate(prompt, ("engine", "is_level_complete"))
        if ok:
            (E3_DIR / game / "world_model.py").write_text(code)
            return True, "local gguf (GPU server) wrote world_model.py"
        return False, code

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None

    def induce(self, game: str, trans: list[Transition], cell: int) -> tuple[bool, str]:
        return self._gen_to_file(
            game,
            induce_prompt(game, trans, cell)
            + "\n\nReturn ONLY one ```python code block with engine + is_level_complete.\n```python\n",
        )

    def refactor(self, game: str, vr: VerifyResult) -> tuple[bool, str]:
        return self._gen_to_file(
            game,
            refactor_prompt(game, vr)
            + "\n\nReturn ONLY the corrected ```python file.\n```python\n",
        )


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
        for cy, cx, _a, _c in comps[:32]:
            cands.append({"action": 6, "data": {"x": int(cx), "y": int(cy)}})
    return cands


def plan_in_model(
    engine,
    is_level_complete,
    start_grid: np.ndarray,
    *,
    max_nodes: int = 20000,
    max_depth: int = 40,
) -> Optional[list]:
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


def plan_and_execute(
    game: str,
    engine,
    is_level_complete,
    *,
    warmup: bool = False,
    max_plan: int = 200,
    max_depth: int = 40,
) -> dict:
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
        for c in rich_action_candidates(f)[:12]:  # candidate actions at logical state
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
                        plan = npath
                        break
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
            return {
                "game": game,
                "planned": True,
                "executed": True,
                "level_up": True,
                "plan_len": len(plan),
            }
        if pred.shape != obs.shape or not np.array_equal(pred, obs):
            return {
                "game": game,
                "planned": True,
                "executed": False,
                "divergence_step": step,
                "reason": "model prediction diverged from observation — halted (verifier-grounded)",
            }
        gp = obs
    return {
        "game": game,
        "planned": True,
        "executed": True,
        "level_up": False,
        "reason": "plan executed but no level-up — model goal predicate imperfect",
    }

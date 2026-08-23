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
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
# THE INDUCED-ENGINE STORE. `load_engine` READS from here and `LocalGGUFProposer._gen_to_file`
# / the refactor path WRITE here, unconditionally, on every successful induction.
#
# REQ-ARC-WMTE-6016 (2026-07-27): `CARNOT_ARC_E3_DIR` makes it redirectable WITHOUT a
# monkeypatch. An A/B that runs several arms against the same store is not running an A/B:
# arm A's inductions overwrite the engines arm B then starts from, so a cross-arm delta
# confounds the flag with starting-engine drift. That is not hypothetical -- a four-arm run
# on 2026-07-27 had rewritten 11 engines from its FIRST arm before it was stopped, including
# both engines named as the origin incident for GAP-WM-TRUST-GATE, and 15 of 75 rows of an
# already-published artifact stopped reproducing as a result (ft09's legacy_accuracy went
# 1.0000 -> 0.0000 because its 12-bare-`return grid` engine had been replaced in place).
#
# Module-global by design so the existing `e3.E3_DIR = ...` monkeypatch (see
# scripts/experiments/arc3_local_scaffold_induction_ab.py and
# tests/python/test_induce_split_fallback.py) keeps working: readers and writers dereference the
# module attribute, so REBINDING IT takes effect immediately. The env var is the same override for
# a subprocess that cannot patch the module.
#
# CORRECTION 2026-07-29 -- THE REDIRECT IS NOT TOTAL, and this comment used to claim it was.
# It previously read "both readers and writers resolve it at CALL time, not at import time". That
# is false about the ENV VAR: the expression below runs ONCE, at module import, so
# `CARNOT_ARC_E3_DIR` only redirects a process that set it BEFORE importing this module. A process
# that imports first and sets the variable afterwards keeps writing to the real evidence store --
# `results/arc_e3/<game>/world_model.py`, which is tracked and is read-only by discipline. The
# monkeypatch path IS effectively call-time (rebinding the attribute changes what every later
# dereference sees); only the env-var path is import-time. Two further writers bypass this constant
# altogether and hardcode `config.repo_root / "results" / "arc_e3" / <game>`, with
# `mkdir(parents=True)`: `arc_e3_named_tail_gate._write_skill_file` and
# `arc_e3_fidelity_gate` (both write `skill_*.json`, not `world_model.py`).
#
# NOT CHANGED HERE, deliberately. Making the env var call-time, or routing those two gates through
# this constant, alters where live code writes; the monkeypatch contract above is depended on by
# existing consumers I have not measured against a change. Recorded so the next reader does not
# re-derive it, and so nobody trusts `CARNOT_ARC_E3_DIR` as a total guarantee -- set it in the
# ENVIRONMENT BEFORE the interpreter starts, not from inside an already-imported process.
E3_DIR = (
    Path(os.environ["CARNOT_ARC_E3_DIR"])
    if os.environ.get("CARNOT_ARC_E3_DIR")
    else (REPO / "results" / "arc_e3")
)
# The tracked evidence store, named separately from `E3_DIR` so the guard below can tell "the
# default location" from "wherever this process was redirected to".
_TRACKED_E3_EVIDENCE_DIR = REPO / "results" / "arc_e3"


def _guard_engine_write(path: Path) -> None:
    """Refuse to overwrite the TRACKED evidence store from inside a test (2026-07-30).

    THE INCIDENT. `tests/python/test_codeonly_induce_scoping.py` drives `LocalGGUFProposer.induce`
    with a stubbed `urlopen`, and did not redirect `E3_DIR`. So running the suite wrote
    `results/arc_e3/g/world_model.py` -- a TRACKED file in a store this project treats as
    read-only evidence. It was caught only because the stubbed response happened to be
    byte-identical to the committed content, so `git status` stayed clean. A different stub body
    would have silently clobbered committed evidence, and the very next `git add -A` would have
    committed the clobber.

    WHY THE GUARD IS SCOPED TO PYTEST, and not to writes-in-general. The LIVE agent legitimately
    writes induced engines into this exact directory -- that is what the store is FOR -- so a
    blanket refusal would break production, which is why the 2026-07-29 note above declined to
    change write routing at all. But a write to the tracked store from inside a test is never
    legitimate: a test that needs an engine store needs its OWN. `PYTEST_CURRENT_TEST` is set by
    pytest for the duration of each test, so it identifies exactly that situation and nothing else.

    The escape hatch (`CARNOT_ARC_E3_ALLOW_EVIDENCE_WRITE=1`) exists because a test whose PURPOSE
    is to exercise the default-path write should be able to say so out loud.
    """
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        return
    if os.environ.get("CARNOT_ARC_E3_ALLOW_EVIDENCE_WRITE") == "1":
        return
    try:
        resolved = path.resolve()
        tracked = _TRACKED_E3_EVIDENCE_DIR.resolve()
    except OSError:  # pragma: no cover - resolve() on an unreadable parent
        return
    if resolved == tracked or tracked in resolved.parents:
        raise RuntimeError(
            f"refusing to write {resolved} from inside a test: results/arc_e3 is TRACKED, "
            f"read-only evidence. Redirect the store for this test -- set CARNOT_ARC_E3_DIR "
            f"before the interpreter starts, or monkeypatch "
            f"`carnot.agentic.arc_executable_world_model.E3_DIR` to a tmp_path. If the write is "
            f"deliberately exercising the default path, set "
            f"CARNOT_ARC_E3_ALLOW_EVIDENCE_WRITE=1."
        )


# REQ-ARC-WMTE-6690: per-attempt retention. The canonical `world_model.py` is keyed by game
# only, so every re-induction overwrites the previous model (measured: 40 attempts -> 25
# surviving files on the 2026-08-22 baseline run). Archiving each producer write makes the
# loss zero without changing what the canonical file holds or how anything reads it.
_ATTEMPT_ARCHIVE_ENV = "CARNOT_ARC_ENGINE_ATTEMPT_ARCHIVE"


def attempt_archive_enabled() -> bool:
    """REQ-ARC-WMTE-6690: archiving is ON unless explicitly disabled."""
    return os.environ.get(_ATTEMPT_ARCHIVE_ENV) != "0"


def _archive_engine_attempt(
    game: str, code: str, *, writer: str, model: str = "", note: str = ""
) -> dict:
    """Archive one produced engine under `E3_DIR/<game>/attempts/` (REQ-ARC-WMTE-6690).

    Writes a content-hash-named copy (deduplicated: the same source twice archives one file)
    and appends a manifest.jsonl line per attempt. `E3_DIR` is read here, at call time, so
    redirects work exactly like canonical writes (REQ-ARC-WMTE-6016).

    Failure direction, stated: the test-guard below FAILS CLOSED (a test reaching the tracked
    store must blow up loudly, same as the canonical write). Everything after it FAILS OPEN --
    a failed archive entry must not fail the induction, because losing one entry is strictly
    no worse than the shipped behaviour, which lost every entry. Failures stay visible in the
    returned dict, which callers record as `self.last_attempt_archive`.
    """
    info: dict = {"enabled": attempt_archive_enabled(), "archived": False, "deduplicated": False}
    if not info["enabled"]:
        return info
    adir = Path(E3_DIR) / game / "attempts"
    _guard_engine_write(adir)  # fail-closed: tests may not write the tracked store
    try:
        import hashlib
        from datetime import datetime, timezone

        raw = code.encode("utf-8", "replace")
        sha = hashlib.sha256(raw).hexdigest()[:16]
        adir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        dedup = bool(next(iter(adir.glob(f"wm_*__{sha}.py")), None))
        fname = f"wm_{stamp}__{sha}.py"
        if not dedup:
            (adir / fname).write_text(code)
        line = {
            "ts": stamp,
            "sha256_16": sha,
            "bytes": len(raw),
            "writer": writer,
            "model": model,
            "note": note[:200],
            "deduplicated": dedup,
            "file": None if dedup else fname,
        }
        with (adir / "manifest.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, sort_keys=True) + "\n")
        info.update({"archived": True, "deduplicated": dedup, "sha256_16": sha})
    except Exception as exc:  # noqa: BLE001 - fail-open past the guard, see docstring
        info["error"] = repr(exc)[:200]
    return info


def _archive_codex_engine(game: str) -> dict:
    """Post-hoc archive for the DEV-ONLY codex path (REQ-ARC-WMTE-6690).

    The codex CLI writes `world_model.py` itself, out-of-band, so the producer seam never
    sees the content -- read it back and archive it. Same fail-open direction."""
    try:
        code = (Path(E3_DIR) / game / "world_model.py").read_text()
    except OSError:
        return {
            "enabled": attempt_archive_enabled(),
            "archived": False,
            "error": "no_file_after_codex",
        }
    return _archive_engine_attempt(game, code, writer="codex")


# Pristine, READ-ONLY copies of the engines as they stood at the commit that named them the
# GAP-WM-TRUST-GATE origin incident. The mutable store above is rewritten by any induction
# run, so a test that asserts "the new gate rejects the real degenerate engines" must read
# from HERE or it silently becomes a test of whatever ran most recently.
E3_ORIGIN_FIXTURES_DIR = REPO / "results" / "arc_e3_origin_fixtures"

# ---------------------------------------------------------------------------
# logical-resolution helpers (the "settled ASCII frame")
# ---------------------------------------------------------------------------


def detect_cell(grid: np.ndarray) -> int:
    """Largest c in {8,4,2,1} (divisor of 64) for which the grid is EXACTLY constant
    within every c x c block -> the logical cell size. Lossless downsample factor."""
    if grid.ndim != 2:
        # A degenerate/malformed frame (e.g. a post-terminal empty sentinel,
        # shape (0,) -- the g50t apply_g50t_label failure class) has no
        # meaningful logical cell size. 1 (no downsampling) is the same safe
        # fallback already used below when no clean divisor is found; this
        # avoids `h, w = grid.shape` raising ValueError on every one of
        # next_move's several call sites (multiple were found unguarded
        # during the 2026-07-12 exp5587 cascade check).
        return 1
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
    if grid.ndim != 2:
        # Can't logically downsample a malformed grid -- return it unchanged
        # (same degenerate-input contract detect_cell now honors above)
        # rather than raising and killing the caller's remaining action
        # budget on the live scored path.
        return grid
    h, w = grid.shape
    return grid[::cell, ::cell] if cell > 1 else grid


def to_ascii(logical: np.ndarray) -> str:
    """Compact one-char-per-cell ASCII (single trailing digit of the color)."""
    return "\n".join("".join(str(int(v))[-1] for v in row) for row in logical)


def _state_key(g: np.ndarray) -> bytes | str:
    """The duplicate-state key for in-model search. Same partition as `to_ascii`, ~5x cheaper.

    WHY THIS EXISTS (REQ-ARC-WMTE-6051, measured 2026-07-30). `plan_in_model` calls `to_ascii`
    once per engine call to decide whether it has already seen a state. A cProfile of a
    shipped-budget ka59 search attributed **38% of the entire search** to that one function --
    essentially all of it in ~1.3M `str.join` calls building a one-char-per-cell Python string
    cell by cell. The search is CPU-bound replay with a hard wall-clock budget per game, so that
    38% is bought at the direct expense of nodes explored.

    WHY IT IS `% 10` AND NOT A PLAIN `tobytes()` -- the part that is easy to get wrong.
    `to_ascii` renders each cell as `str(int(v))[-1]`: the LAST DIGIT ONLY. Colour 14 and colour 4
    therefore render to the same character, as do 15/5, 11/1 and 10/0. The shipped key is LOSSY --
    it MERGES two states that differ only by such a swap, and the search discards the second as
    already-seen. Those colours are live, not hypothetical: ka59's root grid contains 4 AND 14 and
    also 5 AND 15; lp85's contains 1/11, 4/14 and 5/15; cn04's contains 0/10 and 4/14.

    So a plain `g.tobytes()` is NOT a drop-in. It distinguishes those states, inducing a strictly
    FINER partition -- a change to which states the search explores, not a speedup. That is not
    speculative either: measured across ten games, a plain-bytes key changes the search on cn04
    (93 engine calls / 9 unique states becomes 140 / 14).

    Whether the lossy merge is a defect worth fixing is a SEPARATE question, deliberately not
    answered here: fixing it would change search behaviour and belongs in its own change with its
    own evidence. This function's contract is to be indistinguishable from `to_ascii`, faster.

    WHAT THE CONTRACT ACTUALLY IS: A PARTITION, NOT A VALUE. This is the second thing an
    adversarial review caught here, and it invalidated the first fix's reasoning. The claim used to
    be that excluded input classes "fall back to `to_ascii` itself -- so equivalence holds by
    construction". That is false, because the property required of a dedup key is not
    ``key(x) == to_ascii(x)`` per input; it is

        ``key(x) == key(y)``  if and only if  ``to_ascii(x) == to_ascii(y)``

    over the whole set being keyed. Returning `to_ascii(x)` verbatim for SOME inputs and a byte
    encoding for others satisfies the per-input reading and BREAKS the partition, because a `bytes`
    can never equal a `str`: `to_ascii([[4]]) == to_ascii([[-4]])` (both render "4", MERGED) while
    the two-namespace version returned ``b"1:1|\\x04"`` and ``"4"`` (SPLIT). One negative cell
    anywhere in the reachable set silently un-merged it from its aliasing twin. Latent rather than
    active -- real ARC grids are non-negative -- but engines here are arbitrary LLM-written code,
    which is precisely the class of assumption that should not be load-bearing.

    So EVERY non-empty 2-D grid now keys into ONE namespace: the shape prefix followed by one byte
    per cell holding `to_ascii`'s digit for that cell. The fast path computes that digit as `a % 10`
    (exact for v >= 0, one C-level reduction to check); every other 2-D grid computes the identical
    digit the way `to_ascii` does, `int(str(int(v))[-1])`, cell by cell. Same partition, by
    construction this time, because both branches emit the same encoding of the same digit.

    The three conditions below therefore now select the ARITHMETIC, not the namespace. The reasoning
    for each is unchanged and kept verbatim; only where an excluded input goes has changed -- to the
    canonical byte encoding rather than out to a second namespace. The one class still handed to
    `to_ascii` whole is a grid that is not 2-D or is EMPTY, and that is deliberate: `to_ascii`
    collapses every zero-row shape to the same `""`, so a shape prefix would SPLIT states it merges.
    An empty grid's key is a `str` and a non-empty grid's is `bytes`, which cannot collide, and
    `to_ascii` never renders a non-empty grid as an empty-ish string -- so that one boundary is
    partition-safe in both directions.

    * **Non-negative.** This is the condition an adversarial review caught missing, and it is the
      subtlest of the three. `to_ascii` takes the last character of the DECIMAL STRING, which for a
      negative number is the last digit of its ABSOLUTE value: `str(-1)[-1] == "1"`. But `-1 % 10`
      is 9. The two agree only where a digit is its own complement mod 10, i.e. only for 0 and 5 --
      so they disagree on 12 of the 16 values in -15..-1, and `-1` and `-11` would have been keyed
      as 9 and 9 by arithmetic but as "1" and "1" by `to_ascii` (same class, different from the
      class `% 10` assigns). `np.abs(a) % 10` would match everywhere except at a dtype's most
      negative value, where `abs` silently overflows. Rather than take on that edge case for input
      that does not occur, the fast path simply declines negatives. The `min()` guard is one C-level
      reduction over the grid -- negligible beside the per-cell Python work it replaces.
    * **Integer.** For a negative float the encodings also disagree: `to_ascii` truncates toward
      zero (`int(-2.7) == -2`, rendering "2") while `-2.7 % 10 == 7.3` truncates to 7.
    * **2-D.** `to_ascii` iterates rows, so anything else is its business, including how it fails.

    ONE MORE FIDELITY DETAIL a naive swap gets wrong: **the shape prefix is load-bearing.**
    `to_ascii` separates rows with newlines, so a 2x3 grid and a 3x2 grid holding the same six
    values render differently. Raw bytes of the same values are IDENTICAL, so without the prefix
    those two states would collide. `plan_in_model` happens to be safe (it rejects any grid whose
    shape differs from the start grid before keying), but `plan_and_execute` only checks
    `ndim == 2` and would genuinely regress.

    MEASURED: partition identity is verified against `to_ascii` on ten games -- not by comparing
    totals (two different partitions can agree on those) but by hashing the search's ACCEPT TRACE,
    the key-independent sequence of accept/duplicate/skip decisions over every engine call. See
    REQ-ARC-WMTE-6051 for the per-game table.
    """
    a = np.asarray(g)
    if a.ndim == 2 and a.size:
        h, w = a.shape
        prefix = b"%d:%d|" % (h, w)
        if a.dtype.kind in "iu" and a.min() >= 0:
            # v % 10 == int(str(v)[-1]) for every v >= 0, so this is the canonical digit too.
            return prefix + (a % 10).astype(np.uint8).tobytes()
        # Same namespace, same canonical digit, computed the way `to_ascii` computes it.
        return prefix + bytes(int(str(int(v))[-1]) for v in a.flat)
    return to_ascii(a)


def _rle_grid(g: np.ndarray) -> str:
    """Lossless run-length encoding of a FULL grid for the induce prompt: one line per row,
    'r<row>:<v0>x<n0>,<v1>x<n1>,...' -- each row's runs cover ALL columns left-to-right with NO
    gaps, so the column position is implicit (the running sum of prior run counts in that row),
    never spelled out. On large boards (e.g. lp85's 64x64 logical grid), `to_ascii`'s
    one-char-per-cell render was the dominant fixed cost of `induce_prompt` -- a SINGLE full grid
    measured ~6-7K tokens, so an 8-transition window (up to two full-grid renders + per-transition
    deltas) measured 18,355 tokens against a 13,824-token available budget and overflowed with
    `exceed_context_size_error` (ops/known-issues.md task 11, exp5593). An earlier attempt at this
    fix spelled out an explicit `r<row>c<col>:<value>x<count>` per run (matching `_rle_delta`'s
    style) -- measured on lp85's REAL grids, that per-run column overhead made the encoding barely
    smaller than `to_ascii` for medium-length runs and up to 24% LARGER for a grid with many short
    runs (`_rle_delta` pays that overhead once per DIFF, a rare event; a FULL grid pays it once per
    RUN, hundreds of times). Dropping the explicit column (implicit from the row's own running
    count) removed that dominant per-run overhead."""
    g = np.asarray(g)
    h, w = g.shape
    lines = []
    for r in range(h):
        c = 0
        runs = []
        while c < w:
            v = g[r, c]
            c0 = c
            while c < w and g[r, c] == v:
                c += 1
            runs.append(f"{int(v)}x{c - c0}")
        lines.append(f"r{r}:" + ",".join(runs))
    return "\n".join(lines)


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


@dataclass
class ProgrammaticExpert:
    """REQ-ARC-WMTE-4677: one small object-level precondition/effect factor."""

    name: str
    object_class: str
    precondition: Callable[[np.ndarray, int, Any], bool]
    effect: Callable[[np.ndarray, int, Any], np.ndarray]
    action: int | None = None
    trust: float = 0.0
    heldout_correct: int = 0
    heldout_total: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def applies(self, grid: np.ndarray, action: int, data: Any = None) -> bool:
        if self.action is not None and int(action) != int(self.action):
            return False
        try:
            return bool(self.precondition(np.asarray(grid), int(action), data))
        except Exception:
            return False

    def predict(self, grid: np.ndarray, action: int, data: Any = None) -> np.ndarray:
        return np.asarray(self.effect(np.asarray(grid).copy(), int(action), data))

    def summary(self, *, kept: bool) -> dict[str, Any]:
        return {
            "name": self.name,
            "object_class": self.object_class,
            "trust": round(float(self.trust), 6),
            "heldout_correct": int(self.heldout_correct),
            "heldout_total": int(self.heldout_total),
            "kept": bool(kept),
        }


@dataclass
class ProgrammaticExpertInductionResult:
    """REQ-ARC-WMTE-4677: trusted factors plus the rejected-factor ledger."""

    experts: list[ProgrammaticExpert]
    expert_trust_weights: list[dict[str, Any]]
    proposer_used: bool = False
    llm_proposal_ok: bool = False
    residual: str = ""


@dataclass
class FactoredSubgoalPlanResult:
    """SCENARIO-ARC-WMTE-4677-PRODUCT-PLANNING: product-model plan diagnostics."""

    planned: bool
    plan: list[dict[str, Any]] = field(default_factory=list)
    subgoal_decomposition: list[dict[str, Any]] = field(default_factory=list)
    per_subgoal_reachable: list[dict[str, Any]] = field(default_factory=list)
    expert_trust_weights: list[dict[str, Any]] = field(default_factory=list)
    final_grid: np.ndarray | None = None
    residual: str = ""


def _color_rewrite_expert(
    *,
    name: str,
    object_class: str,
    action: int | None,
    from_color: int,
    to_color: int,
    metadata: Mapping[str, Any] | None = None,
) -> ProgrammaticExpert:
    src = int(from_color)
    dst = int(to_color)
    action_id = None if action is None else int(action)

    def _precondition(grid: np.ndarray, candidate_action: int, _data: Any) -> bool:
        return (action_id is None or int(candidate_action) == action_id) and bool(
            np.any(np.asarray(grid) == src)
        )

    def _effect(grid: np.ndarray, _candidate_action: int, _data: Any) -> np.ndarray:
        out = np.asarray(grid).copy()
        out[out == src] = dst
        return out

    return ProgrammaticExpert(
        name=name,
        object_class=object_class,
        precondition=_precondition,
        effect=_effect,
        action=action_id,
        metadata={
            "kind": "color_rewrite",
            "from_color": src,
            "to_color": dst,
            **dict(metadata or {}),
        },
    )


def _exact_delta_expert(transition: Transition, index: int) -> ProgrammaticExpert:
    base = np.asarray(transition.grid).copy()
    target = np.asarray(transition.next_grid).copy()
    action_id = int(transition.action)
    changed = np.argwhere(base != target)
    signature = [(int(r), int(c), int(base[r, c]), int(target[r, c])) for r, c in changed]

    def _precondition(grid: np.ndarray, candidate_action: int, _data: Any) -> bool:
        candidate = np.asarray(grid)
        return (
            int(candidate_action) == action_id
            and candidate.shape == base.shape
            and all(int(candidate[r, c]) == before for r, c, before, _after in signature)
        )

    def _effect(grid: np.ndarray, _candidate_action: int, _data: Any) -> np.ndarray:
        out = np.asarray(grid).copy()
        for r, c, _before, after in signature:
            out[r, c] = after
        return out

    colors = sorted({int(after) for _r, _c, _before, after in signature})
    return ProgrammaticExpert(
        name=f"exact_delta_action_{action_id}_{index}",
        object_class="cells_" + "_".join(str(color) for color in colors[:4]),
        precondition=_precondition,
        effect=_effect,
        action=action_id,
        metadata={"kind": "exact_delta", "changed_cells": len(signature)},
    )


def _normalise_programmatic_experts(rows: Sequence[Any]) -> list[ProgrammaticExpert]:
    experts: list[ProgrammaticExpert] = []
    for index, row in enumerate(rows):
        if isinstance(row, ProgrammaticExpert):
            experts.append(row)
            continue
        if not isinstance(row, Mapping):
            continue
        precondition = row.get("precondition")
        effect = row.get("effect")
        if callable(precondition) and callable(effect):
            experts.append(
                ProgrammaticExpert(
                    name=str(row.get("name") or f"expert_{index}"),
                    object_class=str(row.get("object_class") or row.get("object") or "object"),
                    precondition=precondition,
                    effect=effect,
                    action=(None if row.get("action") is None else int(row["action"])),
                    metadata=dict(row.get("metadata") or {}),
                )
            )
            continue
        if row.get("kind") == "color_rewrite" or {
            "from_color",
            "to_color",
        }.issubset(row.keys()):
            experts.append(
                _color_rewrite_expert(
                    name=str(row.get("name") or f"color_rewrite_{index}"),
                    object_class=str(row.get("object_class") or f"color_{row.get('from_color')}"),
                    action=(None if row.get("action") is None else int(row["action"])),
                    from_color=int(row["from_color"]),
                    to_color=int(row["to_color"]),
                    metadata=dict(row.get("metadata") or {}),
                )
            )
    return experts


def _stratified_prefix_heldout(
    transitions: Sequence[Transition],
    heldout_fraction: float,
) -> tuple[list[Transition], list[Transition]]:
    rows = list(transitions)
    if len(rows) < 2:
        return rows, rows
    n_suffix = max(1, int(round(len(rows) * max(0.0, min(1.0, heldout_fraction)))))
    heldout_indices = set(range(max(0, len(rows) - n_suffix), len(rows)))
    by_action: dict[int, list[int]] = {}
    for i, transition in enumerate(rows):
        by_action.setdefault(int(transition.action), []).append(i)
    for indices in by_action.values():
        if len(indices) > 1:
            heldout_indices.add(indices[-1])
    prefix = [row for i, row in enumerate(rows) if i not in heldout_indices]
    heldout = [row for i, row in enumerate(rows) if i in heldout_indices]
    return (prefix or rows[:1], heldout or rows[-1:])


def _fallback_experts_from_transitions(
    transitions: Sequence[Transition],
) -> list[ProgrammaticExpert]:
    experts: list[ProgrammaticExpert] = []
    seen: set[tuple[Any, ...]] = set()
    for index, transition in enumerate(transitions):
        before = np.asarray(transition.grid)
        after = np.asarray(transition.next_grid)
        if before.shape != after.shape or np.array_equal(before, after):
            continue
        changed = before != after
        from_values = sorted({int(v) for v in before[changed].flatten().tolist()})
        to_values = sorted({int(v) for v in after[changed].flatten().tolist()})
        if len(from_values) == 1 and len(to_values) == 1:
            key = ("color", int(transition.action), from_values[0], to_values[0])
            if key in seen:
                continue
            seen.add(key)
            experts.append(
                _color_rewrite_expert(
                    name=f"color_{from_values[0]}_to_{to_values[0]}_action_{int(transition.action)}",
                    object_class=f"color_{from_values[0]}",
                    action=int(transition.action),
                    from_color=from_values[0],
                    to_color=to_values[0],
                    metadata={"source": "transition_color_delta"},
                )
            )
        else:
            key = ("exact", int(transition.action), to_ascii(before))
            if key in seen:
                continue
            seen.add(key)
            experts.append(_exact_delta_expert(transition, index))
    return experts


def _score_expert_on_transitions(
    expert: ProgrammaticExpert,
    transitions: Sequence[Transition],
) -> ProgrammaticExpert:
    total = 0
    correct = 0
    for transition in transitions:
        if not expert.applies(transition.grid, int(transition.action), transition.data):
            continue
        total += 1
        try:
            pred = expert.predict(transition.grid, int(transition.action), transition.data)
        except Exception:
            continue
        if pred.shape == np.asarray(transition.next_grid).shape and np.array_equal(
            pred,
            np.asarray(transition.next_grid),
        ):
            correct += 1
    expert.heldout_total = int(total)
    expert.heldout_correct = int(correct)
    expert.trust = float(correct) / float(total) if total else 0.0
    return expert


def induce_programmatic_object_experts(
    *,
    game: str,
    transitions: Sequence[Transition],
    proposer: Any = None,
    cell: int = 1,
    trust_threshold: float = 0.75,
    heldout_fraction: float = 0.34,
    max_experts: int = 8,
) -> ProgrammaticExpertInductionResult:
    """REQ-ARC-WMTE-4677: induce factors, weight by held-out trust, keep stable ones."""

    rows = list(transitions)
    if not rows:
        return ProgrammaticExpertInductionResult(
            experts=[],
            expert_trust_weights=[],
            residual="experts_overfit_prefix",
        )
    prefix, heldout = _stratified_prefix_heldout(rows, heldout_fraction)
    proposed_rows: list[Any] = []
    proposer_used = False
    llm_ok = False
    provider = getattr(proposer, "induce_programmatic_experts", None)
    if callable(provider):
        proposer_used = True
        try:
            proposed_rows = list(
                provider(
                    game=game,
                    transitions=list(prefix),
                    heldout_transitions=list(heldout),
                    cell=int(cell),
                    max_experts=int(max_experts),
                )
                or []
            )
            llm_ok = bool(proposed_rows)
        except TypeError:
            try:
                proposed_rows = list(provider(game, list(prefix)) or [])
                llm_ok = bool(proposed_rows)
            except Exception:
                proposed_rows = []
        except Exception:
            proposed_rows = []
    experts = _normalise_programmatic_experts(proposed_rows)
    if not experts:
        experts.extend(_fallback_experts_from_transitions(prefix))

    deduped: list[ProgrammaticExpert] = []
    seen: set[str] = set()
    for expert in experts:
        key = json.dumps(
            {
                "name": expert.name,
                "action": expert.action,
                "object_class": expert.object_class,
                "metadata": expert.metadata,
            },
            sort_keys=True,
            default=str,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(expert)
        if len(deduped) >= int(max_experts):
            break

    threshold = max(0.0, min(1.0, float(trust_threshold)))
    kept: list[ProgrammaticExpert] = []
    weights: list[dict[str, Any]] = []
    for expert in deduped:
        scored = _score_expert_on_transitions(expert, heldout)
        is_kept = scored.heldout_total > 0 and scored.trust >= threshold
        if is_kept:
            kept.append(scored)
        weights.append(scored.summary(kept=is_kept))

    residual = (
        "" if kept else ("experts_overfit_prefix" if deduped else "expert_factors_not_independent")
    )
    return ProgrammaticExpertInductionResult(
        experts=kept,
        expert_trust_weights=weights,
        proposer_used=proposer_used,
        llm_proposal_ok=llm_ok,
        residual=residual,
    )


class ProductWorldModel:
    """REQ-ARC-WMTE-4677: executable product composition of trusted factors."""

    def __init__(self, experts: Sequence[ProgrammaticExpert]) -> None:
        self.experts = list(experts)

    def engine(self, grid: np.ndarray, action: int, data: Any = None) -> np.ndarray:
        start = np.asarray(grid)
        out = start.copy()
        trust = np.full(start.shape, -1.0, dtype=float)
        for expert in self.experts:
            if not expert.applies(start, int(action), data):
                continue
            pred = expert.predict(start, int(action), data)
            if pred.shape != start.shape:
                continue
            changed = pred != start
            stronger = changed & (float(expert.trust) >= trust)
            out[stronger] = pred[stronger]
            trust[stronger] = float(expert.trust)
        return out


def _normalise_factored_subgoals(rows: Sequence[Any]) -> list[dict[str, Any]]:
    subgoals: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if isinstance(row, Mapping):
            predicate = row.get("predicate") or row.get("is_level_complete")
            if callable(predicate):
                subgoals.append(
                    {
                        "name": str(row.get("name") or f"subgoal_{index}"),
                        "predicate": predicate,
                        "source": str(row.get("source") or "a1_goal_induction"),
                        "score": float(row.get("score") or 0.0),
                    }
                )
            continue
        predicate = getattr(row, "predicate", None)
        if callable(predicate):
            subgoals.append(
                {
                    "name": str(getattr(row, "name", f"subgoal_{index}")),
                    "predicate": predicate,
                    "source": str(getattr(row, "source", "a1_goal_induction")),
                    "score": float(getattr(row, "score", 0.0) or 0.0),
                }
            )
    return subgoals


def _apply_factored_plan(
    engine: Callable[[np.ndarray, int, Any], np.ndarray],
    start_grid: np.ndarray,
    plan: Sequence[Mapping[str, Any]] | None,
) -> np.ndarray:
    grid = np.asarray(start_grid)
    for step in list(plan or []):
        grid = np.asarray(engine(grid.copy(), int(step["action"]), step.get("data")))
    return grid


def plan_factored_subgoal_sequence(
    *,
    start_grid: np.ndarray,
    final_goal: Callable[[np.ndarray], bool],
    experts: Sequence[ProgrammaticExpert],
    subgoals: Sequence[Any] = (),
    value_head: Callable[[np.ndarray], float] | None = None,
    max_subgoals: int = 3,
    max_nodes: int = 20000,
    max_depth: int | None = None,
) -> FactoredSubgoalPlanResult:
    """SCENARIO-ARC-WMTE-4677-PRODUCT-PLANNING: plan through the product model.

    `max_depth=None` defers to `plan_max_depth_default()` via the `plan_in_model` calls below,
    rather than re-declaring 40 here. A literal default would have silently kept this path on the
    old horizon after the 2026-07-31 change -- the drift that resolver exists to make impossible.
    """

    product = ProductWorldModel(experts)
    current = np.asarray(start_grid)
    full_plan: list[dict[str, Any]] = []
    decomposition: list[dict[str, Any]] = []
    reachable_rows: list[dict[str, Any]] = []
    weights = [expert.summary(kept=True) for expert in experts]

    def _leg(goal: Callable[[np.ndarray], bool], grid: np.ndarray) -> list[dict[str, Any]] | None:
        try:
            if bool(goal(np.asarray(grid))):
                return []
        except Exception:
            return None
        return plan_in_model(
            product.engine,
            goal,
            np.asarray(grid),
            max_nodes=max_nodes,
            max_depth=max_depth,
            goal_energy=value_head,
        )

    ordered = sorted(
        _normalise_factored_subgoals(subgoals),
        key=lambda row: (float(row.get("score") or 0.0), str(row.get("name") or "")),
        reverse=True,
    )[: max(0, int(max_subgoals))]
    for subgoal in ordered:
        leg = _leg(subgoal["predicate"], current)
        reached = leg is not None
        row = {
            "name": subgoal["name"],
            "source": subgoal["source"],
            "reachable": bool(reached),
            "plan_length": len(leg or []),
            "score": round(float(subgoal.get("score") or 0.0), 6),
        }
        decomposition.append(dict(row))
        reachable_rows.append(dict(row))
        if not reached:
            return FactoredSubgoalPlanResult(
                planned=False,
                plan=full_plan,
                subgoal_decomposition=decomposition,
                per_subgoal_reachable=reachable_rows,
                expert_trust_weights=weights,
                final_grid=current,
                residual="product_model_plans_live_invalid",
            )
        full_plan.extend(dict(step) for step in leg)
        current = _apply_factored_plan(product.engine, current, leg)

    final_leg = _leg(final_goal, current)
    final_reached = final_leg is not None
    final_row = {
        "name": "final_goal",
        "source": "terminal_goal_predicate",
        "reachable": bool(final_reached),
        "plan_length": len(final_leg or []),
        "score": 1.0,
    }
    decomposition.append(dict(final_row))
    reachable_rows.append(dict(final_row))
    if not final_reached:
        return FactoredSubgoalPlanResult(
            planned=False,
            plan=full_plan,
            subgoal_decomposition=decomposition,
            per_subgoal_reachable=reachable_rows,
            expert_trust_weights=weights,
            final_grid=current,
            residual="product_model_plans_live_invalid",
        )
    full_plan.extend(dict(step) for step in final_leg)
    final_grid = _apply_factored_plan(product.engine, current, final_leg)
    return FactoredSubgoalPlanResult(
        planned=True,
        plan=full_plan,
        subgoal_decomposition=decomposition,
        per_subgoal_reachable=reachable_rows,
        expert_trust_weights=weights,
        final_grid=final_grid,
        residual="none",
    )


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


def replay_win_transition(game: str, cell: Optional[int] = None):
    """Replay a BANKED solve to recover one level-up transition, or None.

    WHY THIS EXISTS. `induce_prompt` has dedicated slots for `win_transition` and
    `previous_level_complete_grid` -- it is built to show the model what winning looks
    like. `collect_transitions` supplies neither, and the reason is NOT that it discards
    them (it appends the level-up row before restarting; that earlier diagnosis was wrong
    and is corrected in ops/known-issues.md 2026-08-12). The reason is that a
    salience-biased RANDOM walk essentially never reaches a level-up: measured 0 of 200
    steps on dc22 and 0 of 200 on cn04. So a win exemplar has to come from something that
    already knows one.

    This replays the banked `solution_labels` from `results/arc_loop_solve_<game>.json`
    through the game's OWN `GameAdapter.apply`, and returns the first transition where the
    level counter increments. Using the adapter's apply rather than a local re-implementation
    matters: label dialects differ per game (dc22's ACTION6 rows carry `grid`/`x`/`y`
    instead of `data`), and a fourth copy of that logic would drift.

    DEVELOPMENT PROXY ONLY -- read this before using it in any claim. A hidden game has no
    registry entry, no banked solution and no adapter, so the live agent CANNOT obtain a win
    exemplar this way. On the live path the exemplar comes from the agent's own play and is
    therefore available from level 2 onward, never for level 1. An offline measurement built
    on this answers "does the exemplar improve induction quality?" It does NOT show the live
    agent can get one on a hidden level 1. See the ARC Live-Path Reachability Discipline.

    Returns None whenever anything is missing or the replay never levels up. Callers treat
    None as "no exemplar available" and must not substitute a fabricated one.
    """
    import json as _json

    from carnot.agentic import arc_game_adapters as _adapters
    from carnot.agentic import arc_solver_kit as _kit
    from carnot.agentic.arc_agi3_live_adapter import _levels_completed
    from carnot.agentic.arc_agi3_world_model import grid_of

    banked = REPO / "results" / f"arc_loop_solve_{game}.json"
    adapter = _adapters.get_adapter(game)
    if adapter is None or not banked.exists():
        return None
    try:
        labels = _json.loads(banked.read_text()).get("solution_labels") or []
    except Exception:
        return None
    if not labels:
        return None

    arc = _kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = env.reset()
    if cell is None:
        cell = detect_cell(grid_of(f))
    for label in labels:
        l0 = _levels_completed(f)
        g0 = to_logical(grid_of(f), cell)
        try:
            nf = adapter.apply(env, label, f)
        except Exception:
            return None
        if nf is None:
            return None
        l1 = _levels_completed(nf)
        if l1 > l0:
            action, data = _action_and_data_from_label(label)
            return Transition(g0, action, data, to_logical(grid_of(nf), cell), l0, l1)
        f = nf
    return None


def _action_and_data_from_label(label: str) -> tuple[int, Any]:
    """Best-effort (action_id, data) from a banked solution label.

    Only the action id is reliably present across dialects. `data` is returned when the
    label carries it and None otherwise; a Transition with `data=None` still records the
    grids and the level change, which is what the win exemplar is for.
    """
    import json as _json

    try:
        step = _json.loads(label)
        return int(step.get("action", 0)), step.get("data")
    except Exception:
        return 0, None


# ---------------------------------------------------------------------------
# THE CARNOT VERIFIER — grounds the LLM-induced model against reality
# ---------------------------------------------------------------------------

# ===========================================================================
# REQ-ARC-WMTE-6010 / REQ-ARC-WMTE-6011 -- the two INDEPENDENT, DEFAULT-OFF
# repairs to world-model verification, measured today (2026-07-27) as the two
# reasons `induction_attempts_planned == 0` on 174/174 rows of the first-win
# measurement while the generator itself was healthy (103 calls / 94 responses
# / 0 errors).
#
# THEY PUSH IN OPPOSITE DIRECTIONS, WHICH IS WHY THEY ARE TWO FLAGS AND NOT ONE.
#   * Masking the HUD REMOVES cells that were unattainable by construction, and
#     should therefore RAISE every measured fidelity number.
#   * Closing the trust gate REJECTS degenerate engines that today pass, and
#     should therefore LOWER the pass rate.
# Shipped together behind a single flag and measured together, a null result is
# uninterpretable -- "both worked and cancelled" and "neither did" produce the
# same number. Two independent flags make the four-arm matrix (control /
# mask-only / gate-only / both) possible, which is the only design that can
# attribute the effect. Neither flag is flipped here; the flip is the operator's
# call on the strength of that matrix.
# ===========================================================================

# ---- REQ-ARC-WMTE-6010: the HUD is inside the exact-match comparison --------
# Transitions are recorded as FULL logical grids (arc_competition_agent.py's
# E3AgentPolicy.next_move, via `to_logical(grid_of(latest), self.cell)`). A HUD
# mask EXISTS and is live in the explorer (`_compute_hud_mask_from_frame`, the
# SUBMITTED_EDGE_BAR_HUD_MASK_ENABLED / _COLLAPSE_GUARD / _STAGE2_CONFIRM trio),
# but as of 2026-07-27 `grep hud_mask` returned ZERO hits in this module, in
# arc_llm_reinduction.py, and in arc_world_model_trust_energy.py -- verified
# directly. So on any game with a monotone step counter EVERY frame differs in
# the HUD, full-grid exact match is UNATTAINABLE BY CONSTRUCTION, and
#
# QUALIFICATION (measured 2026-07-27, second review): "a monotone step counter"
# describes a MINORITY of the games this mask is applied to. Measuring
# P(masked region changes | play area unchanged) over real transitions
# (n=60, seed 0) across the 17 games with a resolvable mask:
#     free-running counter (p >= 0.95):  6  -- bp35 lf52 s5i5 sp80 tu93 vc33
#     mixed (0.05 < p < 0.95):           7  -- ka59 cd82 dc22 su15 wa30 m0r0 g50t
#     game-COUPLED chrome (p <= 0.05):   2  -- ar25 ft09
#     unmeasurable (no still frames):    2  -- re86 tr87
# A free-running tick would sit at p = 1.0 everywhere. On the coupled games the
# masked row moves only when the game state also moves -- it is a score or
# progress readout, not an action-independent clock -- so the
# "unattainable by construction" argument does NOT cover them, and masking
# there is discarding real signal rather than removing noise. This does not
# make the mask dishonest; it means the justification is narrower than the
# application, which is why REQ-ARC-WMTE-6015's swallow guard exists.
#
# `cell_recall`'s change mask is dominated by counter cells rather than by game
# state. Part of the measured median-0.0 trust score is therefore a MEASUREMENT
# ARTIFACT, not purely a capability wall.
#
# The repair masks at COMPARE time, never at RECORD time: `Transition` keeps the
# full-fidelity grids it always kept (never-prune; a historical transitions dump
# stays byte-comparable), and only the comparison collapses HUD cells. The mask
# is supplied in LOGICAL coordinates -- see `logical_hud_mask`, which downsamples
# the explorer's FRAME-coordinate mask by the same `grid[::cell, ::cell]` stride
# `to_logical` uses, so the two are aligned by construction rather than by luck.
SUBMITTED_WORLD_MODEL_HUD_MASK_ENABLED = False
SUBMITTED_WORLD_MODEL_HUD_MASK_MODE = "compare_time_logical_mask_from_explorer_frame_mask"

# ---- REQ-ARC-WMTE-6011: GAP-WM-TRUST-GATE (ops/verifier_gaps.md) -----------
# An IDENTITY engine (`return grid`) scores 0.725 on `WorldModelVerifier.score`
# and PASSES the `accuracy >= 0.5` gate -- 0.5 being the threshold recorded in
# ops/verifier_gaps.md's GAP-WM-TRUST-GATE entry and used by
# `binary_exact_gate_pass`'s default. THRESHOLD CORRIGENDUM (2026-07-27): the
# threshold the AGENT ACTUALLY SHIPS is `min_heldout_accuracy=1.0`
# (arc_competition_agent.py:5593 and :5719). The two are not interchangeable --
# admission-flip counts differ by ~10x between them (see
# `change_gate_decision`'s legacy_accuracy_* keys) -- so every claim below is
# tagged with the threshold it was measured at, and any statement about LIVE
# behaviour uses 1.0. The 0.725-passes-0.5 framing is the DOCUMENTED gap, which
# is what this requirement was written against; it remains true at 0.5, and at
# 1.0 the identity engine is rejected by the incumbent gate for a different
# reason (it is not exact) while STILL being ranked and trusted through the
# hidden-state `trust_pass` branch, which is what REQ-6013 addresses.
# `accuracy` is full-grid exact
# match DENOMINATED OVER ALL TRANSITIONS INCLUDING NO-OPS and lp85's corpus is
# ~87 no-ops to 33 changing. Confirmed on disk 2026-07-27:
# results/arc_e3/ft09/world_model.py is 12 bare `return grid` branches and
# `is_level_complete -> False`; results/arc_e3/lp85/world_model.py mutates only
# on `action == 6 and grid[py, px] == 9`. Both were reported to the operator as
# "the good model" and are in fact degenerate.
#
# WHY NOT `cell_recall`, AND WHY NOT THE EXISTING `score_change_weighted_
# consistency`: both mask to TRUE changes only
# (`pred[changed] == next_grid[changed]`), so neither can see a cell the engine
# wrote that reality did NOT change. They punish misses and ignore spurious
# writes -- they are recall, not fidelity. A "write garbage everywhere" engine
# is invisible to them in exactly the way the identity engine is invisible to
# `accuracy`. So the gate quantity here is SYMMETRIC by construction: it scores
# over the UNION of the truly-changed cells and the engine-written cells, which
# makes a spurious write cost exactly what a missed change costs.
SUBMITTED_WORLD_MODEL_CHANGE_GATE_ENABLED = False
SUBMITTED_WORLD_MODEL_CHANGE_GATE_MODE = "symmetric_union_change_fidelity_plus_nondegeneracy_floor"
# Calibrated from the measured separation between the real on-disk degenerates
# and a hand-written correct engine -- see the REQ-ARC-WMTE-6011 witness in
# tests/python/test_arc_world_model_change_gate.py, which asserts the separation
# rather than the constants, so a re-tune cannot silently empty the pass region.
WORLD_MODEL_CHANGE_FIDELITY_THRESHOLD = 0.5
WORLD_MODEL_MIN_CORRECT_CHANGED_CELLS = 1
# Ceiling on the fraction of TRULY-UNCHANGED transitions the engine changes anyway. 0.25 is
# calibrated from the measured separation, not picked round: the honest hand-written dc22
# navigation engine hallucinates on 0.0000 of its no-ops, while the adversarial
# "correct-but-invents-a-change-on-every-no-op" engine that defeated the fidelity-only gate
# sits at 1.0000. Anything in between is a real judgement call, so the threshold is set well
# clear of the honest engine rather than tight against the attack.
WORLD_MODEL_MAX_NOOP_HALLUCINATION_RATE = 0.25

# ---- REQ-ARC-WMTE-6013: the change gate's HIDDEN-STATE branch coverage hole ----
# REQ-6011 above shipped `change_gate_decision` wired into exactly ONE of the agent's two
# admission branches -- the `else` (non-hidden-state) one. The OTHER branch, taken for the
# 11 HIDDEN_STATE_GAME_IDS, admits on `trust_pass` from `select_trusted_world_model` and
# never calls the change gate. That branch covers EVERY one of the 0.08-wall games
# (cn04/ar25/sc25/sk48/wa30), so the gate had zero coverage on precisely the games the
# whole programme exists to move.
#
# THE HOLE, MEASURED (results/experiment_6012_hidden_state_trust_gate_hole.json, 33 matched
# rows = 11 games x 3 seeds): an engine that is correct on every real change AND ALSO writes
# cells reality never wrote is ADMITTED by the live hidden-state gate on 31/33 rows -- the
# SAME 31 rows on which it admits the honest engine. Both attack arms score EXACTLY (not
# approximately) the honest engine's `consistency`. The cause is structural and is the same
# one REQ-6011 names for `cell_recall`: `score_change_weighted_consistency` masks to TRUE
# changes only (`pred[changed] == next[changed]`), so a cell the engine invented outside that
# mask is arithmetically invisible to it. It is recall, not fidelity.
#
# The repair routes the hidden-state branch's ADMIT/REJECT decision through the SAME
# symmetric union-fidelity `change_gate_decision` the plain branch uses -- REPLACING
# `trust_pass`, exactly as the plain branch replaces its `accuracy >= 0.5`, not AND-ing with
# it. AND-ing was considered and rejected on measurement: exp6012 found the live gate is not
# merely blind, it is ALSO too strict in the other direction -- it rejects the hand-written
# honest dc22 engine on 2/3 seeds where REQ-6011 admits it 3/3. Keeping `trust_pass` as a
# conjunct would import that false-reject wholesale, so the arm would confound "the symmetric
# metric helps" with "the old metric still vetoes".
#
# The CANDIDATE RANKING is deliberately NOT touched: `select_trusted_world_model` still picks
# by trust energy. Only the final admit/reject changes. Ranking and admission are separable
# concerns, and changing both at once would make a per-arm delta unattributable.
#
# WHY A THIRD FLAG RATHER THAN FOLDING INTO REQ-6011'S. The two branches gate DISJOINT game
# sets, so per-game attribution is automatic and a shared flag would not confound. But the
# branches replace DIFFERENT incumbent metrics (`trust_pass` from held-out recall here vs
# `accuracy` there) with different measured risk profiles, so an experimenter must be able
# to isolate them. The default is therefore "follow REQ-6011's flag" -- which keeps the
# four-arm matrix at four arms and stops the gate arm from being a silent no-op on the 11
# wall games -- with an explicit env override that separates them when a measurement needs it.
# None means follow; True/False pin it.
SUBMITTED_WORLD_MODEL_CHANGE_GATE_HIDDEN_STATE_ENABLED: Optional[bool] = None

# LEVER (REQ-ARC-WMTE-6241, Phase 3a of the 2026-08-08 ARC live-agent improvement plan).
# Two additive appendices to the induce prompt, both gated by this ONE flag since the plan
# proposes them as a single lever to A/B together: (1) semantic action names (UP/DOWN/LEFT/
# RIGHT/SPACE/MOUSE for actions 1-6 -- action 7 has no established semantic meaning in this
# project and is left as a bare integer) plus an explicit changed-cell COUNT per sampled
# transition (both notebook reference stacks the plan cites converged on "trust numeric
# transitions over visual guesses"); (2) a computed object-identity cross-reference between
# objects_block's two tables (which shape ids are shared, replacing that block's existing
# text HINT with an actual computed list). Both are pure ADDITIONS appended after the existing
# prompt content -- neither `_transitions_block` nor `objects_block` is modified, so their own
# extensively-documented historical correctness fixes stay untouched. Default OFF pending a
# leave-one-game-out held-out change-fidelity A/B; NOT a rerun of exp6214 (a per-transition
# matched before/after component-DELTA construction) -- this is a static per-frame identity/
# topology table plus scalar counts, a different construction entirely.
SUBMITTED_INDUCE_PROMPT_ENRICHMENT_ENABLED = False

# REQ-ARC-WMTE-6282: default-off mechanic-class evidence for the live inducer.
# The route is game-blind. It reads only observed transition deltas and appends a
# short class/uncertainty block to the prompt. Control arms stay byte-identical.
SUBMITTED_MECHANIC_CLASS_ROUTER_ENABLED = False


def _flag_env(name: str, default: bool) -> bool:
    """Read a per-arm override, falling back to the shipped SUBMITTED_* default.

    The four-arm matrix (control / mask-only / gate-only / both) needs to select an arm
    WITHOUT editing the shipped constants, because an arm selected by editing a constant
    cannot be run concurrently with its own control and cannot be reproduced from a command
    line in an artifact. Unset env -> the shipped default, so the submitted path is
    unaffected by the existence of this knob.
    """

    import os

    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.strip().lower() in ("1", "true", "yes", "on")


def world_model_hud_mask_enabled() -> bool:
    """REQ-ARC-WMTE-6010 arm selector. CARNOT_ARC_WM_HUD_MASK=1 turns compare-time masking on."""

    return _flag_env("CARNOT_ARC_WM_HUD_MASK", SUBMITTED_WORLD_MODEL_HUD_MASK_ENABLED)


def world_model_change_gate_enabled() -> bool:
    """REQ-ARC-WMTE-6011 arm selector. CARNOT_ARC_WM_CHANGE_GATE=1 turns the change gate on."""

    return _flag_env("CARNOT_ARC_WM_CHANGE_GATE", SUBMITTED_WORLD_MODEL_CHANGE_GATE_ENABLED)


def world_model_change_gate_hidden_state_enabled() -> bool:
    """REQ-ARC-WMTE-6013 arm selector for the hidden-state branch.

    Resolution order, most specific first:
      1. CARNOT_ARC_WM_CHANGE_GATE_HIDDEN_STATE -- the explicit isolation override.
      2. SUBMITTED_WORLD_MODEL_CHANGE_GATE_HIDDEN_STATE_ENABLED, when pinned to True/False.
      3. Otherwise FOLLOW REQ-6011's flag, so turning the change gate on covers BOTH
         admission branches instead of silently skipping the 11 hidden-state games.

    Step 3 is what keeps the four-arm matrix at four arms. Step 1 is what lets a follow-up
    measurement separate the two branches without a code edit.
    """

    import os

    raw = os.environ.get("CARNOT_ARC_WM_CHANGE_GATE_HIDDEN_STATE")
    if raw is not None and raw != "":
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if SUBMITTED_WORLD_MODEL_CHANGE_GATE_HIDDEN_STATE_ENABLED is not None:
        return bool(SUBMITTED_WORLD_MODEL_CHANGE_GATE_HIDDEN_STATE_ENABLED)
    return world_model_change_gate_enabled()


def induce_prompt_enrichment_enabled() -> bool:
    """REQ-ARC-WMTE-6241 arm selector. CARNOT_ARC_INDUCE_PROMPT_ENRICHMENT=1 turns on the
    semantic-action-names + changed-cell-count + object-identity-crossref appendices."""

    return _flag_env(
        "CARNOT_ARC_INDUCE_PROMPT_ENRICHMENT", SUBMITTED_INDUCE_PROMPT_ENRICHMENT_ENABLED
    )


def mechanic_class_router_enabled() -> bool:
    """REQ-ARC-WMTE-6282 arm selector. CARNOT_ARC_MECHANIC_CLASS_ROUTER=1 turns it on."""

    return _flag_env("CARNOT_ARC_MECHANIC_CLASS_ROUTER", SUBMITTED_MECHANIC_CLASS_ROUTER_ENABLED)


def logical_hud_mask(frame_mask: Any, cell: int) -> Optional[np.ndarray]:
    """Downsample a FRAME-coordinate HUD mask to LOGICAL-grid coordinates.

    `to_logical` is a plain stride `grid[::cell, ::cell]`, so the logical grid's
    (r, c) is the frame's (r*cell, c*cell) and the mask must be taken with the
    IDENTICAL stride. Doing anything cleverer (e.g. "any HUD pixel in the block")
    would silently mask cells `to_logical` never sampled, which is the
    over-masking direction -- and over-masking destroys CORRECTNESS while
    under-masking only costs efficiency, the same asymmetry that keeps the
    explorer's own edge-bar detector behind a Stage-2 confirmation gate.

    Returns None when there is nothing to align (no mask, malformed mask, bad
    cell size). None is NEVER a silent no-op at the call sites: every caller
    records an explicit `hud_mask_status` saying WHY no mask was applied.
    """

    if frame_mask is None:
        return None
    try:
        mask = np.asarray(frame_mask, dtype=bool)
    except Exception:
        return None
    if mask.ndim != 2:
        return None
    c = int(cell) if cell else 1
    if c < 1:
        return None
    out = mask[::c, ::c] if c > 1 else mask
    return out if bool(out.any()) else None


# ---- REQ-ARC-WMTE-6015: THE SWALLOW GUARD ---------------------------------
# Found 2026-07-27 by measuring mask coverage per game rather than trusting the "it is only
# a monotone step counter" story. On two of the 17 games with a resolvable mask, the
# explorer's HUD classifier selects a row where the GAME STATE lives, not chrome:
#
#     game   changed-cells-inside-mask   changing transitions, raw -> masked
#     lf52          1.0000                       60 -> 0     (the entire game is deleted)
#     su15          0.7568                       28 -> 1
#     ...  every other game               0.0000 .. 0.2219   (s5i5 0.2219 is the highest)
#
# CORPUS SCOPE OF THE TABLE ABOVE: `collect_transitions(n=60, seed=0)` -- SIXTY actions, not
# 120. Re-derived 2026-07-28 (REQ-ARC-WMTE-6019) two independent ways, from the frozen
# fixtures sliced to 60 and from a live re-collect, which agree exactly:
#
#     n     lf52 overlap / raw -> surviving      su15 overlap / raw -> surviving
#     60      1.0000   60 -> 0                     0.7568   28 -> 1   <- reproduces the table
#     120     1.0000  120 -> 0                     0.7391   51 -> 2
#
# n=60 is a strict PREFIX of n=120 (same seed draws the same action sequence; the mask is
# computed from the reset frame alone, so it is identical), which is why the fixture slice and
# the live re-collect match. The DECISION is unchanged at either scope and under both
# `edge_bar_detector` settings: lf52 and su15 are REFUSED at n=60 and at n=120, and lp85's
# honest mask is refused at neither (16/16 changing transitions survive at n=60, 33/33 at
# 120). So this is a provenance correction, not a verdict correction -- recorded because a
# threshold calibrated on an unstated corpus scope cannot be re-derived by a reader.
# On lf52 the mask makes the corpus DYNAMICS-FREE: nothing changes, so the IDENTITY engine
# is a perfect model of a game with no mechanics, and it is admitted. That is exactly the
# laundering an adversarial review flagged in the aggregate ("the mask helps a zero-knowledge
# engine ~1.7x more than a real one"), here located in a specific mechanism on specific games
# rather than left as a statistical worry.
#
# `apply_hud_mask`'s own docstring already names the asymmetry that decides the fix:
# "over-masking destroys CORRECTNESS while under-masking only costs efficiency". So the guard
# is REFUSE-ON-DOUBT: a mask that swallows the dynamics is not applied at all, and the reason
# is recorded. Refusing degrades that game to the pre-REQ-6010 behaviour (measurably worse,
# per the very artifact this repair is built on) -- which is the correct direction, because
# the alternative is a metric that scores an engine on a game it has deleted.
#
# THRESHOLD PROVENANCE. 0.5 is not picked round; it is the midpoint of the only wide gap in
# the measured distribution above (0.2219 -> 0.7568), so it is well clear of the worst honest
# game AND of the least-bad swallowing one. The zero-dynamics case is checked SEPARATELY and
# unconditionally, because "the corpus has changes and the mask leaves none" is a swallow at
# any threshold and must not depend on a tunable.
#
# CORPUS-SCOPE CORRIGENDUM (2026-07-28, REQ-ARC-WMTE-6017). The table above was measured on
# ONE corpus shape: RANDOM actions from reset (`collect_transitions`) -- at n=60, per the
# corpus-scope note above; the n=120 re-capture is a SECOND measurement of the same games, not
# the table's own corpus (REQ-ARC-WMTE-6019 correction: this paragraph previously said the
# table was measured at 120, which is the fixture scope, not the table scope). The LIVE path
# judges a different corpus again -- the current episode's transitions, 25 on lf52 -- and the
# verdict FLIPS between random-action and live for the same game and the SAME 64-cell mask:
#
#     corpus                        raw changing  cells inside/total  overlap  verdict
#     lf52, 60 random actions (table)    60            60 /  60        1.0000  REFUSED (b)
#     lf52, 120 random actions          120           120 / 120        1.0000  REFUSED (b)
#     lf52, live episode (25 rows)       25            25 /  81        0.3086  ok, applied
#
# Both records are honest. On the random corpus NOTHING but the counter ever moved, so the
# corpus cannot tell "the mask covers the game" from "this corpus has no state change" --
# case (b) below, refused on doubt. On the live corpus 56 of 81 changed cells fall OUTSIDE
# the mask (23 counter-only ticks are correctly revealed as no-ops, 2 real changes survive),
# which is positive evidence that the mask does NOT cover lf52's game state. So a verdict is
# a statement about (mask, corpus), never about the mask alone, and the fields recording the
# corpus scope below exist so a reviewer can see WHICH corpus produced a verdict instead of
# reading a game-level property off it.
HUD_MASK_MAX_CHANGED_CELL_OVERLAP = 0.5


def hud_mask_swallow_clean(rec: Optional[dict]) -> bool:
    """Is this an AFFIRMATIVE clean verdict -- "checked, and the mask is fine"?

    REQ-ARC-WMTE-6017 (2026-07-28). `hud_mask_swallow_check`'s docstring already states the
    contract: "`swallows=False` with `n_changed_cells_total == 0` is reported as
    `no_dynamics_to_swallow` -- an unmeasurable verdict, NOT a clean one, so a consumer
    cannot read 'we checked and it is fine' off a corpus where the check could not fire."
    Every consumer nonetheless tested `rec.get("swallows")` for truthiness, which reads an
    UNMEASURABLE verdict as a clean one and applies the mask -- the function documented the
    trap and its callers walked into it.

    REAL INSTANCE, not hypothetical: ft09, whose classifier resolves a 64-cell mask while its
    120-action corpus contains ZERO changing transitions (`raw_changing_transitions == 0`).
    That is `no_dynamics_to_swallow`, the guard cannot fire, and before this helper the mask
    was applied with `hud_mask_status == "applied"` -- indistinguishable in the record from a
    mask that had been measured and cleared.

    Clean requires ALL of: the check ran (`checked`), it did not find a swallow, AND its
    reason is the affirmative `ok`. Anything else -- `no_dynamics_to_swallow`,
    `no_transitions`, or any future reason a later requirement adds -- is NOT clean, so the
    default for an unrecognised reason is refuse. That direction is forced by
    `apply_hud_mask`'s stated asymmetry: over-masking destroys correctness, under-masking
    only costs efficiency.

    `no_mask` is also not clean, but it is never reached as a refusal: every consumer checks
    `hud_mask is None` FIRST and reports `disabled`/`unresolved`, which are the honest
    statuses for "there was nothing to judge".
    """

    if not isinstance(rec, dict):
        return False
    return bool(rec.get("checked")) and not rec.get("swallows") and rec.get("reason") == "ok"


def _hud_mask_refusal_status(rec: dict) -> str:
    """Name the refusal precisely -- REQ-ARC-WMTE-6017.

    Four distinct situations reach a refusal and they are NOT the same claim:

      refused_swallows_dynamics          the mask was MEASURED to cover the dynamics
      no_transitions                     there was no corpus to judge at all
      shape_mismatch                     the mask fits NONE of the corpus's grids
      refused_swallow_check_unmeasurable the corpus has grids the mask fits, but not one
                                         state-changing transition among them, so the check
                                         could not fire (`no_dynamics_to_swallow`)

    `no_transitions` and `shape_mismatch` are the SAME STRINGS `score()` already resolved for
    those cases before this requirement moved the decision earlier. Reusing them is not
    cosmetic: collapsing them into the new status would delete a diagnostic distinction that
    already had consumers (`test_arc_world_model_change_gate.py` asserts both), which is the
    "never remove existing content" failure in code rather than docs. Dropping the mask in the
    two of them where it USED to be kept is behaviour-neutral: `apply_hud_mask` already
    returned the grid untouched on a shape mismatch, and an empty corpus grades nothing.
    """

    if rec.get("swallows"):
        return "refused_swallows_dynamics"
    if int(rec.get("n_transitions") or 0) == 0:
        return "no_transitions"
    if rec.get("checked") and int(rec.get("n_transitions_shape_matched") or 0) == 0:
        return "shape_mismatch"
    return "refused_swallow_check_unmeasurable"


def hud_mask_swallow_check(transitions: Sequence["Transition"], mask: Optional[np.ndarray]) -> dict:
    """Does this mask delete the game rather than the chrome? Returns an auditable record.

    A dict, not a bool, for the same reason `change_gate_decision` returns one: a caller must
    be able to show WHY a mask was refused, and a reviewer must be able to see the measured
    quantity next to the threshold that judged it. `swallows=False` with
    `n_changed_cells_total == 0` is reported as `no_dynamics_to_swallow` -- an unmeasurable
    verdict, NOT a clean one, so a consumer cannot read "we checked and it is fine" off a
    corpus where the check could not fire.
    """

    rec = {
        "checked": False,
        "swallows": False,
        "reason": "no_mask",
        "changed_cell_overlap": 0.0,
        "overlap_threshold": float(HUD_MASK_MAX_CHANGED_CELL_OVERLAP),
        "raw_changing_transitions": 0,
        "masked_changing_transitions": 0,
        "n_changed_cells_total": 0,
        "n_changed_cells_inside_mask": 0,
        # ---- REQ-ARC-WMTE-6017 CORPUS SCOPE (2026-07-28) ------------------------------
        # A verdict is about (mask, corpus). Without these a reader cannot tell whether a
        # refusal was measured over 120 transitions or over 25, which is exactly the
        # ambiguity that let lf52's `refused` (120 random actions) and `applied` (25 live
        # rows) both be true and look contradictory.
        "n_transitions": 0,
        "n_transitions_shape_matched": 0,
        "n_transitions_skipped_shape_mismatch": 0,
        # ---- REQ-ARC-WMTE-6017 PER-TRANSITION WITNESS ---------------------------------
        # `changed_cell_overlap` pools CELLS across transitions, so a corpus whose real
        # changes are few-but-large reads as high overlap while one whose real changes are
        # many-but-small reads as low -- the summary is dominated by change DENSITY, which
        # is a property of the corpus, not of the mask. The hazard ("a mechanic observation
        # was deleted") lives at the TRANSITION level, so the transition-level quantities
        # are recorded next to the cell-pooled one. Measured spread, live and offline
        # corpora both: survival 0.0 (tn36) .. 1.0 (lp85, re86, tr87).
        #
        # THEY ARE RECORDED, NOT GATED ON -- deliberately, and for the same reason
        # `invented_changed_cells` below is not a gate condition: on the measured corpora
        # survival does NOT separate swallowing from honest masks (su15 live 0.153 swallow-
        # suspect vs vc33 offline 0.208 honest), so any threshold here would be fitted to
        # the two games it must catch and would tell us nothing about a third. Adding a gate
        # on this evidence would be the forced-gate failure mode this project keeps naming.
        "changing_transitions_deleted": 0,
        "changing_transition_survival": 0.0,
        "mean_changed_cell_overlap_per_changing_transition": 0.0,
    }
    rows = list(transitions)
    # Recorded BEFORE the no-mask return: "we were handed 120 transitions and no mask" and
    # "we were handed nothing" are different facts, and a record that reported 0 for both
    # would be a field that cannot distinguish them -- the same dead-channel shape this
    # requirement is removing elsewhere.
    rec["n_transitions"] = len(rows)
    if mask is None:
        return rec
    if not rows:
        rec["reason"] = "no_transitions"
        return rec
    m = np.asarray(mask, dtype=bool)
    total = inside = raw_changing = masked_changing = 0
    shape_matched = shape_skipped = 0
    per_transition_overlaps: list[float] = []
    for t in rows:
        g0 = np.asarray(t.grid)
        g1 = np.asarray(t.next_grid)
        if g0.shape != g1.shape or g0.shape != m.shape:
            # Counted, not silently dropped: a check that skipped every transition would
            # otherwise report `no_dynamics_to_swallow` -- an unmeasurable verdict wearing
            # the shape of a clean one, which is the defect `hud_mask_swallow_clean` exists
            # to stop being read as clean.
            shape_skipped += 1
            continue
        shape_matched += 1
        ch = g0 != g1
        if not ch.any():
            continue
        raw_changing += 1
        n_ch = int(ch.sum())
        n_in = int((ch & m).sum())
        total += n_ch
        inside += n_in
        per_transition_overlaps.append(float(n_in / n_ch) if n_ch else 0.0)
        if not np.array_equal(apply_hud_mask(g0, m), apply_hud_mask(g1, m)):
            masked_changing += 1
    rec.update(
        {
            "checked": True,
            "raw_changing_transitions": raw_changing,
            "masked_changing_transitions": masked_changing,
            "n_changed_cells_total": total,
            "n_changed_cells_inside_mask": inside,
            "changed_cell_overlap": round(float(inside / total), 6) if total else 0.0,
            "n_transitions_shape_matched": shape_matched,
            "n_transitions_skipped_shape_mismatch": shape_skipped,
            "changing_transitions_deleted": int(raw_changing - masked_changing),
            "changing_transition_survival": (
                round(float(masked_changing / raw_changing), 6) if raw_changing else 0.0
            ),
            "mean_changed_cell_overlap_per_changing_transition": (
                round(float(np.mean(per_transition_overlaps)), 6)
                if per_transition_overlaps
                else 0.0
            ),
        }
    )
    if total == 0:
        rec["reason"] = "no_dynamics_to_swallow"
        return rec
    if raw_changing > 0 and masked_changing == 0:
        # The corpus has changes and the mask leaves none. TWO different situations produce
        # this, and they are NOT distinguishable from inside this corpus:
        #   (a) the mask really does cover the game (lf52: overlap 1.0, 60 -> 0);
        #   (b) the mask is honest chrome and this corpus genuinely contains no state
        #       change, so the only cells that moved were the counter's.
        # Both are refused -- `apply_hud_mask`'s stated asymmetry (over-masking destroys
        # correctness, under-masking only costs efficiency) makes refusing the safe
        # direction -- but they are given DIFFERENT reasons, because (b) is a statement
        # about the corpus and (a) is a defect in the mask, and an operator reading a
        # refusal needs to know which one they are looking at. Collapsing them would be the
        # same clean-vs-unmeasurable conflation `noop_ok_is_vacuous` exists to prevent.
        rec["swallows"] = True
        rec["reason"] = (
            "mask_removes_all_dynamics"
            if inside < total
            else "no_changed_cells_outside_mask_cannot_distinguish"
        )
        return rec
    if rec["changed_cell_overlap"] >= float(HUD_MASK_MAX_CHANGED_CELL_OVERLAP):
        rec["swallows"] = True
        rec["reason"] = "mask_overlaps_majority_of_changed_cells"
        return rec
    rec["reason"] = "ok"
    return rec


def apply_hud_mask(grid: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Collapse HUD cells to a constant so they cannot decide an exact-match test.

    Mirrors `StepwiseExplorer._hash`, which already zeroes `hud_mask` cells before
    hashing a frame for node identity -- same collapse convention, same constant,
    so a state that dedups to one node in the search also compares as one state in
    the world model. Shape-mismatched masks are IGNORED here and reported by the
    caller as `shape_mismatch`; silently applying a wrong-shaped mask (or letting
    numpy broadcast one) is precisely the failure this repair is fixing.
    """

    if mask is None:
        return grid
    g = np.asarray(grid)
    if getattr(mask, "shape", None) != g.shape:
        return grid
    out = g.copy()
    out[mask] = 0
    return out


@dataclass
class VerifyResult:
    n: int
    n_correct: int
    accuracy: float
    mismatches: list[dict] = field(default_factory=list)
    error: Optional[str] = None
    # GRADED companion to `accuracy` (which is exact-FULL-GRID match): mean changed-cell recall over the
    # state-CHANGING transitions. Exact-match reads ~0 for an imperfect (e.g. LLM-induced or learned) world
    # model that is still ~55% changed-cell-accurate, so it gates EVERY such model out of execution-grounded
    # planning -- the single root cause of the 0.08 wall (docs/research-notes/arc-008-wall-root-cause-2026-06-21.md).
    # cell_recall is the granularity-matched gate the coordinated redesign turns on via CARNOT_ARC_TRUST_METRIC.
    cell_recall: float = 0.0

    # ---- REQ-ARC-WMTE-6011 change-weighted fields (GAP-WM-TRUST-GATE) -------
    # Number of recorded transitions that actually changed the (masked) grid. This is
    # the denominator `accuracy` should have had: on lp85 it is 33 of 120, so an
    # engine can be wrong about every single mechanic and still read 0.725.
    n_changing: int = 0
    # Transitions among `n_changing` the engine reproduced EXACTLY. The gap file's
    # literal `n_changes_correct`; reported, and the strictest available witness.
    n_changes_correct: int = 0
    # `n_changes_correct / n_changing` -- the gap file's literal `change_accuracy`.
    # Reported for continuity with the gap entry; NOT the gate quantity, because
    # exact-full-grid match over changing transitions is the same all-or-nothing
    # measure that REQ-6010 shows is unattainable while the HUD is in the compare.
    change_accuracy: float = 0.0
    # THE GATE QUANTITY. Per changing transition, the fraction of the UNION of
    # (cells reality changed) and (cells the engine wrote) that the engine got
    # right, averaged over changing transitions. Symmetric: a miss and a spurious
    # write cost the same. Identity engines score 0.0 here by construction (they
    # write nothing, so the union is exactly the true changes and none are right).
    change_fidelity: float = 0.0
    # Non-degeneracy floor, in CELLS not transitions: how many truly-changed cells
    # the engine predicted correctly. Cell-denominated on purpose -- a
    # transition-denominated floor would be unreachable on a HUD game whenever
    # REQ-6010's mask is off, which would re-introduce exactly the cross-flag
    # coupling the two-flag split exists to prevent.
    correct_changed_cells: int = 0
    # The ASYMMETRY WITNESS: cells the engine wrote that reality did not change to
    # that value. `cell_recall` and `score_change_weighted_consistency` are both
    # structurally blind to this number. Non-zero here with a high `cell_recall` is
    # the "writes garbage everywhere but happens to cover the real changes" engine.
    spurious_changed_cells: int = 0

    # ---- THE NO-OP HALLUCINATION CHANNEL (found by adversarially attacking this gate) ----
    # `change_fidelity` scores GRID-CHANGING transitions only, which leaves it structurally
    # blind to an engine that models every real change correctly AND ALSO invents a change on
    # every NO-OP. Measured on real dc22 transitions: such an engine scores change_fidelity
    # 0.7243 and PASSES, while its full-grid exact accuracy is 0.0000 -- it is wrong about
    # every single transition in the corpus. That engine is catastrophic for `plan_in_model`,
    # which walks the engine forward and would see phantom transitions at every step. The
    # LEGACY accuracy gate caught it (0.0 < 0.5), so without this channel the repair would be
    # strictly WORSE than what it replaces on this failure mode.
    #
    # It is a SEPARATE gate condition rather than being folded into `change_fidelity`: adding
    # no-ops into the same average would give a correctly-idle identity engine credit for
    # every no-op it "predicts", which reproduces exactly the 0.725 blind spot this whole
    # requirement exists to remove.
    n_noop: int = 0
    n_noop_hallucinated: int = 0  # truly-unchanged transitions the engine changed anyway
    noop_hallucination_rate: float = 0.0
    # ---- REQ-ARC-WMTE-6013 DIAGNOSTICS (recorded, deliberately NOT gated on) ----
    # `noop_hallucination_rate` above returns 0.0 when `n_noop == 0`, so the value meaning
    # "this engine invents nothing" is ALSO the value meaning "this could not be measured".
    # That is a structurally dead channel wearing a passing score, and it is not
    # hypothetical: on re86 all 40 held-out transitions change, n_noop is 0, and an engine
    # that writes a cell reality never wrote clears the whole gate at fidelity 0.919 because
    # the one channel that would have caught it cannot fire. This flag separates the two
    # meanings so a consumer can tell "clean" from "unmeasurable".
    noop_channel_measurable: bool = False
    # The PURE invented-write count: cells the engine changed that reality did NOT change at
    # all. Distinct from `spurious_changed_cells`, which is `wrote & ~correct` and therefore
    # CONFLATES two different things -- a cell invented out of nothing, and a genuinely-
    # changed cell predicted with the wrong value (ordinary prediction error, which every
    # imperfect-but-useful engine has). Only this quantity isolates invention.
    #
    # IT IS NOT A GATE CONDITION, ON PURPOSE. It separates perfectly on the corpus measured
    # so far (honest engines 0, the spurious writer one per changing transition), and that is
    # exactly why it must not be thresholded here: a separation measured against an engine
    # built to be caught tells you nothing about where a REALISTICALLY IMPERFECT engine sits,
    # and a threshold fitted to the former would reject the latter. Recalibration against an
    # imperfect engine is follow-up work with the operator, not a side effect of this change.
    invented_changed_cells: int = 0
    invented_change_rate: float = 0.0  # invented_changed_cells / n_changing
    # Explicit provenance for REQ-6010 -- one of "disabled" (flag off, no mask
    # requested), "unresolved" (flag on, caller had no mask to give), "shape_mismatch"
    # (flag on, mask given, but it did not align with the graded grids), or "applied".
    # NEVER a silent no-op: a caller that asked for masking and did not get it can
    # tell which of the three failure reasons it hit.
    hud_mask_status: str = "disabled"
    # Number of logical cells the applied mask covers. 0 whenever status != "applied".
    hud_mask_cells: int = 0
    # REQ-ARC-WMTE-6015 swallow-guard record. Carried on every result, including the
    # unmasked ones (where it reports `no_mask`), so an artifact row always shows whether
    # the guard could have fired and what it measured -- never only when it did fire.
    hud_mask_swallow: dict = field(default_factory=dict)
    # Transitions dropped from grading because they are LEVEL-UP rows, whose `next_grid` is the
    # NEXT level's re-laid-out opening board rather than this action's effect (2026-07-29). No
    # engine can predict a re-layout it has never seen, so grading those rows measures nothing
    # about the dynamics and penalises an HONEST engine for the level-up it correctly caused.
    # Measured: one identical, perfectly-honest engine scores accuracy 1.0000 / change_fidelity
    # 1.0000 on a window with no level-up and 0.6667 / 0.6667 on the same window whose last row
    # is a real level-up. Because the acceptance gate is `heldout_accuracy >= threshold`, that
    # difference is the whole gate at a strict threshold. Reported so the exclusion is never
    # silent: a reader can always see how many rows were dropped and recompute without them.
    n_levelup_rows_excluded: int = 0
    # REQ-ARC-WMTE-6017: "precomputed_by_caller" (the verdict is about the caller's whole
    # corpus) or "computed_on_this_corpus" (about this verifier's slice only).
    hud_mask_swallow_source: str = "computed_on_this_corpus"

    # ---- REQ-ARC-WMTE-6042 WRITE-COLLAPSE INSTRUMENTATION -------------------
    # WHY THESE EXIST. The CEGIS induction-refinement harness records `prefix_accuracy` -- fit
    # on the rows the model was SHOWN, with answers -- and across the two shipped CEGIS shards
    # it collapses between the induce round and the refactor rounds: 28/88 induce rounds reach
    # >0 with a ceiling of 1.0, against 4/160 refactor rounds with a ceiling of 0.125 (Fisher
    # OR 18.2, p=1.03e-10), and 15 of 83 cells fall from a PERFECT 1.0 to 0.0. `accuracy` and
    # `cell_recall` say the resulting engine is bad; they cannot say WHAT it became, and the
    # difference decides the fix. An engine that predicts a plausible-but-wrong change is a
    # modelling error; an engine that returns its input unchanged, or raises on every row, is a
    # DEGENERATE artefact of the write path and needs a different repair entirely. Nothing in
    # `VerifyResult` distinguished those two before this block.
    #
    # ALL FOUR ARE RECORDED, NONE IS GATED ON. They change no acceptance decision anywhere --
    # deliberately, and for the reason `invented_changed_cells` states above: a threshold fitted
    # to engines built to be caught tells you nothing about where a realistically imperfect
    # engine sits.
    #
    # Rows the engine was invoked on -- i.e. after level-up exclusion, so it matches `n`. The
    # denominator every field below is honest about.
    n_engine_called: int = 0
    # Rows where the engine RAISED. Uncapped, unlike the `error` entries in `mismatches`, which
    # stop at `max_mismatch`: a 40-row total wipeout and an 8-row one are indistinguishable
    # there, and they are very different engines.
    n_engine_raised: int = 0
    # Exception TYPE NAME -> count. Names the failure without carrying unbounded repr text.
    engine_raise_kinds: dict = field(default_factory=dict)
    # Answered rows where the graded output equals the graded INPUT -- the behavioural identity
    # measurement. Never inferred from source text; see the loop for why syntax lies both ways.
    n_output_equals_input: int = 0
    # n_output_equals_input / (n_engine_called - n_engine_raised). 0.0 when nothing was
    # answered -- which is ALSO the "wrote something on every row" value, so read it beside
    # `identity_measurable`, never alone.
    identity_rate: float = 0.0
    # True only when the engine ANSWERED at least one row and returned its input on every row
    # it answered. The non-vacuity half is the point: an engine that raises everywhere answers
    # nothing, and "identity" is not the honest description of that.
    functionally_identity: bool = False
    # Separates "not identity" from "could not be measured", exactly as `noop_channel_measurable`
    # does for the no-op channel. False whenever every row raised or the corpus was empty.
    identity_measurable: bool = False


class WorldModelVerifier:
    """Checks that an induced engine(grid, action, data) -> grid reproduces the real
    recorded transitions. This is the verification that makes the LLM accountable: a
    proposed model only earns trust by predicting transitions it was NOT hand-fit to.
    Returns mismatch artifacts (the failing transitions) for the refactor step."""

    def __init__(
        self,
        transitions: list[Transition],
        *,
        hud_mask: Any = None,
        hud_mask_enabled: Optional[bool] = None,
        hud_mask_swallow: Optional[dict] = None,
    ) -> None:
        """`hud_mask` is in LOGICAL-grid coordinates (see `logical_hud_mask`).

        `hud_mask_enabled` defaults to the module flag SUBMITTED_WORLD_MODEL_HUD_MASK_ENABLED
        so the submitted path is byte-identical until the operator flips it; an explicit
        True/False is the per-arm override the four-arm A/B needs.
        """

        self.transitions = transitions
        self.hud_mask_enabled = (
            world_model_hud_mask_enabled() if hud_mask_enabled is None else bool(hud_mask_enabled)
        )
        self.hud_mask = hud_mask if self.hud_mask_enabled else None
        # REQ-ARC-WMTE-6015: refuse a mask that deletes the game instead of the chrome.
        # Run BEFORE the status is settled so a swallowing mask can never reach `_graded`.
        #
        # `hud_mask_swallow` may be PRE-COMPUTED by the caller, and callers that hold the
        # whole corpus SHOULD pre-compute it. Whether a mask covers the game or the chrome
        # is a property of the MASK AND THE WHOLE CORPUS, not of whatever slice this
        # verifier happens to hold. Judging it per-slice produces a real false positive:
        # `select_trusted_world_model` grades on a held-out TAIL, and a tail that happens to
        # contain no genuine state change has ALL of its changed cells inside the HUD -- an
        # honest mask then looks exactly like a swallowing one, and the guard disables the
        # repair on precisely the no-op-heavy corpora the repair exists for (lp85 is ~87
        # no-ops to 33 changing). The full corpus can tell the two apart; a tail cannot.
        self.hud_mask_swallow = (
            dict(hud_mask_swallow)
            if hud_mask_swallow is not None
            else hud_mask_swallow_check(self.transitions, self.hud_mask)
        )
        # REQ-ARC-WMTE-6017: WHOSE corpus produced the verdict. A pre-computed verdict is a
        # statement about the caller's whole corpus; a self-computed one is about this
        # verifier's slice only. Both are legitimate (see the pre-computation note above) but
        # they are different claims, and a reviewer reading a refusal off an artifact cannot
        # otherwise tell which one they are looking at.
        self.hud_mask_swallow_source = (
            "precomputed_by_caller" if hud_mask_swallow is not None else "computed_on_this_corpus"
        )
        # Resolved once, here, so `score()` cannot drift between the status it reports and
        # the grids it actually compared.
        if not self.hud_mask_enabled:
            self.hud_mask_status = "disabled"
        elif self.hud_mask is None:
            # The flag asked for masking and the caller had none to give. This is the
            # explicit record the repair promises instead of a silent no-op.
            self.hud_mask_status = "unresolved"
        elif not hud_mask_swallow_clean(self.hud_mask_swallow):
            # THE SWALLOW GUARD FIRING. Drop the mask entirely and say so. Degrading this
            # game to unmasked grading is the deliberate choice: an unmasked comparison is
            # merely hard to win, while a swallowed one is scoring engines on a game whose
            # dynamics have been deleted -- under which the IDENTITY engine is optimal.
            #
            # REQ-ARC-WMTE-6017: the test used to be `self.hud_mask_swallow.get("swallows")`,
            # i.e. plain truthiness, which let an UNMEASURABLE verdict
            # (`no_dynamics_to_swallow`, real instance: ft09's 64-cell mask over a corpus with
            # zero changing transitions) fall through to `requested` and be APPLIED. The two
            # refusals get DIFFERENT statuses because they are different claims: one says the
            # mask was measured and found to cover the dynamics, the other says the corpus
            # could not measure it at all. Collapsing them would recreate the very
            # clean-vs-unmeasurable conflation this fix removes.
            self.hud_mask = None
            self.hud_mask_status = _hud_mask_refusal_status(self.hud_mask_swallow)
        else:
            self.hud_mask_status = "requested"

    def _graded(self, grid: np.ndarray) -> np.ndarray:
        return apply_hud_mask(grid, self.hud_mask)

    def score(
        self, engine: Callable[[np.ndarray, int, Optional[dict]], np.ndarray], max_mismatch: int = 8
    ) -> VerifyResult:
        n_correct, mism = 0, []
        cell_recalls: list[
            float
        ] = []  # per-CHANGED-transition fraction of changed cells predicted right
        # REQ-ARC-WMTE-6011 accumulators (see VerifyResult for what each one is for).
        n_changing = 0
        n_changes_correct = 0
        fidelities: list[float] = []
        correct_changed_cells = 0
        spurious_changed_cells = 0
        invented_changed_cells = 0
        n_noop = 0
        n_noop_hallucinated = 0
        # ---- REQ-ARC-WMTE-6042 WRITE-COLLAPSE INSTRUMENTATION (pure record, no control flow) ----
        # These three accumulators ride the EXISTING per-transition loop and add ZERO engine
        # calls: `pred`, `g0` and `pred_g` are already computed for the correctness comparison
        # below, so this is arithmetic on values that exist either way. That is deliberate and
        # load-bearing, not an optimisation -- an engine may hold module-level state, so an
        # instrumentation pass that invoked the engine even one extra time could change what a
        # LATER consumer in the same round observes. Riding the existing loop makes trajectory
        # invariance STRUCTURAL rather than merely tested.
        n_engine_called = 0
        n_engine_raised = 0
        engine_raise_kinds: dict[str, int] = {}
        n_output_equals_input = 0
        # ENGINE-CALL GUARD (REQ-ARC-WMTE-6400): this loop runs a GENERATED engine
        # once per transition, on the live scored path, BEFORE plan_in_model ever
        # sees it. A non-terminating call here hangs the per-game thread the same
        # way (the sb26 incident class). A trip counts as an engine raise; past the
        # trip limit the remaining rows are charged as raises WITHOUT running them.
        from carnot.agentic.arc_engine_call_guard import (
            EngineCallGuardError,
            guard_max_trips,
            guarded_call,
        )

        engine_guard_trips = 0
        engine_guard_trip_limit = guard_max_trips()
        engine_guard_skip_kind = "EngineCallGuardError"
        # REQ-ARC-WMTE-6010: resolve the mask's status from the TRANSITIONS ALONE, before the
        # engine runs. Deriving it inside the loop (the first version of this code) made the
        # status depend on ENGINE behaviour: an engine that raised on every transition, or an
        # empty corpus, would `continue` past the alignment check and report `unresolved` even
        # though a perfectly good mask had been supplied. The mask either fits these grids or
        # it does not, and that is a fact about the mask and the corpus, not about the engine.
        mask_status = self.hud_mask_status
        if mask_status == "requested":
            shapes = {np.asarray(t.grid).shape for t in self.transitions}
            if not shapes:
                mask_status = "no_transitions"
            elif any(getattr(self.hud_mask, "shape", None) == s for s in shapes):
                mask_status = "applied"
            else:
                mask_status = "shape_mismatch"
        n_levelup_excluded = 0
        for i, t in enumerate(self.transitions):
            # LEVEL-UP ROWS ARE NOT GRADEABLE DYNAMICS EVIDENCE (2026-07-29). The completing
            # action satisfies the win condition AND re-lays out the playfield atomically, so
            # `t.next_grid` is the NEXT LEVEL'S OPENING BOARD -- on ka59 the winning step rewrites
            # 3527 of 4096 cells against an ordinary-step median of 18.5. No engine induced from
            # THIS level's transitions can predict a layout it has never observed, so scoring the
            # row measures the renderer's level-change, not the engine's dynamics, and it penalises
            # an honest engine precisely for causing the level-up we wanted.
            #
            # Unlike the goal predicate (see `score_goal_predicate_consistency`), there is NO
            # counterfactual available here: the engine's own prediction IS the counterfactual, so
            # grading it against itself would be vacuous. Exclusion is the only sound option.
            #
            # DIRECTION OF THIS CHANGE, STATED PLAINLY: it can only ADMIT engines that were
            # previously rejected, never reject one that previously passed. Every admitted engine
            # was rejected on a row that carried no information about it. `n_levelup_rows_excluded`
            # records the count so no acceptance is silently attributable to this.
            if t.level_after > t.level_before:
                n_levelup_excluded += 1
                continue
            n_engine_called += 1
            if engine_guard_trips >= engine_guard_trip_limit:
                # The engine already proved non-terminating on this corpus. Charge
                # the row as a raise (same accounting as the except branch below)
                # instead of paying another full timeout for a known hang.
                n_engine_raised += 1
                engine_raise_kinds[engine_guard_skip_kind] = (
                    engine_raise_kinds.get(engine_guard_skip_kind, 0) + 1
                )
                if len(mism) < max_mismatch:
                    mism.append(
                        {"i": i, "action": t.action, "error": "engine_guard_trip_limit_reached"}
                    )
                continue
            try:
                pred = np.asarray(guarded_call(engine, t.grid.copy(), t.action, t.data))
            except Exception as e:  # a crashing engine fails the transition
                # REQ-ARC-WMTE-6042: COUNT the raise, do not only sample it. `mism` is capped at
                # `max_mismatch` (default 8), so reading the raise count off the mismatch list
                # silently censors every raise past the cap -- an engine that raises on all 40
                # rows is indistinguishable there from one that raises on 8. The count and the
                # exception KINDS are recorded separately and uncapped.
                #
                # REQ-ARC-WMTE-6400: a guard trip (hang / memory blowup) lands here
                # too -- the row produced no answer, so raise accounting is the
                # honest conversion. The trip is ALSO counted toward the limit above.
                if isinstance(e, EngineCallGuardError):
                    engine_guard_trips += 1
                    engine_guard_skip_kind = type(e).__name__
                n_engine_raised += 1
                kind = type(e).__name__
                engine_raise_kinds[kind] = engine_raise_kinds.get(kind, 0) + 1
                if len(mism) < max_mismatch:
                    mism.append({"i": i, "action": t.action, "error": repr(e)[:160]})
                continue
            # REQ-ARC-WMTE-6010: grade on HUD-collapsed copies. The recorded Transition is
            # left untouched (never-prune: the raw grids stay exactly as observed).
            g0 = self._graded(t.grid)
            g1 = self._graded(t.next_grid)
            pred_g = self._graded(pred) if pred.shape == np.asarray(t.next_grid).shape else pred
            # REQ-ARC-WMTE-6042: did the engine RETURN ITS INPUT? Measured BEHAVIOURALLY, on the
            # graded grids, never by pattern-matching the source for `return grid`. Both syntactic
            # failure modes are real and were observed on the CEGIS residue corpus: an engine that
            # never writes the literal `return grid` and is nonetheless identity on every row it
            # answers, and an engine that writes it a dozen times and is NOT identity on the
            # corpus it was induced from. Only executing it settles the question.
            if pred_g.shape == g0.shape and np.array_equal(pred_g, g0):
                n_output_equals_input += 1
            # graded changed-cell recall (granularity-matched gate); only state-changing transitions count
            changed = not np.array_equal(g0, g1)
            if changed:
                n_changing += 1
                if pred_g.shape == g1.shape:
                    m = g0 != g1
                    cell_recalls.append(float((pred_g[m] == g1[m]).mean()))
                    # ---- symmetric union fidelity (THE GATE QUANTITY) ----------------
                    # `m` is what reality changed; `wrote` is what the engine changed. Scoring
                    # over their UNION is what makes a spurious write cost what a miss costs;
                    # `cell_recall` above scores over `m` alone and therefore cannot see one.
                    wrote = pred_g != g0
                    union = m | wrote
                    correct = pred_g == g1
                    n_union = int(union.sum())
                    fidelities.append(float((correct & union).sum() / n_union) if n_union else 1.0)
                    correct_changed_cells += int((correct & m).sum())
                    spurious_changed_cells += int((wrote & ~correct).sum())
                    # REQ-ARC-WMTE-6013: `wrote & ~m` -- the engine changed a cell that
                    # reality left alone. `~m` (reality did not change it) rather than
                    # `~correct` (the prediction was wrong) is what makes this invention
                    # rather than error.
                    invented_changed_cells += int((wrote & ~m).sum())
                else:
                    cell_recalls.append(0.0)
                    fidelities.append(0.0)
            else:
                # A TRUE no-op. The engine should leave it alone; if it did not, it invented
                # a transition that reality does not contain. Counted separately from
                # `change_fidelity` -- see VerifyResult's NO-OP HALLUCINATION CHANNEL note.
                n_noop += 1
                if pred_g.shape != g1.shape or not np.array_equal(pred_g, g1):
                    n_noop_hallucinated += 1
            if pred_g.shape == g1.shape and np.array_equal(pred_g, g1):
                n_correct += 1
                if changed:
                    n_changes_correct += 1
            elif len(mism) < max_mismatch:
                ok_shape = pred_g.shape == g1.shape
                # COMPACT mismatch (deltas, not full grids — fits a local model's context):
                # what the TRUE action did vs where the engine's prediction was wrong.
                mism.append(
                    {
                        "i": i,
                        "action": t.action,
                        "data": t.data,
                        "true_change": _delta(g0, g1),
                        "your_prediction_was_wrong_at": (
                            _delta(pred_g, g1) if ok_shape else f"wrong shape {pred_g.shape}"
                        ),
                    }
                )
        # Excluded level-up rows leave the DENOMINATOR too. Counting them in `n` while never
        # grading them would cap `accuracy` at (n - n_levelup)/n and hand a strict threshold an
        # automatic failure on any window containing a level-up -- the same wrong-frame penalty,
        # relocated from the numerator to the denominator.
        n = len(self.transitions) - n_levelup_excluded
        cell_recall = float(np.mean(cell_recalls)) if cell_recalls else 0.0
        return VerifyResult(
            n,
            n_correct,
            n_correct / max(1, n),
            mism,
            n_levelup_rows_excluded=n_levelup_excluded,
            cell_recall=cell_recall,
            n_changing=n_changing,
            n_changes_correct=n_changes_correct,
            change_accuracy=float(n_changes_correct / n_changing) if n_changing else 0.0,
            change_fidelity=float(np.mean(fidelities)) if fidelities else 0.0,
            correct_changed_cells=correct_changed_cells,
            spurious_changed_cells=spurious_changed_cells,
            n_noop=n_noop,
            n_noop_hallucinated=n_noop_hallucinated,
            noop_hallucination_rate=(float(n_noop_hallucinated / n_noop) if n_noop else 0.0),
            noop_channel_measurable=bool(n_noop > 0),
            invented_changed_cells=invented_changed_cells,
            invented_change_rate=(
                float(invented_changed_cells / n_changing) if n_changing else 0.0
            ),
            hud_mask_status=mask_status,
            hud_mask_cells=(
                int(np.asarray(self.hud_mask).sum()) if mask_status == "applied" else 0
            ),
            hud_mask_swallow=dict(self.hud_mask_swallow),
            hud_mask_swallow_source=str(self.hud_mask_swallow_source),
            # ---- REQ-ARC-WMTE-6042 write-collapse instrumentation ----
            n_engine_called=n_engine_called,
            n_engine_raised=n_engine_raised,
            engine_raise_kinds=dict(engine_raise_kinds),
            n_output_equals_input=n_output_equals_input,
            # DENOMINATOR IS ANSWERED ROWS, NOT ALL ROWS. A row the engine raised on produced no
            # output, so it is evidence of neither identity nor non-identity; putting it in the
            # denominator would dilute an identity engine that also crashes, and putting it in the
            # numerator would invent an answer it never gave. `n_engine_raised` is reported beside
            # this so the excluded rows are never silent.
            identity_rate=(
                float(n_output_equals_input / (n_engine_called - n_engine_raised))
                if (n_engine_called - n_engine_raised) > 0
                else 0.0
            ),
            # NON-VACUITY IS PART OF THE PREDICATE. With zero answered rows the equality
            # `n_output_equals_input == answered` is trivially true, so without the `> 0` guard an
            # engine that raised on EVERY row -- or an empty corpus -- would be reported as
            # "functionally identity", which is a claim the data cannot support. Missing is not
            # zero, and unmeasurable is not clean.
            functionally_identity=bool(
                (n_engine_called - n_engine_raised) > 0
                and n_output_equals_input == (n_engine_called - n_engine_raised)
            ),
            identity_measurable=bool((n_engine_called - n_engine_raised) > 0),
        )

    def offpath_structural_energy(
        self,
        engine: Callable[[np.ndarray, int, Optional[dict]], np.ndarray],
        *,
        energy_scorer: Any,
    ) -> float:
        """REQ-ARC-WMTE-4791: score candidate predictions without reading true next grids."""

        # ENGINE-CALL GUARD (REQ-ARC-WMTE-6400): same generated-engine exposure and
        # same conversion as `score` above -- a trip scores the row inf, and past
        # the trip limit remaining rows score inf without running the engine.
        from carnot.agentic.arc_engine_call_guard import (
            EngineCallGuardError,
            guard_max_trips,
            guarded_call,
        )

        engine_guard_trips = 0
        engine_guard_trip_limit = guard_max_trips()
        energies: list[float] = []
        for t in self.transitions:
            if engine_guard_trips >= engine_guard_trip_limit:
                energies.append(float("inf"))
                continue
            try:
                pred = np.asarray(guarded_call(engine, t.grid.copy(), t.action, t.data))
                if hasattr(energy_scorer, "transition_energy"):
                    value = energy_scorer.transition_energy(t.grid, t.action, t.data, pred)
                else:
                    value = energy_scorer(t.grid, t.action, t.data, pred)
                value_f = float(value)
                energies.append(value_f if value_f == value_f else float("inf"))
            except EngineCallGuardError:
                engine_guard_trips += 1
                energies.append(float("inf"))
            except Exception:
                energies.append(float("inf"))
        if not energies:
            return float("inf")
        finite = [value for value in energies if value < float("inf")]
        if not finite:
            return float("inf")
        return float(np.mean(finite))

    def rank_offpath_structural_energy(
        self,
        candidates: Sequence[
            tuple[str, Callable[[np.ndarray, int, Optional[dict]], np.ndarray]]
            | dict[str, Any]
            | Any
        ],
        *,
        energy_scorer: Any,
    ) -> list[dict[str, Any]]:
        """REQ-ARC-WMTE-4791: rank candidate engines by lower off-path structural energy."""

        rows: list[dict[str, Any]] = []
        for i, candidate in enumerate(candidates):
            if isinstance(candidate, dict):
                name = str(candidate.get("name") or f"candidate_{i}")
                engine = candidate["engine"]
            elif isinstance(candidate, tuple):
                name = str(candidate[0])
                engine = candidate[1]
            else:
                name = str(getattr(candidate, "name", f"candidate_{i}"))
                engine = getattr(candidate, "engine", candidate)
            rows.append(
                {
                    "candidate_name": name,
                    "offpath_structural_energy": self.offpath_structural_energy(
                        engine,
                        energy_scorer=energy_scorer,
                    ),
                    "n_offpath_transitions": len(self.transitions),
                }
            )
        return sorted(
            rows,
            key=lambda row: (
                float(row["offpath_structural_energy"]),
                str(row["candidate_name"]),
            ),
        )


def change_gate_decision(
    vr: "VerifyResult",
    *,
    enabled: Optional[bool] = None,
    fidelity_threshold: float = WORLD_MODEL_CHANGE_FIDELITY_THRESHOLD,
    min_correct_changed_cells: int = WORLD_MODEL_MIN_CORRECT_CHANGED_CELLS,
    max_noop_hallucination_rate: float = WORLD_MODEL_MAX_NOOP_HALLUCINATION_RATE,
) -> dict:
    """REQ-ARC-WMTE-6011: the change-weighted trust decision, as an auditable record.

    Returns a dict rather than a bare bool ON PURPOSE. A bare bool cannot answer "could
    this gate have failed?" -- and a pass that could not have failed is not evidence. The
    returned record carries the COMPUTED WITNESS at the gate's own aggregation level: the
    two sub-decisions, the two measured quantities, the two thresholds, and the size of
    the population each quantity was computed over. `n_changing == 0` is reported as its
    own reason (`no_changing_transitions`) because a corpus with no state-changing
    transition cannot distinguish a good engine from the identity engine -- refusing there
    is the honest answer, not a pass by default.

    `enabled=False` (the shipped default via SUBMITTED_WORLD_MODEL_CHANGE_GATE_ENABLED)
    still computes and returns every field; it just reports `passed=True` with reason
    `gate_disabled`, so a control arm records the same diagnostics as a treatment arm and
    the four-arm matrix compares like with like.
    """

    on = world_model_change_gate_enabled() if enabled is None else bool(enabled)
    fidelity_ok = float(vr.change_fidelity) >= float(fidelity_threshold)
    nondegenerate = int(vr.correct_changed_cells) >= int(min_correct_changed_cells)
    has_population = int(vr.n_changing) > 0
    noop_ok = float(vr.noop_hallucination_rate) <= float(max_noop_hallucination_rate)
    if not on:
        reason = "gate_disabled"
        passed = True
    elif not has_population:
        reason = "no_changing_transitions"
        passed = False
    elif not nondegenerate:
        # THE ORIGIN INCIDENT. ft09's identity engine and lp85's near-identity engine both
        # land here: they never correctly predict a single changed cell, while `accuracy`
        # reads 0.725 / 1.0 because the corpus is no-op-heavy.
        #
        # HONEST NOTE ON REDUNDANCY AT THE DEFAULT k=1. At `min_correct_changed_cells == 1`
        # this branch cannot fire while `fidelity_ok` is True, and that is a THEOREM, not a
        # coincidence: if no truly-changed cell is predicted correctly, then every cell in
        # the (true-changes UNION engine-writes) set is wrong -- the true-changed ones by
        # assumption, and each engine-written-but-unchanged one because "correct" there
        # would require pred == next == prev, contradicting "the engine wrote it". So the
        # union score is exactly 0 and the fidelity test has already failed. Confirmed
        # empirically over 924 real arms in
        # results/experiment_6011_world_model_change_gate_four_arm.json: the combination
        # (nondegenerate=False, fidelity_ok=True) is never observed.
        #
        # It is kept, and ordered BEFORE the fidelity test, for two reasons. (1) It emits a
        # strictly more diagnostic reason: "this engine never got a single real change
        # right" is actionable where "fidelity 0.0 < 0.5" is not. (2) It becomes an
        # INDEPENDENT gate condition the moment `min_correct_changed_cells > 1`, which is
        # the knob for demanding a minimum evidence base rather than merely a non-zero one
        # -- see test_nondegeneracy_floor_is_redundant_at_k1_and_independent_above_it, which
        # asserts BOTH halves so this cannot quietly become a dead channel.
        reason = "degenerate_engine_no_correct_changed_cells"
        passed = False
    elif not fidelity_ok:
        reason = "change_fidelity_below_threshold"
        passed = False
    elif not noop_ok:
        # Found by attacking this gate rather than by testing it: an engine correct on every
        # real change that ALSO invents one on every no-op scores change_fidelity 0.7243 on
        # real dc22 transitions while being wrong about 100% of them (exact accuracy 0.0000).
        # `plan_in_model` walks the engine forward, so such an engine hallucinates a
        # transition at every step of every plan.
        reason = "engine_hallucinates_changes_on_noop_transitions"
        passed = False
    else:
        reason = "passed"
        passed = True
    return {
        "gate_enabled": on,
        "passed": bool(passed),
        "reason": reason,
        # --- computed witness, at the gate's own aggregation level ---------------
        "change_fidelity": round(float(vr.change_fidelity), 6),
        "fidelity_threshold": float(fidelity_threshold),
        "fidelity_ok": bool(fidelity_ok),
        "correct_changed_cells": int(vr.correct_changed_cells),
        "min_correct_changed_cells": int(min_correct_changed_cells),
        "nondegenerate": bool(nondegenerate),
        "spurious_changed_cells": int(vr.spurious_changed_cells),
        "noop_hallucination_rate": round(float(vr.noop_hallucination_rate), 6),
        "max_noop_hallucination_rate": float(max_noop_hallucination_rate),
        "noop_ok": bool(noop_ok),
        "n_noop": int(vr.n_noop),
        "n_noop_hallucinated": int(vr.n_noop_hallucinated),
        # REQ-ARC-WMTE-6013 diagnostics. Reported, NOT gated on -- see VerifyResult. When
        # `noop_channel_measurable` is False the `noop_ok` verdict above is vacuous (it
        # passed because there was nothing to test, not because the engine is clean), and a
        # consumer that treats those two cases alike will read a false pass.
        "noop_channel_measurable": bool(vr.noop_channel_measurable),
        "noop_ok_is_vacuous": bool(not vr.noop_channel_measurable),
        "invented_changed_cells": int(vr.invented_changed_cells),
        "invented_change_rate": round(float(vr.invented_change_rate), 6),
        "n_changing": int(vr.n_changing),
        "n_transitions": int(vr.n),
        # The legacy quantity this gate replaces, carried alongside so any artifact row
        # shows BOTH verdicts and the disagreement is visible without a re-run.
        "legacy_accuracy": round(float(vr.accuracy), 6),
        # ---- THRESHOLD AMBIGUITY, RESOLVED EXPLICITLY (2026-07-27 corrigendum) --------
        # `legacy_accuracy_would_pass` used to be reported ALONE against a hardcoded 0.5,
        # and 0.5 is NOT the threshold the agent ships. The live admission is
        # `min_heldout_accuracy=1.0` at BOTH call sites (arc_competition_agent.py:5593 and
        # :5719, verified on disk 2026-07-27). The gap is not cosmetic: recomputed over
        # exp6011's 75 rows, mask-induced admission flips for the IDENTITY engine are 29/75
        # at 0.5 but only 3/75 at 1.0, and for the real on-disk engines 12/75 at 0.5 but
        # 0/75 at 1.0 -- an order of magnitude, and a sign change in the headline. Reporting
        # one number against an unnamed threshold made every admission claim unfalsifiable,
        # so BOTH are now reported, each with its threshold named in the key.
        #
        # `legacy_accuracy_would_pass` is retained (not renamed) so already-written
        # consumers keep reading the quantity they read before; it is the DOCUMENTARY
        # threshold from ops/verifier_gaps.md's gap entry, not the live one. Any claim about
        # LIVE behaviour must use `..._at_live_threshold`.
        "legacy_accuracy_would_pass": bool(float(vr.accuracy) >= 0.5),
        "legacy_accuracy_threshold_documented": 0.5,
        "legacy_accuracy_would_pass_at_live_threshold": bool(float(vr.accuracy) >= 1.0),
        "legacy_accuracy_live_threshold": 1.0,
        "legacy_accuracy_live_threshold_source": (
            "arc_competition_agent.py:5593,5719 min_heldout_accuracy=1.0"
        ),
        "change_accuracy": round(float(vr.change_accuracy), 6),
        "n_changes_correct": int(vr.n_changes_correct),
        "cell_recall": round(float(vr.cell_recall), 6),
        "hud_mask_status": str(vr.hud_mask_status),
        "hud_mask_cells": int(vr.hud_mask_cells),
        # REQ-ARC-WMTE-6015. Present on EVERY gate record, fired or not, so a reader can
        # tell "the guard checked and passed" from "the guard never ran" -- the same
        # measurable-vs-clean distinction `noop_ok_is_vacuous` draws above.
        "hud_mask_swallow": dict(vr.hud_mask_swallow),
        "hud_mask_swallow_guard_fired": bool(
            str(vr.hud_mask_status) == "refused_swallows_dynamics"
        ),
        # REQ-ARC-WMTE-6017. The SECOND refusal status. `hud_mask_swallow_guard_fired` above
        # tests one exact string, so a consumer reading only that field would score an
        # unmeasurable-refusal row as "the guard did not fire" -- true of that field, and
        # misleading about the row. Both are reported, each naming its own condition.
        "hud_mask_swallow_refused_unmeasurable": bool(
            str(vr.hud_mask_status) == "refused_swallow_check_unmeasurable"
        ),
        "hud_mask_swallow_source": str(vr.hud_mask_swallow_source),
    }


@dataclass
class GoalPredicateConsistency:
    """REQ-ARC-WMTE-5593: the goal-hypothesis analog of `VerifyResult` -- checks whether
    `is_level_complete` correctly predicts the SIGN of real observed level transitions
    (a real level-up occurred, or it did not), rather than the DYNAMICS `WorldModelVerifier`
    already checks (does `engine()` predict the right next grid). Nothing in the induction
    pipeline validated the goal predicate against real level-progress ground truth before
    this -- `execute_bounded_llm_reinduction` installs `outcome.goal_predicate` as a search
    termination condition on the strength of the proposer's own code, unchecked against any
    observed transition. This is a direct, literal instance of the project's founding thesis
    (verify a claim against ground truth) applied to the goal-hypothesis half of an induced
    world model, mirroring the docs/research-notes/arc-agi3-milestone1-winners-sota-ingestion-
    2026-07-11.md finding that two independent top-3 teams (Reki, Duck) both carry an
    unexploited self-report-vs-ground-truth gap in their own pipelines.
    """

    n: int
    n_correct: int
    accuracy: float
    n_real_levelups: int
    n_real_noops: int
    mismatches: list[dict] = field(default_factory=list)
    # How many level-up rows were graded on the ENGINE'S COUNTERFACTUAL rather than on the
    # (re-laid-out, therefore wrong) rendered `next_grid` -- see score_goal_predicate_consistency's
    # `engine` parameter. 0 means the historical grading was used, so a caller cannot mistake an
    # unfixed run for a fixed one. Defaulted, so existing constructions stay valid.
    #
    # READ THIS ALONGSIDE `counterfactual_grading_is_not_oracle_distinct` BELOW: a non-zero count
    # here means the veto traded independence for correctness on those rows, and that trade must be
    # visible to anyone reading a verdict off this object.
    n_levelups_graded_on_engine_counterfactual: int = 0
    # Level-up rows that could NOT be graded at all, because the engine supplied to grade their
    # counterfactual did not clear the fidelity floor. They are EXCLUDED from `n`/`n_correct`
    # rather than graded on `next_grid`: grading them on the rendered frame is what penalises a
    # CORRECT predicate (measured -- see the docstring), and grading them on an untrusted engine's
    # counterfactual is what loses oracle-distinctness. Neither is acceptable, so the honest state
    # is "ungradeable", recorded here so a silently-inert veto cannot pass for a satisfied one.
    n_levelups_ungradeable_low_engine_fidelity: int = 0
    # True whenever at least one row was graded on the engine's own counterfactual. On those rows
    # the veto is NO LONGER independent of the engine: engine and goal predicate are emitted by the
    # SAME proposer in the SAME call, so a jointly-confabulated (engine, goal) pair -- engine
    # invents a state, goal recognises exactly that state -- agrees with itself and passes. That
    # pair is caught when this is False and can slip through when it is True.
    counterfactual_grading_is_not_oracle_distinct: bool = False
    # The measured engine-quality number the counterfactual decision was made on, and the floor it
    # was compared against, so the decision is reconstructible from the artifact alone.
    engine_fidelity_used_for_counterfactual_decision: Optional[float] = None
    min_engine_fidelity_for_counterfactual: Optional[float] = None
    # SENSITIVITY (REQ-ARC-WMTE-6257, 2026-08-12). `accuracy` above is a SPECIFICITY score:
    # held-out transitions from `collect_transitions` contain no level-ups, so every row is a
    # non-win and a predicate returning False for everything is 100% correct. Measured: 14 of
    # 21 stored predicates scored a perfect 1.0 that way while never firing on a real win. That
    # matters because `plan_in_model` terminates on this predicate -- 10 of them made planning
    # impossible and 4 more produced hollow plans ending on an in-model false win.
    #
    # These three are the other side. They stay 0/None unless the caller supplies `win_grids`,
    # because "nobody checked" and "checked and fine" must never look alike.
    sensitivity_win_grids_tested: int = 0
    sensitivity_win_grids_fired: int = 0
    # True = specific-looking but fires on no real win. None = UNMEASURABLE (no win grid given).
    is_degenerate_constant_false: Optional[bool] = None


def score_goal_predicate_consistency(
    is_level_complete: Callable[[np.ndarray], bool],
    transitions: Sequence[Transition],
    *,
    max_mismatch: int = 8,
    engine: Optional[Callable[[np.ndarray, int, Any], np.ndarray]] = None,
    engine_change_fidelity: Optional[float] = None,
    min_engine_fidelity_for_counterfactual: float = 0.5,
    win_grids: Optional[Sequence[Any]] = None,
) -> GoalPredicateConsistency:
    """REQ-ARC-WMTE-5593: does `is_level_complete`'s sign match real observed level-ups?

    For each transition, the real ground truth is `level_after > level_before` (a genuine
    level-up occurred at that point). The claim under test is
    `is_level_complete(next_grid)`. Agreement is a cheap, deterministic sign check -- no
    second LLM call, matching forge's own competitive-pressure finding that an expensive
    LLM judge was not worth the cost while a deterministic filter was kept.

    CALLER CONTRACT: pass transitions from a SINGLE level boundary (the level
    `is_level_complete` was induced/re-induced for). It is a per-boundary predicate in the
    real pipeline (`execute_bounded_llm_reinduction` re-induces it after every level-up), so
    checking it against transitions spanning multiple boundaries can produce a spurious
    mismatch if a "win"-looking state persists visually into a later, unrelated boundary.

    ``engine`` (optional, 2026-07-29) -- THE LEVEL-UP ROWS CANNOT BE GRADED ON ``next_grid``.
    For a level-up transition this function asks the predicate to return True on ``t.next_grid``,
    but the completing action re-lays out the playfield atomically with winning, so ``next_grid``
    is the NEXT LEVEL'S OPENING BOARD. Measured on ka59's canonical 11-action L1 solve against a
    change-fidelity-1.0000 engine whose predicate encodes the registry's documented win condition:
    the winning step rewrites 3527 of 4096 cells and the CORRECT predicate is False on
    ``next_grid``. So on exactly the rows that carry the positive signal, this veto was scoring a
    correct predicate as WRONG -- it penalised correctness.

    When ``engine`` is supplied, level-up rows are instead graded on the engine's own
    counterfactual ``engine(t.grid, t.action, t.data)``, which is where the terminal configuration
    actually exists (measured True there on the same ka59 case, a 19-cell ordinary-magnitude step).
    Non-level-up rows are unaffected either way.

    DEFAULT IS UNCHANGED. With ``engine=None`` this function behaves EXACTLY as before, because
    this is a live veto and flipping how it scores would change which rounds reach the planner. The
    call site in ``arc_llm_reinduction`` passes the engine explicitly; every other caller keeps the
    historical behaviour until measured. ``n_levelups_graded_on_engine_counterfactual`` reports how
    many rows took the corrected path, so a run cannot silently claim the fix was active.

    THE COST OF THE COUNTERFACTUAL: IT IS NOT ORACLE-DISTINCT (2026-07-29, second pass).
    Grading a level-up row on ``engine(t.grid, t.action)`` makes the goal veto depend on the ENGINE,
    and the engine and the goal predicate are emitted BY THE SAME PROPOSER IN THE SAME CALL. Mutual
    consistency between them is therefore the EXPECTED confabulation mode, not an independent
    check. Demonstrated: an engine that invents colour 9 on the winning action paired with a goal
    that tests for colour 9 -- a colour that appears in NO real observed frame -- scores 0.5 when
    graded on ``next_grid`` (veto FIRES, pair correctly caught) and 1.0 when graded on the
    counterfactual (veto PASSES). So on exactly the rows that carry the level-up signal, a
    jointly-wrong pair now clears a veto that used to catch it.

    Both grading choices are therefore wrong for an UNTRUSTED engine, in opposite directions:
    ``next_grid`` penalises a CORRECT predicate, the counterfactual excuses a JOINTLY-WRONG one.
    So the counterfactual is gated on the engine having independently earned trust:

      * ``engine_change_fidelity`` is the engine's MEASURED quality on the agent's own held-out
        transitions (``heldout_change_consistency`` at the ``arc_llm_reinduction`` call site) --
        computed WITHOUT reference to the goal predicate, which is what keeps the gating decision
        itself independent even though the grading it authorises is not.
      * At or above ``min_engine_fidelity_for_counterfactual`` the counterfactual is used.
      * Below it -- or when no fidelity number was supplied at all -- the level-up row is treated as
        UNGRADEABLE and EXCLUDED from ``n``/``n_correct``, counted in
        ``n_levelups_ungradeable_low_engine_fidelity``. It is NOT quietly graded on ``next_grid``.

    Excluding makes the veto go INERT on such a window (it only fires at
    ``n_real_levelups >= 1``), which is the lesser evil and the honest one: with no trustworthy
    positive there is nothing to grade against, and firing the veto anyway would reject correct
    predicates -- the very failure that blocked the induce->plan path. ``n_real_levelups`` still
    counts the row (it really was a level-up), so the inertness is visible rather than disguised as
    a clean pass, and ``counterfactual_grading_is_not_oracle_distinct`` flags any run whose numbers
    were produced with the independence traded away.
    """

    n_correct = 0
    n_real_levelups = 0
    n_real_noops = 0
    n_cf = 0
    n_ungradeable = 0
    mismatches: list[dict] = []
    # Whether the engine has independently earned the right to supply the counterfactual. The
    # decision is made ONCE, outside the loop, from a number computed without reference to the goal
    # predicate -- so the gating stays independent even though the grading it authorises does not.
    engine_fidelity_ok = (
        engine is not None
        and engine_change_fidelity is not None
        and float(engine_change_fidelity) >= float(min_engine_fidelity_for_counterfactual)
    )

    def _engine_verified_on_action(action: int) -> bool:
        """Is this engine's behaviour ON THIS ACTION corroborated by REAL observed data?

        The overall fidelity floor is necessary but NOT sufficient, and this is the condition that
        actually blocks the confabulation. A jointly-wrong (engine, goal) pair confabulates on the
        WINNING action specifically -- which is exactly the action whose true effect is unobservable
        from `next_grid`, because the completing action's frame is the next level's re-layout. An
        engine can therefore score well overall and still be entirely unconstrained on the one
        action whose counterfactual we are about to trust.

        So we require independent corroboration for that action: at least one NON-level-up
        transition using the SAME action, whose real observed `next_grid` the engine reproduces
        exactly. Those rows are graded against REALITY, not against the goal predicate, which is
        what makes this check oracle-distinct. If the winning action appears nowhere else in the
        window, there is no evidence and the answer is False -- deliberately conservative: absence
        of evidence about the decisive action is not evidence the engine models it.
        """
        corroborated = False
        for other in transitions:
            if int(other.action) != int(action):
                continue
            if other.level_after > other.level_before:
                continue  # its own next_grid is a re-layout; cannot corroborate anything
            try:
                pred = np.asarray(
                    engine(np.asarray(other.grid).copy(), int(other.action), other.data)
                )
            except Exception:
                return False
            observed = np.asarray(other.next_grid)
            if pred.shape != observed.shape or not np.array_equal(pred, observed):
                return False  # demonstrably wrong on this action -> do not trust its counterfactual
            corroborated = True
        return corroborated

    for i, t in enumerate(transitions):
        real_levelup = bool(t.level_after > t.level_before)
        if real_levelup:
            n_real_levelups += 1
        else:
            n_real_noops += 1
        # Grade the row on the frame where a correct predicate should actually be True. For a
        # level-up that is the engine's counterfactual, NOT the rendered next frame (see docstring).
        graded_grid = t.next_grid
        if real_levelup and engine is not None:
            if not (engine_fidelity_ok and _engine_verified_on_action(int(t.action))):
                # UNGRADEABLE (see docstring). We refuse BOTH available gradings: `next_grid` is
                # the next level's re-laid-out board, on which a CORRECT predicate is False, and an
                # untrusted engine's counterfactual is a state the same proposer may have
                # confabulated to match its own goal predicate. Drop the row instead of scoring it
                # wrongly in either direction.
                n_ungradeable += 1
                if len(mismatches) < max_mismatch:
                    mismatches.append(
                        {
                            "i": i,
                            "real_levelup": True,
                            "claimed": None,
                            "ungradeable": (
                                "engine_fidelity_below_counterfactual_floor"
                                if not engine_fidelity_ok
                                else "engine_unverified_on_this_action"
                            ),
                            "engine_change_fidelity": (
                                None
                                if engine_change_fidelity is None
                                else round(float(engine_change_fidelity), 6)
                            ),
                            "floor": round(float(min_engine_fidelity_for_counterfactual), 6),
                        }
                    )
                continue
            try:
                cf = np.asarray(engine(np.asarray(t.grid).copy(), int(t.action), t.data))
                if cf.shape == np.asarray(t.grid).shape:
                    graded_grid = cf
                    n_cf += 1
                else:
                    # A shape-changing engine cannot have produced this level's terminal board, so
                    # its counterfactual is not usable -- and `next_grid` is still wrong. Same
                    # ungradeable treatment rather than a silent fall back to the wrong frame.
                    n_ungradeable += 1
                    continue
            except Exception:
                # A broken engine must not turn into a goal-predicate verdict, and must not be
                # laundered into a `next_grid` grading either (that is what penalises a correct
                # predicate). The row is ungradeable.
                n_ungradeable += 1
                continue
        try:
            claimed = bool(is_level_complete(graded_grid))
        except Exception as e:
            claimed = False
            if len(mismatches) < max_mismatch:
                mismatches.append(
                    {"i": i, "real_levelup": real_levelup, "claimed": None, "error": repr(e)[:160]}
                )
            continue
        if claimed == real_levelup:
            n_correct += 1
        elif len(mismatches) < max_mismatch:
            mismatches.append({"i": i, "real_levelup": real_levelup, "claimed": claimed})

    # Ungradeable rows leave the DENOMINATOR too. Keeping them in `n` would depress `accuracy`
    # toward the veto threshold for a reason that has nothing to do with the predicate under test --
    # which is the same "penalise correctness" failure this whole correction exists to remove.
    n = len(transitions) - n_ungradeable
    # NOTHING GRADEABLE MUST NOT READ AS MAXIMALLY INCONSISTENT -- but only when rows were actually
    # EXCLUDED. With every level-up row dropped, `n == 0`, and the obvious `n_correct / max(1, n)`
    # yields 0.0, which trips any `accuracy < threshold` veto: the veto would fire HARDEST exactly
    # when it has no evidence, re-creating the reject-correct-predicates failure through the
    # denominator instead of the numerator. So an all-excluded window reads 1.0 (vacuously
    # consistent) and `n == 0` is the discriminator a consumer should test.
    #
    # An EMPTY INPUT is a different case and keeps its historical 0.0 (REQ-ARC-WMTE-5593's
    # empty-transitions contract, asserted by test_arc_goal_predicate_consistency). "You gave me no
    # transitions" and "your transitions were all ungradeable" are distinct claims, and collapsing
    # them would silently change a documented return value for every existing caller.
    all_rows_excluded = bool(len(transitions) > 0 and n == 0)
    # SENSITIVITY. Measurable only when the caller supplies grids on which a level-up REALLY
    # happened (`replay_win_transition` produces them). Absent those, the fields stay unmeasured
    # rather than defaulting to a pass -- a metric that cannot fail is not a metric, which is
    # the entire reason this side exists.
    _sens_tested = 0
    _sens_fired = 0
    for _wg in win_grids or []:
        _sens_tested += 1
        try:
            if bool(is_level_complete(np.asarray(_wg))):
                _sens_fired += 1
        except Exception:  # noqa: BLE001
            # A predicate that RAISES on a real win state has not fired on it. Counting the
            # test but not the fire matches how a caller experiences it: the planner would not
            # terminate here either.
            pass
    _spec = 1.0 if all_rows_excluded else float(n_correct / max(1, n))
    _degenerate = None if _sens_tested == 0 else bool(_sens_fired == 0 and _spec >= 0.9)
    return GoalPredicateConsistency(
        n=n,
        n_correct=n_correct,
        accuracy=1.0 if all_rows_excluded else float(n_correct / max(1, n)),
        n_real_levelups=n_real_levelups,
        n_real_noops=n_real_noops,
        mismatches=mismatches,
        n_levelups_graded_on_engine_counterfactual=n_cf,
        n_levelups_ungradeable_low_engine_fidelity=n_ungradeable,
        counterfactual_grading_is_not_oracle_distinct=bool(n_cf > 0),
        engine_fidelity_used_for_counterfactual_decision=(
            None if engine_change_fidelity is None else float(engine_change_fidelity)
        ),
        min_engine_fidelity_for_counterfactual=(
            float(min_engine_fidelity_for_counterfactual) if engine is not None else None
        ),
        sensitivity_win_grids_tested=_sens_tested,
        sensitivity_win_grids_fired=_sens_fired,
        is_degenerate_constant_false=_degenerate,
    )


def predict_hypothesis_transition(
    hypothesis: Any,
    grid: np.ndarray,
    action: int,
    data: Any = None,
) -> np.ndarray:
    """REQ-ARC-WMTE-4727: run one hypothesis' transition model for a candidate probe.

    Active probing needs a narrow, oracle-distinct prediction API: given a
    candidate dynamics hypothesis and a possible live action, return what that
    hypothesis says the next logical grid will be. This helper deliberately
    does not inspect `is_level_complete`; probe routing is about transition
    consequences, not asking the environment's win oracle.
    """

    engine = getattr(hypothesis, "engine", hypothesis)
    if not callable(engine):
        raise TypeError("hypothesis_transition_engine_not_callable")
    return np.asarray(engine(np.asarray(grid).copy(), int(action), data))


def load_engine(game: str):
    """Import the codex-written world_model.py for a game and return (engine,
    is_level_complete). Re-imports fresh each call so a refactor is picked up."""
    import importlib.util

    return _load_engine_from(E3_DIR, game)


def _load_engine_from(root: Path, game: str):
    import importlib.util

    p = Path(root) / game / "world_model.py"
    if not p.exists():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location(f"arc_wm_{game}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return getattr(mod, "engine"), getattr(mod, "is_level_complete", None)


def load_origin_fixture_engine(game: str):
    """Load a GAP-WM-TRUST-GATE origin-incident engine from the FROZEN, never-written copy.

    REQ-ARC-WMTE-6016. `load_engine` reads the MUTABLE store, which any induction run
    rewrites in place. A guard asserted against the mutable store is therefore not asserted
    against its own origin incident for long: on 2026-07-27 a live A/B replaced ft09's
    12-bare-`return grid` identity engine with a 2-branch mutating one within hours of the
    artifact that cited it, and the same run rewrote lp85 -- the ONE game whose degenerate
    engine actually discriminates -- twice.

    A guard that stops firing on the incident that motivated it is the failure mode this
    project has shipped more than once. Reading from `E3_ORIGIN_FIXTURES_DIR` makes the
    origin-incident assertion permanent by construction rather than by luck.
    """

    return _load_engine_from(E3_ORIGIN_FIXTURES_DIR, game)


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


def _rle_delta(g0: np.ndarray, g1: np.ndarray) -> str:
    """LOSSLESS run-length delta for the induce prompt: every changed cell, encoded as maximal
    horizontal runs 'r<row>c<col0>:<new,values>' (values comma-separated so colors >=10 stay
    unambiguous). This REPLACES the old cap=80 raw-tuple delta in the induction evidence, which
    silently TRUNCATED large changes — a 293-cell re-render showed only the first 80 cells (27%),
    starving the model of the very evidence it needs to induce the rule. RLE shows the FULL change
    at ~1/4 the tokens of raw per-cell tuples, so the whole change fits the local model's context
    with no truncation. (The verifier's mismatch examples still use the capped _delta on purpose —
    those are illustrative, not the load-bearing induction evidence.)"""
    g0 = np.asarray(g0)
    g1 = np.asarray(g1)
    if g0.shape != g1.shape:
        return ""
    diff = g0 != g1
    h, w = g0.shape
    runs = []
    for r in range(h):
        c = 0
        while c < w:
            if diff[r, c]:
                c0 = c
                while c < w and diff[r, c]:
                    c += 1
                vals = ",".join(str(int(v)) for v in g1[r, c0:c])
                runs.append(f"r{r}c{c0}:{vals}")
            else:
                c += 1
    return " ".join(runs) if runs else "(no change)"


def _rle_delta_compact(g0: np.ndarray, g1: np.ndarray) -> str:
    """Like `_rle_delta`, but each changed run's NEW values are themselves run-length-collapsed
    as '<value>x<count>' pairs instead of listed one-per-cell. `_rle_grid` fixed
    `induce_prompt`'s full-grid cost, but on lp85's real transitions the per-transition DELTAS
    then became the dominant remaining cost (measured: 8 deltas via `_rle_delta` = 9,308 tokens,
    still over the 13,824-token budget after the full-grid fix) -- large changes are often a
    single-color object moving or a solid region appearing, which `_rle_delta`'s raw
    comma-per-cell listing cannot exploit but this can (measured: same 8 deltas =
    5,992 tokens, a 3,316-token additional saving, closing the remaining gap). Kept as a
    SEPARATE function from `_rle_delta` rather than changing its output format in place --
    `_rle_delta` has its own round-trip tests (test_rle_delta_lossless.py) and another caller
    (scripts/experiments/arc_frontier_tooluse_probe.py) that expect the existing one-value-per-
    comma format; this function is used only by `_transitions_block`'s induction-evidence path.
    The run's starting column stays explicit (unlike `_rle_grid`'s implicit-column full-row
    encoding): a row can have multiple disjoint CHANGED spans separated by unchanged cells, so
    the column position is not implicit here the way it is when every cell in a row is covered."""
    g0 = np.asarray(g0)
    g1 = np.asarray(g1)
    if g0.shape != g1.shape:
        return ""
    diff = g0 != g1
    h, w = g0.shape
    runs = []
    for r in range(h):
        c = 0
        while c < w:
            if diff[r, c]:
                c0 = c
                while c < w and diff[r, c]:
                    c += 1
                sub = []
                i = c0
                while i < c:
                    v = g1[r, i]
                    j = i
                    while j < c and g1[r, j] == v:
                        j += 1
                    sub.append(f"{int(v)}x{j - i}")
                    i = j
                runs.append(f"r{r}c{c0}:" + ",".join(sub))
            else:
                c += 1
    return " ".join(runs) if runs else "(no change)"


# Characters, not tokens, because that is what this module can count without loading a
# tokenizer. ~4 chars/token puts 40,000 chars near 10,000 tokens, against a REAL prompt budget of
# 16,384 (n_ctx 81920, shared 4-way under kv_unified, minus max_tokens 4096). Deliberately a
# backstop rather than a target: the measured worst case is ~5,000 chars, so this should never
# bind. It exists so a future game with far larger deltas degrades by dropping the least
# informative transitions instead of overflowing the context pool.
_INDUCE_TRANSITION_CHAR_BUDGET = 40000


def _transition_salience_key(t: Transition) -> tuple:
    """Ordering key for WHICH transitions to keep when the char budget forces a cut.

    Mage-VL selects patches by residual energy -- how much a region actually moved -- rather than
    by position. The analogue here is that a transition teaches the model something only insofar
    as it is not already explained by one already shown, so the rank is by NOVELTY of the
    (action, change-shape) pair rather than by arrival order. `changed[:6]` is arrival order,
    which is the thing being replaced.

    Returns a sort key; the caller dedupes on the leading signature. Kept deliberately cheap --
    grid comparison, no encoding -- because it runs on every induce call.
    """

    diff = np.asarray(t.grid) != np.asarray(t.next_grid)
    n_changed = int(diff.sum())
    rows = np.flatnonzero(diff.any(axis=1)) if diff.ndim == 2 else np.array([])
    cols = np.flatnonzero(diff.any(axis=0)) if diff.ndim == 2 else np.array([])
    span = (
        (int(rows[-1] - rows[0]) + 1 if rows.size else 0),
        (int(cols[-1] - cols[0]) + 1 if cols.size else 0),
    )
    return (int(t.action), n_changed, span)


def _select_transitions_for_prompt(
    changed: list[Transition],
    noop: list[Transition],
    *,
    k: Optional[int] = None,
    char_budget: Optional[int] = None,
) -> list[Transition]:
    """Choose the transitions the induce prompt shows.

    ORDER OF PREFERENCE, and each step exists for a measured reason:

      1. Every CHANGED transition, in observation order. Observation order is kept because the
         prompt reads as a narrative of what the agent did, and reordering it would imply a
         sequence the agent never took.
      2. Up to two NO-OPS. Unchanged from before: a couple of inert actions tell the model the
         game HAS inert actions, and more than a couple is just budget.
      3. If and only if the result exceeds the char budget, drop the LEAST NOVEL changed
         transitions -- those whose (action, change-shape) signature is already represented --
         rather than simply truncating. Truncation is what `changed[:k-2]` did.

    An explicit `k` still caps the count, so every existing caller and test that pins one keeps
    exactly its old behaviour.
    """

    if k is None:
        # THE WIRING that REQ-ARC-FCP-5699-23 was missing: the live prompt now reads the same
        # resolver the diagnostics have been reading since that REQ shipped.
        k = _induce_transitions_k()
    budget = _INDUCE_TRANSITION_CHAR_BUDGET if char_budget is None else int(char_budget)

    keep_noop = noop[:2]
    if k is not None:
        # LITERALLY the historical expression, `changed[: k - 2] + noop[:2]`, and it must stay
        # literal. The obvious "improvement" -- reserving only as many slots as there are no-ops,
        # `changed[: k - len(keep_noop)]` -- is a BEHAVIOUR CHANGE on any game with no inert
        # actions: tn36 and tu93 have 25 changed and 0 no-ops, so it yields 8 changed where the
        # old code yielded 6. That silently makes `k=8` mean something new and destroys the A/B
        # switch this parameter exists to provide. Written wrong first, caught by comparing
        # against the shipped renderer rather than against the new one.
        return changed[: max(0, int(k) - 2)] + keep_noop

    selected = list(changed)

    # Cheap upper bound on cost: the delta encoding dominates, so estimate from it directly
    # rather than rendering the whole block repeatedly.
    def _cost(t: Transition) -> int:
        try:
            return len(_rle_delta_compact(t.grid, t.next_grid))
        except Exception:  # noqa: BLE001 - a cost estimate must never break the prompt
            return 0

    total = sum(_cost(t) for t in selected)
    if total > budget and selected:
        # Over budget: keep first occurrence of each signature, then re-add the rest in
        # observation order until the budget is spent. Never drops below one transition.
        seen: set[tuple] = set()
        primary: list[Transition] = []
        secondary: list[Transition] = []
        for t in selected:
            sig = _transition_salience_key(t)
            (primary if sig not in seen else secondary).append(t)
            seen.add(sig)
        kept: list[Transition] = []
        spent = 0
        for t in primary + secondary:
            c = _cost(t)
            if kept and spent + c > budget:
                continue
            kept.append(t)
            spent += c
        # Restore observation order: selection is by novelty, presentation is chronological.
        order = {id(t): i for i, t in enumerate(selected)}
        selected = sorted(kept, key=lambda t: order[id(t)])

    return selected + keep_noop


def _transitions_block(
    trans: list[Transition],
    k: Optional[int] = 8,
    *,
    previous_level_complete_grid: Optional[np.ndarray] = None,
    hud_mask: Optional[np.ndarray] = None,
    hud_mask_enabled: Optional[bool] = None,
    # The level-up transition, supplied rather than found. `None` preserves the historical
    # behaviour exactly: scan `trans` for `level_after > level_before`. See the WIN TRANSITION
    # block below for why supplying it is not the same as appending it to `trans`.
    win_transition: Optional[Transition] = None,
) -> str:
    """Compact transition encoding for the induce prompt: ONE full grid (the layout) +
    per-transition DELTAS (changed cells), + the full WIN state if observed. Prefers
    grid-CHANGING transitions; keeps a couple of no-ops. Small enough for a local model's
    context window (the raw one-char-per-cell full-grid form overflowed gemma-4-12B at ~67k
    tokens on small boards, and on large boards like lp85's 64x64 grid overflowed even the
    13,824-token available budget with a SINGLE transition — see `_rle_grid`'s docstring; both
    full-grid renders below use the run-length encoding instead). Deltas use
    `_rle_delta_compact` (not `_rle_delta`) — on lp85's real transitions the raw comma-per-cell
    delta format became the new dominant cost once the full-grid fix landed (see
    `_rle_delta_compact`'s docstring for the measured before/after).

    REQ-ARC-WMTE-6010 PROMPT COHERENCE (added 2026-07-27 after an adversarial review found
    the mask was half-applied). `changed` / `noop` below decide WHICH transitions the LLM is
    shown, and until this date they were computed on RAW grids while the VERIFIER that
    grades the induced engine compared MASKED grids. On a game with a monotone HUD counter
    the two disagree completely: measured on real offline transitions, ft09 has 0 raw no-ops
    and 32 masked no-ops, lf52 has 0 raw and 59 masked. So the prompt asserted "this game has
    no inert actions" and showed six examples labelled as changes, while the grader had
    already decided those same six transitions did not change anything. The model was being
    asked to explain one world and marked against another. `_rle_delta_compact` renders on
    the RAW grids deliberately -- the LLM must still see the true pixels, including the HUD;
    only the CLASSIFICATION of a transition as changing-vs-inert is masked, which is the
    thing the verifier also classifies.

    Default `hud_mask_enabled=None` resolves through the same single flag resolver as every
    other consumer, so with the flag off this function is byte-identical to before.
    """
    if hud_mask_enabled is None:
        hud_mask_enabled = world_model_hud_mask_enabled()
    mask = hud_mask if hud_mask_enabled else None

    def _is_changed(t: Transition) -> bool:
        return not np.array_equal(apply_hud_mask(t.grid, mask), apply_hud_mask(t.next_grid, mask))

    changed = [t for t in trans if _is_changed(t)]
    noop = [t for t in trans if not _is_changed(t)]
    sample = _select_transitions_for_prompt(changed, noop, k=k)
    out = []
    if sample:
        out.append(
            "INITIAL GRID (one full example of the state layout, run-length encoded; "
            "all grids are this shape):\n" + _rle_grid(sample[0].grid)
        )
    for t in sample:
        click = f" data={t.data}" if t.data else ""
        out.append(
            f"--- ACTION{t.action}{click} (level {t.level_before}->{t.level_after}): "
            f"changed cells (FULL, run-length) = {_rle_delta_compact(t.grid, t.next_grid)}"
        )
    # `win_transition` lets a caller SUPPLY the level-up transition instead of relying on one
    # being present in `trans`. That is not a convenience -- on the live path the win transition
    # is structurally absent from `trans`, so this block could never fire (see the parameter's
    # docstring note). Falling back to the scan keeps every existing caller byte-identical.
    #
    # IT IS A SEPARATE PARAMETER, NOT AN APPEND TO `trans`, and that is the load-bearing choice.
    # `changed`/`noop` above are built from `trans`, so a level-up transition placed there would
    # ALSO be rendered as an ordinary dynamics example -- and the completing action re-lays out
    # the whole playfield (measured: 3527 of 4096 cells on ka59, against an ordinary-step median
    # of 18.5). Teaching the proposer that a single action can change 86% of the board is a worse
    # defect than the one being fixed.
    win = win_transition
    if win is None:
        win = next((t for t in trans if t.level_after > t.level_before), None)
    if win is not None:
        # THE WIN STATE IS NOT A RENDERED FRAME (measured 2026-07-29). This block used to emit
        # `win.next_grid` under the label "is_level_complete must return True here". That claim is
        # FALSE, and it was teaching the proposer a wrong win concept.
        #
        # The completing action does two things ATOMICALLY: it satisfies the level's win condition
        # AND it re-lays out the playfield for the next level. So `next_grid` is the NEXT LEVEL'S
        # OPENING BOARD, not a picture of this level completed. Measured on ka59's canonical
        # 11-action L1 solve with the change-fidelity-1.0000 engine whose predicate encodes the
        # registry's documented win condition exactly: the winning step changes 3527 of 4096 cells
        # (86% of the grid) against an ordinary-step median of 18.5 -- a full re-layout -- and the
        # correct predicate is False on `next_grid`. Same shape on the only other two games whose
        # rollout window contains a real level-up (lp85 37.7%, r11l 34.1%).
        #
        # The obvious off-by-one "use the frame BEFORE the increment" (`win.grid`) is ALSO WRONG and
        # was tested and refuted: `win.grid` is one action short of completion, where a correct
        # predicate MUST be False. Measured: the correct predicate is False on `win.grid`, False on
        # `win.next_grid`, and False on ALL 12 observed frames -- but TRUE on
        # `engine(win.grid, win.action)`, a 19-cell ordinary-magnitude step. The terminal
        # configuration exists in the MODEL; the renderer never shows it.
        #
        # So we emit the one thing that IS trustworthy: the labelled TRANSITION EVENT. We give the
        # board before the completing action plus the action, and state the constraint as a joint
        # condition on engine AND is_level_complete. That is strictly more information than the old
        # block (it names the action too) and it asserts nothing false.
        click = f" data={win.data}" if win.data else ""
        out.append(
            f"WIN TRANSITION (this is how the level was completed): applying ACTION{win.action}"
            f"{click} to the grid below completed the level (level {win.level_before}->"
            f"{win.level_after}).\n"
            "CONSTRAINT: is_level_complete(engine(GRID_BELOW, "
            f"{win.action}{', data' if win.data else ''})) must return True.\n"
            "Do NOT assume the rendered next frame is the win state -- the completing action also "
            "re-lays out the board for the next level, so the completed configuration of THIS "
            "level is never drawn. Reason about what the winning move accomplishes, not about how "
            "the following screen looks.\n"
            "GRID BEFORE THE COMPLETING ACTION (run-length encoded):\n" + _rle_grid(win.grid)
        )
    elif previous_level_complete_grid is not None:
        # RELABELLED TRUTHFULLY (2026-07-29). This grid is captured by
        # `arc_competition_agent._observe_level_boundary` from the frame AFTER the level counter
        # incremented, so by the same re-layout measurement above it is the OPENING BOARD OF THE
        # LEVEL THAT JUST STARTED -- not a picture of the previous level completed. Describing it as
        # "a state that COMPLETED the previous level" told the proposer that a level-complete state
        # looks like a fresh board, which is the opposite of the truth.
        # It is still genuinely useful -- it shows the object vocabulary, palette and geometry the
        # game draws with -- so it is kept and described for what it actually is, with no claim
        # about is_level_complete's value on it.
        out.append(
            "BOARD AT THE START OF THE CURRENT LEVEL (full grid, run-length encoded; captured "
            "just after the previous level completed, so this is the CURRENT level's opening "
            "layout). Use it for the object vocabulary, palette and geometry. It is NOT a "
            "level-complete state -- is_level_complete must return False here, because a level is "
            "not complete at its opening screen:\n"
            + _rle_grid(np.asarray(previous_level_complete_grid))
        )
    return "\n".join(out)


def objects_block(
    trans: list[Transition],
    *,
    previous_level_complete_grid: Optional[np.ndarray] = None,
    max_objects: int = 60,
) -> str:
    """LEVER #1 (REQ-ARC-WMTE-5830): object-structured serialization of the layout grid (and WIN state,
    if observed) for the induction prompt. Reuses `arc_color_blob_salience.blob_topology` unchanged.
    Objects are the connected-component partition; `object_hash` is a TRANSLATION-INVARIANT shape id so
    the LLM can recognize the SAME object across frames after it moves -- the raw run-length grid gives
    only order-1 position features that cannot. Defensive by construction: any failure returns "" so
    induction falls back to the raw-grid-only prompt (never breaks the default path), and the per-grid
    object table is capped at `max_objects` to bound prompt length on dense/large boards."""
    try:
        from carnot.agentic.arc_color_blob_salience import blob_topology
    except Exception:
        return ""
    changed = [t for t in trans if not np.array_equal(t.grid, t.next_grid)]
    layout = (changed[0] if changed else trans[0]).grid

    def _table(grid: np.ndarray, title: str) -> str:
        topo = blob_topology(np.asarray(grid))
        blobs = topo.get("blobs", [])
        hashes = topo.get("object_hashes", {})
        n = len(blobs)
        # Show the largest-by-pixel objects first; keep ORIGINAL ids so containment/adjacency stay valid.
        order = sorted(range(n), key=lambda i: -int(getattr(blobs[i], "pixel_count", 0)))[
            :max_objects
        ]
        shown = set(order)
        header = (
            f"{title} OBJECTS (connected components; obj<id>: color bbox=(y0,x0,y1,x1) px=<pixels> "
            f"shape=<translation-invariant id>)"
        )
        if n > len(order):
            header += f"  [showing largest {len(order)} of {n}]"
        rows = [header + ":"]
        for i in order:
            b = blobs[i]
            cy, cx = getattr(b, "centroid", (0.0, 0.0))
            rows.append(
                f"  obj{i}: color={int(b.color)} bbox={tuple(int(v) for v in b.bbox)} "
                f"px={int(b.pixel_count)} centroid=({float(cy):.1f},{float(cx):.1f}) shape={hashes.get(i)}"
            )
        children = {p: cs for p, cs in topo.get("children", {}).items() if cs and p in shown}
        adjacency = [
            pair for pair in topo.get("adjacency_list", []) if all(j in shown for j in pair)
        ]
        rows.append(f"  containment (parent->children): {children}")
        rows.append(f"  adjacency (touching id pairs): {adjacency}")
        rows.append(
            "  NOTE: two objects with the SAME shape id are the SAME object type regardless of "
            "position; use this to track objects across the transition deltas above."
        )
        return "\n".join(rows)

    try:
        parts = [_table(layout, "INITIAL")]
        # Same correction as `_transitions_block` (2026-07-29): neither candidate grid here is a
        # win state. `t.next_grid` on a level-up is the NEXT level's opening board (the completing
        # action re-lays out the playfield -- 86% of cells on ka59), and
        # `previous_level_complete_grid` is captured after the counter incremented, so it is the
        # CURRENT level's opening board. Labelling either "WIN STATE" and handing the proposer an
        # object table for it taught a wrong win concept in object space exactly as the raw-grid
        # block did in pixel space. Both are still emitted -- the object vocabulary is the useful
        # part -- under labels that say what they actually are.
        # TOKEN BUDGET: exactly ONE extra table, same as before this fix. The post-level-up board is
        # deliberately NOT emitted -- it is the NEXT level's layout, so it is irrelevant to inducing
        # THIS level's win condition, and a third table would lengthen a prompt that already caused
        # Qwen3.5-9B to spend its whole budget on win-state chain-of-thought before reaching the code
        # block (see _L2_CODEONLY_DIRECTIVE's history). Removing the poison must not reintroduce the
        # truncation failure it sits next to.
        win_t = next((t for t in trans if t.level_after > t.level_before), None)
        if win_t is not None:
            parts.append(_table(win_t.grid, "BOARD BEFORE THE COMPLETING ACTION"))
        elif previous_level_complete_grid is not None:
            parts.append(
                _table(np.asarray(previous_level_complete_grid), "CURRENT LEVEL OPENING BOARD")
            )
        return "\n\n".join(parts)
    except Exception:
        return ""  # never break the default induction path


# Actions 1-6 per the ARC-AGI-3 competition convention. Action 7 exists in `arcengine`'s
# `GameAction` enum but has no established semantic meaning anywhere in this project's own
# docs or code -- left as a bare integer rather than guessed.
_SEMANTIC_ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SPACE", 6: "MOUSE"}


def _action_label(action: int) -> str:
    """`ACTIONn(NAME)` for actions 1-6, bare `ACTIONn` otherwise. Keeps `ACTIONn` as a literal
    substring (nothing that already greps for that pattern breaks) while adding the semantic
    name a general-purpose LLM's own priors can use -- an opaque integer triggers none of
    them."""
    name = _SEMANTIC_ACTION_NAMES.get(int(action))
    return f"ACTION{action}({name})" if name else f"ACTION{action}"


def _n_changed_cells(g0: np.ndarray, g1: np.ndarray) -> int:
    a, b = np.asarray(g0), np.asarray(g1)
    if a.shape != b.shape:
        return int(a.size)  # a shape mismatch is a maximal, not a zero, disagreement
    return int(np.sum(a != b))


def _action_semantics_and_counts_block(
    trans: list[Transition],
    k: Optional[int] = 8,
    *,
    hud_mask: Optional[np.ndarray] = None,
    hud_mask_enabled: Optional[bool] = None,
) -> str:
    """LEVER (REQ-ARC-WMTE-6241). Additive appendix: semantic action names plus an explicit
    changed-cell COUNT per sampled transition, as separate evidence from `_transitions_block`'s
    own RLE delta string. A SEPARATE function rather than a modification of `_transitions_block`
    itself, so that function's own extensively-documented historical correctness fixes are never
    touched by this lever -- this re-samples via the SAME `_select_transitions_for_prompt` call
    with the same inputs, so the numbered rows line up with `_transitions_block`'s own entries in
    the assembled prompt. Returns "" (and is never called) unless
    `induce_prompt_enrichment_enabled()` is True, so the assembled prompt is byte-identical to
    before this lever whenever the flag is off."""
    if hud_mask_enabled is None:
        hud_mask_enabled = world_model_hud_mask_enabled()
    mask = hud_mask if hud_mask_enabled else None

    def _is_changed(t: Transition) -> bool:
        return not np.array_equal(apply_hud_mask(t.grid, mask), apply_hud_mask(t.next_grid, mask))

    changed = [t for t in trans if _is_changed(t)]
    noop = [t for t in trans if not _is_changed(t)]
    sample = _select_transitions_for_prompt(changed, noop, k=k)
    if not sample:
        return ""
    rows = [
        "ACTION SEMANTICS AND CHANGE COUNTS (same sampled transitions as above, in the same "
        "order; UP/DOWN/LEFT/RIGHT/SPACE/MOUSE per the ARC-AGI-3 action convention -- action 7 "
        "has no established semantic name and is left as a bare integer):"
    ]
    for i, t in enumerate(sample, start=1):
        rows.append(
            f"  #{i}: {_action_label(t.action)} -> changed_cells={_n_changed_cells(t.grid, t.next_grid)}"
        )
    return "\n".join(rows)


def _object_identity_crossref_note(
    trans: list[Transition],
    *,
    previous_level_complete_grid: Optional[np.ndarray] = None,
) -> str:
    """LEVER (REQ-ARC-WMTE-6241). Additive appendix: which object shape ids (per `object_hash`,
    from `objects_block`'s own tables) appear in BOTH the INITIAL grid's table and the second
    table `objects_block` renders (the win/current-level-opening board). `objects_block` already
    carries a text HINT ("two objects with the SAME shape id are the SAME object type") but never
    computes the actual intersection; this does, as a short explicit list, so the model does not
    have to eyeball-match hashes across two separate tables itself. Defensive by construction,
    same as `objects_block`: any failure returns ""."""
    try:
        from carnot.agentic.arc_color_blob_salience import blob_topology
    except Exception:
        return ""
    if not trans:
        return ""
    changed = [t for t in trans if not np.array_equal(t.grid, t.next_grid)]
    layout = (changed[0] if changed else trans[0]).grid
    win_t = next((t for t in trans if t.level_after > t.level_before), None)
    if win_t is not None:
        second = win_t.grid
    elif previous_level_complete_grid is not None:
        second = np.asarray(previous_level_complete_grid)
    else:
        return ""
    try:
        h1 = set(blob_topology(np.asarray(layout)).get("object_hashes", {}).values())
        h2 = set(blob_topology(np.asarray(second)).get("object_hashes", {}).values())
    except Exception:
        return ""
    shared = sorted(h1 & h2)
    if not shared:
        return (
            "OBJECT IDENTITY CROSS-REFERENCE: no shape id is shared between the two object "
            "tables above."
        )
    return (
        f"OBJECT IDENTITY CROSS-REFERENCE: {len(shared)} shape id(s) appear in BOTH object "
        f"tables above (the same object type persists across the two frames): {shared}"
    )


# A forceful CODE-ONLY directive for the L2+ induction call. The L2 induce prompt carries a WIN
# STATE exemplar, which makes Qwen3.5-9B burn its ENTIRE token budget on win-state chain-of-thought
# before reaching the code block (stop_type='limit', 0 code emitted -> goal_predicate_satisfiable
# stays False for ~10 milestones; see proto_l2_proposer_truncation_check + proto_l2_code_only_prefix,
# 2026-06-25). Prepending this directive AND adding a stop-sequence on the closing fence makes the
# model emit ONLY the code and stop: verified 195 tokens / 15.6s (vs 605s rambling / 450s truncated),
# valid engine+is_level_complete. DEFAULT ON (2026-06-25 operator directive); opt out with
# CARNOT_ARC_CODEONLY_INDUCE=0. NB: defeats truncation (emits code) but the induced goal predicate
# can still be degenerate -> see the goal-repair loop in arc_llm_reinduction.execute_bounded_llm_reinduction.
_L2_CODEONLY_DIRECTIVE = (
    "/no_think\n"
    "CRITICAL OUTPUT RULES -- obey EXACTLY:\n"
    "1. Output ONLY one ```python code block. NOTHING before it. NOTHING after it.\n"
    "2. Do NOT analyze the grids. Do NOT describe or reason about the win state. Do NOT write\n"
    "   step-by-step analysis, explanation, or commentary -- not even as comments.\n"
    "3. Your response MUST begin with the characters ```python and end with ```.\n"
    "4. Induce SIMPLE, GENERAL rules and write the requested function(s) directly. Skip all reasoning.\n\n"
)


def _induce_transitions_k() -> Optional[int]:
    """How many transitions the induce prompt shows. `None` means ALL of them (the default since
    2026-08-01). An int caps the sample at `changed[:k-2] + noop[:2]`, the historical shape.

    IT WAS 8, AND -- WORSE -- IT WAS NOT WIRED. REQ-ARC-FCP-5699-22 found that the cap "starves
    the dynamics half to roughly one example per action type, producing hardcoded-literal-
    coordinate memorization instead of general rules". REQ-ARC-FCP-5699-23 responded by adding
    this resolver and `CARNOT_ARC_INDUCE_TRANSITIONS_K` so a diagnostic could test raising it.
    But NOTHING ON THE LIVE PATH EVER CALLED IT: `induce_prompt` carried a literal `k: int = 8`,
    and the only callers of this function were `python/carnot/experiment_*.py`. So the knob moved
    the prompt for experiments and did nothing whatsoever for the agent -- the diagnosis landed,
    the instrument was built, and the fix never reached the thing being diagnosed. It is wired
    now, via `induce_prompt(k=None)` -> `_select_transitions_for_prompt`.

    WHY THE DEFAULT IS NOW "ALL" (measured 2026-08-01; prompted by Mage-VL, arXiv 2607.24904).
    That paper's video result is that once you encode an anchor frame plus DELTAS for the rest,
    subsampling frames stops being necessary -- it keeps every frame at ~1/8 the tokens and gains
    accuracy. This module already did the first half: `_transitions_block` emits ONE full grid
    plus per-transition deltas. It then subsampled anyway, discarding 17 of 25 transitions.

    The cap was correct when written -- before `_rle_grid`, a SINGLE 64x64 transition overflowed
    the budget. After the run-length fixes it is not. Rendering ALL transitions instead of 8,
    measured on the six captured games:

        game   n   changed   k=8 chars   ALL chars   ratio
        ft09  25         6       4,046       4,046   1.00x    <- already showed everything
        lp85  25         2       4,064       4,064   1.00x    <- already showed everything
        sc25  25         7       2,557       2,718   1.06x
        tn36  25        25       3,218       5,023   1.56x
        tu93  25        25       2,532       4,924   1.94x
                                             TOTAL   1.24x

    Worst case ~5,000 chars, about 1,255 tokens, against a 16,384-token prompt budget. The cap
    was discarding 68% of the evidence to save roughly 4% of a budget nothing was near; the
    prompt uses under 8% of what is available.

    WHAT THIS DOES *NOT* CLAIM. Action coverage was checked and was NOT being lost -- `changed[:6]`
    already covers every distinct action on all six games, so the stronger story ("whole actions
    were hidden from the model") is false and is not being told. What the extra transitions buy
    is more examples of the SAME actions, which matters for a positional mechanic (tn36 is 25
    clicks whose effect depends on WHERE you click, of which 6 were shown) and is exactly the
    starvation REQ-5699-22 described. Whether it actually improves induction is an empirical
    question this change makes measurable; the token arithmetic alone does not settle it.

    `CARNOT_ARC_INDUCE_TRANSITIONS_K=8` restores the previous prompt byte-for-byte.
    """
    import os

    override = os.environ.get("CARNOT_ARC_INDUCE_TRANSITIONS_K")
    if override is None:
        return None
    text = str(override).strip().lower()
    if text in {"all", "none", ""}:
        return None
    try:
        value = int(text)
    except (TypeError, ValueError):
        return None
    # A non-positive cap would render an empty transitions block -- an induce prompt with no
    # evidence at all. Treat it as "all" rather than as "show nothing".
    return value if value > 0 else None


def _object_perception_on() -> bool:
    """LEVER #1 (REQ-ARC-WMTE-5830, DEFAULT ON since 2026-08-07). When on, induce_prompt appends a
    connected-component OBJECT table (translation-invariant object_hash for cross-frame identity,
    containment tree, adjacency) ALONGSIDE the raw run-length grid -- feeding the inducer the object
    structure that today only feeds the (gated-off) search salience prior. Attacks
    GAP-ARCH-FEATURES: the raw grid gives the LLM order-1 position-only features (can't track an
    object across frames after it moves); object_hash can.

    DEFAULT FLIPPED ON (2026-08-07, operator directive: adopt the Duck/TAAF leaderboard lesson --
    object-level, not raw-grid, perception -- lever #1 of the ARC six-lever push,
    ops/known-issues.md). Evidence, not a guess: the pre-registered 2026-08-01 held-out A/B
    (results/outer_loop_arc_object_perception_heldout_ab_change_fidelity_20260801.json) measured
    this EXACT flag SIGNIFICANT on its primary metric -- change_fidelity mean delta +0.072084,
    sign-test p=0.0192 over 19/20 discordant games (min reachable p=3.81e-06), on gemma-4-31B,
    13171.65s real GPU wall time, AA control byte-identical, clean on adversarial re-check. That
    artifact shipped with the flag deliberately left off ("moving it is a separate operator
    decision"); this is that decision, made on the evidence already in hand. Opt out with
    CARNOT_ARC_OBJECT_PERCEPTION=0."""
    import os

    return os.environ.get("CARNOT_ARC_OBJECT_PERCEPTION", "1") != "0"


def _object_delta_perception_block(trans: list[Transition]) -> str:
    """REQ-ARC-WMTE-6213: append transition object deltas only for the explicit arm."""

    try:
        from carnot.agentic import arc_object_delta_perception
    except Exception:
        return ""
    if not arc_object_delta_perception.object_delta_perception_on():
        return ""
    try:
        block = arc_object_delta_perception.object_delta_block(trans)
    except Exception:
        return ""
    return ("\n" + block) if block else ""


def _mechanic_class_router_block(trans: list[Transition]) -> str:
    """REQ-ARC-WMTE-6282: append game-blind mechanic class evidence when enabled."""

    if not mechanic_class_router_enabled():
        return ""
    try:
        from carnot.agentic import arc_mechanic_class_detector

        block = arc_mechanic_class_detector.prompt_block(trans)
    except Exception:
        return ""
    return ("\n" + block) if block else ""


# REQ-ARC-WMTE-5717: DEV-ONLY playbook methodology exemplars for the STALL re-induction
# path. Default OFF (env CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED unset -> byte-identical
# prompt, exactly like the CARNOT_ARC_CODEONLY_INDUCE / _REFACTOR_STRUCTURE_REMINDER
# gates above). A SMALL, game-AGNOSTIC few-shot of the recurring "orient, hypothesize,
# test, revise" exploration method distilled from the solve corpus
# (docs/research-notes/arc-exploration-playbook-20260717.md) -- PATTERN statements only,
# never a per-game fact (color/coordinate/mechanic), so they transfer to a HIDDEN game the
# agent has never seen. Deliberately terse: a sibling experiment (exp5714) found that
# long-reasoning induction overruns the token budget and emits zero code, so this biases
# the model's PRIORS without asking it to reason at length.
_PLAYBOOK_EXEMPLAR_BLOCK = """GENERAL EXPLORATION PRINCIPLES (observed across many ARC-AGI-3 games -- apply as PRIORS
when inducing the rules below; do NOT copy any specific game's colors/coordinates):
- Prefer SIMPLE, GENERAL rules over per-cell/hardcoded-coordinate special cases; a rule
  that memorizes exact coordinates rarely generalizes to the next state.
- Action effects can differ from level to level -- induce them from THESE transitions, do
  not assume a mapping carried over from a prior level.
- An object that recolors on contact or selection is the SAME object, not a new one.
- A level-complete state is often the frame AFTER the winning action; ground
  is_level_complete on the STRUCTURAL win condition, not one memorized exact grid.
- Some actions are inert (no change) or reset the level -- model those honestly.
- A fixed goal DISPLAY/legend is not the interactive target; the target is a piece that
  actually moves or changes.

"""


def induce_prompt(
    game: str,
    trans: list[Transition],
    cell: int,
    *,
    previous_level_complete_grid: Optional[np.ndarray] = None,
    # The level-up transition, SUPPLIED. On the live path it is absent from `trans` by
    # construction, so the WIN TRANSITION block could never fire; see `_transitions_block`.
    win_transition: Optional[Transition] = None,
    # `None` -> `_induce_transitions_k()`, which is "all" as of 2026-08-01. This was a literal
    # `8` that never consulted that resolver at all, so REQ-ARC-FCP-5699-23's knob moved the
    # prompt for diagnostics and did nothing for the live agent. An explicit int still caps.
    k: Optional[int] = None,
    include_playbook_exemplars: bool | str = False,
    hud_mask: Optional[np.ndarray] = None,
    hud_mask_enabled: Optional[bool] = None,
) -> str:
    # REQ-ARC-WMTE-6010 PROMPT COHERENCE: `hud_mask` reaches `_transitions_block` so the
    # transitions the LLM is SHOWN are classified changing-vs-inert by the same rule the
    # verifier uses to GRADE the resulting engine. See `_transitions_block`'s docstring for
    # the measured incoherence this closes (ft09: prompt asserted 0 no-ops, grader saw 32).
    # REQ-ARC-FCP-5699-23: k defaults to _transitions_block's own default (8, unchanged
    # production behavior). REQ-ARC-FCP-5699-22 found the default shows the LLM only ~6
    # grid-changing transitions of the 25 collected -- roughly one per action type, a
    # data-starvation signature matching observed hardcoded-literal-coordinate memorization
    # (g50t's engine). Callers may raise k (DEV-ONLY, via CARNOT_ARC_INDUCE_TRANSITIONS_K) to
    # test whether more per-action examples let the LLM infer general rules instead.
    h, w = trans[0].grid.shape
    colors = sorted(set(int(v) for t in trans for v in t.grid.flatten().tolist()))
    # REQ-ARC-WMTE-5717/5718: DEV-ONLY exemplar prefix. The inject/don't-inject DECISION is made
    # by the caller (the agent's stall-only gate). Three modes, all default to byte-identical:
    #   False / ""     -> no injection (the exact pre-existing prompt).
    #   True           -> the STATIC generic exemplar block (REQ-5717).
    #   <non-empty str> -> that exact RETRIEVED block (REQ-5718 RAG: top-K patterns for THIS
    #                      stuck situation), already formatted by arc_playbook_retrieval.
    if isinstance(include_playbook_exemplars, str):
        block = include_playbook_exemplars.strip()
        exemplars = (block + "\n\n") if block else ""
    else:
        exemplars = _PLAYBOOK_EXEMPLAR_BLOCK if include_playbook_exemplars else ""
    return f"""{exemplars}You are inducing an EXECUTABLE WORLD MODEL for the ARC-AGI-3 game '{game}'.

The game state is a {h}x{w} integer grid (logical resolution; colors {colors}). You are
given REAL observed transitions COMPACTLY: one full INITIAL grid (the layout), then per
transition the action and its DELTA = the FULL set of changed cells as run-length runs of the
form r<row>c<col0>:<v0>x<n0>,<v1>x<n1>,... — each run is a horizontal span of changed cells
starting at (row, col0); within that span, the NEW values are themselves given as
<value>x<count> pairs left-to-right (so a span of 6 changed cells that are all now color 5
appears as "5x6", not six separate "5"s). To apply a transition's delta to the prior grid, for
each run walk its <value>x<count> pairs in order, setting <count> consecutive cells starting at
the next unfilled column (starting at col0) to <value>; all other cells are unchanged. The delta
is COMPLETE (not truncated). Full grids (the INITIAL
grid and, if shown, the WIN STATE grid) use a DIFFERENT run-length form to stay compact on large
boards: one line per row, "r<row>:<v0>x<n0>,<v1>x<n1>,...". Each row's runs are listed
left-to-right and cover EVERY column with no gaps and no overlap, so the starting column of each
run is IMPLICIT: it equals the sum of the counts of all runs before it in that row (the first run
in a row starts at column 0). To reconstruct a full grid, for each row walk its runs in order,
placing <n> consecutive cells of value <v> starting at the next unfilled column.
Actions are integers 1-7; ACTION6 is a click
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
{_transitions_block(trans, k, previous_level_complete_grid=previous_level_complete_grid, hud_mask=hud_mask, hud_mask_enabled=hud_mask_enabled, win_transition=win_transition)}
{(_action_semantics_and_counts_block(trans, k, hud_mask=hud_mask, hud_mask_enabled=hud_mask_enabled) + chr(10)) if induce_prompt_enrichment_enabled() else ""}{("OBJECT STRUCTURE (same frames, connected-component view -- use object shape ids to track objects across the deltas above):" + chr(10) + objects_block(trans, previous_level_complete_grid=previous_level_complete_grid)) if _object_perception_on() else ""}{(chr(10) + _object_identity_crossref_note(trans, previous_level_complete_grid=previous_level_complete_grid)) if (induce_prompt_enrichment_enabled() and _object_perception_on()) else ""}{_object_delta_perception_block(trans)}{_mechanic_class_router_block(trans)}"""


_REFACTOR_PROMPT_MAX_CELLS_PER_MISMATCH = 8


def _bounded_mismatches(mismatches: list, *, limit: int = 5) -> list:
    """REQ-ARC-FCP-5699-26: cap each mismatch's cell-diff lists BEFORE JSON-encoding, instead
    of encoding everything and then hard-slicing the resulting string by raw character count.
    The raw-slice approach (the pre-existing `json.dumps(...)[:4000]`) produced genuinely
    INVALID JSON: verified on a real g50t counterexample, 5 real mismatches serialize to
    12,212 characters, and `[:4000]` cuts the string mid-field-name (`"true_chang` with the
    closing `e"` sliced off) -- the model was being shown malformed, truncated JSON and asked
    to debug from it, not a genuine capacity/reasoning limitation. Bounding each mismatch's
    `true_change`/`your_prediction_was_wrong_at` lists to a fixed cell count keeps every
    mismatch entry structurally complete and the overall JSON valid regardless of how large the
    underlying diffs are, while still showing a representative SAMPLE of cells per mismatch (an
    `_omitted_count` companion field records how many were cut, so the count is honest, not
    silently dropped)."""
    bounded = []
    for m in mismatches[:limit]:
        m = dict(m)
        for key in ("true_change", "your_prediction_was_wrong_at"):
            cells = m.get(key)
            if isinstance(cells, list) and len(cells) > _REFACTOR_PROMPT_MAX_CELLS_PER_MISMATCH:
                m[f"{key}_omitted_count"] = len(cells) - _REFACTOR_PROMPT_MAX_CELLS_PER_MISMATCH
                m[key] = cells[:_REFACTOR_PROMPT_MAX_CELLS_PER_MISMATCH]
        bounded.append(m)
    return bounded


# REQ-ARC-FCP-5699-31: DEV-ONLY structural reminder (unset -> exact pre-existing prompt,
# byte-identical), directly targeting the pathology REQ-ARC-FCP-5699-30 found by reading a real
# raw completion: given the SAME real g50t counterexample data, the model abandoned the required
# interface entirely -- it wrote a self-contained `class WorldModel` with `self`-bound methods
# and a fabricated grid representation instead of patching the required TOP-LEVEL engine()/
# is_level_complete() functions, never emitted is_level_complete at all, and stopped mid-
# statement. This is a content/structure reminder, NOT the codeonly "skip all reasoning"
# directive REQ-ARC-FCP-5699-26 confirmed is deliberately excluded from refactor() (refactor
# stays a genuine reasoning task) -- it reminds the model WHAT shape its answer must take without
# telling it not to think.
_REFACTOR_STRUCTURE_REMINDER_BEFORE = """
REQUIRED OUTPUT STRUCTURE -- do not deviate from this: your fixed code must define EXACTLY two
TOP-LEVEL functions, in the SAME format as the file you are correcting:

    def engine(grid, action, data):
        ...
    def is_level_complete(grid):
        ...

Do NOT wrap them in a class. Do NOT use `self`. Do NOT invent a new internal grid
representation or shape -- `grid` is the SAME real grid shape/format already used by the
code above; reuse it exactly, do not redesign it.
"""
_REFACTOR_STRUCTURE_REMINDER_AFTER = """
Reminder: return ONLY the corrected `engine(grid, action, data)` and `is_level_complete(grid)`
top-level functions -- no classes, no `self`, no invented state, no renamed parameters.
"""


# REQ-ARC-WMTE-6091: DEFAULT OFF. Unset/"0" -> the rendered prompt is BYTE-IDENTICAL to the
# shipped one, including the defect below.
_REFACTOR_SHOW_ENGINE_DEFAULT = "0"
# Bound on the engine source spliced into the prompt. The 13 roster engines run 413..15,396 bytes
# (measured, not assumed), so this clips nothing today; it exists so a pathological engine cannot
# silently evict the mismatch block from the context pool. Truncation is ANNOUNCED in the prompt
# and counted in the return of `_current_engine_source`, never silent.
_REFACTOR_ENGINE_SOURCE_MAX_CHARS = 24000


def refactor_show_engine_enabled() -> bool:
    """Is the engine being refactored actually SHOWN to the model? (REQ-ARC-WMTE-6091)

    THE DEFECT, MEASURED (results/outer_loop_arc_refine_instrument_repro_20260803.json).
    `refactor_prompt` takes a game id and a `VerifyResult`. Neither carries the engine. So the
    rendered prompt tells the model to "REFACTOR toward simpler, more general rules (replace
    special cases with shared rules) while keeping the cases it already gets right" -- about
    code it cannot see, and about passing cases it is never shown. Across the 13 roster games,
    ZERO of the engines' substantive source lines reach the prompt string; the only lines that
    match are this module's own REQUIRED OUTPUT STRUCTURE boilerplate (`import numpy as np`,
    `def engine(grid, action, data):`), which is the prompt quoting its own template, not the
    engine.

    WHAT THAT MAKES THE SHIPPED "REFINEMENT" ROUND. Not refinement. A blind RE-INDUCTION from a
    handful of failing deltas, with LESS evidence than round 0 had -- round 0 at least sees the
    induce prompt's transition block, while a refactor round sees at most five mismatches
    (`_bounded_mismatches`). That is a coherent explanation for the shipped CEGIS result
    (exp5766: 0 of 39 cells with delta_heldout > 0, and prefix_accuracy COLLAPSING from a 1.0
    ceiling at induce to a 0.125 ceiling at refactor -- REQ-ARC-WMTE-6042's write-collapse
    instrumentation): a round that is told to preserve what it cannot see does not preserve it.

    SO THE PRIOR CEGIS NULL DOES NOT FALSIFY REFINEMENT. It falsifies the instrument. Turning
    this ON is what makes the refinement hypothesis testable for the first time.

    DEFAULT OFF, because every existing measurement -- exp5760, exp5766, and exp5766's
    `retire_if_same_verdict` -- was taken against the blind prompt, and an interpretable A/B
    needs the old arm reproducible byte for byte. `CARNOT_ARC_REFACTOR_SHOW_ENGINE=1` enables it.
    """

    return _flag_env("CARNOT_ARC_REFACTOR_SHOW_ENGINE", _REFACTOR_SHOW_ENGINE_DEFAULT == "1")


def _current_engine_source(game: str, *, max_chars: int = _REFACTOR_ENGINE_SOURCE_MAX_CHARS):
    """The engine `refactor` is being asked to fix, read off disk. Returns (text, truncated_chars).

    READ-ONLY, and deliberately failure-tolerant: an unreadable/absent engine yields "" so the
    ON arm degrades to exactly the OFF prompt rather than raising inside a live refinement round.
    The caller records which happened; it is never inferred from the prompt's length.
    """

    try:
        src = (E3_DIR / game / "world_model.py").read_text()
    except OSError:
        return "", 0
    if len(src) <= int(max_chars):
        return src, 0
    return src[: int(max_chars)], len(src) - int(max_chars)


_REFACTOR_ENGINE_BLOCK_HEADER = """
THE CURRENT ENGINE YOU ARE FIXING (results/arc_e3/{game}/world_model.py) -- this is the code
that produced the predictions below. Keep every rule of it that is already correct; change only
what the mismatches show to be wrong:

```python
{source}
```{truncation_note}
"""


def refactor_prompt(game: str, vr: VerifyResult, *, engine_source: Optional[str] = None) -> str:
    import os

    mism = json.dumps(_bounded_mismatches(vr.mismatches), indent=1)
    engine_block = ""
    if refactor_show_engine_enabled():
        # An explicit `engine_source` (a caller that already holds the text) wins over the disk
        # read, so a caller driving a redirected store does not depend on E3_DIR resolution.
        if engine_source is None:
            src, dropped = _current_engine_source(game)
        else:
            src, dropped = str(engine_source), 0
        if src.strip():
            engine_block = _REFACTOR_ENGINE_BLOCK_HEADER.format(
                game=game,
                source=src.rstrip(),
                truncation_note=(
                    f"\n(NOTE: {dropped} further characters of this file were omitted for length.)"
                    if dropped
                    else ""
                ),
            )
    before = ""
    after = ""
    # Graduated to default-on (REQ-ARC-FCP-5699-35): REQ-31/32 validated this reminder fixes a
    # real class-wrapping/missing-function pathology with zero observed regressions (6/6 success
    # across a full-budget live run). CARNOT_ARC_REFACTOR_STRUCTURE_REMINDER=0 remains an
    # explicit opt-out escape hatch, matching the CARNOT_ARC_MTP=0 pattern elsewhere in this file.
    if os.environ.get("CARNOT_ARC_REFACTOR_STRUCTURE_REMINDER", "1") != "0":
        before = _REFACTOR_STRUCTURE_REMINDER_BEFORE
        after = _REFACTOR_STRUCTURE_REMINDER_AFTER
    return f"""The executable world model at results/arc_e3/{game}/world_model.py reproduces
only {vr.n_correct}/{vr.n} ({vr.accuracy:.0%}) of the observed transitions. Below are
failing cases (BEFORE / your PREDICTED / the true OBSERVED next grid). Fix engine() so it
reproduces these too, and REFACTOR toward simpler, more general rules (replace special
cases with shared rules) while keeping the cases it already gets right. Edit only that
file.
{engine_block}{before}
MISMATCHES:
{mism}
{after}"""


@dataclass
class CodexProposer:
    """DEV-ONLY proposer (codex/gpt-5.5). Requires INTERNET, so it is NOT legal in the
    OFFLINE competition eval — use it only to validate the E3 loop during development.
    For the competition, use LocalGGUFProposer (open-weight, offline)."""

    timeout: int = 420
    offline_legal: bool = False
    # REQ-ARC-WMTE-5717: DEV-ONLY (see LocalGGUFProposer's field); default False -> byte-identical.
    include_playbook_exemplars: bool | str = False

    def induce(
        self,
        game: str,
        trans: list[Transition],
        cell: int,
        *,
        previous_level_complete_grid: Optional[np.ndarray] = None,
        win_transition: Optional[Transition] = None,
    ) -> tuple[bool, str]:
        _guard_engine_write(E3_DIR / game)
        (E3_DIR / game).mkdir(parents=True, exist_ok=True)
        ok, tail = _codex(
            induce_prompt(
                game,
                trans,
                cell,
                previous_level_complete_grid=previous_level_complete_grid,
                win_transition=win_transition,
                k=_induce_transitions_k(),
                include_playbook_exemplars=self.include_playbook_exemplars,
            ),
            self.timeout,
        )
        if ok:
            # REQ-ARC-WMTE-6690: codex wrote the file out-of-band; archive it post-hoc.
            self.last_attempt_archive = _archive_codex_engine(game)
        return ok, tail

    def refactor(self, game: str, vr: VerifyResult) -> tuple[bool, str]:
        ok, tail = _codex(refactor_prompt(game, vr), self.timeout)
        if ok:
            # REQ-ARC-WMTE-6690: same post-hoc archive as induce.
            self.last_attempt_archive = _archive_codex_engine(game)
        return ok, tail


# ==============================================================================================
# THE LIVE ARC GENERATOR (operator directive 2026-07-28). ONE definition, read by every live
# construction site, so `_proposer()` and `_load_sge_candidate_router()` cannot drift apart and
# quietly load two different models onto two ports.
#
# WHAT CHANGED AND WHY. The generator was Qwen3.5-9B-MTP, pinned there for exactly one reason: an
# assumed 16 GB Kaggle VRAM ceiling that a 5.9 GB Q4 model fits and an 18.3 GB one does not. The
# operator has declared that ceiling void ("the Kaggle hardware is 96G since May"), which removes
# the only constraint that was holding the pin, and directed the switch to gemma-4-31B-it.
#
# THE MEASUREMENT BEHIND IT (2026-07-28, 13 games x 3 replicates, Q4_K_M both sides, n_ctx 32768,
# one model per card, per-arm engine store, no wedge):
#
#                  induce_ok    fail-as-zero    survivor-mean    nonzero cells
#   gemma-4-31B      38/39          0.3843          0.3944            23
#   Qwen3.6-27B      21/39          0.0627          0.1164             4
#
#   matched per-game tally 11-0-2, two-sided sign test p = 0.00098 -- which is EXACTLY the
#   smallest p reachable at 11 discordant pairs, i.e. the result is as strong as this design can
#   express. The 31B independently reproduces exp5764 (0.3843 here vs 0.3785 there).
#
# READ THE DOMINANT DRIVER HONESTLY: it is LOADABILITY, not subtle induction quality. The 27B
# failed to emit an importable world_model.py on 18 of 39 attempts. Its survivor mean is therefore
# survivorship-biased and fail-as-zero is the honest column.
#
# NOT AN MTP MODEL. Qwen3.5-9B-MTP carried native multi-token-prediction heads and the live sites
# defaulted `CARNOT_ARC_MTP` to "1". gemma-4-31B-it declares no `nextn_predict_layers` in its GGUF
# header at all (checked directly). Leaving the old default would have made `_ensure_server()`
# emit `--spec-type draft-mtp --model-draft <the same 18.3 GB file>` -- loading the weights TWICE,
# ~36.6 GB, a guaranteed cudaMalloc failure on a 24 GB card and a 180 s burn ending in a SILENT
# LLM-off run that still reports itself as the LLM-on scored path. Hence the default flips to "0".
# The env var still works in both directions for anyone pointing CARNOT_ARC_GGUF_PATH at a real
# MTP model.
#
# ^^^ CORRECTION, 2026-07-28 (SAME DAY, MEASURED -- the paragraph above is WRONG in its premise and
# is preserved per never-prune because its CONCLUSION about `--model-draft <the main gguf>` is
# still exactly right and is now enforced mechanically). gemma-4-31B-it DOES have real MTP. The
# reason the check above found no `nextn_predict_layers` is that for this model family the head is
# NOT embedded in the main GGUF at all -- it ships as a SEPARATE 491 MiB file
# (`unsloth/gemma-4-31B-it-GGUF` -> `MTP/mtp-gemma-4-31B-it-Q8_0.gguf`) whose own header declares
# `general.architecture = gemma4-assistant` and the key `gemma4-assistant.nextn_predict_layers`.
# Read directly out of the file's GGUF header, not inferred.
#
# So the correct statement is: the main GGUF is not SELF-drafting. `--model-draft` must point at
# the HEAD, never at the main weights. Passing the main file is what produces the double-load the
# paragraph above describes, so that warning stands -- it is now enforced by `_resolve_mtp_head()`
# plus the head-absent guard in `_ensure_server()` rather than by leaving MTP permanently off.
#
# MEASURED, this session, matched prompt / n_ctx 32768 / q8_0 KV / one model alone on an RTX 3090,
# using the BINARY THE SUBMISSION ACTUALLY BUNDLES (`iancblenke/carnot-llamacpp-mtp-binary`, which
# was selected for the retired Qwen MTP path months before gemma-4 MTP existed and therefore could
# not be assumed to support it):
#
#     MTP OFF  35.88 tok/s        MTP ON  50.16 tok/s     -> 1.398x decode, +862 MiB
#     draft_n_accepted / draft_n = 319/576 (55.4%)        -> speculation is provably DOING work,
#                                                            not merely logged as initialised
#
# THE SILENT-DEGRADATION TRAP THIS ARMS AGAINST. When `--spec-type draft-mtp` is handed a draft the
# runtime cannot use, llama.cpp does NOT fail. It warns and serves normally with speculation
# silently DISABLED:
#     W llama_init_from_model: context type MTP requested but model doesn't contain MTP layers
#     W common_speculative_init: no implementations specified for speculative decoding
# -> server UP, /health 200, generation normal, ZERO speedup. The SUCCESS signature is instead:
#     I common_speculative_impl_draft_mtp: adding speculative implementation 'draft-mtp'
#     I srv    load_model: speculative decoding context initialized
# A misconfigured MTP is indistinguishable from a working one EXCEPT by tok/s. Never conclude "MTP
# is enabled" from a healthy server or an absent error.
#
# WHY THE LOCAL DEFAULT STAYS "0" ANYWAY (this is an arithmetic result, not caution). The head is
# not free and its cost SCALES WITH THE CONTEXT POOL -- measured 862 MiB at n_ctx 32768 and
# 1290 MiB at n_ctx 81920 (see `_VRAM_MTP_HEAD_*` below). At the shipped n_ctx 81920 a 24 GB 3090
# must offload ~14 FFN blocks to system RAM to host the MTP-on server versus ~7 for MTP-off, and
# the measured offload curve costs more decode throughput than MTP buys back:
#     MTP-off @ 7 offloaded layers  ~= 23.8 tok/s
#     MTP-on  @ 14 offloaded layers ~= 13.9 tok/s x 1.398 = ~19.4 tok/s   <- NET LOSS locally
# On the 96 GB Kaggle card no offload is needed at all, so there MTP is a pure ~1.4x win. Hence the
# split below: "0" is the correct LOCAL default and "1" is the correct SCORED default. They are two
# different hardware answers to the same question, so they are two named constants rather than one
# constant somebody has to remember to flip.
#
# DO NOT TREAT 1.4x AS A PLANNING CONSTANT. Speculative speedup scales with how predictable the
# continuation is; the matched prompt above accepted 55.4% of drafted tokens. An earlier
# measurement on a different prompt reached 1.76x. The defensible range on real induction prompts
# is ~1.4-1.8x, and 1.4x is the conservative floor an actual measurement supports.
#
# MTP IS **NOT** OUTPUT-NEUTRAL HERE, AND THAT IS A MEASURED FACT, NOT A CAVEAT. Speculative
# decoding is textbook-exact -- the target model verifies every drafted token, so the accepted
# sequence is supposed to be the one the target would have produced alone. This implementation on
# this model does not deliver that. From the SAME matched run recorded in
# `results/arc_gemma31b_migration_evidence_20260728/mtp_shipped_binary_t1/`, at temperature 0,
# top_k 1, seed 1234, byte-identical prompt, 3 iterations per arm:
#
#     MTP ON  -> content_len 1890, identical across all 3 iterations
#     MTP OFF -> content_len 1917, identical across all 3 iterations
#
# So each arm is internally deterministic and the two arms DISAGREE with each other. Both
# completions begin with the same 60 characters, so this is a divergence part-way through a
# generation, not two unrelated answers. The likely mechanism is that batched draft verification
# takes a different kernel/reduction path than single-token decode and the resulting last-bit
# differences change an argmax at some token; we have not isolated it, and this comment does not
# claim to have.
#
# WHY THIS MATTERS FOR THE SUBMISSION, STATED HONESTLY. Every number backing the gemma-over-Qwen
# migration -- the 11-0-2 tally, the 0.3843 fail-as-zero score, every induction timing -- was taken
# MTP-OFF. The scored path launches MTP-ON. No induction-QUALITY A/B has been run in the MTP-on
# configuration, so the 1.398x is a measured SPEED claim and the quality transfer is an ASSUMPTION.
# Recorded here as an accepted risk rather than papered over: the operator authorised MTP for the
# scored run explicitly, the divergence is small and mid-generation rather than structural, and an
# induction-quality A/B is the correct way to retire the assumption if it ever becomes load-bearing
# for a headline claim. Do NOT describe MTP as output-neutral anywhere in this codebase; the
# artifacts say otherwise.
#
# NO `/no_think`. That prefix is a Qwen3 hybrid-thinking control token. Gemma-4 has no such token
# and would consume it as literal prompt text -- the silently-dead-channel defect class this
# project keeps finding. (The 31B scored 0.3843 in the head-to-head WITH the prefix still present,
# so it was not load-bearing; that is a reason to remove it cleanly, not a reason to keep it.)
# QAT (2026-07-31, operator directive after the head-to-head below). The repo substring is
# the QAT one specifically: "gemma-4-31B-it" alone matches BOTH cache dirs
# (`...-it-GGUF` and `...-it-qat-GGUF`) and would resolve ambiguously.
# MOVED 2026-08-16 by operator directive, gemma-4-31B-it-qat -> Qwen3.8-27B. What the swap costs,
# measured, so it is not rediscovered later: Qwen3.8 writes far longer completions (49,244-83,544
# tokens against gemma's ~6,584) and no MTP draft head ships for it, so speculative decoding turns
# off. Together that is roughly 17x the wall clock per induction on the scored card -- about 49% of
# the 12h eval budget against gemma's 3%. It is bought with better induction quality on the three
# games measured head to head so far, which is thin evidence; see the h2h evidence directory.
ARC_LIVE_GENERATOR_REPO_SUBSTR = "Qwen3.8-27B"
ARC_LIVE_GENERATOR_MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
ARC_LIVE_GENERATOR_MODEL_FILENAME = "Qwen3.8-27B-Q4_K_M.gguf"
# WHY QAT, AND WHY NOT FOR QUALITY. A 20-game x 3-trial head-to-head against the shipped
# Q4_K_M (results/outer_loop_arc_qat_vs_q4km_h2h_20260731.json) came back INDISTINGUISHABLE:
# mean-B 6-6 with 8 ties, exact two-sided sign test p = 1.0 over 12 discordant pairs (min
# reachable p 0.000488, so the null had real power). Pooled mean-B 0.1937 QAT vs 0.1999
# Q4_K_M -- if anything marginally worse, well inside noise.
#
# An earlier 13-game read had QAT ahead 5-2 (p = 0.453) and pooled +0.028. That lead did not
# survive the pre-registered extension to 20 games; it inverted. Recorded because it is the
# reason the switch is NOT justified on quality and must never be described as if it were.
#
# The switch is on the two axes QAT actually wins, both measured:
#   VRAM  20430 MiB resident vs 21418 MiB  (~1 GB, observed across 60 cells per arm)
#   disk  17.3 GB vs 18.3 GB
# plus it ships matching QAT MTP drafters, which Google's model card states are REQUIRED when
# speculating against a QAT target: "the assistant model must also be a QAT checkpoint with
# the same precision".
# LOCAL default. "0" because at n_ctx 81920 a 24 GB card must offload ~14 FFN blocks to host the
# MTP-on server and the offload costs more throughput than MTP returns -- see the arithmetic above.
ARC_LIVE_GENERATOR_MTP_DEFAULT = "0"
# SCORED (Kaggle 96 GB) default. "1" because no offload is needed there, so MTP is a pure ~1.4x
# decode win, and the bundled `iancblenke/carnot-llamacpp-mtp-binary` was VERIFIED this session to
# engage `draft-mtp` on the `gemma4-assistant` head (positive marker + 319/576 accepted draft
# tokens + a matched throughput delta). Operator-authorised 2026-07-28: "when we submit we will
# want MTP enabled for speed when running on the Kaggle 96G GPU hardware."
#
# THIS IS A SEPARATE CONSTANT ON PURPOSE. The scored path and the dev box have DIFFERENT correct
# answers, and the failure mode of collapsing them into one is asymmetric: an operator who forgets
# to flip a single shared constant before submitting silently ships the slower configuration, and
# nothing anywhere reports it -- the run just takes ~1.4x longer per induction. Naming both makes
# the divergence a fact the tests can pin instead of a step in a runbook.
# FLIPPED TO "0" 2026-08-17 with the move to Qwen3.8-27B, which has no MTP draft head published.
# Kernel v19 caught the disagreement at runtime and named it exactly: "SUBMITTED_AGENT_CONFIG
# declares mtp=True but this kernel resolved mtp=False ... the readiness gates describe a
# configuration this run is NOT using". Leaving it at "1" costs nothing at run time (the kernel
# resolves head_present=False and proceeds without speculative decoding) but it makes every
# readiness gate report a config that is not the one running, which is the precise failure this
# constant's own comment above warns about, pointed the other way.
# FLIPPED BACK TO "1" 2026-08-17 (later the same day, operator-directed) with the move to the
# NVFP4 Qwen3.8-27B conversion. The "0" above was correct for exactly one reason -- that model
# had no published draft head -- and that reason is now gone: the NVFP4 build carries its own MTP
# layers and declares `nextn_predict_layers` in its GGUF, so `_gguf_declares_baked_mtp` resolves
# it and `_ensure_server` emits `--spec-type draft-mtp` with no separate head. Head presence is
# therefore no longer the right question to ask about this model; self-drafting is.
# The v19 disagreement this constant's comment describes does NOT return: that fired because the
# config CLAIMED mtp=True while the kernel resolved False. Here both read True, and the run is
# verifiable rather than assumed -- llama.cpp prints `adding speculative implementation
# 'draft-mtp'` when speculation is genuinely wired, and this file already greps for that marker.
# Confirm it in the save-run log before trusting the speedup; tok/s is otherwise the only tell.
# FLIPPED TO "0" AGAIN 2026-08-18, and this time the save-run answered the question the paragraph
# above says to ask. Kernel v27 reported `mtp_requested=False mtp_engaged=False` and printed the
# config/run disagreement banner verbatim -- the same v19 failure the comment above predicted would
# NOT return. Two things changed under it:
#   * The scored path is vLLM now, not llama.cpp. `_ensure_vllm_server` deliberately does not
#     configure speculation, so on the path that actually scores, mtp is False as a matter of fact,
#     whatever this constant says.
#   * MTP is the WRONG config for the fallback path too, on measurement rather than taste. The eval
#     runs one thread per game, and at k=16 llama.cpp served 228.3 tok/s with MTP off against 108.8
#     with it on. Speculation and batching compete for the same compute; batching wins at the
#     concurrency this eval produces.
# So "0" is now correct on both paths at once -- honest on the vLLM path, and faster on the
# llama.cpp fallback. `tests/python/test_arc_submitted_agent_parity.py` pins the literal False and
# went red on the "1" above, which is how this regression surfaced.
ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT = "0"
# The MTP head is a SEPARATE FILE, not a section of the main GGUF. Both the filename and the
# substring are named here because the Kaggle kernel matches by name against an order-undefined
# `rglob`, and the head and the main model are both `*.gguf` under the same mount root.
ARC_LIVE_GENERATOR_MTP_HEAD_FILENAME = "mtp-gemma-4-31B-it-Q8_0.gguf"
ARC_LIVE_GENERATOR_MTP_HEAD_SUBSTR = "mtp-gemma-4-31B-it"
# The drafter must come from the SAME repo as the target. Both the QAT and non-QAT repos ship
# a file called `mtp-gemma-4-31B-it-Q8_0.gguf`, so the filename substring above CANNOT tell
# them apart -- and `_resolve_mtp_head` globs every `models--*GGUF` root and takes sorted()[0].
# Before this constant existed, switching the target to QAT while leaving a non-QAT head on
# disk would have silently paired a non-QAT drafter with a QAT target: accepted by llama.cpp,
# forbidden by Google's card, and invisible except as degraded acceptance.
ARC_LIVE_GENERATOR_MTP_HEAD_REPO_SUBSTR = "gemma-4-31B-it-qat"
ARC_LIVE_GENERATOR_MTP_HEAD_ARCH = "gemma4-assistant"  # read from the head GGUF's own header
ARC_LIVE_GENERATOR_NO_THINK_PREFIX = ""  # /no_think is a Qwen3 token; inert on gemma-4

# THINK MODE (2026-08-07, operator directive: "we want /think with the 31B model" -- lever #6,
# ops/known-issues.md, amending the 2026-08-03 one-slot ruling; the June /no_think freeze is
# LIFTED). Gemma-4 has no Qwen-style /think-/no_think soft-switch token -- exp5764's own note
# discloses this. Its mechanism is instead a NATIVE thought channel that the bundled llama-server
# splits into `reasoning_content` on the /v1/chat/completions endpoint (proof:
# experiment_5764_gemma31b_singleshot_induction_ab.json, n_reason_engaged=39/39 on
# use_chat_template=True). The best gemma induction numbers on record (exp5764 pooled heldout
# 0.3785) were ALREADY taken in that reasoning-engaged chat configuration; the LIVE path today
# ships the reasoning-SUPPRESSED raw+codeonly configuration instead (the codeonly directive +
# pre-opened fence exist specifically to suppress reasoning for a different failure mode -- see
# _L2_CODEONLY_DIRECTIVE below). Both configurations are real and neither is a straw man; which
# one wins on gemma is an open, UN-A/B'd question this flag exists to answer.
#
# WAS "0" ON BOTH AXES pending the matched-budget offline A/B this project's standing
# convention required before a live-path default flip. That A/B landed 2026-08-08
# (experiment_6199_gemma_think_mode_ab.json, REQ-ARC-WMTE-6198): 10/10 games induced on both
# arms, think mode engaged real reasoning 10/10 vs 0/10 for no_think, and think mode had a
# consistent edge on induction quality (heldout_accuracy, goal_predicate_accuracy) -- never
# losing where the arms differed. The narrower levelup_positive_recall metric was inconclusive
# (7 of 10 games showed no signal either way). Operator directive 2026-08-08, on that evidence:
# flip think mode on going forward. `ARC_LIVE_GENERATOR_THINK_DEFAULT` is flipped alongside the
# SCORED constant for consistency even though `induce_think_on()` only ever reads the SCORED one
# (see its own docstring) -- nothing in the tree consults the non-scored constant at runtime.
ARC_LIVE_GENERATOR_THINK_DEFAULT = "1"
ARC_LIVE_GENERATOR_THINK_SCORED_DEFAULT = "1"

# INDUCE BUDGET AND TIMEOUT MOVE WITH THE PIN (2026-08-21, REQ-ARC-WMTE-6620).
# Qwen3.8-27B thinking inductions measure 36,406-83,444 generated tokens (median 62,490;
# one censored draw at >=100,988). The old agent-side default of 4096 was validated for the
# retired 9B (REQ-ARC-FCP-5699-32) and never moved through two generator swaps, so every
# local run failed 100% of inductions while the scored kernel, which pins these same two
# values via env (scripts/kaggle/submission_kernel/main.py), kept working. These constants
# are the kernel's pins, named next to the generator pin so the next swap moves them too.
# `test_arc_induction_budget_defaults.py` asserts the kernel literals stay equal to these.
ARC_LIVE_GENERATOR_INDUCE_MAX_TOKENS_DEFAULT = 131072
ARC_LIVE_GENERATOR_INDUCE_TIMEOUT_FLOOR_S = 2400


def induce_think_on() -> bool:
    """REQ-ARC-WMTE-6198 (lever #6) arm selector. CARNOT_ARC_INDUCE_THINK=1/0 overrides;
    unset -> ARC_LIVE_GENERATOR_THINK_SCORED_DEFAULT. STALE UNTIL 2026-08-08: this docstring
    said both constants were "0" (default OFF); the operator flipped both to "1" that day on
    exp6199's induction-quality evidence (see the comment above ARC_LIVE_GENERATOR_THINK_
    DEFAULT), so this resolver is now live by default, not inert. Mirrors `_flag_env`'s
    override-then-shipped-default shape but keys off the SCORED constant, matching the MTP
    resolver pattern above (`ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT`) -- there is no separate
    local-default consulted at runtime here because, unlike MTP, think mode has no local-hardware
    reason to differ from the scored intent; the local/scored split exists so the two answers CAN
    diverge later without collapsing back into one shared constant, not because they must today.
    """

    import os

    raw = os.environ.get("CARNOT_ARC_INDUCE_THINK")
    if raw is not None and raw != "":
        return raw.strip() == "1"
    return ARC_LIVE_GENERATOR_THINK_SCORED_DEFAULT != "0"


def _is_mtp_head_file(name: str) -> bool:
    """True if `name` looks like an MTP DRAFT HEAD rather than a main weights file.

    Exists because the head and the main model are both `*.gguf`, both live under the same
    HuggingFace repo, and are told apart by NOTHING except their names. Every place that picks "the
    model" out of a directory listing has to exclude the head, or it can bind 491 MiB of draft
    head as if it were the 18.3 GB generator -- which loads, serves, and answers nonsense.

    MATCHES THE `mtp-` PREFIX, NOT THE SUBSTRING `"mtp-"` ANYWHERE (fixed 2026-07-28, third pass).
    The substring form was a false-positive generator on a naming convention this project actually
    uses: `Qwen3.5-9B-MTP-Q4_K_M.gguf` is a MAIN WEIGHTS file whose name contains "MTP-", and the
    substring test classified it as a draft head. Consequences, both silent:

      * `_resolve_gguf` filters `if not _is_mtp_head_file(...)`, so for a repo whose weights file
        is named `...-MTP-<quant>.gguf` it filtered out the only candidate and returned None --
        i.e. the documented "the retired 9B remains a legitimate CARNOT_ARC_GGUF_PATH override for
        a genuinely 16GB-class box" escape hatch could not resolve its own model from cache.
      * Any caller asking "is this a self-drafting MTP build?" got the answer "no, it's a head",
        which is the exact inversion of the truth for that model.

    Upstream's convention is a PREFIX (`unsloth/gemma-4-31B-it-GGUF` -> `MTP/mtp-gemma-4-31B-it-
    Q8_0.gguf`), and `ARC_LIVE_GENERATOR_MTP_HEAD_FILENAME` follows it, so the prefix is both the
    precise test and the documented one. A model whose name merely CONTAINS "-MTP-" is a
    self-drafting build, not a head.
    """
    return Path(name).name.lower().startswith("mtp-")


# The GGUF metadata key a SELF-DRAFTING model declares. Seen as `qwen35.nextn_predict_layers`;
# the prefix is the architecture string, so match the suffix and never the whole key.
_BAKED_MTP_METADATA_KEY = b"nextn_predict_layers"
# GGUF puts its metadata block at the head of the file, so a bounded read finds the key without
# loading 23 GB of weights. 8 MiB clears the metadata of every model this project ships.
_GGUF_METADATA_SCAN_BYTES = 8 * 1024 * 1024


def _gguf_declares_baked_mtp(path: str | Path) -> bool:
    """True if this GGUF carries its own MTP layers and needs no separate draft head.

    Two different things are both called "MTP". gemma-4-31B-it has NO MTP layers in its main
    weights and needs `--model-draft <a separate head file>`. A self-drafting build (for example
    the NVFP4 Qwen3.8-27B conversion) has the head baked in and declares `nextn_predict_layers`,
    so it takes `--spec-type draft-mtp` ALONE. Passing `--model-draft` for that second kind would
    reintroduce the exact silent failure documented in `_ensure_server`.

    Fails CLOSED: any read error returns False, which costs the speedup and never enables
    speculation on a model that cannot do it.
    """
    try:
        with open(path, "rb") as fh:
            return _BAKED_MTP_METADATA_KEY in fh.read(_GGUF_METADATA_SCAN_BYTES)
    except OSError:
        return False


# --- vLLM backend (REQ-ARC-WMTE-6510, operator-directed migration 2026-08-18) -----------------
#
# A SECOND backend, not a replacement. The dev 3090s are sm_86 and cannot execute NVFP4 at all,
# so llama.cpp stays the local/conductor default and every existing test keeps pinning it. vLLM
# is selected only where it was measured to win: the Kaggle Blackwell kernel, where native FP4 +
# fp8 KV + continuous batching measured 651.8 tok/s aggregate at k=32 against 228.3 for the best
# llama.cpp config and ~52 for what shipped before (see okf/claims/vllm-native-fp4-blackwell.md).
# Default UNSET keeps the llama.cpp path byte-identical.

_VLLM_BACKEND_ENV = "CARNOT_ARC_LLM_BACKEND"
_VLLM_MODEL_DIR_ENV = "CARNOT_ARC_VLLM_MODEL_DIR"
_VLLM_MAX_SEQS_ENV = "CARNOT_ARC_VLLM_MAX_SEQS"


def _vllm_backend_active() -> bool:
    """True only on the explicit opt-in. Anything else -- unset, empty, llamacpp, typo -- is the
    llama.cpp path, so a broken env var cannot silently migrate the generator."""
    return os.environ.get(_VLLM_BACKEND_ENV, "").strip().lower() == "vllm"


def _resolve_vllm_model_dir() -> Optional[str]:
    """The safetensors MODEL DIRECTORY for the vLLM backend (config.json + *.safetensors).

    Precedence mirrors `_resolve_gguf`: the explicit env pin wins (the Kaggle kernel sets it
    after resolving by rglob -- dataset mount depth is NOT stable across runs, so fixed-depth
    paths are forbidden); otherwise scan /kaggle/input the same way the kernel does. No
    HuggingFace-cache fallback on purpose: this backend only exists where the weights ship as an
    attached dataset, and a silent cache hit on a dev box would run vLLM somewhere it was never
    validated."""
    env = (os.environ.get(_VLLM_MODEL_DIR_ENV) or "").strip()
    if env and Path(env).is_dir() and (Path(env) / "config.json").exists():
        return env
    root = Path("/kaggle/input")
    if root.exists():
        for cfg in root.rglob("config.json"):
            if any(cfg.parent.glob("*.safetensors")):
                return str(cfg.parent)
    return None


def _vllm_max_seqs() -> int:
    """Concurrent-sequence cap for the vLLM server. Default 8, LOWERED FROM 24 on 2026-08-19.

    WHY IT WAS 24: fp8 KV fits ~22 concurrent sessions at a capped ~54k-token length in the
    ~61 GB KV budget, and the throughput curve is already bending by k=32. Both still true. The
    number was chosen against MEMORY and DIMINISHING RETURNS, and neither is what binds.

    WHAT BINDS IS THE PER-CALL TIMEOUT. The scored kernel sets CARNOT_ARC_INDUCE_TIMEOUT=2400,
    and aggregate throughput at concurrency is bought by LOWERING the per-stream rate -- so the
    more sessions run, the longer each induction takes, against a fixed per-call ceiling. A
    timeout that fires does not slow the run, it loses that induction: `generate()` fails and the
    agent proceeds with no world model for that call.

    THE LENGTHS ARE MEASURED, not assumed. Nine real Qwen3.8-27B inductions from the 2026-08-18
    goal-defect A/B, in generated tokens: 36406, 47817, 57492, 61951, 62490, 69095, 70944, 78851,
    83444 -- median 62,490 and max 83,444. Against per-stream rates measured on the scored
    Blackwell card:

        k    tok/s/stream   median      max        aggregate
        4        45.2       1383 s ok   1846 s ok  ~181 tok/s
        8        40.0       1562 s ok   2086 s ok  ~320 tok/s   <- the default
        16       30.2       2069 s ok   2763 s OVER ~483 tok/s
        24       23.6       2648 s OVER 3536 s OVER ~567 tok/s   <- the old default
        32       20.4       3063 s OVER 4090 s OVER ~652 tok/s

    At 24 the MEDIAN induction exceeds the timeout, not merely the tail. 8 still runs ~6.1x the
    throughput of the llama.cpp configuration this replaces.

    CORRECTION, hours later the same night. The line above used to claim 8 is "the highest
    concurrency at which even the longest observed induction fits". That was true of the longest
    COMPLETED induction (83,444 tokens, 2086 s at k=8) and false in general, because the length
    distribution is RIGHT-CENSORED BY THIS VERY TIMEOUT. Three draws in that corpus were cancelled
    at 3599 s and never produced an `eval time` line, so they are invisible to any distribution
    built from completed calls. The largest reached 100,988 generated tokens -- 98.6% of the
    102,400 cap -- and was still climbing when the client killed it. At k=8 that draw needs 2525 s
    against a 2400 s ceiling: OVER.

    So no concurrency guarantees the tail fits, and none can, because the mechanism that would
    record the tail is the mechanism that destroys it. 8 is a risk/throughput choice, not a
    guarantee. For reference, at the measured per-stream rates:

                    83,444 (max completed)   >=100,988 (censored)   131,072 (the cap)
        k=4              1846 s ok                 2234 s ok             2900 s OVER
        k=8              2086 s ok                 2525 s OVER           3277 s OVER
        k=16             2763 s OVER               3344 s OVER           4340 s OVER

    THE BETTER LEVER IS PROBABLY THE TIMEOUT, NOT THIS CONSTANT. `CARNOT_ARC_INDUCE_TIMEOUT` is
    2400 s and `CARNOT_ARC_INDUCE_MAX_TOKENS` is 131,072, and the two were set independently -- so
    the budget cannot be spent inside the window at ANY concurrency, including k=1. Making them
    consistent (about 3277 s at k=8) would cover the full cap without giving up throughput.
    Operator decision, not taken here.

    HONEST LIMITS. The draws were produced on a dev 3090 under the A/B's think envelope, not the
    scored 131072-token one, so the length distribution transfers only as an estimate; the k=8
    and k=24 per-stream rates are interpolated between measured k=4, k=16 and k=32; and the eval's
    real concurrency is the number of games in flight, which may never reach either value. Raising
    this above 8 without re-measuring induction length is the specific thing not to do.

    Floor at 1, malformed values fall back.
    """
    raw = os.environ.get(_VLLM_MAX_SEQS_ENV)
    try:
        return max(1, int(raw)) if raw else 8
    except (TypeError, ValueError):
        return 8


def _resolve_gguf(repo_substr: str) -> Optional[str]:
    """Find a cached GGUF weight file for an open-weight SOTA model (offline).

    EXCLUDES MTP DRAFT HEADS (2026-07-28). `unsloth/gemma-4-31B-it-GGUF` contains BOTH the main
    Q4_K_M weights and, under `MTP/`, a 491 MiB `mtp-gemma-4-31B-it-Q8_0.gguf` draft head. The old
    body did `sorted(d.glob("snapshots/*/*.gguf"))[0]` and survived only by alphabetical luck
    ("gemma-..." sorts before "mtp-..."), and only for as long as the head stayed in its `MTP/`
    subdirectory where the non-recursive glob could not see it. Either of those changing -- a
    rename, a flattened download, an `hf download` of the whole repo -- silently returns the DRAFT
    HEAD as the generator. Excluding it by name removes the dependence on both accidents.
    """
    base = Path.home() / ".cache" / "huggingface" / "hub"
    for d in base.glob(f"models--*{repo_substr}*GGUF"):
        hits = sorted(p for p in d.glob("snapshots/*/*.gguf") if not _is_mtp_head_file(p.name))
        if hits:
            return str(hits[0])
    return None


def _resolve_mtp_head(head_substr: str = ARC_LIVE_GENERATOR_MTP_HEAD_SUBSTR) -> Optional[str]:
    """Path to the MTP draft head GGUF, or None if it is not present on this machine.

    RETURNING None IS A FIRST-CLASS ANSWER, not an error path. `_ensure_server()` uses it to decide
    between "launch with speculative decoding" and "launch without it, and SAY SO". The one thing
    it must never do is let a caller fall back to the main weights file: `--model-draft <main
    gguf>` is precisely the configuration that llama.cpp accepts, warns about, and then serves with
    speculation silently disabled -- a misconfiguration indistinguishable from success except by
    measuring tok/s.

    Search order, most explicit first:
      1. `CARNOT_ARC_MTP_GGUF_PATH` -- how the Kaggle kernel hands over the path it resolved from
         the attached dataset mount. Honoured ONLY if the file exists, so a stale env var pointing
         at a deleted path degrades to "no head" (MTP off, loudly) rather than to a launch failure.
      2. The HuggingFace cache, searched RECURSIVELY, because upstream nests the head under an
         `MTP/` subdirectory inside the snapshot rather than beside the main weights.
      3. `~/.cache/kaggle_mtp_head_upload/` -- the operator's staging directory for the Kaggle
         dataset upload, which on this box is where the head actually is.
    """
    import os

    env = (os.environ.get("CARNOT_ARC_MTP_GGUF_PATH") or "").strip()
    if env:
        return env if Path(env).exists() else None
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    # REPO-SCOPED FIRST (2026-07-31). Both `gemma-4-31B-it-GGUF` and `gemma-4-31B-it-qat-GGUF`
    # ship a file named `mtp-gemma-4-31B-it-Q8_0.gguf`, so `head_substr` cannot tell them
    # apart and the old `sorted(...)[0]` over every GGUF root bound whichever sorted first.
    # After the 2026-07-31 switch to a QAT target that would have paired a NON-QAT drafter
    # with it -- accepted by llama.cpp, forbidden by Google's card ("the assistant model must
    # also be a QAT checkpoint with the same precision"), and invisible except as degraded
    # draft acceptance. So prefer a head living in the target's own repo, and only fall back
    # to the unscoped search when none is found there.
    repo_substr = ARC_LIVE_GENERATOR_MTP_HEAD_REPO_SUBSTR
    scoped = [r for r in hub.glob("models--*GGUF") if repo_substr in r.name]
    unscoped = [r for r in hub.glob("models--*GGUF") if repo_substr not in r.name]
    roots = scoped + unscoped + [Path.home() / ".cache" / "kaggle_mtp_head_upload"]
    for root in roots:
        try:
            hits = sorted(p for p in root.rglob("*.gguf") if head_substr in p.name)
        except OSError:  # a directory that does not exist / is unreadable is simply "no head here"
            continue
        if hits:
            return str(hits[0])
    return None


def _mtp_default_on() -> bool:
    """Whether MTP is on for a generator that was not given an explicit `mtp=`.

    Reads `CARNOT_ARC_MTP` against `ARC_LIVE_GENERATOR_MTP_DEFAULT`, which is the SAME expression
    both live construction sites in `arc_competition_agent.py` evaluate. It exists so the VRAM
    arithmetic can ask the question too: the guard has to budget for the head, and before this it
    had no way to know whether the launch it was sizing would load one.
    """
    import os

    return (os.environ.get("CARNOT_ARC_MTP", ARC_LIVE_GENERATOR_MTP_DEFAULT) or "0") != "0"


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

# ---------------------------------------------------------------------------------------------
# THE MEASURED VRAM ENVELOPE for the local 3090 generator launch (exp5866, 9 configs, refit max
# error 0.19%). Named constants rather than a magic literal because TWO things depend on the same
# arithmetic and MUST NOT drift apart: the context-pool size (_default_induce_n_ctx) and the
# free-VRAM guard that decides whether the card can hold the resulting server.
#
# SCOPE (2026-07-27 adversarial review, finding "envelope scoped to a shape the scored path never
# runs"): this fit was taken with `--spec-type draft-mtp` ON, i.e. the LOCAL/dev launch shape, where
# the MTP self-draft loads a second copy of the weights. It over-predicts the SCORED Kaggle shape by
# ~6.1 GB, because scripts/kaggle/submission_kernel/main.py forces CARNOT_ARC_MTP=0 there. The
# mtp-OFF pair is recorded separately below and is the one to reason about for the 16GB-class card.
# Over-prediction is the safe direction for a guard, so this constant is deliberately the mtp-ON fit.
#
# HISTORICAL AS OF 2026-07-28. These two constants describe the Qwen3.5-9B-MTP generator, which the
# operator directive of that date RETIRED in favour of gemma-4-31B-it (see the GENERATOR SWITCH
# block below). They are kept -- not deleted -- because they are the provenance of every VRAM
# number recorded in artifacts before that date, and because the 9B remains a legitimate
# CARNOT_ARC_GGUF_PATH override for a genuinely 16GB-class box. They are NO LONGER what
# `_generator_cuda_min_free_mb()` computes with; the gemma-4-31B fit below is.
_VRAM_MTP_ON_INTERCEPT_MIB = 10547.0  # weights + MTP self-draft copy + fixed overhead
_VRAM_MTP_ON_PER_CTX_MIB = 0.02519  # q8_0 KV per shared-pool cell
_VRAM_PER_SLOT_MIB = 206.83  # per llama.cpp slot, independent of n_ctx
# ---------------------------------------------------------------------------------------------
# THE MEASURED VRAM ENVELOPE FOR THE CURRENT GENERATOR: gemma-4-31B-it Q4_K_M, mtp OFF (the model
# is not an MTP model at all -- its GGUF header declares no `nextn_predict_layers`), q8_0 KV, the
# default 4 llama-server slots. Measured 2026-07-28 on an RTX 3090 as per-PID resident VRAM
# (`nvidia-smi --query-compute-apps`, joined PID -> GPU UUID -> index, never the env var):
#
#     n_ctx 32768, 0 FFN layers on CPU -> 21416 MiB
#     n_ctx 81920, 0 FFN layers on CPU -> 23888 MiB
#
# Two points, so the line below is an EXACT fit with no residual to report -- weaker evidence than
# the 9-config 9B refit above, and stated as such rather than dressed up. Its per-context slope is
# 0.0503 MiB/cell, very close to 2x the 9B's 0.02519: this model has roughly double the per-token
# KV, exactly the case `_default_induce_n_ctx()`'s docstring warned would invalidate the old guard.
# Leaving the 9B constants in place here would have under-predicted the 31B's footprint by ~9 GB,
# admitting a card the server then cudaMalloc-fails on -- i.e. the silent-LLM-off fault the guard
# exists to prevent, re-created by a model swap instead of by an n_ctx change.
_VRAM_GEMMA31B_PER_CTX_MIB = 0.050293  # (23888 - 21416) / (81920 - 32768)
_VRAM_GEMMA31B_INTERCEPT_MIB = 18940.7  # 21416 - 0.050293*32768 - 206.83*4
# ---------------------------------------------------------------------------------------------
# THE MTP DRAFT-HEAD SURCHARGE. Added when `--spec-type draft-mtp --model-draft <head>` is on.
#
# THIS IS NOT A FLAT CONSTANT, AND ASSUMING IT WAS IS THE BUG THIS BLOCK EXISTS TO PREVENT.
# Measured per-PID residency (`nvidia-smi --query-compute-apps`, PID -> GPU UUID -> index, never
# the env var), gemma-4-31B-it Q4_K_M + the real 491 MiB `mtp-gemma-4-31B-it-Q8_0.gguf` head,
# q8_0 KV, one server alone on an RTX 3090:
#
#     n_ctx 32768, 0 CPU-FFN layers -> 21402 MiB off / 22264 MiB on -> head costs  862 MiB
#     n_ctx 81920, 11 CPU-FFN layers -> 21730 MiB off / 23020 MiB on -> head costs 1290 MiB
#
# The head is a 491 MiB FILE but carries its own KV allocation proportional to the context pool,
# so its cost grows with `-c`. A flat 862 (or the ~840 recorded from an earlier single reading)
# would UNDER-PREDICT the shipped n_ctx 81920 configuration by ~428 MiB -- i.e. admit a card that
# then cudaMalloc-fails, which is exactly the silent-LLM-off fault the whole guard exists to stop,
# re-created by an MTP flag instead of by a model swap or an n_ctx change.
#
# Two points, so this is an EXACT fit with no residual to report -- stated plainly rather than
# dressed up. Its slope is ~17% of the main model's per-cell KV, consistent with a small draft
# head sharing the same pool geometry.
#
# THE 81920 PAIR IS NOW PERSISTED, WHICH IT WAS NOT WHEN THESE CONSTANTS WERE FIRST WRITTEN. The
# 32768 pair was always corroborated by `mtp_shipped_binary_t1/shipped_mtp_{on,off}.json`, but the
# 81920 readings were taken once off a live `nvidia-smi` and never written down -- an unrecorded
# number is indistinguishable from a remembered one, and this file asks the reader to trust it as
# evidence. Re-measured and RECORDED 2026-07-28 at
# `results/arc_gemma31b_migration_evidence_20260728/vram_81920_residency/`, per-PID residency
# joined PID -> GPU UUID -> index (never the env var, which only renames devices inside the child):
#
#     mtp OFF, 11 CPU-FFN layers -> 21716 MiB   (constant below predicts 21730, error 0.06%)
#     mtp ON,  11 CPU-FFN layers -> 23010 MiB   -> head surcharge 1294 MiB (fitted 1290, error 0.3%)
#
# Both arms landed on GPU-b52387a2 (index 0), and the ON arm's log carries the positive
# `adding speculative implementation 'draft-mtp'` marker while the OFF arm's does not -- so the
# surcharge is the cost of speculation ACTUALLY ENGAGING, not of an argument being accepted and
# silently ignored. The fitted constants are kept unchanged: they now have a re-measurement
# agreeing to well within the 1500 MiB guard margin, which is stronger than moving them to chase
# ~14 MiB of run-to-run scatter.
_VRAM_MTP_HEAD_PER_CTX_MIB = 0.008708  # (1290 - 862) / (81920 - 32768)
_VRAM_MTP_HEAD_INTERCEPT_MIB = 576.6  # 862 - 0.008708*32768
# The mtp-OFF arm of the 81920/11-layer measurement above is also an INDEPENDENT CHECK of the base
# envelope at a point it was never fitted on (11 offloaded layers, a count chosen by the auto-fit
# rather than by the original sweep): predicted 18940.7 + 0.050293*81920 + 206.83*4 - 195.3*11 =
# 21740 MiB, measured 21730 MiB, error 0.05%. Recorded here because a two-point fit that also
# predicts a third, differently-shaped configuration is materially stronger evidence than the
# two points alone.
#
# RE-MEASURED AND PERSISTED 2026-07-28 (same run as the MTP-head surcharge above):
# `results/arc_gemma31b_migration_evidence_20260728/vram_81920_residency/f13_81920_residency.json`
# records 21716 MiB for this exact shape. The literal below is kept at the original 21730 because
# it is the number the envelope was CHECKED against; the re-measurement agrees to 14 MiB (0.06%),
# which is the point -- it confirms the check rather than replacing it.
_VRAM_GEMMA31B_11LAYER_81920_CHECK_MIB = 21730
# VRAM freed per transformer block whose FFN weights are pushed to system RAM via `-ot` (see
# `_ffn_cpu_override_regex`). Measured at n_ctx 32768 across 0/12/24/40 CPU-FFN layers:
# 21416 / 19072 / 16728 / 13580 MiB -- dead linear, and cross-checked at n_ctx 81920 (12 layers
# -> 21544 MiB, predicted 21544). A least-squares slope over those four points is 195.9; the
# smaller 195.3 is used deliberately, because UNDER-crediting the saving makes the guard demand
# MORE free VRAM, which is the conservative direction for a guard.
_VRAM_PER_CPU_FFN_LAYER_MIB = 195.3
# llama-server with no explicit --parallel: n_parallel=4 AND kv_unified=true (server.cpp:106-110).
# READ from the source of the local build and CONFIRMED from a running server's own /props
# (`total_slots: 4`, 2026-07-27). It is the K the shared-pool admission arithmetic has to survive,
# because the eval framework starts one thread per game with no pool (swarm.py:91) and llama.cpp
# queues everything past its own slot count.
_LLAMA_SERVER_DEFAULT_SLOTS = 4


def _kv_quant_for_launch(field_value: Optional[str]) -> Optional[str]:
    """KV cache type the server should launch with. Env wins over the field.

    `CARNOT_ARC_KV_QUANT` (2026-08-11, REQ-ARC-WMTE-6253) exists because q8_0 was chosen
    when every card held 16-24 GB, and the scored card now holds 96 GB, where f16 KV is
    affordable. The env must OUTRANK the dataclass field: both live construction sites
    pass `kv_quant="q8_0"` explicitly, so a knob that lost to the argument could never be
    reached in production.

    Returns "none" unchanged; the caller drops the flags on that value. Unset env keeps
    today's behaviour exactly.

    This is a module-level function, not an inline `os.environ` read inside
    `_ensure_server`. That method has its own local `import os` further down, which makes
    `os` a function-local name for the WHOLE method, so an earlier `os.environ` read
    there raises UnboundLocalError and breaks the live launch. Caught by
    tests/python/test_arc_generator_config_knobs_6253.py before it shipped.
    """
    import os as _os

    raw = _os.environ.get("CARNOT_ARC_KV_QUANT", "")
    return raw.strip() if raw.strip() else field_value


def _llama_server_slots() -> int:
    """Slot count K the shared-pool arithmetic must survive. Env-overridable since
    2026-08-11 (REQ-ARC-WMTE-6253).

    WHY A KNOB NOW. The constant above is llama-server's own no-`--parallel` default,
    and it was correct while every card in play held 16-24 GB. The scored card is now a
    single 96 GB RTX PRO 6000, measured 2026-08-11. K is the multiplier in
    `_default_induce_n_ctx()`, so it decides the context pool, and there was no way to
    change it without editing source. That made the sizing untestable on the real card.

    THE DEFAULT DOES NOT MOVE. Unset env == 4 == today's behaviour, byte for byte.
    Raising K here changes the ARITHMETIC only. It does NOT add `--parallel` to the
    server launch, so the running server still serves its own default slot count. Any
    change to the real slot count must also pass `--parallel`, and must first be
    measured on the scored card through the preview A/B channel -- the
    `_default_induce_n_ctx` docstring records that a naive `--parallel 4` DIVIDES the
    pool per slot and was strictly worse.
    """
    import os

    raw = os.environ.get("CARNOT_ARC_LLAMA_SERVER_SLOTS")
    if raw is None or raw.strip() == "":
        return _LLAMA_SERVER_DEFAULT_SLOTS
    try:
        val = int(raw.strip())
    except ValueError:
        return _LLAMA_SERVER_DEFAULT_SLOTS
    # Refuse a nonsense K rather than propagate it into the VRAM and n_ctx arithmetic.
    return val if 1 <= val <= 64 else _LLAMA_SERVER_DEFAULT_SLOTS


# The real `induce_prompt()` for the largest logical grid in ops/arc_solve_registry.yaml (64x64),
# measured through the model's own tokenizer rather than estimated. The WORST case, not the
# typical one, because the generated length is unknowable in advance.
#
# RE-MEASURED 2026-07-28 for the gemma-4-31B generator, because a token count is a property of the
# TOKENIZER and the old 15734 was measured with Qwen3.5-9B's. Method: build one worst-case
# `induce_prompt()` (64x64 grids, k=8) and tokenize the SAME string with BOTH GGUFs via
# `llama_cpp.Llama(model_path=..., vocab_only=True)` -- the .gguf path, never AutoTokenizer on a
# GGUF repo id (CLAUDE.md GGUF tokenizer rule). Paired result: Qwen 17893 tokens, gemma 17930,
# ratio 1.00207. So 15734 * 1.00207 = 15767. The two vocabularies are within 0.2% on this prompt
# shape, which is why `_default_induce_n_ctx()` still returns 81920 -- 4*(15767+4096) = 79452,
# and the round-up to a 4096 multiple absorbs the difference entirely. The constant moves anyway:
# a stale tokenizer-derived number that happens not to change the answer today is still a landmine
# for the next person who changes max_tokens.
#
# RE-MEASURED AGAIN 2026-08-08 (REQ-ARC-WMTE-6227, 2026-08-08 adversarial review, Correctness
# finding 5). The 15767 figure above was measured at `k=8` transitions with NO object table.
# Neither is the shipped default any more: `k` defaults to ALL transitions (2026-08-01, see
# `_induce_transitions_k`), and the object-perception table is default ON (2026-08-07, see
# `_object_perception_on`). Method, same as the 2026-07-28 pass: built ONE worst-case
# `induce_prompt()` call (64x64 grid, 25 transitions, same rng seed 5900 as the prior
# measurement, so the two are apples-to-apples) via `results/arc_gemma31b_migration_evidence_
# 20260728/fitgrid/mkprompt.py`'s own methodology, extended to `k=None` (the current default)
# with `CARNOT_ARC_OBJECT_PERCEPTION` left unset (its own default, "1" -- object table ON) and
# `CARNOT_ARC_OBJECT_DELTA_PERCEPTION` left unset (REQ-ARC-WMTE-6213's block, still an uncommitted
# opt-in arm at measurement time, correctly excluded from a shipped-defaults figure). Tokenized
# through the gemma-4-31B-it Q4_K_M GGUF via `llama_cpp.Llama(vocab_only=True)`, same as before.
#
# RESULT: 22352 tokens under current defaults, vs 20071 tokens at k=8 with the object table
# (the object table alone already moved the k=8 figure from 17930 to 20071 -- +11.9% -- before
# the k=all default is even applied). 22352 is +41.8% over the stale 15767 this constant held,
# and it EXCEEDS the OLD per-slot budget (81920/4 - 4096 = 16384 tokens) by 5968 tokens -- a
# worst-case induce call at K=4 concurrency could genuinely overflow the old pool, exactly the
# HTTP-500 / server-death / silent-truncation failure modes `_default_induce_n_ctx()`'s own
# docstring documents. The constant moves to the real, current-defaults worst case rather than
# to a rounded or padded figure, because `_default_induce_n_ctx()` already rounds UP to the next
# 4096-token pool size and adding padding here would just be double-rounding.
_INDUCE_WORST_CASE_PROMPT_TOKENS = 22352
# Mirrors LocalGGUFProposer.max_tokens and the CARNOT_ARC_INDUCE_MAX_TOKENS default read at both
# construction sites in arc_competition_agent.py. Named here so the context-pool derivation and
# the completion budget cannot drift apart -- see _default_induce_n_ctx().
# CORRECTION 2026-08-21 (REQ-ARC-WMTE-6620): the mirror claim above is now historical. This
# constant is ONLY the pool-sizing per-slot completion reserve read by _default_induce_n_ctx().
# The per-request budget default is ARC_LIVE_GENERATOR_INDUCE_MAX_TOKENS_DEFAULT (131072, the
# scored kernel's own pin), resolved by _induce_max_tokens_default() below. The two split on
# purpose: a 24 GB card can host the 106,496-cell pool this reserve derives, but not the
# 614,400-cell pool the full budget would derive at 4 slots. A single local request instead
# uses the whole unified pool, clamped at request time -- see _pool_clamped_n_predict().
# The env var still raises BOTH together, exactly as before.
_INDUCE_DEFAULT_MAX_TOKENS = 4096


def _induce_max_tokens_default() -> int:
    """Per-request completion budget: CARNOT_ARC_INDUCE_MAX_TOKENS, else the generator pin's
    default. One resolver for every construction site (both in arc_competition_agent.py, the
    lever harness, and the dataclass default), so the budget cannot silently stay behind at
    the next generator swap -- the drift that produced the 2026-08-21 zero-world-model runs.
    Malformed env values fall back to the pin default rather than crash a live episode."""
    raw = os.environ.get("CARNOT_ARC_INDUCE_MAX_TOKENS")
    try:
        return int(raw) if raw else ARC_LIVE_GENERATOR_INDUCE_MAX_TOKENS_DEFAULT
    except (TypeError, ValueError):
        return ARC_LIVE_GENERATOR_INDUCE_MAX_TOKENS_DEFAULT


def _pool_clamped_n_predict(max_tokens: int, observed_n_ctx: Optional[int]) -> int:
    """The n_predict a request may actually ask for against the RUNNING server's pool.

    llama-server admission counts prompt + n_predict against the shared pool (see
    _default_induce_n_ctx: n_ctx >= K * (prompt + max_tokens)), so a 131,072-token budget
    sent at a local 106,496-cell pool is refused outright (HTTP 500), not merely truncated.
    Clamping to `pool - worst_case_prompt` lets a single stream use the pool's real room:
    84,144 tokens at the local default pool, which covers the measured Qwen3.8 median
    (62,490) and the max completed draw (83,444). No-op when the pool is large (scored
    path: 614,400 cells) or unobservable (vLLM serves no /props -> None).

    FLOOR (adversarial review of 512eca0e6b): a room of a few tokens would produce a
    "successful" 1-token completion -- worse than the loud admission refusal, because it
    looks like a healthy-but-terse model (exp5866's mode C). Below 1024 tokens of room no
    world-model engine can complete anyway, so pass through and let the server refuse."""
    if isinstance(observed_n_ctx, int) and observed_n_ctx > 0:
        room = observed_n_ctx - _INDUCE_WORST_CASE_PROMPT_TOKENS
        if 1024 <= room < max_tokens:
            return room
    return int(max_tokens)


# Yield-if-the-conductor-needs-it margin. Must cover measurement scatter (the same 81920/mtp-on
# launch measured 13452 and 13518 MiB per-PID on two occasions), the driver/context overhead
# nvidia-smi attributes outside the fit, and enough slack that we do not admit a card we will then
# cudaMalloc-fail on. A failed bind costs 180s in _ensure_server() and returns the agent to a
# SILENT LLM-off state -- exactly the class of fault this file's n_ctx fix exists to remove, so the
# guard must be conservative in the direction of declining the card.
_GENERATOR_CUDA_GUARD_MARGIN_MIB = 1500

# ---------------------------------------------------------------------------------------------
# INDUCE-PATH DECODE SAMPLER (wired 2026-07-31 -- REQ-ARC-FCP-5699-41)
#
# WHAT FAILED. The shipped induce path sends NO repetition penalty, so `llama-server` applies its
# default of 1.0 (read from the running server's own `/props`) -- i.e. none. The dominant induce
# failure is a decode-level repetition loop: the model reaches an impasse part-way through
# `engine()`, emits the same comment line until `n_predict` is exhausted, and never writes a
# `return`. ft09's live engine is 1112 of 1144 lines of duplicated comment. The defect census over
# 36 paired attempts:  `missing_return` 13 -> 2 and `engine_returned_none` 12 -> 2 with the penalty
# on, and the share of calls hitting the 4096-token cap falls 20/36 -> 2/36.
#
# WHAT IS CLAIMED, AND WHAT IS NOT. This ships on VALIDITY and COST, which is what was measured:
# attempts producing a mechanically-usable engine 13/36 -> 22/36, attempt-matched sign test
# p = 0.049 (17 discordant pairs), at 47.2 s per attempt against 100.3 s. It is NOT claimed to
# improve engine QUALITY: the strict out-of-sample quality funnel moved 2/6 games -> 3/6 on a
# single attempt, p = 1.000, and that channel had only 5 discordant pairs so its best reachable
# two-sided p was 0.0625 -- it could not have shown an effect at that n either way. On 2 of 6 games
# NO arm ever produced a correct engine. See
# docs/research-notes/arc-induce-repeat-penalty-confirm-2026-07-31.md.
#
# WHY DEFAULT ON, when `sampling_seed()` above argues at length that quietly changing how the
# scored agent samples is "a behaviour change, and it is not this function's to make". The two are
# consistent: that argument is against changing sampling as an unmeasured side effect of wanting a
# measurement. This is the sanctioned opposite -- a behaviour change adopted BECAUSE a paired
# measurement on the live prompt says it is better, under an explicit operator decision to ship it.
# The env var exists so a future A/B can turn it back off without a code edit; 1.0 disables the
# penalty exactly (llama.cpp treats 1.0 as identity), so `CARNOT_ARC_INDUCE_REPEAT_PENALTY=1.0`
# restores the pre-2026-07-31 payload byte-for-byte.
#
# SCOPE. Applied ONLY on the ENGINE code-only induce path -- `codeonly_eligible` AND `engine` in
# `required` -- which is exactly where it was measured. `refactor()` is a reasoning task on a
# different prompt shape and is deliberately untouched: penalising repetition in a debugging
# explanation is not the same intervention and has no measurement behind it here.
#
# NARROWED 2026-07-31 (adversarial review). As first wired, the gate was `codeonly_eligible`
# alone -- which is ALSO True for the focused goal-only call in `_split_induce`
# (`required=("is_level_complete",)`, see the `_goal_only_prompt` call site). So the penalty was
# reaching a second prompt shape that Phase 1 never measured: the confirm harness sent ENGINE
# prompts only. The source comment claimed the scope was "where it was measured", and that claim
# was true of the flag but not of the measurement.
#
# The condition now matches the defect gate's (`_defect_check_on`), which was already correctly
# narrower. Two reasons to narrow rather than merely disclose: (1) the project's standing rule is
# that behaviour changes ship WITH measurement, and the goal-only call had none in either
# direction -- so its correct state is the long-standing no-penalty baseline every banked result
# was produced under, not an untested 1.1; (2) the failure mode the penalty targets is a decode
# repetition loop that exhausts `n_predict`, and the goal-only call was split out of the combined
# call precisely BECAUSE it does not have that problem ("valid in ~3.5s where the combined call
# fails"). There is no mechanism for it to help there and no measurement saying it does.
#
# COST OF THE NARROWING, stated rather than buried: the 2026-07-31 Phase 3 pre-flight
# (`results/arc_phase3_preflight_20260731/`) exercised the WIDER configuration, so the shipped
# code no longer matches that artifact's config byte-for-byte. That artifact was a REFUSAL --
# nothing downstream depends on it, and its finding (arms emit byte-identical action traces;
# `n_plans_found` is 0 in every completed cell but one) is not a function of the goal-only
# sampler. Recorded in the Phase 2 note under "config drift".
_INDUCE_REPEAT_PENALTY = 1.1
# The window the penalty looks back over. 256 tokens is what was measured; it must be wide enough
# to span the repeating unit (a duplicated comment line plus its neighbours) or the penalty cannot
# see the loop it is meant to break.
_INDUCE_REPEAT_LAST_N = 256
# How many times a MECHANICALLY DEFECTIVE candidate may be rejected and re-asked. 1 is what was
# measured (round 0 + one plain re-ask). This is a floor on quality, never a cap on attempts: when
# the budget is spent the candidate is accepted anyway rather than failing the call -- see
# `generate()`.
_INDUCE_DEFECT_OWNS_ATTEMPTS_DEFAULT = "0"


def _defect_gate_owns_attempts() -> bool:
    """Do the defect gates get attempts a CONTENT failure cannot consume? Default NO.

    THE INCIDENT (2026-08-01, found in a live A/B's first treatment cell, not in review). ar25
    accepted a textbook `return False` -- the model's own comment reads "no win state was given
    ... maybe just return False" -- with `goal_defect_reasks == 0` AND `engine_defect_reasks == 0`.
    Both gates silent at once is what pointed at a shared cause rather than a bug in either.

    `attempt < tries - 1` guards both defect gates, and it is there for a real reason: a
    `continue` on the final attempt falls out of the loop into the content-failure return, which
    would convert an accept into a hard failure. But `tries` is ALSO the budget that content
    failures draw from -- a reply with no code block, a missing `def`, a syntax error. So every
    attempt the model wastes on malformed output is an attempt the defect gate needed, and an
    answer that finally parses on the last attempt is never checked at all.

    THE GATE IS THEREFORE QUIETEST EXACTLY WHERE IT IS MOST NEEDED. A game the model finds hard
    burns its attempts on unusable output, then lands its one parseable answer on the attempt
    where nothing is armed. Confirmed against a scripted server: accepted on attempt 0 gives 2
    re-asks; two content failures followed by the SAME defective answer gives 0, accepted
    unchecked.

    NOT A REGRESSION FROM THE GOAL GATE. The shipped ENGINE gate carries the identical guard and
    the identical blind spot, so its measured 13/36 -> 22/36 is a FLOOR on what that gate can do,
    not a ceiling.

    WHEN ON, a defect re-ask GRANTS one extra attempt instead of consuming a content attempt. That
    keeps the "never fails where the old path succeeded" guarantee intact by construction: the
    change only ever ADDS an iteration, and can never turn an accept into a failure. It is bounded
    without needing a cap, because the grant is gated on `_reasks_left` / `_goal_reasks_left`,
    which are finite (1 each by default) -- so the loop can grow by at most the re-ask budgets.

    DEFAULT OFF, and deliberately so: this changes shipped induce behaviour, and it was written
    while the A/B that found it was still collecting. Flipping it mid-run would have measured two
    treatments under one label. `CARNOT_ARC_INDUCE_DEFECT_OWNS_ATTEMPTS=1` enables it.
    """

    raw = os.environ.get("CARNOT_ARC_INDUCE_DEFECT_OWNS_ATTEMPTS")
    if raw is None:
        raw = _INDUCE_DEFECT_OWNS_ATTEMPTS_DEFAULT
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


_INDUCE_DEFECT_REASKS = 1
# The re-ask text. It names NOTHING about what went wrong, and that is a measured choice rather
# than laziness: `arc_engine_static_validation.repair_prompt_block()` builds a defect-naming block,
# and a head-to-head of repair-text against this neutral block over 5 discordant pairs on disjoint
# games came back p = 1.000. The defect TEXT bought nothing; the second ASK is the whole effect.
# `repair_prompt_block` is therefore deliberately left unwired -- it is a better-looking prompt
# with no evidence behind it.
_INDUCE_PLAIN_REASK_BLOCK = """
YOUR PREVIOUS ANSWER WAS RUN AGAINST THE OBSERVED TRANSITIONS AND WAS NOT SATISFACTORY.
Please try again from the same evidence:

  * re-read the observed transitions above before writing anything
  * take the simplest rule that is consistent with all of them
  * prefer a general rule over a table of the specific cases you were shown

Write the function again. `engine(grid, action, data)` must return a numpy array of the SAME
SHAPE as `grid` on EVERY path, and must not raise on any observed transition.
"""


def _induce_repeat_penalty() -> float:
    """The induce-path repetition penalty, env-overridable. 1.0 disables it exactly.

    A malformed value falls back to the measured default rather than raising: a typo'd env var
    must not take down a live episode, and the default is the behaviour we have evidence for.
    """
    raw = os.environ.get("CARNOT_ARC_INDUCE_REPEAT_PENALTY")
    if raw is None or raw.strip() == "":
        return _INDUCE_REPEAT_PENALTY
    try:
        return float(raw)
    except (TypeError, ValueError):
        return _INDUCE_REPEAT_PENALTY


def _induce_defect_reasks() -> int:
    """How many defect-triggered re-asks `generate()` may spend. 0 disables the re-ask entirely,
    leaving `repeat_penalty` as the only wired change (the two ship independently on purpose --
    the penalty carries 11 of the 13 paired wins, the re-ask 2)."""
    raw = os.environ.get("CARNOT_ARC_INDUCE_DEFECT_REASKS")
    if raw is None or raw.strip() == "":
        return _INDUCE_DEFECT_REASKS
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return _INDUCE_DEFECT_REASKS


_GOAL_DEFECT_REASKS = 2
# Wall-clock ceiling for the whole goal probe, and the cap on how many observed grids it runs
# against. Constancy needs TWO DISTINCT ANSWERS, not every frame, so capping the grid count
# costs almost no detection power while making the check's cost bounded and roughly constant
# across games -- wa30's window would otherwise probe 66 grids per candidate against cd82's 10.
_GOAL_PROBE_TIMEOUT_S = 10.0
_GOAL_PROBE_MAX_GRIDS = 12


def _goal_probe_sample(grids: list) -> list:
    """Grids to run the goal predicate on: HEAD AND TAIL, never the head alone.

    WHY THIS EXISTS (REQ-ARC-WMTE-6530, 2026-08-18). The probe used to take
    `grids[:_GOAL_PROBE_MAX_GRIDS]`, the first 12. Grids are appended (grid, next_grid) per
    transition, so 12 grids is the first SIX transitions -- and every window is built to END at
    its level flip, so the win frame is always the LAST grid. The win was therefore visible only
    for windows of six transitions or fewer.

    That inverts the check on longer windows. A CORRECT predicate -- one true only on win frames
    -- reads constant-False across everything the probe can see, so it is rejected and re-asked,
    while a predicate that fires early on a non-win frame reads non-constant and is kept. The
    gate rejected correct predicates and rewarded false positives, on exactly the windows where
    induction is hardest. Measured over `window_meta.json`: 7 of 13 games have more than six
    transitions, and the builder caps windows at `WINDOW_K = 12` TRANSITIONS while this cap is 12
    GRIDS -- the same number in two different units, so a capped window is exactly twice the
    probe's reach and the probe read the half that can never hold the win.

    The fix is not a bigger cap. A bigger number repairs the symptom until some window outgrows
    it, and the probe would still be pointed at the wrong end. Sampling both ends fixes it for
    any length at the same cost: the budget is split, the head keeps the early-frame coverage the
    original had, and the tail guarantees the win frame is always included.

    Cost is unchanged -- at most `_GOAL_PROBE_MAX_GRIDS` grids are returned, which is what bounds
    the hang exposure the caller's docstring describes. Order is preserved and duplicates are
    dropped, so a window at or under the cap probes exactly what it always did.
    """
    cap = _GOAL_PROBE_MAX_GRIDS
    if len(grids) <= cap:
        return list(grids)
    head = cap // 2
    tail = cap - head
    return list(grids[:head]) + list(grids[-tail:])


class _GoalProbeTimeout(Exception):
    """The goal probe exceeded its wall-clock ceiling. Deliberately NOT a defect verdict: a
    probe that could not finish is an absence of evidence, and the caller converts it to
    "accept, as before" rather than to `goal_raises`. Conflating "I could not check" with "I
    checked and it is broken" is how a guard starts inventing findings."""


def _goal_probe_timeout(_signum, _frame):  # pragma: no cover - signal path
    raise _GoalProbeTimeout


# The re-ask text for a mechanically defective GOAL. It is deliberately the same SHAPE as
# `_INDUCE_PLAIN_REASK_BLOCK` -- neutral, naming nothing about what was wrong -- because the
# project already ran that head-to-head for the engine: a defect-NAMING block against this
# neutral one over 5 discordant pairs on disjoint games came back p = 1.000. The defect TEXT
# bought nothing there; the second ASK was the whole effect. Re-deriving that on the goal
# would spend GPU time to relearn a settled negative, so the goal inherits the settled answer.
#
# What it does NOT say, and this is the line the whole intervention lives or dies on: it never
# tells the model what winning looks like. It says the answer was CONSTANT, which is a property
# of the answer, not a fact about the game. A predicate that returns the same value on every
# frame the agent has seen carries no information regardless of which game it is, so this text
# is exactly as available on a hidden game as on a solved one.
#
# SCOPED 2026-08-02 after adversarial review, because the paragraph above was written as if it
# covered all three bullets and it does not. Bullets 1-2 are properties of the ANSWER, computed
# on frames the agent observed for itself, and are bootstrap-free exactly as claimed. BULLET 3
# ("prefer a simple condition on a specific region, row, column or object over a whole-board
# property") IS NOT. It is a distributional prior about the SHAPE of ARC-AGI-3 win conditions,
# derived by the 2026-08-01 taxonomy from the 25 SOLVED public games: C_UNIFORMITY never wins,
# and E_FIXED_BAND is 11 of the 22 successes. A live agent meeting a hidden game would not have
# derived it on its own.
#
# It is disclosed rather than removed because it does not cross the line the discipline draws:
# it is game-agnostic, it names no specific win condition, and it would still function on a game
# nobody has ever solved -- which is the operative test. It is the same class of public-corpus
# scaffolding the ARC Solve Reproducibility discipline positively MANDATES be captured and
# reused (`general_gotchas`). Nothing scored depends on it either way: the flag ships OFF and
# the run that measured it recommends against flipping it. But if that default is ever flipped,
# this bullet -- not bullets 1-2 -- is the sentence an adversarial reviewer will pull on, and it
# should be disclosed before that happens rather than after.
_GOAL_PLAIN_REASK_BLOCK = """
YOUR PREVIOUS `is_level_complete` WAS RUN ON THE OBSERVED FRAMES AND RETURNED THE SAME ANSWER
ON EVERY ONE OF THEM, so it carries no information. Please try again from the same evidence:

  * a win condition that is never true, or always true, cannot distinguish anything
  * look at what the deltas above actually CHANGE, and write a condition on THAT
  * prefer a simple condition on a specific region, row, column or object over a whole-board
    property like "every cell is one colour"

Write `def is_level_complete(grid):` again. It must RETURN a bool on every path.
"""


def _goal_defect_check_on() -> bool:
    """Should a mechanically defective induced `is_level_complete` be rejected and re-asked?
    DEFAULT OFF (env `CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK=1`).

    WHY THIS EXISTS. `generate()` has dry-run defect rejection for the emitted `engine()` and
    NONE for the emitted goal -- its own `_engine_induce_call` gate is keyed on `"engine" in
    required`, so the focused goal-only call in `_split_induce` (`required=("is_level_complete",)`)
    is not merely unchecked, it is unreachable by the check. That asymmetry is not incidental:
    measured over 138 induced engines from 21 games, 71 of the 93 LIVE engines (76%) cannot
    yield a plan because of the GOAL rather than the dynamics, and the single largest slice --
    34 unconditional `return False` plus 3 with no return at all -- is mechanically detectable
    without knowing anything whatsoever about the game.

    THE BOOTSTRAP PROBLEM, and exactly how much of it this escapes. On a game the agent has
    never solved it has never seen a win, so it has NO positive example by construction; you
    need a win to learn what a win looks like. DETECTION escapes that completely -- "is this
    predicate constant over the frames I have already observed" needs no win, no positive
    example, and no environment, only the agent's own observations. REPAIR does not escape it:
    a model that has seen no win may simply re-emit a different trope, and this flag has no
    answer to that. So the honest claim is narrow: this converts a detectable generation defect
    into another sample, and makes no claim to supply information the agent does not have.

    WHY IT SHIPS OFF. It is a live-path behaviour change on the scored agent whose only
    evidence is observational (a frozen corpus, scored after the fact). Default OFF makes the
    control arm of any measurement the SHIPPED path rather than a reimplementation of it.
    Flipping the default is an operator decision that belongs after the measurement.

    A malformed value falls back to OFF rather than raising -- a typo'd env var must not change
    how the scored agent behaves, and must certainly not take down a live episode.
    """
    raw = os.environ.get("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK")
    return bool(raw) and raw.strip() == "1"


def _goal_defect_reasks() -> int:
    """How many goal-defect re-asks `generate()` may spend. Its OWN budget, deliberately NOT
    shared with `_induce_defect_reasks()`.

    THE CONFOUND THIS AVOIDS, which is a measurement bug and not a nicety. On the combined
    induce call one budget serves both functions. 89% of induced goals are constant over every
    observed frame (measured, pre-flight over 115 frozen engines), so a shared counter would be
    consumed by the goal on almost every cell and the ENGINE would silently lose its re-ask in
    the treatment arm only. The arm difference would then be part goal-check and part
    engine-check-removal, and no analysis could separate them afterwards.

    A malformed value falls back to the default rather than raising.
    """
    raw = os.environ.get("CARNOT_ARC_INDUCE_GOAL_DEFECT_REASKS")
    if raw is None or raw.strip() == "":
        return _GOAL_DEFECT_REASKS
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return _GOAL_DEFECT_REASKS


def _goal_prompt_transitions_on() -> bool:
    """Should the focused goal-only prompt CARRY the observed transitions? DEFAULT OFF
    (env `CARNOT_ARC_GOAL_PROMPT_TRANSITIONS=1`).

    `_goal_only_prompt` today receives a game name and one grid, and no transitions at all --
    it is the evidence-free prompt in the pair, and the 2026-08-01 taxonomy traced 12 of 13
    whole-board "every cell is one colour" predicates to exactly the cells it produced. Those
    cells are split-induce cells, where the model writes an evidence-GROUNDED predicate
    alongside the engine and it is then SHADOWED by a second definition generated from a
    prompt containing no grid and no deltas; Python binds the second one.

    The transitions are the agent's OWN observations, so showing them crosses no line: it is
    help USING what the agent has already seen for itself, not a fact about the game supplied
    from outside.

    NARROWED 2026-08-02. This paragraph used to end "It is exactly as available on a hidden game
    as on a solved one." That is true of the FIELD and untested for the DISTRIBUTION, and the
    difference matters. The 2026-08-02 A/B that measured this flag scored it against windows from
    `build_progress_window`, i.e. the last k actions of a BANKED WINNING ROUTE cut at the L0->L1
    boundary -- a game that cannot be solved to L1 offline produces no window at all. On a hidden
    game the `trans` reaching this prompt is whatever the stall-triggered exploration buffer
    holds, which is a strictly weaker sample from a different distribution. The sentence had been
    read as an empirical result and had propagated into an artifact as
    `works_on_an_unsolved_game: true`; that claim was retracted. Whether the flag helps on
    exploration-buffer transitions is OPEN and needs an adapter-free harness to answer.
    """
    raw = os.environ.get("CARNOT_ARC_GOAL_PROMPT_TRANSITIONS")
    return bool(raw) and raw.strip() == "1"


def _goal_dedup_on() -> bool:
    """Should the split-induce path guarantee EXACTLY ONE `is_level_complete`? DEFAULT OFF
    (env `CARNOT_ARC_GOAL_DEDUP=1`).

    THE DEFECT. On the split-induce fallback, `_combine_world_model` concatenates the
    engine-only completion and the goal-only completion. The engine-only prompt is
    `induce_prompt(...)` and carries the observed transitions; the goal-only prompt is
    `_goal_only_prompt(...)` and by default carries none. The model frequently writes an
    `is_level_complete` in the ENGINE-only response as well (the base prompt describes the
    whole interface), so the concatenation ships TWO top-level definitions and Python binds
    the second -- the evidence-free one. Measured 2026-08-02 over the frozen corpus at
    `results/arc_goal_predicate_shadowing_20260802/`: 23 of 116 concatenated world models
    carry two definitions, against 0 of 40 raw single-call completions, so the duplication
    is produced by the concatenation and not by the model redefining the function.

    WHAT THE MEASUREMENT DID **NOT** SHOW, recorded here because the docstring on
    `_goal_prompt_transitions_on` above asserts the motivating half of this story and it is
    only PARTLY right. Grading both definitions of all 23 cells through the shipped goal gate
    (`arc_llm_reinduction._goal_satisfiability_check`, same engine, same root grid, one
    killable subprocess each) does NOT show the shadowed definition is systematically better:
    2 shadowed arms were satisfiable against 1 bound arm, and in 20 of 23 cells BOTH were
    unsatisfiable, so the gate cannot separate them at all. Verified stable against the root
    grid used -- on the 3 games with a recorded planner root, that root is byte-identical to
    the arcade opening board and 0 of 10 arm verdicts changed. So this flag must NOT be sold
    as recovering a good predicate; on the gate's own criterion the expected gain is +1 cell
    in 23, which is noise.

    WHAT IT DID SHOW, and the only reason this ships at all: a ONE-SIDED VALIDITY defect.
    4 of 23 BOUND definitions are not usable predicates in the first place -- two return
    `None` (the body falls through), one raises `NameError` on a variable the model never
    bound, and one does not terminate inside 120s -- against 0 of 23 shadowed definitions.
    Binding the second definition is therefore strictly worse on validity while being a coin
    flip on quality. The failure modes also differ sharply: every one of the 11 unconditional
    `return False` predicates sits in the ENGINE half, and every one of the 8 whole-board
    "one colour" tropes sits in the GOAL-ONLY half.

    WHY IT SHIPS OFF. It changes what the scored agent writes to `world_model.py`, and its
    evidence is observational -- a frozen corpus graded after the fact, not an A/B on the live
    path. Default OFF keeps the control arm of any future measurement the SHIPPED path rather
    than a re-rendering of it, exactly as `_goal_defect_check_on` and
    `_goal_prompt_transitions_on` do. Flipping the default is an operator decision.

    A malformed value falls back to OFF: a typo'd env var must not change how the scored agent
    behaves.
    """
    raw = os.environ.get("CARNOT_ARC_GOAL_DEDUP")
    return bool(raw) and raw.strip() == "1"


def _goal_predicate_is_constant_false(code: str) -> bool:
    """Does the LAST top-level `is_level_complete` in `code` do nothing but `return False`?

    A declined predicate is not a bug -- given a prompt that says nothing about the win
    condition, `return False` is arguably the honest answer, and 11 of the 23 engine halves in
    the frozen corpus say so in their own comments. But it carries no information, so it must
    NOT out-rank a goal-only completion: `_split_induce` treats "the engine half already
    answered" as false when the answer is this one, and still spends the focused goal call.

    Docstrings and bare constant expressions are ignored so a commented-out or documented
    decline still counts as one. Unparseable input is NOT constant-false -- an unparseable
    half is a different defect and is caught by the static validator, not here.
    """
    import ast as _ast

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return False
    fns = [
        n for n in tree.body if isinstance(n, _ast.FunctionDef) and n.name == "is_level_complete"
    ]
    if not fns:
        return False
    body = [
        s
        for s in fns[-1].body
        if not (isinstance(s, _ast.Expr) and isinstance(s.value, _ast.Constant))
    ]
    return (
        len(body) == 1
        and isinstance(body[0], _ast.Return)
        and isinstance(body[0].value, _ast.Constant)
        and body[0].value.value is False
    )


def _engine_half_goal_usable(engine_code: str) -> bool:
    """Did the ENGINE-only completion already supply a goal predicate worth keeping?

    Three conditions, all necessary. It must be PRESENT; it must be free of the shipped
    static validator's defects for that function name (`missing_return` is the one that
    matters -- a predicate whose body falls through returns `None`, and `None` is not an
    answer); and it must not be the constant-false decline, which carries no information.

    Deliberately reuses `arc_engine_static_validation.missing_return_defects` rather than
    re-deriving the check, so this cannot drift away from the validator the project already
    ships for exactly this question.
    """
    import ast as _ast

    try:
        tree = _ast.parse(engine_code)
    except SyntaxError:
        return False
    if not any(
        isinstance(n, _ast.FunctionDef) and n.name == "is_level_complete" for n in tree.body
    ):
        return False
    if _goal_predicate_is_constant_false(engine_code):
        return False
    from carnot.agentic import arc_engine_static_validation as _sv

    return not _sv.missing_return_defects(engine_code, "is_level_complete")


def _strip_top_level_goal_defs(code: str) -> str:
    """Remove every top-level `def is_level_complete` from `code`, preserving all else.

    Line-range excision on the original text, NOT `ast.unparse`: the engine half carries the
    model's own comments and helper functions, and a round-trip through `unparse` would
    silently rewrite all of it. A definition nested inside another function is left alone --
    `exec` never binds it at module level, so it is not a competing definition.
    """
    import ast as _ast

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return code
    drop: set[int] = set()
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name == "is_level_complete":
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            for ln in range(start, (node.end_lineno or node.lineno) + 1):
                drop.add(ln)
    if not drop:
        return code
    kept = [l for n, l in enumerate(code.splitlines(), start=1) if n not in drop]
    return "\n".join(kept).rstrip() + "\n"


def _reject_inert_engines() -> bool:
    """Should a CLEAN-BUT-INERT induced engine be rejected and re-asked? DEFAULT OFF.

    An INERT engine is one that predicts no action changes anything -- the identity function,
    modulo whatever code shape it wears. It clears every mechanical check in
    `arc_engine_static_validation` and it is, per the 2026-08-01 taxonomy, the LARGEST single
    failure class of the live generator: 26 of 172 gemma-4-31B candidates (15.1%), more than
    every code-validity class combined, and the only class the induce path took no action on
    despite `engine_changes_anything` being shipped and already imported here.

    WHY IT SHIPS OFF. This is a live-path behaviour change on the scored agent, and the only
    evidence for it is observational (a frozen corpus, scored after the fact). Default OFF makes
    it a true A/B: with the flag unset `_engine_defects` is behaviourally identical to the code
    that preceded it, so the control arm of any measurement IS the shipped path rather than a
    reimplementation of it. Flipping the default is an operator decision that belongs after the
    measurement, not inside it.

    A malformed value falls back to OFF rather than raising -- a typo'd env var must not change
    how the scored agent behaves, and must certainly not take down a live episode.
    """
    raw = os.environ.get("CARNOT_ARC_INDUCE_REJECT_INERT")
    return bool(raw) and raw.strip() == "1"


def _generator_cuda_min_free_mb(
    ffn_cpu_layers: Optional[int] = None, mtp: Optional[bool] = None
) -> int:
    """Free VRAM (MiB) the opt-in 3090 generator path requires before it will bind a card.

    `mtp` MUST be the value the server will actually launch with, for exactly the reason
    `ffn_cpu_layers` must be: the MTP draft head is a real, `n_ctx`-dependent VRAM cost
    (+1290 MiB at the shipped n_ctx 81920) and a guard blind to it validates a configuration the
    server is not about to run. It defaults to `_mtp_default_on()` for module-level callers;
    `_ensure_server()` passes `self.mtp` explicitly, and that is the important case -- a proposer
    can carry an explicit constructor value the env default knows nothing about.

    `ffn_cpu_layers` MUST be the count that will actually reach the launch argv as `-ot`. It
    defaults to `_default_ffn_cpu_layers()` for the module-level callers, but `_ensure_server()`
    passes `self.ffn_cpu_layers` explicitly, and that is the important case: a proposer can carry
    an EXPLICIT constructor value that the default factory knows nothing about, and even when it
    does not, the auto-fit re-reads free VRAM and could return a different number at construction
    time than at launch time. Either way the guard would then be validating a configuration the
    server is not about to run -- admitting a card on the strength of an offload that never
    happens is the same cudaMalloc-then-silent-LLM-off failure this guard exists to prevent,
    reintroduced through the back door by the fix for it.

    DERIVED, never a hand-typed literal. It was a literal (13000, commented "loads ~11.5GB") and
    the 2026-07-27 n_ctx 16384 -> 81920 fix raised the real footprint to ~13.4-13.5 GiB WITHOUT
    touching it, so the guard would have admitted a card with 13000-13452 MiB free and then
    cudaMalloc-failed: server exits, `_ensure_server()` burns its full retry budget, `generate()`
    returns `(False, msg)`, and the agent runs LLM-off while still reporting itself as the LLM-on
    scored path. That is a NEW silent-degradation path of exactly the shape the fix was removing.

    Computing it from the SAME `_default_induce_n_ctx()` the server is actually launched with means
    an operator raising CARNOT_ARC_INDUCE_N_CTX automatically raises the guard too. Pinned by
    `tests/python/test_arc_generator_vram_guard.py`.

    2026-07-28, THE GENERATOR SWITCH. Two things changed here and both had to, together:

      1. The envelope is now the gemma-4-31B fit, not the Qwen3.5-9B one. The 9B constants
         under-predict the 31B by ~9 GB; keeping them would have let the guard admit a 3090 that
         the 31B server then cudaMalloc-fails on, which is the silent-LLM-off fault this function
         exists to prevent -- caused by a model swap rather than by an n_ctx change.
      2. It now subtracts the FFN-to-system-RAM credit, because `CARNOT_ARC_FFN_CPU_LAYERS` moves
         the real footprint by ~195 MiB per layer and a guard that ignores the lever would decline
         a card the configured server would in fact have fitted on.

    CONSEQUENCE AT THE DEFAULTS, AND THE CORRECTION THAT FOLLOWED. gemma-4-31B at n_ctx 81920 with
    NO FFN offload predicts 23888 MiB + 1500 MiB margin = 25388 MiB required free, which EXCEEDS a
    24576 MiB 3090 outright -- the guard becomes unsatisfiable by arithmetic, not by contention.

    This function's first version recorded that as "the correct answer and not a bug", on the
    stated ground that the iGPU fallback was "slower, but functional and LOUD, never a silent
    LLM-off". BOTH HALVES OF THAT WERE FALSE, and the second pass of the 2026-07-28 review proved
    it by measurement rather than argument:

      * NOT FUNCTIONAL. The iGPU HIP build runs gemma-4-31B Q4_K_M at ~2 tok/s decode. Against
        `max_tokens=4096` and a 600 s timeout, a single induce call cannot finish. `generate()`
        returns `(False, msg)` and the agent proceeds LLM-OFF while still reporting itself as the
        LLM-on scored path -- precisely the silent degradation this guard exists to prevent.
      * NOT LOUD. `_generator_server_and_env()` fell through with no output whatsoever. There is
        now an actual channel (`GENERATOR_SELECTION_LOG`, mirrored to stderr and copied onto the
        proposer) and every placement decision writes to it.

    The fix is NOT to weaken this arithmetic -- it is measured and correct. It is that
    `_default_ffn_cpu_layers()` now AUTO-SELECTS the fewest offload layers that make the card fit
    whenever the operator has opted into a local CUDA card, so this requirement comes down to meet
    the hardware instead of the generator silently leaving it. Because that credit is subtracted
    below, raising the offload lowers what this returns, and the two stay consistent by
    construction.
    """
    layers = _default_ffn_cpu_layers() if ffn_cpu_layers is None else int(ffn_cpu_layers)
    predicted = _predicted_generator_vram_mib(_default_induce_n_ctx(), layers, mtp)
    return int(predicted + _GENERATOR_CUDA_GUARD_MARGIN_MIB)


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


def _cuda_gpu_total_mb(idx: int) -> int:
    """TOTAL VRAM (MiB) on CUDA GPU `idx` via nvidia-smi; -1 if unavailable.

    WHY THIS EXISTS separately from `_cuda_gpu_free_mb`. The retry loop in
    `_cuda_gpu_has_headroom` waits for a *transient* condition to clear -- a just-crashed
    process whose VRAM the driver has not reclaimed yet. It is worth 20 s of patience for that.
    But if the requirement exceeds the card's TOTAL capacity, no amount of waiting can help: the
    condition is not transient, it is arithmetic. Distinguishing the two needs the total, and
    that distinction is what turns an 18 s pause-then-silently-degrade into an immediate,
    explained refusal.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return int(lines[idx]) if 0 <= idx < len(lines) else -1
    except Exception:
        return -1


# THE GENERATOR-SELECTION AUDIT LOG. Append-only, module-level, and mirrored to stderr.
#
# WHY A MODULE GLOBAL. `_generator_server_and_env()` and `_default_ffn_cpu_layers()` are plain
# module functions with no instance to hang a record on, and this module has NO logger (deliberately
# -- it is imported into the Kaggle kernel where logging config is not ours). Yet the decisions they
# make are exactly the ones that were invisible in the 2026-07-28 review: the guard declined the
# CUDA card and `_generator_server_and_env()` fell through to the iGPU with ZERO output, despite its
# docstring asserting the fallback was "functional and LOUD". It was functional. It was not loud.
#
# `_ensure_server()` copies this onto the proposer instance (`generator_selection_log`) so an
# artifact can carry the reason a run was slow, instead of the reason being unrecoverable after the
# fact. Bounded so a long-lived process cannot grow it without limit.
GENERATOR_SELECTION_LOG: list[str] = []
_GENERATOR_SELECTION_LOG_MAX = 200
_GENERATOR_SELECTION_SEEN: set = set()


def _note_generator_selection(msg: str) -> None:
    """Record + announce a generator-placement decision. Never raises: an audit channel that can
    break the generator it audits is worse than no audit channel.

    DEDUPED BY MESSAGE TEXT. `_default_ffn_cpu_layers()` is a dataclass `default_factory` and is
    also re-read by the guard, so a single launch evaluates it several times and an un-deduped
    channel prints the same paragraph five times before the server even starts. Repetition trains
    the reader to skip the channel, which defeats the point of adding it. The message text embeds
    the free-VRAM reading, so a genuinely CHANGED situation still produces a new line.
    """
    try:
        line = f"[carnot.arc.generator] {msg}"
        if line in _GENERATOR_SELECTION_SEEN:
            return
        _GENERATOR_SELECTION_SEEN.add(line)
        if len(GENERATOR_SELECTION_LOG) < _GENERATOR_SELECTION_LOG_MAX:
            GENERATOR_SELECTION_LOG.append(line)
        print(line, file=sys.stderr, flush=True)
    except Exception:
        pass


def _predicted_generator_vram_mib(
    n_ctx: int, ffn_cpu_layers: int, mtp: Optional[bool] = None
) -> float:
    """Predicted resident VRAM (MiB) for the CURRENT generator at `n_ctx` with `ffn_cpu_layers`
    FFN blocks in system RAM. The single arithmetic both `_generator_cuda_min_free_mb()` and the
    auto-fit search below evaluate, so a change to the envelope cannot move one and not the other.

    `mtp` defaults to `_mtp_default_on()` -- the same expression the live construction sites use --
    so a module-level caller that does not thread it still budgets for the configuration that will
    actually launch. Pass it explicitly when the answer is known (a proposer instance carries its
    own `mtp` field, which may differ from the env default).
    """
    on = _mtp_default_on() if mtp is None else bool(mtp)
    head = (_VRAM_MTP_HEAD_INTERCEPT_MIB + _VRAM_MTP_HEAD_PER_CTX_MIB * float(n_ctx)) if on else 0.0
    return (
        _VRAM_GEMMA31B_INTERCEPT_MIB
        + _VRAM_GEMMA31B_PER_CTX_MIB * float(n_ctx)
        + _VRAM_PER_SLOT_MIB * float(_llama_server_slots())
        + head
        - _VRAM_PER_CPU_FFN_LAYER_MIB * float(max(0, ffn_cpu_layers))
    )


# gemma-4-31B-it has 60 transformer blocks (read from the GGUF tensor table: blk.0..blk.59, three
# FFN tensors each). Offloading all 60 is the most VRAM the `-ot` lever can free; beyond that the
# regex simply matches nothing more.
_GEMMA31B_N_BLOCKS = 60
# Cap the AUTO-selected offload. The measured table in `_default_ffn_cpu_layers()` shows prefill --
# which is what the induce path is bound by at a 15767-token worst-case prompt -- degrading 4.7x at
# 12 layers and 13x at 40. Auto-selecting past this point would trade a silent slow fallback for a
# silent slow CUDA path, which is not an improvement. If more than this is genuinely needed, the
# card is too full and the operator should be told rather than accommodated.
#
# WAS 24, LOWERED TO 12 (2026-07-28, third pass) BECAUSE 24 CONTRADICTED THE DOCSTRING THAT
# JUSTIFIES IT. `_default_ffn_cpu_layers()` states in terms: "treat anything past ~12 layers as
# likely to push real induction into timeout". A cap of 24 let the auto-fit select 23 layers at
# 21000 MiB free and 24 at 20750 -- i.e. straight through the threshold its own measured table
# names, into the region the docstring calls a timeout. `_default_induce_timeout_s()` does scale
# the budget with the layer count (3.8x at 24 layers), but it interpolates a SINGLE-STREAM decode
# table while the live path runs 4 `kv_unified` slots, so the real per-request rate is lower than
# that curve and the scaled timeout is not known to cover it. Nothing measured 4-slot decode at
# 20+ layers, so the honest move is to stop auto-selecting into unmeasured territory rather than
# to keep a cap justified by a number nobody took.
#
# CONSEQUENCE, STATED RATHER THAN DISCOVERED LATER: at the shipped n_ctx 81920 on a 24123 MiB
# card, MTP-OFF needs 7 layers (fits under this cap) and MTP-ON needs 14 (does NOT). So a LOCAL
# MTP-on launch now hits the `-1` branch and says loudly that it cannot fit, instead of quietly
# auto-selecting a 14-layer offload. That is the intended outcome and not a regression: the local
# MTP-on configuration is independently measured as a NET THROUGHPUT LOSS (a 14-layer offload
# costs more decode than MTP's 1.398x returns -- see the GENERATOR SWITCH block), which is exactly
# why `ARC_LIVE_GENERATOR_MTP_DEFAULT` is "0". The cap now enforces that conclusion instead of
# leaving a path that silently contradicts it. The SCORED path is untouched: the 96 GB card needs
# no offload at all, and the Kaggle kernel never reaches this code (see `_default_ffn_cpu_layers()`
# "KAGGLE IS UNAFFECTED").
_FFN_CPU_AUTOFIT_MAX_LAYERS = 12
# `LocalGGUFProposer.ffn_cpu_layers` default meaning "nobody chose -- fit it in `__post_init__`,
# where `self.mtp` is visible". Negative so it can never collide with a real layer count, and
# distinct from 0 so "the caller explicitly asked for no offload" stays a statement the caller can
# make. See `LocalGGUFProposer.__post_init__` for why a `default_factory` could not do this job.
_FFN_CPU_LAYERS_AUTO = -1


def _ffn_cpu_layers_to_fit(free_mb: int, n_ctx: int, mtp: Optional[bool] = None) -> int:
    """Fewest FFN-on-CPU layers that brings the predicted footprint + guard margin under `free_mb`.

    Returns -1 when even `_FFN_CPU_AUTOFIT_MAX_LAYERS` is not enough -- i.e. the card genuinely
    cannot host this generator and the honest answer is to say so, not to keep offloading until
    the thing is slower than the fallback it was meant to beat.

    THE POSTCONDITION IS THE POINT, and it is asserted in the tests rather than merely intended:
    a non-negative return MUST satisfy `_generator_cuda_min_free_mb(n, mtp) <= free_mb`. Both
    functions evaluate `_predicted_generator_vram_mib` with the same margin, so this holds by
    construction -- but only for as long as they keep taking the same arguments, which is precisely
    what broke when `mtp` entered the arithmetic on one side only.
    """
    for n in range(0, _FFN_CPU_AUTOFIT_MAX_LAYERS + 1):
        need = _predicted_generator_vram_mib(n_ctx, n, mtp) + _GENERATOR_CUDA_GUARD_MARGIN_MIB
        if need <= free_mb:
            return n
    return -1


# How many times _generator_server_and_env retries the free-VRAM check before conceding to the
# iGPU fallback, and how long it waits between retries. A just-crashed CUDA process does not
# release its VRAM allocation to the driver instantaneously; LocalGGUFProposer's self-heal path
# (_ensure_server) calls _generator_server_and_env() immediately after detecting an unhealthy
# server, so a single free-memory snapshot can catch the dying process's VRAM still "in use" and
# wrongly fall back to the iGPU for that server's entire subsequent lifetime. Found 2026-07-21
# (exp5768): three consecutive self-heals after a CUDA server crash all silently landed on the HIP
# build with near-zero VRAM, running a 31B model on CPU for hours before being noticed. An initial
# fix used 4 attempts / 1.5s apart (~6s total) -- confirmed insufficient 2026-07-22 when the exact
# same failure recurred (a longer reclaim window than 6s in that instance). Widened to 10 attempts
# / 2s apart (~20s total) after that. Still bounded small enough that a genuinely busy card (a real
# conductor job actually holding the VRAM) yields to the iGPU within seconds, not a long stall.
_GENERATOR_CUDA_FREE_RETRY_ATTEMPTS = 10
_GENERATOR_CUDA_FREE_RETRY_DELAY_S = 2.0


def _cuda_gpu_has_headroom(idx: int, min_free_mb: int) -> bool:
    """True if GPU `idx` has >= `min_free_mb` free, retrying briefly across
    _GENERATOR_CUDA_FREE_RETRY_ATTEMPTS attempts to survive a just-crashed process's VRAM not yet
    being reclaimed by the driver (see the constants' docstring above).

    TWO FIXES, 2026-07-28, both from the same root cause -- the generator switch raised
    `_generator_cuda_min_free_mb()` from 14937 MiB (which a 3090 passes) to 25388 MiB (which
    EXCEEDS a 3090's 24576 MiB total), making the guard unsatisfiable by arithmetic rather than by
    contention:

      1. SHORT-CIRCUIT THE WAIT. Retrying is for a transient condition. When the requirement
         exceeds the card's TOTAL capacity, waiting cannot ever help, and the old loop burned
         10 x 2.0 s = 18 s of deterministic sleep per launch before conceding. Now that case is
         detected on the first pass and returns immediately.
      2. SAY WHY. The old function returned a bare False and the caller fell through silently, so
         a run that lost its CUDA card looked identical to a run that never asked for one. Every
         refusal now names the card, the requirement, the free amount, and -- when it is the
         arithmetic case -- the total, which is what tells the reader this is not a busy-card
         problem they can wait out.
    """
    total = _cuda_gpu_total_mb(idx)
    if 0 < total < min_free_mb:
        _note_generator_selection(
            f"CUDA gpu{idx} DECLINED (not transient): the configured generator needs "
            f"{min_free_mb} MiB free but the card's TOTAL capacity is {total} MiB. No wait can "
            f"satisfy this. Lower CARNOT_ARC_INDUCE_N_CTX or raise CARNOT_ARC_FFN_CPU_LAYERS."
        )
        return False
    last_free = -1
    for attempt in range(_GENERATOR_CUDA_FREE_RETRY_ATTEMPTS):
        last_free = _cuda_gpu_free_mb(idx)
        if last_free >= min_free_mb:
            return True
        if attempt < _GENERATOR_CUDA_FREE_RETRY_ATTEMPTS - 1:
            time.sleep(_GENERATOR_CUDA_FREE_RETRY_DELAY_S)
    # REPORT the reading we already took. An earlier draft called `_cuda_gpu_free_mb(idx)` again
    # inside this message, which spawns an extra nvidia-smi subprocess purely to render a log line
    # -- and, worse, makes the number in the message a DIFFERENT observation from the one that
    # drove the decision. `test_generator_server_and_env_cuda_retry.py` caught it as an off-by-one
    # in the probe count, which is the honest symptom of a diagnostic that perturbs what it reports.
    _note_generator_selection(
        f"CUDA gpu{idx} DECLINED after "
        f"{_GENERATOR_CUDA_FREE_RETRY_ATTEMPTS} x {_GENERATOR_CUDA_FREE_RETRY_DELAY_S}s: needs "
        f"{min_free_mb} MiB free, last saw {last_free} MiB free of {total} MiB total "
        f"-- another process is holding the card."
    )
    return False


def _free_port() -> int:
    """An OS-assigned free localhost port, for the case where the port we wanted is already
    held by a server whose context pool is too small for us (see `_reusable`). Binding to 0
    and reading the assignment back is the only race-free way to pick one; the tiny window
    between close() and llama-server's bind() is accepted because the alternative -- reusing
    a mismatched server -- is the silent-degradation fault this whole path exists to remove."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# Aggregate VRAM across all cards RISES when a model is layer-split, because each card carries
# its own compute buffers and non-layer state. Measured 2026-07-31 over five arms on 2x RTX 3090
# (results/outer_loop_arc_gpu_layer_split_sweep_20260731.json):
#     n_ctx 32768: 20428 MiB single -> 21412 MiB summed across two cards (x1.048)
#     n_ctx 81920: 22900 MiB single -> 24604 MiB summed across two cards (x1.074)
# 1.10 rounds the worse of the two up. It is deliberately a headroom multiplier and NOT a
# saving: a split does not reduce total memory, it redistributes it. The per-card requirement is
# min_free * this / n_cards, so admitting a card is still gated on real measured arithmetic.
_SPLIT_AGGREGATE_OVERHEAD = 1.10


def _parse_generator_cuda_gpus(raw: str) -> list[int]:
    """Parse CARNOT_ARC_GENERATOR_CUDA_GPU into physical CUDA indices.

    Accepts a single index ("1", the historical form -- behaviour unchanged) or a comma list
    ("0,1") requesting a layer-split across those cards. Malformed entries yield [] so the caller
    falls through to its existing refusal path rather than binding an arbitrary card.
    """
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part)
        except ValueError:
            return []
        if idx < 0 or idx in out:  # negatives and duplicates are both operator typos
            return []
        out.append(idx)
    return out


def _split_args_for_env(launch_env: Optional[dict]) -> list[str]:
    """Layer-split flags derived FROM the env that will actually launch the server.

    WHY DERIVED AND NOT DECIDED SEPARATELY. The card list and the `-ts` ratio must agree, and
    this module already carries a hard-won rule that a guard which validates a configuration the
    server is not about to run is worse than no guard (see `_generator_cuda_min_free_mb`). If
    this read the env var a second time it could size `-ts` for a split the headroom check had
    already refused. Reading CUDA_VISIBLE_DEVICES makes the argv a function of the one thing
    that determines what the process can see, so the two cannot drift apart.

    `-sm layer` IS llama.cpp's default and is passed anyway, deliberately: it is the difference
    between an argv that documents its own intent and one where a reader has to know the default
    to understand what the run did. `last_launch_argv` is what tests and artifacts assert on.

    `-ts` values are positional over the VISIBLE devices, not physical indices, so `1,1` is
    correct for CUDA_VISIBLE_DEVICES="0,1" and equally for "1,2". It is passed because
    llama.cpp's default splits by AVAILABLE VRAM -- on two cards holding different amounts of
    someone else's work that yields a lopsided split, which is exactly the failure the sweep's
    evenness gate was written to catch.

    Returns [] for a single card (the flags are meaningless with one visible device) and [] when
    `launch_env` is None. None is the Kaggle/scored path, which inherits the ambient environment
    with ALL devices visible -- llama.cpp is already layer-splitting there by default, and this
    function deliberately does not second-guess a placement that has never been measured on L4s.
    """
    if not launch_env:
        return []
    visible = [p for p in (launch_env.get("CUDA_VISIBLE_DEVICES") or "").split(",") if p.strip()]
    if len(visible) < 2:
        return []
    return ["-sm", "layer", "-ts", ",".join("1" for _ in visible)]


class GeneratorCudaRequiredError(RuntimeError):
    """Raised by `_generator_server_and_env()` when `CARNOT_ARC_GENERATOR_REQUIRE_CUDA=1` is set,
    a CUDA card was explicitly requested via `CARNOT_ARC_GENERATOR_CUDA_GPU`, and the guard could
    not place it -- so the caller asked to be told loudly rather than silently degraded to the
    iGPU HIP build.

    WHY THIS EXISTS (2026-08-07). The HIP fallback below is a deliberate, LOUD-but-permissive
    default: for the conductor's own routine ARC work, a slow generator beats no generator, so
    `_generator_server_and_env()` degrades split -> single-card -> iGPU rather than raising. That
    is correct for that caller. It is WRONG for a CUDA-substrate-specific measurement (e.g. a
    think-mode A/B whose entire point is comparing decode behavior on the CUDA build): a silent
    ~2 tok/s HIP substitution there does not degrade the result, it CORRUPTS it -- induce calls
    time out and look like ordinary induction failures, not an infrastructure fallback. This is
    exactly what happened to `experiment_6199_gemma_think_mode_ab.py`'s first run attempt (a
    transient 161 MiB shortfall on gpu1 exhausted the retry window and fell through to HIP with
    no crash, so the corruption was only caught by a human reading the run log).

    This exception is opt-in (`CARNOT_ARC_GENERATOR_REQUIRE_CUDA=1`, default unset) so the
    conductor's own generator resolution -- and every other existing caller -- is byte-identical
    to before. A caller that needs CUDA-or-nothing sets the env var and catches this to emit
    `honest_verdict: blocked_cuda_unavailable` per the Pre-Launch Preconditions Discipline,
    instead of running degraded and reporting a corrupted number.
    """


def _generator_server_and_env(
    ffn_cpu_layers: Optional[int] = None, mtp: Optional[bool] = None
) -> tuple[Path, Optional[dict]]:
    """Resolve the llama-server binary + launch env for the generator, evaluated at LAUNCH time so the
    3090 guard sees current GPU state.

    Priority:
      1. CARNOT_LLAMA_SERVER (Kaggle/live bundled CUDA binary) -- unchanged; inherits ambient env.
      2. OPT-IN CARNOT_ARC_GENERATOR_CUDA_GPU=<idx> -> the local CUDA build pinned to that 3090 via
         CUDA_VISIBLE_DEVICES, but ONLY if the card has >=_generator_cuda_min_free_mb() free (checked
         via _cuda_gpu_has_headroom, which retries briefly to survive a just-crashed process's VRAM
         not yet being reclaimed -- see that function's docstring). This is the operator-approved
         (2026-06-19) use of one idle 3090 for generator throughput now that the TRM run is retired;
         the free-memory guard yields to any conductor job already on the card.
      3. Default: the iGPU HIP build (no conductor contention), else the CUDA build.
    Returns (server_path, env_or_None); env=None means inherit the ambient environment (legacy behavior).

    THE VRAM GUARD IS A **LOCAL-DEV** MECHANISM AND DOES NOT PROTECT THE SCORED RUN. Stated here
    because the surrounding code repeatedly justifies the guard as preventing "a silent LLM-off run
    that still reports itself as the LLM-on scored path", and that framing over-claims its reach.
    Priority 1 returns on `CARNOT_LLAMA_SERVER`, which the Kaggle kernel ALWAYS sets, so on the
    scored path this function never reaches step 2 -- `_generator_cuda_min_free_mb()`, the auto-fit,
    and the fit invariant are all skipped. Everything they guarantee applies to this dev box only.

    What that means concretely: if the scored `machine_shape` ever resolved to a 24GB-class card,
    MTP-on at n_ctx 81920 would need ~25.2 GB and NOTHING in this module would refuse it. The
    server would cudaMalloc-fail, `_ensure_server()` would burn its retry budget, and the agent
    would run LLM-off. The scored-side check for that lives in the kernel's own pre-flight probe
    (`scripts/kaggle/submission_kernel/main.py`), which reads the device's real free VRAM and
    compares it against `_generator_cuda_min_free_mb()` before launching -- that is the only place
    the arithmetic reaches the scored hardware.
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
        idxs = _parse_generator_cuda_gpus(gpu)
        # `ffn_cpu_layers` is threaded through from `_ensure_server()` so the guard budgets for the
        # offload the server will REALLY launch with -- see `_generator_cuda_min_free_mb()`.
        layers = _default_ffn_cpu_layers() if ffn_cpu_layers is None else int(ffn_cpu_layers)
        # ...and `mtp` for the same reason: the draft head is a real, n_ctx-scaled VRAM cost, so a
        # guard that does not know whether this launch loads one is sizing a different server.
        mtp_on = _mtp_default_on() if mtp is None else bool(mtp)
        need_total = _generator_cuda_min_free_mb(layers, mtp_on)
        # MULTI-CARD (layer split). The per-card requirement is the total scaled by the measured
        # aggregate overhead and divided by the card count -- NOT the total.
        #
        # WHAT THIS BUYS, measured 2026-07-31 through this exact code path at the shipped
        # n_ctx 81920 (NOT extrapolated from the sweep harness):
        #     single card, CARNOT_ARC_GENERATOR_CUDA_GPU=1  -> 20.47 tok/s decode, 288 prefill
        #     layer split, CARNOT_ARC_GENERATOR_CUDA_GPU=0,1 -> 38.83 tok/s decode, 908 prefill
        #     i.e. +89.7% decode and +215% prefill.
        #
        # WHY the single-card number is so much worse, and a correction to a wrong first
        # explanation. `need_total` is 25388 MiB here while a 3090's TOTAL is 24576, so the
        # single-card branch below cannot be satisfied outright. It does NOT then fail over to
        # the iGPU -- the auto-fit re-reads free VRAM and spills FFN layers to system RAM until
        # the model fits, landing on ffn_cpu_layers=7. That launch SUCCEEDS, which is precisely
        # why the cost was invisible: nothing errors, the CUDA build is used, and the only
        # symptom is roughly half the throughput.
        #
        # So the split's win is avoiding that forced offload (ffn_cpu 7 -> 0), not rescuing a
        # launch that would otherwise not happen. Both configurations run; one runs degraded.
        #
        # EVERY card must clear, and a partial pass is a refusal, not a smaller split. Silently
        # binding one card when two were asked for would put the operator back on the degraded
        # auto-fit path while believing the split was in effect.
        if len(idxs) > 1:
            per_card = int(need_total * _SPLIT_AGGREGATE_OVERHEAD / len(idxs)) + 1
            missing = [i for i in idxs if not _cuda_gpu_has_headroom(i, per_card)]
            if not missing:
                _note_generator_selection(
                    f"using the CUDA build LAYER-SPLIT across gpus {idxs} "
                    f"(per-card need {per_card} MiB = {need_total} x "
                    f"{_SPLIT_AGGREGATE_OVERHEAD} / {len(idxs)}; ffn_cpu_layers={layers}, "
                    f"n_ctx={_default_induce_n_ctx()}, mtp={mtp_on})."
                )
                return cuda, dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(str(i) for i in idxs))
            _note_generator_selection(
                f"LAYER-SPLIT across {idxs} REFUSED: gpu(s) {missing} lack the {per_card} MiB "
                "per-card requirement (each card's own reason is logged above). Degrading to a "
                "SINGLE-CARD pin on the first requested card that has room."
            )
        # DEGRADE TO ONE CARD, NEVER STRAIGHT TO THE iGPU. Without this, a refused split left
        # `idx = -1` and fell through to the HIP build at ~2 tok/s -- i.e. asking for "0,1" would
        # have been MORE fragile than asking for "1", because any transient contention on either
        # card cost the whole LLM (agent runs LLM-off) instead of dropping to the ~20 tok/s
        # auto-fit path. A degraded single card is worse than a split and enormously better than
        # no generator, so the ordering is split -> single -> iGPU.
        #
        # The narrower split (e.g. 3 cards requested, 2 available) is deliberately NOT attempted:
        # fewer cards means a HIGHER per-card requirement, which is the arithmetic that was just
        # refused, so it would be a guess dressed as a fallback.
        # Single evaluation per card: `_cuda_gpu_has_headroom` RETRIES WITH SLEEPS, so a
        # check-then-recheck would double that latency on every launch and, worse, could return
        # a different answer the second time on a card whose VRAM is in flux.
        idx = next((c for c in idxs if _cuda_gpu_has_headroom(c, need_total)), -1)
        if idx >= 0:
            _note_generator_selection(
                f"using the CUDA build pinned to gpu{idx} "
                f"(ffn_cpu_layers={layers}, n_ctx={_default_induce_n_ctx()}, mtp={mtp_on})."
            )
            return cuda, dict(os.environ, CUDA_VISIBLE_DEVICES=str(idx))
        # guard tripped (card busy / unavailable / bad idx) -> fall through to the iGPU path,
        # never fight the conductor for the 3090.
        #
        # THIS FALL-THROUGH USED TO BE SILENT, and after the 2026-07-28 generator switch it became
        # the default outcome on a 3090 rather than a rare one. The operator ASKED for a specific
        # CUDA card and did not get it; that is exactly the kind of thing that must never be
        # inferred later from a throughput anomaly. `_cuda_gpu_has_headroom` has already logged
        # WHY it refused; this logs what we are doing about it, and how bad that is, because the
        # HIP fallback runs this 31B model at ~2 tok/s and will time out every induce call.
        _note_generator_selection(
            f"CARNOT_ARC_GENERATOR_CUDA_GPU={gpu!r} was requested but the guard refused it "
            "(reason logged above). FALLING BACK to the iGPU HIP build. WARNING: measured at "
            "~2 tok/s decode for gemma-4-31B-it Q4_K_M, which CANNOT complete a "
            "max_tokens=4096 induce call inside CARNOT_ARC_INDUCE_TIMEOUT -- expect "
            "generate() to fail and the agent to run LLM-OFF."
        )
    if gpu and os.environ.get("CARNOT_ARC_GENERATOR_REQUIRE_CUDA") == "1":
        # Opt-in hard stop -- see GeneratorCudaRequiredError's docstring. `gpu` truthy means CUDA
        # was explicitly requested (this is the only branch that can reach here with `gpu` set);
        # default behavior (no CUDA pin requested at all) is completely unaffected.
        raise GeneratorCudaRequiredError(
            f"CARNOT_ARC_GENERATOR_REQUIRE_CUDA=1 and CARNOT_ARC_GENERATOR_CUDA_GPU={gpu!r} were "
            "both set, but no requested CUDA card had headroom (or the CUDA build is missing --  "
            f"cuda.exists()={cuda.exists()}). Refusing the iGPU HIP fallback instead of silently "
            "running a CUDA-substrate measurement on HIP at ~2 tok/s. See the "
            "generator-selection log above for why the CUDA card was refused."
        )
    return (hip if hip.exists() else cuda), None


def _describe_http_failure(exc: BaseException) -> str:
    """Render a completion-request exception INCLUDING the server's own response body.

    WHY (exp5866 finding 4). The old code did `f"...failed: {exc!r}"`, and for a
    urllib HTTPError that repr is just `<HTTPError 500: 'Internal Server Error'>` --
    the generic reason phrase. The ONE informative string in the whole failure was
    thrown away unread:

      * the 500 body says `Context size has been exceeded.` (the concurrency fault)
      * the 400 body says `request (15754 tokens) exceeds the available context size
        (8192 tokens), try increasing it` -- literally the fix, in the message

    Two independent sessions spent effort re-deriving what these bodies already said,
    because nothing ever printed them. This is a RECORD change only: same exception
    handling, same (False, msg) return, same control flow -- just a message that
    contains the evidence. Never raises: a body that cannot be read degrades to the
    plain repr rather than replacing one silent failure with another.
    """
    base = repr(exc)[:200]
    body = ""
    try:  # urllib.error.HTTPError is a file-like object over the response body
        reader = getattr(exc, "read", None)
        if callable(reader):
            raw = reader()
            if isinstance(raw, bytes):
                raw = raw.decode(errors="replace")
            body = str(raw or "")[:400]
    except Exception:
        body = ""
    return f"{base} body={body!r}" if body else base


def _default_induce_n_ctx() -> int:
    """The generator server's SHARED context-pool size, in tokens (llama-server `-c`).

    WHY 81920 AND NOT 16384 (the concurrency fault, measured 2026-07-27, exp5866).
    llama-server with no explicit `--parallel` sets `n_parallel=4` AND `kv_unified=true`
    (its own default, server.cpp:106-110). kv_unified means the 4 slots share ONE pool of
    `-c` cells -- they do NOT each get `-c` cells, and they do NOT get `-c / 4` either
    (that is the DIVIDED-context branch, which only happens when you pass `--parallel`
    explicitly). So the real admission requirement is:

        n_ctx  >=  K_concurrent * (prompt_tokens + max_tokens)

    The eval framework starts ONE THREAD PER GAME with no pool (swarm.py:91), so induce
    requests arrive together; llama.cpp caps concurrency at its own 4 slots and QUEUES the
    rest, which is why K=4 -- not the ~110 game count -- is the number that has to fit.

    At the previous 16384 with max_tokens=4096, the fault fired at K=2 (2 * (5968+4096) =
    20128 > 16384), and it had THREE distinct shapes, all invisible to the concurrency-1
    probing every prior measurement used:
      A. HTTP 500 "Context size has been exceeded." -- server survives (large prompts).
      B. server DEATH: `GGML_ASSERT(logits != nullptr)` -> `ggml_abort` inside
         `update_slots()` -- permanent, every later request gets RemoteDisconnected
         (small prompts admitted, generations collectively overrun the pool).
      C. WORST: HTTP **200** with a silently truncated completion, when the prompt nearly
         fills the pool and only the leftover cells remain for generation.
    Because `generate()` returns `(False, msg)` instead of raising, A and B degrade the
    agent to LLM-OFF while it still reports itself as the LLM-on scored path, and C is not
    even visible as a failure.

    WHAT SIZING THE POOL ACTUALLY REMOVED -- narrowed 2026-07-27 after an adversarial review
    of the shipping commit found this docstring claiming all three, contradicted by that
    commit's own end-to-end evidence.

      A. REMOVED, measured. `n_context_exceeded` went 36 -> 0 across all pre-fix (16384) vs
         all post-fix (81920) cells; the direct back-to-back control at the same prompt in
         the same tree went control 2/2 and 4/4 HTTP 500, fix 2/2 and 4/4 HTTP 200.
      C. REMOVED for the worst measured prompt: every fix request reported
         `predicted_n == 4096 == max_tokens`, i.e. `pool_exhaustion_limit == 0` in every
         cell, where the pre-fix K=1 cell at the same prompt truncated to 2133 chars with
         630 cells of generation room.
      B. **NOT DEMONSTRATED REMOVED.** 6 of the 12 `llm_on_fix_probe__*` cells still carry
         `RemoteDisconnected` server-failure diagnostics at `generator_n_ctx=81920` -- 16
         diagnostics in total, 2 cells ending `generator_healthy_after=False`, and
         `lp85_color04` fully LLM-off at calls=4 / responses=0 / errors=4. The
         requantification records this as `n_remote_disconnected_post_fix: 16` and sets it
         aside as confounded with an external process killer, which may well be right --
         but the discriminating evidence (the server's own `ggml_abort` line in its log)
         was never captured, and `RemoteDisconnected` with the server gone is exactly mode
         B's recorded signature. The 6-cell HTTP gate that reported "fix 0 failures" cannot
         see mode B at all: it fires ONE worst-case prompt shape at K in {2,4}, and mode B's
         trigger is the opposite shape (many SMALL prompts, individually admitted, whose
         GENERATIONS collectively overrun the pool over a long horizon).
    So: treat A and C as fixed, and B as open. Before claiming B is fixed, run a
    mode-B-specific arm (many small concurrent prompts, long horizon, external killers
    excluded) and capture the server's stderr so `ggml_abort` can be told apart from SIGTERM.

    81920 = 4 * (15734 + 4096) rounded up to a 4096 multiple, where 15734 tokens is the
    real `induce_prompt()` for the largest logical grid in `ops/arc_solve_registry.yaml`
    (64x64), measured through the server's own `/tokenize` -- not estimated. Worst case,
    not typical, because the GENERATED length is unknowable in advance.

    COMPUTED, NOT HARDCODED (2026-07-27 review). The first version of this function returned
    a literal `81920` while `max_tokens` was independently read from
    `CARNOT_ARC_INDUCE_MAX_TOKENS` at BOTH construction sites in arc_competition_agent.py
    (:889 and :5014). So the two halves of the admission inequality could diverge exactly the
    way this docstring warns the construction sites once did: `CARNOT_ARC_INDUCE_MAX_TOKENS
    =8192` needs 4*(15734+8192) = 95704 cells and would have silently re-broken K=4 against
    an unchanged 81920. Both halves now come from the same arithmetic, so an operator raising
    the completion budget raises the pool with it. At the default 4096 this returns exactly
    the 81920 that was measured and shipped -- the change is a derivation, not a re-sizing.

    HOW LITTLE SLACK THERE IS. 81920/4 - 4096 = 16384 tokens is the largest prompt this pool
    admits at K=4, versus the 15734-token worst case it is sized for: 650 tokens of margin per
    slot. That is not much, and it is why the pre-flight probe must use a prompt of the SAME
    measured worst-case size rather than an eyeballed synthetic one -- the kernel's original
    synthetic probe string measured 17238 tokens through the model's own tokenizer, i.e.
    854 tokens OVER what the pool admits, and at K=4 it returns 4/4 HTTP 500 "Context size
    has been exceeded" (measured directly, 2026-07-27, RTX 3090, mtp-off, per-slot n_tokens
    20469..20493 at release == 81920/4 exactly). It passed the shipped probe only because
    that probe ran K=2.

    WHY THIS AXIS AND NOT ANOTHER. Measured VRAM envelope (9 configs, refit max error
    0.19%): `MiB = 10547 + 0.02519*n_ctx + 206.8*slots`. Context is the CHEAP axis --
    16384 -> 81920 costs +1668 MiB, while a slot costs ~207 MiB regardless of n_ctx. The
    alternatives were measured and rejected: an explicit `--parallel 4` DIVIDES the pool
    (4096/slot) and is strictly worse; `--parallel 1` passes an HTTP gate and costs LESS
    VRAM but generated 648/650/184/648 tokens against a 4096 budget -- i.e. it converts
    the loud 500 into silent mode C, the exact defect under investigation. `n_ctx_train`
    is 262144, so 81920 is well inside the model's trained context.

    OVERRIDE with CARNOT_ARC_INDUCE_N_CTX for a tight-VRAM box or a model with a fatter
    per-token KV than the frozen 9B live generator (this default is sized for that model;
    a ~3x-larger model's KV would cost ~3x the 1668 MiB).

    THAT WARNING CAME TRUE ON 2026-07-28 -- recorded rather than deleted, because it is the
    rare case of a docstring correctly predicting its own obsolescence. The generator was
    re-pinned to gemma-4-31B-it, whose measured per-cell KV is 0.0503 MiB against the 9B's
    0.02519, i.e. almost exactly 2x. The POOL SIZE did not need to change (81920 is a token
    count, and the two tokenizers are within 0.2% on the worst-case induce prompt -- measured
    paired, 17930 vs 17893) but the VRAM ARITHMETIC did: see `_generator_cuda_min_free_mb()`
    and the gemma envelope constants. On a 24 GB card this default now resides at 23888 MiB,
    so `CARNOT_ARC_INDUCE_N_CTX` and the new `CARNOT_ARC_FFN_CPU_LAYERS` are the two levers
    that make the local 3090 viable at all. Read via default_factory so the
    literal lives in exactly ONE place -- both construction sites in
    arc_competition_agent.py (`_proposer()` and `_load_sge_candidate_router()`) omit n_ctx
    and therefore cannot silently diverge from each other, which is the failure the
    REQ-ARC-FCP-5699-35 comment at that second site records having already happened once
    for max_tokens.

    CORRECTION 2026-08-08 (REQ-ARC-WMTE-6227). The "81920 = 4*(15734+4096)" arithmetic above is
    now historical: `_INDUCE_WORST_CASE_PROMPT_TOKENS` moved 15767 -> 22352 (see that constant's
    own comment for the re-measurement under current defaults -- k=all transitions, object table
    on). This function's arithmetic is unchanged and picks the new figure up automatically
    (`need = 4*(22352+4096) = 105792`, rounded up to `n_ctx = 106496`), so no code here needed to
    change -- only the worst-case-prompt constant it reads. Recorded so a reader doing the old
    arithmetic by hand does not conclude this function is wrong; it is the constant's docstring
    that moved.
    """
    import os

    override = os.environ.get("CARNOT_ARC_INDUCE_N_CTX")
    if override:
        return int(override)
    max_tokens = int(
        os.environ.get("CARNOT_ARC_INDUCE_MAX_TOKENS", str(_INDUCE_DEFAULT_MAX_TOKENS))
    )
    need = _llama_server_slots() * (_INDUCE_WORST_CASE_PROMPT_TOKENS + max_tokens)
    # Round UP to a 4096 multiple: llama.cpp allocates in blocks and a round pool is easier to
    # reason about against the published VRAM envelope, whose n_ctx samples are all multiples.
    return int(-(-need // 4096) * 4096)


def _default_ffn_cpu_layers(mtp: Optional[bool] = None) -> int:
    """How many transformer blocks' FFN weights to keep in SYSTEM RAM instead of VRAM.

    `mtp` MUST be the value the server will actually launch with whenever the caller knows it.
    It defaults to `_mtp_default_on()` (the env expression) so a bare module-level call still sizes
    something sensible, but a `LocalGGUFProposer` passes ITS OWN `self.mtp` via `__post_init__` --
    and that is the case this parameter exists for. See that method for the failure it removes.

    OPT-IN, DEFAULT 0 -- unset means byte-identical launch args to before this knob existed.
    Set `CARNOT_ARC_FFN_CPU_LAYERS=<n>` to free VRAM on a card that cannot hold the whole model
    plus its KV pool (operator directive 2026-07-28: "when running on eGPU locally with llama.cpp
    we can offload the FFN weights to system RAM to free up VRAM").

    WHY `-ot` AND NOT `-cmoe`/`-ncmoe`. llama-server ships `--cpu-moe` / `--n-cpu-moe`, and they
    are the obvious-looking answer, but they match MoE expert tensors (`ffn_*_exps`) only.
    gemma-4-31B-it is DENSE: its GGUF contains `blk.<i>.ffn_{gate,up,down}.weight` and NO
    `ffn_*_exps` tensor at all (read directly from the GGUF tensor table, 2026-07-28). So both MoE
    flags are accepted and do NOTHING on this model -- a silent no-op, which is worse than no
    flag. The dense lever is `--override-tensor` with a regex over the real tensor names.

    MEASURED COST, NOT ASSUMED (RTX 3090, gemma-4-31B-it Q4_K_M, n_ctx 32768, q8_0 KV, fixed
    238-token prompt / 256-token completion):

        CPU FFN layers |   VRAM  |  freed  | decode tok/s | prefill tok/s
                     0 |  21416  |    --   |    36.14     |    826.8
                    12 |  19072  |  -2344  |    15.17     |    177.0
                    24 |  16728  |  -4688  |     9.81     |    101.4
                    40 |  13580  |  -7836  |     6.33     |     64.0

    This is a REAL TRADEOFF, not a free win, and the throughput column is the half that matters
    most for this workload: the first 12 layers cost 58% of decode speed to buy 11% of VRAM, and
    PREFILL -- which is what the induce path is actually bound by, at a 15767-token worst-case
    prompt -- degrades 4.7x at 12 layers and 13x at 40. That worst-case prompt costs ~19 s to
    prefill at full offload and ~246 s at 40 CPU layers, against this proposer's 600 s timeout
    with 4 concurrent slots. Treat anything past ~12 layers as likely to push real induction into
    timeout, and prefer lowering `CARNOT_ARC_INDUCE_N_CTX` first if the goal is purely to fit.

    `--no-mmap` does NOT recover the loss (measured 9.69 vs 9.81 tok/s at 24 layers, identical
    residency), despite llama.cpp printing a hint about mmap when `-ot` is used.

    AUTO-FIT WHEN THE OPERATOR OPTED INTO A LOCAL CUDA CARD (added 2026-07-28, second pass).
    "Default 0" was correct as a statement about the KNOB and wrong as a shipped configuration,
    because the generator switch moved the requirement past what a 3090 has:

        gemma-4-31B @ n_ctx 81920, 0 CPU-FFN layers -> 23888 MiB + 1500 margin = 25388 required
        an RTX 3090 has 24576 MiB TOTAL

    so with `CARNOT_ARC_GENERATOR_CUDA_GPU=0` set -- which the conductor's standing systemd
    drop-in DOES set -- the guard declined the card unconditionally and the generator fell back to
    the iGPU HIP build. That fallback was never measured before it became the default; measured
    here (same 238-token prompt / 256-token completion as the table above) it runs this model at
    ~2 tok/s decode against a 600 s induce timeout, i.e. every induce call times out, `generate()`
    returns `(False, msg)`, and the agent runs LLM-OFF while still reporting itself LLM-on. A
    working local path became a non-working one with no error anywhere.

    So when (a) the env var is UNSET, and (b) `CARNOT_ARC_GENERATOR_CUDA_GPU` names a real CUDA
    card, this returns the FEWEST layers that make the configured server fit that card, and says
    so on `GENERATOR_SELECTION_LOG`. Rationale for auto-fitting rather than hard-failing: the
    operator directive that introduced this knob asked for the local eGPU to be usable, and the
    smallest offload that fits is strictly better than both alternatives (a 2 tok/s iGPU, or a
    refusal). It is capped at `_FFN_CPU_AUTOFIT_MAX_LAYERS` so auto-fit can never silently trade a
    slow fallback for an equally slow CUDA path -- past the cap it returns 0 and lets the guard
    decline the card loudly.

    KAGGLE IS UNAFFECTED, by construction: the scored kernel sets `CARNOT_LLAMA_SERVER`, which
    `_generator_server_and_env()` honours at priority 1 and never reaches the CUDA guard, and it
    does not set `CARNOT_ARC_GENERATOR_CUDA_GPU`. Both conditions must hold for auto-fit to
    engage, so the 96 GB submission path launches with byte-identical argv to before.
    """
    import os

    raw = (os.environ.get("CARNOT_ARC_FFN_CPU_LAYERS") or "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            # A typo must not silently disable the knob the operator thinks they set, nor crash
            # the live path. We still return 0 -- there is no safe way to guess what they meant --
            # but the DOCSTRING USED TO CLAIM `_ensure_server` recorded the bad value, and it did
            # not: `raw` was discarded inside this function before anything could see it. So the
            # promise is kept here instead, on the audit channel that is mirrored to stderr and
            # copied onto the proposer. Naming the rejected string is the whole point; "invalid
            # value" without the value sends the operator looking in the wrong shell.
            _note_generator_selection(
                f"CARNOT_ARC_FFN_CPU_LAYERS={raw!r} is not an integer -- IGNORED, the FFN offload "
                "is DISABLED for this launch (0 layers). Set an integer to enable it."
            )
            return 0

    gpu = (os.environ.get("CARNOT_ARC_GENERATOR_CUDA_GPU") or "").strip()
    if not gpu:
        return 0  # no local-CUDA opt-in -> byte-identical legacy argv, no -ot at all
    try:
        idx = int(gpu)
    except ValueError:
        return 0
    if idx < 0:
        return 0
    free = _cuda_gpu_free_mb(idx)
    if free < 0:
        return 0  # no nvidia-smi / no such card: not our problem to guess, the guard will refuse
    n_ctx = _default_induce_n_ctx()
    # Budget for the MTP draft head if this launch will load one: at n_ctx 81920 the head adds
    # 1290 MiB, which is ~7 additional offloaded layers -- an auto-fit blind to it picks a layer
    # count the guard then rejects, and the card falls through to the ~2 tok/s iGPU for the run's
    # whole lifetime.
    #
    # PREFER THE CALLER'S VALUE OVER THE ENV DEFAULT. `_mtp_default_on()` answers "what would a
    # proposer that was given no `mtp=` argument do", which is the WRONG question when the proposer
    # was in fact given one. Dozens of harnesses construct `LocalGGUFProposer(mtp=...)` explicitly,
    # so sizing the offload from the env default while the server launches with the instance value
    # is a guaranteed mismatch in whichever direction they differ.
    mtp_on = _mtp_default_on() if mtp is None else bool(mtp)
    if _predicted_generator_vram_mib(n_ctx, 0, mtp_on) + _GENERATOR_CUDA_GUARD_MARGIN_MIB <= free:
        return 0  # full offload already fits -- never pay the throughput cost for nothing
    needed = _ffn_cpu_layers_to_fit(free, n_ctx, mtp_on)
    if needed < 0:
        _note_generator_selection(
            f"CUDA gpu{idx} has {free} MiB free; even {_FFN_CPU_AUTOFIT_MAX_LAYERS} CPU-FFN layers "
            f"cannot fit the generator at n_ctx={n_ctx} (mtp={mtp_on}; would still need "
            f"{_predicted_generator_vram_mib(n_ctx, _FFN_CPU_AUTOFIT_MAX_LAYERS, mtp_on) + _GENERATOR_CUDA_GUARD_MARGIN_MIB:.0f} MiB). "
            "NOT auto-offloading; the guard will decline this card and the generator will fall "
            "back to the iGPU HIP build, which runs gemma-4-31B at ~2 tok/s and WILL time out. "
            "Free the card, lower CARNOT_ARC_INDUCE_N_CTX, or set CARNOT_ARC_MTP=0 (the draft "
            f"head alone costs {_VRAM_MTP_HEAD_INTERCEPT_MIB + _VRAM_MTP_HEAD_PER_CTX_MIB * n_ctx:.0f} MiB here)."
        )
        return 0
    _note_generator_selection(
        f"CUDA gpu{idx} has {free} MiB free, generator needs "
        f"{_predicted_generator_vram_mib(n_ctx, 0, mtp_on) + _GENERATOR_CUDA_GUARD_MARGIN_MIB:.0f} MiB at "
        f"n_ctx={n_ctx} (mtp={mtp_on}) with no offload -- AUTO-SELECTING "
        f"CARNOT_ARC_FFN_CPU_LAYERS={needed} "
        f"(predicted {_predicted_generator_vram_mib(n_ctx, needed, mtp_on):.0f} MiB resident). This costs "
        "throughput -- see _default_ffn_cpu_layers()'s measured table -- and is taken because the "
        "alternative is the iGPU fallback at ~2 tok/s, which cannot meet the induce timeout."
    )
    return needed


# Measured decode throughput (tok/s) vs CPU-FFN layers, RTX 3090 / gemma-4-31B-it Q4_K_M / n_ctx
# 32768 / q8_0 KV / fixed 238-token prompt / 256-token completion, per-PID residency joined
# PID -> GPU UUID -> index (card 1 confirmed for every arm).
#
# PROVENANCE, and why these are the RE-MEASURED numbers rather than the first sweep's. The first
# sweep's results file was overwritten by a later n_ctx-81920 run (the script hardcoded its output
# path), leaving its VRAM column alive only as prose in a docstring -- a traceability gap, not a
# fabrication. It was re-measured 2026-07-28 to a distinct file
# (`ffn_offload_results_ctx32768_reemit.json`), which independently reproduced the VRAM column
# EXACTLY -- 21416 / 19072 / 16728 / 13580 MiB at 0/12/24/40 layers -- confirming the prose.
#
# Decode reproduced to within 3% at 0/12/24 layers (36.07 vs 36.14, 15.05 vs 15.17, 9.54 vs 9.81)
# but the 40-layer arm came in 22% slower (4.91 vs 6.33). That arm is the most CPU-bound of the
# four by construction, and the re-measurement ran while a concurrent session held the other card
# and its CPU cores, so the discrepancy is best explained by host contention rather than by either
# reading being wrong. The SLOWER re-measured values are the ones used here, deliberately: they
# come from the file that still exists, and for a TIMEOUT a conservative (slower) rate is the safe
# direction -- it lengthens the budget rather than shortening it.
_FFN_DECODE_TOK_S_BY_LAYERS = ((0, 36.07), (12, 15.05), (24, 9.54), (40, 4.91))
# The slowest induce call observed in the 2026-07-28 gemma-4-31B head-to-head (39 runs, 13 games x
# 3 replicates): mean 383.9 s, median 366.5 s, max 572.0 s. Measured on CUDA, SINGLE-STREAM, at
# n_ctx 32768 and ZERO FFN offload -- i.e. the fastest configuration we ship, and it still landed
# within 4.7% of the old 600 s timeout. Nothing about that margin was re-examined when the
# generator changed from a 9B to a 31B; the 600 s literal was calibrated for the 9B.
_INDUCE_OBSERVED_MAX_WALL_S = 572.0


def _default_induce_timeout_s() -> int:
    """Per-call induce timeout (s), DERIVED from measured throughput rather than a fixed literal.

    THE PROBLEM WITH THE OLD 600 s LITERAL, in two parts:

      1. It was calibrated for Qwen3.5-9B and never revisited for a model 3.4x its size. The
         head-to-head's own numbers show the risk: the slowest of 39 real induce calls took
         572.0 s, 4.7% inside the limit, in the FASTEST configuration we run (single-stream, no
         offload, n_ctx 32768). The live path runs n_ctx 81920 with 4 `kv_unified` slots, where
         per-request throughput is lower, so the true headroom is smaller than 4.7% and is not
         measured.
      2. The 2026-07-28 FFN auto-fit makes it strictly worse. Offloading FFN blocks to system RAM
         is what lets a 24 GB card host this generator at all, and it costs 2.4x decode throughput
         at 12 layers. A timeout that does not account for the offload turns the fix for one
         silent-LLM-off failure into a cause of another.

    So the timeout scales with the SAME offload setting that slows the generation down. Floored at
    the historical 600 s so this can never SHORTEN an existing deployment's budget, and the env
    override is untouched.

    FLOOR RAISED 600 -> ARC_LIVE_GENERATOR_INDUCE_TIMEOUT_FLOOR_S (2400) on 2026-08-21
    (REQ-ARC-WMTE-6620). Same drift as max_tokens: 600 was calibrated for the 9B; Qwen3.8's
    median induction is 62,490 tokens, ~1,730 s at the local 3090's measured 36 tok/s, so the
    old floor cut off the median draw less than halfway through. 2400 is the scored kernel's
    own pin for this generator and covers the pool-clamped worst case (84,144 tokens, ~2,340 s).
    Raising a floor can only LENGTHEN a budget, so the no-shortening contract above holds.

    KAGGLE IS UNAFFECTED: the scored kernel runs with zero FFN offload (a 96 GB card needs none),
    the slowdown factor is 1.0, and this returns the floor -- the same 600 s it always used.

    The counter-argument to simply raising this -- "a longer timeout means a hung call blocks
    longer" -- is real but much weaker than it looks: a timeout that fires does NOT surface an
    error, it returns `(False, msg)` and the agent proceeds LLM-OFF while still reporting itself
    as the LLM-on path. Waiting longer for a real answer beats promptly recording a fake one.
    """
    import os

    override = os.environ.get("CARNOT_ARC_INDUCE_TIMEOUT")
    if override:
        # Malformed env falls through to the derived default rather than crash -- this
        # resolver is now a dataclass default_factory (REQ-ARC-WMTE-6620), so a bare int()
        # would turn a typo'd env var into a constructor crash on every proposer build.
        # Same contract as _induce_max_tokens_default.
        try:
            return int(override)
        except (TypeError, ValueError):
            pass
    layers = _default_ffn_cpu_layers()
    # Piecewise-linear interpolation of the measured decode curve. Linear rather than fitted
    # because four points do not justify a model, and because interpolating BETWEEN measurements
    # is defensible where extrapolating beyond them is not -- past the last anchor we hold the
    # final (slowest) rate rather than extending the trend into numbers nobody measured.
    rate = _FFN_DECODE_TOK_S_BY_LAYERS[-1][1]
    if layers <= 0:
        rate = _FFN_DECODE_TOK_S_BY_LAYERS[0][1]
    else:
        for (lo_n, lo_r), (hi_n, hi_r) in zip(
            _FFN_DECODE_TOK_S_BY_LAYERS, _FFN_DECODE_TOK_S_BY_LAYERS[1:]
        ):
            if lo_n <= layers <= hi_n:
                frac = (layers - lo_n) / float(hi_n - lo_n)
                rate = lo_r + frac * (hi_r - lo_r)
                break
    slowdown = _FFN_DECODE_TOK_S_BY_LAYERS[0][1] / max(rate, 1e-6)
    return int(
        max(
            float(ARC_LIVE_GENERATOR_INDUCE_TIMEOUT_FLOOR_S), _INDUCE_OBSERVED_MAX_WALL_S * slowdown
        )
    )


def _ffn_cpu_override_regex(n_cpu_layers: int) -> str:
    """The `-ot/--override-tensor` pattern that keeps the FFN weights of the FIRST `n_cpu_layers`
    transformer blocks on the CPU.

    Tensor names are REAL, read out of the gemma-4-31B-it Q4_K_M GGUF tensor table rather than
    guessed: `blk.<i>.ffn_gate.weight`, `blk.<i>.ffn_up.weight`, `blk.<i>.ffn_down.weight`
    (60 blocks, three FFN tensors each). The alternation is written out explicitly per index
    instead of as a numeric range, because llama.cpp's override matcher is a plain regex with no
    numeric-range support and `blk\\.[0-9]+\\.` would match EVERY block, silently offloading the
    whole model when the caller asked for twelve layers.

    Returns "" for n_cpu_layers <= 0 so the caller can append nothing at all -- the default path
    must not gain an argument.
    """
    if n_cpu_layers <= 0:
        return ""
    idx = "|".join(str(i) for i in range(int(n_cpu_layers)))
    return rf"blk\.({idx})\.ffn_(gate|up|down)\.weight=CPU"


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

    # THE DATACLASS DEFAULT IS NOW THE LIVE PIN (2026-07-28, second pass). It used to be
    # `"gemma-4-12B-it"`, on the stated ground that the ~25 historical `experiment_4xxx/5xxx_*`
    # modules constructing a bare proposer were measured against the 12B and re-pointing them
    # "would rewrite what those experiments mean". That argument does not survive contact with the
    # operator directive, and it was protecting the wrong thing:
    #
    #   * The 12B is a THIRD model -- neither the retired Qwen nor the directed gemma-4-31B. A
    #     default nobody chose is not a preserved measurement, it is an unowned configuration.
    #   * Historical artifacts already in `results/` are not affected by a default in live code;
    #     they record what they ran. Never-prune protects the RECORD, not the default that
    #     produced it. RE-RUNNING one of those modules today under a 12B default would in fact be
    #     the misleading outcome -- a "current" measurement of a model the project no longer runs.
    #   * The failure mode is silent in the direction that matters: a bare proposer that quietly
    #     loads a different model produces plausible-looking induction numbers attributed to the
    #     live generator.
    #
    # Any module that genuinely needs the 12B can still say so explicitly; that is a default, not
    # a hardcode. Pinned by tests/python/test_arc_generator_migration_defects.py.
    repo_substr: str = ARC_LIVE_GENERATOR_REPO_SUBSTR
    # SHARED context pool (llama-server -c). 81920 by measurement, env-overridable --
    # see _default_induce_n_ctx() above for the full derivation and the rejected alternatives.
    n_ctx: int = field(default_factory=_default_induce_n_ctx)
    # WAS the literals 4096 / 300 -- 9B-era values that survived two generator swaps and made
    # every env-less local run fail 100% of Qwen3.8 inductions (REQ-ARC-WMTE-6620, 2026-08-21).
    # Both resolvers read the env override first, then the generator pin's defaults, exactly
    # like `n_ctx` above -- construction sites omit these arguments so they cannot drift.
    max_tokens: int = field(default_factory=_induce_max_tokens_default)
    timeout: int = field(default_factory=_default_induce_timeout_s)
    port: int = 8919
    offline_legal: bool = True
    # Live-submission deploy config (all OPT-IN; defaults preserve legacy behavior). Validated 2026-06-19:
    # Qwen3.5-9B-MTP is the selected ARC live generator (62.5% Layer-B grounding vs DeepSeek-Flash 25%,
    # ~13 tok/s with MTP, 5.9GB Q4 fits 16GB). See docs/research-notes/arc-16gb-model-alternatives-2026-06-18.md.
    # `--spec-type draft-mtp` + `--model-draft <the SEPARATE head GGUF>`. Reads the same env
    # expression the live construction sites read, so a bare proposer and a live one agree.
    mtp: bool = field(default_factory=_mtp_default_on)
    # EXPLICIT path to the MTP draft head. None -> `_resolve_mtp_head()` finds it (env var, HF
    # cache, operator staging dir). The Kaggle kernel sets CARNOT_ARC_MTP_GGUF_PATH from the
    # attached dataset mount rather than passing this.
    mtp_model_path: Optional[str] = None
    # WAS `None` (= f16), WHICH CANNOT LOAD THIS MODEL AT ANY USEFUL CONTEXT. Measured 2026-07-28
    # on an RTX 3090: gemma-4-31B-it Q4_K_M at n_ctx 32768 with f16 KV OOMs at 23902 MiB, and at
    # the shipped n_ctx 81920 it SIGSEGVs. q8_0 is what makes the model fit at all (23906 of
    # 24123 MiB free), and every live construction site already passed it explicitly -- so the
    # `None` default was reachable ONLY by a caller who omitted the argument, and handed exactly
    # that caller an unloadable configuration. The default now matches the only value the project
    # ships. Near-lossless; halves the KV cache.
    kv_quant: Optional[str] = "q8_0"  # --cache-type-k/v q8_0
    # -ngl: how many transformer layers' weights live on the GPU. 999 = all on GPU (fast, default).
    # Operator prefill-to-RAM lever (2026-06-21): on the shared 16GB eval GPU the LLM (5.9GB MTP-off)
    # coexists with the live per-game CNN dynamics fit (measured 1.45GB peak) + the q8 KV-cache. Full
    # offload already fits (~9.4GB of 16GB), so 999 stays the default. But if a heavier config (deeper
    # search, larger CNN, bigger ctx) pushes VRAM, set CARNOT_ARC_NGL below the layer count: the
    # un-offloaded layers stay PREFILLED in system RAM (llama.cpp mmaps the GGUF -> host page cache;
    # 125GB RAM trivially holds the 5.9GB Q4 weights) and compute on CPU, freeing VRAM for KV + CNN
    # training. The wall-clock cost is acceptable because the ARC eval has NO time limit (only the 12h
    # Kaggle-notebook cap + the 600 RPM real-env rate limit, neither of which gates internal generation).
    n_gpu_layers: int = 999
    # OPT-IN dense-FFN offload to system RAM (`-ot`). 0 = byte-identical argv to before the knob
    # existed. See `_default_ffn_cpu_layers()` for the measured VRAM/throughput tradeoff table and
    # for why `-cmoe`/`-ncmoe` are the WRONG lever on this dense model (they are silent no-ops).
    #
    # THE DEFAULT IS A SENTINEL, NOT A `default_factory`, AND THE DIFFERENCE IS LOAD-BEARING. A
    # dataclass evaluates default factories with no access to sibling fields, so a factory here
    # sized the offload against the ENVIRONMENT's `mtp` answer while the server launches with THIS
    # instance's `mtp`. `_FFN_CPU_LAYERS_AUTO` defers the decision to `__post_init__`, which does
    # have `self.mtp`. An explicitly-passed value is left exactly as given.
    ffn_cpu_layers: int = _FFN_CPU_LAYERS_AUTO
    no_think_prefix: str = ""  # e.g. "/no_think\n" -> suppress hybrid-thinking CoT (Qwen3)
    # REQ-ARC-WMTE-5725: OPT-IN. When True, generate()/complete_text() POST to the OpenAI-compatible
    # /v1/chat/completions endpoint (a single user turn) instead of the raw /completion endpoint. The
    # server then applies the GGUF's OWN embedded chat template (turn delimiters, e.g. Qwen3.6's
    # <|im_start|>assistant), which Qwen3.6-family models (ThinkingCap-27B) REQUIRE to know a turn has
    # started -- the raw /completion path (no template) made those models emit an immediate EOS with ~0
    # output on ~10/12 genuine-reasoning induce cells (REQ-ARC-WMTE-5724 measurement-validity caveat).
    # Default False keeps the FROZEN live-generator path (Qwen3.5-9B raw /completion) byte-identical.
    # The response is normalized back into llama.cpp's {content, stop_type, truncated} shape so
    # _record_completion_diagnostics + every caller works unchanged; a split-out reasoning_content (some
    # builds extract <think> into its own field) is folded back into `content` wrapped in <think> tags
    # so reason_engaged detection + max_raw_completion_len stay faithful to what the model generated.
    use_chat_template: bool = False
    model_path: Optional[str] = (
        None  # explicit .gguf path; on Kaggle set to the bundled /kaggle/input/... path
    )
    tries: int = 3
    extra_server_args: tuple = ()  # e.g. ("-fit", "off") -- raw args appended to the launch
    # command verbatim. Added for exp5705 after llama-server's default -fit heuristic hard-hung
    # (confirmed via /proc/PID/io: zero read progress for 12+ minutes) loading a large hybrid
    # linear/full-attention model (Qwen3.6-27B) on this project's HIP/ROCm build -- -fit off has
    # no downside when n_gpu_layers and n_ctx are both already explicit (nothing left to auto-fit).
    _proc: Any = None
    # MANDATORY truncation detection (operator directive, REQ-ARC-FCP-5699-27): every completion
    # request (generate() AND complete_text()) sets these from llama.cpp's own response, so ANY
    # caller can check them after a call regardless of whether that call's own return contract
    # treats truncation as a failure. last_stop_type == "limit" means the response was cut off by
    # hitting n_predict (self.max_tokens) before finishing naturally ("eos"); last_prompt_truncated
    # means the INPUT prompt itself exceeded the server's context window (n_ctx) and was cut --
    # a different, upstream failure mode. Both were previously silently discarded (only the
    # "content" field was read from the response).
    last_stop_type: str = ""
    last_prompt_truncated: bool = False
    # The EXACT argv `_ensure_server()` last handed to subprocess.Popen, and the `-ot` value it
    # derived. Recorded because "the flag reaches the server" is otherwise unfalsifiable from
    # outside the process: with stderr going to a log file and the launch happening inside a
    # private method, a silently-dropped argument looks identical to a working one. Empty until
    # the first launch. Pinned by tests/python/test_arc_ffn_cpu_offload.py.
    last_launch_argv: tuple = ()
    # The KV cache type THIS launch actually used, or None when the flags were dropped. Recorded
    # for the same reason as `last_launch_argv`: `CARNOT_ARC_KV_QUANT` can override the field, so
    # the field alone no longer tells an artifact what the server ran with.
    last_kv_quant_used: Optional[str] = None
    last_ffn_cpu_override: str = ""
    # The `-sm layer -ts 1,1...` flags THIS launch used, or () for a single card / the scored
    # path. Same rationale as `last_ffn_cpu_override`: without it, "did this run split?" is only
    # answerable by re-deriving it from the env, which is precisely the second read that
    # `_split_args_for_env` exists to avoid.
    last_split_args: tuple = ()
    # WHAT `--model-draft` ACTUALLY RECEIVED, and -- when MTP was asked for and not delivered --
    # why. These exist because a misconfigured MTP is invisible: llama.cpp accepts a draft it
    # cannot use, warns, and serves normally with speculation silently disabled. "The server is
    # healthy" is therefore NOT evidence that MTP is engaged, so the launch decision has to be
    # recorded at the point it is made rather than inferred later from throughput.
    last_mtp_draft_path: str = ""
    mtp_disabled_reason: str = ""
    # DID SPECULATION ACTUALLY ENGAGE, per the RUNTIME'S OWN STDERR -- as opposed to
    # `last_mtp_draft_path`, which only records what we PASSED. Three-valued:
    #   True  -> the positive `adding speculative implementation 'draft-mtp'` marker was found
    #   False -> the server is healthy and the marker is ABSENT, i.e. silently disabled
    #   None  -> not requested, or stderr was not captured so it cannot be determined
    # See `_verify_mtp_engaged()` for why "healthy server" is not evidence and None is not False.
    last_mtp_engaged: Optional[bool] = None
    mtp_engaged_evidence: str = ""
    # WHERE THIS GENERATOR ACTUALLY LANDED, and why. Populated by `_ensure_server()` from the
    # module-level `GENERATOR_SELECTION_LOG`. Before this existed, "the CUDA guard refused the card
    # and we silently fell back to a ~2 tok/s iGPU" was a control-flow branch with no trace: the
    # only symptom was an induce timeout much later, which reads as a model problem rather than a
    # placement problem. An artifact carrying these two fields can distinguish them after the fact.
    generator_selection_log: list = field(default_factory=list)
    generator_server_path: str = ""
    # REQ-ARC-WMTE-5717: DEV-ONLY. When True (set by the agent ONLY on the stall/first-contact
    # re-induction path) AND CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED=1, induce() prepends the
    # game-agnostic exploration-playbook exemplars. Default False -> byte-identical induce prompt.
    include_playbook_exemplars: bool | str = False
    # REQ-ARC-FCP-5699-30: the raw completion text, captured on EVERY call regardless of
    # success/failure -- generate()'s failure path previously discarded `text` entirely once it
    # decided the required functions were missing, so there was no way to see WHAT the model
    # actually produced (reasoning-only? malformed code? nothing?) on a failed try, only THAT it
    # failed. Closes the diagnostic gap REQ-ARC-FCP-5699-23 through -29 all ran into without ever
    # inspecting.
    last_raw_completion: str = ""
    # THE TWO CHANNELS, RECORDED SEPARATELY (2026-08-05). `last_raw_completion` holds the FOLDED
    # text (`<think>reasoning</think>final`), which is faithful to everything the model emitted but
    # cannot answer the one question that matters when an induce fails on the chat endpoint: WHICH
    # CHANNEL WAS EMPTY. A completion that ends at `</think>` with 6603 generated tokens looks, in
    # the folded view, exactly like a model that reasoned at length and wrote no code -- and looks
    # exactly the same as a model that wrote perfectly good code into a channel the extractor never
    # read. Those two have opposite fixes (better prompt vs. read the other channel), and exp6091
    # spent a whole 19-cell run unable to tell them apart. These two fields are PURE OBSERVATION:
    # nothing branches on them, so recording them changes no behaviour on any path.
    last_final_content: str = ""
    last_reasoning_content: str = ""
    # The n_predict the most recent generate() call actually SENT after the pool clamp
    # (REQ-ARC-WMTE-6620); -1 until a call runs. _limit_diagnostic reads this so its message
    # names the budget the server was really asked for, not the configured cap.
    last_requested_n_predict: int = -1
    # LIVENESS WITNESS (2026-07-27, exp5866 finding 4). The scored ARC path had NO channel
    # at all for "did the generator actually answer": generate()/complete_text() return
    # (False, msg) on a dead or refusing server, every caller treats that as "no induction
    # this stall" and continues, and the message string is discarded by 4 of the 11 call
    # sites outright. So a run whose generator died at action 3 completed all 400 actions,
    # exited 0, and was recorded as an LLM-on measurement. The census
    # (results/outer_loop_arc_generator_failure_swallow_census_20260727.json) found the
    # harness-side `errors` counter is STRUCTURALLY dead -- 877 stat blocks, zero non-zero,
    # including all 8 cells where the generator provably died -- because it only counts
    # exceptions that PROPAGATE, and none do.
    #
    # These counters live on the PROPOSER because it is the single choke point all 11 call
    # sites funnel through; instrumenting the call sites individually would have to be
    # redone for every new caller and would miss exactly the ones that discard the message.
    # SERVER failures (unreachable / HTTP error / transport death) are counted separately
    # from CONTENT failures (the server answered, the answer was unusable) because only the
    # first is a liveness fact -- conflating them would let a healthy-but-unhelpful model
    # read as a dead generator and vice versa.
    n_completion_calls: int = 0
    n_completion_ok: int = 0
    n_server_failures: int = 0
    n_content_failures: int = 0
    # Times a candidate that PASSED parse/required/validate was rejected as mechanically
    # defective and re-asked (2026-07-31). Counted separately from the three above because it is
    # neither a liveness fact nor a content failure: the server answered and the answer looked
    # well-formed. It is the count of accepts the old path would have made and this one did not,
    # which is the only number that says whether the wired re-ask is doing anything at all in a
    # live episode -- 0 here means the gate never fired, not that it fired and found nothing.
    n_induce_defect_reasks: int = 0
    # The same fact for the GOAL half (2026-08-01), counted SEPARATELY rather than folded into
    # the number above. Two reasons, both about what a witness has to be able to say. First,
    # the two gates have independent budgets, so one shared count could not tell an operator
    # which budget was spent. Second, the goal gate is opt-in and default OFF: a 0 here on a
    # run with the flag on is evidence the gate never fired, and merging it into a counter
    # that the always-on engine gate also increments would make that reading impossible.
    n_goal_defect_reasks: int = 0
    server_failure_diagnostics: list = field(default_factory=list)
    last_generated_tokens: int = -1
    # DECLARED-VS-ACTUAL (2026-07-27 review finding 1). `n_ctx` above is what we INTEND to
    # launch with. These two record what a RUNNING server on our port actually reports, so
    # the liveness witness can publish an OBSERVED value instead of re-publishing our own
    # intent -- the exact gap that let the n_ctx fix be a silent no-op against a stale server.
    observed_server_n_ctx: Optional[int] = None
    reuse_n_ctx_check: str = "not_checked"
    reuse_refusals: list = field(default_factory=list)
    # Live children this instance TERMINATED because a later step was about to overwrite the
    # `self._proc` reference that was the only thing keeping them reachable (2026-08-08 review
    # finding 4: a still-loading server left running past the wait budget, or a refused-for-reuse
    # server, used to be silently dropped when the next launch replaced `self._proc`, orphaning a
    # process that could hold ~20 GB of VRAM with nothing left able to stop it). Bounded and kept
    # on its OWN channel, like `reuse_refusals` above, and for the same reason: this is bookkeeping
    # about a PRIOR attempt, not a failure of the CURRENT one, so it must not feed
    # `n_server_failures` and flip a liveness gate for a run whose eventual server is healthy.
    orphaned_child_cleanups: list = field(default_factory=list)
    # ...and the same declared-vs-actual treatment for WHICH MODEL is loaded (2026-07-28). Added
    # with the generator switch: `repo_substr` is our INTENT, and a stale server from the previous
    # pin satisfied every prior reuse condition, so the witness could report gemma while the run
    # induced on Qwen. These record what the running server actually says.
    observed_server_model_path: Optional[str] = None
    reuse_model_check: str = "not_checked"
    # Set by `__post_init__` when it re-fitted `ffn_cpu_layers` because this instance's `mtp`
    # disagreed with the environment default the dataclass factory had used. Recorded rather than
    # silently corrected: a re-fit means somebody's mental model of this launch was wrong, and the
    # artifact should be able to say so.
    ffn_cpu_layers_refit_note: str = ""

    def __post_init__(self) -> None:
        """Re-fit `ffn_cpu_layers` from THIS INSTANCE'S `mtp`, not from the environment default.

        THE BUG THIS CLOSES, WHICH IS A SILENT-LLM-OFF BUG AND NOT A TUNING NICETY.
        `ffn_cpu_layers` is a `default_factory=_default_ffn_cpu_layers`, and a dataclass evaluates
        its default factories with NO ACCESS to the other fields being constructed. So the
        auto-fit sized the offload against `_mtp_default_on()` -- the ENVIRONMENT's answer -- while
        the server this proposer is about to launch uses `self.mtp`, the CONSTRUCTOR's answer.
        Whenever those two disagree the offload is fitted for the wrong configuration.

        They disagree constantly in practice. Eight-plus harnesses build
        `LocalGGUFProposer(mtp=os.environ.get("CARNOT_ARC_MTP", "1") != "0")` -- note the literal
        `"1"`, which is NOT the canonical local default of `"0"` -- so with `CARNOT_ARC_MTP` unset
        the instance gets `mtp=True` while the factory fitted layers for `mtp=False`. Worked
        through at the shipped n_ctx 81920 on a 24123 MiB card:

            factory fits for mtp=False -> 7 layers, `_generator_cuda_min_free_mb(7, True)` = 25311
            no re-fit, mtp=True         -> 0 layers, `_generator_cuda_min_free_mb(0, True)` = 26678

        Both exceed what the card has, so the guard declines CUDA, the generator falls back to the
        iGPU HIP build at ~2 tok/s, every induce call exceeds its timeout, `generate()` returns
        `(False, msg)`, and the agent proceeds LLM-OFF while still reporting itself as the LLM-on
        scored path. Re-fitting here makes the two sides agree BY CONSTRUCTION for every call site
        at once, present and future, instead of requiring each one to be found and corrected.

        WHY A SENTINEL RATHER THAN ALWAYS RE-FITTING. A caller who names `ffn_cpu_layers=` has
        stated a fact about the launch they want, and silently overriding it would be the same
        class of error in the other direction -- the tests that pin exact argv for a given layer
        count depend on the explicit value surviving. `_FFN_CPU_LAYERS_AUTO` (-1) means "nobody
        chose; fit it", which is what the default factory's result is re-expressed as below.

        KAGGLE IS UNAFFECTED. `_default_ffn_cpu_layers()` returns 0 unless
        `CARNOT_ARC_GENERATOR_CUDA_GPU` names a real CUDA card, and the scored kernel does not set
        it. A re-fit of 0 to 0 changes no argv.
        """
        if self.ffn_cpu_layers != _FFN_CPU_LAYERS_AUTO:
            return  # the caller named a value; it is not ours to override
        try:
            self.ffn_cpu_layers = _default_ffn_cpu_layers(mtp=self.mtp)
        except Exception:  # an audit convenience must never break the generator it audits
            self.ffn_cpu_layers = 0
            return
        env_answer_differs = self.mtp != _mtp_default_on()
        if env_answer_differs:
            self.ffn_cpu_layers_refit_note = (
                f"ffn_cpu_layers auto-fitted to {self.ffn_cpu_layers} against THIS proposer's "
                f"mtp={self.mtp}, which differs from the environment default "
                f"mtp={_mtp_default_on()}. Sizing it from the environment instead would have "
                "fitted the offload for a configuration this server is not about to launch."
            )
            _note_generator_selection(self.ffn_cpu_layers_refit_note)

    def _url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @staticmethod
    def sampling_seed(attempt: int = 0) -> int | None:
        """OPT-IN sampler seed for the `/completion` payload. ``None`` = today's behaviour exactly.

        WHY THIS EXISTS (measured 2026-07-29, and it is the root cause of three wasted A/B runs).
        Every generation this class issues goes out at ``temperature = 0.2 + 0.1*attempt`` --
        NONZERO -- and with NO ``seed`` field. `llama-server` defaults an absent seed to -1, which
        means "pick a fresh random one", so **two runs of identical code on the identical game with
        the identical harness `seed` produce different LLM output**. The harness `seed` argument
        seeds `random`/`numpy` inside the driver; it never reached the server's sampler.

        The cost of that gap, measured rather than assumed: comparing two runs that share the same
        treatment, seed, model file and game (`ret1` in
        `results/arc_engine_retention_20260729/cells` against `31b` in
        `results/arc_heldout_31b_vs_9b_20260728/cells`), **2 of 5 cells diverge under IDENTICAL
        CODE** -- a 40% nondeterminism rate. That floor is at least as large as any treatment
        effect yet measured on this path, so an A/B here is uninterpretable without an A/A control
        no matter how many cells it runs. It is why the engine-retention grid's single "perturbed"
        cell (vc33, divergence at action index 17) turned out to perturb under A/A too, and why
        four runs spanning two different treatments in two different experiments all diverge at
        that same index 17 while the partition crosses treatment lines.

        THE DESIGN, and why each choice is deliberate:

        * **Default OFF.** With ``CARNOT_ARC_GENERATOR_SEED`` unset this returns None and the
          caller omits the field entirely, so the payload is byte-identical to today's. The live
          scored agent's behaviour is unchanged unless an operator opts in. Determinism is a
          measurement property, and quietly changing how the scored agent samples is not a
          measurement change -- it is a behaviour change, and it is not this function's to make.
        * **The seed VARIES WITH `attempt`.** A single fixed seed would break the retry ladder:
          the whole point of ``0.2 + 0.1*attempt`` is that a failed induction is retried with more
          diversity, and re-sending the same seed at a higher temperature would still re-explore,
          but pinning it would make attempt 2 far more correlated with attempt 1 than intended.
          ``base * 1000 + attempt`` keeps every attempt distinct while making the WHOLE RUN
          reproducible, which is the property an A/B needs.
        * **Non-integer or absent values fall back to None** rather than raising. A malformed env
          var must not take down a live episode; it should just leave behaviour as it is today.

        With this set, a `pre` vs `post` trace difference is attributable to the code change
        without needing a third arm -- and an A/A arm should come back byte-identical, which is a
        cheap positive control on the determinism itself.
        """
        raw = os.environ.get("CARNOT_ARC_GENERATOR_SEED")
        if raw is None or raw.strip() == "":
            return None
        try:
            base = int(raw)
        except (TypeError, ValueError):
            return None
        return base * 1000 + int(attempt)

    def _effective_model_label(self) -> str:
        """The model this proposer is ACTUALLY configured for, for human-facing messages.

        `model_path` (the CARNOT_ARC_GGUF_PATH override) supersedes `repo_substr` at load
        time, so a failure message must name it first — same rule `liveness_witness()`
        applies for `generator_model_declared`. REQ-ARC-WMTE-6670: the supab5 A/B failed
        with a message naming the harness's frozen 9B pin while the 27B was what actually
        loaded and served, and that label misdirected a whole investigation.
        """
        if self.model_path:
            return Path(str(self.model_path)).name
        return str(self.repo_substr)

    # The lines llama-server prints when it receives SIGINT/SIGTERM. Their presence in
    # the stderr tail means the server was told to stop from OUTSIDE — not a crash, not
    # a resource fault. See ops/known-issues.md 2026-08-23 (the reaper resolution).
    _EXTERNAL_KILL_MARKERS = ("cleaning up before exit", "Received second interrupt")

    def _server_death_signature(self) -> str:
        """One short line naming what a launched server's own log says about its death.

        Checks the CURRENT launch's log first, then the PREVIOUS one: when a killed
        server is relaunched and the relaunch fails, the kill evidence sits in the
        previous log, not the fresh one. Returns '' when no log carries the marker.
        Never raises: this runs inside the failure-note path, and a diagnostic helper
        that can take down the note it is enriching is worse than no helper
        (REQ-ARC-WMTE-6670, SCENARIO-6670-3).

        Wording note: llama-server prints these lines for SIGTERM and SIGINT alike,
        including a SIGTERM this class itself sent via stop()/_terminate_stale_proc —
        so the hint says "termination signal", not "external kill". The reader decides
        attribution; the note's job is to surface the evidence and where it lives.
        """
        for which, log_path in (
            ("", getattr(self, "_stderr_log_path", None)),
            ("previous ", getattr(self, "_prev_stderr_log_path", None)),
        ):
            if not log_path:
                continue
            try:
                # Bounded read: seek to the tail instead of loading a whole
                # multi-hour server log per failure note.
                with open(log_path, "rb") as fh:
                    fh.seek(0, 2)
                    fh.seek(max(0, fh.tell() - 2000))
                    tail = fh.read().decode("utf-8", "replace")
            except OSError:
                continue
            if any(marker in tail for marker in self._EXTERNAL_KILL_MARKERS):
                return (
                    f"{which}server log records a termination signal (SIGTERM/SIGINT) "
                    f"— see {log_path}"
                )
        return ""

    def _note_server_failure(self, diagnostic: str) -> None:
        """Count + KEEP a server-side failure diagnostic (bounded, so a storm cannot grow
        without limit). This is the record the scored path never had.

        REQ-ARC-WMTE-6670: when the launched server's own stderr says it was terminated
        from outside, that evidence is appended here — so the row note names the true
        cause instead of leaving it in a log nobody reads.
        """
        self.n_server_failures += 1
        hint = self._server_death_signature()
        if hint:
            diagnostic = f"{diagnostic} [{hint}]"
        if len(self.server_failure_diagnostics) < 24:
            self.server_failure_diagnostics.append(diagnostic[:400])

    def liveness_witness(self) -> dict:
        """The generator-liveness primitives, in the SHAPE `scripts/arc_llm_on_liveness_lint.py`
        already recomputes from (`llm.responses`, `generator_healthy_after`), so a scored-path
        row can be audited by the SAME gate as a harness row rather than needing a second,
        differently-buggy checker."""
        healthy = self._healthy()
        # Only ask the server what it is if it is actually up; on a dead server /props costs
        # a 3s timeout per witness call and returns nothing useful anyway.
        observed_n_ctx = self.observed_n_ctx() if healthy else None
        observed_slots = self.observed_total_slots() if healthy else None
        return {
            "llm": {
                "calls": int(self.n_completion_calls),
                "responses": int(self.n_completion_ok),
                "errors": int(self.n_server_failures),
                "content_failures": int(self.n_content_failures),
                "induce_defect_reasks": int(self.n_induce_defect_reasks),
                "goal_defect_reasks": int(self.n_goal_defect_reasks),
            },
            "generator_healthy_after": bool(healthy),
            "generator_server_failure_diagnostics": list(self.server_failure_diagnostics),
            "generator_port": int(self.port),
            # OBSERVED, not declared (2026-07-27 review finding 1). This used to publish
            # `int(self.n_ctx)` -- our own INTENT -- so a run that reused a stale server with a
            # smaller pool reported the pool it wished it had. Reading /props makes the witness
            # a measurement of the server rather than an echo of the caller, which is the whole
            # point of a liveness witness. `generator_n_ctx_source` is published alongside so a
            # reader can tell an observation from the declared fallback rather than having to
            # assume; `declared_only` is exactly the state in which the number is NOT evidence.
            "generator_n_ctx": int(observed_n_ctx if observed_n_ctx is not None else self.n_ctx),
            "generator_n_ctx_declared": int(self.n_ctx),
            "generator_n_ctx_source": (
                "server_props_observed" if observed_n_ctx is not None else "declared_only"
            ),
            "generator_total_slots_observed": observed_slots,
            "generator_reuse_n_ctx_check": str(self.reuse_n_ctx_check),
            "generator_reuse_refusals": list(self.reuse_refusals),
            # OBSERVED model identity, alongside the declared repo_substr. Without the observed
            # value a witness that says "gemma-4-31B-it" is only restating our own configuration.
            "generator_model_declared": str(self.model_path or self.repo_substr),
            "generator_model_observed": self.observed_server_model_path,
            "generator_reuse_model_check": str(self.reuse_model_check),
            "generator_max_tokens": int(self.max_tokens),
            # MTP: what we ASKED for, versus what the runtime SAID it did. Published as three
            # separate fields on purpose. `generator_mtp_requested` is our configuration;
            # `generator_mtp_draft_path` is the argv we built; `generator_mtp_engaged` is the only
            # one of the three that is EVIDENCE, because it comes from the server's own stderr.
            # Before this, an artifact could report a fully "MTP-on" run that had speculation
            # silently disabled -- llama.cpp accepts an unusable draft, warns, and serves normally,
            # so every other field on this witness looks identical either way.
            "generator_mtp_requested": bool(self.mtp),
            "generator_mtp_draft_path": str(self.last_mtp_draft_path),
            "generator_mtp_engaged": self.last_mtp_engaged,
            "generator_mtp_evidence": str(self.mtp_engaged_evidence or self.mtp_disabled_reason),
            # Recorded because a re-fit means the offload was sized against a different `mtp` than
            # the launch used -- see `LocalGGUFProposer.__post_init__`. Empty in the normal case.
            "generator_ffn_cpu_layers": int(self.ffn_cpu_layers),
            "generator_ffn_cpu_layers_refit_note": str(self.ffn_cpu_layers_refit_note),
            # A prior launch attempt's child that this instance had to stop itself, so the leak
            # is visible in the artifact rather than only in `nvidia-smi` on the host.
            "generator_orphaned_child_cleanups": list(self.orphaned_child_cleanups),
        }

    def _record_completion_diagnostics(self, response: dict) -> None:
        self.last_stop_type = str(response.get("stop_type") or "")
        self.last_prompt_truncated = bool(response.get("truncated"))
        self.last_raw_completion = str(response.get("content") or "")
        # How many tokens the server ACTUALLY generated. Load-bearing for telling the two
        # "stop_type == limit" cases apart -- see _limit_diagnostic().
        timings = response.get("timings")
        got = (timings or {}).get("predicted_n") if isinstance(timings, dict) else None
        self.last_generated_tokens = int(got) if isinstance(got, int) else -1

    def _limit_diagnostic(self) -> str:
        """Distinguish the TWO different faults that both report stop_type == 'limit'.

        The old message said "HIT n_predict=<max_tokens> OUTPUT LIMIT" for both, which is
        actively misleading in the second case and is why exp5866's mode C went unnoticed:

          * INTENDED BUDGET LIMIT -- the model generated the full max_tokens we asked for
            and was still going. The fix is a bigger max_tokens.
          * SHARED-POOL TRUNCATION -- the model was cut off FAR short of max_tokens because
            the prompt had already consumed most of the server's shared context pool, so
            only the leftover cells were available to generate into. The fix is a bigger
            -c / CARNOT_ARC_INDUCE_N_CTX, and a bigger max_tokens would make it WORSE.
            Measured shape: a 15754-token prompt in a 16384 pool left 630 cells, produced
            2133 characters, and returned HTTP 200 -- indistinguishable, before this
            change, from a healthy-but-terse model.
        """
        got = self.last_generated_tokens
        # Compare against the budget the request actually CARRIED (pool-clamped, REQ-ARC-WMTE-
        # 6620), not the configured cap -- otherwise every clamped request that ran its full
        # clamped budget would misreport as pool truncation. -1 (no call yet) falls back.
        asked = (
            self.last_requested_n_predict if self.last_requested_n_predict > 0 else self.max_tokens
        )
        if isinstance(got, int) and 0 <= got < asked - 8:
            return (
                f" [TRUNCATED BY SHARED CONTEXT POOL: generated only {got} of the "
                f"{asked}-token budget in an n_ctx={self.n_ctx} pool -- the prompt "
                f"consumed the rest. RAISE -c / CARNOT_ARC_INDUCE_N_CTX; raising max_tokens "
                f"would make this worse]"
            )
        return f" [HIT n_predict={asked} OUTPUT LIMIT before completing]"

    def _chat_complete_request(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: Optional[list],
        attempt: int = 0,
        repeat_penalty: Optional[float] = None,
        repeat_last_n: Optional[int] = None,
        _continuation_prefix: Optional[str] = None,
    ) -> tuple[dict, str]:
        """POST one user turn to the OpenAI-compatible /v1/chat/completions endpoint (the server
        applies the GGUF's OWN embedded chat template -- the turn delimiters Qwen3.6/ThinkingCap
        need) and normalize the OpenAI-shaped reply back into llama.cpp's native
        {content, stop_type, truncated} shape. Returns (normalized_response, extraction_text):

          * normalized_response["content"] -> the FULL generated text. Some llama.cpp builds
            extract the <think> reasoning into a separate `reasoning_content` field and strip it
            from `content`; we fold it back in (wrapped in <think></think>) so
            _record_completion_diagnostics, reason_engaged detection, and max_raw_completion_len
            stay faithful to EVERYTHING the model emitted (reasoning + answer).
          * extraction_text -> the FINAL answer only (reasoning stripped when the build split it),
            so _extract_python cannot accidentally grab a ```python block written INSIDE the
            model's reasoning trace.

        `repeat_penalty`/`repeat_last_n` (added 2026-08-08, adversarial review): the raw
        /completion path (this method's sibling) has scoped these to the engine-induce prompt
        since REQ-ARC-WMTE-6198's repeat-penalty fix (13/36 -> 22/36 usable engines). This
        method silently dropped both, so any call routed here -- which INCLUDES every think-mode
        call, since think mode forces the chat endpoint regardless of `use_chat_template`
        -- ran without repetition control the raw path would have applied to the identical
        prompt. `None` (the default) omits the field exactly as before this fix.

        `_continuation_prefix`, internal-only (set only by the retry this method issues on
        itself -- see CARNOT_ARC_CHAT_FORCE_ANSWER_CONTINUATION below), appends a trailing
        assistant-role message so the server continues generation from that exact text instead
        of starting a fresh turn. Not part of the public call surface.

        Raises on a network/transport error; the caller converts that to its failure tuple,
        exactly like the raw /completion path."""
        import json as _json
        import urllib.request

        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        if _continuation_prefix is not None:
            messages.append({"role": "assistant", "content": _continuation_prefix})
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "cache_prompt": True,
        }
        if _continuation_prefix is not None:
            # llama-server's convention for resuming generation from a supplied assistant
            # prefix: skip re-adding the template's own generation prompt (the turn markers
            # that would otherwise be inserted AFTER our partial assistant text, corrupting it).
            payload["add_generation_prompt"] = False
        if repeat_penalty is not None:
            payload["repeat_penalty"] = repeat_penalty
            if repeat_last_n is not None:
                payload["repeat_last_n"] = repeat_last_n
        # Same opt-in determinism as the raw /completion path -- it must be on BOTH, or the
        # chat-template route (which Qwen3.6/ThinkingCap take) would stay nondeterministic while
        # the artifact claimed a seeded run. `attempt` is threaded in from the retry ladder so
        # this route gets the SAME per-attempt seed the /completion route does; without it the
        # chat path would reuse one seed across a ladder whose whole purpose is diversity.
        _seed = self.sampling_seed(attempt)
        if _seed is not None:
            payload["seed"] = _seed
        # OPT-IN THINKING BUDGET (2026-08-17, default OFF -- unset means byte-identical requests).
        #
        # On a reasoning generator the think channel is the bulk of the tokens: measured
        # completions of 38k-94k where the final answer is 1-8k. `thinking_budget_tokens` is a
        # per-request llama.cpp field (server-common.cpp reads it into reasoning_budget) that
        # forces `</think>` after N think tokens and lets the model still answer. That is NOT the
        # same as capping `max_tokens`, which hard-truncates mid-answer and produced dead cells
        # in earlier work -- the difference is a graceful close versus a cut.
        #
        # Deliberately not enabled by default. Independent review argues thinking HELPS held-out
        # accuracy on the incumbent generator (gemma /think, exp6221: 0.196 vs 0.083), so a cap is
        # a mitigation for a long-completion model, not a general win. Wired so it can be measured
        # without another code change; the A/B is a separate decision.
        _think_budget = os.environ.get("CARNOT_ARC_INDUCE_THINKING_BUDGET")
        if _think_budget:
            try:
                _tb = int(_think_budget)
            except ValueError:
                _tb = 0
            if _tb > 0:
                payload["thinking_budget_tokens"] = _tb
        if stop:
            payload["stop"] = list(stop)
        req = urllib.request.Request(
            self._url() + "/v1/chat/completions",
            data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            raw = _json.load(r)
        choice = (raw.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        final = str(msg.get("content") or "")
        reasoning = str(msg.get("reasoning_content") or "")
        # OpenAI finish_reason 'length' == hit max_tokens == llama.cpp stop_type 'limit' (overran).
        stop_type = "limit" if choice.get("finish_reason") == "length" else "eos"
        full = f"<think>\n{reasoning}\n</think>\n{final}" if reasoning else final
        # PURE OBSERVATION -- see the field declarations. Recorded before any branch so the
        # per-channel truth is available even when the caller goes on to fail the candidate.
        self.last_final_content = final
        self.last_reasoning_content = reasoning
        # OPT-IN EMPTY-ANSWER-CHANNEL FALLBACK (CARNOT_ARC_CHAT_EMPTY_CONTENT_FALLBACK=1).
        #
        # WHAT IT IS FOR. This build splits the model's thought channel out into
        # `reasoning_content` and leaves `content` holding only the post-thought answer. That is
        # correct and is why `extraction_text` is `final` -- so `_extract_python` cannot grab a
        # ```python block the model wrote as a DRAFT inside its own reasoning. But when the model
        # closes its thought channel and emits EOS without writing an answer, `content` is empty,
        # `extraction_text` is empty, and the candidate is a guaranteed miss no matter what the
        # model actually worked out. With this flag on, an empty answer channel falls back to the
        # reasoning channel rather than to nothing.
        #
        # WHY IT IS DEFAULT-OFF AND MUST STAY SO. The fallback reads a channel the model did not
        # nominate as its answer, so it can surface a draft the model was in the middle of
        # rejecting. That is a real quality risk and the frozen live/scored generator path must not
        # inherit it silently. It is also strictly a NO-OP whenever `final` is non-empty: every
        # existing passing path is byte-identical with the flag on or off, which is the property
        # the both-directions test pins.
        extraction = final
        if (
            not final.strip()
            and reasoning.strip()
            and os.environ.get("CARNOT_ARC_CHAT_EMPTY_CONTENT_FALLBACK") == "1"
        ):
            extraction = reasoning
        # OPT-IN FORCED-CONTINUATION RETRY (CARNOT_ARC_CHAT_FORCE_ANSWER_CONTINUATION=1).
        #
        # WHAT IT IS FOR. Measured live 2026-08-05, n=5 cells, one call each: the model reasons
        # correctly (sb26 derives the exact right numpy slice; r11l explicitly decides "let's
        # just return the grid as is") and then emits EOS at </think> WITHOUT EVER WRITING CODE,
        # at 4.5k-11.7k tokens -- all well under the 16384 budget, so this is not truncation. The
        # channel-fallback above cannot help here: it reads the reasoning channel INSTEAD of the
        # answer channel, but in this failure shape neither channel contains a code block, so the
        # fallback just substitutes one codeless text for another.
        #
        # THE FIX. Re-issue ONE request with the model's own reasoning fed back as a trailing
        # assistant-role message (an "assistant prefill" -- the standard llama.cpp-server
        # mechanism for resuming generation from supplied text) plus an explicit nudge to
        # continue into the code fence. The model continues from exactly where it stopped rather
        # than re-planning from scratch, so this costs one short follow-up call, not a full retry.
        #
        # WHY IT IS DEFAULT-OFF, SEPARATE FROM THE FLAG ABOVE, AND ONLY FIRES ONCE. Same
        # quality-risk reasoning as the channel fallback -- forcing continuation past a model's
        # own EOS is a real intervention the frozen live/scored path must not inherit silently.
        # Kept as a SEPARATE flag (not folded into CARNOT_ARC_EMPTY_CONTENT_FALLBACK) because it
        # is a materially different mechanism: a second network call, not a different read. Fires
        # at most once per call -- `_continuation_prefix is None` is what distinguishes a
        # top-level call from the retry it issues on itself, so the retry can never retry again.
        # No-op whenever `extraction` already has a code marker: every passing path is untouched
        # with the flag on or off, which is the property the both-directions test pins.
        if (
            _continuation_prefix is None
            and "```" not in extraction
            and "def " not in extraction
            and reasoning.strip()
            and os.environ.get("CARNOT_ARC_CHAT_FORCE_ANSWER_CONTINUATION") == "1"
        ):
            fence = "```python\n"
            _prefix = reasoning.rstrip() + "\n</think>\n\n" + fence
            retry_normalized, retry_extraction = self._chat_complete_request(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                attempt=attempt,
                repeat_penalty=repeat_penalty,
                repeat_last_n=repeat_last_n,
                _continuation_prefix=_prefix,
            )
            # DEFENSIVE, NOT ASSUMED: whether this llama.cpp build echoes the supplied prefix
            # back in its response, or returns only the newly-generated continuation tokens, is
            # server-specific and was NOT confirmed before this shipped (see the commit message
            # for the live-probe result). Handle both without guessing which one is live: if the
            # retry's own extraction already carries a fence/def marker, the prefix was echoed
            # and `retry_extraction` is already complete; otherwise it is continuation-only and
            # our forced fence belongs in front of it.
            final = (
                retry_extraction
                if ("```" in retry_extraction or "def " in retry_extraction)
                else (fence + retry_extraction if retry_extraction.strip() else "")
            )
            full = str(retry_normalized.get("content") or final)
            extraction = final
            self.last_final_content = final
            raw = retry_normalized
        # HOW MANY TOKENS THE SERVER ACTUALLY GENERATED -- normalized into llama.cpp's native
        # `timings.predicted_n` shape. WITHOUT THIS the mode-C detector is STRUCTURALLY DEAD on
        # this endpoint (found 2026-07-27, adversarial review): the normalized dict carried no
        # `timings` key at all, so `_record_completion_diagnostics` set `last_generated_tokens =
        # -1`, and `_limit_diagnostic()`'s pool-truncation branch (`0 <= got < max_tokens - 8`)
        # could NEVER be true when use_chat_template=True -- it always fell through to the
        # actively-misleading "HIT n_predict OUTPUT LIMIT" message, whose prescription (raise
        # max_tokens) is the OPPOSITE of the correct one (raise n_ctx). That is the same
        # dead-channel class the diagnostic was added to fix, reintroduced on the sibling
        # endpoint. Two sources because llama.cpp builds differ: newer ones attach a native
        # top-level `timings`, all of them fill OpenAI `usage.completion_tokens`.
        timings = raw.get("timings") if isinstance(raw.get("timings"), dict) else None
        predicted_n = (timings or {}).get("predicted_n")
        if not isinstance(predicted_n, int):
            usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
            ct = usage.get("completion_tokens")
            predicted_n = ct if isinstance(ct, int) else None
        normalized: dict[str, Any] = {
            "content": full,
            "stop_type": stop_type,
            "truncated": bool(raw.get("truncated")),
        }
        if isinstance(predicted_n, int):
            normalized["timings"] = {"predicted_n": predicted_n}
        return normalized, extraction

    def _healthy(self) -> bool:
        import urllib.request

        try:
            with urllib.request.urlopen(self._url() + "/health", timeout=2) as r:
                return b"ok" in r.read()
        except Exception:
            return False

    def server_props(self) -> dict:
        """Read the RUNNING server's own /props. This is the only channel that reports what
        the server was actually LAUNCHED with; every other field on this object reports what
        we INTENDED. Returns {} when /props is unreachable or unparseable (never raises)."""
        import json as _json
        import urllib.request

        try:
            with urllib.request.urlopen(self._url() + "/props", timeout=3) as r:
                raw = _json.load(r)
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        return raw

    def observed_n_ctx(self) -> Optional[int]:
        """The n_ctx the RUNNING server reports, or None if /props is unreachable.

        llama.cpp reports the context pool under default_generation_settings.n_ctx; some
        builds also surface a bare top-level n_ctx. Both are read so a build difference
        cannot silently degrade this into 'unobservable' (which would re-open exactly the
        declared-vs-actual gap this method exists to close)."""
        props = self.server_props()
        if not props:
            return None
        gen = props.get("default_generation_settings")
        for candidate in (
            (gen or {}).get("n_ctx") if isinstance(gen, dict) else None,
            props.get("n_ctx"),
        ):
            if isinstance(candidate, int) and candidate > 0:
                return int(candidate)
        return None

    def observed_model_path(self) -> Optional[str]:
        """The GGUF path the RUNNING server was launched with, or None if unreadable.

        llama.cpp's /props reports it top-level as `model_path` (verified directly against this
        project's own build, 2026-07-28: the key is present and absolute, while
        `default_generation_settings` carries only `n_ctx`/`params` and no model field at all).
        `model_alias` is also present but is a display name, so `model_path` is the load-bearing
        one. Read defensively -- an unreadable value must degrade to None, never raise.
        """
        props = self.server_props()
        if not props:
            return None
        raw = props.get("model_path") or props.get("model") or props.get("model_alias")
        return str(raw) if isinstance(raw, str) and raw.strip() else None

    def observed_total_slots(self) -> Optional[int]:
        props = self.server_props()
        slots = props.get("total_slots") if props else None
        return int(slots) if isinstance(slots, int) and slots > 0 else None

    def _model_path_matches(self, observed: str) -> bool:
        """Does the running server's GGUF correspond to the one THIS proposer is configured for?

        Two configuration shapes, so two comparisons:

          * An explicit `model_path` (the Kaggle bundle sets `CARNOT_ARC_GGUF_PATH`): compare
            BASENAMES, not full paths. The same weights legitimately live at different absolute
            paths in different environments, and a full-path compare would refuse a perfectly good
            warm server for a directory-layout difference.
          * Only a `repo_substr` (the local cache path): require the substring to appear in the
            observed path, case-insensitively. `_resolve_gguf` finds the file by exactly this
            substring, so anything it would have resolved will match, and a different model's path
            will not.

        Deliberately permissive about QUANT: a Q4_K_M vs Q5 of the same model differs in quality,
        not in identity, and refusing across quants would relaunch a second copy of what is
        substantially the right model. The failure this guards is a DIFFERENT MODEL entirely.
        """
        obs = str(observed or "")
        if not obs:
            return False
        if self.model_path:
            from pathlib import Path as _P

            return _P(obs).name == _P(str(self.model_path)).name
        return str(self.repo_substr).lower() in obs.lower()

    def _reusable(self) -> bool:
        """Is an ALREADY-RUNNING server on our port usable as OUR configured generator?

        THE HOLE THIS CLOSES (2026-07-27 review finding 1). `_ensure_server` used to return
        True on a bare /health check. /health only says "a llama-server is listening"; it
        says NOTHING about the context pool that server was launched with. So the 2026-07-27
        n_ctx 16384 -> 81920 fix was a SILENT NO-OP against any long-lived server already on
        the port: verified live on the dev box, port 8919 (this class's DEFAULT port) was
        serving n_ctx=16384 from a launch the previous evening, `_ensure_server()` returned
        True without launching anything, and `liveness_witness()` reported 81920 -- the
        INTENDED value read off `self.n_ctx`. A run in that state self-certifies as fixed
        while running on the faulty pool, which is the same declared-vs-actual silent
        degradation the fix was chartered to eliminate, one layer up.

        Refusing to reuse (rather than adopting the observed value) is deliberate: adopting
        would make the process quietly run a configuration nobody asked for, and the
        admission arithmetic (K_concurrent * (prompt + max_tokens) <= n_ctx) that the
        shipped default was sized against would no longer hold.

        A server whose /props cannot be read is reused with a WARNING record rather than
        refused, so a llama.cpp build that does not serve /props does not brick the path.

        THE SECOND HOLE, closed 2026-07-28 with the generator switch. This check compared the
        context POOL and nothing else -- so a running server was adopted regardless of WHICH MODEL
        it had loaded. That was near-harmless while exactly one model was ever served on port 8919.
        It stops being harmless the moment the pin changes: a Qwen3.5-9B server left over from a
        previous run, on the default port, with n_ctx 81920, satisfies every condition above and
        gets adopted -- so the run induces with the RETIRED model while `liveness_witness()`
        faithfully reports `repo_substr` read off `self.repo_substr`, i.e. gemma. That is the exact
        declared-vs-actual shape described two paragraphs up, in the model dimension instead of the
        context dimension, and it is most likely to fire precisely during a model transition when
        stale servers are lying around. Same policy as the n_ctx check: refuse and relaunch on a
        fresh port (never adopt, never fight for the port), and fail OPEN if /props is unreadable.
        """
        observed_model = self.observed_model_path()
        if observed_model is None:
            self.reuse_model_check = "unobserved_model_path_unreadable"
        else:
            self.observed_server_model_path = observed_model
            if self._model_path_matches(observed_model):
                self.reuse_model_check = "match"
            else:
                self.reuse_model_check = f"refused_wrong_model observed={observed_model} want={self.model_path or self.repo_substr}"
                return False
        observed = self.observed_n_ctx()
        if observed is None:
            self.reuse_n_ctx_check = "unobserved_props_unreachable"
            return True
        self.observed_server_n_ctx = observed
        if observed >= int(self.n_ctx):
            # >= not ==: a LARGER pool than we asked for still satisfies our admission
            # arithmetic. Only a SMALLER pool can silently truncate/500 under concurrency.
            self.reuse_n_ctx_check = "match" if observed == int(self.n_ctx) else "larger_ok"
            return True
        self.reuse_n_ctx_check = f"refused_smaller_pool observed={observed} want={self.n_ctx}"
        return False

    def _terminate_stale_proc(self, reason: str) -> None:
        """Terminate whatever `self._proc` currently holds, if it is still alive, then clear it.

        THE LEAK THIS CLOSES (2026-08-08 review finding 4). `_ensure_server` had two places that
        replace `self._proc` -- the fresh Popen after a wait-exhaustion timeout, and the fresh
        Popen after refusing to reuse an already-running server on our port -- and neither one
        stopped the PREVIOUS child first. `stop()` can only ever terminate whatever `self._proc`
        currently points at, so the instant it was reassigned the earlier process became
        unreachable by any cleanup path this class has, and kept running (and holding its VRAM)
        until something outside this process noticed and killed it by hand. Calling this right
        before every such reassignment, and once more on the wait-exhaustion return itself, means
        a live child is never dropped without first being asked (then told) to exit.

        SIGTERM first, like `stop()` already uses, so a graceful shutdown is always tried first;
        a short wait; SIGKILL only if it refuses to die within that wait. A cleanup step must
        never itself take down the launch it is protecting, so every failure mode here is caught
        and folded into the diagnostic instead of raised.
        """
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return  # never launched, or already exited on its own -- nothing to clean up
        diagnostic = f"{reason} (pid={getattr(proc, 'pid', '?')}, port={self.port})"
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    diagnostic += "; did not exit even after kill()"
        except Exception as exc:  # cleanup must never crash the launch it is protecting
            diagnostic += f"; termination raised {exc!r}"
        if len(self.orphaned_child_cleanups) < 24:
            self.orphaned_child_cleanups.append(diagnostic[:400])

    def _ensure_vllm_server(self) -> bool:
        """Launch or reuse a vLLM OpenAI server (REQ-ARC-WMTE-6510). Kaggle-Blackwell only.

        Deliberately self-contained: health, reuse and launch live here so the llama.cpp path
        above is byte-untouched and its tests keep meaning what they meant. The launch recipe is
        the one PROVEN by bench v13/v15 on the actual scored card -- native FP4 selected
        (`FlashInferCutlassNvFp4LinearKernel` in the log), fp8 KV engaged and measured free, and
        the toolchain/linker environment (coherent 13.0.x pip CUDA + dev-symlinks) prepared by
        the kernel BEFORE the agent starts, not here.

        MTP is deliberately NOT configured on this backend: batching was measured to beat
        speculation decisively at concurrency (llama.cpp: 228.3 batched vs 108.8 MTP-on at k=16),
        and vLLM's own MTP interaction with batching is unmeasured. Recorded on the instance so a
        run cannot silently believe it had speculation.
        """
        self.last_mtp_draft_path = ""
        self.mtp_disabled_reason = (
            "vllm_backend: MTP not configured (batching beats speculation at concurrency; unmeasured under vLLM)"
            if self.mtp
            else ""
        )
        model_dir = _resolve_vllm_model_dir()
        if not model_dir:
            self._note_server_failure(
                "vllm backend active but no safetensors model dir resolved "
                f"(set {_VLLM_MODEL_DIR_ENV} or attach the dataset)"
            )
            return False
        if self._vllm_healthy() and self._vllm_reusable(model_dir):
            return True
        self._terminate_stale_proc("replacing server for vllm launch")
        # max-model-len from the SAME admission arithmetic the llama.cpp path uses, expressed
        # per-sequence: prompt + completion + slack. PagedAttention shares the pool, so this is
        # a per-request ceiling rather than a divided static pool.
        max_len = int(_INDUCE_WORST_CASE_PROMPT_TOKENS + self.max_tokens + 2048)
        args = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_dir,
            "--served-model-name",
            "m",
            "--max-model-len",
            str(max_len),
            "--gpu-memory-utilization",
            "0.90",
            "--max-num-seqs",
            str(_vllm_max_seqs()),
            "--kv-cache-dtype",
            "fp8",
            "--port",
            str(self.port),
            "--host",
            "127.0.0.1",
        ]
        self.last_launch_argv = list(args)
        self.generator_server_path = f"vllm:{model_dir}"
        # Honour CARNOT_ARC_SERVER_LOG_DIR, the same knob the llama.cpp launch below uses. On
        # Kaggle the kernel points it at /kaggle/working so the server log survives the run --
        # in /tmp it dies with the container, which is what made three failed scored runs
        # undiagnosable until the log was copied out by hand.
        import os as _os

        log_path = (
            Path(_os.environ.get("CARNOT_ARC_SERVER_LOG_DIR", tempfile.gettempdir()))
            / f"vllm_server_{self.port}.log"
        )
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            log_path = Path(tempfile.gettempdir()) / f"vllm_server_{self.port}.log"
        lf = open(log_path, "ab")
        self._proc = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT)
        # Weights (23 GB) + torch.compile + CUDA-graph capture: measured cold boots run minutes,
        # not seconds. 240 * 5s = 20 min ceiling, generous but bounded.
        for _ in range(240):
            time.sleep(5)
            if self._vllm_healthy():
                return True
            if self._proc.poll() is not None:
                break
        tail = ""
        try:
            tail = log_path.read_bytes()[-1500:].decode(errors="replace")
        except OSError:
            pass
        self._note_server_failure(f"vllm server failed to become healthy; log tail: {tail}"[:400])
        return False

    def _vllm_healthy(self) -> bool:
        """Health for vLLM is HTTP 200 on /health with an EMPTY body.

        `_healthy()` returns `b"ok" in r.read()`, which is right for llama.cpp -- it answers
        `{"status":"ok"}`. vLLM answers 200 and no body, so that check is False forever no matter
        how healthy the server is. Kernel v32 is the proof: the server logged `Application startup
        complete`, then served 157 consecutive `GET /health` 200s while the launcher waited out its
        full 20-minute budget and reported `server_up=False`.

        Worth naming plainly, because this file already contains the lesson: the whole point of the
        backend-aware pre-flight probe is that a llama.cpp-shaped check reports the wrong thing
        about vLLM -- and a llama.cpp-shaped health check was left sitting inside the vLLM launcher
        while that probe was being written.
        """
        import urllib.request

        try:
            with urllib.request.urlopen(self._url() + "/health", timeout=3) as r:
                return 200 <= int(r.status) < 300
        except Exception:
            return False

    def _vllm_reusable(self, model_dir: str) -> bool:
        """Reuse check for a running vLLM server: same policy as `_reusable` (refuse rather than
        adopt), sourced from /v1/models because vLLM serves no /props. The served id is the model
        path when launched with --model <dir>; --served-model-name aliases it as 'm', so match
        either. max_model_len is checked with the same >= rule as the n_ctx check."""
        import json as _json
        import urllib.request

        try:
            with urllib.request.urlopen(self._url() + "/v1/models", timeout=3) as r:
                data = _json.load(r)
        except Exception:
            self.reuse_model_check = "unobserved_v1_models_unreachable"
            return True  # same fail-open as /props-unreadable in _reusable
        rows = data.get("data") if isinstance(data, dict) else None
        row = rows[0] if isinstance(rows, list) and rows else {}
        served = str(row.get("id") or "")
        root = str(row.get("root") or "")
        if served and served not in ("m",) and model_dir not in (served, root):
            self.reuse_model_check = f"refused_wrong_model observed={served} want={model_dir}"
            return False
        self.reuse_model_check = "match"
        mml = row.get("max_model_len")
        need = int(_INDUCE_WORST_CASE_PROMPT_TOKENS + self.max_tokens + 2048)
        if isinstance(mml, int) and mml > 0:
            self.observed_server_n_ctx = int(mml)
            if mml < need:
                self.reuse_n_ctx_check = f"refused_smaller_pool observed={mml} want={need}"
                return False
            self.reuse_n_ctx_check = "match" if mml == need else "larger_ok"
        else:
            self.reuse_n_ctx_check = "unobserved_props_unreachable"
        return True

    def _vllm_raw_completion(self, payload: dict) -> dict:
        """POST the llama.cpp-shaped raw payload to vLLM's /v1/completions and return the answer
        RESHAPED to llama.cpp's response contract, so `_record_completion_diagnostics` and every
        downstream truncation check keep working unchanged: `content` from choices[0].text,
        `stop_type` mapped from finish_reason (length->limit, stop->eos), and
        `timings.predicted_n` from usage.completion_tokens."""
        import json as _json
        import urllib.request

        body = _json.dumps(
            {
                "model": "m",
                "prompt": payload.get("prompt", ""),
                "max_tokens": int(payload.get("n_predict") or self.max_tokens),
                "temperature": float(payload.get("temperature", 0.0)),
                **({"seed": payload["seed"]} if "seed" in payload else {}),
                **({"stop": payload["stop"]} if payload.get("stop") else {}),
            }
        ).encode()
        req = urllib.request.Request(
            self._url() + "/v1/completions", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            d = _json.load(r)
        ch = (d.get("choices") or [{}])[0]
        fr = str(ch.get("finish_reason") or "")
        usage = d.get("usage") or {}
        return {
            "content": str(ch.get("text") or ""),
            "stop_type": {"length": "limit", "stop": "eos"}.get(fr, fr),
            "truncated": False,
            "timings": {"predicted_n": usage.get("completion_tokens")},
        }

    def _ensure_server(self) -> bool:
        if _vllm_backend_active():
            return self._ensure_vllm_server()
        if self._healthy():
            if self._reusable():
                return True  # reuse an already-running server (loaded model)
            # A live server on our port is unusable: either a SMALLER context pool than this
            # proposer needs, or (since 2026-07-28) a DIFFERENT MODEL than the one we are pinned
            # to. Do not adopt it and do not fight it for the port -- move to a fresh port and
            # launch our own, so a stale/foreign server cannot silently degrade this run.
            # Recorded on its OWN channel, NOT via _note_server_failure. A port relaunch is a
            # configuration event, not a generator failure: routing it into n_server_failures
            # would flip llm_on_row_valid to False for a run whose generator then worked
            # perfectly, i.e. over-firing the very gate that has to stay trustworthy.
            #
            # The message names WHICH check refused, because the two have different operator
            # remedies (raise the pool / kill the stale server) and a single generic "unusable"
            # line would send the reader to the wrong one.
            self.reuse_refusals.append(
                f"port {self.port} unusable: model_check={self.reuse_model_check} "
                f"n_ctx_check={self.reuse_n_ctx_check} "
                f"(observed model={self.observed_server_model_path}, "
                f"n_ctx={self.observed_server_n_ctx}; required n_ctx {self.n_ctx}); "
                "relaunched on a fresh port -- reusing it would silently restore the "
                "concurrency fault or run the wrong model"
            )
            self.port = _free_port()
        path = self.model_path or _resolve_gguf(
            self.repo_substr
        )  # explicit path (Kaggle bundle) else cache
        # Resolve the server + launch env at LAUNCH time so the opt-in 3090 guard sees current GPU state
        # (CARNOT_ARC_GENERATOR_CUDA_GPU=<idx> -> CUDA build pinned to that card iff it has headroom).
        server, launch_env = _generator_server_and_env(self.ffn_cpu_layers, self.mtp)
        # LIFT the placement decisions onto the instance. `_generator_server_and_env()` and
        # `_default_ffn_cpu_layers()` are module functions with nowhere to record, so an artifact
        # could not previously answer "why was this run slow / why was the LLM off": the guard
        # refusal and the iGPU fall-back existed only as a control-flow branch. Copied (not
        # aliased) so a later launch cannot retroactively edit what an earlier artifact recorded.
        self.generator_selection_log = list(GENERATOR_SELECTION_LOG)
        self.generator_server_path = str(server)
        if not path or not server.exists():
            return False  # GPU enforcement: no CPU fallback
        args = [
            str(server),
            "-m",
            path,
            "-ngl",
            str(
                self.n_gpu_layers
            ),  # 999=all-GPU (default); lower spills weights to system RAM (frees VRAM)
            "-c",
            str(self.n_ctx),
            "--port",
            str(self.port),
            "--host",
            "127.0.0.1",
        ]
        # NATIVE llama.cpp MTP SPECULATIVE DECODING. `--model-draft` MUST be the SEPARATE draft
        # head GGUF, never `path` (the main weights).
        #
        # THIS LINE USED TO READ `--model-draft <path>` -- the main model -- and that is the single
        # most dangerous configuration in this file, because it does not fail. llama.cpp emits
        #     W llama_init_from_model: context type MTP requested but model doesn't contain MTP layers
        #     W common_speculative_init: no implementations specified for speculative decoding
        # and then serves normally with speculation SILENTLY DISABLED: /health returns 200,
        # generation is correct, and the only observable difference is tok/s. Directly reproduced
        # 2026-07-28. So "MTP is on" was unfalsifiable from inside the process, which is why it is
        # now recorded on the instance instead.
        #
        # HEAD ABSENT -> MTP OFF, LOUDLY. If the head cannot be resolved we drop the flags entirely
        # rather than passing something bogus. Dropping them costs the ~1.4x speedup; passing a
        # bogus draft costs the same speedup AND leaves a run that believes it had MTP. The
        # `-- reason --` is written to `mtp_disabled_reason` and to the audit channel, because the
        # remedy (attach the head dataset / set CARNOT_ARC_MTP_GGUF_PATH) is not guessable from a
        # missing speedup.
        self.last_mtp_draft_path = ""
        self.mtp_disabled_reason = ""
        if self.mtp:
            # SELF-DRAFTING MODEL FIRST. If the main weights declare their own MTP layers, the
            # correct launch is `--spec-type draft-mtp` with NO `--model-draft` at all. This is a
            # different shape from the head case below, not a fallback to it: handing such a model
            # a `--model-draft` -- even a real head -- is not what it wants, and handing it its own
            # path is the silent-degradation trap this block exists to prevent. Checked before head
            # resolution so a stale head left under /kaggle/input cannot outrank a baked-in head.
            if _gguf_declares_baked_mtp(path):
                args += ["--spec-type", "draft-mtp"]
                self.last_mtp_draft_path = "<baked-in>"
                _note_generator_selection(
                    f"MTP: {Path(path).name} declares its own MTP layers; launching "
                    "--spec-type draft-mtp with no separate draft head."
                )
            elif (
                (head := (self.mtp_model_path or _resolve_mtp_head()))
                and Path(head).exists()
                and _is_mtp_head_file(Path(head).name)
            ):
                args += ["--spec-type", "draft-mtp", "--model-draft", head]
                self.last_mtp_draft_path = str(head)
            else:
                self.mtp_disabled_reason = (
                    f"mtp=True but no usable MTP draft head was resolved (looked for "
                    f"{ARC_LIVE_GENERATOR_MTP_HEAD_FILENAME!r}; got {head!r}). Launching WITHOUT "
                    "speculative decoding rather than passing the main weights as the draft -- "
                    "llama.cpp would accept that, warn, and serve with speculation silently "
                    "disabled. Set CARNOT_ARC_MTP_GGUF_PATH, or attach the "
                    "iancblenke/carnot-gemma4-31b-mtp-head dataset on Kaggle."
                )
                _note_generator_selection(self.mtp_disabled_reason)
        # 8-bit KV cache doubles usable context and is near-lossless. `CARNOT_ARC_KV_QUANT`
        # (2026-08-11, REQ-ARC-WMTE-6253) lets an operator pick another type without editing
        # source, because q8_0 was chosen when every card held 16-24 GB and the scored card now
        # holds 96 GB, where f16 KV is affordable. Set it to "f16" to test that. Set it to "none"
        # to drop the flags entirely. Unset keeps today's q8_0 behaviour exactly.
        _kv = _kv_quant_for_launch(self.kv_quant)
        if _kv and _kv.lower() != "none":
            args += ["--cache-type-k", _kv, "--cache-type-v", _kv]
        self.last_kv_quant_used = _kv if (_kv and _kv.lower() != "none") else None
        # OPT-IN dense-FFN offload to system RAM. Appended ONLY when the operator asked for it, so
        # the default launch argv is unchanged. Recorded on the instance because a flag that is
        # accepted and silently ignored is worse than no flag: `last_ffn_cpu_override` lets a test
        # (and an artifact) assert the regex actually reached the process argv, and the server's
        # own stderr log -- captured a few lines below -- prints `tensor overrides` on the load
        # path, which is the independent confirmation that it was ACTED on and not just parsed.
        self.last_ffn_cpu_override = _ffn_cpu_override_regex(self.ffn_cpu_layers)
        if self.last_ffn_cpu_override:
            args += ["-ot", self.last_ffn_cpu_override]
        # LAYER SPLIT. Derived from `launch_env` -- the env this very launch will use -- so the
        # `-ts` ratio cannot be sized for a split the headroom guard already refused. [] on a
        # single card and on the scored path (env=None, all devices visible, llama.cpp already
        # layer-splits by default there). Recorded on the instance for the same reason as
        # `last_ffn_cpu_override`: a flag that is accepted and ignored is worse than no flag, and
        # an artifact must be able to answer "was this run split?" without re-deriving it.
        self.last_split_args = tuple(_split_args_for_env(launch_env))
        if self.last_split_args:
            args += list(self.last_split_args)
        if self.extra_server_args:  # e.g. ("-fit", "off") -- see field docstring
            args += list(self.extra_server_args)
        # env=launch_env: None inherits the ambient env (legacy iGPU path); a dict pins CUDA_VISIBLE_DEVICES.
        #
        # STDERR IS CAPTURED TO A FILE, NOT DISCARDED (2026-07-27). It used to go to DEVNULL, and
        # that single choice is why the K>=2 concurrency fault stayed invisible for months: the
        # server DIAGNOSES itself on stderr and we threw the diagnosis away.
        #
        # WHAT WE WERE DISCARDING, concretely. llama.cpp's decode-failure handler
        # (tools/server/server-context.cpp:3200-3230) does NOT raise -- it checks the RETURN CODE of
        # llama_decode() and logs `SRV_ERR("%s i = %d, n_batch = %d, ret = %d")`. That `ret` is the
        # discriminator between our failure modes and nothing else distinguishes them:
        #     ret == 1  -> "Context size has been exceeded."  (mode A: pool exhaustion, survivable)
        #     ret == -1 -> "Invalid input batch."
        #     ret <  -1 -> "Compute error."
        #     ret == 2  -> explicitly UNHANDLED upstream (`// TODO: handle ret == 2 (abort)`)
        # A hard GGML_ASSERT abort (mode B, the server DIES) also prints only to stderr. So with
        # DEVNULL, mode A and mode B are indistinguishable from the client -- which is exactly the
        # state the 2026-07-27 review left open ("Mode B is UNRESOLVED at 81920; needs the server's
        # stderr captured").
        #
        # Note also that the graceful path is a DIFFERENT site: the per-request admission check at
        # :2704-2712 sends a 400 ("try increasing it") BEFORE decoding. It is per-request, so it
        # cannot catch the aggregate case where K requests each fit but jointly exhaust the shared
        # kv_unified pool -- that only fails later inside llama_decode, as a 500. Concurrency
        # escapes the graceful path by construction, and the 500 handler then errors EVERY
        # processing slot (`for (auto & slot : slots) ... send_error`), which is why we measure
        # 2/2 at K=2 and 4/4 at K=4 rather than a single victim.
        #
        # A FILE, NOT A PIPE, DELIBERATELY. subprocess.PIPE with no reader deadlocks the server the
        # moment the OS pipe buffer fills (~64KB) -- llama-server is chatty enough to hit that
        # during a long run, and the hang would look exactly like the fault we are diagnosing.
        # Writing to a file has no such backpressure. Best-effort: if the log cannot be opened we
        # fall back to DEVNULL rather than failing the launch, because losing diagnostics must
        # never cost us the generator itself.
        # Local imports: this module has NO module-level `import os` (every user imports it inside
        # its own function) and no `tempfile` at all. Ruff passed on the module-attribute version
        # anyway -- a green lint is not evidence the code runs, so this is imported where it is used
        # and the path below is exercised by a real test rather than trusted.
        import os
        import tempfile

        # Keep the PREVIOUS launch's log reachable before this launch replaces the
        # path. The death evidence for a killed-then-relaunched server lives in the
        # OLD log; reading only the fresh one misses exactly the flagship case
        # (REQ-ARC-WMTE-6670, adversarial-review finding 5, 2026-08-23).
        if getattr(self, "_stderr_log_path", None):
            self._prev_stderr_log_path = self._stderr_log_path
        self._stderr_log_path = None
        _err_sink = subprocess.DEVNULL
        try:
            log_dir = (
                Path(os.environ.get("CARNOT_ARC_SERVER_LOG_DIR", tempfile.gettempdir()))
                / "carnot_llama_server_logs"
            )
            log_dir.mkdir(parents=True, exist_ok=True)
            self._stderr_log_path = log_dir / f"llama_server_p{self.port}_{int(time.time())}.log"
            _err_sink = open(self._stderr_log_path, "ab", buffering=0)  # noqa: SIM115
        except OSError:
            self._stderr_log_path = None
            _err_sink = subprocess.DEVNULL
        # A live child from an earlier launch attempt on this instance (still loading past a
        # prior wait budget, or refused above for reuse) is about to be dropped by the assignment
        # below. Stop it first -- see `_terminate_stale_proc` for why an unreferenced live server
        # is worse than a failed launch.
        self._terminate_stale_proc("terminated before launching a replacement llama-server")
        self.last_launch_argv = tuple(args)
        self._proc = subprocess.Popen(
            args, stdout=subprocess.DEVNULL, stderr=_err_sink, env=launch_env
        )
        load_wait_attempts = max(90, int(self.timeout / 2))  # large full-precision models (e.g.
        # a 62GB BF16 GGUF) can take far longer than the 180s the fixed 90-attempt budget allows
        # CAPPED at 300 attempts (10 min) since 2026-08-21: this budget is for MODEL LOAD time,
        # not generation, and it was accidentally coupled to the request timeout. When the
        # timeout default moved 600 -> 2400 with the Qwen3.8 pin (REQ-ARC-WMTE-6620), an
        # uncapped derivation would have turned a crash-at-startup server into a silent
        # 40-minute stall. 300 preserves the old ceiling (timeout 600 -> 300 attempts) exactly.
        load_wait_attempts = min(300, load_wait_attempts)
        for _ in range(load_wait_attempts):
            if self._healthy():
                self._verify_mtp_engaged()
                return True
            time.sleep(2)
        # WAIT EXHAUSTED: the server never answered /health in time. It may still be loading, or
        # it may be permanently stuck -- either way, returning failure here used to leave it
        # running, referenced only by `self._proc`, one call away from being silently orphaned
        # the next time this method Popens a replacement. Stop it now instead of hoping a later
        # call remembers to.
        self._terminate_stale_proc("terminated: never became healthy within the wait budget")
        return False

    # The POSITIVE marker llama.cpp prints when `--spec-type draft-mtp` is genuinely wired up. Read
    # off a real successful launch's stderr (2026-07-28), not guessed:
    #     I common_speculative_impl_draft_mtp: adding speculative implementation 'draft-mtp'
    #     I srv    load_model: speculative decoding context initialized
    # The first is specific to the draft-mtp implementation, so it is the one matched.
    _MTP_ENGAGED_MARKER = "common_speculative_impl_draft_mtp: adding speculative implementation"
    # ...and the WARNINGS it prints instead when the draft is accepted but unusable. Matched only to
    # give the operator the runtime's own words in the failure note; absence of these is NOT
    # evidence of success (that is the whole trap), so the positive marker is what decides.
    _MTP_REFUSED_MARKERS = (
        "context type MTP requested but model doesn't contain MTP layers",
        "no implementations specified for speculative decoding",
    )

    def _verify_mtp_engaged(self) -> None:
        """After the server reports healthy, CHECK ITS OWN STDERR for the MTP positive marker.

        WHY THIS EXISTS, AND WHY ITS ABSENCE WAS THE SHARPEST GAP IN THIS FILE. This module already
        states the doctrine, twice, in capitals: "never conclude MTP is enabled from a healthy
        server or an absent error; a misconfigured MTP is indistinguishable from a working one
        except by tok/s". It then did not apply that doctrine to its OWN server. The only place the
        marker was ever grepped was `scripts/kaggle/submission_kernel/main.py`'s pre-flight probe --
        a DIFFERENT process, on a different port, torn down before the agent starts. Inside this
        class, `last_mtp_draft_path` recorded only what was PASSED to `--model-draft`, which is a
        fact about our argv and not about the runtime's behaviour: the entire failure mode is that
        llama.cpp ACCEPTS a draft it cannot use.

        So `mtp=True` + a resolvable head + a healthy server was treated as "MTP is on" by exactly
        the inference the file forbids. Now `last_mtp_engaged` records what the runtime SAID.

        THREE-VALUED ON PURPOSE. `None` means "could not determine" -- stderr went to DEVNULL
        because the log directory was unwritable, so there is nothing to read. That is genuinely
        different from `False` ("we read the log and speculation is NOT engaged"), and collapsing
        them would either invent a failure or, worse, let an unreadable log read as success. A
        `False` is written loudly to the audit channel; a `None` is recorded quietly, because a
        missing log is not evidence of a broken launch.

        NEVER RAISES. An audit channel that can kill the generator it audits is worse than no audit
        channel -- the whole point is to make a degraded run visible, not to convert it into a
        crashed one.
        """
        if not self.mtp or not self.last_mtp_draft_path:
            # MTP was not requested, or was already declined with a recorded reason. Nothing to
            # verify; leave `last_mtp_engaged` at its "not applicable" default of None.
            return
        try:
            log_path = getattr(self, "_stderr_log_path", None)
            if not log_path or not Path(log_path).exists():
                self.last_mtp_engaged = None
                self.mtp_engaged_evidence = (
                    "server stderr was not captured (log dir unwritable), so whether speculative "
                    "decoding engaged CANNOT be determined from this launch. Not treated as "
                    "success: an absent error is exactly what a silently-disabled MTP looks like."
                )
                return
            text = Path(log_path).read_text(errors="replace")
            if self._MTP_ENGAGED_MARKER in text:
                self.last_mtp_engaged = True
                self.mtp_engaged_evidence = self._MTP_ENGAGED_MARKER
                return
            self.last_mtp_engaged = False
            refused = [m for m in self._MTP_REFUSED_MARKERS if m in text]
            self.mtp_engaged_evidence = (
                f"MTP was requested with --model-draft {self.last_mtp_draft_path!r} and the server "
                f"is HEALTHY, but its stderr does NOT contain {self._MTP_ENGAGED_MARKER!r}. "
                "Speculative decoding is therefore NOT engaged and this run gets no speedup, while "
                "otherwise behaving exactly like a working one."
                + (f" The runtime's own warnings: {refused}." if refused else "")
            )
            _note_generator_selection(self.mtp_engaged_evidence)
        except Exception:
            self.last_mtp_engaged = None
            self.mtp_engaged_evidence = "mtp verification raised; treated as could-not-determine"

    def _engine_defects(self, code: str, transitions: Optional[Sequence[Any]]) -> list[str]:
        """The MECHANICAL defect kinds in an emitted engine, or [] -- never a quality judgement.

        Returning [] means nothing detectable is broken. It does NOT mean the engine models the
        game: the out-of-sample measurement behind this wiring found a game where every arm
        produced clean-but-wrong engines on 19 of 19 held-out transitions. `usable` is not `good`,
        and this gate only ever claims the former.

        The import is LOCAL, not module-scope, and deliberately so: `arc_engine_static_validation`
        must not become a hard import of the world-model module, because a failure to import it
        would take the induce path down entirely in order to run an optional quality gate. Any
        exception here is swallowed to [] -- i.e. "accept, as before". A defect DETECTOR that can
        break the thing it is checking is worse than no detector.
        """
        try:
            from . import arc_engine_static_validation as _sv

            defects = _sv.validate_engine_code(
                code,
                transitions=list(transitions) if transitions else None,
                stop_type=self.last_stop_type,
                required=("engine",),
                # The budget the request actually carried (pool-clamped), so truncation-at-
                # budget detection judges against what was asked, not the configured cap.
                budget=(
                    self.last_requested_n_predict
                    if self.last_requested_n_predict > 0
                    else self.max_tokens
                ),
            )
            # OPT-IN INERTNESS REJECTION (2026-08-01, CARNOT_ARC_INDUCE_REJECT_INERT, DEFAULT
            # OFF). Deliberately OUTSIDE `validate_engine_code` rather than folded into it, for
            # two reasons. First, `validate_engine_code` is the definition an A/B measures
            # usable-engine yield WITH; moving the flag inside it would make the treatment and
            # the outcome the same object and the measurement circular. Second, the ordering is
            # load-bearing: it runs only when the mechanical checks came back CLEAN, so an
            # engine that hangs or raises is reported as hanging or raising -- its actual
            # defect -- and is never relabelled "inert" by a probe that could not run it.
            if _reject_inert_engines() and not defects and transitions:
                inert = _sv.engine_inertness_defect(code, list(transitions))
                if inert is not None:
                    defects = [inert]
            return sorted({d.kind for d in defects})
        except Exception:
            return []

    def _goal_defects(self, code: str, transitions: Optional[Sequence[Any]]) -> list[str]:
        """The MECHANICAL defect kinds in an emitted `is_level_complete`, or [] -- never a
        quality judgement. OPT-IN via `_goal_defect_check_on()`; returns [] when off, so with
        the flag unset this method is behaviourally identical to not existing.

        The three kinds, and why each needs no knowledge of the game:

          * `goal_missing_return` -- the function body contains no `return` on any path, so it
            yields None. Purely syntactic (AST), no execution, no evidence required. This is
            the `D_NO_PREDICATE` class: a bare `import numpy as np` body, a truncated
            `thought // no_think` fragment.
          * `goal_raises` -- raises on an observed frame. Detected by RUNNING it on grids the
            agent has already seen, which is the same dry-run principle `_engine_defects`
            already applies to `engine()`.
          * `goal_constant` -- returns the SAME value on every observed frame. This is the big
            one: 34 unconditional `return False` (`A_DECLINED`, four of which say in their own
            docstring that they had been told nothing about the goal), plus every whole-board
            trope that happens to be false everywhere the agent has been.

        WHAT `goal_constant` DOES NOT CLAIM. Constant-over-observed-frames is not "wrong". A
        correct win condition on a game the agent has never won IS false on every frame it has
        seen -- that is what not having won means. So this check cannot distinguish a lazy
        `return False` from a correct-but-unreached predicate, and it is not trying to: it is
        a check on whether the answer carries INFORMATION for the search that is about to use
        it, and a predicate constant across everything the agent has observed carries none
        either way. The cost of being wrong here is bounded to one extra sample.

        SANDBOXING. The probe runs LLM-written code, so every call is wrapped: an exception is
        a DEFECT (`goal_raises`), and any failure of the machinery itself falls through to []
        -- "accept, as before". A defect detector that can break the thing it is checking is
        worse than no detector.

        THE HANG, named because this flag INTRODUCES the exposure rather than inheriting it.
        The shipped induce path never executes `is_level_complete`; with this flag on, it does.
        So a predicate containing an unbounded loop would hang induction where it previously
        would not. Two bounds, neither perfect and both stated:

          * a SIGALRM watchdog around the whole probe. It is best-effort by construction --
            `signal.setitimer` only works on the main thread of the main interpreter and raises
            otherwise, and the induce path is sometimes threaded. The failure is safe in the
            right direction: setting the alarm is itself inside the try, so a thread that
            cannot arm it proceeds unguarded rather than refusing to check, and a fired alarm
            lands in the same `except` that returns [] -- accept, as before.
          * the probe grid count is CAPPED (`_GOAL_PROBE_MAX_GRIDS`). Constancy needs two
            distinct answers, not every frame, so a cap costs almost no detection power while
            making the cost of the check bounded and roughly constant across games. Without it
            a 33-transition window would probe 66 grids on every candidate.

        `_engine_defects` carries the same hang exposure for `engine()` and does not bound it;
        that is a pre-existing gap, not one this method should silently inherit.
        """
        if not _goal_defect_check_on():
            return []
        import ast

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []  # the pre-existing parse gate above already owns this case
        fn = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "is_level_complete":
                    fn = node  # LAST definition wins, mirroring Python's own binding
        if fn is None:
            return []  # the `required` check above already owns this case
        defects: list[str] = []
        if not any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(fn)):
            defects.append("goal_missing_return")

        grids = []
        for t in list(transitions or []):
            for attr in ("grid", "next_grid"):
                g = getattr(t, attr, None)
                if g is not None:
                    grids.append(g)
        if not grids:
            # NO EVIDENCE IS NOT A CLEAN BILL. Without frames the runtime probes cannot run,
            # so only the syntactic defect is reported -- and it is reported, rather than
            # returning [] wholesale, because a body with no return is defective whether or
            # not anything was observed.
            return sorted(set(defects))

        seen: set = set()
        raised = False
        _alarm_armed = False
        try:
            # Best-effort watchdog; see the docstring's THE HANG note for why it is inside the
            # try and why failing to arm it must not refuse the check.
            import signal as _signal

            try:
                _signal.signal(_signal.SIGALRM, _goal_probe_timeout)
                _signal.setitimer(_signal.ITIMER_REAL, _GOAL_PROBE_TIMEOUT_S)
                _alarm_armed = True
            except (ValueError, OSError, AttributeError):
                _alarm_armed = False
            ns: dict = {}
            exec(compile(code, "<induced_goal>", "exec"), ns)  # noqa: S102
            probe = ns.get("is_level_complete")
            if probe is None:
                return sorted(set(defects))
            for g in _goal_probe_sample(grids):
                try:
                    v = probe(np.asarray(g))
                except _GoalProbeTimeout:
                    raise  # the watchdog, not the predicate: do NOT swallow it as goal_raises
                except Exception:  # noqa: BLE001
                    raised = True
                    continue
                try:
                    seen.add(bool(v))
                except Exception:  # noqa: BLE001
                    raised = True
        except Exception:  # noqa: BLE001
            # Includes _GoalProbeTimeout. A probe we could not finish is NOT evidence of a
            # defect, so this returns "accept, as before" rather than inventing a verdict.
            return sorted(set(defects))
        finally:
            if _alarm_armed:
                import signal as _signal

                _signal.setitimer(_signal.ITIMER_REAL, 0)
        if raised:
            defects.append("goal_raises")
        if len(seen) <= 1:
            defects.append("goal_constant")
        return sorted(set(defects))

    def generate(
        self,
        prompt: str,
        required: tuple = ("engine", "is_level_complete"),
        validate=None,
        tries: int = 3,
        *,
        codeonly_eligible: bool = False,
        engine_transitions: Optional[Sequence[Any]] = None,
    ) -> tuple[bool, str]:
        """Generic GPU-server completion: returns (True, code) where `code` contains every
        `def <name>` in `required`, PARSES, and (if `validate` is given) passes the
        runtime check `validate(code) -> bool`. Retries on the iGPU (fast). This is the
        gap-filler entry point: the LLM writes a FOCUSED component (a goal_distance
        heuristic, a state_key, a verifier invariant) — not a full solver. `validate` lets
        the caller reject runtime-buggy code (e.g. a heuristic that returns None).

        `engine_transitions` (induce path only) supplies the observed transitions the emitted
        `engine()` is DRY-RUN against before it is accepted. Without them the defect check is
        static-only (truncation, syntax, missing `return`), which is still the majority of what
        the shipped path was accepting: 22 of 36 measured attempts returned a mechanically
        defective candidate as success, `missing_return` being the single largest kind.
        """
        import ast
        import json as _json
        import os
        import urllib.request

        self.n_completion_calls += 1
        if not self._ensure_server():
            msg = (
                # REQ-ARC-WMTE-6670: name the EFFECTIVE model (model_path wins over
                # the repo pin), so this message cannot point at a retired pin while
                # different weights are what actually loaded.
                f"GPU llama-server failed for {self._effective_model_label()}; SOTA models "
                "must run on GPU (no CPU fallback)"
            )
            self._note_server_failure(msg)
            return False, msg
        # L2 induction truncation fix (proto_l2_code_only_prefix, 2026-06-25). Scope to the INDUCE
        # call ONLY (codeonly_eligible, set True solely by induce->_gen_to_file). refactor() also
        # routes through _gen_to_file with the same `required` tuple, but it is a REASONING task
        # (debug BEFORE/PREDICTED/OBSERVED mismatches) that must NOT be told to "skip all reasoning";
        # keying on codeonly_eligible (set True ONLY by the induce paths) keeps refactor and
        # gap-fillers untouched. It is NOT keyed on `required` because the focused split-induce calls
        # request just ("engine",) or ("is_level_complete",) yet still need the code-only path.
        # When on: prepend the code-only directive + an opened fence, and add a stop-sequence on the
        # closing fence so the model emits ONLY the code (no win-state CoT). DEFAULT ON (2026-06-25
        # operator directive): a strict improvement (emits valid code in ~10s where the unpatched
        # path truncates to 0 code at 450s); opt out with CARNOT_ARC_CODEONLY_INDUCE=0.
        # THINK MODE (lever #6 -- see induce_think_on()). Takes priority over codeonly: the two
        # are mutually exclusive by construction (codeonly exists to SUPPRESS reasoning; think
        # mode exists to ALLOW it), so an operator flipping CARNOT_ARC_INDUCE_THINK does not need
        # to also touch CARNOT_ARC_CODEONLY_INDUCE. STALE UNTIL 2026-08-08: this said "default
        # OFF" and described think-off as the shipped default; the operator flipped think ON that
        # day (exp6199 induction-quality evidence), so THIS branch (not the codeonly branch below)
        # is now the shipped default path.
        _think_on = induce_think_on() and codeonly_eligible
        _codeonly = (
            (not _think_on)
            and (os.environ.get("CARNOT_ARC_CODEONLY_INDUCE", "1") != "0")
            and codeonly_eligible
        )
        _stop_seq = ["```"] if _codeonly else None
        # KNOWN LATENT BUG, FLAGGED 2026-08-08 (adversarial review), NOT FIXED: `_codeonly` and
        # `use_chat_template` are independent switches, so `_codeonly=True` with
        # `self.use_chat_template=True` is reachable in principle (currently NOT on the live
        # gemma-4-31B path -- `use_chat_template` defaults False and nothing sets it True there;
        # `_think_on` being on forces `_codeonly=False` anyway). If it ever IS reached: the
        # pre-opened fence below primes a RAW-COMPLETION continuation (the model resumes
        # mid-code-block), which is meaningless under a chat template (the fence sits inside the
        # USER message; the model starts a FRESH assistant turn and typically emits its OWN
        # opening fence first). `stop=["```"]` then fires on that opening fence, truncating the
        # response before any code. Do not flip `use_chat_template=True` for a codeonly-eligible
        # call without redesigning this interaction first -- see ops/known-issues.md.
        if _codeonly:
            prompt = _L2_CODEONLY_DIRECTIVE + prompt + "\n```python\n"
        elif _think_on:
            pass  # no directive, no pre-opened fence: let the model reason before it answers
        elif self.no_think_prefix:  # suppress hybrid-thinking CoT so the model emits code directly
            prompt = self.no_think_prefix + prompt
        # DEFECT-REJECTION + PLAIN RE-ASK (2026-07-31). Armed only where it was measured: the
        # code-only induce path emitting an `engine`. Gap-fillers (a goal_distance heuristic, a
        # state_key) are a different artifact with a different contract and are left alone.
        # THE MEASURED CALL SHAPE. Phase 1 sent ENGINE induce prompts and nothing else, so this
        # is the scope both 2026-07-31 interventions are armed on. Named once rather than
        # spelled twice so the penalty and the defect gate cannot silently drift apart -- and
        # named separately from `_defect_check_on` below so that narrowing one later does not
        # move the other by accident.
        _engine_induce_call = bool(codeonly_eligible) and "engine" in tuple(required)
        _defect_check_on = _engine_induce_call
        _reasks_left = _induce_defect_reasks() if _defect_check_on else 0
        # OPT-IN GOAL DEFECT REJECTION (2026-08-01, CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK,
        # DEFAULT OFF). Scoped to `"is_level_complete" in required`, which is DELIBERATELY a
        # wider gate than the engine's: it covers BOTH the combined induce call (where ~84% of
        # live inductions happen) AND the focused goal-only call in `_split_induce`, which the
        # engine gate cannot reach at all because `"engine"` is not in its `required`. That
        # unreachability is the defect this closes, so keying on the same token would have
        # reproduced it. `codeonly_eligible` is still required so that gap-fillers and the
        # reasoning-mode refactor path stay untouched.
        _goal_check_on = bool(codeonly_eligible) and "is_level_complete" in tuple(required)
        _goal_reasks_left = _goal_defect_reasks() if _goal_check_on else 0
        _reask_suffix = ""
        last = ""
        # ATTEMPT BUDGET. `tries` is the CONTENT budget -- attempts spent on a reply that is
        # unusable (no code block, missing `def`, syntax error). With
        # `_defect_gate_owns_attempts()` OFF, `_budget` never grows and this loop is exactly
        # `for attempt in range(tries)`; with it ON, a defect re-ask GRANTS an attempt rather
        # than consuming one. See that resolver for the incident.
        _own_attempts = _defect_gate_owns_attempts()
        _budget = int(tries)
        # POOL-CLAMPED COMPLETION BUDGET (REQ-ARC-WMTE-6620). llama-server admission counts
        # prompt + n_predict against the shared pool, so asking for more than the RUNNING
        # server's pool has room for is refused outright, not truncated. Read the pool once
        # per generate() call (one /props round-trip on a server _ensure_server just proved
        # up); clamp only when the pool is real and smaller than the configured budget.
        _n_predict = _pool_clamped_n_predict(int(self.max_tokens), self.observed_n_ctx())
        self.last_requested_n_predict = int(_n_predict)
        attempt = -1
        while True:
            attempt += 1
            if attempt >= _budget:
                break
            _payload = {
                "prompt": prompt + _reask_suffix,
                "n_predict": _n_predict,
                "temperature": 0.2 + 0.1 * attempt,
                "cache_prompt": True,
            }
            # OPT-IN SAMPLING OVERRIDE (2026-08-16). Absent all three env vars this block is a
            # no-op and the payload is byte-identical to before -- the same default-off contract
            # as `sampling_seed`, and for the same reason: changing how the SCORED agent samples
            # is a behaviour change, not a measurement change.
            #
            # WHY IT EXISTS. The ladder above is near-greedy (0.2/0.3/0.4) and we send no top_p,
            # top_k or penalties at all. Qwen3's own model card warns that near-greedy decoding on
            # their thinking builds produces ENDLESS THINKING REPETITION, and recommends
            # temperature 0.6 / top_p 0.95 / top_k 20. Endless thinking is exactly the observed
            # failure -- Qwen3.8 overran its budget on 5 of 5 tu93 draws while a plain call on the
            # same prompt terminates at ~41k tokens. That is external evidence, not measured here,
            # which is precisely why this ships as an opt-in probe rather than a default change.
            #
            # The +0.1 ladder's original job -- decorrelating retries -- is now done by
            # `sampling_seed(attempt)`, so pinning a constant temperature no longer costs
            # diversity.
            _t_override = os.environ.get("CARNOT_ARC_INDUCE_TEMPERATURE")
            if _t_override:
                try:
                    _payload["temperature"] = float(_t_override)
                except ValueError:
                    pass  # malformed env must never take down a live episode
            for _env, _key, _cast in (
                ("CARNOT_ARC_INDUCE_TOP_P", "top_p", float),
                ("CARNOT_ARC_INDUCE_TOP_K", "top_k", int),
            ):
                _v = os.environ.get(_env)
                if _v:
                    try:
                        _payload[_key] = _cast(_v)
                    except ValueError:
                        pass
            # See _INDUCE_REPEAT_PENALTY: breaks the decode-level repetition loop that is the
            # dominant induce failure. Scoped to the ENGINE induce prompt -- the same condition
            # as the defect gate, and the only prompt shape Phase 1 measured. 1.0 restores the
            # old payload byte-for-byte.
            if _engine_induce_call:
                _rp = _induce_repeat_penalty()
                if _rp != 1.0:
                    _payload["repeat_penalty"] = _rp
                    _payload["repeat_last_n"] = _INDUCE_REPEAT_LAST_N
            # OPT-IN determinism. Absent CARNOT_ARC_GENERATOR_SEED this adds nothing and the
            # payload is byte-identical to before. See `sampling_seed` for why the default must
            # stay off and why the seed varies with `attempt`.
            _seed = self.sampling_seed(attempt)
            if _seed is not None:
                _payload["seed"] = _seed
            if _stop_seq:
                _payload["stop"] = _stop_seq
            body = _json.dumps(_payload).encode()
            try:
                # `or _think_on`: think mode NEEDS the chat endpoint regardless of the instance's
                # own use_chat_template setting -- gemma's native thought channel is split into
                # reasoning_content only on /v1/chat/completions (exp5764), never on raw
                # /completion. With _think_on False (the shipped default) this is exactly
                # `self.use_chat_template` and nothing here changes.
                if self.use_chat_template or _think_on:
                    # OpenAI /v1/chat/completions -> server applies the GGUF's embedded chat template
                    # (Qwen3.6/ThinkingCap need the assistant-turn structure; REQ-ARC-WMTE-5725).
                    # CORRECTED 2026-08-08 (adversarial review): repeat_penalty/repeat_last_n were
                    # already computed into _payload above (when _engine_induce_call applies) but
                    # never forwarded here, so every call routed through this branch -- which
                    # INCLUDES every think-mode call, since `_think_on` forces this branch
                    # regardless of `use_chat_template` -- silently ran without the repetition
                    # control the raw /completion branch below applies to the identical prompt.
                    # `.get(...)` reads None on the (common) case where _engine_induce_call is
                    # False or _induce_repeat_penalty() returned 1.0, matching _chat_complete_
                    # request's own no-op default.
                    _response, text = self._chat_complete_request(
                        _payload["prompt"],
                        # The SAME pool-clamped budget the raw payload carries -- passing
                        # self.max_tokens here would re-open the admission refusal on the
                        # exact branch (think mode) the live path takes.
                        max_tokens=_payload["n_predict"],
                        temperature=_payload["temperature"],
                        stop=_stop_seq,
                        attempt=attempt,
                        repeat_penalty=_payload.get("repeat_penalty"),
                        repeat_last_n=_payload.get("repeat_last_n"),
                    )
                elif _vllm_backend_active():
                    # THE INDUCE PATH under the vLLM backend. Same translation as the
                    # complete_text route: /v1/completions in, llama.cpp-shaped reply out, so
                    # `_record_completion_diagnostics` and the truncation/limit diagnostics that
                    # gate every induction keep reading the fields they were written against.
                    _response = self._vllm_raw_completion(_payload)
                    text = _response.get("content", "")
                else:
                    req = urllib.request.Request(
                        self._url() + "/completion",
                        data=body,
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=self.timeout) as r:
                        _response = _json.load(r)
                    text = _response.get("content", "")
            except Exception as e:
                msg = f"local gguf (GPU server) failed: {_describe_http_failure(e)}"[:400]
                self._note_server_failure(msg)
                return False, msg
            self._record_completion_diagnostics(_response)  # MANDATORY truncation detection
            code = _extract_python(text)
            if not code and _codeonly:
                # the stop-sequence consumed the closing fence and the opener was in the prompt, so
                # the raw completion IS the code block body.
                code = text.strip()
            if not code or any(f"def {fn}" not in code for fn in required):
                _diag = ""
                if self.last_stop_type == "limit":
                    _diag += self._limit_diagnostic()
                if self.last_prompt_truncated:
                    _diag += " [PROMPT TRUNCATED -- exceeded server context window]"
                last = f"missing {required} in output{_diag}"
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
            # The candidate now parses and defines everything asked for -- which is exactly the
            # bar the shipped path stopped at, and exactly the bar 22 of 36 measured attempts
            # cleared while returning code that could not work: an `engine()` with no `return`
            # on some path yields None, and `WorldModelVerifier.score` wraps that in
            # `np.asarray(None)` rather than raising, so it degrades SILENTLY into a world model
            # that predicts nothing.
            #
            # NEVER FAILS WHERE THE OLD PATH SUCCEEDED. A defect only ever converts an accept
            # into one more ask, and only while `_reasks_left` allows; once the budget is spent
            # the candidate is accepted exactly as before. So the worst case of this block is the
            # old behaviour plus one wasted call, and a hard failure here can still only come
            # from the pre-existing parse/required/validate checks above.
            # `attempt < tries - 1` is what makes the "never fails where the old path succeeded"
            # guarantee TRUE rather than merely intended: on the final attempt a `continue` would
            # fall out of the loop into the content-failure return and convert an accept into a
            # hard False. On the last try we take the defective candidate, exactly as before.
            if _defect_check_on and _reasks_left > 0 and (attempt < _budget - 1 or _own_attempts):
                _defects = self._engine_defects(code, engine_transitions)
                if _defects:
                    # Grant rather than consume: without this the `continue` below would fall out
                    # of the loop and turn an accept into a hard failure, which is precisely what
                    # the old `attempt < tries - 1` guard existed to prevent.
                    if attempt >= _budget - 1:
                        _budget += 1
                    _reasks_left -= 1
                    _reask_suffix = _INDUCE_PLAIN_REASK_BLOCK
                    last = f"mechanically defective engine: {_defects}"
                    self.n_induce_defect_reasks += 1
                    continue
            # CORRECTION 2026-08-02, MEASURED, and it applies to the ENGINE block ABOVE as much
            # as to the goal block below. The comment above says a defect re-ask "NEVER FAILS
            # WHERE THE OLD PATH SUCCEEDED". THAT CLAIM IS FALSE, and the live A/B is what
            # falsified it: the treatment arm hard-failed induction on 17 of 21 cells against
            # 1 of 22 in control, and 0 of 21 in an A/A arm. (An earlier revision of this
            # comment said "16 of 20 against 1 of 20" -- those were interim counts written
            # while the run was still in flight; the final artifact's numbers are the ones
            # above.)
            #
            # `attempt < tries - 1` stops the LAST attempt from `continue`-ing out of the loop.
            # It does NOT stop an EARLIER re-ask from spending the attempt that would have been
            # the accept. Concretely, with `tries=3` and a candidate that is usable-but-defective
            # on attempt 0 followed by two malformed completions:
            #     flag off -> attempt 0 is ACCEPTED                       -> (True, code)
            #     flag on  -> attempt 0 is re-asked, 1 and 2 are content
            #                 failures, the loop ends                     -> (False, "unusable")
            # Reproduced deterministically against a scripted server in
            # tests/python/test_arc_goal_defect_reask_wiring.py::
            # test_reask_CAN_convert_an_accept_into_a_hard_failure.
            #
            # The re-ask is therefore NOT free: it trades a defective-but-usable artifact for a
            # fresh sample AND for the risk that no later attempt parses at all. Both defect
            # gates carry this, so the shipped engine gate's measured 13/36 -> 22/36 was obtained
            # under the same trade. The fix is to give the defect gates their own attempts rather
            # than borrowing from the content-failure retry ladder; that is a behaviour change to
            # shipped code and is deliberately NOT made here, because it was discovered by a
            # measurement that is still running and would be invalidated by changing its subject.
            #
            # The GOAL half. It runs AFTER the engine check, and that order is load-bearing:
            # on the combined call one answer carries both functions, so a re-ask regenerates
            # both. Checking the engine first means a candidate with a broken engine is
            # re-asked for its engine and reported as such, and is never relabelled a goal
            # defect. `_goal_reasks_left` is its OWN budget (see `_goal_defect_reasks`), so
            # the goal can never consume the engine's and silently degrade the engine in the
            # treatment arm only. The `attempt < tries - 1` guard is repeated verbatim for the
            # same reason it exists above: on the final attempt a `continue` would fall out of
            # the loop and convert an accept into a hard False, so the last try takes the
            # defective candidate exactly as the shipped path does.
            if (
                _goal_check_on
                and _goal_reasks_left > 0
                and (attempt < _budget - 1 or _own_attempts)
            ):
                _gdefects = self._goal_defects(code, engine_transitions)
                if _gdefects:
                    if attempt >= _budget - 1:
                        _budget += 1
                    _goal_reasks_left -= 1
                    _reask_suffix = _GOAL_PLAIN_REASK_BLOCK
                    last = f"mechanically defective goal: {_gdefects}"
                    self.n_goal_defect_reasks += 1
                    continue
            self.n_completion_ok += 1
            return True, code
        # CONTENT failure, not a liveness failure: the server answered every try, the
        # answers were unusable. Counted separately so a terse-but-alive model can never
        # read as a dead generator (and vice versa) in the liveness witness.
        self.n_content_failures += 1
        return False, f"local model code unusable after {tries} tries ({last})"

    def complete_text(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: float = 0.1,
        stop: Optional[list] = None,
    ) -> tuple[bool, str]:
        """Raw free-text/JSON completion (NOT the code-extraction path of generate()).

        WHY a separate method: generate() runs `_extract_python` + an `ast.parse` gate +
        a `def <name>` presence check, which is exactly right for inducing a world-model
        engine but WRONG for a short reasoning answer (e.g. "which archived cell is most
        promising to explore from -> reply with one integer"). complete_text returns the
        server's raw `content` string with no code gating, so callers that want JSON/an
        index/a short rationale get it directly. Reuses the SAME warm GPU server as
        generate() via _ensure_server()/_url() (no second llama-server, no CPU fallback).

        Returns (ok, text). ok=False (with a diagnostic string) when the GPU server is
        unavailable or the request errors — the caller is expected to fall back to its
        own heuristic rather than fabricate, per the no-silent-degradation discipline.
        """
        import json as _json
        import urllib.request

        self.n_completion_calls += 1
        if not self._ensure_server():
            msg = (
                # REQ-ARC-WMTE-6670: name the EFFECTIVE model (model_path wins over
                # the repo pin), so this message cannot point at a retired pin while
                # different weights are what actually loaded.
                f"GPU llama-server failed for {self._effective_model_label()}; SOTA models "
                "must run on GPU (no CPU fallback)"
            )
            self._note_server_failure(msg)
            return False, msg
        full_prompt = (self.no_think_prefix + prompt) if self.no_think_prefix else prompt
        # Same pool clamp as generate() (REQ-ARC-WMTE-6620): with the budget default now the
        # generator pin's 131,072, an unclamped fallback here would be refused at admission
        # by any local pool. Callers passing an explicit small max_tokens are unaffected.
        _n_predict = _pool_clamped_n_predict(
            int(max_tokens or self.max_tokens), self.observed_n_ctx()
        )
        self.last_requested_n_predict = int(_n_predict)
        payload = {
            "prompt": full_prompt,
            "n_predict": _n_predict,
            "temperature": float(temperature),
            "cache_prompt": True,
        }
        # Opt-in determinism on the third and last generation path (`complete_text`), so no route
        # out of this class can be nondeterministic while another is seeded.
        _seed = self.sampling_seed(0)
        if _seed is not None:
            payload["seed"] = _seed
        if stop:
            payload["stop"] = list(stop)
        body = _json.dumps(payload).encode()
        try:
            if self.use_chat_template:
                # OpenAI /v1/chat/completions applies the GGUF's embedded chat template; the
                # normalized "content" folds any split-out reasoning back in so callers/smoke
                # tests can see the <think> trace (REQ-ARC-WMTE-5725).
                _response, _ = self._chat_complete_request(
                    full_prompt,
                    max_tokens=_n_predict,
                    temperature=temperature,
                    stop=stop,
                )
            elif _vllm_backend_active():
                # vLLM serves no llama.cpp /completion; translate to /v1/completions and reshape
                # the reply to llama.cpp's contract so truncation detection is unchanged.
                _response = self._vllm_raw_completion(payload)
            else:
                req = urllib.request.Request(
                    self._url() + "/completion",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    _response = _json.load(r)
        except Exception as e:
            msg = f"local gguf (GPU server) failed: {_describe_http_failure(e)}"[:400]
            self._note_server_failure(msg)
            return False, msg
        self._record_completion_diagnostics(_response)  # MANDATORY truncation detection
        self.n_completion_ok += 1
        _content = str(_response.get("content", ""))
        if not _content.strip():
            # HTTP 200 WITH NOTHING IN IT. `n_completion_ok` deliberately still counts this --
            # it is a liveness fact (the server answered), and conflating "answered emptily" with
            # "did not answer" is the exact confusion the server/content split exists to prevent.
            # But counting it ONLY as a success would make `responses > 0` read as evidence of
            # usable output when there was none, so it is ALSO recorded as a content failure.
            # Found 2026-07-27 (adversarial review): before this, an alive server returning empty
            # strings for every call produced calls=N / responses=N / errors=0 / content_failures=0
            # -- a perfectly healthy-looking witness for a run that induced nothing.
            self.n_content_failures += 1
        return True, _content

    def _gen_to_file(
        self, game: str, prompt: str, *, codeonly_eligible: bool = False
    ) -> tuple[bool, str]:
        _guard_engine_write(E3_DIR / game)
        (E3_DIR / game).mkdir(parents=True, exist_ok=True)
        ok, code = self.generate(
            prompt,
            ("engine", "is_level_complete"),
            tries=self.tries,
            codeonly_eligible=codeonly_eligible,
        )
        if ok:
            (E3_DIR / game / "world_model.py").write_text(code)
            # REQ-ARC-WMTE-6690: retain this attempt; the canonical write above is unchanged.
            self.last_attempt_archive = _archive_engine_attempt(
                game, code, writer="gen_to_file", model=str(self.repo_substr)
            )
            return True, "local gguf (GPU server) wrote world_model.py"
        return False, code

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None

    def _write_world_model(self, game: str, code: str, note: str = "") -> tuple[bool, str]:
        _guard_engine_write(E3_DIR / game)
        (E3_DIR / game).mkdir(parents=True, exist_ok=True)
        (E3_DIR / game / "world_model.py").write_text(code)
        # REQ-ARC-WMTE-6690: retain this attempt; the canonical write above is unchanged.
        self.last_attempt_archive = _archive_engine_attempt(
            game, code, writer="write_world_model", model=str(self.repo_substr), note=note
        )
        msg = "local gguf (GPU server) wrote world_model.py"
        return True, (f"{msg} ({note})" if note else msg)

    def _goal_only_prompt(
        self,
        game: str,
        previous_level_complete_grid: Optional[np.ndarray],
        trans: Optional[list] = None,
    ) -> str:
        """A FOCUSED is_level_complete-only prompt, so the model spends its whole budget on the win
        condition (not the engine).

        CORRECTED 2026-07-29 -- this was the most damaging instance of the win-state poison. The
        block used to assert, about `previous_level_complete_grid`: "The level is COMPLETE at this
        WIN STATE grid (is_level_complete must return True here, and False elsewhere)." That grid is
        captured by `arc_competition_agent._observe_level_boundary` from the frame AFTER the level
        counter incremented, so it is the CURRENT level's OPENING BOARD. The prompt was therefore
        telling the model, as its single most emphasised fact, that a level-complete state looks like
        a freshly-laid-out board -- the exact opposite of the truth, in a prompt whose entire budget
        goes to the win condition.
        It also directly contradicted `_goal_satisfiability_check`, which now REJECTS any predicate
        that is True at the level root: a model obeying the old instruction produced a predicate the
        gate immediately threw out as `goal_predicate_true_at_root`.
        The grid is still shown -- the object vocabulary and geometry are genuinely useful -- but
        labelled for what it is, with the polarity stated correctly.

        `trans` (2026-08-01, OPT-IN via `_goal_prompt_transitions_on()`, DEFAULT OFF) attaches
        the SAME observed transitions the engine half is built from. Until this existed the
        block received a game name and one grid and nothing else -- it was the evidence-free
        prompt of the pair, and the taxonomy traced 12 of 13 whole-board "every cell is one
        colour" predicates to precisely the cells it produced. When the flag is off the
        argument is ignored and the returned string is byte-identical to the shipped one, which
        is what keeps the control arm of the measurement the SHIPPED prompt rather than a
        re-rendering of it.

        This shows the model nothing the live agent does not already have: the transitions ARE
        the agent's own observations, rendered exactly as `induce_prompt` renders them. It is
        help USING what has already been seen, not a fact about the game from outside, so it
        works identically on a game nobody has ever solved.
        """
        obs = ""
        if trans and _goal_prompt_transitions_on():
            # `_transitions_block` -- the SAME renderer `induce_prompt` uses, at the same k
            # resolver. Reusing it rather than writing a second encoder means the two prompts
            # cannot drift into disagreeing about what the agent observed, and the model meets
            # one delta format instead of two.
            obs = (
                "These are the transitions YOU observed in this game -- the action taken and "
                "the FULL set of cells it changed. Whatever a win requires, it has to be "
                "expressible in terms of what these deltas move, clear, fill or count:\n"
                + _transitions_block(
                    list(trans),
                    _induce_transitions_k(),
                    previous_level_complete_grid=None,
                )
                + "\n\n"
            )
        win = ""
        if previous_level_complete_grid is not None:
            win = (
                "This is the board at the START of the current level (captured just after the "
                "previous level completed). is_level_complete must return False here -- a level is "
                "NOT complete at its opening screen. Use it for the object vocabulary, palette and "
                "geometry, and infer what would have to CHANGE for the level to be complete:\n"
                + to_ascii(np.asarray(previous_level_complete_grid))
                + "\n"
            )
        # CORRECTED 2026-08-08 (adversarial review, think-routing gap). `generate()`'s own
        # think-mode branch (`elif _think_on: pass`) only skips ADDING its own pre-opened fence
        # -- it does nothing to strip one this prompt already ends with. Before this fix, this
        # was one of two call shapes (the split-induce engine-half fallback below is the other)
        # that unconditionally primed a `\`\`\`python\n` fence, so it suppressed reasoning even
        # on a think-mode call. Only the combined induce prompt in `induce()` was gated. Same
        # conditional shape as that gate.
        _goal_suffix = (
            "Return ONLY one ```python code block defining is_level_complete."
            if induce_think_on()
            else "Return ONLY one ```python code block defining is_level_complete.\n```python\n"
        )
        return (
            f"You are inducing ONLY the win condition for the ARC-AGI-3 game '{game}'.\n"
            + obs
            + win
            + "Write ONLY `def is_level_complete(grid):` returning True iff `grid` is a level-complete "
            "/ win state, else False. numpy + stdlib only; pure and deterministic. Prefer a SIMPLE "
            "GENERAL rule over an exact full-grid match.\n\n" + _goal_suffix
        )

    def _combine_world_model(self, engine_code: str, goal_code: str) -> str:
        """Concatenate a focused engine block and a focused is_level_complete block into one world
        model. Both pieces already parse individually (generate validates each); duplicate imports
        are valid Python, but we verify the concatenation parses and fall back to a raw join.

        UNDER `_goal_dedup_on()` (DEFAULT OFF) the engine half's own top-level
        `is_level_complete` is excised before the join, so the result defines the function
        EXACTLY ONCE. This is the belt to `_split_induce`'s braces: that path normally avoids
        calling the goal generator at all when the engine half already answered usefully, but
        when it does call it -- because the engine half declined, or was structurally defective
        -- the two halves would still both define the function and the reader would be back to
        inferring which one runs. Making it impossible here means no caller of this function can
        produce a shadowed file, whatever it passes in.

        See `_goal_dedup_on` for what the frozen-corpus measurement did and did NOT establish;
        in particular this is justified by a validity asymmetry, not by the shadowed predicate
        being better.
        """
        import ast

        if _goal_dedup_on():
            engine_code = _strip_top_level_goal_defs(engine_code)

        combined = (
            "import numpy as np\n\n" + engine_code.strip() + "\n\n" + goal_code.strip() + "\n"
        )
        try:
            ast.parse(combined)
            return combined
        except SyntaxError:
            return engine_code.strip() + "\n\n" + goal_code.strip() + "\n"

    def induce(
        self,
        game: str,
        trans: list[Transition],
        cell: int,
        *,
        previous_level_complete_grid: Optional[np.ndarray] = None,
        win_transition: Optional[Transition] = None,
    ) -> tuple[bool, str]:
        # OPT-IN TOOL-CALLING LOOP (REQ-ARC-WMTE-6460, 2026-08-17, DEFAULT OFF). Unset ->
        # this block is dead and induction is byte-identical to the shipped single-shot.
        # The loop replaces mental grid simulation (the ~95%-of-decode think channel)
        # with in-process tool execution; see arc_induction_tool_loop's docstring. A
        # loop failure returns (False, ...) and we FALL THROUGH to the shipped path,
        # so the worst case is today's behaviour plus the loop's bounded cost.
        if os.environ.get("CARNOT_ARC_INDUCE_TOOL_LOOP") == "1":
            from carnot.agentic.arc_induction_tool_loop import induce_with_tool_loop

            ok_tool, note_tool = induce_with_tool_loop(
                self,
                game,
                list(trans),
                int(cell),
                previous_level_complete_grid=previous_level_complete_grid,
                win_transition=win_transition,
            )
            if ok_tool:
                return ok_tool, note_tool
        base = induce_prompt(
            game,
            trans,
            cell,
            previous_level_complete_grid=previous_level_complete_grid,
            win_transition=win_transition,
            k=_induce_transitions_k(),
            include_playbook_exemplars=self.include_playbook_exemplars,
        )
        # Happy path: one combined engine+is_level_complete induction (code-only eligible: it is the
        # win-state-exemplar prompt whose CoT caused the truncation; refactor stays reasoning).
        #
        # THINK MODE (lever #6, default ON since 2026-08-08 -- was OFF when this comment was
        # first written). The PRE-OPENED ```python fence below is exactly what exp5714 proved
        # suppresses reasoning even on its own, independent of the codeonly directive -- so a
        # think-mode call must not send it either. The instruction to return one code block
        # stays either way; only the priming fence differs. The two other induce prompt shapes
        # (the split-induce engine-half fallback and `_goal_only_prompt`, both below) had the
        # same unconditional fence and were fixed the same way on 2026-08-08 -- until then only
        # this combined-call path actually respected think mode.
        _induce_suffix = (
            "\n\nReturn ONLY one ```python code block with engine + is_level_complete."
            if induce_think_on()
            else "\n\nReturn ONLY one ```python code block with engine + is_level_complete.\n```python\n"
        )
        # GAP-6260 layer 1, before a token is spent. A window whose actions the planner cannot
        # emit yields an engine nothing can plan in, however well it scores on its own window.
        self.last_window_vocabulary_violation = sorted(
            window_actions_outside_planner_vocabulary(trans)
        )
        if self.last_window_vocabulary_violation:
            print(
                "WINDOW VOCABULARY VIOLATION: actions "
                f"{self.last_window_vocabulary_violation} appear in this window and the in-model "
                "planner cannot emit them. The induced engine will model transitions the planner "
                "never asks for, and may be inert at the planning root even at cell_recall 1.0. "
                "Inducing anyway -- see ops/verifier_gaps.md GAP-6260.",
                flush=True,
            )
        ok, code = self.generate(
            base + _induce_suffix,
            ("engine", "is_level_complete"),
            tries=self.tries,
            codeonly_eligible=True,
            # The SAME transitions the prompt was built from. This is not a held-out set and is
            # not scored as one: it is the dry run that catches an `engine()` which raises or
            # returns None on evidence the model was literally shown.
            engine_transitions=trans,
        )
        if ok:
            return self._write_world_model(game, code)
        # FALLBACK (proto_l2_fix_finder, 2026-06-25): on complex real L2 prompts the combined call
        # commonly fails because the model rambles its analysis INTO engine() comments, exhausts the
        # token budget, and never writes is_level_complete. Induce each function in its OWN focused
        # call so the engine ramble cannot starve the goal -- the focused goal call is valid in ~3.5s
        # where the combined call fails (a budget bump does NOT help; the model just rambles more).
        # CORRECTED 2026-08-08 (adversarial review, think-routing gap). Same fix as
        # `_goal_only_prompt`'s trailing fence, same reason: `generate()`'s think-mode branch
        # only skips ADDING its own pre-opened fence, it does not strip one already in the
        # prompt text, so this call sent a reasoning-suppressing fence even under think mode.
        _engine_only_suffix = (
            "\n\nReturn ONLY one ```python code block defining engine(grid, action, data)."
            if induce_think_on()
            else (
                "\n\nReturn ONLY one ```python code block defining engine(grid, action, data)."
                "\n```python\n"
            )
        )
        ok_e, eng = self.generate(
            base + _engine_only_suffix,
            ("engine",),
            tries=self.tries,
            codeonly_eligible=True,
            engine_transitions=trans,
        )
        if not ok_e:
            return False, f"split induce: engine failed: {str(eng)[:150]}"
        # THE ENGINE HALF MAY ALREADY HAVE ANSWERED. `required=("engine",)` means the engine is
        # the only thing this call had to produce, but the base prompt describes the whole
        # interface, so the model routinely writes an `is_level_complete` here too -- and this
        # one saw the transitions and the opening grid, which the goal-only prompt does not.
        # Generating a second definition and appending it after this one used to be
        # unconditional, and Python binds the last, so the evidence-carrying predicate was
        # overwritten by construction on every such cell.
        #
        # Skipping the goal call when the engine half already supplied a STRUCTURALLY VALID,
        # NON-DECLINED predicate makes the shadowing impossible rather than unlikely, and costs
        # one fewer generation call. When the engine half declined (`return False`) or is
        # defective, the goal call still runs -- there is genuinely nothing to preserve, and
        # `_combine_world_model` then excises the dud so the file still defines the function
        # once.
        #
        # DEFAULT OFF. Read `_goal_dedup_on` before flipping it: the measurement behind this
        # establishes a validity asymmetry (4 of 23 bound definitions are not usable predicates
        # at all, against 0 of 23 shadowed), NOT that the preserved predicate scores better on
        # the goal gate -- there it is 2 against 1 across 23 cells, which is noise.
        if _goal_dedup_on() and _engine_half_goal_usable(eng):
            # The `import numpy as np` prefix is NOT decoration: `_combine_world_model` prepends
            # it on every split-induce write, so skipping the join must not quietly drop that
            # guarantee. A completion that uses `np` without importing it at top level would
            # load fine through the old path and raise `NameError` through this one. Measured
            # over the 40 raw completions in `results/arc_induce_bestofn_20260731`, 0 lack the
            # import -- so this is belt-and-braces, not a live failure. It is here anyway
            # because "the corpus happens not to exercise it" is a reason to keep an invariant
            # cheaply, not a reason to drop it. A duplicate import is valid Python and is
            # exactly what the shipped path already produces.
            return self._write_world_model(
                game,
                "import numpy as np\n\n" + eng.strip() + "\n",
                note="split induce: engine half supplied is_level_complete (dedup)",
            )
        ok_g, goal = self.generate(
            self._goal_only_prompt(game, previous_level_complete_grid, trans),
            ("is_level_complete",),
            tries=self.tries,
            codeonly_eligible=True,
            # The observed transitions reach BOTH halves of the opt-in goal intervention: the
            # prompt (gated by `_goal_prompt_transitions_on`) and the dry-run probe (gated by
            # `_goal_defect_check_on`). Passing them unconditionally is inert with the flags
            # off -- `_goal_defects` returns [] immediately when its flag is unset, and
            # `_goal_only_prompt` ignores the argument when its own flag is unset -- so the
            # shipped path is byte-identical either way.
            engine_transitions=trans,
        )
        if not ok_g:
            return False, f"split induce: goal failed: {str(goal)[:150]}"
        return self._write_world_model(
            game, self._combine_world_model(eng, goal), note="split induce: engine + focused goal"
        )

    def refactor(self, game: str, vr: VerifyResult) -> tuple[bool, str]:
        # NOT codeonly_eligible: refactor asks the model to reason about BEFORE/PREDICTED/OBSERVED
        # mismatches; the code-only "skip all reasoning" directive would degrade exactly that.
        return self._gen_to_file(
            game,
            refactor_prompt(game, vr)
            + "\n\nReturn ONLY the corrected ```python file.\n```python\n",
            codeonly_eligible=False,
        )

    def induce_programmatic_experts(
        self,
        *,
        game: str,
        transitions: Sequence[Transition],
        heldout_transitions: Sequence[Transition] | None = None,
        cell: int = 1,
        max_experts: int = 8,
    ) -> list[dict[str, Any]]:
        """REQ-ARC-WMTE-4677: ask the local GGUF for small serializable expert rules."""

        examples = _transitions_block(list(transitions), k=min(6, max(1, len(transitions))))
        prompt = f"""You are proposing SMALL programmatic object-level experts for ARC-AGI-3 game '{game}'.

Each expert must be a SERIALIZABLE dictionary with:
  name, object_class, kind='color_rewrite', action, from_color, to_color.

Use only the observed transitions. Prefer simple object/color rewrite factors that can be
held-out replay verified. Do not include brittle grid-sized programs. Return a Python code
block defining:

def expert_rules():
    return [{{...}}, ...]

Limit to {int(max_experts)} experts. Actions are integer ARC actions; click data is in pixel
coords where one logical cell is {int(cell)} pixels.

OBSERVED PREFIX TRANSITIONS:
{examples}
"""
        ok, code = self.generate(
            prompt + "\n```python\n",
            required=("expert_rules",),
            validate=None,
            tries=1,
        )
        if not ok:
            return []
        namespace: dict[str, Any] = {}
        try:
            exec(code, {"np": np, "numpy": np}, namespace)
            rows = namespace["expert_rules"]()
        except Exception:
            return []
        return [dict(row) for row in list(rows or []) if isinstance(row, Mapping)]


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


# The action ids the in-model planner can actually emit: the five keyboard actions plus a click.
# `_model_candidates` below is the authority; this mirrors its action set so a window can be
# checked WITHOUT a grid, which is what lets the check run before induction spends a token.
# `tests/python/test_arc_window_vocabulary.py` asserts the two stay in agreement.
_PLANNER_ACTION_VOCABULARY = frozenset({1, 2, 3, 4, 5, 6})


def window_actions_outside_planner_vocabulary(trans: list) -> set:
    """Actions present in an induction window that the in-model planner CANNOT emit.

    WHY THIS EXISTS (GAP-6260, 2026-08-18). An induced engine is only useful if the planner can
    drive it, and the planner's move set is `_model_candidates`: actions 1-5 plus clicks. A window
    whose observed actions fall outside that set teaches the engine transitions the planner will
    never ask for, so the engine can score a perfect `cell_recall` on its own window and still be
    unplannable.

    That is not hypothetical. Every action in lp85's cached window is 0 (RESET), which the planner
    does not plan through, so its induced engine produced ZERO novel successors from the planning
    root: 37 nodes expanded -- exactly one candidate sweep of no-ops -- the win predicate never
    consulted, and the search exhausted its queue far under budget. Six A/B cells across both arms
    were spent on it. `cell_recall = 1.0` was honest; what it measured was not gameplay.

    This is the CHEAP half of the two-layer check in `ops/verifier_gaps.md` GAP-6260. It costs no
    engine calls and no generation, so it can run BEFORE induction rather than diagnosing the
    wreckage after. The second layer -- does the induced engine yield a novel successor from the
    root -- needs an engine and is not implemented here.

    REPORTS, DOES NOT BLOCK. Returning the offending actions lets the caller log, record, or skip;
    it deliberately does not abort induction on its own, because refusing to induce is a live-path
    behaviour change that wants its own measurement first.
    """
    actions = set()
    for t in list(trans or []):
        a = getattr(t, "action", None)
        if isinstance(a, (int, np.integer)):
            actions.add(int(a))
    return actions - _PLANNER_ACTION_VOCABULARY


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
        # Defensive unpack (`*_` absorbs any extra trailing fields): `_components_detailed` was widened
        # from a 4-tuple (cy, cx, area, color) to a 5-tuple (+ is_grid_fallback) in the
        # GAP-ARC-BP35-CLICK-CANDIDATE-GENERATION-MISS fix (commit 2f0760307), which updated the
        # arc_graph_explore consumer defensively but MISSED this one -- a rigid `cy, cx, _a, _c` unpack then
        # crashed plan_in_model on ANY grid with components (e.g. tu93's 65), silently disabling the entire
        # world-model planning tier for those games. `*_` handles both the 4-tuple (test doubles monkeypatch
        # the old shape) and the 5-tuple (real) forms. (REQ-ARC-WMTE-5841 regression fix.)
        for cy, cx, _a, _c, *_ in comps[:32]:
            cands.append({"action": 6, "data": {"x": int(cx), "y": int(cy)}})
    return cands


def _termination_reason(nodes: int, max_nodes: int, depth_truncated_nodes: int) -> str:
    """Why a `plan_in_model` search ended without a plan. Three-way, not two-way.

    WHY THIS EXISTS (2026-07-31). Both search loops in `plan_in_model` used to close with
    ``"max_nodes_reached" if nodes >= max_nodes else "queue_exhausted"``, which forces every
    non-budget ending into a single label meaning "the frontier emptied, so I searched the reachable
    set exhaustively and there is no plan." That is FALSE whenever the loop dropped a node at
    ``max_depth``: those nodes were popped and thrown away UNEXPANDED, so their subtrees were never
    looked at, and the frontier emptied for a reason that says nothing at all about whether a plan
    exists. A caller reading `queue_exhausted` concludes the ENGINE or the GOAL is wrong; the true
    answer in that case is that the DEPTH BUDGET is too small. Those lead to opposite fixes, which
    is the whole cost of the mislabel -- the identical bug in the goal gate
    (`arc_llm_reinduction._goal_satisfiability_check`) spent a milestone pointing "your win
    condition is degenerate" at tn36's measured-reachable predicate.

    PRECEDENCE is budget, then depth, then genuine exhaustion. A run that spent its node budget is
    budget-limited whatever else it also did -- raising `max_depth` alone would not have helped it,
    because it never got to look. Only when the budget is intact does the depth cap become the
    binding explanation. `queue_exhausted` is now what it always claimed to be: nothing left to
    search AND nothing thrown away, so the negative result is real evidence.

    This is a REPORTING change. `plan_in_model` returns exactly what it returned before on every
    path; nothing in the tree branches on the string.
    """
    if nodes >= max_nodes:
        return "max_nodes_reached"
    if depth_truncated_nodes > 0:
        return "depth_capped"
    return "queue_exhausted"


_PLAN_DEFAULT_MAX_DEPTH = 80


def plan_max_depth_default() -> int:
    """THE single source of truth for the search horizon, shared by the planner and the goal gate.

    WHY IT IS SHARED RATHER THAN TWO CONSTANTS (this is a soundness requirement, not tidiness).
    `arc_llm_reinduction._goal_satisfiability_check` vetoes a goal it cannot reach within its
    depth cap, and that veto is only honest because `plan_in_model` -- the planner the gate exists
    to guard -- is bounded by the SAME cap: a goal unreachable within the horizon is genuinely
    unreachable BY THE PLANNER, so refusing it wastes nothing. Let the two drift and both
    directions break:
      * gate deeper than planner -> the gate certifies goals the planner then fails on, turning
        an honest veto into a false accept one layer down;
      * planner deeper than gate -> the gate vetoes goals the planner could now reach.
    Two module-level constants would drift silently on the next edit. One resolver cannot.

    WHY 80 AND NOT 40 (measured 2026-07-31, twice, by two independently-written harnesses).
    At the previous default of 40, re-running the shipped gate and planner over 48 frozen
    induction candidates gave an EMPTY intersection between "clears held-out dynamics" and "a
    plan was found" -- 9 candidates in the first set, 2 in the second, 0 in both. That looked
    like evidence that selecting engines on dynamics is anti-selective for plannability. It was
    not. It was the horizon:

        max_depth   clears (i)   plan found   BOTH
               40            9            2      0
               61            9            6      3
               80            9            6      3
              200            9            6      3

    Three tn36 engines that predict 17 of 17 held-out changing transitions EXACTLY, and that
    already pass the shipped trust gate, need a 61-action plan. At 40 they were reported
    `goal_unreached_within_depth` and discarded.

    THE COST IS ABSENT, NOT MERELY SMALL. Mean planner wall time across all 48 candidates was
    0.357s at depth 40, 0.339 at 61, 0.334 at 80, 0.355 at 200 -- flat within noise, and LOWEST
    where the conversions happen, because a search that succeeds stops early instead of draining
    its frontier. The reason is structural and was measured rather than assumed: tn36's 32
    changing root actions collapse to ONE distinct successor (the engine fills the next cell
    wherever the click lands) and both search sites dedup by state key, so the tree is a PATH.
    On a path the node budget buys depth one-for-one and only `max_depth` can stop the search --
    which is why raising the horizon and NOT `max_nodes` is the correct fix, and why this is a
    horizon correction rather than a threshold relaxation. `max_nodes` is untouched at 20000, and
    every quality check is untouched: the same sweep also moved 3 candidates from UNDECIDED to a
    firm `degenerate_goal_predicate`, which is the gate rejecting MORE confidently, not less.

    80 rather than 61: 61 is one game's exact measured requirement (tn36's row is 61 cells wide)
    and pinning the default to it would be overfitting to the only game in the corpus that
    converts. 80 carries ~30% headroom at a cost measured to be zero. Nothing above 61 bought
    anything on this corpus, so this is headroom over a measured need, NOT a tuned optimum -- do
    not read 80 as evidence that 80 is special.

    HONEST SCOPE. Every conversion observed was tn36. Of the other 36 stall candidates, 21 never
    had a usable engine at all (13 predict that no action changes anything; 8 are disproved at
    their own dynamical fixed point) and 8 are not valid Python. Raising the horizon does nothing
    for any of them. This unblocks a real and previously-invisible class; it does not fix
    induction.

    `CARNOT_ARC_PLAN_MAX_DEPTH` restores any prior behaviour exactly -- set it to 40 to recover
    the pre-2026-07-31 default byte-for-byte.
    """

    raw = os.environ.get("CARNOT_ARC_PLAN_MAX_DEPTH")
    if raw is None:
        return _PLAN_DEFAULT_MAX_DEPTH
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return _PLAN_DEFAULT_MAX_DEPTH
    return value if value > 0 else _PLAN_DEFAULT_MAX_DEPTH


def plan_in_model(
    engine,
    is_level_complete,
    start_grid: np.ndarray,
    *,
    max_nodes: int = 20000,
    max_depth: int | None = None,
    goal_energy=None,
    diagnostics: Optional[dict] = None,
) -> Optional[list]:
    """BFS a path to an is_level_complete state ENTIRELY INSIDE the induced model
    (engine is pure: grid,action,data -> grid; no environment). Returns the action
    sequence [{"action","data"}] that the model believes reaches a win, or None. This
    is the harness-friendly planner: the agent computes the plan with zero real actions,
    then executes it in the real env (few real actions = the EFFICIENCY win), halting if
    reality diverges from the model.

    GOAL-ENERGY (2026-06-23, closes GAP-ARCH-GOAL-NOT-VERIFIED): when ``goal_energy`` (grid -> float,
    LOWER = closer to the induced win) is supplied, the search is BEST-FIRST by goal_energy -- it DESCENDS
    toward the goal predicate instead of exploring blind breadth-first, so it reaches the win in FEWER
    nodes (the action-efficiency win). ``goal_energy`` is induced per-game from the agent's OWN observed
    win/non-win states (``arc_agi3_goal_induction.induce_goal_energy``), NOT a frozen transfer. Backward-
    compatible: ``goal_energy=None`` keeps the exact original FIFO BFS. The terminal check stays
    ``is_level_complete`` (the energy only orders the frontier); an ablation control is mandatory.

    DIAGNOSTICS (REQ-ARC-FCP-5699-15, closes the "trust gate passes but no plan found" question
    REQ-ARC-FCP-5699-14 left open): when ``diagnostics`` (a caller-owned dict) is supplied, this
    populates it with ``is_level_complete_was_none`` (bool), ``nodes_expanded`` (int),
    ``termination_reason`` (one of ``"is_level_complete_none"`` / ``"plan_found"`` /
    ``"max_nodes_reached"`` / ``"depth_capped"`` / ``"queue_exhausted"``), and
    ``used_goal_energy_search`` (bool) before
    returning -- so a caller can tell WHY an empty return happened without re-deriving the search.

    DEPTH AXIS (2026-07-31, REQ-ARC-WMTE-6047-D sibling fix). ``termination_reason`` used to be a
    two-way choice, ``"max_nodes_reached" if nodes >= max_nodes else "queue_exhausted"`` -- but the
    loop above it does ``if len(path) >= max_depth: continue``, which DISCARDS a popped node WITHOUT
    EXPANDING IT. So the frontier can drain while the reachable set was never searched out, and the
    old code reported that as ``queue_exhausted``: "I looked everywhere and there is no plan," when
    the truth is "I stopped looking at depth ``max_depth``." Those are opposite conclusions for a
    caller deciding whether the ENGINE is wrong or the BUDGET is. The same conflation was fixed on
    2026-07-31 in ``arc_llm_reinduction._goal_satisfiability_check`` (the goal gate); this is the
    identical bug in the function that gate guards, and the gate's own evidence proves it lives here
    too -- ``results/arc_goal_gate_depth_20260731/tn36_depth_label.json`` records
    ``plan_at_max_depth_40.diagnostics.termination_reason == "queue_exhausted"`` on a goal this same
    engine reaches at depth 61 once the cap is lifted.

    So there is now a third value, ``"depth_capped"``, plus ``depth_truncated_nodes`` (int, always
    populated): the number of popped nodes discarded unexpanded at the cap. Precedence is
    budget-first (``nodes >= max_nodes``), then depth, then genuine exhaustion -- a search that
    spent its node budget is budget-limited whatever else happened. **This changes what is RECORDED,
    never what is RETURNED**: every path still returns the same plan or the same ``None``. No caller
    in the tree branches on ``"queue_exhausted"``; the value is read for reporting only
    (``scripts/arc_plan_in_model_nav_solve.py``, the ``ttt_prior_engine_plan_diagnostics`` blob).
    REQ-ARC-FCP-5699-18 adds ``initial_goal_energy``/``min_goal_energy_observed`` (floats, only when
    ``used_goal_energy_search`` is True): the goal-energy value at ``start_grid`` and the lowest
    value seen across every state the search visited -- lets a caller tell whether a failed search
    got structurally CLOSE to the induced goal (min << initial, "coherent but ran out of budget")
    or never moved toward it at all (min ~= initial, "the model's rollout doesn't structurally
    connect toward the goal"). Backward-compatible: ``diagnostics=None`` (the default) changes
    nothing about the search or the return value.

    ENGINE-CALL GUARD (2026-08-17, REQ-ARC-WMTE-6400). ``engine`` and
    ``is_level_complete`` are LLM-generated. Nothing here bounded ONE call's cost:
    a generated sb26 engine contained a non-terminating, allocating flood fill, and
    a single click candidate drove the per-game process to ~78 GB RSS and an OOM
    kill. Both callables now run through ``arc_engine_call_guard.guarded_call``,
    which raises inside the calling thread (works off the main thread, where
    ``signal.alarm`` silently cannot). A trip skips the candidate; after
    ``guard_max_trips()`` trips the search is abandoned, because a hanging engine
    usually hangs on many inputs and each trip costs a full timeout. This adds a
    fourth ``termination_reason`` value, ``"engine_guard_tripped"``, and a new
    always-populated diagnostics key ``engine_guard_trips`` (int). Return values
    are unchanged on every path a guard trip does not touch."""
    # `None` means "no caller opinion" -> the shared default (or its env override). An EXPLICIT
    # `max_depth` still wins outright, so every existing test and diagnostic caller that pins a
    # horizon keeps pinning it. Resolved HERE rather than in the signature default so the env
    # override is read at CALL time, matching how `max_nodes` is handled in the goal gate.
    if max_depth is None:
        max_depth = plan_max_depth_default()
    if is_level_complete is None:
        if diagnostics is not None:
            diagnostics["is_level_complete_was_none"] = True
            diagnostics["nodes_expanded"] = 0
            diagnostics["termination_reason"] = "is_level_complete_none"
        return None
    start = np.asarray(start_grid)
    seen = {_state_key(start)}
    nodes = 0
    # Counts popped-but-never-expanded nodes -- the depth axis. See the DEPTH AXIS note in the
    # docstring: without this the frontier draining at the cap is indistinguishable from the
    # frontier draining because there was nothing left to search, and those mean opposite things.
    depth_truncated_nodes = 0

    # ENGINE-CALL GUARD (REQ-ARC-WMTE-6400): every `engine`/`is_level_complete`
    # call below runs LLM-generated code with no bound of its own. One generated
    # sb26 call hung forever while allocating (~78 GB, OOM kill). `guarded_call`
    # bounds ONE call's wall time and RSS growth; `max_nodes` only bounds COUNT.
    from carnot.agentic.arc_engine_call_guard import (
        EngineCallGuardError,
        guard_max_trips,
        guarded_call,
    )

    engine_guard_trips = 0
    engine_guard_abort = False
    engine_guard_trip_limit = guard_max_trips()

    if goal_energy is not None:
        # BEST-FIRST by goal-energy: descend toward the induced goal predicate.
        import heapq
        import itertools

        def _h(g):
            try:
                return float(goal_energy(g))
            except Exception:
                return 0.0

        counter = itertools.count()
        initial_energy = _h(start)
        min_energy = initial_energy
        heap = [(initial_energy, next(counter), start, [])]
        while heap and nodes < max_nodes:
            _, _, grid, path = heapq.heappop(heap)
            if len(path) >= max_depth:
                depth_truncated_nodes += 1
                continue
            for c in _model_candidates(grid):
                try:
                    ng = np.asarray(guarded_call(engine, grid.copy(), c["action"], c["data"]))
                except EngineCallGuardError:
                    # A hanging engine usually hangs on many inputs. Skip this
                    # candidate; at the trip limit abandon the whole search rather
                    # than paying a full timeout for thousands of candidates.
                    engine_guard_trips += 1
                    if engine_guard_trips >= engine_guard_trip_limit:
                        engine_guard_abort = True
                        break
                    continue
                except Exception:
                    continue
                nodes += 1
                if ng.shape != start.shape:
                    continue
                key = _state_key(ng)
                if key in seen:
                    continue
                seen.add(key)
                npath = path + [c]
                try:
                    if bool(guarded_call(is_level_complete, ng)):
                        if diagnostics is not None:
                            diagnostics["is_level_complete_was_none"] = False
                            diagnostics["nodes_expanded"] = nodes
                            diagnostics["termination_reason"] = "plan_found"
                            diagnostics["depth_truncated_nodes"] = depth_truncated_nodes
                            diagnostics["used_goal_energy_search"] = True
                            diagnostics["initial_goal_energy"] = initial_energy
                            diagnostics["min_goal_energy_observed"] = min_energy
                            diagnostics["engine_guard_trips"] = engine_guard_trips
                        return npath
                except EngineCallGuardError:
                    # The goal predicate is generated too, so it carries the same
                    # hang exposure. Unlike the plain-exception `pass`, skip the
                    # state: pushing it would only re-run the hang on every pop.
                    engine_guard_trips += 1
                    if engine_guard_trips >= engine_guard_trip_limit:
                        engine_guard_abort = True
                        break
                    continue
                except Exception:
                    pass
                ng_energy = _h(ng)
                if ng_energy < min_energy:
                    min_energy = ng_energy
                heapq.heappush(heap, (ng_energy, next(counter), ng, npath))
            if engine_guard_abort:
                break
        if diagnostics is not None:
            diagnostics["is_level_complete_was_none"] = False
            diagnostics["nodes_expanded"] = nodes
            diagnostics["used_goal_energy_search"] = True
            diagnostics["initial_goal_energy"] = initial_energy
            diagnostics["min_goal_energy_observed"] = min_energy
            diagnostics["depth_truncated_nodes"] = depth_truncated_nodes
            diagnostics["engine_guard_trips"] = engine_guard_trips
            diagnostics["termination_reason"] = (
                "engine_guard_tripped"
                if engine_guard_abort
                else _termination_reason(nodes, max_nodes, depth_truncated_nodes)
            )
        return None

    # ---- original blind FIFO BFS (goal_energy=None; unchanged) ----
    from collections import deque

    q = deque([(start, [])])
    while q and nodes < max_nodes:
        grid, path = q.popleft()
        if len(path) >= max_depth:
            depth_truncated_nodes += 1
            continue
        for c in _model_candidates(grid):
            try:
                ng = np.asarray(guarded_call(engine, grid.copy(), c["action"], c["data"]))
            except EngineCallGuardError:
                # Same trip policy as the goal-energy branch above.
                engine_guard_trips += 1
                if engine_guard_trips >= engine_guard_trip_limit:
                    engine_guard_abort = True
                    break
                continue
            except Exception:
                continue
            nodes += 1
            if ng.shape != start.shape:
                continue
            key = _state_key(ng)
            if key in seen:
                continue
            seen.add(key)
            npath = path + [c]
            try:
                if bool(guarded_call(is_level_complete, ng)):
                    if diagnostics is not None:
                        diagnostics["is_level_complete_was_none"] = False
                        diagnostics["nodes_expanded"] = nodes
                        diagnostics["termination_reason"] = "plan_found"
                        diagnostics["depth_truncated_nodes"] = depth_truncated_nodes
                        diagnostics["used_goal_energy_search"] = False
                        diagnostics["engine_guard_trips"] = engine_guard_trips
                    return npath
            except EngineCallGuardError:
                # Same trip policy as the goal-energy branch above.
                engine_guard_trips += 1
                if engine_guard_trips >= engine_guard_trip_limit:
                    engine_guard_abort = True
                    break
                continue
            except Exception:
                pass
            q.append((ng, npath))
        if engine_guard_abort:
            break
    if diagnostics is not None:
        diagnostics["is_level_complete_was_none"] = False
        diagnostics["nodes_expanded"] = nodes
        diagnostics["used_goal_energy_search"] = False
        diagnostics["depth_truncated_nodes"] = depth_truncated_nodes
        diagnostics["engine_guard_trips"] = engine_guard_trips
        diagnostics["termination_reason"] = (
            "engine_guard_tripped"
            if engine_guard_abort
            else _termination_reason(nodes, max_nodes, depth_truncated_nodes)
        )
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
    seen = {_state_key(start)}
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
            key = _state_key(ng) if ng.ndim == 2 else None
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

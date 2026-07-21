"""Exp 5768: Does DIRECT in-context next-grid prediction (NO code) beat CODE-SYNTHESIS
world-model induction? (REQ-ARC-WMTE-5768)

WHY THIS EXISTS
---------------
Tonight's induction-quality diagnosis
(docs/research-notes/arc-world-model-induction-quality-diagnosis-2026-07-20.md) found that
Carnot's ARC world-model induction asks the LLM to do something HARDER than pattern recognition:
not just infer the transition rule from a few examples, but correctly EXTERNALIZE it as
syntactically-valid, general Python `engine(grid, action, data)` code. That code-writing step is
where things systematically break -- near-zero `heldout_accuracy` (29/37 cells exactly 0.0), with
failure patterns including literal window-coordinate memorization (hardcoding observed pixel
positions instead of inferring a general rule). Even gemma-4-31B-it (the strongest model tested,
REQ-ARC-WMTE-5764) floored code-synthesis single-shot at pooled heldout 0.378487.

The operator (after discussing Google's TabFM -- a foundation model that predicts tabular outputs
via direct in-context learning in a single forward pass, no code generation -- and the GPT-3
few-shot paper, arXiv:2005.14165) asked a cheap, low-risk question: does asking the SAME local
model to DIRECTLY predict the next grid state in-context (no code, no function -- "here are N
examples of (before,action,data)->after; given this NEW (before,action,data), what's after?")
outperform asking it to write code implementing the general rule?

THE MECHANISM -- GENUINELY DIFFERENT (Failed-Experiment Rerun Discipline)
-------------------------------------------------------------------------
NO code is generated at ANY point. The model's raw text output IS the predicted grid, parsed back
and compared to the true next_grid via the SAME exact full-grid match (`np.array_equal`) that
`WorldModelVerifier.accuracy` / `heldout_accuracy` uses. This is a NEW induction MECHANISM, not a
rerun of:
  * single-shot CODE induction (REQ-ARC-WMTE-5726/5764: LLM writes engine() Python), or
  * CEGIS refinement (REQ-ARC-WMTE-5760/5766: iteratively repair the induced engine() Python).
Both cited in prior_work_extended; what is mechanistically different is stated precisely there.

LEAVE-ONE-OUT, LEAK-FREE (the adversarial concern the brief flags)
------------------------------------------------------------------
The naive "show all N window transitions as examples then re-predict them" would let the model
COPY the after-grid straight out of its own prompt (answer leakage) -- and separately, a model
could score well by NEAR-IDENTITY-COPYING the input grid on a no-op-heavy window. Both are guarded:

  * LEAVE-ONE-OUT: the transition being predicted is NEVER in its own in-context example set, so
    there is zero verbatim-answer leakage. Every prediction is a genuine generalization from the
    OTHER observed transitions -- a STRICTER test than the code baseline (whose induced engine may
    still memorize its k=8 shown transitions). So a direct-prediction win here is conservative.
  * IDENTITY-COPY instrumentation: `identity_copy_rate` (fraction of predictions equal to the
    INPUT grid), plus `heldout_accuracy` split into CHANGING vs NO-OP transitions -- an identity
    copier trivially clears no-ops but fails every changing transition. The changing-only number
    is the honest "did it predict real dynamics" signal.
  * PARSE-SUCCESS: free-text grid output can be malformed / truncated / wrong-shape -- a NEW
    failure mode code-synthesis did not have. `parse_success_rate` is reported SEPARATELY; a parse
    failure counts as an incorrect prediction (cannot match) but is NOT silently conflated with a
    genuine wrong-dynamics prediction. Both the all-predictions and parseable-only accuracies are
    reported.

APPLES-TO-APPLES: same 13-game/3-trial pre-registered roster as the CEGIS runs (ROSTER/TRIALS
imported from exp5760 to guarantee an exact match), same `build_progress_window` window, same
gemma-4-31B-it-GGUF model (strongest tested), same exact-match metric definition. The primary
comparison is pooled `heldout_accuracy` vs the gemma-4-31B code-synthesis single-shot baseline
(0.378487, REQ-ARC-WMTE-5764).

PROVENANCE: development_proxy on PUBLIC games (NOT a hidden-game self-discovery solve). This is a
DIAGNOSTIC over the SAME induction input the live mechanism uses, with the induction METHOD swapped
from code-synthesis to direct prediction -- it is NOT a live-path modification (nothing here writes
world_model.py or plans/executes) and NOT an orphan solver. verifier_is_oracle False (the exact-match
metric is oracle-distinct from any win oracle). NEVER flips the frozen live default, NEVER submits.

GPU: GPU 1 by default (operator-preferred; GPU 0 sometimes carries the conductor's own ARC
generator). Reuses exp5764's server-launch ladder + gemma config VERBATIM for a byte-identical
serving substrate. Runtime GPU-offload assertion (VRAM jump) refuses a silent CPU fallback.

RESUMABLE: every (game, trial) cell appends to a JSONL shard as it completes.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Reuse exp5764's gemma server machinery VERBATIM (byte-identical serving substrate for a fair
# model-held-constant comparison). Importing exp5764 runs only module-level config (it in turn
# imports exp5726/exp5760 module-level config); run_all()/main() are under its own __main__.
from carnot.experiment_5764_gemma31b_singleshot_induction_ab import (  # noqa: E402
    GEMMA,
    GPU_INDEX,
    check_preconditions,
    launch_server_ladder,
    make_gemma_proposer,
)
from carnot.experiment_5726_thinkingcap_16k_dualgpu_reason_ab import (  # noqa: E402
    _gpu_mem_used_mib,
    terminate,
)
from carnot.experiment_5760_cegis_refinement_induction_ab import ROSTER, TRIALS  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SHARD = REPO / "results" / "exp5768_direct_incontext_prediction_shard.jsonl"
ARTIFACT = REPO / "results" / "experiment_5768_direct_incontext_prediction_ab.json"

# gemma-4-31B code-synthesis single-shot baseline (REQ-ARC-WMTE-5764), read for the side-by-side.
GEMMA_CODE_BASELINE_ARTIFACT = REPO / "results" / "experiment_5764_gemma31b_singleshot_induction_ab.json"
GEMMA_CODE_BASELINE_POOLED = 0.378487  # from REQ-ARC-WMTE-5764 (fallback if artifact unreadable)

K_EXAMPLES = 8  # in-context example cap, matching the code path's induce k (_induce_transitions_k=8)
TEMPERATURE = 0.2  # matches the code-synthesis induce first-try temperature (generate: 0.2+0.1*att)
# Adaptive size bounds. A logical grid of H*W cells serialized as space-separated ints costs
# ~ H*W tokens per grid (digits tokenize ~1/char). K examples show 2 grids each (before+after) plus
# the query's before + its answer -> ~ (2*K+2)*H*W token-cells. Keep the whole prompt+answer under
# a fraction of the deployed n_ctx; if a grid is so large that even 1 example + query + answer does
# not fit, that cell is recorded blocked_grid_too_large (disclosed, NOT silently scored 0).
MAX_ANSWER_TOKENS_CAP = 8192  # hard cap on a single prediction's output budget
# space-separated small ints tokenize at ~1.4-1.7 tokens/cell (the digit(s) + the joining space),
# NOT 1 token/cell -- undercounting silently truncates the prompt on large grids. Empirically ~1.6.
TOKENS_PER_CELL = 1.6
CTX_SAFETY_FRACTION = 0.92  # keep prompt+answer under this fraction of n_ctx (margin for the estimate)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------------------
# Grid <-> text serialization (LOSSLESS, unlike to_ascii's last-digit form -- colors can exceed 9)
# ---------------------------------------------------------------------------
def grid_to_text(g: np.ndarray) -> str:
    """Row-major, one line per row, space-separated integers. Lossless for any color value."""
    g = np.asarray(g)
    return "\n".join(" ".join(str(int(v)) for v in row) for row in g)


_ROW_RE = re.compile(r"-?\d+")


def parse_grid(text: str, expected_shape: tuple[int, int]) -> Optional[np.ndarray]:
    """Parse the model's raw text into a rectangular integer grid.

    Robust to markdown fences, a leading 'AFTER:'-style marker, and trailing prose: scan every
    line, classify each as an int-row (all whitespace/comma-separated tokens are integers) or not,
    find maximal CONTIGUOUS blocks of equal-width int-rows, and choose the block that best matches
    ``expected_shape`` (exact match preferred, else the largest rectangular block, last one wins on
    ties -- models usually put the answer last). Returns None if no rectangular int-grid is found.
    parse_success is defined as "returned a rectangular int grid" REGARDLESS of whether its shape
    matches expected (a wrong-shape grid is a parsed-but-wrong prediction, mirroring
    WorldModelVerifier's `pred.shape == next.shape and array_equal`)."""
    if not text:
        return None
    eh, ew = expected_shape
    # Collect (row_values) for lines that are pure integer rows.
    classified: list[Optional[list[int]]] = []
    for raw in text.splitlines():
        s = raw.strip().strip("`").strip()
        if not s:
            classified.append(None)
            continue
        toks = [t for t in re.split(r"[,\s]+", s) if t]
        if toks and all(_ROW_RE.fullmatch(t) for t in toks):
            classified.append([int(t) for t in toks])
        else:
            classified.append(None)
    # Find maximal contiguous blocks of equal-width int-rows.
    blocks: list[list[list[int]]] = []
    cur: list[list[int]] = []
    cur_w: Optional[int] = None
    for row in classified:
        if row is None:
            if cur:
                blocks.append(cur)
            cur, cur_w = [], None
            continue
        w = len(row)
        if cur and w != cur_w:
            blocks.append(cur)
            cur, cur_w = [row], w
        else:
            cur.append(row)
            cur_w = w
    if cur:
        blocks.append(cur)
    if not blocks:
        return None

    def _key(b: list[list[int]]) -> tuple[int, int, int]:
        h, w = len(b), len(b[0])
        exact = 1 if (h == eh and w == ew) else 0
        return (exact, h * w, 0)

    # Prefer exact-shape blocks; among equals prefer larger; ties -> last (stable via index).
    best_idx = -1
    best_key: tuple[int, int, int] = (-1, -1, -1)
    for i, b in enumerate(blocks):
        k = _key(b)
        if k >= best_key:  # >= so a LATER block of equal key wins (answer usually last)
            best_key, best_idx = k, i
    chosen = blocks[best_idx]
    w0 = len(chosen[0])
    if any(len(r) != w0 for r in chosen):  # not rectangular
        return None
    try:
        return np.asarray(chosen, dtype=int)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Prompt construction -- DIRECT prediction, NO code (leave-one-out example set)
# ---------------------------------------------------------------------------
def _serialize_action(t: Any) -> str:
    data = getattr(t, "data", None)
    return f"ACTION {int(t.action)}" + (f"  DATA {data}" if data else "  DATA none")


def build_prediction_prompt(
    examples: list[Any], query: Any, *, game: str, cell: int, colors: list[int], shape: tuple[int, int]
) -> str:
    """A pure 'predict the output grid directly' prompt. NO instruction to write a function; the
    only requested output is the after-grid as rows of space-separated integers."""
    h, w = shape
    ex_lines = []
    for i, t in enumerate(examples, 1):
        ex_lines.append(
            f"EXAMPLE {i}\n"
            f"BEFORE:\n{grid_to_text(t.grid)}\n"
            f"{_serialize_action(t)}\n"
            f"AFTER:\n{grid_to_text(t.next_grid)}"
        )
    examples_block = "\n\n".join(ex_lines)
    return (
        f"You are given observed state transitions of the ARC-AGI-3 game '{game}'. Each state is a "
        f"{h}x{w} integer grid (colors {colors}). An ACTION transforms BEFORE into AFTER. ACTION 6 is "
        f"a click at pixel DATA={{'x':px,'y':py}} (one logical cell = {cell} pixels); other actions "
        f"are directional/keyboard with DATA none. Infer the transition rule from the examples and "
        f"apply it to predict the AFTER grid for the QUERY.\n\n"
        f"{examples_block}\n\n"
        f"QUERY\n"
        f"BEFORE:\n{grid_to_text(query.grid)}\n"
        f"{_serialize_action(query)}\n\n"
        f"Output ONLY the predicted AFTER grid as exactly {h} rows of {w} space-separated integers. "
        f"No explanation, no code, no other text.\n"
        f"AFTER:\n"
    )


def _select_examples(window: list, query_idx: int, *, k: int) -> list[Any]:
    """Leave-one-out example selection: up to k transitions from window EXCLUDING query_idx,
    preferring state-CHANGING transitions (the informative ones) then keeping a couple of no-ops,
    in original window order. The query transition is NEVER included (zero answer leakage)."""
    others = [(j, t) for j, t in enumerate(window) if j != query_idx]
    changing = [t for j, t in others if not np.array_equal(t.grid, t.next_grid)]
    noop = [t for j, t in others if np.array_equal(t.grid, t.next_grid)]
    sel = changing[: max(1, k - 2)] + noop[:2]
    return sel[:k]


def _answer_budget(shape: tuple[int, int]) -> int:
    """Output-token budget for ONE full-grid prediction: enough to emit the whole HxW grid at
    ~TOKENS_PER_CELL tokens/cell plus per-row newlines, capped. A grid that needs more than the
    cap will truncate -> that prediction is recorded as a parse failure (honest), never silently 0."""
    h, w = shape
    return int(min(MAX_ANSWER_TOKENS_CAP, TOKENS_PER_CELL * h * w + h + 64))


def _fits_context(shape: tuple[int, int], k: int, n_ctx: int) -> tuple[bool, int]:
    """Return (fits, k_usable). Estimate prompt tokens as TOKENS_PER_CELL * (2k+1) full grids
    (k examples show before+after each, plus the query's before) + header overhead, and require
    prompt+answer under CTX_SAFETY_FRACTION*n_ctx. Reduce k until it fits; (False, 0) if even
    k=1 does not fit (that game is recorded blocked_grid_too_large, disclosed)."""
    h, w = shape
    cells = h * w
    answer = _answer_budget(shape)
    ceil = int(CTX_SAFETY_FRACTION * n_ctx)
    for kk in range(k, 0, -1):
        prompt_tokens = int(TOKENS_PER_CELL * (2 * kk + 1) * cells) + 600  # +header/marker overhead
        if prompt_tokens + answer <= ceil:
            return True, kk
    return False, 0


# ---------------------------------------------------------------------------
# One (game, trial) cell: leave-one-out predict every window transition
# ---------------------------------------------------------------------------
def run_prediction_cell(
    game: str, prop: Any, *, trial: int, window: list, cell: int, n_ctx: int
) -> dict[str, Any]:
    shape = tuple(int(x) for x in np.asarray(window[0].grid).shape)
    colors = sorted({int(v) for t in window for v in np.asarray(t.grid).flatten().tolist()})
    fits, k_use = _fits_context(shape, K_EXAMPLES, n_ctx)
    if not fits:
        return {
            "game": game,
            "trial": trial,
            "grid_shape": list(shape),
            "n_transitions": len(window),
            "blocked": "grid_too_large_for_context",
            "n_ctx": n_ctx,
            "heldout_accuracy": None,
            "parse_success_rate": None,
        }
    ans_budget = _answer_budget(shape)
    per_trans: list[dict[str, Any]] = []
    for i, q in enumerate(window):
        examples = _select_examples(window, i, k=k_use)
        prompt = build_prediction_prompt(
            examples, q, game=game, cell=cell, colors=colors, shape=shape
        )
        t0 = time.time()
        try:
            ok, text = prop.complete_text(prompt, max_tokens=ans_budget, temperature=TEMPERATURE)
        except Exception as exc:  # network/server boundary -> a datum, not a crash
            ok, text = False, f"complete_text_error: {type(exc).__name__}: {exc}"
        pred = parse_grid(text, shape) if ok else None
        next_grid = np.asarray(q.next_grid)
        before = np.asarray(q.grid)
        changed = not np.array_equal(before, next_grid)
        parse_ok = pred is not None
        exact = bool(parse_ok and pred.shape == next_grid.shape and np.array_equal(pred, next_grid))
        identity = bool(parse_ok and pred.shape == before.shape and np.array_equal(pred, before))
        shape_ok = bool(parse_ok and pred.shape == next_grid.shape)
        per_trans.append(
            {
                "i": i,
                "action": int(q.action),
                "changed": changed,
                "server_ok": bool(ok),
                "parse_ok": parse_ok,
                "shape_ok": shape_ok,
                "exact": exact,
                "identity_copy": identity,
                "wall_s": round(time.time() - t0, 2),
                "stop_type": getattr(prop, "last_stop_type", ""),
                "prompt_truncated": bool(getattr(prop, "last_prompt_truncated", False)),
            }
        )

    def _rate(pred_key: str, subset: Optional[str] = None) -> Optional[float]:
        rows = per_trans
        if subset == "changing":
            rows = [r for r in per_trans if r["changed"]]
        elif subset == "noop":
            rows = [r for r in per_trans if not r["changed"]]
        elif subset == "parseable":
            rows = [r for r in per_trans if r["parse_ok"]]
        if not rows:
            return None
        return round(float(np.mean([bool(r[pred_key]) for r in rows])), 4)

    n = len(per_trans)
    n_changing = sum(1 for r in per_trans if r["changed"])
    return {
        "game": game,
        "trial": trial,
        "grid_shape": list(shape),
        "n_transitions": n,
        "n_changing": n_changing,
        "n_noop": n - n_changing,
        "k_examples_used": k_use,
        "answer_budget_tokens": ans_budget,
        "n_ctx": n_ctx,
        # PRIMARY metric -- exact full-grid match over ALL window transitions (leave-one-out), the
        # SAME definition/denominator as WorldModelVerifier.accuracy / heldout_accuracy.
        "heldout_accuracy": _rate("exact"),
        "heldout_accuracy_changing": _rate("exact", "changing"),
        "heldout_accuracy_noop": _rate("exact", "noop"),
        "heldout_accuracy_parseable_only": _rate("exact", "parseable"),
        "parse_success_rate": _rate("parse_ok"),
        "shape_match_rate": _rate("shape_ok"),
        "identity_copy_rate": _rate("identity_copy"),
        "identity_copy_rate_changing": _rate("identity_copy", "changing"),
        "n_truncated": sum(1 for r in per_trans if r["stop_type"] == "limit"),
        "n_prompt_truncated": sum(1 for r in per_trans if r["prompt_truncated"]),
        "per_transition": per_trans,
    }


# ---------------------------------------------------------------------------
# Shard IO (resumable) -- mirrors the sibling scripts
# ---------------------------------------------------------------------------
def _load_shard() -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    if SHARD.exists():
        for line in SHARD.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[(r["game"], int(r["trial"]))] = r
    return rows


def _append_shard(row: dict[str, Any]) -> None:
    SHARD.parent.mkdir(parents=True, exist_ok=True)
    with SHARD.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _write_blocked_artifact(precond: dict[str, Any], duration_s: float) -> None:
    missing = [c["resource"] for c in precond["checks"] if not c["available"]]
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(
        json.dumps(
            {
                "experiment": "experiment_5768_direct_incontext_prediction_ab",
                "schema": "carnot.exp5768.direct_incontext_prediction_ab.v1",
                "requirements": ["REQ-ARC-WMTE-5768"],
                "honest_verdict": f"blocked_{'_'.join(missing)[:80]}",
                "inference_substrate": "live_llm_inference",
                "preconditions_checked": precond["checks"],
                "solve_provenance": "development_proxy",
                "verifier_is_oracle": False,
                "duration_s": round(duration_s, 2),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


# ---------------------------------------------------------------------------
# Run loop -- single gemma server on GPU 1, sequential cells
# ---------------------------------------------------------------------------
def run_all() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from carnot.agentic import arc_actions_to_progress as atp

    done = _load_shard()
    total = len(ROSTER) * len(TRIALS)
    log(f"resume: {len(done)}/{total} cells already in shard")
    log(f"building {len(ROSTER)} windows...")
    windows: dict[str, Any] = {}
    for game in ROSTER:
        w = atp.build_progress_window(game)
        windows[game] = w
        if w is None:
            log(f"SKIP {game}: no offline L1 window")
        else:
            shp = np.asarray(w[0][0].grid).shape if w[0] else None
            log(f"  window {game}: n_trans={len(w[0])} shape={shp} cell={w[2]}")

    pending = [
        (game, t)
        for game in ROSTER
        for t in TRIALS
        if windows.get(game) is not None and (game, t) not in done
    ]
    server_meta: dict[str, Any] = {"n_ctx": None, "vram_before": None, "vram_after": None}
    if not pending:
        log("all cells present, skipping inference")
        return list(done.values()), server_meta

    log(f"=== gemma-4-31B-it DIRECT prediction : {len(pending)} cells | CUDA={GEMMA['cuda_visible']} ===")
    proc = None
    try:
        proc, n_ctx, v_before, v_after = launch_server_ladder()
        server_meta = {"n_ctx": n_ctx, "vram_before": v_before, "vram_after": v_after}
        log(f"  deployed n_ctx={n_ctx}; gpu0={_gpu_mem_used_mib(0)} gpu1={_gpu_mem_used_mib(1)} MiB")
        prop = make_gemma_proposer(n_ctx)
        for game, t in pending:
            window, _full_traj, cell = windows[game]
            log(f"RUN gemma31b-predict {game} trial={t}")
            c0 = time.time()
            try:
                row = run_prediction_cell(game, prop, trial=t, window=window, cell=cell, n_ctx=n_ctx)
            except Exception as exc:
                row = {
                    "game": game,
                    "trial": t,
                    "error": f"cell_crash: {type(exc).__name__}: {exc}"[:300],
                    "heldout_accuracy": None,
                    "parse_success_rate": None,
                }
            row["arm"] = "gemma31b_direct_prediction"
            row["game"] = game
            row["trial"] = t
            row["server_n_ctx"] = server_meta["n_ctx"]
            _append_shard(row)
            done[(game, t)] = row
            log(
                f"  -> heldout={row.get('heldout_accuracy')} "
                f"changing={row.get('heldout_accuracy_changing')} "
                f"parse={row.get('parse_success_rate')} "
                f"identity={row.get('identity_copy_rate')} "
                f"blocked={row.get('blocked')} ({time.time() - c0:.0f}s)"
            )
    finally:
        terminate(proc)
    return list(done.values()), server_meta


# ---------------------------------------------------------------------------
# gemma-4-31B code-synthesis single-shot baseline (REQ-ARC-WMTE-5764) side-by-side
# ---------------------------------------------------------------------------
def _code_baseline_by_game() -> tuple[dict[str, Optional[float]], Optional[float]]:
    """Per-game + pooled heldout from REQ-ARC-WMTE-5764's artifact (code-synthesis single-shot)."""
    by_game: dict[str, Optional[float]] = {}
    pooled = None
    try:
        art = json.loads(GEMMA_CODE_BASELINE_ARTIFACT.read_text())
        by_game = dict(art.get("heldout_accuracy_by_game") or {})
        pooled = (art.get("comparison_to_thinkingcap27_baseline") or {}).get(
            "gemma31b_pooled_mean_heldout"
        )
    except Exception:
        pass
    if pooled is None:
        pooled = GEMMA_CODE_BASELINE_POOLED
    return by_game, pooled


def _mean(xs: list[float]) -> Optional[float]:
    return round(float(np.mean(xs)), 6) if xs else None


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------
def build_artifact(
    duration_s: float, precond: dict[str, Any], server_meta: dict[str, Any]
) -> dict[str, Any]:
    rows = [r for r in _load_shard().values() if r.get("arm") == "gemma31b_direct_prediction"]

    def _by_game(key: str) -> dict[str, Optional[float]]:
        acc: dict[str, list[float]] = {}
        for r in rows:
            v = r.get(key)
            if isinstance(v, (int, float)):
                acc.setdefault(r["game"], []).append(float(v))
        return {g: _mean(v) for g, v in sorted(acc.items())}

    heldout_by_game = _by_game("heldout_accuracy")
    heldout_changing_by_game = _by_game("heldout_accuracy_changing")
    parse_by_game = _by_game("parse_success_rate")
    identity_by_game = _by_game("identity_copy_rate")

    pooled_vals = [m for m in heldout_by_game.values() if m is not None]
    pooled_mean = _mean(pooled_vals)
    pooled_max = round(max(pooled_vals), 6) if pooled_vals else None
    games_nonzero = sum(1 for m in pooled_vals if m is not None and m > 1e-9)

    changing_vals = [m for m in heldout_changing_by_game.values() if m is not None]
    pooled_changing = _mean(changing_vals)
    parse_vals = [m for m in parse_by_game.values() if m is not None]
    pooled_parse = _mean(parse_vals)
    identity_vals = [m for m in identity_by_game.values() if m is not None]
    pooled_identity = _mean(identity_vals)

    code_by_game, code_pooled = _code_baseline_by_game()
    code_vals = [v for g, v in code_by_game.items() if g in ROSTER and isinstance(v, (int, float))]
    code_pooled_on_roster = _mean([float(v) for v in code_vals]) if code_vals else code_pooled

    per_game_cmp = {}
    for g in ROSTER:
        per_game_cmp[g] = {
            "direct_prediction_heldout": heldout_by_game.get(g),
            "direct_prediction_heldout_changing": heldout_changing_by_game.get(g),
            "code_synthesis_heldout": code_by_game.get(g),
            "parse_success_rate": parse_by_game.get(g),
            "identity_copy_rate": identity_by_game.get(g),
        }

    pooled_delta = (
        round(pooled_mean - code_pooled, 6)
        if (pooled_mean is not None and code_pooled is not None)
        else None
    )

    n_blocked = sum(1 for r in rows if r.get("blocked"))
    blocked_games = sorted({r["game"] for r in rows if r.get("blocked")})

    # ---- honest verdict (terminal-prefixed, numbers-first) + recommendation.
    # A direct-prediction "win" is judged CONSERVATIVELY: pooled leave-one-out heldout must exceed
    # the code baseline AND the CHANGING-only heldout must be non-trivial (so the win is real
    # dynamics prediction, not identity-copying no-ops).
    beats_code = (
        pooled_mean is not None
        and code_pooled is not None
        and pooled_delta is not None
        and pooled_delta > 0.05
    )
    changing_real = pooled_changing is not None and pooled_changing > 0.05
    if not pooled_vals:
        branch = "no_data"
        verdict = "complete_direct_incontext_prediction_no_cells_completed_see_errors_or_blocked"
        recommendation = "no scored cells produced -- inspect the shard/log/blocked before any follow-up."
    elif beats_code and changing_real:
        branch = "promising_direct_prediction_beats_code_synthesis"
        verdict = (
            f"complete_direct_incontext_prediction_pooled_heldout_{pooled_mean}_"
            f"vs_code_synthesis_{code_pooled}_delta_{pooled_delta}_"
            f"changing_{pooled_changing}_parse_{pooled_parse}_beats_code_N{len(pooled_vals)}"
        )
        recommendation = (
            f"PROMISING: direct in-context prediction pooled leave-one-out heldout {pooled_mean} "
            f"exceeds gemma-4-31B code-synthesis {code_pooled} (delta {pooled_delta}), and the "
            f"CHANGING-only heldout {pooled_changing} confirms the win is real dynamics prediction "
            f"(identity-copy rate {pooled_identity}), not no-op copying. Direct prediction surfaces "
            f"latent in-context transition capability the code-writing step was suppressing. WORTH "
            f"investigating wiring this as an alternative world-model source in the live plan_in_model "
            f"path, or the specialized small-predictor-model follow-up. OPERATOR-ONLY whether to invest."
        )
    else:
        branch = "direct_prediction_does_not_beat_code_synthesis"
        verdict = (
            f"complete_direct_incontext_prediction_pooled_heldout_{pooled_mean}_"
            f"vs_code_synthesis_{code_pooled}_delta_{pooled_delta}_"
            f"changing_{pooled_changing}_parse_{pooled_parse}_no_win_N{len(pooled_vals)}"
        )
        recommendation = (
            f"NOT A WIN: direct in-context prediction pooled leave-one-out heldout {pooled_mean} "
            f"(changing-only {pooled_changing}, parse-success {pooled_parse}, identity-copy "
            f"{pooled_identity}) does not clear the gemma-4-31B code-synthesis baseline {code_pooled} "
            f"by a meaningful margin (delta {pooled_delta}). Extends GAP-ARC-INDUCTION-REFINEMENT-NULL "
            f"with a fourth negative: neither a bigger model, CEGIS refinement, nor a no-code "
            f"direct-prediction paradigm moves single-shot induction quality off its floor on these "
            f"near-optimal-baseline corpora. Do NOT re-propose direct-prediction variants without a "
            f"different corpus/mechanic-class prior. OPERATOR-ONLY regardless."
        )

    return {
        "experiment": "experiment_5768_direct_incontext_prediction_ab",
        "schema": "carnot.exp5768.direct_incontext_prediction_ab.v1",
        "requirements": ["REQ-ARC-WMTE-5768"],
        "prior_work_extended": [
            {
                "req": "REQ-ARC-WMTE-5764",
                "relation": "gemma-4-31B-it CODE-SYNTHESIS single-shot induction baseline (LLM writes "
                "engine(grid,action,data) Python), pooled heldout 0.378487 on this exact 13-game roster "
                "-- the DIRECT comparison arm. What is mechanistically DIFFERENT here: NO code is "
                "generated at any point; the model's raw text output IS the predicted grid, parsed and "
                "exact-matched against the true next_grid. This tests whether the code-EXTERNALIZATION "
                "step (not the pattern inference) is the binding wall.",
                "verdict": "complete_gemma31b_singleshot_induction_pooled_heldout_0.378487",
            },
            {
                "req": "REQ-ARC-WMTE-5760",
                "relation": "CEGIS refinement of the induced engine() CODE (iterative counterexample-"
                "guided repair) -- pooled delta_heldout -0.0128, zero games improved. DIFFERENT here: "
                "no code to refine; a single direct-prediction pass per held-out transition.",
                "verdict": "complete_cegis_refinement_partial_pooled_delta_-0.0128_does_not_cleanly_meet_a_preregistered_branch",
            },
            {
                "req": "REQ-ARC-WMTE-5766",
                "relation": "CEGIS refinement on gemma-4-31B specifically -- pooled delta_heldout "
                "-0.0598, zero games improved. Same DIFFERENCE: this experiment removes code generation "
                "entirely rather than refining it.",
                "verdict": "complete_cegis_refinement_partial_pooled_delta_-0.0598_does_not_cleanly_meet_a_preregistered_branch",
            },
        ],
        "question": (
            "Does asking the SAME local model (gemma-4-31B-it) to DIRECTLY predict the next grid "
            "in-context (no code, no function) outperform asking it to write general Python code "
            "implementing the transition rule, on the SAME 13 games / 3 trials / same window / same "
            "exact-match metric? A win would show the code-EXTERNALIZATION step, not pattern "
            "inference, is the binding wall the induction-quality diagnosis found."
        ),
        "inference_substrate": "live_llm_inference",
        "honest_verdict": verdict,
        "gate_branch": branch,
        "recommendation": recommendation,
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "read_game_source": False,
        "used_env_source": True,
        "submitted_to_leaderboard": False,
        "random_seed": TRIALS[0],
        "trials_per_game": len(TRIALS),
        "model_specs": [
            {
                "name": GEMMA["repo_substr"],
                "hf_id": GEMMA["hf_id"],
                "quant": "Q4_K_M",
                "gguf_path": GEMMA["gguf"],
                "role": "direct in-context next-grid predictor (NO code generation)",
                "kv_quant": GEMMA["kv_quant"],
                "use_chat_template": True,
                "n_ctx_deployed": server_meta.get("n_ctx"),
                "temperature": TEMPERATURE,
                "k_examples_cap": K_EXAMPLES,
                "server": (
                    f"CUDA llama-server single-GPU (CUDA_VISIBLE_DEVICES={GEMMA['cuda_visible']}), "
                    f"-ngl 999, q8_0 KV, port {GEMMA['port']}, /v1/chat/completions"
                ),
                "vram_gpu_before_after_mib": [
                    server_meta.get("vram_before"),
                    server_meta.get("vram_after"),
                ],
            }
        ],
        "prompt_format_used": {
            "paradigm": "direct_incontext_grid_prediction_no_code",
            "grid_serialization": "row-major, one line per row, space-separated integers (LOSSLESS; "
            "handles colors > 9, unlike to_ascii's last-digit form)",
            "example_selection": "LEAVE-ONE-OUT: up to k=8 in-context (before,action,data)->after "
            "examples drawn from the window EXCLUDING the query transition (prefer changing, keep <=2 "
            "no-ops); the query transition is NEVER shown -> zero verbatim-answer leakage.",
            "query": "BEFORE grid + ACTION int + DATA (click pixel dict or none); the model outputs "
            "ONLY the predicted AFTER grid as H rows of W space-separated integers.",
            "action_encoding": "ACTION <int>; ACTION 6 is a click with DATA {'x':px,'y':py} in pixel "
            "coords (one logical cell = <cell> pixels); other actions DATA none.",
            "temperature": TEMPERATURE,
            "parse": "extract maximal contiguous blocks of equal-width integer rows, choose the block "
            "best matching the expected shape (exact preferred, else largest, last-wins on ties); "
            "parse_success = a rectangular int grid was produced (wrong-shape counts as parsed-but-"
            "wrong, mirroring WorldModelVerifier's pred.shape==next.shape AND np.array_equal).",
            "template_example": (
                "You are given observed state transitions of the ARC-AGI-3 game '<game>'. Each state "
                "is a HxW integer grid (colors [...]). An ACTION transforms BEFORE into AFTER. ... "
                "EXAMPLE 1\\nBEFORE:\\n<grid>\\nACTION 6  DATA {'x':.., 'y':..}\\nAFTER:\\n<grid>\\n\\n"
                "... QUERY\\nBEFORE:\\n<grid>\\nACTION 3  DATA none\\n\\nOutput ONLY the predicted AFTER "
                "grid as exactly H rows of W space-separated integers. No explanation, no code, no "
                "other text.\\nAFTER:\\n"
            ),
        },
        "parse_success_rate": pooled_parse,
        "heldout_accuracy_by_game": heldout_by_game,
        "heldout_accuracy_changing_by_game": heldout_changing_by_game,
        "identity_copy_rate_by_game": identity_by_game,
        "parse_success_rate_by_game": parse_by_game,
        "comparison_to_gemma31b_code_synthesis_baseline": {
            "note": (
                "direct in-context prediction (leave-one-out, NO code) vs REQ-ARC-WMTE-5764's "
                "gemma-4-31B CODE-SYNTHESIS single-shot induction, SAME 13-game roster, SAME "
                "exact-match heldout metric, SAME model. Leave-one-out never shows the model the "
                "queried transition, so this is a STRICTER test than the code baseline (whose induced "
                "engine may still memorize its k=8 shown transitions) -- a direct-prediction win is "
                "therefore conservative."
            ),
            "direct_prediction_pooled_mean_heldout": pooled_mean,
            "direct_prediction_pooled_max_game_heldout": pooled_max,
            "direct_prediction_pooled_mean_heldout_changing": pooled_changing,
            "direct_prediction_nonzero_games": games_nonzero,
            "code_synthesis_pooled_mean_heldout": code_pooled,
            "code_synthesis_pooled_mean_heldout_on_roster": code_pooled_on_roster,
            "pooled_mean_delta_direct_minus_code": pooled_delta,
            "n_games_compared": len(pooled_vals),
            "per_game": per_game_cmp,
        },
        "identity_copy_analysis": {
            "note": (
                "The top adversarial concern: a high parse+heldout could be near-identity-copying the "
                "input grid rather than predicting real dynamics. identity_copy_rate = fraction of "
                "predictions equal to the INPUT (before) grid; heldout_accuracy_changing isolates the "
                "state-CHANGING transitions where identity-copy necessarily FAILS -- that is the honest "
                "'did it predict real dynamics' number."
            ),
            "pooled_identity_copy_rate": pooled_identity,
            "pooled_heldout_changing": pooled_changing,
            "pooled_heldout_all": pooled_mean,
        },
        "attribution": {
            "n_cells": len(rows),
            "n_blocked_grid_too_large": n_blocked,
            "blocked_games": blocked_games,
            "pooled_parse_success_rate": pooled_parse,
            "note": (
                "parse_success_rate is a NEW failure mode code-synthesis lacked: free-text grid output "
                "can be malformed/truncated/wrong-shape. A parse failure counts as an incorrect "
                "prediction but is reported separately; heldout_accuracy_parseable_only (per-cell) is "
                "the conditional accuracy over parseable predictions. Grids too large to fit "
                "in-context (n_ctx) are recorded blocked, NOT silently scored 0."
            ),
        },
        "field_principles": {
            "honest_verdict": "terminal-prefixed, numbers-first; a continuous leave-one-out heldout "
            "mean with real headroom (baseline pooled 0.378487) cannot come back 'no headroom'.",
            "inference_substrate": "live_llm_inference -- real gemma-4-31B-it Q4_K_M GGUF generation on "
            "a CUDA llama-server; 60s duration floor. Runtime VRAM-jump assertion refuses CPU fallback.",
            "solve_provenance": "development_proxy -- PUBLIC-game offline measurement of the induction "
            "INPUT the live mechanism uses, with the METHOD swapped to direct prediction; NOT a "
            "hidden-game self-discovery solve and NOT a live-path modification (nothing writes "
            "world_model.py or plans/executes).",
            "verifier_is_oracle": "False -- heldout_accuracy is exact-match against real recorded "
            "transitions; oracle-distinct from any win oracle.",
            "random_seed": "LLM sampling is server-side stochastic (temperature 0.2); trials are "
            "per-game replicates (same window), NOT independent game-level samples -- reported per "
            "game + pooled.",
            "reproducibility_checksum": "content hash over harness + induce/e3 code + generator/roster "
            "config + rows.",
            "duration_s": "real wall-clock; the 60s floor guards against a fabricated fast run.",
            "prior_work_extended": "traces this to REQ-ARC-WMTE-5764 (code-synthesis baseline) + "
            "5760/5766 (CEGIS), stating precisely what is mechanistically different (no code generated).",
            "prompt_format_used": "the exact in-context prediction template + grid serialization, so "
            "the paradigm is reproducible and auditable (a NEW-mechanism claim needs its mechanism "
            "written down).",
            "parse_success_rate": "a NEW failure mode code-synthesis lacked; reported separately so a "
            "parse failure is never silently conflated with a genuine wrong-dynamics prediction.",
            "heldout_accuracy_by_game": "the PRIMARY signal -- does no-code direct prediction move the "
            "exact quantity (exact full-grid match) the diagnosis named as the binding wall, per game?",
            "identity_copy_rate": "guards the top adversarial concern -- distinguishes real dynamics "
            "prediction from near-identity-copying the input grid on no-op-heavy windows.",
            "recommendation": "screening call ONLY (does direct prediction show real promise vs code "
            "synthesis?); whether to invest further (live plan_in_model wiring / specialized predictor) "
            "is OPERATOR-ONLY.",
        },
        "preconditions_checked": precond["checks"],
        "sample_size": {
            "games": len(pooled_vals),
            "roster_n": len(ROSTER),
            "roster": ROSTER,
            "trials_per_game": len(TRIALS),
            "paired_unit": "game (heldout averaged over trials, compared by game vs the code baseline)",
            "note": (
                "SAME 13-game pre-registered roster + 3-trial count as REQ-ARC-WMTE-5760/5764/5766 "
                "(ROSTER/TRIALS imported from exp5760 to guarantee an exact match). Each cell "
                "leave-one-out predicts EVERY window transition, so the per-cell heldout aggregates "
                "over all ~N window transitions, not a single grid."
            ),
        },
        "methodology_note": (
            "DIRECT in-context grid prediction: no code generated at any point. Per (game, trial) the "
            "SAME build_progress_window window (from exp5717.build_window, imported via "
            "arc_actions_to_progress) is used. For EACH window transition (leave-one-out), up to k=8 "
            "OTHER transitions are serialized as (before-grid, action, data)->after-grid in-context "
            "examples (query never shown -> zero answer leakage), the model is asked to output ONLY the "
            "predicted after-grid as rows of space-separated integers, the output is parsed back into a "
            "grid, and scored by exact full-grid np.array_equal against the true next_grid -- the SAME "
            "metric definition WorldModelVerifier.accuracy / heldout_accuracy uses. gemma-4-31B-it "
            "served via /v1/chat/completions (its embedded chat template) on GPU "
            f"{GPU_INDEX} (CUDA_VISIBLE_DEVICES={GEMMA['cuda_visible']}), temperature {TEMPERATURE} "
            "(matching the code-synthesis induce first-try temperature), n_ctx by the exp5764 launch "
            "ladder. Grids too large to fit in-context are recorded blocked (disclosed). This is a "
            "DIAGNOSTIC over the same induction input the live mechanism uses with the induction METHOD "
            "swapped -- NOT a live-path modification (no world_model.py written, no plan/execute) and "
            "NOT an orphan solver."
        ),
        "duration_s": round(duration_s, 2),
        "reproducibility_checksum": _repro_checksum(rows, server_meta),
    }


def _repro_checksum(rows: list[dict[str, Any]], server_meta: dict[str, Any]) -> str:
    from carnot.agentic import arc_executable_world_model as e3

    h = hashlib.sha256()
    for mod_file in (
        __file__,
        e3.__file__,
        REPO / "python" / "carnot" / "experiment_5764_gemma31b_singleshot_induction_ab.py",
        REPO / "python" / "carnot" / "agentic" / "arc_actions_to_progress.py",
    ):
        try:
            h.update(Path(mod_file).read_bytes())
        except Exception:
            pass
    h.update(
        json.dumps(
            {
                "roster": ROSTER,
                "trials": TRIALS,
                "k_examples": K_EXAMPLES,
                "temperature": TEMPERATURE,
                "gpu_index": GPU_INDEX,
                "gemma": {k: v for k, v in GEMMA.items() if k != "gguf"},
                "server_meta": server_meta,
            },
            sort_keys=True,
            default=str,
        ).encode()
    )
    h.update(json.dumps(sorted(json.dumps(r, sort_keys=True, default=str) for r in rows)).encode())
    return "sha256:" + h.hexdigest()


def main() -> int:
    started = time.time()
    precond = check_preconditions()
    for c in precond["checks"]:
        log(f"PRECOND {c['resource']}: available={c['available']} {c['detail']}")
    if not precond["all_ok"]:
        log("PRECONDITIONS FAILED -- writing blocked artifact and STOPPING (no inference).")
        _write_blocked_artifact(precond, time.time() - started)
        return 0
    log("preconditions OK -- starting gemma-4-31B DIRECT in-context prediction run.")
    _, server_meta = run_all()
    artifact = build_artifact(time.time() - started, precond, server_meta)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    log(f"DONE. verdict={artifact['honest_verdict']}")
    log(f"artifact -> {ARTIFACT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

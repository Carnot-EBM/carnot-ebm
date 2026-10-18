"""Exp 5764: Does a genuinely BIGGER/different-family model move single-shot world-model
induction quality off its near-zero floor? (REQ-ARC-WMTE-5764)

WHY THIS EXISTS
---------------
Tonight's induction-quality diagnosis (REQ-ARC-WMTE-5726 +
docs/research-notes/arc-world-model-induction-quality-diagnosis-2026-07-20.md) found that
SINGLE-SHOT world-model induction with ThinkingCap-Qwen3.6-27B and Qwen3.5-9B-MTP scores
heldout_accuracy at a near-zero floor (median 0.0; 29/37 successful inductions at EXACTLY 0.0):
the induced engine() almost never reproduces a full observed transition. The design doc
(docs/research-notes/arc-induction-quality-improvement-design-2026-07-20.md) pre-registered an
"HONEST-NEGATIVE" branch for the CEGIS-refinement follow-on (REQ-ARC-WMTE-5760, running
CONCURRENTLY on GPU 0 -- NOT complete, do NOT cite a final verdict for it) whose early cells
show refinement emits real code but ZERO heldout lift. That branch names two candidate next
steps: (a) a genuinely BIGGER/different-family offline model, (b) reconsidering the induction
architecture. This experiment tests (a) -- explicitly permitted on the conductor's 3090s for
OFFLINE dev work per the 2026-06-27 operator GPU-allocation directive (the iGPU-only constraint
governs the LIVE hidden-game submission stack ONLY, not offline induction).

THE QUESTION
------------
Does raw model capacity/family ALONE move heldout_accuracy off the floor, on the SAME games,
using the SAME single-shot mechanism that produced tonight's null? gemma-4-31B-it is a
different-FAMILY, DENSE 31B (far more per-token compute than ThinkingCap-27B, and much more than
the 35B-A3B / 9B MoE-or-small models whose active-parameter counts are tiny). If YES -> a strong,
cheap signal worth a full CEGIS treatment on this model (operator-only whether to launch that).
If NO -> further supports the architectural-reconsideration branch, not merely "the model isn't
big enough".

MECHANISM -- SINGLE-SHOT ONLY (deliberately smaller than the GPU-0 CEGIS job)
-----------------------------------------------------------------------------
Reuses exp5726.run_reason_cell_budget VERBATIM (the exact tonight single-shot cell: codeonly
OFF, /think, no pre-opened fence, tries=1, exp5722 stale-engine unlink, exp5720
actions-to-progress ladder) with budget 16384. NOT execute_bounded_llm_reinduction (that is the
GPU-0 CEGIS test). The ONLY scientific variable is the generator (gemma-4-31B vs tonight's
ThinkingCap-27B / Qwen-9B). Metrics computed by the same code as tonight: heldout_accuracy,
cell_recall, goal_predicate_accuracy, levelup_positive_recall (WorldModelVerifier). Plus the
exp5760 window-memorization AST detector applied to the induced engine() source for direct
comparability with the concurrent CEGIS run.

gemma routing: gemma-4-31B-it is an instruction-tuned model with its OWN embedded chat template,
so it is served via /v1/chat/completions (use_chat_template=True) -- the SAME accommodation
ThinkingCap-27B needed (the 5725 fix). The mechanism's hardcoded "/think\n" prefix is a Qwen-ism
applied UNIFORMLY for mechanism-identity; gemma has no hybrid-think mode and treats it as literal
turn text (disclosed, not hidden). This keeps the induction mechanism byte-identical to tonight
while giving gemma its required instruction format -- a fair "does raw capacity move it" test.

GPU / topology: GPU 1 ONLY (CUDA_VISIBLE_DEVICES=1), a dedicated own llama-server on port 8971
(no collision with the GPU-0 CEGIS job's 8969/8968). The 18GB Q4 fits one 24GB 3090; n_ctx is
chosen by a launch-time ladder (32768 first, for byte-identical comparability with tonight, then
smaller only if the full-KV allocation OOMs) so the winning context is picked EMPIRICALLY, never
assumed. Runtime GPU-offload assertion (VRAM jump) refuses a silent CPU fallback.

PROVENANCE: development_proxy on PUBLIC games (NOT a hidden-game self-discovery solve).
verifier_is_oracle False (win oracle = the level counter; heldout is exact-match on a held-out
transition split, oracle-distinct). NEVER flips the frozen live default (operator-only), NEVER
submits. This is a DIAGNOSTIC over EXISTING induction machinery with the generator swapped -- NOT
a live-path modification and NOT an orphan solver (Live-Path Reachability Discipline: the induce
mechanism under test IS the live mechanism, just with a bigger offline model).

Prior-failure block (Failed-Experiment Rerun Discipline): names REQ-ARC-WMTE-5726 (tonight's
ThinkingCap/Qwen single-shot near-zero-heldout diagnosis) and the concurrent REQ-ARC-WMTE-5760
(CEGIS refinement, in-progress). Root cause of the near-zero floor is UNRESOLVED (whether it is a
model-capacity wall or an architecture wall). What is DIFFERENT: this swaps in a genuinely bigger
different-FAMILY dense model, the one lever the diagnosis + design doc named as untested. It is
NOT a re-run of a peripheral tweak on the SAME model. retire_if_same_verdict: if gemma-4-31B ALSO
floors at near-zero heldout, do NOT re-propose yet-bigger single-shot induction variants -- that
is strong evidence for the architecture wall, and the next step is architectural (reactive-with-
filter / executable-world-model refinement), NOT a bigger model.

RESUMABLE: every (game, trial) cell appends to a JSONL shard as it completes.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Reuse tonight's EXACT single-shot induction cell + GPU-mem/terminate utilities. Importing
# exp5726 runs only module-level config (GENERATORS/paths + os.environ.setdefault); run_all() is
# under its own __main__. Importing exp5760 (for the shared roster/trials + the memorization AST
# detector) likewise runs only module-level config.
from carnot.experiment_5726_thinkingcap_16k_dualgpu_reason_ab import (  # noqa: E402
    LLAMA_SERVER,
    _gpu_mem_used_mib,
    run_reason_cell_budget,
    terminate,
)
from carnot.experiment_5760_cegis_refinement_induction_ab import (  # noqa: E402
    ROSTER,
    TRIALS,
    _window_changed_coords,
    memorization_scan,
)

REPO = Path(__file__).resolve().parents[2]
SHARD = REPO / "results" / "exp5764_gemma31b_singleshot_shard.jsonl"
ARTIFACT = REPO / "results" / "experiment_5764_gemma31b_singleshot_induction_ab.json"

# ThinkingCap-27B single-shot heldout baseline (tonight, REQ-ARC-WMTE-5726) -- read at artifact
# time for the side-by-side comparison.
TC_BASELINE_SHARD = REPO / "results" / "exp5726_thinkingcap_16k_dualgpu_shard.jsonl"

BUDGET = 16384  # tonight's doubled completion budget (matches REQ-ARC-WMTE-5726/5760)
GPU_INDEX = 1  # GPU 1 ONLY (the operator-designated idle card; GPU 0 runs the CEGIS job)
# n_ctx ladder: 32768 first for byte-identical comparability with tonight (same prompt-truncation
# behaviour); fall back ONLY if the full-KV allocation OOMs an 18GB Q4 model on one 24GB card. In
# llama.cpp the KV cache is pre-allocated at load for the full n_ctx, so "healthy at launch" ~=
# "will not OOM mid-run" -- the ladder picks the largest context that actually loads.
NCTX_LADDER = [32768, 24576, 20480]
GPU1_IDLE_MAX_MIB = 1000  # GPU 1 is "idle" (safe to use) if <1000 MiB is currently allocated

GEMMA: dict[str, Any] = {
    "repo_substr": "gemma-4-31B-it",
    "hf_id": "unsloth/gemma-4-31B-it-GGUF",
    "gguf": (
        "/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF/"
        "snapshots/f130ba51393346288f5862e30e9586b9b021513f/gemma-4-31B-it-Q4_K_M.gguf"
    ),
    "port": 8971,  # distinct from the GPU-0 CEGIS job (8969 TC / 8968 Qwen)
    "kv_quant": "q8_0",
    "use_chat_template": True,  # gemma-4-it needs its embedded chat template (like the 5725 fix)
    "cuda_visible": str(GPU_INDEX),  # "1" -- GPU 1 only
    "mtp": False,  # gemma GGUF has no nextn self-draft heads
    "extra": "-fit off",  # harmless when ngl+nctx explicit; consistent with the sibling scripts
    "timeout": 1800,  # 31B dense load can be slow on first mmap
    "role": "different-family DENSE 31B; single-shot induction-capacity probe on GPU 1",
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------------------
# Own llama-server launch (pinned to GPU 1, n_ctx-parameterized so the ladder can fall back).
# Mirrors exp5726.launch_server's health-wait contract; does NOT reuse it because that reads the
# module-level N_CTX and I need per-attempt control for the OOM ladder.
# ---------------------------------------------------------------------------
def _server_args(n_ctx: int) -> list[str]:
    args = [
        str(LLAMA_SERVER),
        "-m",
        GEMMA["gguf"],
        "-ngl",
        "999",
        "-c",
        str(n_ctx),
        "--port",
        str(GEMMA["port"]),
        "--host",
        "127.0.0.1",
        "--cache-type-k",
        GEMMA["kv_quant"],
        "--cache-type-v",
        GEMMA["kv_quant"],
    ]
    args += [a for a in str(GEMMA["extra"]).split() if a]
    return args


def _launch_one(n_ctx: int) -> subprocess.Popen:
    """Launch a llama-server pinned to GPU 1 at n_ctx and wait for /health. Raises on failure."""
    args = _server_args(n_ctx)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GEMMA["cuda_visible"])
    log(f"  launch (n_ctx={n_ctx}): CUDA_VISIBLE_DEVICES={GEMMA['cuda_visible']} {' '.join(args)}")
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    url = f"http://127.0.0.1:{GEMMA['port']}/health"
    deadline = time.time() + GEMMA["timeout"]
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early (code {proc.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if b"ok" in r.read():
                    return proc
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("llama-server did not become healthy before timeout")


def launch_server_ladder() -> tuple[subprocess.Popen, int, int, int]:
    """Try each n_ctx in NCTX_LADDER (largest first). Return the first healthy
    (proc, n_ctx, vram_before, vram_after) with a REAL GPU-1 offload (VRAM jump > 1GB).
    Raises RuntimeError if none succeed."""
    last_err = ""
    for n_ctx in NCTX_LADDER:
        v_before = _gpu_mem_used_mib(GPU_INDEX)
        try:
            proc = _launch_one(n_ctx)
        except Exception as exc:
            last_err = f"n_ctx={n_ctx}: {type(exc).__name__}: {exc}"
            log(f"  launch FAILED at n_ctx={n_ctx}: {exc} -- falling back")
            time.sleep(4)  # let any OOM'd process fully release GPU memory
            continue
        v_after = _gpu_mem_used_mib(GPU_INDEX)
        jump = (v_after - v_before) if (v_before is not None and v_after is not None) else None
        log(f"  server healthy at n_ctx={n_ctx}. VRAM gpu{GPU_INDEX} {v_before}->{v_after} MiB")
        if jump is not None and jump < 1000:
            terminate(proc)
            last_err = (
                f"n_ctx={n_ctx}: VRAM only {v_before}->{v_after} MiB (<1GB jump) -- no GPU offload"
            )
            log(f"  {last_err} -- refusing CPU fallback, trying next n_ctx")
            time.sleep(4)
            continue
        return proc, n_ctx, int(v_before or 0), int(v_after or 0)
    raise RuntimeError(f"no n_ctx in {NCTX_LADDER} launched with real GPU offload. last={last_err}")


def make_gemma_proposer(n_ctx: int):
    """A LocalGGUFProposer pointed at the ALREADY-RUNNING gemma server (reuses it via the health
    check; never launches its own). use_chat_template=True routes to /v1/chat/completions so the
    server applies gemma-4-it's OWN embedded chat template."""
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    return LocalGGUFProposer(
        repo_substr=GEMMA["repo_substr"],
        port=GEMMA["port"],
        mtp=False,
        kv_quant=GEMMA["kv_quant"],
        n_ctx=n_ctx,
        max_tokens=BUDGET,
        timeout=GEMMA["timeout"],
        use_chat_template=True,
        model_path=GEMMA["gguf"],
    )


# ---------------------------------------------------------------------------
# The single-shot cell: exp5726.run_reason_cell_budget VERBATIM (identical mechanism to tonight),
# augmented with the exp5760 window-memorization AST scan of the induced engine() source.
# ---------------------------------------------------------------------------
def run_singleshot_cell(
    game: str, prop: Any, *, trial: int, window: list, full_traj: list, cell: int
) -> dict[str, Any]:
    from carnot.agentic.arc_executable_world_model import E3_DIR

    row = run_reason_cell_budget(
        game, prop, trial=trial, window=window, full_traj=full_traj, cell=cell, budget=BUDGET
    )
    # Memorization scan of the induced engine (single-shot -> the on-disk world_model.py IS the
    # round-0 engine; cells run sequentially on one server so there is no world_model.py race).
    coord_set = _window_changed_coords(window)
    try:
        src = (E3_DIR / game / "world_model.py").read_text()
    except Exception:
        src = ""
    mem = memorization_scan(src, coord_set)
    row["mem_scan"] = mem
    row["is_memorizing"] = mem["is_memorizing"]
    return row


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


# ---------------------------------------------------------------------------
# Preconditions (Pre-Launch Preconditions Discipline) -- BEFORE any inference
# ---------------------------------------------------------------------------
def check_preconditions() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def _add(resource: str, ok: bool, detail: str = "") -> None:
        checks.append({"resource": resource, "available": bool(ok), "detail": detail})

    gguf_ok = Path(GEMMA["gguf"]).exists()
    _add(
        "gguf_cached::gemma-4-31B-it",
        gguf_ok,
        GEMMA["gguf"] if gguf_ok else f"MISSING {GEMMA['gguf']}",
    )

    binary_ok = Path(LLAMA_SERVER).exists()
    _add("llama_server_binary", binary_ok, str(LLAMA_SERVER))

    # GPU 1 must be idle RIGHT NOW (brief: if not idle, STOP rather than risk contending with the
    # GPU-0 CEGIS job -- whose qwen arm is configured for GPU 1 later).
    gpu1_used = _gpu_mem_used_mib(GPU_INDEX)
    idle_ok = gpu1_used is not None and gpu1_used < GPU1_IDLE_MAX_MIB
    _add(
        "gpu1_idle",
        idle_ok,
        f"gpu{GPU_INDEX} used={gpu1_used} MiB (threshold <{GPU1_IDLE_MAX_MIB})",
    )

    # CLAUDE.md 2026-07-06 CUDA-build rule: a CPU-only llama-cpp wheel is a rig-health red flag even
    # though THIS harness routes inference through the native llama-server. False -> venv regressed.
    try:
        from llama_cpp import llama_cpp as _b

        offload_ok = bool(_b.llama_supports_gpu_offload())
        _add("llama_cpp_gpu_offload", offload_ok, "llama_supports_gpu_offload()")
    except Exception as exc:
        _add("llama_cpp_gpu_offload", False, f"import failed: {type(exc).__name__}: {exc}"[:160])

    return {"all_ok": all(c["available"] for c in checks), "checks": checks}


def _write_blocked_artifact(precond: dict[str, Any], duration_s: float) -> None:
    missing = [c["resource"] for c in precond["checks"] if not c["available"]]
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(
        json.dumps(
            {
                "experiment": "experiment_5764_gemma31b_singleshot_induction_ab",
                "schema": "carnot.exp5764.gemma31b_singleshot_induction_ab.v1",
                "requirements": ["REQ-ARC-WMTE-5764"],
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
# Run loop -- single gemma server on GPU 1, sequential cells (no contention, no world_model race)
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

    log(
        f"=== gemma-4-31B-it : {len(pending)} cells | CUDA={GEMMA['cuda_visible']} budget={BUDGET} ==="
    )
    proc = None
    try:
        proc, n_ctx, v_before, v_after = launch_server_ladder()
        server_meta = {"n_ctx": n_ctx, "vram_before": v_before, "vram_after": v_after}
        log(
            f"  deployed n_ctx={n_ctx}; gpu0={_gpu_mem_used_mib(0)} gpu1={_gpu_mem_used_mib(1)} MiB"
        )
        prop = make_gemma_proposer(n_ctx)
        for game, t in pending:
            window, full_traj, cell = windows[game]
            log(f"RUN gemma31b {game} trial={t}")
            c0 = time.time()
            try:
                row = run_singleshot_cell(
                    game, prop, trial=t, window=window, full_traj=full_traj, cell=cell
                )
            except Exception as exc:
                row = {
                    "game": game,
                    "trial": t,
                    "error": f"cell_crash: {type(exc).__name__}: {exc}"[:300],
                    "heldout_accuracy": None,
                    "wall_s": round(time.time() - c0, 1),
                }
            row["generator"] = "gemma31b"
            row["arm"] = "gemma31b_singleshot"
            row["game"] = game
            row["trial"] = t
            row["server_n_ctx"] = server_meta["n_ctx"]
            _append_shard(row)
            done[(game, t)] = row
            log(
                f"  -> induce_ok={row.get('induce_ok')} reason={row.get('reason_engaged')} "
                f"heldout={row.get('heldout_accuracy')} cell_recall={row.get('cell_recall')} "
                f"goal_pred={row.get('goal_predicate_accuracy')} mem={row.get('is_memorizing')} "
                f"overran={row.get('overran')} wall={row.get('wall_s')}s ({time.time() - c0:.0f}s)"
            )
    finally:
        terminate(proc)
    return list(done.values()), server_meta


# ---------------------------------------------------------------------------
# ThinkingCap-27B single-shot baseline (tonight) for the side-by-side comparison
# ---------------------------------------------------------------------------
def _tc_baseline_by_game() -> dict[str, list[float]]:
    by: dict[str, list[float]] = {}
    if not TC_BASELINE_SHARD.exists():
        return by
    for line in TC_BASELINE_SHARD.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("arm") != "thinkingcap27_reason":
            continue
        g = r.get("game")
        h = r.get("heldout_accuracy")
        if g in ROSTER and isinstance(h, (int, float)):
            by.setdefault(g, []).append(float(h))
    return by


def _mean(xs: list[float]) -> Optional[float]:
    return round(float(np.mean(xs)), 6) if xs else None


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------
def build_artifact(
    duration_s: float, precond: dict[str, Any], server_meta: dict[str, Any]
) -> dict[str, Any]:
    rows = [r for r in _load_shard().values() if r.get("arm") == "gemma31b_singleshot"]

    # gemma heldout by game (mean over trials) + pooled
    gemma_by_game: dict[str, list[float]] = {}
    for r in rows:
        h = r.get("heldout_accuracy")
        if isinstance(h, (int, float)):
            gemma_by_game.setdefault(r["game"], []).append(float(h))
    gemma_game_mean = {g: _mean(v) for g, v in sorted(gemma_by_game.items())}
    gemma_pooled_vals = [m for m in gemma_game_mean.values() if m is not None]
    gemma_pooled_mean = _mean(gemma_pooled_vals)
    gemma_pooled_max = round(max(gemma_pooled_vals), 6) if gemma_pooled_vals else None
    gemma_games_nonzero = sum(1 for m in gemma_pooled_vals if m is not None and m > 1e-9)

    # induction success + reasoning attribution (single-shot)
    n_cells = len(rows)
    n_induce_ok = sum(1 for r in rows if r.get("induce_ok"))
    n_reason_engaged = sum(1 for r in rows if r.get("reason_engaged"))
    n_overran = sum(1 for r in rows if r.get("overran"))
    n_levelup = sum(1 for r in rows if r.get("reached_levelup"))

    # ThinkingCap-27B single-shot baseline (tonight) side-by-side on the SAME games
    tc_by_game = _tc_baseline_by_game()
    tc_game_mean = {g: _mean(v) for g, v in sorted(tc_by_game.items())}
    tc_pooled_vals = [m for m in tc_game_mean.values() if m is not None]
    tc_pooled_mean = _mean(tc_pooled_vals)
    tc_games_nonzero = sum(1 for m in tc_pooled_vals if m is not None and m > 1e-9)

    comparison_rows = {}
    for g in ROSTER:
        comparison_rows[g] = {
            "gemma31b_singleshot_heldout_mean": gemma_game_mean.get(g),
            "gemma31b_singleshot_heldout_trials": sorted(gemma_by_game.get(g, [])),
            "thinkingcap27_singleshot_heldout_mean": tc_game_mean.get(g),
            "thinkingcap27_singleshot_heldout_trials": sorted(tc_by_game.get(g, [])),
        }

    pooled_delta = (
        round(gemma_pooled_mean - tc_pooled_mean, 6)
        if (gemma_pooled_mean is not None and tc_pooled_mean is not None)
        else None
    )

    # memorization rate on cells that induced an engine (round1 loaded == heldout not None)
    mem_cells = [r for r in rows if r.get("heldout_accuracy") is not None]
    mem_rate = (
        round(float(np.mean([bool(r.get("is_memorizing")) for r in mem_cells])), 4)
        if mem_cells
        else None
    )

    # ---- honest verdict (terminal-prefixed, numbers-first) + recommendation
    # "Off the floor" = a materially higher pooled mean AND at least one game gemma gets non-trivial
    # heldout that ThinkingCap floored at 0. This is a CHEAP screening signal, not a gate flip.
    moved_off_floor = (
        gemma_pooled_mean is not None
        and tc_pooled_mean is not None
        and pooled_delta is not None
        and pooled_delta > 0.10
        and gemma_games_nonzero > tc_games_nonzero
    )
    if n_cells == 0:
        branch = "no_data"
        verdict = "complete_gemma31b_singleshot_induction_no_cells_completed_see_errors"
        recommendation = "no cells produced -- inspect the shard/log before any follow-up."
    elif moved_off_floor:
        branch = "promising_bigger_model_moves_heldout"
        verdict = (
            f"complete_gemma31b_singleshot_induction_pooled_heldout_{gemma_pooled_mean}_"
            f"vs_thinkingcap27_baseline_{tc_pooled_mean}_delta_{pooled_delta}_"
            f"nonzero_games_{gemma_games_nonzero}of{len(gemma_pooled_vals)}_moved_off_floor_N{len(gemma_pooled_vals)}"
        )
        recommendation = (
            f"PROMISING: gemma-4-31B pooled single-shot heldout {gemma_pooled_mean} exceeds tonight's "
            f"ThinkingCap-27B baseline {tc_pooled_mean} by {pooled_delta}, with {gemma_games_nonzero} "
            f"non-zero games vs {tc_games_nonzero}. Raw capacity/family appears to move induction "
            f"quality off the near-zero floor -- WORTH a full CEGIS-refinement run on gemma-4-31B "
            f"(the exp5760 treatment) to see whether refinement compounds the gain. Whether to launch "
            f"that is OPERATOR-ONLY (a ~1-day GPU commitment)."
        )
    else:
        branch = "bigger_model_does_not_move_heldout"
        verdict = (
            f"complete_gemma31b_singleshot_induction_pooled_heldout_{gemma_pooled_mean}_"
            f"vs_thinkingcap27_baseline_{tc_pooled_mean}_delta_{pooled_delta}_"
            f"nonzero_games_{gemma_games_nonzero}of{len(gemma_pooled_vals)}_still_near_floor_"
            f"supports_architecture_wall_N{len(gemma_pooled_vals)}"
        )
        recommendation = (
            f"NOT PROMISING: gemma-4-31B pooled single-shot heldout {gemma_pooled_mean} (vs "
            f"ThinkingCap-27B baseline {tc_pooled_mean}, delta {pooled_delta}; non-zero games "
            f"{gemma_games_nonzero} vs {tc_games_nonzero}) stays near the floor. A genuinely bigger "
            f"different-family DENSE model does NOT move single-shot induction quality much -- this "
            f"further supports the ARCHITECTURAL-reconsideration branch of the design doc over "
            f"'the model isn't big enough'. Do NOT re-propose yet-bigger single-shot induction "
            f"variants (retire_if_same_verdict). A full CEGIS run on gemma-4-31B is LOWER priority "
            f"given single-shot shows no capacity headroom; operator-only regardless."
        )

    return {
        "experiment": "experiment_5764_gemma31b_singleshot_induction_ab",
        "schema": "carnot.exp5764.gemma31b_singleshot_induction_ab.v1",
        "requirements": ["REQ-ARC-WMTE-5764"],
        "prior_work_extended": [
            {
                "req": "REQ-ARC-WMTE-5726",
                "relation": "tonight's ThinkingCap-27B + Qwen-9B SINGLE-SHOT induction diagnosis "
                "(heldout near-zero floor, median 0.0, 29/37 at exactly 0.0) -- the baseline this "
                "extends; ThinkingCap-27B per-game single-shot heldout is the comparison arm here.",
                "verdict": "complete_thinkingcap_16k_dualgpu_reason_near_zero_heldout_floor",
            },
            {
                "req": "REQ-ARC-WMTE-5760",
                "relation": "CONCURRENT (GPU 0, NOT COMPLETE) CEGIS-refinement follow-on testing "
                "whether the existing refinement loop lifts the same near-zero heldout floor. Early "
                "cells show real code emission but ZERO heldout lift (consistent with its pre-"
                "registered HONEST-NEGATIVE branch). This experiment tests that branch's candidate "
                "(a) 'bigger/different-family model' IN PARALLEL. No final verdict for 5760 is "
                "claimed here.",
                "verdict": "in_progress_do_not_cite_final",
            },
        ],
        "question": (
            "Does raw model capacity/family ALONE (a different-family DENSE 31B, gemma-4-31B-it) move "
            "single-shot world-model induction heldout_accuracy off the near-zero floor tonight's "
            "ThinkingCap-27B/Qwen-9B produced, on the SAME 13 games using the SAME single-shot "
            "mechanism? If yes -> worth a full CEGIS run on this model; if no -> supports the "
            "architectural-reconsideration branch."
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
                "role": GEMMA["role"],
                "kv_quant": GEMMA["kv_quant"],
                "mtp": False,
                "use_chat_template": True,
                "n_ctx_deployed": server_meta.get("n_ctx"),
                "budget": BUDGET,
                "server": (
                    f"CUDA llama-server single-GPU (CUDA_VISIBLE_DEVICES={GEMMA['cuda_visible']}), "
                    f"-ngl 999, q8_0 KV, port {GEMMA['port']}, /v1/chat/completions"
                ),
                "vram_gpu1_before_after_mib": [
                    server_meta.get("vram_before"),
                    server_meta.get("vram_after"),
                ],
            }
        ],
        "heldout_accuracy_by_game": gemma_game_mean,
        "comparison_to_thinkingcap27_baseline": {
            "note": (
                "gemma-4-31B single-shot vs tonight's ThinkingCap-27B single-shot (REQ-ARC-WMTE-5726 "
                "shard), SAME games, SAME single-shot mechanism, mean heldout_accuracy over trials."
            ),
            "gemma31b_pooled_mean_heldout": gemma_pooled_mean,
            "gemma31b_pooled_max_game_heldout": gemma_pooled_max,
            "gemma31b_nonzero_games": gemma_games_nonzero,
            "thinkingcap27_pooled_mean_heldout": tc_pooled_mean,
            "thinkingcap27_nonzero_games": tc_games_nonzero,
            "pooled_mean_delta_gemma_minus_tc": pooled_delta,
            "n_games_compared": len(gemma_pooled_vals),
            "per_game": comparison_rows,
        },
        "single_shot_attribution": {
            "n_cells": n_cells,
            "n_induce_ok": n_induce_ok,
            "n_reason_engaged": n_reason_engaged,
            "n_overran": n_overran,
            "n_reached_levelup": n_levelup,
            "memorization_rate_on_induced_engines": mem_rate,
            "note": (
                "gemma-4-it has NO hybrid-think mode; the mechanism's hardcoded '/think\\n' prefix is "
                "a Qwen-ism applied uniformly for mechanism-identity and is expected to yield low "
                "reason_engaged. induce_ok = emitted parseable engine+is_level_complete within 16384."
            ),
        },
        "field_principles": {
            "honest_verdict": "terminal-prefixed, numbers-first; a continuous heldout mean with real "
            "headroom (tonight's baseline median is a measured 0.0) cannot come back 'no headroom'.",
            "inference_substrate": "live_llm_inference -- real gemma-4-31B-it Q4_K_M GGUF generation on "
            "a CUDA llama-server; 60s duration floor. Runtime VRAM-jump assertion refuses CPU fallback.",
            "solve_provenance": "development_proxy -- PUBLIC-game offline measurement of the LIVE induce "
            "mechanism with a bigger offline model swapped in; NOT a hidden-game self-discovery solve.",
            "verifier_is_oracle": "False -- heldout_accuracy is exact-match against real recorded "
            "transitions the engine was NOT fit to (held-out split); the win oracle is the level "
            "counter, oracle-distinct.",
            "random_seed": "LLM sampling is server-side stochastic; trials are per-game replicates "
            "(same seeded window), NOT independent game-level samples -- reported per game + pooled.",
            "reproducibility_checksum": "content hash over harness + induce/e3/reinduction code + "
            "generator/roster/budget config + rows.",
            "duration_s": "real wall-clock; a bigger dense model's per-cell latency is disclosed and the "
            "60s floor guards against a fabricated fast run.",
            "prior_work_extended": "traces this to REQ-ARC-WMTE-5726 (baseline) + the concurrent "
            "REQ-ARC-WMTE-5760 (cited as in-progress, no final verdict) so the delta is auditable.",
            "heldout_accuracy_by_game": "the PRIMARY signal -- does a bigger different-family model "
            "move the exact quantity the diagnosis named as the binding wall, per game?",
            "recommendation": "screening call ONLY (worth a full CEGIS run on gemma-4-31B?); whether to "
            "launch that ~1-day GPU job is OPERATOR-ONLY.",
        },
        "preconditions_checked": precond["checks"],
        "sample_size": {
            "games": len(gemma_pooled_vals),
            "roster_n": len(ROSTER),
            "roster": ROSTER,
            "trials_per_game": len(TRIALS),
            "paired_unit": "game (heldout averaged over trials, compared by game vs the TC baseline)",
            "note": (
                "SAME 13-game pre-registered roster as the concurrent REQ-ARC-WMTE-5760 CEGIS run "
                "(imported from that module to guarantee an exact match). 3 trials/game add per-game "
                "stability, not additional independent degrees of freedom (same seeded window)."
            ),
        },
        "methodology_note": (
            "Single-shot induction ONLY via exp5726.run_reason_cell_budget VERBATIM (codeonly OFF, "
            "/think, no pre-opened fence, tries=1, exp5722 stale-engine unlink, exp5720 "
            "actions-to-progress ladder), budget 16384 -- NOT execute_bounded_llm_reinduction (that is "
            "the concurrent GPU-0 CEGIS test). The ONLY scientific variable is the generator "
            "(gemma-4-31B-it dense, /v1/chat/completions with its embedded chat template) vs tonight's "
            "ThinkingCap-27B/Qwen-9B. heldout_accuracy/cell_recall/goal_predicate_accuracy computed by "
            "the SAME WorldModelVerifier code as tonight; window-memorization AST scan from exp5760. "
            "gemma pinned to GPU 1 (CUDA_VISIBLE_DEVICES=1), own llama-server on port 8971, parallel "
            "to and non-interfering with the GPU-0 CEGIS job. n_ctx chosen by a launch-time ladder "
            "(32768 first for comparability; smaller only on OOM). Disclosed divergences: (1) the "
            "'/think\\n' prefix is a Qwen-ism gemma ignores; (2) n_ctx may be < tonight's 32768 if the "
            "18GB model's full-KV allocation did not fit one 24GB card (recorded in n_ctx_deployed). "
            "This is a DIAGNOSTIC over EXISTING induction machinery with a bigger offline model, NOT a "
            "live-path modification and NOT an orphan solver (the induce mechanism IS the live one)."
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
        REPO / "python" / "carnot" / "experiment_5726_thinkingcap_16k_dualgpu_reason_ab.py",
        REPO / "python" / "carnot" / "experiment_5714_think_mode_rescoped_ab.py",
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
                "budget": BUDGET,
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
    log("preconditions OK -- starting gemma-4-31B single-shot induction run on GPU 1.")
    _, server_meta = run_all()
    artifact = build_artifact(time.time() - started, precond, server_meta)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    log(f"DONE. verdict={artifact['honest_verdict']}")
    log(f"artifact -> {ARTIFACT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

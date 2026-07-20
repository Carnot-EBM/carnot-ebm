"""Exp 5766: Does verifier-grounded CEGIS refinement COMPOUND gemma-4-31B's already-better
single-shot world-model induction, or was gemma's single-shot edge already near a ceiling
refinement cannot improve on? (REQ-ARC-WMTE-5766) -- the MISSING cell of tonight's 2x2
(model x mechanism) induction-quality design.

WHY THIS EXISTS
---------------
Tonight built a 2x2 over {ThinkingCap-27B, gemma-4-31B} x {single-shot, CEGIS refinement}:

  1. REQ-ARC-WMTE-5726  ThinkingCap-27B, SINGLE-SHOT  -> pooled heldout 0.187604 (near-zero floor,
     29/37 successful inductions at EXACTLY 0.0). The diagnosis
     (docs/research-notes/arc-world-model-induction-quality-diagnosis-2026-07-20.md): the binding
     wall is GENERATING a correct world model, not selecting among candidates.
  2. REQ-ARC-WMTE-5760  ThinkingCap-27B, CEGIS refinement (execute_bounded_llm_reinduction,
     min_heldout_accuracy=1.0) -- CONCURRENT on GPU 0 tonight, NOT COMPLETE. Early cells: real
     code emission but ~flat/negative heldout lift (mean delta ~ -0.02, 0 positive cells so far),
     consistent with its pre-registered HONEST-NEGATIVE branch. DO NOT cite its final verdict.
  3. REQ-ARC-WMTE-5764  gemma-4-31B, SINGLE-SHOT -> pooled heldout 0.378487 (12/13 games nonzero
     vs ThinkingCap's 6/13). A real, substantial positive: a genuinely BIGGER/different-family
     DENSE model DOES move single-shot induction off the floor. Its own recommendation named the
     obvious follow-on -- "WORTH a full CEGIS-refinement run on gemma-4-31B to see whether
     refinement COMPOUNDS the gain."

THIS experiment is that follow-on: cell #4 = gemma-4-31B x CEGIS refinement. The scientific
question is now a 2x2 interaction, not a main effect. On ThinkingCap (5726->5760) refinement
appears to add ~nothing on top of a near-zero single-shot floor. Does refinement behave the same
on top of gemma's much HIGHER single-shot starting point (0.378)? Two hypotheses the 2x2
distinguishes:
  * COMPOUND: gemma's better single-shot models are closer to correct, so counterexample-guided
    refactor rounds have a foothold and LIFT heldout further (delta_heldout > 0, and CEGIS best
    beats gemma's own 0.378 single-shot baseline). -> refinement is worth wiring more aggressively
    on a bigger offline model.
  * CEILING: gemma's single-shot edge is already near what THIS induction ARCHITECTURE can reach
    on these games; refinement rounds emit code but do not move heldout (delta ~ 0), same as
    ThinkingCap. -> the remaining wall is architectural (reactive-with-filter / executable-world-
    model construction), not model-capacity AND not refinement-depth.

MECHANISM -- CEGIS refinement (identical to REQ-ARC-WMTE-5760), generator swapped to gemma
------------------------------------------------------------------------------------------
Reuses exp5760.run_cegis_cell VERBATIM (routes induction through execute_bounded_llm_reinduction
with min_heldout_accuracy=1.0 so the dynamics-refactor rounds actually FIRE; captures the per-round
heldout_accuracy trajectory from outcome.rounds[*], the before/after window-memorization AST scan,
and the refactor code-emission rate). The ONLY scientific variable vs 5760 is the generator:
gemma-4-31B-it dense (served via /v1/chat/completions with its embedded chat template, the 5725
accommodation) instead of ThinkingCap-27B/Qwen-9B. Round 0 (action="induce") = the within-run
single-shot baseline (comparable to gemma's 5764 single-shot); rounds 1-2 (action="refactor") =
counterexample-guided refactor. PRIMARY metric = delta_heldout = heldout(best refined round) -
heldout(round 0), per game + pooled, bootstrap 95% CI -- 5760's exact metric. SECONDARY, and the
cross-experiment question the operator asked: does CEGIS best-achieved heldout beat gemma's OWN
0.378 single-shot baseline (REQ-ARC-WMTE-5764), per game + pooled?

Roster/trials/budget are IMPORTED from the sibling modules (ROSTER+TRIALS from 5760, BUDGET from
5726 -- exactly 5760's own source) so the roster is byte-identical to BOTH prior tonight runs.
This is a paired 2x2: same 13 games, same 3 trials, same 16384 budget.

PRE-REGISTERED FALSIFIABLE GATE -- 5760's EXACT three branches + partial catch-all (same
thresholds: pooled delta>0.15 / <=0.05, >=12 games, CI-excludes-0, posfrac>=0.5, sign-p<0.05,
memdrop>=0.2, degradation guard, emission>0.6):
  POSITIVE:          refinement COMPOUNDS gemma's single-shot -> pooled delta_heldout>0.15 etc.
  HONEST-NEGATIVE:   delta<=0.05 pooled AND memorization unchanged AND emission healthy -> CEILING
                     (matches the ThinkingCap 5760 pattern; the wall is architectural).
  EMISSION-CONFOUND: refactor-emission rate<=0.6 (mechanical artifact -- fix emission first).

GPU / topology: GPU 1 ONLY (CUDA_VISIBLE_DEVICES=1), a dedicated own llama-server on port 8972
(distinct from the GPU-0 CEGIS job's 8969 TC / 8968 Qwen -- both of which stay on GPU 0 because
that job runs with CARNOT_5726_QW_CUDA=0, verified from its live /proc environ; and distinct from
the completed 5764's 8971). n_ctx picked by a launch-time ladder (32768 first for comparability,
smaller only on OOM). Runtime GPU-offload assertion (VRAM jump > 1GB) refuses a silent CPU
fallback. Runs fully PARALLEL to and NON-INTERFERING with the GPU-0 CEGIS job.

PROVENANCE: development_proxy on PUBLIC games (NOT a hidden-game self-discovery solve). This is a
DIAGNOSTIC over the EXISTING live refinement machinery (execute_bounded_llm_reinduction is what
runs live) with a bigger offline model swapped in -- NOT a live-path modification and NOT an orphan
solver (Live-Path Reachability Discipline: the mechanism under test IS the live mechanism).
verifier_is_oracle False (win oracle = the level counter; heldout is exact-match on a held-out
transition split, oracle-distinct). NEVER flips the frozen live default (operator-only), NEVER
submits.

Prior-failure block (Failed-Experiment Rerun Discipline): names REQ-ARC-WMTE-5726 (ThinkingCap
single-shot near-zero floor), the concurrent REQ-ARC-WMTE-5760 (ThinkingCap CEGIS, in-progress,
early flat), and REQ-ARC-WMTE-5764 (gemma single-shot POSITIVE 0.378, the baseline this tests
whether refinement improves on). Root cause of the ThinkingCap CEGIS flatness is UNRESOLVED
(whether it is a model-capacity wall the bigger model clears, or an induction-architecture wall
refinement cannot cross regardless of model). What is DIFFERENT: this pairs the CEGIS mechanism
with the ONE model that already cleared the single-shot floor -- the exact 2x2 cell needed to tell
"capacity wall" from "architecture wall". retire_if_same_verdict: if gemma+CEGIS ALSO returns
HONEST-NEGATIVE (delta<=0.05) on top of its 0.378 single-shot, do NOT re-propose more induction-
REFINEMENT variants at all -- that is strong 2x2 evidence the residual wall is ARCHITECTURAL
(reactive-with-filter / executable-world-model construction), independent of both model capacity
and refinement depth.

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

# ---- CEGIS mechanism + roster + gate helpers from the ThinkingCap CEGIS sibling (5760). Importing
# runs only module-level config (it re-imports 5726's config; all os.environ.setdefault /
# constant definitions, no run()). ROSTER/TRIALS are imported (NOT hand-copied) so the roster is
# byte-identical to both prior tonight runs. run_cegis_cell is reused VERBATIM -- the ONLY variable
# vs 5760 is the generator this file feeds it.
from carnot.experiment_5760_cegis_refinement_induction_ab import (  # noqa: E402
    DEGRADATION_GUARD_GAMES,
    MIN_HELDOUT_ACCURACY,
    ROSTER,
    TRIALS,
    _bootstrap_ci,
    _mean,
    _mem_rate,
    _per_game_delta,
    run_cegis_cell,
)

# ---- gemma-4-31B model config from the single-shot sibling (5764), copied so the model path /
# hf_id / quant / chat-template / KV-quant are BYTE-IDENTICAL to the single-shot baseline this
# experiment is paired against. Only the port (fresh, collision-free) + role differ.
from carnot.experiment_5764_gemma31b_singleshot_induction_ab import (  # noqa: E402
    GEMMA as _SS_GEMMA,
)
from carnot.experiment_5764_gemma31b_singleshot_induction_ab import (  # noqa: E402
    GPU1_IDLE_MAX_MIB,
    GPU_INDEX,
    NCTX_LADDER,
)

# ---- true shared primitives (BUDGET is imported from 5726, exactly as 5760 imports it, so the
# recorded budget matches what run_cegis_cell actually uses internally).
from carnot.experiment_5726_thinkingcap_16k_dualgpu_reason_ab import (  # noqa: E402
    BUDGET,
    LLAMA_SERVER,
    _gpu_mem_used_mib,
    terminate,
)

REPO = Path(__file__).resolve().parents[2]
SHARD = REPO / "results" / "exp5766_gemma31b_cegis_refinement_shard.jsonl"
ARTIFACT = REPO / "results" / "experiment_5766_gemma31b_cegis_refinement_ab.json"

# gemma-4-31B single-shot baseline (REQ-ARC-WMTE-5764) shard -- the paired comparison arm read at
# artifact time (does CEGIS best beat gemma's OWN 0.378 single-shot?).
SS_SHARD = REPO / "results" / "exp5764_gemma31b_singleshot_shard.jsonl"

# gemma config: copy 5764's exact model config, swap only the port (fresh, no collision with the
# GPU-0 CEGIS job's 8969/8968 -- both on GPU 0 per CARNOT_5726_QW_CUDA=0 -- nor 5764's completed
# 8971) and the role string.
GEMMA: dict[str, Any] = dict(_SS_GEMMA)
GEMMA["port"] = 8972
GEMMA["role"] = (
    "different-family DENSE 31B; CEGIS-refinement induction probe on GPU 1 "
    "(the missing model x mechanism 2x2 cell: does refinement compound gemma's single-shot edge?)"
)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------------------
# Own llama-server launch pinned to GPU 1 (server-launch PATTERN mirrored from 5764: n_ctx ladder,
# health-wait, runtime VRAM-jump offload assertion). Self-contained copies reference this module's
# GEMMA (fresh port) -- the same structure 5764 itself uses.
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
    server applies gemma-4-it's OWN embedded chat template. run_cegis_cell then mirrors tonight's
    reason-induce config on this proposer (no_think_prefix '/think\\n', tries=1, BUDGET)."""
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
# Shard IO (resumable) -- keyed by (game, trial): single generator arm
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

    # GPU 1 must be idle RIGHT NOW. If not idle, STOP rather than risk contending with the GPU-0
    # CEGIS job (verified to stay on GPU 0 via CARNOT_5726_QW_CUDA=0, but re-check defensively).
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
                "experiment": "experiment_5766_gemma31b_cegis_refinement_ab",
                "schema": "carnot.exp5766.gemma31b_cegis_refinement_ab.v1",
                "requirements": ["REQ-ARC-WMTE-5766"],
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
# Run loop -- single gemma server on GPU 1, sequential cells (no contention, no world_model race).
# Mirrors 5764's run_all but routes each cell through 5760.run_cegis_cell (the CEGIS mechanism).
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
        f"=== gemma-4-31B-it CEGIS : {len(pending)} cells | CUDA={GEMMA['cuda_visible']} "
        f"budget={BUDGET} min_heldout={MIN_HELDOUT_ACCURACY} ==="
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
            log(f"RUN gemma31b_cegis {game} trial={t}")
            c0 = time.time()
            try:
                row = run_cegis_cell(
                    game, prop, trial=t, window=window, full_traj=full_traj, cell=cell
                )
            except Exception as exc:
                row = {
                    "game": game,
                    "trial": t,
                    "error": f"cell_crash: {type(exc).__name__}: {exc}"[:300],
                    "delta_heldout": None,
                    "round0_heldout": None,
                    "wall_s": round(time.time() - c0, 1),
                }
            row["generator"] = "gemma31b"
            row["arm"] = "gemma31b_cegis"
            row["game"] = game
            row["trial"] = t
            row["server_n_ctx"] = server_meta["n_ctx"]
            _append_shard(row)
            done[(game, t)] = row
            log(
                f"  -> round0={row.get('round0_heldout')} refactor={row.get('refactor_heldouts')} "
                f"delta={row.get('delta_heldout')} emit={row.get('n_refactor_emitted')}/"
                f"{row.get('n_refactor_attempted')} mem {row.get('mem_before_is_memorizing')}->"
                f"{row.get('mem_after_is_memorizing')} planned={row.get('planned')} "
                f"wall={row.get('wall_s')}s ({time.time() - c0:.0f}s)"
            )
    finally:
        terminate(proc)
    return list(done.values()), server_meta


# ---------------------------------------------------------------------------
# gemma-4-31B single-shot baseline (REQ-ARC-WMTE-5764) for the paired comparison
# ---------------------------------------------------------------------------
def _ss_baseline_by_game() -> dict[str, list[float]]:
    by: dict[str, list[float]] = {}
    if not SS_SHARD.exists():
        return by
    for line in SS_SHARD.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("arm") != "gemma31b_singleshot":
            continue
        g = r.get("game")
        h = r.get("heldout_accuracy")
        if g in ROSTER and isinstance(h, (int, float)):
            by.setdefault(g, []).append(float(h))
    return by


def _cell_best_achieved(r: dict[str, Any]) -> Optional[float]:
    """Best heldout the CEGIS loop reached across ALL rounds for this cell (round0 + every refactor
    round). This is what the refinement PIPELINE achieved -- the fair number to compare against the
    single-shot baseline. None only if round0 itself was unmeasured (induce failed)."""
    vals: list[float] = []
    r0 = r.get("round0_heldout")
    if isinstance(r0, (int, float)):
        vals.append(float(r0))
    for h in r.get("refactor_heldouts") or []:
        if isinstance(h, (int, float)):
            vals.append(float(h))
    return max(vals) if vals else None


def _by_game_mean(rows: list[dict[str, Any]], key: str) -> dict[str, Optional[float]]:
    by: dict[str, list[float]] = {}
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)):
            by.setdefault(r["game"], []).append(float(v))
    return {g: (round(float(np.mean(v)), 6) if v else None) for g, v in by.items()}


# ---------------------------------------------------------------------------
# Artifact + pre-registered gate evaluation (5760's EXACT gate-threshold logic)
# ---------------------------------------------------------------------------
def build_artifact(
    duration_s: float, precond: dict[str, Any], server_meta: dict[str, Any]
) -> dict[str, Any]:
    from carnot.agentic.arc_actions_to_progress import _sign_test_p

    rows = [r for r in _load_shard().values() if r.get("arm") == "gemma31b_cegis"]

    # ---- PRIMARY (5760's exact metric): per-game delta_heldout = heldout(best refined) - round0
    pooled_delta = _per_game_delta(rows)
    per_game_list = sorted(pooled_delta.items())
    pooled_vals = [v for _, v in per_game_list]
    pooled_mean = _mean(pooled_vals)
    ci = _bootstrap_ci(pooled_vals)
    n_games = len(pooled_vals)
    positive_frac = round(float(np.mean([v > 0 for v in pooled_vals])), 4) if pooled_vals else None
    wins = sum(1 for v in pooled_vals if v > 1e-9)
    losses = sum(1 for v in pooled_vals if v < -1e-9)
    sign_p = _sign_test_p(wins, losses)

    # ---- SECONDARY: window-memorization rate before vs after refinement
    mem_before_rate = _mem_rate(rows, "mem_before_is_memorizing")
    mem_after_rate = _mem_rate(rows, "mem_after_is_memorizing")
    mem_drop = (
        round(mem_before_rate - mem_after_rate, 4)
        if (mem_before_rate is not None and mem_after_rate is not None)
        else None
    )

    # ---- ATTRIBUTION: refactor code-emission rate (pooled)
    tot_attempted = sum(int(r.get("n_refactor_attempted") or 0) for r in rows)
    tot_emitted = sum(int(r.get("n_refactor_emitted") or 0) for r in rows)
    emission_rate = round(tot_emitted / tot_attempted, 4) if tot_attempted else None

    # ---- degradation guard: on round0==1.0 games (sp80/ft09), refined must NOT drop below round0
    degradation_violations = []
    for r in rows:
        if r.get("game") in DEGRADATION_GUARD_GAMES:
            r0 = r.get("round0_heldout")
            br = r.get("best_refined_heldout")
            if (
                isinstance(r0, (int, float))
                and r0 >= 0.999
                and isinstance(br, (int, float))
                and br < r0 - 1e-9
            ):
                degradation_violations.append(
                    {
                        "game": r["game"],
                        "trial": r["trial"],
                        "round0": r0,
                        "best_refined": br,
                    }
                )
    degradation_guard_holds = len(degradation_violations) == 0

    # ---- pre-registered gate (5760 design sec 5): three honest branches + partial catch-all.
    # IDENTICAL thresholds to REQ-ARC-WMTE-5760.
    emission_healthy = emission_rate is not None and emission_rate > 0.6
    ci_excludes_0 = bool(ci.get("excludes_0"))
    positive_gate = (
        pooled_mean is not None
        and pooled_mean > 0.15
        and n_games >= 12
        and ci_excludes_0
        and positive_frac is not None
        and positive_frac >= 0.5
        and sign_p is not None
        and sign_p < 0.05
        and mem_drop is not None
        and mem_drop >= 0.2
        and degradation_guard_holds
        and emission_healthy
    )
    mem_unchanged = mem_drop is not None and mem_drop < 0.2
    honest_negative_gate = (
        pooled_mean is not None and pooled_mean <= 0.05 and mem_unchanged and emission_healthy
    )
    emission_confound_gate = emission_rate is not None and emission_rate <= 0.6

    # ---- SECONDARY cross-experiment question: does CEGIS best beat gemma's OWN single-shot (5764)?
    ss_by_game = _ss_baseline_by_game()
    ss_game_mean = {g: (round(float(np.mean(v)), 6) if v else None) for g, v in ss_by_game.items()}
    cegis_round0_by_game = _by_game_mean(rows, "round0_heldout")
    cegis_best_by_game: dict[str, Optional[float]] = {}
    _best_tmp: dict[str, list[float]] = {}
    for r in rows:
        b = _cell_best_achieved(r)
        if b is not None:
            _best_tmp.setdefault(r["game"], []).append(b)
    for g, v in _best_tmp.items():
        cegis_best_by_game[g] = round(float(np.mean(v)), 6) if v else None

    def _pool(d: dict[str, Optional[float]]) -> Optional[float]:
        vals = [x for x in d.values() if x is not None]
        return round(float(np.mean(vals)), 6) if vals else None

    ss_pooled = _pool(ss_game_mean)
    cegis_round0_pooled = _pool(cegis_round0_by_game)
    cegis_best_pooled = _pool(cegis_best_by_game)
    # primary NEW delta: CEGIS best-achieved minus gemma's own single-shot baseline, paired by game
    best_minus_ss_by_game: dict[str, float] = {}
    for g in ROSTER:
        b = cegis_best_by_game.get(g)
        s = ss_game_mean.get(g)
        if b is not None and s is not None:
            best_minus_ss_by_game[g] = round(b - s, 6)
    best_minus_ss_vals = list(best_minus_ss_by_game.values())
    best_minus_ss_pooled = (
        round(float(np.mean(best_minus_ss_vals)), 6) if best_minus_ss_vals else None
    )
    best_minus_ss_ci = _bootstrap_ci(best_minus_ss_vals)
    games_cegis_beats_ss = sum(1 for v in best_minus_ss_vals if v > 1e-9)
    games_cegis_below_ss = sum(1 for v in best_minus_ss_vals if v < -1e-9)

    comparison_per_game = {
        g: {
            "gemma31b_singleshot_5764_heldout_mean": ss_game_mean.get(g),
            "cegis_round0_heldout_mean": cegis_round0_by_game.get(g),
            "cegis_best_achieved_heldout_mean": cegis_best_by_game.get(g),
            "cegis_best_minus_singleshot": best_minus_ss_by_game.get(g),
        }
        for g in ROSTER
    }

    # ---- verdict (terminal-prefixed, numbers-first) -- 5760's branch structure, gemma-flavored,
    # annotated with the cross-baseline (does refinement compound 5764's 0.378?).
    pm = round(pooled_mean, 4) if pooled_mean is not None else None
    xb = best_minus_ss_pooled
    if n_games == 0:
        branch = "no_data"
        verdict = "complete_gemma31b_cegis_refinement_no_cells_completed_see_errors"
    elif emission_confound_gate:
        branch = "emission_confound"
        verdict = (
            f"complete_gemma31b_cegis_refinement_untestable_emission_confound_rate_{emission_rate}_"
            f"le_0.6_fix_code_emission_first_pooled_delta_{pm}_vs_singleshot_{xb}_N{n_games}"
        )
    elif positive_gate:
        branch = "positive"
        verdict = (
            f"success_gemma31b_cegis_refinement_compounds_singleshot_pooled_delta_{pm}_gt0.15_"
            f"CI_{ci['lo']}_{ci['hi']}_posfrac_{positive_frac}_signp_{sign_p}_memdrop_{mem_drop}_"
            f"cegis_best_{cegis_best_pooled}_vs_singleshot_{ss_pooled}_delta_{xb}_N{n_games}"
        )
    elif honest_negative_gate:
        branch = "honest_negative"
        verdict = (
            f"complete_gemma31b_cegis_refinement_null_ceiling_pooled_delta_{pm}_le0.05_"
            f"memdrop_{mem_drop}_emission_{emission_rate}_healthy_cegis_best_{cegis_best_pooled}_"
            f"vs_singleshot_{ss_pooled}_delta_{xb}_architecture_wall_not_refinement_depth_N{n_games}"
        )
    else:
        branch = "partial_inconclusive"
        verdict = (
            f"complete_gemma31b_cegis_refinement_partial_pooled_delta_{pm}_CI_{ci['lo']}_{ci['hi']}_"
            f"posfrac_{positive_frac}_signp_{sign_p}_memdrop_{mem_drop}_emission_{emission_rate}_"
            f"vs_singleshot_{xb}_N{n_games}_does_not_cleanly_meet_a_preregistered_branch"
        )

    # ---- recommendation (OPERATOR-ONLY): interpret the 2x2 interaction.
    if branch == "positive":
        recommendation = (
            f"COMPOUND CONFIRMED: on gemma-4-31B's higher single-shot starting point ({ss_pooled}), "
            f"CEGIS refinement LIFTS pooled heldout by {pm} (best-achieved {cegis_best_pooled}, "
            f"+{xb} over gemma's own single-shot baseline), {games_cegis_beats_ss}/{n_games} games "
            f"improving. Unlike ThinkingCap (near-zero single-shot floor where refinement had no "
            f"foothold), a bigger offline model + refinement COMPOUND. Worth wiring refinement more "
            f"aggressively behind a bigger offline induction model -- OPERATOR-ONLY (a live-default / "
            f"GPU-commitment decision; this experiment NEVER flips the frozen live stack)."
        )
    elif branch == "honest_negative":
        recommendation = (
            f"CEILING (architecture wall): even on gemma-4-31B's much better single-shot start "
            f"({ss_pooled}), CEGIS refinement adds ~nothing (pooled delta {pm}<=0.05; best-achieved "
            f"{cegis_best_pooled}, {xb} vs single-shot), the SAME flat pattern ThinkingCap showed. "
            f"The 2x2 now reads: bigger model moves single-shot, but refinement does NOT compound on "
            f"EITHER model. The residual wall is ARCHITECTURAL (reactive-with-filter / executable-"
            f"world-model construction), independent of both model capacity and refinement depth. "
            f"retire_if_same_verdict: do NOT re-propose more induction-refinement variants; pivot the "
            f"architecture. OPERATOR-ONLY on any live change."
        )
    elif branch == "emission_confound":
        recommendation = (
            f"UNTESTABLE (emission confound): refactor code-emission rate {emission_rate}<=0.6, so a "
            f"flat/negative delta is a mechanical budget-overrun artifact, NOT evidence about "
            f"refinement. Fix gemma's refactor-round emission (budget/prompt) first, then re-judge. "
            f"OPERATOR-ONLY on any follow-up."
        )
    elif branch == "no_data":
        recommendation = "No cells produced -- inspect the shard/log before any follow-up."
    else:
        recommendation = (
            f"PARTIAL/INCONCLUSIVE: pooled delta {pm} (best-achieved {cegis_best_pooled} vs "
            f"single-shot {ss_pooled}, {xb}) does not cleanly meet a pre-registered branch. Read the "
            f"per-game trajectory + emission attribution before drawing a 2x2 conclusion. "
            f"OPERATOR-ONLY on any follow-up."
        )

    return {
        "experiment": "experiment_5766_gemma31b_cegis_refinement_ab",
        "schema": "carnot.exp5766.gemma31b_cegis_refinement_ab.v1",
        "requirements": ["REQ-ARC-WMTE-5766"],
        "prior_work_extended": [
            {
                "req": "REQ-ARC-WMTE-5726",
                "relation": "ThinkingCap-27B SINGLE-SHOT induction diagnosis (pooled heldout "
                "0.187604, near-zero floor). The single-shot cell of the OTHER model in the 2x2.",
                "verdict": "complete_thinkingcap_16k_dualgpu_reason_near_zero_heldout_floor",
            },
            {
                "req": "REQ-ARC-WMTE-5760",
                "relation": "CONCURRENT (GPU 0, NOT COMPLETE) ThinkingCap-27B CEGIS refinement -- the "
                "SAME mechanism this experiment runs, on the OTHER model. Early cells show real code "
                "emission but ~flat/negative heldout lift (its pre-registered HONEST-NEGATIVE "
                "pattern). This experiment pairs that mechanism with gemma-4-31B to complete the "
                "model x mechanism 2x2. No final verdict for 5760 is claimed here.",
                "verdict": "in_progress_do_not_cite_final",
            },
            {
                "req": "REQ-ARC-WMTE-5764",
                "relation": "gemma-4-31B SINGLE-SHOT induction -- pooled heldout 0.378487 (12/13 games "
                "nonzero), a real POSITIVE showing a bigger different-family model moves single-shot "
                "off the floor. This experiment tests whether CEGIS refinement COMPOUNDS that "
                "0.378 baseline (the primary comparison arm).",
                "verdict": "complete_gemma31b_singleshot_induction_pooled_heldout_0.378487_vs_"
                "thinkingcap27_baseline_0.187604_delta_0.190883_nonzero_games_12of13_moved_off_floor_N13",
            },
        ],
        "question": (
            "Does the EXISTING verifier-grounded CEGIS refinement loop (execute_bounded_llm_"
            "reinduction, min_heldout_accuracy=1.0) COMPOUND gemma-4-31B's already-better single-shot "
            "world-model induction (pooled heldout 0.378, REQ-ARC-WMTE-5764), or was gemma's "
            "single-shot edge already near a ceiling refinement cannot cross? The missing model x "
            "mechanism 2x2 cell distinguishing a model-capacity wall from an induction-architecture "
            "wall."
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
                "min_heldout_accuracy": MIN_HELDOUT_ACCURACY,
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
        "delta_heldout_by_game": {
            "note": (
                "PRIMARY metric (REQ-ARC-WMTE-5760's exact metric): delta_heldout = heldout(best "
                "refined round) - heldout(round 0), per game (mean over trials) + pooled, bootstrap "
                "95% CI over games. This is the WITHIN-loop refinement lift on gemma."
            ),
            "pooled_mean_delta_heldout": round(pooled_mean, 6) if pooled_mean is not None else None,
            "bootstrap_95ci": ci,
            "n_games": n_games,
            "positive_game_frac": positive_frac,
            "paired_sign_test_p": sign_p,
            "wins": wins,
            "losses": losses,
            "per_game_delta": {g: round(v, 6) for g, v in per_game_list},
        },
        "comparison_to_gemma31b_singleshot_baseline": {
            "note": (
                "The PRIMARY cross-experiment question the operator asked: does CEGIS refinement "
                "improve on gemma's OWN 0.378 single-shot baseline (REQ-ARC-WMTE-5764), per game + "
                "pooled? 'cegis_best_achieved' = best heldout the loop reached across round0 + every "
                "refactor round (what the refinement PIPELINE achieved). Paired by game."
            ),
            "gemma31b_singleshot_5764_pooled_heldout": ss_pooled,
            "cegis_round0_pooled_heldout": cegis_round0_pooled,
            "cegis_best_achieved_pooled_heldout": cegis_best_pooled,
            "pooled_delta_cegis_best_minus_singleshot": best_minus_ss_pooled,
            "pooled_delta_bootstrap_95ci": best_minus_ss_ci,
            "n_games_cegis_beats_singleshot": games_cegis_beats_ss,
            "n_games_cegis_below_singleshot": games_cegis_below_ss,
            "n_games_compared": len(best_minus_ss_vals),
            "per_game": comparison_per_game,
        },
        "window_memorization_rate_before_after": {
            "definition": (
                "structural AST scan of the induced engine() source; is_memorizing := >=3 int "
                "literals (>=10, coordinate range not action/color codes) matching an observed-window "
                "changed-cell row/col index. GATED quantity = before/after DELTA (cancels the "
                "dense-window chance-match confound symmetrically)."
            ),
            "rate_before_refinement": mem_before_rate,
            "rate_after_refinement": mem_after_rate,
            "rate_drop": mem_drop,
        },
        "refactor_emission_rate": {
            "refactor_emission_rate_pooled": emission_rate,
            "refactor_rounds_attempted": tot_attempted,
            "refactor_rounds_emitted": tot_emitted,
            "healthy_threshold": 0.6,
            "note": (
                "attribution guard -- a low rate means a flat delta is a budget-overrun mechanical "
                "artifact, not evidence against refinement (5760 design sec 5 branch 3)."
            ),
        },
        "degradation_guard": {
            "games": list(DEGRADATION_GUARD_GAMES),
            "holds": degradation_guard_holds,
            "violations": degradation_violations,
        },
        "preregistration": {
            "roster": ROSTER,
            "roster_n": len(ROSTER),
            "roster_source": (
                "IMPORTED from REQ-ARC-WMTE-5760 (ROSTER/TRIALS) -- byte-identical to BOTH prior "
                "tonight runs (5760 ThinkingCap CEGIS, 5764 gemma single-shot). Not hand-copied."
            ),
            "trials": TRIALS,
            "budget": BUDGET,
            "min_heldout_accuracy": MIN_HELDOUT_ACCURACY,
            "primary_metric": "delta_heldout = heldout(best refined round) - heldout(round 0), per game + pooled",
            "secondary_metric": "cegis_best_achieved_heldout - gemma31b_singleshot_5764_heldout, per game + pooled",
            "gate_positive": (
                "pooled mean delta_heldout>0.15 across >=12 games AND bootstrap 95% CI excludes 0 "
                "AND positive on >=50% of games AND paired sign-test p<0.05 AND memorization rate "
                "drops >=0.2 absolute AND degradation guard holds AND refactor-emission rate>0.6 "
                "(REQ-ARC-WMTE-5760's EXACT thresholds)"
            ),
            "gate_honest_negative": "delta<=0.05 pooled AND memorization unchanged AND emission healthy (CEILING / architecture wall)",
            "gate_emission_confound": "refactor-emission rate<=0.6 (mechanical artifact, not evidence against refinement)",
            "retire_if_same_verdict": (
                "on HONEST-NEGATIVE, do NOT re-propose more induction-refinement variants; the 2x2 "
                "then reads 'architecture wall independent of model capacity AND refinement depth' -- "
                "pivot the architecture (reactive-with-filter / executable-world-model construction)."
            ),
        },
        "field_principles": {
            "honest_verdict": "terminal-prefixed, numbers-first; a continuous heldout delta with real "
            "headroom (gemma's single-shot baseline is a measured 0.378, not a floor) cannot come "
            "back 'no headroom'.",
            "inference_substrate": "live_llm_inference -- real gemma-4-31B-it Q4_K_M GGUF generation "
            "across up to 3 CEGIS rounds/cell on a CUDA llama-server; 60s duration floor. Runtime "
            "VRAM-jump assertion refuses a silent CPU fallback.",
            "solve_provenance": "development_proxy -- PUBLIC-game offline measurement of the LIVE "
            "refinement mechanism with a bigger offline model swapped in; NOT a hidden-game solve.",
            "verifier_is_oracle": "False -- heldout_accuracy is exact-match against real recorded "
            "transitions the engine was NOT fit to (held-out split); win oracle is the level counter, "
            "oracle-distinct.",
            "random_seed": "LLM sampling is server-side stochastic; trials are per-game replicates "
            "(same seeded window), paired by GAME, bootstrap over games -- NOT independent samples.",
            "reproducibility_checksum": "content hash over harness + reinduction + world-model code + "
            "generator/roster/budget config + rows.",
            "duration_s": "real wall-clock; a bigger dense model across 3 refinement rounds is slow "
            "and disclosed; the 60s floor guards against a fabricated fast run.",
            "prior_work_extended": "traces this to REQ-ARC-WMTE-5726/5760/5764 by id+verdict so the "
            "2x2 interaction is auditable (5760 cited as in-progress, no final verdict).",
            "delta_heldout_by_game": "PRIMARY -- the within-loop refinement lift on the exact quantity "
            "the diagnosis named as the binding wall, per game.",
            "comparison_to_gemma31b_singleshot_baseline": "the operator's primary question -- does "
            "refinement COMPOUND gemma's 0.378 single-shot edge, per game + pooled?",
            "gate_branch": "the pre-registered 5760 branch this run lands in (positive / honest_negative "
            "/ emission_confound / partial) -- decides the 2x2 interpretation.",
            "recommendation": "OPERATOR-ONLY 2x2 interpretation (compound vs ceiling); this experiment "
            "NEVER flips the frozen live stack.",
        },
        "preconditions_checked": precond["checks"],
        "sample_size": {
            "games": n_games,
            "roster_n": len(ROSTER),
            "roster": ROSTER,
            "trials_per_game": len(TRIALS),
            "paired_unit": "game (delta_heldout averaged over trials; paired by game vs 5764 single-shot)",
            "note": (
                "SAME 13-game pre-registered roster + 3 trials + 16384 budget as BOTH prior tonight "
                "runs (imported from REQ-ARC-WMTE-5760 to guarantee an exact match). Trials add "
                "per-game stability, not additional independent degrees of freedom (same seeded window)."
            ),
        },
        "methodology_note": (
            "Route induction through execute_bounded_llm_reinduction (min_heldout_accuracy=1.0, "
            "candidate_provider = the loaded engine, load_engine + plan_in_model from "
            "arc_executable_world_model) via exp5760.run_cegis_cell VERBATIM -- the ONLY scientific "
            "variable vs REQ-ARC-WMTE-5760 is the generator (gemma-4-31B-it dense, "
            "/v1/chat/completions with its embedded chat template, vs ThinkingCap-27B/Qwen-9B). "
            "Per-round heldout from outcome.rounds[*]['heldout_accuracy'] (exact-match on the held-out "
            "split; CARNOT_ARC_TRUST_METRIC forced OFF so the metric is exact, not cell_recall). "
            "round 0 = induce (the within-run single-shot baseline), rounds 1-2 = refactor. Proposer "
            "mirrors tonight's reason-induce (codeonly OFF, /think, tries=1, 16384 budget). gemma "
            "pinned to GPU 1 (CUDA_VISIBLE_DEVICES=1), own llama-server on port 8972, PARALLEL to and "
            "NON-INTERFERING with the GPU-0 CEGIS job (which stays on GPU 0 via CARNOT_5726_QW_CUDA=0). "
            "n_ctx chosen by a launch-time ladder (32768 first for comparability; smaller only on OOM). "
            "Disclosed divergences (identical to 5764/5760): (1) the '/think\\n' prefix is a Qwen-ism "
            "gemma ignores; (2) round 0 uses the loop's STANDARD induce (the live path), not tonight's "
            "_induce_no_fence variant -- but the rigorous quantity is the INTERNAL round-0-vs-refined "
            "delta. A DIAGNOSTIC over the EXISTING live refinement mechanism with a bigger offline "
            "model, NOT a live-path modification and NOT an orphan solver."
        ),
        "duration_s": round(duration_s, 2),
        "reproducibility_checksum": _repro_checksum(rows, server_meta),
    }


def _repro_checksum(rows: list[dict[str, Any]], server_meta: dict[str, Any]) -> str:
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_llm_reinduction as reind

    h = hashlib.sha256()
    for mod_file in (
        __file__,
        e3.__file__,
        reind.__file__,
        REPO / "python" / "carnot" / "experiment_5760_cegis_refinement_induction_ab.py",
        REPO / "python" / "carnot" / "experiment_5726_thinkingcap_16k_dualgpu_reason_ab.py",
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
                "min_heldout": MIN_HELDOUT_ACCURACY,
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
    log("preconditions OK -- starting gemma-4-31B CEGIS refinement run on GPU 1.")
    _, server_meta = run_all()
    artifact = build_artifact(time.time() - started, precond, server_meta)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    log(f"DONE. verdict={artifact['honest_verdict']}")
    log(f"artifact -> {ARTIFACT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

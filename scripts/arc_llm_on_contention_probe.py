#!/usr/bin/env python3
"""LLM-ON CONTENTION LADDER -- the per-game cost level the MAX_ACTIONS answer actually needs.

WHY THIS EXISTS
===============
`results/outer_loop_arc_llm_on_wallclock_envelope_20260726.json` measured LLM-ON per-game wall
clock at budgets 400/1000/2000 in ONE process against ONE warm generator, and said so plainly:

    "This probe ran ONE process against ONE warm server. The real eval runs ~110 concurrent
     threads in one process. The measured numbers are therefore a FLOOR on eval wall clock, not
     an estimate of it."

Its conservative reading -- the one that decides whether a MAX_ACTIONS raise fits the cap -- is
therefore built on an IMPORTED contention factor: 1.72x (range 1.436-1.976), measured on
**LLM-OFF** cells in `outer_loop_scored_path_budget_sweep_20260726.json`. That factor cannot be
assumed to carry: an LLM-OFF cell contends for CPU, while an LLM-ON cell contends for a SINGLE
llama-server whose slot count is FINITE (4, read off `/props`). Those are different queues with
different shapes, and one of them can REFUSE service rather than slow down -- the sibling lane's
three-way isolation found concurrent induction requests at `-c 16384` returning HTTP 500
"Context size has been exceeded" (~4096 tokens/slot at 4 slots, against a `max_tokens=4096`
request) and killing the server in 4 of 5 observations.

So the decision-relevant number -- LLM-ON cost per game UNDER CONCURRENCY -- has never been
measured. This probe measures it directly, as a ladder in K (concurrent games), per-seed matched
against a K=1 control on the same games.

WHAT IT MEASURES, AND IN WHICH UNIT
===================================
Per-game WALL CLOCK SECONDS on the SCORED path (`E3AgentPolicy` via `arc_leaderboard_eval.run_game`
-- one of the two live entrypoints per CLAUDE.md's ARC Live-Path Reachability Discipline), with the
FROZEN live generator running, at K concurrent games sharing ONE llama-server.

It also records, per cell, the three counters this project keeps conflating -- `actions` (OFFLINE
actions, resets EXCLUDED), `n_frames` (loop iterations, resets INCLUDED), `n_resets` -- plus the
per-level reset attribution the instrumented `run_game` now emits, so the gateway-charged unit is
recoverable from these rows without re-running anything.

DESIGN DECISIONS, AND WHY
=========================
1. CONCURRENCY IS THE SWEPT PARAMETER AND THE INNERMOST STRUCTURE IS THE GAME SET. Every K arm runs
   the SAME games at the SAME seed and the SAME budget. The only thing that changes between arms is
   how many run at once. Per-game paired ratios are therefore matched by construction (this
   project's measurement-failure #3: never score a union across seeds).

2. K CONCURRENT PROCESSES, NOT THREADS -- and the limitation is stated rather than hidden. The real
   eval is single-process/multi-thread (`ARC-AGI-3-Agents/agents/swarm.py:76-99` starts one thread
   per game, then joins them all), so it ALSO pays GIL serialisation on the non-LLM component that
   separate processes do not. Threads are unusable here for a different reason: `run_cell` sets
   `os.environ["CARNOT_ARC_DISABLE_INDUCTION"]` and calls `random.seed`/`np.random.seed` -- process-
   global state that concurrent threads would interleave, confounding the arm with a seeding bug.
   So this probe measures the SERVER-side contention exactly and UNDERSTATES the GIL-side. The
   GIL-side floor is separately bounded by the sibling artifact's `unbatchable_floor_s_per_game`.

3. OVERLAP IS VERIFIED, NOT ASSUMED. Each worker records its own start/end epoch; the parent
   computes the measured overlap of every K-batch. An arm labelled "K=4" whose cells did not in fact
   overlap is an UNINTERPRETABLE arm (this project's measurement-failure #2/#5: a treatment that
   silently failed to apply reads as a clean null). The overlap fraction travels with the arm.

4. ONE SERVER, HEALED BETWEEN BATCHES, NEVER DURING ONE. Workers call `forbid_spawn()` so a
   transient health blip costs one induction instead of forking a second 12 GB model onto the card
   (a measured failure mode: `InstrumentedProposer.forbid_spawn`'s docstring records three
   simultaneous servers). The parent heals between batches, where the reload cost lands OUTSIDE
   any measured cell.

5. GPU 1 ONLY, VERIFIED NOT ASSUMED. Per the 2026-06-27 allocation the conductor owns GPU 0 and the
   outer loop owns GPU 1. `_generator_server_and_env` falls through to the AMD iGPU HIP build
   SILENTLY if the CUDA headroom guard trips, and that build exists on this box -- so "I set the
   env var" is not evidence the env var took. The probe resolves the binary + launch env through the
   proposer's OWN resolver, cross-checks per-PID VRAM against the physical card index via GPU UUID,
   and records both on every row. It also samples GPU utilisation THROUGHOUT each batch, which is
   what distinguishes "the server was the bottleneck" from "the server was idle and we were
   CPU-bound".

6. A FRESH PORT. A stale wedged llama-server was observed on the default 8924 (alive, ~300 MiB
   resident -- i.e. NO model on the card -- and `/health` silent). Binding a fresh port forces a
   clean spawn whose configuration we know.

WHAT IT DOES NOT DO
===================
It does not submit anything, does not touch GPU 0, and does not change MAX_ACTIONS or any
`SUBMITTED_*` flag. Budget is a `run_game` parameter here exactly as in the existing lever harness.
The flag decision is the operator's.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))


# --------------------------------------------------------------------------------------------
# Device resolution + GPU witnesses. Imported from the sibling probe so the two cannot drift:
# one definition of "which card did we actually get", used by both.
# --------------------------------------------------------------------------------------------
def _device_helpers():
    import arc_llm_on_wallclock_budget_probe as sib

    return sib.resolve_generator_device, sib.gpu_compute_apps


def gpu_utilisation_sample() -> dict:
    """One instantaneous read of both cards: utilisation + memory.

    Recorded as a TIME SERIES across each batch rather than once, because a per-game wall-clock
    number under concurrency is only interpretable if we know whether the shared GPU was saturated.
    A K=4 arm that is slower than K=1 while GPU utilisation stays low is CPU/queueing-bound, not
    GPU-bound, and the two have different extrapolations to K=110.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
        per = {}
        for ln in out.strip().splitlines():
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 3:
                per[int(parts[0])] = {"util_pct": int(parts[1]), "mem_used_mib": int(parts[2])}
        return {"t": round(time.time(), 3), "gpus": per}
    except Exception as exc:
        return {"t": round(time.time(), 3), "error": f"{type(exc).__name__}:{exc}"}


class GpuSampler:
    """Background utilisation sampler for the duration of one batch."""

    def __init__(self, period_s: float = 2.0):
        self.period_s = period_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    def start(self) -> None:
        def loop():
            while not self._stop.is_set():
                self.samples.append(gpu_utilisation_sample())
                self._stop.wait(self.period_s)

        self._th = threading.Thread(target=loop, daemon=True)
        self._th.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=5)
        g1 = [s["gpus"][1]["util_pct"] for s in self.samples if s.get("gpus", {}).get(1)]
        g0 = [s["gpus"][0]["util_pct"] for s in self.samples if s.get("gpus", {}).get(0)]
        m1 = [s["gpus"][1]["mem_used_mib"] for s in self.samples if s.get("gpus", {}).get(1)]
        return {
            "n_samples": len(self.samples),
            "gpu1_util_mean": round(sum(g1) / len(g1), 1) if g1 else None,
            "gpu1_util_max": max(g1) if g1 else None,
            "gpu1_util_frac_above_50": (
                round(sum(1 for x in g1 if x > 50) / len(g1), 3) if g1 else None
            ),
            "gpu1_mem_used_mib_max": max(m1) if m1 else None,
            "gpu0_util_mean_CONDUCTORS_CARD_recorded_not_gated": (
                round(sum(g0) / len(g0), 1) if g0 else None
            ),
            "gpu0_util_max": max(g0) if g0 else None,
        }


# --------------------------------------------------------------------------------------------
# Worker: exactly one cell, in its own process.
# --------------------------------------------------------------------------------------------
def run_worker(a) -> int:
    """One (game, seed, budget) cell with the LLM ON, against an ALREADY-RUNNING server.

    Writes the row plus its own start/end epochs so the parent can VERIFY the batch overlapped.
    Never spawns a server (`forbid_spawn`): under K-way concurrency a self-heal would fork a second
    copy of the model onto the very card whose contention is being measured.
    """
    import arc_scored_path_lever_harness as harness

    t_start = time.time()
    rec: dict = {
        "game": a.game,
        "seed": a.seed,
        "budget": a.budget,
        "concurrency_K": a.concurrency,
        "pid": os.getpid(),
        "t_start_epoch": round(t_start, 3),
    }
    try:
        proposer = harness.build_proposer(a.port)
        if a.n_ctx:
            proposer._inner.n_ctx = int(a.n_ctx)
        proposer.forbid_spawn()
        healthy_at_start = bool(proposer._inner._healthy())
        rec["server_healthy_at_worker_start"] = healthy_at_start
        row = harness.run_cell(
            a.game,
            a.seed,
            budget=a.budget,
            proposer=proposer,
            llm=True,
            extra_kwargs=dict(harness.ARMS["S"]),
            arm=f"S_llmon_b{a.budget}_K{a.concurrency}",
        )
        rec["row"] = row
        rec["worker_ok"] = True
    except Exception as exc:  # a crashed worker must be VISIBLE, not an absent row
        rec["worker_ok"] = False
        rec["worker_error"] = f"{type(exc).__name__}:{exc}"
    rec["t_end_epoch"] = round(time.time(), 3)
    rec["worker_wall_s"] = round(rec["t_end_epoch"] - t_start, 2)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str))
    return 0 if rec.get("worker_ok") else 3


# --------------------------------------------------------------------------------------------
# Parent: the ladder.
# --------------------------------------------------------------------------------------------
def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help="internal: run ONE cell")
    ap.add_argument("--game")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument(
        "--games",
        default="dc22,ft09,sc25,su15",
        help="the 4 games with COMPLETE budget coverage in the sibling uncontended probe, so the "
        "K=1 arm is a direct same-game/same-seed replicate of a published measurement",
    )
    ap.add_argument("--seed", type=int, default=20260724)
    ap.add_argument("--budget", type=int, default=400, help="the SHIPPED MAX_ACTIONS value")
    ap.add_argument("--ladder", default="1,2,4", help="K values, ascending")
    ap.add_argument(
        "--n-ctx",
        type=int,
        default=0,
        dest="n_ctx",
        help="OPTIONAL server context-window override for a CONFIGURATION test (0 = the shipped "
        "LocalGGUFProposer default of 16384, i.e. change nothing). Exists because the K=4 arm killed "
        "the generator outright at the shipped setting, and the shipped per-slot context "
        "(n_ctx/total_slots = 4096 at 4 slots) is exactly the agent's own max_tokens request -- so "
        "'is the death context-driven?' is a one-flag experiment. Requires a FRESH port: "
        "`_ensure_server` REUSES a healthy server and would otherwise silently keep the old -c.",
    )
    ap.add_argument("--port", type=int, default=8971)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows-dir", default=None)
    a = ap.parse_args(argv)

    if a.worker:
        return run_worker(a)

    resolve_generator_device, gpu_compute_apps = _device_helpers()
    games = [g for g in a.games.split(",") if g]
    ladder = [int(k) for k in a.ladder.split(",") if k]
    rows_dir = Path(a.rows_dir or (Path(a.out).parent / "cells"))
    rows_dir.mkdir(parents=True, exist_ok=True)

    # ---- PRECONDITIONS (before any measurement) --------------------------------------------
    pre = []
    gguf = list(
        (Path.home() / ".cache/huggingface/hub/models--unsloth--Qwen3.5-9B-MTP-GGUF").glob(
            "snapshots/*/*.gguf"
        )
    )
    pre.append(
        {
            "resource": "frozen_live_generator_gguf_cached",
            "available": bool(gguf),
            "detail": str(gguf[0]) if gguf else "MISSING",
        }
    )
    dev = resolve_generator_device()
    resolver_says_cuda_gpu1 = (not dev["is_hip_igpu_build"]) and dev[
        "cuda_visible_devices_pinned"
    ] == "1"
    # A SERVER ALREADY ON THIS PORT IS THE DEVICE WE GET, whatever the resolver would choose NOW.
    # This clause exists because of a real false block observed 2026-07-26: the resolver's headroom
    # guard needs >=13000 MiB FREE on the card, and a healthy CUDA-GPU1 generator from a previous run
    # was itself holding 12.1 GiB -- so the resolver reported the iGPU fallback for a launch that
    # would never happen (`_ensure_server` reuses a healthy server without consulting the resolver).
    # Blocking there would be a false negative. The reuse clause is only allowed to pass on a
    # PER-PID VRAM WITNESS: the process bound to this port must hold multi-GiB on physical card 1.
    reuse = _existing_server_on_gpu1(a.port, gpu_compute_apps)
    pre.append(
        {
            "resource": "generator_resolves_to_cuda_gpu1_not_igpu",
            "available": bool(resolver_says_cuda_gpu1 or reuse["reusable_gpu1_server"]),
            "detail": {
                "resolver": dev,
                "resolver_says_cuda_gpu1": resolver_says_cuda_gpu1,
                "existing_server_reuse_witness": reuse,
                "note": (
                    "PASS via the resolver means a fresh spawn will be CUDA-pinned to GPU 1. PASS via "
                    "the reuse witness means a server already bound to this port is holding multi-GiB "
                    "on physical card 1, so that is the device this run will actually use."
                ),
            },
        }
    )
    envdir = REPO / "environment_files"
    missing = [g for g in games if not (envdir / g).is_dir()]
    pre.append(
        {
            "resource": "offline_environment_files_for_requested_games",
            "available": not missing,
            "detail": f"missing={missing}",
        }
    )
    gpu0 = [x for x in gpu_compute_apps() if x.get("gpu_index") == 0]
    pre.append(
        {
            "resource": "gpu0_left_to_the_conductor",
            "available": True,
            "detail": f"gpu0_compute_apps_at_start={gpu0} (recorded, NOT a gate)",
        }
    )
    for p in pre:
        print(f"[precondition] {p['resource']}: {p['available']} :: {p['detail']}", flush=True)
    failed = [p["resource"] for p in pre if not p["available"]]
    if failed:
        Path(a.out).write_text(
            json.dumps(
                {
                    "honest_verdict": "blocked_" + failed[0],
                    "preconditions_checked": pre,
                    "cells": [],
                },
                indent=1,
            )
        )
        print(f"BLOCKED on {failed} -- no measurement attempted, no numbers invented", flush=True)
        return 2

    # ---- one server, up front ---------------------------------------------------------------
    import arc_scored_path_lever_harness as harness

    parity = harness.assert_shipped_dict_matches_module_globals()
    if parity.get("pinned_vs_live_drift"):
        print(f"[WARNING] pinned SHIPPED dict differs from live globals: {parity}", flush=True)
    parent_proposer = harness.build_proposer(a.port)
    if a.n_ctx:
        parent_proposer._inner.n_ctx = int(a.n_ctx)
        print(
            f"[generator] n_ctx OVERRIDE requested: {a.n_ctx} (shipped default is 16384)",
            flush=True,
        )
    t_srv = time.time()
    ok = parent_proposer._inner._ensure_server()
    srv_s = round(time.time() - t_srv, 2)
    print(f"[generator] ensure_server={ok} in {srv_s}s port={a.port}", flush=True)
    if not ok:
        Path(a.out).write_text(
            json.dumps(
                {
                    "honest_verdict": "blocked_generator_server_would_not_start",
                    "preconditions_checked": pre,
                    "cells": [],
                },
                indent=1,
            )
        )
        return 2
    apps_after = gpu_compute_apps()
    on1 = [x for x in apps_after if x.get("gpu_index") == 1]
    print(f"[generator] GPU1 residency witness: {on1}", flush=True)
    server_props = _read_props(a.port)
    print(f"[generator] /props: {server_props}", flush=True)

    cells: list[dict] = []
    batches: list[dict] = []
    heals: list[dict] = []
    t0 = time.time()

    def emit():
        Path(a.out).write_text(
            json.dumps(
                {
                    "probe": "arc_llm_on_contention_ladder",
                    "preconditions_checked": pre,
                    "generator_device_resolved": dev,
                    "gpu_compute_apps_after_model_load": apps_after,
                    "gpu1_residency_witness": on1,
                    "server_props": server_props,
                    "server_spawn_s": srv_s,
                    "port": a.port,
                    "n_ctx_requested": a.n_ctx or 16384,
                    "n_ctx_is_an_override_of_the_shipped_default": bool(a.n_ctx),
                    "n_ctx_readback_from_server_props": (server_props or {}).get("n_ctx"),
                    "seed": a.seed,
                    "budget": a.budget,
                    "games_requested": games,
                    "ladder_requested": ladder,
                    "concurrency_is_the_swept_parameter": True,
                    "workers_are_processes_not_threads": True,
                    "flag_parity_vs_live_globals": parity,
                    "shipped_max_actions_for_reference": 400,
                    "generator_heal_events": heals,
                    "batches": batches,
                    "cells": cells,
                    "elapsed_s": round(time.time() - t0, 1),
                },
                indent=1,
                default=str,
            )
        )

    emit()

    def heal(tag: str) -> dict:
        if parent_proposer._inner._healthy():
            return {"healed": False, "before_tag": tag}
        t = time.time()
        hok = parent_proposer._inner._ensure_server()
        rec = {
            "healed": True,
            "heal_ok": bool(hok),
            "heal_s": round(time.time() - t, 2),
            "before_tag": tag,
        }
        heals.append(rec)
        print(f"[generator] HEALED before {tag}: {rec}", flush=True)
        return rec

    for K in ladder:
        chunks = [games[i : i + K] for i in range(0, len(games), K)]
        for ci, chunk in enumerate(chunks):
            hrec = heal(f"K{K}/chunk{ci}")
            sampler = GpuSampler(2.0)
            sampler.start()
            t_batch = time.time()
            procs = []
            for g in chunk:
                cell_out = rows_dir / f"cell_K{K}_{g}_{a.seed}_b{a.budget}.json"
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--game",
                    g,
                    "--seed",
                    str(a.seed),
                    "--budget",
                    str(a.budget),
                    "--concurrency",
                    str(K),
                    "--port",
                    str(a.port),
                    "--out",
                    str(cell_out),
                ]
                env = dict(os.environ)
                env.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
                procs.append((g, cell_out, subprocess.Popen(cmd, env=env)))
            rcs = {}
            for g, _co, pr in procs:
                rcs[g] = pr.wait()
            batch_wall = round(time.time() - t_batch, 2)
            gpu = sampler.stop()

            got = []
            for g, co, _pr in procs:
                try:
                    rec = json.loads(co.read_text())
                except Exception as exc:
                    rec = {
                        "game": g,
                        "concurrency_K": K,
                        "worker_ok": False,
                        "worker_error": f"row_unreadable:{type(exc).__name__}:{exc}",
                    }
                rec["worker_returncode"] = rcs.get(g)
                rec["batch_index"] = ci
                rec["batch_wall_s"] = batch_wall
                rec["heal_before_batch"] = hrec
                rec["gpu_during_batch"] = gpu
                got.append(rec)
                cells.append(rec)

            # OVERLAP WITNESS: did the cells in this batch actually run at the same time?
            spans = [
                (r["t_start_epoch"], r["t_end_epoch"])
                for r in got
                if r.get("t_start_epoch") and r.get("t_end_epoch")
            ]
            overlap = _overlap_seconds(spans)
            longest = max((s[1] - s[0] for s in spans), default=0.0)
            batches.append(
                {
                    "K": K,
                    "batch_index": ci,
                    "games": chunk,
                    "batch_wall_s": batch_wall,
                    "n_cells": len(got),
                    "all_pairwise_overlap_s": round(overlap, 2),
                    "longest_cell_s": round(longest, 2),
                    "overlap_fraction_of_longest_cell": (
                        round(overlap / longest, 3) if longest > 0 else None
                    ),
                    "concurrency_actually_achieved": bool(K == 1 or overlap > 0.5 * longest),
                    "gpu_during_batch": gpu,
                    "worker_returncodes": rcs,
                }
            )
            for r in got:
                row = r.get("row") or {}
                L = row.get("llm") or {}
                print(
                    f"K={K} {r['game']:5} wall={row.get('wall_s')}s "
                    f"llm={L.get('llm_wall_s')}s act={row.get('actions')} "
                    f"frames={row.get('n_frames')} resets={row.get('n_resets')} "
                    f"lv={row.get('levels')} ind={row.get('induction_attempts')}"
                    f"(llm={row.get('induction_attempts_llm_reached')}) "
                    f"resp={L.get('responses')} err={L.get('errors')} "
                    f"tok_out={L.get('tokens_predicted')} "
                    f"genok={row.get('generator_healthy_before')}->"
                    f"{row.get('generator_healthy_after')} "
                    f"srv={row.get('llama_servers_before')}->{row.get('llama_servers_after')} "
                    f"VALID={row.get('llm_on_row_valid')} rc={r.get('worker_returncode')}",
                    flush=True,
                )
            print(
                f"[batch] K={K} chunk{ci} wall={batch_wall}s overlap={round(overlap, 1)}s "
                f"of longest {round(longest, 1)}s gpu1_util_mean={gpu.get('gpu1_util_mean')}",
                flush=True,
            )
            emit()

    print(f"TOTAL {round(time.time() - t0, 1)}s cells={len(cells)}", flush=True)
    emit()
    return 0


def _existing_server_on_gpu1(port: int, gpu_compute_apps) -> dict:
    """Is a llama-server already bound to `port`, and is IT holding VRAM on physical card 1?

    Two independent facts, both required: the process must be bound to the port we are about to use
    (so it is the server this run will talk to), and per-PID VRAM attribution -- matched to the
    PHYSICAL card index via GPU UUID, not via CUDA_VISIBLE_DEVICES, which renumbers -- must show
    multi-GiB resident. A few hundred MiB means the weights are NOT on the card (a wedged server in
    exactly that state was observed on the default port), so the threshold is deliberately high.
    """
    try:
        out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, timeout=15)
        pids = [
            ln.strip().split()[0]
            for ln in out.stdout.splitlines()
            if "llama-server" in ln and f"--port {port}" in ln
        ]
    except Exception as exc:
        return {"reusable_gpu1_server": False, "error": f"{type(exc).__name__}:{exc}"}
    apps = gpu_compute_apps()
    matched = []
    for app in apps:
        if str(app.get("pid")) in pids:
            try:
                mib = int(str(app.get("used_mib", "0")).split()[0])
            except Exception:
                mib = 0
            matched.append({**app, "used_mib_int": mib})
    on1 = [m for m in matched if m.get("gpu_index") == 1 and m["used_mib_int"] > 4000]
    return {
        "reusable_gpu1_server": bool(on1),
        "pids_bound_to_port": pids,
        "matched_compute_apps": matched,
        "gpu1_resident_multi_gib": on1,
    }


def _overlap_seconds(spans: list[tuple[float, float]]) -> float:
    """Seconds during which ALL spans were simultaneously active.

    The strict reading on purpose: a K=4 batch where only 2 cells ever overlapped is not a K=4
    measurement, and the intersection (not the union) is what makes that visible.
    """
    if not spans:
        return 0.0
    lo = max(s[0] for s in spans)
    hi = min(s[1] for s in spans)
    return max(0.0, hi - lo)


def _read_props(port: int) -> dict:
    """Read the server's own `/props`: the SLOT COUNT and context size are the capacity that
    concurrency contends for, and they are read from the server rather than inferred from flags
    (the sibling lane retracted a claim built on inferring 1 slot from the absence of a flag)."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=20) as r:
            d = json.loads(r.read().decode())
        return {
            "total_slots": d.get("total_slots"),
            "n_ctx": d.get("default_generation_settings", {}).get("n_ctx") or d.get("n_ctx"),
            "model_path": str(d.get("model_path"))[-60:],
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}:{exc}"}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

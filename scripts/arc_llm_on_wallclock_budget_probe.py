#!/usr/bin/env python3
"""LLM-ON WALL-CLOCK PROBE ACROSS ACTION BUDGETS -- the missing anchor above b400.

WHY THIS EXISTS
===============
`results/outer_loop_scored_path_budget_sweep_20260726.json` fits an LLM-ON cost model and
projects per-game wall clock at budgets 1000/2000/4000. Its own `residuals` list says, verbatim:

    "one LLM-on run at b1000/b2000 would replace this model with a direct anchor"

and

    "THE BUDGET-400 ROW IS ARITHMETICALLY FORCED, NOT A TEST. The model is calibrated to
     reproduce the measured b400 anchor exactly ... that row's FITS/OVER verdict carries ZERO
     independent information. Only the b1000+ rows are model OUTPUT."

So every LLM-ON number above budget 400 that this project holds is EXTRAPOLATION from a single
budget. The MAX_ACTIONS decision (shipped 400; 4000 buys 12 median wins where 400 buys 4) turns
entirely on whether that extrapolation is right, because wall clock is the only axis that opposes a
raise (the charge model is settled: the post-solve tail is free, and depth dominates the score).
This script produces the direct anchor.

WHAT IT MEASURES, AND IN WHICH UNIT
===================================
Per-game WALL CLOCK SECONDS on the SCORED path (`E3AgentPolicy` via `arc_leaderboard_eval.run_game`,
one of the two live entrypoints per CLAUDE.md's ARC Live-Path Reachability Discipline), with the
FROZEN live generator actually running (Qwen3.5-9B-MTP on a real llama-server).

It ALSO records, per cell, the three action/frame/reset counters the project keeps confusing, so
the budget->reset-traffic coupling can be read off the same rows rather than assumed:
`actions` (offline actions, EXCLUDES resets), `n_frames` (loop iterations, INCLUDES resets), and
`n_resets`. A raise in MAX_ACTIONS that buys wins by spending resets is a raise whose cost is
INVISIBLE in offline actions and VISIBLE in the gateway's charge -- so the reset counter has to
travel with the wall-clock number.

THE DESIGN DECISIONS, AND WHY
=============================
1. BUDGET IS THE INNERMOST LOOP. The prior sweep ran one budget per process invocation, making
   budget the OUTERMOST loop, so machine-condition drift between invocations is confounded with the
   swept parameter. Here every budget for a given (game, seed) runs back-to-back in one process
   against one already-warm generator.

2. BUDGET ORDER ALTERNATES BY GAME. Even-indexed games ascend (400, 1000, 2000); odd-indexed games
   descend. A single fixed order would let a within-cell warm-up or KV-cache effect masquerade as a
   budget effect. Alternating balances order against budget.

3. A SAME-CONFIG REPLICATE AT THE TOP BUDGET. With the LLM ON the run is NOT a deterministic
   function of the seed -- the generator samples. The prior artifact's own `S_replicate` arm
   measured this and the same-config wall ratio spanned 0.52x to 2.47x across only 3 pairs, i.e. a
   4.7-fold spread. The cost model's predicted b400->b2000 effect is 1.34x. THE NOISE IS LARGER
   THAN THE EFFECT at the per-cell level. Reporting a budget->wall number without a same-config
   replicate beside it would be reporting sampling noise as a budget effect (this project's
   measurement-failure #5: an uninstrumented arm reads as a clean result). The replicate runs at the
   TOP budget because that is where there is no prior data at all; the b400 cells can be read
   against the 17 b400 cells already on file.

4. ONE SHARED PROPOSER / ONE SERVER. Mirrors `arc_leaderboard_eval.py`'s `_PROPOSER` global and the
   real eval: the ~14s model load is paid once per process, not once per game.

5. GPU 1 ONLY, VERIFIED NOT ASSUMED. Per the 2026-06-27 allocation the conductor owns GPU 0 and the
   outer loop owns GPU 1. `_generator_server_and_env` FALLS THROUGH TO THE iGPU HIP BUILD SILENTLY
   if the CUDA headroom guard trips (arc_executable_world_model.py:1539), and the HIP build exists
   on this box -- so "I set the flag" is not evidence the flag took. This script resolves the
   binary + env the same way the proposer does, prints it, and records it on every row.

WHAT IT DOES NOT DO
===================
It does not submit anything, does not touch GPU 0, and does not change any SUBMITTED_* flag or
MAX_ACTIONS. It sweeps `budget` as a parameter of `run_game` exactly as the existing lever harness
already does, and leaves the flag decision to the operator.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))


def resolve_generator_device() -> dict:
    """What binary + env will the proposer ACTUALLY launch, and on which card?

    Resolved by calling the proposer module's own resolver, so this cannot drift from what the
    generator really does. The distinction that matters: the CUDA build with CUDA_VISIBLE_DEVICES
    pinned is the 3090; the `build-hip` binary is the AMD iGPU, which is ~6x slower and would make
    every wall-clock number in this probe an artifact of the wrong device.
    """
    from carnot.agentic.arc_executable_world_model import _generator_server_and_env

    path, env = _generator_server_and_env()
    return {
        "server_binary": str(path),
        "is_hip_igpu_build": "build-hip" in str(path),
        "cuda_visible_devices_pinned": (env or {}).get("CUDA_VISIBLE_DEVICES"),
        "env_is_pinned_dict": env is not None,
        "requested_gpu": os.environ.get("CARNOT_ARC_GENERATOR_CUDA_GPU"),
    }


def gpu_compute_apps() -> list[dict]:
    """Which PIDs hold VRAM on which physical card, right now.

    This is the independent cross-check on the pinning: the resolver can say
    CUDA_VISIBLE_DEVICES=1 and still be wrong if, say, the index mapping differs. Reading the
    per-process attribution back off nvidia-smi and matching the GPU UUID to an index is what makes
    "it ran on GPU 1" an observation rather than a claim. Also the only way to notice we have
    started fighting the conductor for GPU 0.
    """
    try:
        uuids = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
        idx_of = {}
        for ln in uuids.splitlines():
            if "," in ln:
                i, u = ln.split(",", 1)
                idx_of[u.strip()] = int(i.strip())
        apps = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
        out = []
        for ln in apps.splitlines():
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 3:
                out.append(
                    {
                        "pid": parts[0],
                        "gpu_index": idx_of.get(parts[1]),
                        "used_mib": parts[2],
                    }
                )
        return out
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}:{exc}"}]


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--games",
        default="dc22,cd82,su15,ft09,tu93",
        help="cheap->expensive at b400 per the prior LLM-ON table, so an early cut still leaves "
        "COMPLETE budget triples for the games that did run (budget is the inner loop)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=20260724,
        help="the seed the prior b400 LLM-ON cells used, so the anchor is comparable",
    )
    ap.add_argument("--budgets", default="400,1000,2000")
    ap.add_argument(
        "--replicate-at",
        type=int,
        default=2000,
        help="same-config repeat at this budget: the noise floor where no prior data "
        "exists. 0 disables (do not disable -- see module docstring point 3)",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=8951,
        help="NOT the default 8931/8924: a stale wedged server was observed on 8924 "
        "(alive, 296MiB, /health silent). A fresh port forces a clean spawn.",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--order-offset",
        type=int,
        default=0,
        help="shift the budget-order alternation parity. Needed when a run is SPLIT across "
        "invocations (run 1 died when the generator died): the games in run 2 must keep the "
        "parity they would have had in the un-split run, or the alternation that balances "
        "order against budget is silently broken for the merged dataset.",
    )
    a = ap.parse_args(argv)

    games = [g for g in a.games.split(",") if g]
    budgets = [int(b) for b in a.budgets.split(",") if b]

    # ---- PRECONDITIONS (before any measurement, per the Pre-Launch Preconditions Discipline) ----
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
    # The load-bearing precondition. An iGPU run would produce wall-clock numbers that describe the
    # wrong device, and the fallback is SILENT.
    pre.append(
        {
            "resource": "generator_resolves_to_cuda_gpu1_not_igpu",
            "available": (not dev["is_hip_igpu_build"])
            and dev["cuda_visible_devices_pinned"] == "1",
            "detail": dev,
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
            "detail": f"gpu0_compute_apps_at_start={gpu0} "
            f"(recorded, NOT a gate -- the conductor is entitled to GPU 0)",
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
                    "rows": [],
                },
                indent=1,
            )
        )
        print(f"BLOCKED on {failed} -- no measurement attempted, no numbers invented", flush=True)
        return 2

    # ---- generator up (one server, shared across every cell) ----
    import arc_scored_path_lever_harness as harness

    parity = harness.assert_shipped_dict_matches_module_globals()
    if parity.get("pinned_vs_live_drift"):
        print(
            f"[WARNING] pinned SHIPPED dict differs from live SUBMITTED_* globals: "
            f"{parity['pinned_vs_live_drift']}",
            flush=True,
        )

    proposer = harness.build_proposer(a.port)
    t_srv = time.time()
    ok = proposer._inner._ensure_server()
    srv_s = round(time.time() - t_srv, 2)
    print(f"[generator] ensure_server={ok} in {srv_s}s port={a.port}", flush=True)
    if not ok:
        Path(a.out).write_text(
            json.dumps(
                {
                    "honest_verdict": "blocked_generator_server_would_not_start",
                    "preconditions_checked": pre,
                    "rows": [],
                },
                indent=1,
            )
        )
        return 2
    # SPAWNING IS *NOT* FROZEN, and that is a corrected decision (2026-07-26, run 1).
    #
    # Run 1 called `forbid_spawn()` to prevent the server-storm this project has measured before (a
    # mid-cell self-heal spawns a SECOND server, and two servers on one card contend for exactly the
    # resource being measured). But the frozen live generator DIED after ~10 minutes / 5 inductions
    # of real work -- a failure the lever harness's own comments record as observed twice on
    # 2026-07-26 -- and with spawning forbidden that death was UNRECOVERABLE: every subsequent cell
    # would have run with no generator, producing rows that are correctly marked invalid but that
    # cost full wall clock to produce. Forbidding the spawn converted a recoverable fault into a
    # run-ending one.
    #
    # The fix keeps both properties: heal BETWEEN cells (where the cost lands outside the timed
    # region and cannot contaminate a wall-clock number), never DURING one. `ensure_healthy_between_
    # cells` below does the pre-cell check. A self-heal that still happens mid-cell is caught by the
    # harness's own `server_storm_suspected` (llama_servers_after > before) and that row is dropped
    # by the analyser's validity gate -- so the storm remains detectable rather than silent.
    apps_after = gpu_compute_apps()
    print(f"[generator] compute apps after load: {apps_after}", flush=True)
    # Independent witness that the model is on GPU 1 and is actually resident (a 9B Q4 with a
    # 22k q8 KV cache is several GiB; a few hundred MiB means the weights are NOT on the card).
    on1 = [x for x in apps_after if x.get("gpu_index") == 1]
    print(f"[generator] GPU1 residency witness: {on1}", flush=True)

    heals: list[dict] = []

    def ensure_healthy_between_cells(tag: str) -> dict:
        """Bring the generator back BEFORE a cell starts, so a reload's cost never lands inside a
        measured wall clock.

        Returns a record that travels with the row. `healed: True` means this cell was preceded by a
        model reload -- worth knowing, because a freshly-loaded server has a cold prompt cache and
        its first induction may be slower than a warm one, which is a per-cell confound the row
        should carry rather than hide.
        """
        if proposer._inner._healthy():
            return {"healed": False, "heal_s": 0.0, "before_tag": tag}
        t = time.time()
        ok = proposer._inner._ensure_server()
        rec = {
            "healed": True,
            "heal_ok": bool(ok),
            "heal_s": round(time.time() - t, 2),
            "before_tag": tag,
        }
        heals.append(rec)
        print(
            f"[generator] HEALED before {tag}: ok={ok} in {rec['heal_s']}s "
            f"(cost is OUTSIDE the cell's wall_s)",
            flush=True,
        )
        return rec

    rows: list[dict] = []
    t0 = time.time()

    def emit():
        Path(a.out).write_text(
            json.dumps(
                {
                    "probe": "llm_on_wallclock_budget_probe",
                    "preconditions_checked": pre,
                    "generator_device_resolved": dev,
                    "gpu_compute_apps_after_model_load": apps_after,
                    "server_spawn_s": srv_s,
                    "port": a.port,
                    "seed": a.seed,
                    "budgets_requested": budgets,
                    "replicate_at": a.replicate_at,
                    "games_requested": games,
                    "flag_parity_vs_live_globals": parity,
                    "budget_is_innermost_loop": True,
                    "budget_order_alternates_by_game_index": True,
                    "order_offset": a.order_offset,
                    "shipped_max_actions_for_reference": 400,
                    # The frozen live generator DIES under sustained load. Recorded as a first-class
                    # observation, not swept up: in a 9h eval this will recur many times, each costing a
                    # model reload, and a death mid-induction wastes that induction's work.
                    "generator_heal_events": heals,
                    "generator_heal_count": len(heals),
                    "generator_heal_total_s": round(sum(h["heal_s"] for h in heals), 2),
                    "rows": rows,
                    "elapsed_s": round(time.time() - t0, 1),
                },
                indent=1,
            )
        )

    emit()
    for gi, g in enumerate(games):
        # Order alternates so a warm-up/KV effect cannot be read as a budget effect.
        order = list(budgets) if (gi + a.order_offset) % 2 == 0 else list(reversed(budgets))
        plan = [(b, "S") for b in order]
        if a.replicate_at:
            plan.append((a.replicate_at, "S_replicate"))
        for b, arm_key in plan:
            heal = ensure_healthy_between_cells(f"{g}/b{b}/{arm_key}")
            row = harness.run_cell(
                g,
                a.seed,
                budget=b,
                proposer=proposer,
                llm=True,
                extra_kwargs=dict(harness.ARMS[arm_key]),
                arm=f"{arm_key}_llmon_b{b}",
            )
            row["probe_budget_order_index"] = order.index(b) if b in order else None
            row["probe_budget_order"] = order
            row["probe_is_replicate"] = arm_key == "S_replicate"
            row["probe_heal_before_cell"] = heal
            rows.append(row)
            L = row.get("llm") or {}
            print(
                f"{g:5} b{b:<5} {arm_key:12} wall={row.get('wall_s')}s "
                f"llm={L.get('llm_wall_s')}s share="
                f"{round((L.get('llm_wall_s') or 0) / max(row.get('wall_s') or 1, 1e-9), 3)} "
                f"act={row.get('actions')} frames={row.get('n_frames')} "
                f"resets={row.get('n_resets')} lv={row.get('levels')} "
                f"ind={row.get('induction_attempts')}"
                f"(llm={row.get('induction_attempts_llm_reached')}) "
                f"resp={L.get('responses')} gen={L.get('generate_calls')} "
                f"tok_out={L.get('tokens_predicted')} tok_in={L.get('tokens_prompt')} "
                f"states={row.get('states_expanded')} "
                f"genok={row.get('generator_healthy_before')}->"
                f"{row.get('generator_healthy_after')} "
                f"srv={row.get('llama_servers_before')}->{row.get('llama_servers_after')} "
                f"VALID={row.get('llm_on_row_valid')}",
                flush=True,
            )
            emit()
    print(f"TOTAL {round(time.time() - t0, 1)}s n={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

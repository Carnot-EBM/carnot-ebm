#!/usr/bin/env python3
"""THE MAX_ACTIONS ANSWER: a budget with uncertainty, priced in the unit the competition pays on.

WHAT QUESTION THIS CLOSES
=========================
"What should MAX_ACTIONS be?" has been answered three times this project-week in three different
units, and no two of them are comparable:

  * `outer_loop_scored_path_budget_sweep_20260726.json` -- "budget 4000 fits" -- in OFFLINE ACTIONS
    against a 12h cap that turns out to be our own `subprocess(timeout=43200)`.
  * `outer_loop_arc_llm_on_wallclock_envelope_20260726.json` -- "only 400 fits, conservatively" --
    per-game WALL CLOCK measured UNCONTENDED, then scaled by a contention factor imported from
    LLM-OFF cells.
  * `outer_loop_arc_gateway_accurate_rescore_20260726.json` -- the per-level efficiency numbers all
    of the above rest on are OPTIMISTIC by a median 3.7% because the gateway charges a RESET an
    action and our harness charges it zero.

This file gives ONE answer in ONE unit chain, with the uncertainty attached: a budget, the LEVEL-UP
CELLS it buys, and the GATEWAY-CHARGED score those cells are worth. It changes no flag.

THE WORD "WIN" IS NOT USED AS A COUNT ANYWHERE IN THIS FILE (2026-07-27 correction). It used to be:
an accumulator named `wins` incremented on `levels > 0`, i.e. a cell that banked AT LEAST ONE level,
while two lines away the same function computed the real game-complete predicate as a DIFFERENT
variable (`won = lv >= len(baselines)`) and used that for scoring. So "median wins 3 -> 11" read as
"games won" and meant "cells reaching a first level-up". Every count is now named for its predicate:
`cells_with_at_least_one_levelup*` (levels > 0), and cross-seed sums carry the
`_summed_across_seeds` suffix so a per-seed number can never be read as a corpus total.

THE THREE UNITS, NEVER CONFLATED (stated on every number this file emits)
========================================================================
  (1) OFFLINE ACTIONS  -- our harness `actions`; EXCLUDES resets.
  (2) FRAMES           -- loop iterations; INCLUDES resets.
  (3) GATEWAY-CHARGED  -- non-RESET moves PLUS resets; the ONLY unit the score is a function of.
And separately, WALL CLOCK SECONDS, which is the unit the competition CAP is in.

WHAT IS NEW HERE, AND WHY IT WAS NEEDED
=======================================
1. THE CONTENTION LEVEL IS MEASURED WITH THE LLM ON, not imported from LLM-OFF cells. The prior
   conservative reading multiplied uncontended LLM-ON cost by 1.72x, a factor measured on cells that
   contend for CPU. LLM-ON cells contend for ONE llama-server with FOUR slots. Those are different
   queues, and the difference decides the answer: per-game LATENCY and total THROUGHPUT do not move
   together when the bottleneck is a saturated GPU. `arc_llm_on_contention_probe.py` measures the
   ladder K=1,2,4 on matched games/seed, and this analyser reads THROUGHPUT (games per hour) off it,
   because throughput -- not per-game latency -- is what a fixed wall-clock cap divides.

2. THE SCORE CURVE IS RE-PRICED IN THE GATEWAY UNIT. Level-up counts are not scored; efficiency-squared is,
   and resets are charged. Every budget's score is recomputed under three charge models (M0 offline,
   M1 all-resets-charged, M2 bootstrap-free -- the opening reset is FREE, per the installed
   `arc_agi/api.py` + `scorecard.py` chain), through the INSTALLED scorer, never a paraphrase.

3. THE EARLY-STOP LEVER IS PRICED ON THE BENEFIT SIDE. The grace sweep measured its SAFETY (no
   score regression) and found the shipped-budget saving negligible -- 0.072% of actions at b400 --
   for a FIXED window. That is a statement about fixed windows, not about the mechanism. This file
   computes the ORACLE CEILING of any early-stop rule (all actions after the last level-up are
   score-free by the settled charge model, so cutting them is free score-wise) and the ADAPTIVE
   window multiplier that would preserve every level-up actually observed in the corpus. If wall
   clock binds, that ceiling is the size of the prize for building the adaptive version.

MEASUREMENT FAILURES THIS FILE IS BUILT TO AVOID (each named in the brief, each guarded)
=======================================================================================
 #2  Forced gates: every gate carries a COMPUTED witness that its pass region was non-empty and
     could have failed.
 #3  Any-seed unions: score and win curves are PER-SEED MATCHED, then aggregated across seeds.
 #4  One-sided tests: every test reports BOTH tails, the favoured direction, and the MINIMUM
     REACHABLE p at the available support.
 #5  Dead channels: the human-baseline channel, the LLM-liveness channel, and the reset-attribution
     channel are each asserted POPULATED before any number derived from them is reported.
 #8  Analyser clock vs measurement clock: `duration_s` is this analyser's runtime;
     `measurement_wall_s` is summed from each ROW FILE's own `elapsed_s`.
 #11 Single-game witnesses: every verdict carries the games and cells it rests on.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import math
import os
import statistics as st
import sys
import time
from typing import Any, Sequence

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import arc_gateway_rescore as gw  # noqa: E402  (the prior lane's scorer-driving library, reused)

T_START = time.time()

CAPS = {
    # Ranked by provenance strength, exactly as the sibling envelope ranked them. The 12h figure is
    # OUR OWN subprocess timeout plus an external ARC-Prize-VERIFIED number that is explicitly NOT
    # the Kaggle leaderboard, so it is carried but never used as the headline basis.
    "kaggle_9h_max_notebook_runtime": 9 * 3600,
    "preview_8h": 8 * 3600,
    "kaggle_6h_arcagi3_specific": 6 * 3600,
    "our_own_subprocess_timeout_12h": 12 * 3600,
}
KERNEL_OVERHEAD_S_ASSUMED = 980.0  # model load + dataset mount + imports; inherited, ASSUMED
MARGIN_FRACTION = 0.80  # "fits with margin" == consumes <= 80% of the usable loop budget


# ---------------------------------------------------------------------------------- small helpers
def _median(xs: Sequence[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(st.median(xs)) if xs else None


def _mean(xs: Sequence[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(st.fmean(xs)) if xs else None


def both_tails_sign_test(deltas: Sequence[float]) -> dict[str, Any]:
    """Paired sign test reporting BOTH tails, the favoured direction, and the reachable p-floor.

    A one-sided test makes a REVERSAL read as "no effect"; a two-sided p on a tiny support can never
    clear 0.05 and must say so rather than be quoted as a null. Both are named failure modes.
    """
    d = [x for x in deltas if x is not None and x != 0]
    n = len(d)
    pos = sum(1 for x in d if x > 0)
    neg = n - pos

    def _binom_tail(k: int, n_: int) -> float:
        if n_ == 0:
            return 1.0
        return sum(math.comb(n_, i) for i in range(k, n_ + 1)) / (2.0**n_)

    p_greater = _binom_tail(pos, n) if n else 1.0
    p_less = _binom_tail(neg, n) if n else 1.0
    p_two = min(1.0, 2 * min(p_greater, p_less)) if n else 1.0
    p_floor = 2.0 / (2.0**n) if n else 1.0
    return {
        "n_nonzero_pairs": n,
        "n_positive": pos,
        "n_negative": neg,
        "p_two_sided": round(p_two, 4),
        "p_one_sided_increase": round(p_greater, 4),
        "p_one_sided_decrease": round(p_less, 4),
        "direction_favoured": ("increase" if pos > neg else "decrease" if neg > pos else "tie"),
        "min_reachable_two_sided_p_at_this_support": round(p_floor, 4),
        "can_ever_reach_0_05": bool(p_floor <= 0.05),
        "verdict": (
            "UNDERPOWERED_p_floor_above_0.05"
            if p_floor > 0.05
            else ("SIGNIFICANT" if p_two <= 0.05 else "NOT_SIGNIFICANT")
        ),
    }


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_fingerprint(path: str) -> dict[str, Any]:
    """Path + sha256 + size, or an explicit `unreadable` marker.

    Never a silent skip: the freshness lint distinguishes STALE from UNVERIFIABLE, and it can only do
    that if a missing input is recorded as missing rather than dropped.
    """
    try:
        return {
            "path": os.path.relpath(path, REPO),
            "sha256": _sha256(path),
            "bytes": os.path.getsize(path),
        }
    except Exception as exc:
        return {"path": os.path.relpath(path, REPO), "unreadable": f"{type(exc).__name__}:{exc}"}


def _git_head() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover -- recorded, never fatal
        return f"unavailable:{type(exc).__name__}"


def _code_dependencies() -> list[str]:
    """Every code file whose CONTENT can change a number in this artifact.

    Includes the two measurement-side modules (`arc_llm_on_contention_probe.py` produced the ladder
    rows; `arc_scored_path_lever_harness.py` + `arc_leaderboard_eval.py` produced every row's fields)
    as well as the analysis side. The freshness lint's own trigger regex is REGENERATED from this
    union, so a newly-added dependency cannot fall silently outside the hook.
    """
    return [
        os.path.join(REPO, p)
        for p in (
            "scripts/analyze_arc_max_actions_answer.py",
            "scripts/arc_gateway_rescore.py",
            "scripts/arc_llm_on_contention_probe.py",
            "scripts/arc_scored_path_lever_harness.py",
            "scripts/arc_leaderboard_eval.py",
        )
    ]


def _load_json(path: str) -> Any:
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    with open(path) as fh:
        return json.load(fh)


# ============================================================== PART 1: WALL CLOCK / CONTENTION
def part1_contention(ladder_paths: list[str], sibling_envelope: dict) -> dict[str, Any]:
    """Per-game LLM-ON wall clock and THROUGHPUT as a function of concurrency K.

    The distinction this part exists to make: when the bottleneck is a GPU already saturated by ONE
    game, running K games at once multiplies each game's LATENCY by ~K while leaving TOTAL
    THROUGHPUT flat. A fixed wall-clock cap divides throughput, not latency -- so scaling an
    uncontended per-game latency by a contention factor DOUBLE-COUNTS. Whether that is what happens
    here is measured, not assumed: the GPU utilisation trace during each batch says whether the card
    was saturated, and the batch walls say what throughput actually did.
    """
    cells: list[dict] = []
    batches: list[dict] = []
    files: list[dict] = []
    for p in ladder_paths:
        d = _load_json(p)
        files.append(
            {
                "path": os.path.relpath(p, REPO),
                "sha256": _sha256(p),
                "elapsed_s": d.get("elapsed_s"),
                "server_props": d.get("server_props"),
                "generator_device_resolved": d.get("generator_device_resolved"),
                "gpu1_residency_witness": d.get("gpu1_residency_witness"),
                "budget": d.get("budget"),
                "seed": d.get("seed"),
                "n_ctx_requested": d.get("n_ctx_requested"),
                "n_ctx_is_an_override_of_the_shipped_default": bool(
                    d.get("n_ctx_is_an_override_of_the_shipped_default")
                ),
                "honest_verdict_if_blocked": d.get("honest_verdict"),
            }
        )
        # The seed lives on the FILE, not the cell. Attaching it here is what makes the paired
        # ratios below per-seed MATCHED rather than an any-seed union (failure #3).
        # CONFIGURATION also lives on the FILE. A cell run against a server with a non-shipped
        # context window is a DIFFERENT configuration and must not be pooled into the cost analysis:
        # doing so would make an n_ctx experiment look like a concurrency result.
        cfg_override = bool(d.get("n_ctx_is_an_override_of_the_shipped_default"))
        for c in d.get("cells") or []:
            c = dict(c)
            c["_seed"] = d.get("seed")
            c["_n_ctx"] = d.get("n_ctx_requested") or 16384
            c["_is_config_override"] = cfg_override
            cells.append(c)
        for b in d.get("batches") or []:
            b = dict(b)
            b["_seed"] = d.get("seed")
            b["_is_config_override"] = cfg_override
            batches.append(b)

    # The SHIPPED-configuration cells are the cost measurement; override cells are a separate test.
    config_test_cells = [c for c in cells if c.get("_is_config_override")]
    cells = [c for c in cells if not c.get("_is_config_override")]
    batches = [b for b in batches if not b.get("_is_config_override")]

    if not cells:
        return {"usable": False, "reason": "no_ladder_cells_found", "files": files}

    # --- the device we ACTUALLY got, reported not assumed --------------------------------------
    # Taken from the SHIPPED-config files only, and asserted consistent across them -- reading
    # `files[0]` would silently report a configuration-test run's server settings as the headline's
    # once a second configuration exists in the same directory.
    shipped_files = [f for f in files if not f.get("n_ctx_is_an_override_of_the_shipped_default")]
    src = shipped_files[0] if shipped_files else files[0]
    dev = src.get("generator_device_resolved") or {}
    resident = src.get("gpu1_residency_witness") or []
    resident_mib = []
    for r in resident:
        try:
            resident_mib.append(int(str(r.get("used_mib", "0")).split()[0]))
        except Exception:
            pass
    resident_ok = bool(resident_mib and max(resident_mib) > 4000)
    resolver_cuda = (not dev.get("is_hip_igpu_build")) and dev.get(
        "cuda_visible_devices_pinned"
    ) == "1"
    device_report = {
        # TWO INDEPENDENT SIGNALS, and the verdict must not rest on the resolver alone. The resolver
        # answers "what would a FRESH spawn launch NOW"; when a healthy server is already bound to the
        # port, `_ensure_server` reuses it and never consults the resolver. On the seed-20260724 run
        # that is exactly what happened -- the resolver reported the iGPU build because the CUDA
        # headroom guard saw the (CUDA-spawned) server's own 12 GiB -- so reading the resolver as "the
        # device we got" would report the WRONG CARD while per-PID VRAM proved otherwise. An earlier
        # draft of this file did exactly that and printed NOT_THE_REQUESTED_DEVICE next to a G1 gate
        # that passed, i.e. contradicted itself.
        "resolver_would_launch_binary": dev.get("server_binary"),
        "resolver_would_launch_is_hip_igpu_build": dev.get("is_hip_igpu_build"),
        "resolver_says_cuda_gpu1": resolver_cuda,
        "cuda_visible_devices_pinned": dev.get("cuda_visible_devices_pinned"),
        "requested_gpu_env": dev.get("requested_gpu"),
        "gpu1_compute_apps": resident,
        "max_resident_mib_on_gpu1": max(resident_mib) if resident_mib else None,
        "weights_are_resident_on_gpu1": resident_ok,
        "per_file_device_basis": [
            {
                "path": f.get("path"),
                "resolver_says_cuda_gpu1": bool(
                    (not (f.get("generator_device_resolved") or {}).get("is_hip_igpu_build"))
                    and (f.get("generator_device_resolved") or {}).get(
                        "cuda_visible_devices_pinned"
                    )
                    == "1"
                ),
                "gpu1_residency_witness": f.get("gpu1_residency_witness"),
                "server_props": f.get("server_props"),
            }
            for f in files
        ],
        "server_props": src.get("server_props"),
        "n_shipped_config_row_files": len(shipped_files),
        "all_shipped_files_report_the_same_server_props": bool(
            len({json.dumps(f.get("server_props"), sort_keys=True) for f in shipped_files}) <= 1
        ),
        "verdict": (
            "CUDA_GPU1"
            if (resolver_cuda or resident_ok)
            else "NOT_THE_REQUESTED_DEVICE__NEITHER_SIGNAL_CONFIRMS_CARD_1"
        ),
        "verdict_basis": (
            "resolver+residency"
            if (resolver_cuda and resident_ok)
            else ("per_PID_VRAM_residency_on_physical_card_1" if resident_ok else "resolver_only")
        ),
        "why_this_is_checked": (
            "the proposer FALLS THROUGH to the AMD iGPU HIP build silently when the CUDA headroom "
            "guard trips, and that build exists on this box -- so setting the env var is not "
            "evidence the env var took. Resolved through the proposer's own resolver and "
            "cross-checked against per-PID VRAM by GPU UUID."
        ),
    }

    # --- per (K, game, seed) -------------------------------------------------------------------
    by_kg: dict[tuple[int, str, int], dict] = {}
    for c in cells:
        row = c.get("row") or {}
        K = int(c.get("concurrency_K") or 0)
        g = str(c.get("game") or "")
        sd = int(c.get("_seed") or c.get("seed") or 0)
        L = row.get("llm") or {}
        by_kg[(K, g, sd)] = {
            "K": K,
            "game": g,
            "seed": sd,
            "worker_ok": bool(c.get("worker_ok")),
            "wall_s": row.get("wall_s"),
            "llm_wall_s": L.get("llm_wall_s"),
            "non_llm_wall_s": (
                round(float(row["wall_s"]) - float(L.get("llm_wall_s") or 0.0), 2)
                if row.get("wall_s") is not None
                else None
            ),
            "llm_responses": L.get("responses"),
            "llm_errors": L.get("errors"),
            "tokens_predicted": L.get("tokens_predicted"),
            "tokens_prompt_CUMULATIVE_SUM_not_per_request": L.get("tokens_prompt"),
            "prompt_truncated": L.get("prompt_truncated"),
            "inductions": row.get("induction_attempts"),
            "inductions_llm_reached": row.get("induction_attempts_llm_reached"),
            "generator_healthy_after": row.get("generator_healthy_after"),
            "llm_on_row_valid": row.get("llm_on_row_valid"),
            "server_storm_suspected": row.get("server_storm_suspected"),
            "levels": row.get("levels"),
            "offline_actions": row.get("actions"),
            "frames": row.get("n_frames"),
            "resets": row.get("n_resets"),
            "states_expanded": row.get("states_expanded"),
            "llama_servers_before": row.get("llama_servers_before"),
            "llama_servers_after": row.get("llama_servers_after"),
            "gpu0_util_mean_during_batch": (c.get("gpu_during_batch") or {}).get(
                "gpu0_util_mean_CONDUCTORS_CARD_recorded_not_gated"
            ),
            "gpu1_util_mean_during_batch": (c.get("gpu_during_batch") or {}).get("gpu1_util_mean"),
        }
        # WHY a cell is invalid, distinguished -- because the two reasons have different owners. A
        # generator DEATH is this configuration failing. A rising llama-server COUNT is
        # `server_storm_suspected`, and that counter is SYSTEM-WIDE: the conductor runs its own ARC
        # generator on GPU 0, so a server it starts mid-cell trips this probe's storm flag without any
        # storm of ours. Both still invalidate the cell (its wall clock is contended by something we
        # did not control), but conflating them would mis-attribute the cause.
        _sb, _sa = row.get("llama_servers_before"), row.get("llama_servers_after")
        by_kg[(K, g, sd)]["invalid_reasons"] = [
            r
            for r in (
                None if c.get("worker_ok") else "worker_crashed",
                "generator_unhealthy_after"
                if row.get("generator_healthy_after") is False
                else None,
                "server_count_increased_storm_suspected"
                if (isinstance(_sb, int) and isinstance(_sa, int) and _sa > _sb)
                else None,
                "no_llm_responses" if not (row.get("llm") or {}).get("responses") else None,
            )
            if r
        ]
        # VALIDITY, and why it has to gate the wall-clock analysis rather than just be reported: when
        # the generator dies, `generate()` returns (False, msg) instead of raising, so the cell
        # COMPLETES a full 400-action run with almost no LLM work -- i.e. it looks FAST. Averaging a
        # dead-server arm into a throughput number does not add noise, it inverts the conclusion
        # ("concurrency is 2.7x faster!"). Invalid cells are therefore excluded from every cost number
        # and reported separately as the concurrency-failure finding they are.
        by_kg[(K, g, sd)]["row_valid"] = bool(c.get("worker_ok")) and bool(
            row.get("llm_on_row_valid")
        )
    Ks = sorted({k[0] for k in by_kg})
    games = sorted({k[1] for k in by_kg})
    seeds = sorted({k[2] for k in by_kg})

    # --- the LLM-liveness channel must be ALIVE on every arm ----------------------------------
    liveness = {}
    for K in Ks:
        arm = [v for k, v in by_kg.items() if k[0] == K]
        liveness[str(K)] = {
            "n_cells": len(arm),
            "n_valid": sum(1 for a in arm if a["llm_on_row_valid"]),
            "n_with_llm_responses": sum(1 for a in arm if (a["llm_responses"] or 0) > 0),
            "n_llm_errors_total": sum(int(a["llm_errors"] or 0) for a in arm),
            "n_inductions_reaching_llm": sum(int(a["inductions_llm_reached"] or 0) for a in arm),
            "n_generator_unhealthy_after": sum(
                1 for a in arm if a["generator_healthy_after"] is False
            ),
            "n_server_storm_suspected": sum(1 for a in arm if a["server_storm_suspected"]),
            "arm_is_genuinely_llm_on": bool(arm)
            and all((a["llm_responses"] or 0) > 0 for a in arm),
        }

    # --- the concurrency FAILURE finding, kept out of every cost number ------------------------
    invalid = [v for v in by_kg.values() if not v["row_valid"]]
    death = {
        "n_invalid_cells": len(invalid),
        "invalid_cells": [
            {
                "K": v["K"],
                "game": v["game"],
                "seed": v["seed"],
                "wall_s": v["wall_s"],
                "llm_responses": v["llm_responses"],
                "tokens_predicted": v["tokens_predicted"],
                "generator_healthy_after": v["generator_healthy_after"],
                "llama_servers_before_after": [v["llama_servers_before"], v["llama_servers_after"]],
                "inductions_llm_reached": v["inductions_llm_reached"],
                "invalid_reasons": v.get("invalid_reasons"),
                "gpu0_util_mean_during_batch_CONDUCTORS_CARD": v.get("gpu0_util_mean_during_batch"),
            }
            for v in invalid
        ],
        "invalid_reason_census": {
            r: sum(1 for v in invalid if r in (v.get("invalid_reasons") or []))
            for r in (
                "worker_crashed",
                "generator_unhealthy_after",
                "server_count_increased_storm_suspected",
                "no_llm_responses",
            )
        },
        "storm_flag_is_system_wide_not_port_scoped": (
            "`_llama_server_count()` counts every llama-server process on the box. The conductor runs "
            "its own ARC generator, so a server IT starts during one of these cells trips the flag with "
            "no storm of ours. Such a cell is still excluded -- its wall clock was contended by "
            "something outside this experiment -- but the cause is attributed rather than assumed, and "
            "the GPU-0 utilisation during that batch is recorded beside it as the available evidence."
        ),
        "Ks_with_any_invalid_cell": sorted({v["K"] for v in invalid}),
        "Ks_fully_invalid": sorted(
            {
                K
                for K in Ks
                if all(not v["row_valid"] for k, v in by_kg.items() if k[0] == K)
                and any(k[0] == K for k in by_kg)
            }
        ),
        "what_this_is": (
            "The generator DIED under concurrency. Signature: `llama_servers` drops from 2 to 1 (the "
            "probe's own server disappears; an unrelated wedged one survives), `generator_healthy_"
            "after` flips True->False, `llm_responses` collapses to 0-1, `tokens_predicted` to ~0 -- "
            "and every cell still returns exit code 0 with a complete 400-action run. `forbid_spawn` "
            "is why it stays dead for the rest of the batch: that is the deliberate trade (one lost "
            "batch instead of a second 12 GiB model forked onto the card being measured)."
        ),
        "why_it_is_excluded_from_the_cost_numbers": (
            "A dead-generator cell is FASTER, not noisier. Including these arms would report "
            "concurrency as a large throughput WIN when what actually happened is that the LLM work "
            "stopped happening."
        ),
        "llm_errors_are_zero_and_that_is_not_reassuring": (
            "The wrapper's `errors` counter only sees raised exceptions. `generate()` handles the "
            "server failure internally and returns (False, msg), so a total generator loss is "
            "consistent with errors=0. The liveness witnesses, not the error count, are what detect it."
        ),
    }

    # --- per-game paired ratios vs K=1 (matched game AND seed by construction) -----------------
    ratios: dict[str, Any] = {}
    for K in Ks:
        if K == 1:
            continue
        pairs = []
        for g in games:
            for sd in seeds:
                a1, aK = by_kg.get((1, g, sd)), by_kg.get((K, g, sd))
                if not a1 or not aK or not a1["wall_s"] or not aK["wall_s"]:
                    continue
                if not (a1["row_valid"] and aK["row_valid"]):
                    continue
                pairs.append(
                    {
                        "game": g,
                        "seed": sd,
                        "wall_K1_s": a1["wall_s"],
                        "wall_KN_s": aK["wall_s"],
                        "latency_ratio": round(aK["wall_s"] / a1["wall_s"], 3),
                        "llm_wall_ratio": (
                            round((aK["llm_wall_s"] or 0) / (a1["llm_wall_s"] or 1e-9), 3)
                            if a1["llm_wall_s"]
                            else None
                        ),
                        "tokens_predicted_K1": a1["tokens_predicted"],
                        "tokens_predicted_KN": aK["tokens_predicted"],
                    }
                )
        ratios[f"K{K}_vs_K1"] = {
            "usable": bool(pairs),
            "unusable_reason": (
                None
                if pairs
                else "no matched pair where BOTH the K=1 and the K>1 cell are valid -- this arm was "
                "invalidated (see generator_death_under_concurrency), so there is no latency ratio to "
                "report and the test below is empty by construction, not null by measurement"
            ),
            "n_matched_game_seed_pairs": len(pairs),
            "pairs": pairs,
            "latency_ratio_median": _median([p["latency_ratio"] for p in pairs]),
            "latency_ratio_values": [p["latency_ratio"] for p in pairs],
            "sign_test_latency_ratio_gt_1": both_tails_sign_test(
                [p["latency_ratio"] - 1.0 for p in pairs]
            ),
        }

    # --- THROUGHPUT: what a fixed cap actually divides ------------------------------------------
    # A batch is a throughput point ONLY if every cell in it is valid: the batch wall of a
    # dead-generator batch measures a degraded run, not the configuration under test.
    valid_batch_keys = set()
    for b in batches:
        K = int(b.get("K") or 0)
        sd = int(b.get("_seed") or 0)
        gs = list(b.get("games") or [])
        if gs and all((by_kg.get((K, g, sd)) or {}).get("row_valid") for g in gs):
            valid_batch_keys.add((K, sd, b.get("batch_index")))
    valid_batches = [
        b
        for b in batches
        if (int(b.get("K") or 0), int(b.get("_seed") or 0), b.get("batch_index"))
        in valid_batch_keys
    ]

    thr = {}
    for K in Ks:
        bs = [b for b in valid_batches if int(b.get("K") or 0) == K]
        per_seed_s = {}
        for sd in seeds:
            bss = [b for b in bs if int(b.get("_seed") or 0) == sd]
            nc = sum(int(b.get("n_cells") or 0) for b in bss)
            wl = sum(float(b.get("batch_wall_s") or 0.0) for b in bss)
            per_seed_s[str(sd)] = round(wl / nc, 1) if nc else None
        n_cells = sum(int(b.get("n_cells") or 0) for b in bs)
        wall = sum(float(b.get("batch_wall_s") or 0.0) for b in bs)
        util = [
            b.get("gpu_during_batch", {}).get("gpu1_util_mean")
            for b in bs
            if b.get("gpu_during_batch")
        ]
        overlap_ok = all(bool(b.get("concurrency_actually_achieved")) for b in bs)
        thr[str(K)] = {
            "n_valid_batches": len(bs),
            "n_batches_excluded_as_invalid": sum(1 for b in batches if int(b.get("K") or 0) == K)
            - len(bs),
            "is_a_usable_throughput_point": bool(bs),
            "n_cells": n_cells,
            "total_batch_wall_s": round(wall, 1),
            "s_per_game_throughput": round(wall / n_cells, 1) if n_cells else None,
            "s_per_game_throughput_per_seed": per_seed_s,
            "games_per_hour": round(3600.0 * n_cells / wall, 2) if wall else None,
            "gpu1_util_mean_across_batches": _mean([u for u in util if u is not None]),
            "gpu1_util_max": max(
                [b.get("gpu_during_batch", {}).get("gpu1_util_max") or 0 for b in bs],
                default=None,
            ),
            "concurrency_actually_achieved_all_batches": overlap_ok,
            "overlap_fraction_of_longest_cell": [
                b.get("overlap_fraction_of_longest_cell") for b in bs
            ],
        }
    # MATCHED-SET throughput ratio. The unmatched version divides "all valid K>1 batches" by "all
    # K=1 batches", and when a K>1 batch is invalidated the two arms no longer cover the same GAMES --
    # comparing totals over different sets instead of matched sets is a defect this project has shipped
    # before. So for each (K, seed) the ratio is computed over exactly the games that have a VALID cell
    # in BOTH arms, and the game set is published with the ratio.
    matched_ratio: dict[str, Any] = {}
    for K in Ks:
        if K == 1:
            continue
        per_seed = {}
        for sd in seeds:
            bss = [
                b
                for b in valid_batches
                if int(b.get("K") or 0) == K and int(b.get("_seed") or 0) == sd
            ]
            gs = [g for b in bss for g in (b.get("games") or [])]
            k1 = [
                b
                for b in valid_batches
                if int(b.get("K") or 0) == 1
                and int(b.get("_seed") or 0) == sd
                and all(g in gs for g in (b.get("games") or []))
            ]
            wk = sum(float(b.get("batch_wall_s") or 0.0) for b in bss)
            nk = sum(int(b.get("n_cells") or 0) for b in bss)
            w1 = sum(float(b.get("batch_wall_s") or 0.0) for b in k1)
            n1 = sum(int(b.get("n_cells") or 0) for b in k1)
            if nk and n1 and sorted(gs) == sorted([g for b in k1 for g in (b.get("games") or [])]):
                per_seed[str(sd)] = {
                    "games_matched": sorted(gs),
                    "s_per_game_KN": round(wk / nk, 1),
                    "s_per_game_K1": round(w1 / n1, 1),
                    "ratio_KN_over_K1": round((wk / nk) / (w1 / n1), 3),
                }
            else:
                per_seed[str(sd)] = {
                    "games_matched": sorted(gs),
                    "usable": False,
                    "reason": "no matched valid K=1/K>1 game set for this seed",
                }
        vals = [
            v["ratio_KN_over_K1"]
            for v in per_seed.values()
            if v.get("ratio_KN_over_K1") is not None
        ]
        matched_ratio[str(K)] = {
            "per_seed": per_seed,
            "ratio_values": vals,
            "ratio_median": _median(vals),
            "n_seeds_usable": len(vals),
            # NOT named `min_reachable_two_sided_p_at_this_support`: G7 walks the artifact for any dict
            # carrying that key and requires BOTH tails on it. This block is a SUMMARY, not a test --
            # the test is nested below -- and using the reserved name here made G7 fail on my own code
            # (correctly). Renamed rather than exempted.
            "p_floor_for_a_test_at_this_seed_support": (
                round(2.0 / (2.0 ** len(vals)), 4) if vals else 1.0
            ),
            "sign_test_ratio_gt_1": both_tails_sign_test([v - 1.0 for v in vals]),
        }

    base = thr.get("1", {}).get("s_per_game_throughput")
    for K in Ks:
        s = thr[str(K)].get("s_per_game_throughput")
        thr[str(K)]["throughput_s_per_game_ratio_vs_K1_UNMATCHED_do_not_use_for_the_answer"] = (
            round(s / base, 3) if (s and base) else None
        )
        thr[str(K)]["throughput_s_per_game_ratio_vs_K1"] = (
            (matched_ratio.get(str(K)) or {}).get("ratio_median") if K != 1 else 1.0
        )

    # --- the K=1 arm as a REPLICATE of the published uncontended level -------------------------
    # The published b400 per-game walls, paired to games by POSITION. The sibling artifact stores
    # `wall_s_values` as a list ordered by `games_with_complete_budget_coverage`, so the zip below is
    # the only available join -- asserted rather than assumed by checking the lengths match.
    bc = (sibling_envelope.get("budget_curve") or {}).get("400") or {}
    sib_games = sibling_envelope.get("games_with_complete_budget_coverage") or []
    sib_walls = bc.get("wall_s_values") or []
    sib_join_ok = len(sib_games) == len(sib_walls) and bool(sib_games)
    sib_b400 = dict(zip(sib_games, sib_walls)) if sib_join_ok else {}
    sib_seed = sibling_envelope.get("random_seed")
    repl = []
    for g in games:
        for sd in seeds:
            a1 = by_kg.get((1, g, sd))
            if a1 and a1["row_valid"] and a1["wall_s"] and g in sib_b400:
                repl.append(
                    {
                        "game": g,
                        "my_seed": sd,
                        "published_seed": sib_seed,
                        "same_seed_as_published": bool(sd == sib_seed),
                        "published_uncontended_wall_s": sib_b400[g],
                        "my_K1_wall_s": a1["wall_s"],
                        "ratio_mine_over_published": round(a1["wall_s"] / sib_b400[g], 3),
                    }
                )
    replicate = {
        "positional_join_to_the_published_values_verified": sib_join_ok,
        "why": (
            "The K=1 arm re-runs the SAME games at the SAME seed and budget as the published "
            "uncontended anchor. Two same-config runs of an LLM-ON cell are NOT expected to agree "
            "exactly (the generator samples; the published same-config fold change spans 0.52x-2.47x) "
            "-- so this tests the LEVEL for gross disagreement, not for equality."
        ),
        "pairs": repl,
        "ratio_median": _median([r["ratio_mine_over_published"] for r in repl]),
        "ratio_range": (
            [
                min(r["ratio_mine_over_published"] for r in repl),
                max(r["ratio_mine_over_published"] for r in repl),
            ]
            if repl
            else None
        ),
        "published_same_config_fold_change_median": (
            (sibling_envelope.get("noise_floor") or {}).get("same_config_fold_change_median")
        ),
        "sign_test": both_tails_sign_test([r["ratio_mine_over_published"] - 1.0 for r in repl]),
        "level_reproduces_within_the_published_noise_floor": bool(
            repl
            and max(
                max(r["ratio_mine_over_published"] for r in repl),
                1.0 / min(r["ratio_mine_over_published"] for r in repl),
            )
            <= (
                (sibling_envelope.get("noise_floor") or {}).get("same_config_fold_change_max")
                or 2.468
            )
        ),
    }

    return {
        "usable": True,
        "row_files": files,
        "device_actually_used": device_report,
        "Ks_measured": Ks,
        "games": games,
        "seeds": seeds,
        "per_cell": [by_kg[k] for k in sorted(by_kg)],
        "llm_liveness_by_K": liveness,
        "generator_death_under_concurrency": death,
        "config_test_does_a_bigger_server_context_survive_the_same_concurrency": _config_test(
            config_test_cells, by_kg
        ),
        "latency_ratios": ratios,
        "throughput_by_K": thr,
        "k1_replicate_of_published_uncontended": replicate,
        "matched_set_throughput_ratio_by_K": matched_ratio,
        "batches": batches,
    }


def _config_test(config_cells: list[dict], by_kg: dict) -> dict[str, Any]:
    """Matched configuration test: SAME games/seed/budget/K, only the server context window differs.

    WHY IT IS SEPARATE FROM THE COST ANALYSIS. These cells ran against a server with a non-shipped
    `-c`, so pooling them into the throughput numbers would report a configuration change as a
    concurrency result. They answer one question: was the K=4 generator death CONTEXT-driven? The
    shipped per-slot context is `n_ctx / total_slots` = 4096 at 4 slots, which is exactly the agent's
    own `max_tokens` request -- leaving nothing for the prompt. If that is the mechanism, a window
    large enough for prompt+completion per slot should survive the same concurrency.
    """
    if not config_cells:
        return {
            "ran": False,
            "why_not": "no configuration-override row file was present",
            "hypothesis_it_would_test": (
                "per-slot context (n_ctx/total_slots) smaller than prompt+max_tokens is what kills "
                "the server at K=4"
            ),
        }
    out = []
    for c in config_cells:
        row = c.get("row") or {}
        L = row.get("llm") or {}
        K = int(c.get("concurrency_K") or 0)
        g = str(c.get("game") or "")
        sd = int(c.get("_seed") or 0)
        shipped = by_kg.get((K, g, sd)) or {}
        out.append(
            {
                "game": g,
                "seed": sd,
                "K": K,
                "n_ctx": c.get("_n_ctx"),
                "wall_s": row.get("wall_s"),
                "llm_responses": L.get("responses"),
                "tokens_predicted": L.get("tokens_predicted"),
                "generator_healthy_after": row.get("generator_healthy_after"),
                "llm_on_row_valid": row.get("llm_on_row_valid"),
                "levels": row.get("levels"),
                "llama_servers_before_after": [
                    row.get("llama_servers_before"),
                    row.get("llama_servers_after"),
                ],
                "SHIPPED_CONFIG_SAME_CELL": (
                    {
                        "n_ctx": 16384,
                        "wall_s": shipped.get("wall_s"),
                        "llm_responses": shipped.get("llm_responses"),
                        "generator_healthy_after": shipped.get("generator_healthy_after"),
                        "llm_on_row_valid": shipped.get("row_valid"),
                        "invalid_reasons": shipped.get("invalid_reasons"),
                    }
                    if shipped
                    else None
                ),
            }
        )
    n_valid = sum(1 for r in out if r["llm_on_row_valid"])
    matched = [r for r in out if r["SHIPPED_CONFIG_SAME_CELL"]]
    # SECONDARY observation, explicitly NOT folded into the shipped-config cost levels: if the bigger
    # window survives, this arm also yields a K=4 THROUGHPUT point. The batch wall is reconstructed
    # from the workers' own start/end epochs (max end - min start) rather than from a batch record, so
    # it is a floor on the true batch wall (it excludes the parent's per-batch overhead).
    starts = [float(c["t_start_epoch"]) for c in config_cells if c.get("t_start_epoch")]
    ends = [float(c["t_end_epoch"]) for c in config_cells if c.get("t_end_epoch")]
    batch_wall = (max(ends) - min(starts)) if (starts and ends) else None
    throughput = {
        "reconstructed_batch_wall_s": round(batch_wall, 1) if batch_wall else None,
        "n_cells": len(config_cells),
        "s_per_game": round(batch_wall / len(config_cells), 1)
        if (batch_wall and config_cells)
        else None,
        "per_game_latency_s": [r["wall_s"] for r in out],
        "why_this_is_reported_separately": (
            "It is a DIFFERENT server configuration from the one the cost levels are built on. It is "
            "worth stating because it tests the same latency-vs-throughput decomposition one step "
            "further out in K: if per-game latency scales ~K while s/game stays near the K=1 value, "
            "the cap-relevant quantity is unchanged at 4-way concurrency too."
        ),
    }
    shipped_valid = sum(1 for r in matched if r["SHIPPED_CONFIG_SAME_CELL"]["llm_on_row_valid"])
    return {
        "ran": True,
        "n_cells": len(out),
        "n_valid_under_the_bigger_context": n_valid,
        "n_matched_shipped_cells": len(matched),
        "n_valid_under_the_shipped_context": shipped_valid,
        "throughput_under_the_bigger_context_SEPARATE_CONFIG": throughput,
        "cells": out,
        "verdict": (
            "CONTEXT_DRIVEN__the_bigger_window_survived_the_same_concurrency_that_killed_the_shipped_one"
            if (n_valid == len(out) and matched and shipped_valid == 0)
            else (
                "NOT_CONTEXT_DRIVEN__the_bigger_window_died_too"
                if n_valid == 0
                else "MIXED_OR_UNMATCHED__see_cells"
            )
        ),
        "what_a_pass_does_NOT_license": (
            "It does not license flipping `-c` in the shipped config: per-slot VRAM cost at eval scale "
            "is unmeasured here, `--parallel 1` is a cheaper alternative if throughput can be traded, "
            "and the decision is the operator's. It identifies the MECHANISM, nothing more."
        ),
        "power": (
            "One batch of 4 cells at one seed against one matched shipped batch. Enough to identify a "
            "mechanism when the contrast is total (all survive vs all die); NOT enough to characterise "
            "an intermittent failure rate, and the shipped-config death has itself been observed to be "
            "intermittent."
        ),
    }


# ================================================== PART 2: THE SCORE CURVE IN THE GATEWAY UNIT
def _spans_from_row(row: dict) -> tuple[list[int], int, list[int]]:
    """(per-level OFFLINE spans, tail offline actions, per-level human baselines).

    `level_up_actions` in these rows is CUMULATIVE offline actions at each level-up; `per_level`
    carries the human baseline per level. Both conventions are asserted by the callers below rather
    than trusted, because mixing cumulative with per-span is a live misreading trap in this schema.
    """
    lua = [int(x) for x in (row.get("level_up_actions") or [])]
    spans, prev = [], 0
    for at in lua:
        spans.append(at - prev)
        prev = at
    tail = int(row.get("actions") or 0) - prev
    baselines = [int(p.get("human_actions") or 0) for p in (row.get("per_level") or [])]
    return spans, max(0, tail), baselines


def _score(baselines: Sequence[int], spans: Sequence[int], tail: int, game_won: bool) -> float:
    s, _ = gw.gateway_score_via_calculator(baselines, spans, tail, game_won=game_won)
    return float(s)


def _uniform_rate_cum_resets(spans: Sequence[int], total_offline: int, n_resets: int) -> list[int]:
    """Estimate cumulative resets at each level-up by assuming resets are spread UNIFORMLY over
    offline actions.

    This is the only estimator available corpus-wide: per-level reset attribution was added to the
    harness on 2026-07-26 and exists on 48 cells. It is VALIDATED against those 48 in
    `part2_score_curve.estimator_validation` and its error is published, so a corpus number derived
    from it carries its own accuracy rather than borrowing the exact cells' credibility.
    """
    out, run = [], 0
    for s in spans:
        run += int(s)
        out.append(int(round(n_resets * (run / total_offline))) if total_offline > 0 else 0)
    # monotone non-decreasing, capped at the run total
    fixed, prev = [], 0
    for v in out:
        v = max(prev, min(int(n_resets), v))
        fixed.append(v)
        prev = v
    return fixed


def _charged_spans(
    spans: Sequence[int], cum_resets: Sequence[int], free_opening: bool
) -> list[int]:
    """Per-level CHARGED spans from per-level OFFLINE spans + CUMULATIVE resets at each level-up.

    `free_opening` implements M2: the FIRST reset of a play is routed by `update_scorecard` to
    `new_play` -> `inc_play_count`, which appends a zeroed counter row and charges NOTHING; only
    `reset` -> `inc_reset_count` charges an action. The opening RESET is therefore free.
    """
    out, prev = [], 0
    for i, s in enumerate(spans):
        add = int(cum_resets[i]) - prev
        prev = int(cum_resets[i])
        if free_opening and i == 0:
            add = max(0, add - 1)
        out.append(int(s) + add)
    return out


def part2_score_curve(sweep_files: list[str], exact_files: list[str]) -> dict[str, Any]:
    """Per-budget authoritative score in three charge models, per-seed matched, plus win counts."""
    # ---- corpus: the shipped (grace=None) arm of every early-stop sweep file -----------------
    rows: list[dict] = []
    files_meta = []
    for p in sweep_files:
        d = _load_json(p)
        files_meta.append(
            {
                "path": os.path.relpath(p, REPO),
                "sha256": _sha256(p),
                "elapsed_s": d.get("elapsed_s"),
                "tag": d.get("tag"),
                "llm_enabled": d.get("llm_enabled"),
            }
        )
        for r in d.get("rows") or []:
            if r.get("early_stop_grace") is None:  # the SHIPPED configuration
                rows.append(r)

    # ---- dead-channel guard: the human-baseline channel -------------------------------------
    with_levels = [r for r in rows if int(r.get("levels") or 0) > 0]
    baseline_alive = 0
    for r in with_levels:
        _, _, b = _spans_from_row(r)
        if b and all(b):
            baseline_alive += 1
    channel = {
        "n_rows_grace_none": len(rows),
        "n_rows_with_a_levelup": len(with_levels),
        "n_rows_with_live_baselines": baseline_alive,
        "baseline_channel_alive": bool(with_levels and baseline_alive == len(with_levels)),
        "why": (
            "A zero human baseline makes every charge model agree at score 0 and reads as a clean "
            "'no optimism' null -- the dead-channel failure mode. A prior agent hit exactly this by "
            "reading `env.baseline_actions` when the value lives on `env.info`."
        ),
        "n_rows_with_reset_count": sum(1 for r in rows if r.get("n_resets") is not None),
    }

    # ---- estimator validation against the 48 exact-attribution cells ------------------------
    exact_cells = []
    for p in exact_files:
        d = _load_json(p)
        files_meta.append(
            {
                "path": os.path.relpath(p, REPO),
                "sha256": _sha256(p),
                "measurement_wall_s": d.get("measurement_wall_s"),
                "n_cells": d.get("n_cells"),
            }
        )
        exact_cells.extend(d.get("cells") or [])

    val_rows = []
    for c in exact_cells:
        lv = int(c.get("levels") or 0)
        if lv <= 0:
            continue
        spans = [int(x) for x in (c.get("level_up_actions_offline") or [])]
        cum_r = [int(x) for x in (c.get("resets_before_levelups") or [])]
        cum_c = [int(x) for x in (c.get("level_up_charged") or [])]
        baselines = [int(p.get("human_actions") or 0) for p in (c.get("per_level") or [])]
        if (
            not (len(spans) == len(cum_r) == len(cum_c) == lv)
            or not baselines
            or not all(baselines)
        ):
            continue
        # convention assertion: charged == cumsum(offline) + cumulative resets
        cum_off, run = [], 0
        for s in spans:
            run += s
            cum_off.append(run)
        conv_ok = all(cum_c[i] == cum_off[i] + cum_r[i] for i in range(lv))
        total_off = int(c.get("offline_actions") or 0)
        n_res = int(c.get("n_resets") or 0)
        tail = max(0, total_off - sum(spans))
        won = lv >= len(baselines)
        est_cum = _uniform_rate_cum_resets(spans, total_off, n_res)
        m1_exact = _score(baselines, _charged_spans(spans, cum_r, False), tail, won)
        m1_est = _score(baselines, _charged_spans(spans, est_cum, False), tail, won)
        m2_exact = _score(baselines, _charged_spans(spans, cum_r, True), tail, won)
        m0 = _score(baselines, list(spans), tail, won)
        val_rows.append(
            {
                "game": c.get("game"),
                "seed": c.get("seed"),
                "budget": c.get("budget"),
                "levels": lv,
                "n_resets": n_res,
                "convention_identity_holds": conv_ok,
                "exact_cum_resets_at_levelups": cum_r,
                "estimated_cum_resets_at_levelups": est_cum,
                "m0_offline_score": round(m0, 6),
                "m1_exact_score": round(m1_exact, 6),
                "m1_estimated_score": round(m1_est, 6),
                "m2_exact_score": round(m2_exact, 6),
                "estimator_signed_rel_error_vs_exact_m1": (
                    round((m1_est - m1_exact) / m1_exact, 6) if m1_exact > 0 else None
                ),
                "exact_m1_rel_optimism_of_m0": (round((m0 - m1_exact) / m0, 6) if m0 > 0 else None),
                "exact_m2_rel_optimism_of_m0": (round((m0 - m2_exact) / m0, 6) if m0 > 0 else None),
            }
        )
    errs = [v["estimator_signed_rel_error_vs_exact_m1"] for v in val_rows]
    errs = [e for e in errs if e is not None]
    estimator_validation = {
        "n_exact_cells_usable": len(val_rows),
        "convention_identity_holds_on_all": all(v["convention_identity_holds"] for v in val_rows),
        "signed_rel_error_median": _median(errs),
        "signed_rel_error_mean": _mean(errs),
        "abs_rel_error_median": _median([abs(e) for e in errs]),
        "abs_rel_error_p90": (
            round(sorted(abs(e) for e in errs)[int(0.9 * (len(errs) - 1))], 6) if errs else None
        ),
        "abs_rel_error_max": (round(max(abs(e) for e in errs), 6) if errs else None),
        "n_cells_estimator_too_optimistic": sum(1 for e in errs if e > 0),
        "n_cells_estimator_too_pessimistic": sum(1 for e in errs if e < 0),
        "direction_test": both_tails_sign_test(errs),
        "cells": val_rows,
        "reading": (
            "The estimator's error is what a corpus-wide M1/M2 point estimate inherits. It is "
            "reported per cell and in aggregate so no corpus number here can be read as exact."
        ),
    }

    # ---- per (budget, seed) score sums under M0 / M1 / M2 + hard bounds ----------------------
    per_bs: dict[tuple[int, int], dict] = {}
    for r in rows:
        b, s = int(r.get("budget") or 0), int(r.get("seed") or 0)
        key = (b, s)
        acc = per_bs.setdefault(
            key,
            {
                "budget": b,
                "seed": s,
                "n_games": 0,
                "cells_with_at_least_one_levelup": 0,
                "levels_total": 0,
                "m0": 0.0,
                "m1_est": 0.0,
                "m2_est": 0.0,
                "m1_worst_bound": 0.0,
                "resets_total": 0,
                "offline_actions_total": 0,
                "frames_total": 0,
                "wall_s_total": 0.0,
                "games": [],
            },
        )
        acc["n_games"] += 1
        acc["games"].append(r.get("game"))
        acc["resets_total"] += int(r.get("n_resets") or 0)
        acc["offline_actions_total"] += int(r.get("actions") or 0)
        acc["frames_total"] += int(r.get("n_frames") or 0)
        acc["wall_s_total"] += float(r.get("wall_s") or 0.0)
        lv = int(r.get("levels") or 0)
        acc["levels_total"] += lv
        if lv > 0:
            acc["cells_with_at_least_one_levelup"] += 1
            spans, tail, baselines = _spans_from_row(r)
            if not baselines or not all(baselines):
                continue
            won = lv >= len(baselines)
            total_off = int(r.get("actions") or 0)
            n_res = int(r.get("n_resets") or 0)
            rec_rbl = r.get("resets_before_levelups")
            cum = (
                [int(x) for x in rec_rbl]
                if isinstance(rec_rbl, list) and len(rec_rbl) == lv
                else _uniform_rate_cum_resets(spans, total_off, n_res)
            )
            m0_cell = _score(baselines, spans, tail, won)
            # INDEPENDENT-PATH CHECK: the row's own `efficiency` was written by
            # arc_leaderboard_eval's `_calculate_score`, i.e. by the INSTALLED scorer at measurement
            # time. My re-derivation must reproduce it per cell, or my scorer drive is wrong and every
            # number in this part is wrong with it.
            rec_eff = r.get("efficiency")
            if rec_eff is not None:
                acc.setdefault("recorded_efficiency_sum", 0.0)
                acc["recorded_efficiency_sum"] += float(rec_eff)
                acc.setdefault("m0_vs_recorded_max_abs_diff", 0.0)
                acc["m0_vs_recorded_max_abs_diff"] = max(
                    acc["m0_vs_recorded_max_abs_diff"], abs(m0_cell - float(rec_eff))
                )
                acc.setdefault("n_cells_m0_matches_recorded", 0)
                # the row rounds `efficiency` to 4dp, so agreement is asserted at that tolerance
                if abs(m0_cell - float(rec_eff)) <= 5e-5:
                    acc["n_cells_m0_matches_recorded"] += 1
            acc["m0"] += m0_cell
            acc["m1_est"] += _score(baselines, _charged_spans(spans, cum, False), tail, won)
            acc["m2_est"] += _score(baselines, _charged_spans(spans, cum, True), tail, won)
            wsc, _ = gw.worst_case_allocation(baselines, spans, n_res, tail, game_won=won)
            acc["m1_worst_bound"] += float(wsc)

    by_budget: dict[str, Any] = {}
    for b in sorted({k[0] for k in per_bs}):
        arms = [per_bs[k] for k in per_bs if k[0] == b]
        by_budget[str(b)] = {
            "n_seeds": len(arms),
            "seeds": sorted(a["seed"] for a in arms),
            "n_games_per_seed": sorted({a["n_games"] for a in arms}),
            "cells_with_at_least_one_levelup_per_seed": {
                str(a["seed"]): a["cells_with_at_least_one_levelup"] for a in arms
            },
            "cells_with_at_least_one_levelup_median_per_seed": _median(
                [a["cells_with_at_least_one_levelup"] for a in arms]
            ),
            "levels_total_median": _median([a["levels_total"] for a in arms]),
            "resets_total_median": _median([a["resets_total"] for a in arms]),
            "offline_actions_total_median": _median([a["offline_actions_total"] for a in arms]),
            "frames_total_median": _median([a["frames_total"] for a in arms]),
            "llm_off_wall_s_total_median": _median([round(a["wall_s_total"], 1) for a in arms]),
            "score_M0_offline_per_seed": {str(a["seed"]): round(a["m0"], 4) for a in arms},
            "recorded_efficiency_sum_per_seed": {
                str(a["seed"]): round(a.get("recorded_efficiency_sum") or 0.0, 4) for a in arms
            },
            "m0_vs_recorded_max_abs_diff": max(
                (a.get("m0_vs_recorded_max_abs_diff") or 0.0) for a in arms
            ),
            "n_cells_m0_matches_recorded": sum(
                int(a.get("n_cells_m0_matches_recorded") or 0) for a in arms
            ),
            "n_levelup_cells_summed_across_seeds": sum(
                int(a["cells_with_at_least_one_levelup"]) for a in arms
            ),
            "score_M0_offline_median": _median([a["m0"] for a in arms]),
            "score_M1_all_resets_charged_median": _median([a["m1_est"] for a in arms]),
            "score_M2_bootstrap_free_median": _median([a["m2_est"] for a in arms]),
            "score_M1_worst_case_bound_median": _median([a["m1_worst_bound"] for a in arms]),
            # PER-SEED sums, published because the corpus score is DOMINATED by which seed you look
            # at (b400 spans 1.16 to 5.74 across three seeds -- a 5x range). A median or a sum across
            # seeds hides that; a matched per-seed delta is the only comparison that survives it.
            "score_M1_per_seed": {str(a["seed"]): round(a["m1_est"], 4) for a in arms},
            "score_M2_per_seed": {str(a["seed"]): round(a["m2_est"], 4) for a in arms},
            "M2_rel_optimism_of_M0_median": (
                round(
                    (
                        (_median([a["m0"] for a in arms]) - _median([a["m2_est"] for a in arms]))
                        / _median([a["m0"] for a in arms])
                    ),
                    4,
                )
                if _median([a["m0"] for a in arms])
                else None
            ),
        }
    # ---- MATCHED per-seed budget deltas (the only comparison that survives the seed spread) ----
    matched: dict[str, Any] = {}
    budgets_full = sorted(
        int(b) for b, v in by_budget.items() if (v.get("n_games_per_seed") or [None])[0] == 25
    )
    for i in range(len(budgets_full) - 1):
        lo, hi = budgets_full[i], budgets_full[i + 1]
        pairs = []
        for sd in sorted({k[1] for k in per_bs}):
            a, b_ = per_bs.get((lo, sd)), per_bs.get((hi, sd))
            if not a or not b_:
                continue
            pairs.append(
                {
                    "seed": sd,
                    "levelup_cells_lo": a["cells_with_at_least_one_levelup"],
                    "levelup_cells_hi": b_["cells_with_at_least_one_levelup"],
                    "d_levelup_cells": (
                        b_["cells_with_at_least_one_levelup"] - a["cells_with_at_least_one_levelup"]
                    ),
                    "M0_lo": round(a["m0"], 4),
                    "M0_hi": round(b_["m0"], 4),
                    "d_M0": round(b_["m0"] - a["m0"], 4),
                    "M2_lo": round(a["m2_est"], 4),
                    "M2_hi": round(b_["m2_est"], 4),
                    "d_M2_ESTIMATED": round(b_["m2_est"] - a["m2_est"], 4),
                    "resets_lo": a["resets_total"],
                    "resets_hi": b_["resets_total"],
                }
            )
        est_err = estimator_validation.get("abs_rel_error_median") or 0.0
        typical_scale = _mean([p["M2_lo"] for p in pairs]) or 0.0
        resolvable = (
            abs(_median([p["d_M2_ESTIMATED"] for p in pairs]) or 0.0) > est_err * typical_scale
        )
        matched[f"b{lo}_to_b{hi}"] = {
            "n_matched_seeds": len(pairs),
            "pairs": pairs,
            "d_levelup_cells_median": _median([p["d_levelup_cells"] for p in pairs]),
            "d_M0_median": _median([p["d_M0"] for p in pairs]),
            "d_M0_sign_test": both_tails_sign_test([p["d_M0"] for p in pairs]),
            "d_M2_ESTIMATED_median": _median([p["d_M2_ESTIMATED"] for p in pairs]),
            "d_M2_sign_test": both_tails_sign_test([p["d_M2_ESTIMATED"] for p in pairs]),
            "estimator_abs_rel_error_median": est_err,
            "typical_M2_scale": round(typical_scale, 4),
            "estimator_noise_floor_on_this_delta": round(est_err * typical_scale, 4),
            "d_M2_is_RESOLVABLE_above_the_estimator_error": bool(resolvable),
            "verdict": (
                "M2_DELTA_UNRESOLVABLE_SMALLER_THAN_THE_ESTIMATORS_OWN_ERROR"
                if not resolvable
                else "M2_DELTA_RESOLVABLE"
            ),
            "why_this_matters": (
                "The M0 (offline) delta is exact -- it is the installed scorer on recorded per-level "
                "actions. The M2 delta on the FULL corpus is an ESTIMATE, and if it is smaller than "
                "the estimator's own median error there is no sign to report. The exact-subset "
                "comparison below is the resolvable version."
            ),
        }

    # ---- EXACT matched budget delta on the exactly-attributed subset --------------------------
    exact_by_key: dict[tuple[str, int, int], dict] = {}
    for v in val_rows:
        exact_by_key[(str(v["game"]), int(v["seed"]), int(v["budget"]))] = v
    exact_budgets = sorted({k[2] for k in exact_by_key})
    exact_matched: dict[str, Any] = {}
    for i in range(len(exact_budgets) - 1):
        lo, hi = exact_budgets[i], exact_budgets[-1]
        rows_x = []
        for g, sd, b_ in list(exact_by_key):
            if b_ != lo:
                continue
            other = exact_by_key.get((g, sd, hi))
            if not other:
                continue
            a = exact_by_key[(g, sd, lo)]
            rows_x.append(
                {
                    "game": g,
                    "seed": sd,
                    "levels_lo": a["levels"],
                    "levels_hi": other["levels"],
                    "M0_lo": a["m0_offline_score"],
                    "M0_hi": other["m0_offline_score"],
                    "d_M0": round(other["m0_offline_score"] - a["m0_offline_score"], 6),
                    "M2_lo_EXACT": a["m2_exact_score"],
                    "M2_hi_EXACT": other["m2_exact_score"],
                    "d_M2_EXACT": round(other["m2_exact_score"] - a["m2_exact_score"], 6),
                    "resets_lo": a["n_resets"],
                    "resets_hi": other["n_resets"],
                }
            )
        if not rows_x:
            continue
        d_m2 = [r["d_M2_EXACT"] for r in rows_x]
        exact_matched[f"b{lo}_to_b{hi}_EXACT_SUBSET"] = {
            "n_matched_game_seed_cells": len(rows_x),
            "games": sorted({r["game"] for r in rows_x}),
            "cells": rows_x,
            "n_exact_ties": sum(1 for x in d_m2 if x == 0),
            "d_M2_EXACT_sum": round(sum(d_m2), 6),
            "d_M2_EXACT_median": _median(d_m2),
            "d_M2_EXACT_sign_test": both_tails_sign_test(d_m2),
            "d_M0_sum": round(sum(r["d_M0"] for r in rows_x), 6),
            "reading": (
                "EXACT, not estimated: both budgets' cells carry per-level reset attribution, so the "
                "gateway-charged score is computed rather than inferred. Ties are cells whose level "
                "reaching did not change with the budget -- structurally frozen, excluded from the "
                "sign test's support, and counted here so the support is not mistaken for the sample."
            ),
        }

    return {
        "channel_checks": channel,
        "row_files": files_meta,
        "estimator_validation": estimator_validation,
        "per_budget": by_budget,
        "matched_per_seed_budget_deltas": matched,
        "exact_subset_matched_budget_delta": exact_matched,
        "per_budget_seed_cells": [per_bs[k] for k in sorted(per_bs)],
        "unit_of_every_score_here": "GATEWAY-CHARGED for M1/M2; OFFLINE ACTIONS for M0",
        # UNIT OF EVERY *COUNT* (2026-07-27). Added after a review found the old `wins` accumulator
        # was read as "games won" when its predicate is `levels > 0`. Two DIFFERENT predicates
        # coexist in this analyser and must never be conflated.
        "unit_of_every_count_here": {
            "cells_with_at_least_one_levelup": {
                "predicate": "levels > 0",
                "meaning": (
                    "a (game, seed, budget) CELL that banked at least ONE level-up. NOT a game won, "
                    "NOT a level count."
                ),
                "aggregation": "PER SEED unless the key carries the _summed_across_seeds suffix",
            },
            "won": {
                "predicate": "levels >= len(baselines)",
                "meaning": "the game-COMPLETE predicate; used ONLY to decide scorer completion flags",
                "note": "never emitted as a count; it is a per-cell boolean inside the scorer drive",
            },
            "n_levelup_cells_summed_across_seeds": {
                "predicate": "levels > 0",
                "aggregation": "SUMMED ACROSS SEEDS -- not comparable to the per-seed medians above",
            },
        },
        "scope": (
            "grace=None (SHIPPED) arm only, LLM-OFF cells. b400/b2000 are the full 25-game corpus x "
            "3 seeds; b4000 is a 13-game LEVEL-REACHING SUBSET and is therefore biased UPWARD on "
            "level-up cells and NOT comparable to the 25-game rows without saying so."
        ),
    }


# ============================================ PART 3: THE EARLY-STOP LEVER, PRICED ON THE BENEFIT
def part3_early_stop_ceiling(sweep_files: list[str]) -> dict[str, Any]:
    """How much wall clock is spent where NO score can be earned, and what window would keep it.

    Two numbers, both computed:
      * ORACLE CEILING -- the share of offline actions spent after the LAST level-up (plus ALL
        actions in cells that never level up). By the settled charge model an incomplete level scores
        0.0 whatever it is charged, so cutting that share costs EXACTLY ZERO score. It is an ORACLE
        (a real rule cannot know a level-up will never come), hence a ceiling, not a forecast.
      * ADAPTIVE MULTIPLIER -- the smallest c such that a window of c x (the game's own FIRST
        level-up cost) is longer than every SUBSEQUENT inter-level-up gap actually observed. That is
        the design constant an adaptive rule needs, and it is measured off the corpus rather than
        guessed.
    """
    out: dict[str, Any] = {"per_budget": {}, "adaptive_window": {}}
    gaps_all: list[dict] = []
    for p in sweep_files:
        d = _load_json(p)
        for r in d.get("rows") or []:
            if r.get("early_stop_grace") is not None:
                continue
            b = str(int(r.get("budget") or 0))
            acc = out["per_budget"].setdefault(
                b,
                {
                    "n_cells": 0,
                    "n_cells_zero_levels": 0,
                    "offline_actions_total": 0,
                    "score_free_offline_actions_total": 0,
                    "frames_total": 0,
                    "score_free_frames_total": 0,
                    "wall_s_total": 0.0,
                    "score_free_wall_s_est_total": 0.0,
                },
            )
            acts = int(r.get("actions") or 0)
            frames = int(r.get("n_frames") or 0)
            wall = float(r.get("wall_s") or 0.0)
            lua = [int(x) for x in (r.get("level_up_actions") or [])]
            tail = acts - (lua[-1] if lua else 0)
            acc["n_cells"] += 1
            acc["offline_actions_total"] += acts
            acc["frames_total"] += frames
            acc["wall_s_total"] += wall
            if not lua:
                acc["n_cells_zero_levels"] += 1
                acc["score_free_offline_actions_total"] += acts
                acc["score_free_frames_total"] += frames
                acc["score_free_wall_s_est_total"] += wall
            else:
                acc["score_free_offline_actions_total"] += tail
                acc["score_free_frames_total"] += int(round(frames * (tail / acts))) if acts else 0
                acc["score_free_wall_s_est_total"] += wall * (tail / acts) if acts else 0.0
            if len(lua) >= 2:
                first = lua[0]
                for i in range(1, len(lua)):
                    gaps_all.append(
                        {
                            "game": r.get("game"),
                            "seed": r.get("seed"),
                            "budget": int(r.get("budget") or 0),
                            "level_index": i,
                            "first_levelup_cost": first,
                            "gap": lua[i] - lua[i - 1],
                            "gap_over_first": round((lua[i] - lua[i - 1]) / first, 3)
                            if first
                            else None,
                        }
                    )
    for b, acc in out["per_budget"].items():
        acc["score_free_share_of_offline_actions"] = (
            round(acc["score_free_offline_actions_total"] / acc["offline_actions_total"], 4)
            if acc["offline_actions_total"]
            else None
        )
        acc["score_free_share_of_frames"] = (
            round(acc["score_free_frames_total"] / acc["frames_total"], 4)
            if acc["frames_total"]
            else None
        )
        acc["score_free_share_of_llm_off_wall"] = (
            round(acc["score_free_wall_s_est_total"] / acc["wall_s_total"], 4)
            if acc["wall_s_total"]
            else None
        )
        acc["wall_s_total"] = round(acc["wall_s_total"], 1)
        acc["score_free_wall_s_est_total"] = round(acc["score_free_wall_s_est_total"], 1)
    ratios = [g["gap_over_first"] for g in gaps_all if g["gap_over_first"] is not None]

    # ---- THE TWO WINDOWS, AS TRADE CURVES ----------------------------------------------------
    # A single "safe multiplier" is not a design answer, because the two buckets of score-free work
    # need DIFFERENT rules and each has its own safety cost:
    #   (A) START window -- cells that NEVER level up (the majority of the waste). A rule can only cut
    #       them by giving up after W actions with no level-up at all. Its cost is measured by the
    #       distribution of FIRST level-up costs: any cell whose first level-up lands beyond W is lost.
    #   (B) CONTINUATION window -- cells that levelled up and then kept spending. A rule scaled to the
    #       game's own first level-up cost (c x first) is cut short by the LONG TAIL of inter-level-up
    #       gaps, which is what the max-c number hides.
    first_costs: list[dict] = []
    per_cell_rows: list[dict] = []
    for p in sweep_files:
        d = _load_json(p)
        for r in d.get("rows") or []:
            if r.get("early_stop_grace") is not None:
                continue
            lua = [int(x) for x in (r.get("level_up_actions") or [])]
            per_cell_rows.append(
                {
                    "budget": int(r.get("budget") or 0),
                    "game": r.get("game"),
                    "seed": r.get("seed"),
                    "actions": int(r.get("actions") or 0),
                    "level_up_actions": lua,
                }
            )
            if lua:
                first_costs.append(
                    {
                        "budget": int(r.get("budget") or 0),
                        "game": r.get("game"),
                        "seed": r.get("seed"),
                        "first_levelup_actions": lua[0],
                    }
                )
    start_curve = {}
    for b in sorted({c["budget"] for c in per_cell_rows}):
        rows_b = [c for c in per_cell_rows if c["budget"] == b]
        firsts = [c["first_levelup_actions"] for c in first_costs if c["budget"] == b]
        total_actions = sum(c["actions"] for c in rows_b)
        pts = []
        for W in (100, 200, 400, 800, 1300, 2000, 4000):
            if W > b:
                continue
            # A start-window stops EVERY cell that has not levelled up by W -- which includes the
            # cells whose first level-up would have landed LATER. Those contribute BOTH a saving and a
            # lost win, and counting only the never-levelling cells' saving (an earlier version of
            # this line) understates the benefit while still charging the full cost.
            cut = [c for c in rows_b if (not c["level_up_actions"]) or c["level_up_actions"][0] > W]
            saved = sum(max(0, c["actions"] - W) for c in cut)
            lost = sum(1 for f in firsts if f > W)
            pts.append(
                {
                    "W_actions": W,
                    "actions_saved_on_cells_cut_at_W": saved,
                    "n_cells_cut_at_W": len(cut),
                    "share_of_all_offline_actions_saved": (
                        round(saved / total_actions, 4) if total_actions else None
                    ),
                    "first_levelups_that_would_be_LOST": lost,
                    "first_levelups_observed": len(firsts),
                    "share_of_first_levelups_lost": round(lost / len(firsts), 4)
                    if firsts
                    else None,
                }
            )
        start_curve[str(b)] = {
            "n_cells": len(rows_b),
            "n_cells_never_levelling": sum(1 for c in rows_b if not c["level_up_actions"]),
            "first_levelup_actions_median": _median(firsts),
            "first_levelup_actions_max": max(firsts) if firsts else None,
            "trade_points": pts,
            "reading": (
                "A start-window W stops a game that has not levelled up within W actions. Saving comes "
                "from the never-levelling cells; the cost is the cells whose FIRST level-up lands beyond "
                "W. Both are counted from the same rows, so the trade is measured, not modelled."
            ),
        }
    cont_curve = []
    for c_mult in (1, 2, 3, 5, 8, 13, 22):
        lost = sum(1 for g in gaps_all if (g["gap_over_first"] or 0) > c_mult)
        cont_curve.append(
            {
                "c_multiplier_of_first_levelup_cost": c_mult,
                "subsequent_levelups_that_would_be_LOST": lost,
                "subsequent_levelups_observed": len(gaps_all),
                "share_lost": round(lost / len(gaps_all), 4) if gaps_all else None,
            }
        )
    out["start_window_trade_curve"] = start_curve
    out["continuation_window_trade_curve"] = cont_curve
    out["adaptive_window"] = {
        "n_subsequent_levelups_observed": len(gaps_all),
        "gap_over_first_levelup_cost_median": _median(ratios),
        "gap_over_first_levelup_cost_max": (max(ratios) if ratios else None),
        "smallest_c_that_preserves_every_observed_levelup": (
            round(max(ratios), 3) if ratios else None
        ),
        "gaps": gaps_all,
        "caveat": (
            "This c is the smallest multiplier that would not have cut any level-up THIS CORPUS "
            "produced. A deeper level in an unseen game can have a longer gap, so c is a lower "
            "bound on a safe design constant, not a guarantee -- and the corpus is the 25 PUBLIC "
            "games, which are not the hidden set."
        ),
    }
    return out


# ================================================================== PART 4: THE ANSWER + GATES
def _bootstrap_ci(values: Sequence[float], n_boot: int = 20000, seed: int = 20260724) -> dict:
    """Percentile bootstrap over the measured (game, seed) CELLS.

    Reported because a 4-game mean has real uncertainty and quoting it as a point estimate is how a
    per-game cost becomes a false precision at 110 games.
    """
    import random

    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(st.fmean(rng.choices(vals, k=len(vals))))
    means.sort()
    return {
        "mean": round(st.fmean(vals), 2),
        "lo": round(means[int(0.025 * (n_boot - 1))], 2),
        "hi": round(means[int(0.975 * (n_boot - 1))], 2),
        "n": len(vals),
        "note": (
            "percentile bootstrap over the measured (game, seed) CELLS, 20k resamples. Not over games "
            "alone: with more than one seed the resampling unit is the cell, and saying 'over games' "
            "would overstate how independent the units are."
        ),
    }


def _walk_p_floors(node) -> list:
    """Collect every emitted POSITIVE `min_reachable_two_sided_p_at_this_support` in a subtree.

    Non-positive values are excluded deliberately. A floor of 0.0 is not a reachable p -- it is what a
    sign test emits when its support is EMPTY (2/2**0 style degeneracies and empty-list guards). Left
    in, a single degenerate 0.0 anywhere in the artifact drags the minimum to zero and makes the
    self-consistency check unfailable-in-one-direction and permanently false in the other.
    """
    out: list = []
    if isinstance(node, dict):
        v = node.get("min_reachable_two_sided_p_at_this_support")
        if isinstance(v, (int, float)) and float(v) > 0.0:
            out.append(float(v))
        for x in node.values():
            out.extend(_walk_p_floors(x))
    elif isinstance(node, list):
        for x in node:
            out.extend(_walk_p_floors(x))
    return out


def _leaf_floor_values(node) -> list:
    """Every `min_reachable_two_sided_p_at_this_support` value INCLUDING non-positive ones."""
    out: list = []
    if isinstance(node, dict):
        v = node.get("min_reachable_two_sided_p_at_this_support")
        if isinstance(v, (int, float)):
            out.append(float(v))
        for x in node.values():
            out.extend(_leaf_floor_values(x))
    elif isinstance(node, list):
        for x in node:
            out.extend(_leaf_floor_values(x))
    return out


def _min_emitted_p_floor(node) -> float | None:
    floors = _walk_p_floors(node)
    return round(min(floors), 4) if floors else None


def _throughput_power_stamp(p1: dict) -> dict:
    """The THROUGHPUT arm's power, extracted so it can be stamped wherever it is relied on.

    WHY (2026-07-27). The latency-vs-throughput decomposition is what VOIDS the prior cost model,
    and every downstream budget number inherits it. But the throughput arm has 2 usable seeds
    (one of which matched only 2 games), a two-sided sign-test p-floor of 0.5, and therefore
    `can_ever_reach_0_05: false`. The LATENCY arm is the powered one (p=0.0312) and it measures the
    quantity the cap argument declares irrelevant. Reading the artifact previously required finding
    the nested test to learn this; the stamp now travels with the claim.
    """
    mr = (p1.get("matched_set_throughput_ratio_by_K") or {}).get("2") or {}
    test = mr.get("sign_test_ratio_gt_1") or {}
    per_seed = mr.get("per_seed") or {}
    return {
        "arm": "matched_set_throughput_ratio_K2_vs_K1",
        "n_seeds_usable": mr.get("n_seeds_usable"),
        "games_matched_per_seed": {k: (v.get("games_matched") or []) for k, v in per_seed.items()},
        "min_games_matched_on_any_usable_seed": min(
            [
                len(v.get("games_matched") or [])
                for v in per_seed.values()
                if v.get("ratio_KN_over_K1")
            ]
            or [0]
        ),
        "p_floor": mr.get("p_floor_for_a_test_at_this_seed_support"),
        "can_ever_reach_0_05": test.get("can_ever_reach_0_05"),
        "verdict": test.get("verdict"),
        "STAMP": (
            "UNDERPOWERED__the_latency_vs_throughput_decomposition_that_voids_the_prior_cost_model_"
            "rests_on_this_arm_and_this_arm_cannot_reach_p_0.05_at_its_seed_support"
        )
        if test.get("can_ever_reach_0_05") is False
        else "POWERED",
        "what_this_does_NOT_invalidate": (
            "the measured NUMBERS (K=1 and K=2 s/game) are real; what is underpowered is the "
            "significance of the RATIO being at/below 1.0. A reader should treat 'throughput is flat' "
            "as a 2-seed observation consistent with a saturated GPU, not as a powered result."
        ),
    }


def part4_answer(
    p1: dict, p2: dict, p3: dict, sibling_envelope: dict, n_games_list: list[int]
) -> dict:
    """Compose the feasible-budget answer, with the uncertainty attached."""
    thr = p1.get("throughput_by_K") or {}
    K1 = thr.get("1") or {}
    per_cell = p1.get("per_cell") or []
    k1_walls = [
        c["wall_s"] for c in per_cell if c["K"] == 1 and c.get("wall_s") and c.get("row_valid")
    ]
    k1_ci = _bootstrap_ci(k1_walls)

    # THROUGHPUT RATIO: does running K at once change games-per-hour? This -- not per-game latency --
    # is what a fixed cap divides. A ratio of ~1.0 with a saturated GPU means concurrency neither
    # helps nor hurts TOTAL time, and that scaling an uncontended latency by a contention factor
    # would double-count.
    ratios_by_K = {
        K: v.get("throughput_s_per_game_ratio_vs_K1")
        for K, v in thr.items()
        if K != "1" and v.get("is_a_usable_throughput_point")
    }
    measured = [r for r in ratios_by_K.values() if r]
    best_ratio = min(measured) if measured else 1.0  # most favourable concurrency measured
    worst_ratio = max(measured) if measured else 1.0  # least favourable concurrency measured
    # LEVEL-C FALSIFIABILITY (2026-07-27). `worst_ratio` is a max over `measured`, and `measured`
    # has exactly ONE usable element (K=2; K=4 died and is `is_a_usable_throughput_point: False`).
    # So min == max == that single value, and cost level C (CI-high x WORST measured concurrency) is
    # BIT-IDENTICAL to level B (CI-high x BEST measured concurrency) BY CONSTRUCTION. The prior
    # headline "survives all three cost levels" therefore counted two distinct levels as three. Per
    # this project's own uninterpretable-gate rule, that is stamped rather than quietly counted.
    n_distinct_ratios = len({round(float(r), 6) for r in measured})
    level_c_is_distinct = n_distinct_ratios >= 2
    level_c_stamp = (
        None if level_c_is_distinct else "UNFALSIFIABLE_AS_CONSTRUCTED__single_element_ratio_set"
    )
    # The ONE measured adverse-concurrency point that exists anywhere in this ladder: K=4 under the
    # BIGGER server context (n_ctx=32768), s/game 116.2 against the K=1 SAME-SEED comparator 100.7.
    # It is CONFOUNDED (there is no K=1 run at n_ctx=32768, so context and concurrency move
    # together) and is therefore NOT promoted to a cost level -- it is published as a SENSITIVITY
    # row so a reader can see where the answer breaks, instead of inferring from a level that could
    # not differ.
    cfg = p1.get("config_test_does_a_bigger_server_context_survive_the_same_concurrency") or {}
    adverse_sep = (cfg.get("throughput_under_the_bigger_context_SEPARATE_CONFIG") or {}).get(
        "s_per_game"
    )
    k1_by_seed = K1.get("s_per_game_throughput_per_seed") or {}
    adverse_seeds = sorted({str(c.get("seed")) for c in (cfg.get("cells") or [])})
    k1_same_seed = None
    if len(adverse_seeds) == 1:
        k1_same_seed = k1_by_seed.get(adverse_seeds[0])
    adverse_ratio = (
        round(float(adverse_sep) / float(k1_same_seed), 4) if adverse_sep and k1_same_seed else None
    )

    budget_ratio: dict[str, float] = {}
    for b, v in (sibling_envelope.get("budget_curve") or {}).items():
        r = v.get("paired_ratio_vs_b400_median")
        if r:
            budget_ratio[b] = float(r)

    # EXTRAPOLATED budgets above the measured grid. Marked as extrapolation on every row, and needed
    # for a reason of measurement hygiene rather than ambition: if every budget in the table fits, the
    # feasibility answer is produced by construction and carries no information (gate G6). The fit is
    # log-linear in budget because that is what the three MEASURED points look like -- the LLM cost
    # does not scale with the action budget, it scales with how many INDUCTIONS the budget buys.
    measured_budgets = sorted(int(b) for b in budget_ratio)
    extrapolated: dict[str, dict] = {}
    if len(measured_budgets) >= 2:
        xs = [math.log(b) for b in measured_budgets]
        ys = [budget_ratio[str(b)] for b in measured_budgets]
        xm, ym = st.fmean(xs), st.fmean(ys)
        denom = sum((x - xm) ** 2 for x in xs)
        slope = (sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denom) if denom else 0.0
        inter = ym - slope * xm
        ss_res = sum((y - (inter + slope * x)) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - ym) ** 2 for y in ys)
        for b in (4000, 8000):
            budget_ratio[str(b)] = round(inter + slope * math.log(b), 3)
            extrapolated[str(b)] = {"extrapolated": True}
        extrapolated["_fit"] = {
            "form": "ratio_vs_b400 = intercept + slope * ln(budget)",
            "slope": round(slope, 4),
            "intercept": round(inter, 4),
            "r2": round(1 - ss_res / ss_tot, 4) if ss_tot else None,
            "fitted_on_measured_budgets": measured_budgets,
            "caveat": (
                "budgets above the measured grid are EXTRAPOLATION from three points with an "
                "UNDERPOWERED underlying test. They exist so the cap table can contain a non-fit; a "
                "verdict must not rest on an extrapolated row alone."
            ),
        }

    # THREE cost levels. A deliberately-omitted fourth: an earlier draft multiplied the CI-high by the
    # published same-config fold-change MAX (2.468x). That is wrong statistics -- a per-CELL noise
    # statistic does not multiply the uncertainty of a 110-game MEAN, which the bootstrap CI already
    # covers. The honest replacement for "how bad could it be" is the critical-N table below: the
    # number of games at which each budget stops fitting.
    levels = {
        "A_central_best_measured_throughput": (k1_ci["mean"] or 0.0) * best_ratio,
        "B_ci_hi_best_measured_throughput": (k1_ci["hi"] or 0.0) * best_ratio,
        "C_ci_hi_worst_measured_throughput": (k1_ci["hi"] or 0.0) * worst_ratio,
    }
    n_distinct_cost_levels = len({round(float(v), 6) for v in levels.values()})

    rows = []
    for cap_name, cap_s in CAPS.items():
        for n_games in n_games_list:
            usable = cap_s - KERNEL_OVERHEAD_S_ASSUMED
            for b, ratio in sorted(budget_ratio.items(), key=lambda kv: int(kv[0])):
                if ratio is None:
                    continue
                row = {
                    "cap": cap_name,
                    "cap_s": cap_s,
                    "n_games": n_games,
                    "budget": int(b),
                    "budget_scaling_ratio_vs_b400": ratio,
                    "budget_ratio_is_EXTRAPOLATED": bool(str(b) in extrapolated),
                    "usable_loop_s": round(usable, 1),
                }
                for lname, per_game_b400 in levels.items():
                    total = per_game_b400 * ratio * n_games
                    row[f"total_s__{lname}"] = round(total, 1)
                    row[f"fraction_of_usable__{lname}"] = (
                        round(total / usable, 3) if usable > 0 else None
                    )
                    row[f"fits_with_margin__{lname}"] = bool(total <= MARGIN_FRACTION * usable)
                rows.append(row)

    # ---- SENSITIVITY (2026-07-27): the two uncertainties the cap table treated as EXACT --------
    # (i) the CONCURRENCY ratio. The ladder's cost levels cover no ratio >= 1.0 (see
    #     `cost_level_falsifiability`). Rows at ratio 1.0 (the K=1 arm, exact by definition) and at
    #     the confounded adverse 1.15x are published so the reader sees the boundary.
    # (ii) the BUDGET-SCALING ratio. The cap table multiplies the bootstrapped per-game wall CI by
    #     `budget_ratio` treated as a POINT value, although it is a median over 4 games with a real
    #     spread (b2000: 1.044 / 1.611 / 1.680 / 1.827). Propagating only one of two multiplicands'
    #     uncertainty understates the interval. Both are now bootstrapped and the FITTING BOUNDARY
    #     is published, so "fits with margin" is a statement with a stated breaking point.
    SENS_CAP = "kaggle_9h_max_notebook_runtime"
    SENS_N = 110
    sens_usable = CAPS[SENS_CAP] - KERNEL_OVERHEAD_S_ASSUMED
    adverse_rows = []
    for label, ratio_k in (
        ("K1_arm_ratio_1.0_EXACT_by_definition", 1.0),
        ("adverse_K4_bigger_context_CONFOUNDED", adverse_ratio),
    ):
        if ratio_k is None:
            continue
        for b, br in sorted(budget_ratio.items(), key=lambda kv: int(kv[0])):
            if br is None:
                continue
            total = (k1_ci["hi"] or 0.0) * ratio_k * br * SENS_N
            adverse_rows.append(
                {
                    "concurrency_ratio_label": label,
                    "concurrency_ratio": ratio_k,
                    "cap": SENS_CAP,
                    "n_games": SENS_N,
                    "budget": int(b),
                    "budget_scaling_ratio_vs_b400": br,
                    "budget_ratio_is_EXTRAPOLATED": bool(str(b) in extrapolated),
                    "total_s_at_ci_hi": round(total, 1),
                    "fraction_of_usable": round(total / sens_usable, 4)
                    if sens_usable > 0
                    else None,
                    "fits_with_margin": bool(total <= MARGIN_FRACTION * sens_usable),
                }
            )
    # budget-scaling-ratio uncertainty, bootstrapped over the per-GAME paired ratios the sibling
    # envelope publishes (4 games per budget). Emits lo/median/hi cap rows plus the ratio at which
    # each budget crosses the 0.80 margin at cost level B.
    ratio_uncertainty: dict[str, Any] = {}
    for b, v in (sibling_envelope.get("budget_curve") or {}).items():
        vals = [float(x) for x in (v.get("paired_ratio_vs_b400_values") or []) if x]
        if len(vals) < 2:
            continue
        boot = _bootstrap_ci(vals, seed=int(b))
        per_game_at_b400 = (k1_ci["hi"] or 0.0) * best_ratio
        denom = per_game_at_b400 * SENS_N
        crossing = (MARGIN_FRACTION * sens_usable / denom) if denom else None
        ratio_uncertainty[str(b)] = {
            "n_games_behind_the_ratio": len(vals),
            "observed_values": sorted(vals),
            "median": round(st.median(vals), 4),
            "bootstrap_mean_ci": boot,
            "observed_min": round(min(vals), 4),
            "observed_max": round(max(vals), 4),
            "cap_rows_at_level_B": {
                lbl: {
                    "ratio": round(float(rv), 4),
                    "total_s": round(per_game_at_b400 * float(rv) * SENS_N, 1),
                    "fraction_of_usable": (
                        round(per_game_at_b400 * float(rv) * SENS_N / sens_usable, 4)
                        if sens_usable > 0
                        else None
                    ),
                    "fits_with_margin": bool(
                        per_game_at_b400 * float(rv) * SENS_N <= MARGIN_FRACTION * sens_usable
                    ),
                }
                for lbl, rv in (
                    ("ratio_lo_bootstrap", boot.get("lo")),
                    ("ratio_median", st.median(vals)),
                    ("ratio_hi_bootstrap", boot.get("hi")),
                    ("ratio_observed_max", max(vals)),
                )
                if rv
            },
            "margin_crossing_ratio_at_level_B": round(crossing, 4) if crossing else None,
            "reading": (
                "the budget-scaling ratio is a 4-game median with a real spread; at its observed "
                "MAXIMUM this budget's fit can flip. `margin_crossing_ratio_at_level_B` is the ratio "
                "at which it does."
            ),
        }

    feasible: dict[str, Any] = {}
    for cap_name in CAPS:
        for n_games in n_games_list:
            key = f"{cap_name}|n_games={n_games}"
            entry = {}
            for lname in levels:
                sel = [
                    r
                    for r in rows
                    if r["cap"] == cap_name
                    and r["n_games"] == n_games
                    and r[f"fits_with_margin__{lname}"]
                ]
                entry[lname] = max((r["budget"] for r in sel), default=None)
            level_answers = [entry[lname] for lname in levels]
            entry["ANSWER_central"] = entry["A_central_best_measured_throughput"]
            # The conservative answer is the budget that fits under EVERY cost level. If any level
            # admits nothing, the conservative answer is None -- "no measured budget survives the
            # harshest reading" -- and must NOT be silently reported as the smallest budget.
            entry["ANSWER_conservative_all_levels"] = (
                None if any(v is None for v in level_answers) else min(level_answers)
            )
            entry["ANSWER_central_is_EXTRAPOLATED"] = bool(
                entry["ANSWER_central"] is not None and str(entry["ANSWER_central"]) in extrapolated
            )
            entry["largest_MEASURED_budget_fitting_central"] = max(
                (
                    r["budget"]
                    for r in rows
                    if r["cap"] == cap_name
                    and r["n_games"] == n_games
                    and not r["budget_ratio_is_EXTRAPOLATED"]
                    and r["fits_with_margin__A_central_best_measured_throughput"]
                ),
                default=None,
            )
            feasible[key] = entry

    # ---- CRITICAL N: how many games each budget can carry before the cap binds ----------------
    # This replaces the arbitrary "how bad could it be" multiplier with a number the reader can
    # compare against whatever the hidden set size turns out to be, since that size is NOT known.
    critical_n: dict[str, Any] = {}
    for cap_name, cap_s in CAPS.items():
        usable = cap_s - KERNEL_OVERHEAD_S_ASSUMED
        for b, ratio in sorted(budget_ratio.items(), key=lambda kv: int(kv[0])):
            per_level = {}
            for lname, per_game_b400 in levels.items():
                per_game = per_game_b400 * ratio
                per_level[lname] = (
                    int((MARGIN_FRACTION * usable) // per_game) if per_game > 0 else None
                )
            critical_n[f"{cap_name}|budget={b}"] = {
                "max_games_fitting_with_margin": per_level,
                "budget_ratio_is_EXTRAPOLATED": bool(str(b) in extrapolated),
            }

    # ---- what the budget BUYS, in LEVEL-UP CELLS and in the paid unit ------------------------
    pb = p2.get("per_budget") or {}
    buys = {}
    for b, v in pb.items():
        buys[b] = {
            "cells_with_at_least_one_levelup_median_per_seed": v.get(
                "cells_with_at_least_one_levelup_median_per_seed"
            ),
            "cells_with_at_least_one_levelup_per_seed": v.get(
                "cells_with_at_least_one_levelup_per_seed"
            ),
            "n_games_per_seed": v.get("n_games_per_seed"),
            "score_M0_offline_median": (
                round(v["score_M0_offline_median"], 4) if v.get("score_M0_offline_median") else None
            ),
            "score_M1_all_resets_charged_median": (
                round(v["score_M1_all_resets_charged_median"], 4)
                if v.get("score_M1_all_resets_charged_median")
                else None
            ),
            "score_M2_bootstrap_free_median_THE_PAID_UNIT": (
                round(v["score_M2_bootstrap_free_median"], 4)
                if v.get("score_M2_bootstrap_free_median")
                else None
            ),
            "score_M1_worst_case_bound_median": (
                round(v["score_M1_worst_case_bound_median"], 4)
                if v.get("score_M1_worst_case_bound_median")
                else None
            ),
            "max_possible_score_for_this_game_count": (
                100 * (v.get("n_games_per_seed") or [0])[0] if v.get("n_games_per_seed") else None
            ),
            "resets_total_median": v.get("resets_total_median"),
        }
    # THE SCORE-MAXIMISING BUDGET, decided from MATCHED per-seed deltas rather than from a median of
    # per-seed sums. The corpus score spans 1.16-5.74 across three seeds at a single budget, so an
    # across-seed median is dominated by which seed happens to be central; only the matched delta is
    # interpretable. Two separate answers are given because the two units genuinely differ:
    #   * M0 (OFFLINE actions) -- exact, from the installed scorer on recorded per-level actions.
    #   * M2 (GATEWAY-CHARGED) -- the paid unit; estimated corpus-wide, EXACT on the 44-cell subset.
    md = p2.get("matched_per_seed_budget_deltas") or {}
    exm = p2.get("exact_subset_matched_budget_delta") or {}
    m0_direction = {k: (v.get("d_M0_median"), v.get("d_M0_sign_test", {})) for k, v in md.items()}
    score_max_M0 = None
    comparable_budgets = sorted(
        int(b) for b, v in buys.items() if (v.get("n_games_per_seed") or [None])[0] == 25
    )
    if comparable_budgets:
        score_max_M0 = comparable_budgets[0]
        for i in range(len(comparable_budgets) - 1):
            k = f"b{comparable_budgets[i]}_to_b{comparable_budgets[i + 1]}"
            d = (md.get(k) or {}).get("d_M0_median")
            if d is not None and d > 0:
                score_max_M0 = comparable_budgets[i + 1]
    m2_resolvable = all(v.get("d_M2_is_RESOLVABLE_above_the_estimator_error") for v in md.values())
    exact_key = next(iter(exm), None)
    exact_block = exm.get(exact_key) if exact_key else None
    score_max_M2_verdict = (
        "UNRESOLVABLE_ON_THE_FULL_CORPUS__the_M2_delta_between_budgets_is_smaller_than_the_"
        "estimators_own_median_error"
        if not m2_resolvable
        else "RESOLVABLE"
    )
    # ---- TEST THE PRIOR FITTED COST MODEL, at the LEVEL (its MECHANISM was already falsified) ----
    # The task this file answers requires testing the previously fitted LLM-ON cost model against what
    # was measured, and voiding its extrapolated crossing point if it does not hold. Two separate
    # questions, both answered:
    #   MECHANISM -- already falsified by the sibling artifact's own test (LLM cost grows at CONSTANT
    #     induction count, while the model attributes all budget-scaling to induction COUNT).
    #   LEVEL -- tested here. The prior model projects a per-game cost measured under 3-way process
    #     CONTENTION and then multiplies by the game count. This file's measurement says total
    #     THROUGHPUT is what a cap divides and that it does not degrade with concurrency, so the prior
    #     projections should be systematically HIGH by roughly the contention factor they embed.
    prior_proj = {
        str(k): v
        for k, v in (
            (sibling_envelope.get("headline") or {}).get("prior_model_projection_for_comparison")
            or {}
        ).items()
    }
    prior_test = {
        "prior_model_projections_s_per_game": prior_proj,
        "my_throughput_based_cost_s_per_game": {
            b: round((levels["A_central_best_measured_throughput"]) * r, 1)
            for b, r in sorted(budget_ratio.items(), key=lambda kv: int(kv[0]))
        },
        "ratio_prior_over_mine": {
            b: round(
                float(prior_proj[b])
                / (levels["A_central_best_measured_throughput"] * budget_ratio[b]),
                3,
            )
            for b in prior_proj
            if b in budget_ratio and levels["A_central_best_measured_throughput"]
        },
        "mechanism_verdict_from_the_sibling_artifact": (
            (sibling_envelope.get("cost_model_mechanism_test") or {}).get("verdict")
        ),
        "level_verdict": (
            "PRIOR_MODEL_IS_SYSTEMATICALLY_HIGH_AT_THE_LEVEL__it_projects_a_CONTENDED_per_game_LATENCY "
            "and then multiplies by the game count, which double-counts once throughput is measured to "
            "be flat under concurrency __ BUT_THE_THROUGHPUT_ARM_IS_UNDERPOWERED (see "
            "`throughput_arm_power_stamp` beside this field)"
        ),
        # POWER STAMP CARRIED INLINE (2026-07-27). The whole reversal of the prior cost model rests
        # on "a wall-clock CAP divides THROUGHPUT, not latency". The THROUGHPUT arm is the
        # UNDERPOWERED one (2 usable seeds, p-floor 0.5, `can_ever_reach_0_05: false`), while the arm
        # that IS adequately powered (LATENCY, p=0.0312) measures the quantity declared irrelevant to
        # the cap. A review found the stamp existed but only in the nested test, so the conclusion
        # built on it read as powered. It is now attached wherever the decomposition is asserted.
        "throughput_arm_power_stamp": _throughput_power_stamp(p1),
        "consequence_for_the_extrapolated_crossing_point": (
            "The prior model's implied crossing point (where a budget stops fitting) is VOIDED in both "
            "directions: its mechanism is falsified and its level is ~2x high. It is replaced by the "
            "critical-N table in this artifact, which is derived from measured throughput and carries "
            "the bootstrap CI as its uncertainty."
        ),
        "what_this_test_does_NOT_show": (
            "It does not show the prior model was WRONG about its own measurement -- its b400 anchor was "
            "measured under real 3-way contention and this file's K=1 arm reproduces the published "
            "UNCONTENDED anchor within its noise floor. The disagreement is about which quantity a cap "
            "divides, not about either measurement's honesty."
        ),
    }
    return {
        "prior_cost_model_test": prior_test,
        "cost_levels_s_per_game_at_b400": {k: round(v, 1) for k, v in levels.items()},
        # FALSIFIABILITY OF THE COST LADDER (2026-07-27). Emitted because a review found level C is
        # identical to level B by construction: `worst_ratio` is a max over a ONE-ELEMENT set, so
        # min == max. Two distinct levels were being counted as three in the headline.
        "cost_level_falsifiability": {
            "n_usable_throughput_ratios_measured": len(measured),
            "n_distinct_throughput_ratios": n_distinct_ratios,
            "n_DISTINCT_cost_levels": n_distinct_cost_levels,
            "n_cost_levels_emitted": len(levels),
            "level_C_is_distinct_from_level_B": bool(level_c_is_distinct),
            "level_C_stamp": level_c_stamp,
            "why": (
                "level C is CI-high x the WORST measured throughput ratio and level B is CI-high x "
                "the BEST. `measured` holds exactly one usable ratio (K=2 = "
                f"{measured[0] if measured else None}); K=4 died and is not a usable throughput "
                "point. min == max == that value, so C cannot differ from B and cannot fail when B "
                "passes. Any claim of the form 'survives all three cost levels' is therefore a "
                "claim about TWO distinct levels."
            )
            if not level_c_is_distinct
            else "two or more distinct ratios were measured; C is a genuinely separate level",
            "no_adverse_concurrency_level_is_measurable_from_this_ladder": not level_c_is_distinct,
        },
        "adverse_concurrency_sensitivity_NOT_a_cost_level": {
            "adverse_ratio_measured": adverse_ratio,
            "basis": (
                "K=4 under the BIGGER server context (n_ctx=32768) s/game "
                f"{adverse_sep} against the SAME-SEED K=1 comparator {k1_same_seed} "
                f"(seed {adverse_seeds[0] if len(adverse_seeds) == 1 else adverse_seeds})"
            ),
            "CONFOUND": (
                "there is no K=1 run at n_ctx=32768, so server context and concurrency move together "
                "in this contrast. It is NOT promoted to a cost level for that reason. It is "
                "published so a reader can see where the answer breaks rather than infer safety "
                "from a level that could not differ."
            ),
            "rows": adverse_rows,
            "reading": (
                "The cost ladder covers no ratio >= 1.0. The directly measured K=1 arm sits at ratio "
                "1.0 by definition and is shown here for reference; the confounded adverse ratio is "
                "where the headline budget stops fitting the margin."
            ),
        },
        "cost_level_definitions": {
            "A_central_best_measured_throughput": (
                "mean measured K=1 per-game wall x the most favourable measured throughput ratio"
            ),
            "B_ci_hi_best_measured_throughput": "bootstrap-CI-high per-game wall, best throughput",
            "C_ci_hi_worst_measured_throughput": "CI-high x the LEAST favourable measured concurrency",
            "a_fourth_level_was_DELETED_on_purpose": (
                "An earlier draft multiplied C by the published same-config fold-change MAX (2.468x). "
                "That is wrong statistics: a per-CELL noise statistic does not multiply the "
                "uncertainty of a 110-game MEAN, which the bootstrap CI already covers. It also "
                "single-handedly produced a 'nothing fits' answer. Replaced by the critical-N table."
            ),
        },
        "k1_per_game_wall_ci": k1_ci,
        "budget_scaling_ratio_uncertainty": ratio_uncertainty,
        "budget_scaling_ratio_uncertainty_why": (
            "The cap table multiplies the bootstrapped per-game wall CI by `budget_ratio` as if that "
            "ratio were EXACT. It is a median over 4 games. Propagating one multiplicand's "
            "uncertainty and not the other's understates the interval, and for the headline budget "
            "the fit flips inside the ratio's own observed range -- so the boundary is published."
        ),
        "throughput_ratio_by_K_vs_K1": ratios_by_K,
        "cap_table": rows,
        "largest_budget_fitting_WITH_MARGIN": feasible,
        "critical_n_games_per_cap_and_budget": critical_n,
        "budget_ratio_extrapolation": extrapolated,
        "margin_definition": f"total loop wall <= {MARGIN_FRACTION:.0%} of (cap - kernel overhead)",
        "kernel_overhead_s_ASSUMED_not_measured": KERNEL_OVERHEAD_S_ASSUMED,
        "what_the_budget_buys": buys,
        "score_maximising_budget_M0_offline_unit": score_max_M0,
        "score_maximising_budget_M0_basis": m0_direction,
        "score_maximising_budget_M2_paid_unit_verdict": score_max_M2_verdict,
        "score_M2_exact_subset_matched_delta": exact_block,
        "budget_scaling_provenance": (
            "paired per-game wall ratios vs b400 measured in "
            "outer_loop_arc_llm_on_wallclock_envelope_20260726.json (LLM ON, uncontended, 4 games, "
            "1 seed). Its own adjacent-step tests are UNDERPOWERED (p-floor 0.0625/0.125), so this "
            "scaling carries that uncertainty into every row above."
        ),
    }


def _walk_sign_tests(obj: Any, path: str = "") -> list[tuple[str, dict]]:
    """Every emitted sign test, so a structural gate can check they all publish both tails."""
    out = []
    if isinstance(obj, dict):
        if "min_reachable_two_sided_p_at_this_support" in obj:
            out.append((path, obj))
        for k, v in obj.items():
            out.extend(_walk_sign_tests(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_walk_sign_tests(v, f"{path}[{i}]"))
    return out


def part5_headline_and_gates(
    p1: dict, p2: dict, p3: dict, p4: dict, headline_cap: str, headline_n: int
) -> dict:
    """The answer, and the gates that make it falsifiable.

    Every gate carries a COMPUTED witness that its pass region was non-empty AND that it could have
    failed -- the forced-gate failure this project has shipped before. Where a gate's failure was
    OBSERVED this session, the observation is cited as the witness.
    """
    gates: list[dict] = []

    dev = p1.get("device_actually_used") or {}
    gates.append(
        {
            "gate": "G1_generator_ran_on_cuda_physical_gpu1_with_weights_resident",
            "passed": bool(
                dev.get("verdict") == "CUDA_GPU1" or dev.get("weights_are_resident_on_gpu1")
            ),
            "witness": {
                # RENAMED 2026-07-27: this field is NOT the resolver's own answer. The resolver
                # reports `resolver_says_cuda_gpu1: false`; the verdict below is derived from
                # per-PID VRAM residency on GPU1. Calling it `resolver_verdict` invited exactly the
                # misreading the artifact elsewhere guards against.
                "device_verdict_from_vram_residency": dev.get("verdict"),
                "resolver_says_cuda_gpu1_RAW": dev.get("resolver_says_cuda_gpu1"),
                "max_resident_mib_on_gpu1": dev.get("max_resident_mib_on_gpu1"),
                "weights_are_resident_on_gpu1": dev.get("weights_are_resident_on_gpu1"),
                "server_props": dev.get("server_props"),
            },
            "could_have_failed_and_DID_this_session": (
                "The FIRST launch of this probe at 02:16Z emitted "
                "blocked_generator_resolves_to_cuda_gpu1_not_igpu because the resolver returned the "
                "build-hip iGPU binary: a healthy CUDA-GPU1 server from the smoke run was itself "
                "holding 12.1 GiB and the resolver's guard wants >=13000 MiB FREE. The gate is "
                "demonstrably falsifiable, not decorative."
            ),
        }
    )

    conc_batches = [b for b in (p1.get("batches") or []) if int(b.get("K") or 1) > 1]
    gates.append(
        {
            "gate": "G2_concurrency_was_actually_achieved_on_every_K_gt_1_batch",
            "passed": bool(conc_batches)
            and all(bool(b.get("concurrency_actually_achieved")) for b in conc_batches),
            "witness": {
                "n_K_gt_1_batches": len(conc_batches),
                "overlap_fraction_of_longest_cell": [
                    b.get("overlap_fraction_of_longest_cell") for b in conc_batches
                ],
                "rule": "strict INTERSECTION of all cells' [start,end] > 50% of the longest cell",
            },
            "non_vacuity": (
                "The gate needs at least one K>1 batch to exist -- a ladder of K=1 alone would make "
                "it vacuous, so the batch count is part of the pass condition."
            ),
        }
    )

    live = p1.get("llm_liveness_by_K") or {}
    broke = [K for K, v in live.items() if not v.get("arm_is_genuinely_llm_on")]
    gates.append(
        {
            "gate": "G3_llm_channel_is_instrumented_on_every_arm_and_alive_on_the_K1_control",
            "passed": bool(live)
            and bool((live.get("1") or {}).get("arm_is_genuinely_llm_on"))
            and all((v.get("n_cells") or 0) > 0 for v in live.values()),
            "witness": live,
            "arms_where_the_llm_channel_BROKE": broke,
            "deliberately_NOT_gated_on_all_arms_being_llm_on": (
                "Whether the LLM channel SURVIVES concurrency is the measurement, not a precondition "
                "of it. Gating on it would make a real breakage unreportable. What IS gated is that "
                "the channel is instrumented everywhere and alive on the K=1 control -- otherwise a "
                "breakage could not be attributed to concurrency."
            ),
            "why_this_channel_needs_a_witness_at_all": (
                "generate() returns (False, msg) rather than raising when the server is unhealthy, so "
                "a broken generator yields a complete, error-free row that is silently LLM-OFF. The "
                "sibling lane observed exactly that (resp=0, genok True->False) on a cell whose server "
                "was demonstrably serving."
            ),
        }
    )

    ch = p2.get("channel_checks") or {}
    gates.append(
        {
            "gate": "G4_human_baseline_channel_alive_on_every_scored_row",
            "passed": bool(ch.get("baseline_channel_alive")),
            "witness": ch,
            "could_have_failed": (
                "A zero baseline makes M0/M1/M2 agree at score 0 and reads as a clean 'no optimism' "
                "null; a prior agent hit exactly that by reading env.baseline_actions instead of "
                "env.info.baseline_actions."
            ),
        }
    )

    ev = p2.get("estimator_validation") or {}
    gates.append(
        {
            "gate": "G5_corpus_wide_attribution_estimator_carries_its_measured_error",
            "passed": bool(
                (ev.get("n_exact_cells_usable") or 0) >= 20
                and ev.get("abs_rel_error_median") is not None
                and ev.get("abs_rel_error_max") is not None
                and ev.get("convention_identity_holds_on_all")
            ),
            "witness": {
                "n_exact_cells_usable": ev.get("n_exact_cells_usable"),
                "abs_rel_error_median": ev.get("abs_rel_error_median"),
                "abs_rel_error_p90": ev.get("abs_rel_error_p90"),
                "abs_rel_error_max": ev.get("abs_rel_error_max"),
                "signed_rel_error_median": ev.get("signed_rel_error_median"),
                "direction_test": ev.get("direction_test"),
                "convention_identity_holds_on_all": ev.get("convention_identity_holds_on_all"),
            },
            "why": (
                "This closes limitation (6) of REQ-ARC-WMTE-5986 (rows without per-span attribution "
                "had only a structurally-uninformative bound) but ONLY at the published accuracy -- "
                "the corpus numbers are estimates with a measured error, never exact."
            ),
        }
    )

    pb_all = p2.get("per_budget") or {}
    worst_diff = max(
        [(v.get("m0_vs_recorded_max_abs_diff") or 0.0) for v in pb_all.values()] or [1.0]
    )
    matched = sum(int(v.get("n_cells_m0_matches_recorded") or 0) for v in pb_all.values())
    # KEY RENAMED 2026-07-27 (`n_won_cells` -> `n_levelup_cells_summed_across_seeds`, because the
    # predicate is `levels > 0`, not "game won"). This gate READS that key, and reading the stale name
    # silently produced 0 and failed the gate -- which is the gate working. Kept as a single source of
    # truth rather than a fallback chain, so a future rename fails loudly here too.
    won_cells = sum(int(v.get("n_levelup_cells_summed_across_seeds") or 0) for v in pb_all.values())
    gates.append(
        {
            "gate": "G8_my_scorer_drive_reproduces_the_rows_own_installed_scorer_output_per_cell",
            "passed": bool(won_cells > 0 and matched == won_cells and worst_diff <= 5e-5),
            "witness": {
                "n_levelup_cells_summed_across_seeds": won_cells,
                "n_cells_where_my_M0_matches_the_recorded_efficiency": matched,
                "max_abs_diff": worst_diff,
                "tolerance": 5e-5,
                "note": (
                    "the row's `efficiency` was written at MEASUREMENT time by "
                    "arc_leaderboard_eval's `_calculate_score` through the INSTALLED scorer, and is "
                    "rounded to 4dp -- so this is an independent-path agreement check, not a tautology"
                ),
            },
            "why": (
                "If my re-derivation of M0 disagreed with the scorer that produced the corpus, every "
                "M1/M2 number here would be wrong with it and the disagreement would be invisible."
            ),
        }
    )

    key = f"{headline_cap}|n_games={headline_n}"
    feas = (p4.get("largest_budget_fitting_WITH_MARGIN") or {}).get(key) or {}
    tbl = [
        r
        for r in (p4.get("cap_table") or [])
        if r["cap"] == headline_cap and r["n_games"] == headline_n
    ]
    n_fit = sum(1 for r in tbl if r.get("fits_with_margin__A_central_best_measured_throughput"))
    n_nofit = len(tbl) - n_fit
    all_rows = p4.get("cap_table") or []
    fit_any = sum(
        1 for r in all_rows if r.get("fits_with_margin__A_central_best_measured_throughput")
    )
    nofit_any = len(all_rows) - fit_any
    gates.append(
        {
            "gate": "G6_the_feasibility_question_is_falsifiable_somewhere_in_the_cap_table",
            "passed": bool(fit_any >= 1 and nofit_any >= 1),
            "witness": {
                "whole_table_n_fitting_central": fit_any,
                "whole_table_n_NOT_fitting_central": nofit_any,
                "headline_cap": headline_cap,
                "headline_n_games": headline_n,
                "headline_row_n_budgets_fitting": n_fit,
                "headline_row_n_budgets_NOT_fitting": n_nofit,
                "headline_row_verdict": (
                    "EVERY_MEASURED_BUDGET_FITS_AT_THE_HEADLINE_CAP__so_the_clock_is_not_the_binding_"
                    "constraint_here"
                    if n_nofit == 0
                    else "THE_HEADLINE_ROW_ITSELF_CONTAINS_BOTH_A_FIT_AND_A_NON_FIT"
                ),
                "budgets_examined": sorted({r["budget"] for r in tbl}),
            },
            "why": (
                "If the whole table fitted, the answer would be produced by construction and carry no "
                "information. The gate is at TABLE scope on purpose: 'every budget fits at 9h/110' is "
                "itself the finding, and gating that row to contain a non-fit would have forced the "
                "opposite conclusion. The headline row's own verdict is reported separately."
            ),
        }
    )

    death = p1.get("generator_death_under_concurrency") or {}
    per_cell_all = p1.get("per_cell") or []
    naive_walls = [c["wall_s"] for c in per_cell_all if c["K"] == 1 and c.get("wall_s")]
    naive_invalid = [
        c["wall_s"]
        for c in per_cell_all
        if not c.get("row_valid") and c.get("wall_s") and c["K"] > 1
    ]
    gates.append(
        {
            "gate": "G9_dead_generator_cells_are_excluded_from_every_cost_number",
            "passed": bool(
                all(c.get("row_valid") for c in per_cell_all if c["K"] == 1)
                and (p4.get("k1_per_game_wall_ci") or {}).get("n") == len(naive_walls)
            ),
            "witness": {
                "n_invalid_cells_found": death.get("n_invalid_cells"),
                "Ks_fully_invalidated": death.get("Ks_fully_invalid"),
                "invalid_cell_walls_s": naive_invalid,
                "counterfactual_if_they_had_been_averaged_in": (
                    "a dead-generator cell is FASTER, not noisier -- these walls are LOWER than the "
                    "valid K=2 cells', so including them would have reported concurrency as a large "
                    "throughput WIN. That is the inverted conclusion this gate exists to prevent."
                ),
                "k1_ci_n_matches_valid_k1_cell_count": (p4.get("k1_per_game_wall_ci") or {}).get(
                    "n"
                ),
            },
            "why": (
                "The exclusion is load-bearing rather than cosmetic here: an entire arm was invalidated "
                "by a real generator death, and that arm's numbers look GOOD."
            ),
        }
    )

    tests = _walk_sign_tests({"p1": p1, "p2": p2, "p3": p3, "p4": p4})
    bad = [
        path
        for path, t in tests
        if "p_one_sided_increase" not in t or "p_one_sided_decrease" not in t
    ]
    gates.append(
        {
            "gate": "G7_every_emitted_test_publishes_both_tails_and_the_reachable_p_floor",
            "passed": bool(tests) and not bad,
            "witness": {
                "n_tests_emitted": len(tests),
                "tests_missing_a_tail": bad,
                "n_underpowered_stamped": sum(
                    1 for _p, t in tests if t.get("verdict") == "UNDERPOWERED_p_floor_above_0.05"
                ),
            },
            "why": "A one-sided test makes a REVERSAL read as no effect; a p that cannot reach 0.05 must say so.",
        }
    )

    # ---- the answer -------------------------------------------------------------------------
    buys = p4.get("what_the_budget_buys") or {}
    ans_central = feas.get("ANSWER_central")
    ans_cons = feas.get("ANSWER_conservative_all_levels")
    smax = p4.get("score_maximising_budget_M0_offline_unit")
    thr = p1.get("throughput_by_K") or {}
    gpu_sat_k1 = (thr.get("1") or {}).get("gpu1_util_mean_across_batches")
    ratios = p4.get("throughput_ratio_by_K_vs_K1") or {}
    concurrency_verdict = (
        "THROUGHPUT_FLAT_UNDER_CONCURRENCY__the_card_is_already_saturated_at_K1"
        if all((r or 1.0) >= 0.85 for r in ratios.values())
        else "CONCURRENCY_CHANGES_THROUGHPUT__see_throughput_ratio_by_K_vs_K1"
    )

    def _score_at(b) -> float | None:
        v = buys.get(str(b)) if b is not None else None
        return v.get("score_M2_bootstrap_free_median_THE_PAID_UNIT") if v else None

    wall_binds = bool(smax is not None and ans_central is not None and int(smax) > int(ans_central))
    es = p3.get("per_budget") or {}
    ceiling_at_smax = (es.get(str(smax)) or {}).get("score_free_share_of_offline_actions")

    headline = {
        "question": (
            "What should MAX_ACTIONS be? Answer as a budget with uncertainty, the median "
            "CELLS-WITH-AT-LEAST-ONE-LEVEL-UP it buys, and the GATEWAY-CHARGED score those cells "
            "are worth."
        ),
        "device_actually_used": {
            "verdict": dev.get("verdict"),
            "verdict_basis": dev.get("verdict_basis"),
            "resolver_would_launch_binary": dev.get("resolver_would_launch_binary"),
            "resolver_says_cuda_gpu1": dev.get("resolver_says_cuda_gpu1"),
            "physical_card": 1,
            "resident_mib_on_card_1": dev.get("max_resident_mib_on_gpu1"),
            "server_slots": (dev.get("server_props") or {}).get("total_slots"),
            "server_n_ctx": (dev.get("server_props") or {}).get("n_ctx"),
            "gpu0_left_to_the_conductor": True,
        },
        "headline_cap": headline_cap,
        "headline_n_games": headline_n,
        "ANSWER_budget_central": ans_central,
        "ANSWER_budget_central_is_EXTRAPOLATED": feas.get("ANSWER_central_is_EXTRAPOLATED"),
        "ANSWER_largest_MEASURED_budget_fitting_central": feas.get(
            "largest_MEASURED_budget_fitting_central"
        ),
        "ANSWER_budget_conservative_across_all_three_cost_levels": ans_cons,
        "ANSWER_per_cap_and_game_count": p4.get("largest_budget_fitting_WITH_MARGIN"),
        "ANSWER_critical_n_games_per_cap_and_budget": p4.get("critical_n_games_per_cap_and_budget"),
        "cost_levels_s_per_game_at_b400": p4.get("cost_levels_s_per_game_at_b400"),
        "k1_per_game_wall_ci": p4.get("k1_per_game_wall_ci"),
        "prior_cost_model_test": p4.get("prior_cost_model_test"),
        "concurrency_verdict": concurrency_verdict,
        "concurrency_verdict_what_it_does_and_does_NOT_claim": (
            "CLAIMS: total throughput (games per hour) did NOT degrade when games ran concurrently, "
            "while per-game LATENCY nearly doubled. That decomposition is what matters for a wall-clock "
            "CAP, because a cap divides throughput, not latency -- so multiplying an uncontended "
            "per-game latency by a contention factor and then by the game count DOUBLE-COUNTS. "
            "DOES NOT CLAIM that concurrency is a batching WIN: the small measured improvement is "
            "within this measurement's noise and is partly attributable to per-process setup "
            "(construct_s ~3-4 s per cell) overlapping rather than to GPU batching. The defensible "
            "statement is 'throughput is flat', not 'concurrency is faster'."
        ),
        "gpu1_util_mean_at_K1": gpu_sat_k1,
        "gpu1_util_reading": (
            "The card is already ~75-85% busy with a SINGLE game running, which is the mechanism behind "
            "flat throughput: there is little idle GPU for a second game to claim."
        ),
        "throughput_ratio_by_K_vs_K1": ratios,
        "what_the_answer_buys": {
            "budget": ans_central,
            "cells_with_at_least_one_levelup_median_per_seed": (
                buys.get(str(ans_central)) or {}
            ).get("cells_with_at_least_one_levelup_median_per_seed"),
            "gateway_score_M2_median": _score_at(ans_central),
            "out_of": (buys.get(str(ans_central)) or {}).get(
                "max_possible_score_for_this_game_count"
            ),
        },
        "score_maximising_budget_in_the_OFFLINE_unit_M0": smax,
        "score_maximising_budget_in_the_PAID_unit_M2": p4.get(
            "score_maximising_budget_M2_paid_unit_verdict"
        ),
        "what_the_score_maximising_budget_would_buy": {
            "budget": smax,
            "cells_with_at_least_one_levelup_median_per_seed": (buys.get(str(smax)) or {}).get(
                "cells_with_at_least_one_levelup_median_per_seed"
            ),
            "gateway_score_M2_median": _score_at(smax),
        },
        "the_raise_400_to_score_max_priced_in_both_units": {
            "matched_per_seed_deltas": p2.get("matched_per_seed_budget_deltas"),
            "exact_subset_matched_delta": p4.get("score_M2_exact_subset_matched_delta"),
            "reading": (
                "The OFFLINE-unit delta is exact and positive; the PAID-unit delta on the full corpus "
                "is smaller than the attribution estimator's own error, so its SIGN is not resolvable "
                "there. The exact-attribution subset is where a paid-unit sign can be read at all."
            ),
        },
        "does_wall_clock_bind_at_the_score_maximising_budget": wall_binds,
        "the_binding_constraint": (
            "NOT_WALL_CLOCK_AT_THE_HEADLINE_CAP__the_prize_is_the_binding_consideration"
            if not wall_binds
            else "WALL_CLOCK"
        ),
        "adaptive_early_stop_revival_condition": {
            "condition": (
                "wall clock binds at the score-maximising budget AND a large share of the wall clock "
                "is spent where no score can be earned"
            ),
            "condition_met": bool(wall_binds and (ceiling_at_smax or 0) > 0.25),
            "oracle_ceiling_score_free_share_of_offline_actions_at_score_max_budget": ceiling_at_smax,
            "oracle_ceiling_by_budget": {
                b: v.get("score_free_share_of_offline_actions") for b, v in es.items()
            },
            "smallest_adaptive_multiplier_preserving_every_observed_levelup": (
                p3.get("adaptive_window") or {}
            ).get("smallest_c_that_preserves_every_observed_levelup"),
            "start_window_trade_curve": p3.get("start_window_trade_curve"),
            "continuation_window_trade_curve": p3.get("continuation_window_trade_curve"),
            "why_the_ceiling_is_not_the_prize": (
                "The oracle ceiling is large (most of the corpus's actions earn no score) but it is an "
                "ORACLE: the bulk of it sits in cells that NEVER level up, which a real rule can only "
                "cut by giving up after W actions -- and the first-level-up cost distribution says what "
                "that costs in lost first-level-ups. The continuation window has the same problem from the other "
                "end: preserving EVERY observed subsequent level-up needs c=22x the game's own first "
                "level-up cost, so a tight window is not safe and a safe window barely fires. Both "
                "trade curves are published instead of a single multiplier."
            ),
            "reading": (
                "The grace sweep's 0.072%-of-actions saving at b400 is a fact about a FIXED window, "
                "not about the mechanism. The oracle ceiling is what an ADAPTIVE window is competing "
                "for, and it is score-free by construction: an incomplete level scores 0.0 whatever "
                "it is charged."
            ),
        },
        "a_concurrency_CEILING_was_hit_before_any_wall_clock_ceiling": {
            "what_happened": death.get("what_this_is"),
            "Ks_fully_invalidated": death.get("Ks_fully_invalid"),
            "n_invalid_cells": death.get("n_invalid_cells"),
            "server_config_at_the_time": {
                "total_slots": (dev.get("server_props") or {}).get("total_slots"),
                "n_ctx": (dev.get("server_props") or {}).get("n_ctx"),
                "per_slot_ctx_implied": (
                    int((dev.get("server_props") or {}).get("n_ctx", 0))
                    // max(1, int((dev.get("server_props") or {}).get("total_slots", 1)))
                    if (dev.get("server_props") or {}).get("n_ctx")
                    else None
                ),
                "agent_requests_max_tokens": 4096,
            },
            "consequence_for_the_MAX_ACTIONS_ANSWER": (
                "Every wall-clock number above assumes the LLM tier is RUNNING. At the eval's ~110 "
                "concurrent games it is not clear that it would be: 4 concurrent games killed this "
                "server outright, and the shipped per-slot context (n_ctx/total_slots) is smaller than "
                "the agent's own max_tokens request. If induction dies under eval concurrency the "
                "agent silently degrades to LLM-OFF -- which is CHEAPER in wall clock and WORSE in "
                "capability, so the budget question would then be answered by a different cost model "
                "entirely (the LLM-off sweep's, where 4000 was already affordable)."
            ),
            "this_reproduces_a_sibling_finding_in_the_LIVE_AGENT_PATH": (
                "The sibling lane found the same failure by isolated three-way prompt testing "
                "(HTTP 500 'Context size has been exceeded' at concurrency 4, server death in 4 of 5 "
                "observations). This is the same failure reached through arc_leaderboard_eval.run_game "
                "with the real E3AgentPolicy, so it is not an artifact of a synthetic prompt harness."
            ),
        },
        "which_flag_this_answer_is_about": {
            "the_live_scored_cap": (
                "python/carnot/agentic/arc_competition_agent.py:6244 -- `CarnotAgent.MAX_ACTIONS = 400`, "
                "the class attribute inside make_carnot_agent that the competition framework's own loop "
                "reads. This is the value the answer above is about."
            ),
            "a_second_MAX_ACTIONS_exists_and_is_a_different_budget": (
                "arc_competition_agent.py:117 -- module-level `MAX_ACTIONS = 200`. It is NOT the scored "
                "cap: it is imported by scripts/arc_competition_validate.py:21 and used there as the "
                "validation budget (`budget = 6000 if explore else MAX_ACTIONS`). Flipping the wrong one "
                "changes validation, not the submission. Both line numbers were already recorded in "
                "results/outer_loop_scored_path_budget_sweep_20260726.json -- noted here so the answer "
                "names its own referent rather than inheriting the ambiguity."
            ),
        },
        "NOT_a_recommendation": True,
        "no_flag_was_flipped": True,
    }
    return {
        "acceptance_gates": gates,
        "acceptance_gates_all_passed": all(g["passed"] for g in gates),
        "acceptance_gate_failures": [g["gate"] for g in gates if not g["passed"]],
        "headline": headline,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ladder", nargs="*", default=None)
    ap.add_argument(
        "--out",
        default=os.path.join(REPO, "results/outer_loop_arc_max_actions_answer_20260726.json"),
    )
    a = ap.parse_args(argv)

    ladder_paths = a.ladder or sorted(
        glob.glob(os.path.join(REPO, "results/llm_on_contention_rows_20260726/*.json"))
    )
    sweep_files = sorted(
        [
            p
            for p in glob.glob(os.path.join(REPO, "results/early_stop_sweep_20260726/rows_b*.json"))
            if "g350" not in p
        ]
    )
    exact_files = sorted(
        glob.glob(
            os.path.join(REPO, "results/early_stop_sweep_20260726/rows_exact_attribution*.json")
        )
    )
    sibling = _load_json(
        os.path.join(REPO, "results/outer_loop_arc_llm_on_wallclock_envelope_20260726.json")
    )
    rescore = _load_json(
        os.path.join(REPO, "results/outer_loop_arc_gateway_accurate_rescore_20260726.json")
    )

    p1 = part1_contention(ladder_paths, sibling)
    p2 = part2_score_curve(sweep_files, exact_files)
    p3 = part3_early_stop_ceiling(sweep_files)
    p4 = part4_answer(p1, p2, p3, sibling, [110, 60, 25])
    p5 = part5_headline_and_gates(p1, p2, p3, p4, "kaggle_9h_max_notebook_runtime", 110)

    # ---- measurement clock vs analyser clock (failure #8) ------------------------------------
    # LATENT DOUBLE-COUNT FIXED 2026-07-27. This loop used to add BOTH candidate fields per file
    # (`for k in ("elapsed_s","measurement_wall_s"): if f.get(k): meas += ...`). It happened to be
    # correct because no row file currently carries both (6 have `elapsed_s`, 2 have
    # `measurement_wall_s`, none both) -- but the first time an upstream row file grows the second
    # field, the measurement clock silently INFLATES rather than failing. One value per file now,
    # with explicit precedence, an assertion that no file carries both, and a per-file record of
    # which field contributed so the composition is auditable.
    meas = 0.0
    meas_composition = []
    meas_files_with_both = []
    for f in (p1.get("row_files") or []) + (p2.get("row_files") or []):
        has = [k for k in ("elapsed_s", "measurement_wall_s") if f.get(k)]
        if len(has) > 1:
            meas_files_with_both.append({"path": f.get("path"), "fields": has})
        field = has[0] if has else None
        v = float(f[field]) if field else 0.0
        meas += v
        meas_composition.append(
            {"path": f.get("path"), "field_used": field, "value_s": round(v, 3)}
        )
    assert not meas_files_with_both, (
        "a row file carries BOTH elapsed_s and measurement_wall_s; the precedence rule must be "
        f"restated before this clock can be trusted: {meas_files_with_both}"
    )
    art: dict[str, Any] = {
        "experiment": "outer_loop_arc_max_actions_answer_20260726",
        "title": (
            "The MAX_ACTIONS answer: LLM-ON contention measured (not imported), the budget->score "
            "curve re-priced in the GATEWAY-CHARGED unit, and the early-stop lever priced on its "
            "benefit side"
        ),
        # The session-local date, matching every sibling artifact from this session. The BUILD's UTC
        # timestamp is later (this work ran past local midnight-minus-2h, i.e. after 00:00Z), so the
        # two are recorded separately rather than letting a reader think the file is misdated.
        "run_date": "2026-07-26",
        "run_date_convention": "session-local (America/New_York); see build_timestamp_utc",
        "build_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema": "carnot.arc_max_actions_answer.v1",
        # THIS FILE performs no inference: it is an analyser pass over persisted rows. Declared as a
        # BARE STRING deliberately -- a dict here is unreadable to the substrate classifier in
        # scripts/adversarial_verify.py, which then falls back to the strict live-inference duration
        # floor and CRITICAL-flags an honest aggregation (observed on the first build of this file).
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_principle": (
            "An analyser pass inherits its methodology from the artifacts it cites; declaring it as "
            "live inference would borrow credibility this file has not earned, and declaring it as a "
            "dict makes the classifier unable to read it at all."
        ),
        "rows_inference_substrate": {
            "contention_ladder_rows": "live_llm_inference",
            "contention_ladder_rows_detail": (
                "results/llm_on_contention_rows_20260726/*.json -- the frozen Qwen3.5-9B-MTP "
                "llama-server on CUDA GPU 1 with real induction calls, per-cell wall clock 65-190 s. "
                "Those files carry their own preconditions, device witnesses and elapsed_s."
            ),
            "early_stop_sweep_rows": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
            "exact_attribution_rows": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        },
        "model_specs": {
            "generator_used_by_the_cited_ladder_rows": "unsloth/Qwen3.5-9B-MTP-GGUF (Q4_K_M)",
            "invoked_by_this_file": False,
            "note": (
                "Named because the ladder rows this file aggregates DID load it; this analyser did not. "
                "Recorded rather than omitted so the methodology chain is followable."
            ),
        },
        "measurement_wall_s": round(meas, 1),
        "measurement_wall_s_composition": meas_composition,
        "measurement_wall_s_basis": (
            "one value per ROW FILE with explicit precedence (`elapsed_s` before "
            "`measurement_wall_s`), asserted to be unambiguous (no file carries both). NOT a sum of "
            "per-cell `wall_s`, which undercounts ~25% because it omits per-cell setup."
        ),
        "duration_s": None,  # filled at the end
        "random_seed": 20260724,
        "verifier_is_oracle": False,
        "verifier_is_oracle_principle": (
            "No verifier-value or moat claim is made here; this measures WALL CLOCK and re-prices a "
            "score. Recorded because the circularity discipline requires the field on any artifact "
            "that could be read as a verifier claim."
        ),
        "solve_provenance": "development_proxy",
        "solve_provenance_principle": (
            "Cells run the SCORED policy (E3AgentPolicy) against OFFLINE public-game envs via "
            "arc_leaderboard_eval. No new level is claimed and no registry entry is written."
        ),
        "arc_solve_claim": False,
        "claims_new_solve": False,
        "what_was_NOT_changed": [
            "MAX_ACTIONS is still 400 in the tree -- no flag was flipped",
            "no SUBMITTED_* global was touched",
            "nothing was submitted to Kaggle or the ARC gateway",
            "no historical artifact's recorded numbers were rewritten",
        ],
        "upstream_artifacts_cited": {
            "wallclock_envelope": {
                "path": "results/outer_loop_arc_llm_on_wallclock_envelope_20260726.json",
                "used_for": "per-game budget scaling ratios, published uncontended level, noise floor",
                "honest_verdict": sibling.get("honest_verdict"),
            },
            "gateway_accurate_rescore": {
                "path": "results/outer_loop_arc_gateway_accurate_rescore_20260726.json",
                "used_for": "the M2 bootstrap-free charge model this file re-uses",
                "honest_verdict": rescore.get("honest_verdict"),
            },
        },
        "known_caveats_about_this_files_own_construction": {
            "probe_file_was_ruff_formatted_between_ladder_runs": (
                "scripts/arc_llm_on_contention_probe.py was ruff-FORMATTED (whitespace only, by the "
                "formatter -- not edited) at 02:26Z, after the seed-20260724 ladder's K=1 and first "
                "K=2 batch had already run. The provenance sha256 recorded here is the POST-format "
                "content, so those early batches were produced by byte-different-but-semantically-"
                "identical source. This is disclosed rather than papered over: the seed-20260725 "
                "ladder was run ENTIRELY from the recorded bytes, and the throughput result is "
                "reported per seed so a reader can check the two agree."
            ),
            "the_gpu_sampler_is_part_of_the_measured_process": (
                "The parent samples nvidia-smi every 2s during each batch. That cost is inside the "
                "batch wall clock. It is a subprocess call per 2s on a 24-core box and is not "
                "corrected for -- it inflates every arm equally, including the K=1 control."
            ),
        },
        "part_1_contention_and_throughput": p1,
        "part_2_score_curve_in_the_gateway_unit": p2,
        "part_3_early_stop_lever_benefit_side": p3,
        "part_4_the_answer": p4,
        "acceptance_gates": p5["acceptance_gates"],
        "acceptance_gates_all_passed": p5["acceptance_gates_all_passed"],
        "acceptance_gate_failures": p5["acceptance_gate_failures"],
        "headline": p5["headline"],
        "scope_and_power": {
            "contention_ladder": {
                "n_games": len(p1.get("games") or []),
                "games": p1.get("games"),
                # DERIVED, NOT LITERAL (2026-07-27). These two fields used to be hardcoded
                # `n_seeds: 1, seed: 20260724`, and the p-floor was computed from the GAME count
                # alone (2/2**n_games = 0.125), ignoring seeds entirely. The ladder actually spans
                # TWO seeds and 24 valid cells, and the tests this artifact emits run over matched
                # game-seed PAIRS with floors as low as 0.0312. A reader auditing the headline
                # p=0.0312 against a stated floor of 0.125 would have concluded the artifact
                # reported significance below its own floor -- a false violation manufactured by a
                # stale literal. Both are now computed, and the block-level floor is the MINIMUM of
                # the floors of the tests actually emitted (see the self-consistency assertion in
                # `_scope_floor_is_consistent`).
                "n_seeds": len(p1.get("seeds") or []),
                "seeds": sorted(p1.get("seeds") or []),
                "n_valid_cells": sum(1 for c in (p1.get("per_cell") or []) if c.get("row_valid")),
                "budget": 400,
                "Ks": p1.get("Ks_measured"),
                "min_reachable_two_sided_p_at_this_support": _min_emitted_p_floor(p1),
                "min_reachable_two_sided_p_at_this_support_basis": (
                    "MINIMUM over the `min_reachable_two_sided_p_at_this_support` of every sign test "
                    "this part actually emits (each computed from that test's own matched-pair "
                    "count), NOT 2/2**n_games. See the top-level "
                    "`scope_and_power_self_consistency` block for the artifact-wide check that no "
                    "emitted test reports a p below the floor its own scope block claims."
                ),
                "game_selection_is_NOT_random": (
                    "the 4 games with complete budget coverage in the sibling uncontended probe, "
                    "themselves chosen because budget 2000 GAINS A WIN on them -- deliberately "
                    "adverse for a cost question, so the per-game cost is an OVER-estimate and the "
                    "affordable budget an UNDER-estimate"
                ),
                "processes_not_threads": (
                    "the real eval is single-process/multi-thread (swarm.py:76-99), so it ALSO pays "
                    "GIL serialisation on the non-LLM component that separate processes do not. This "
                    "probe measures SERVER-side contention exactly and UNDERSTATES the GIL side."
                ),
                "hidden_set_size_is_unknown": (
                    "n_games=110 is inherited from the prior artifact's assumption, not measured. "
                    "The answer is therefore reported for 110 / 60 / 25 separately."
                ),
            },
            "score_curve": {
                "scope": (p2.get("scope") or ""),
                "n_rows_grace_none": (p2.get("channel_checks") or {}).get("n_rows_grace_none"),
                "llm_off": True,
                "llm_off_caveat": (
                    "the score curve is measured on LLM-OFF cells (the only corpus with 3 seeds x 25 "
                    "games x reset counts). The LLM-ON path can change WHICH levels are reached, so "
                    "the score curve is a proxy for the scored path, not the scored path itself."
                ),
                "b4000_is_a_subset": (
                    "b4000 exists only for a 13-game LEVEL-REACHING subset, so it is biased UPWARD on "
                    "level-up cells and is excluded from the score-maximising-budget comparison, which is "
                    "restricted to budgets with the full 25-game corpus."
                ),
            },
            "public_games_are_not_the_hidden_set": (
                "Every cell is a PUBLIC game played OFFLINE. The hidden competition set is "
                "out-of-distribution relative to all of them; nothing here is a hidden-set forecast."
            ),
        },
    }
    # ---- SELF-CONSISTENCY: no emitted test may report a p below its scope block's stated floor ---
    # Added 2026-07-27 after a review found `scope_and_power.contention_ladder` published a
    # hardcoded 1-seed scope and a p-floor of 0.125 derived from the game count alone, while the
    # artifact's own headline reported p=0.0312. A reader checking one against the other would have
    # concluded the artifact reported significance below its own floor. This check makes that class
    # of drift a first-class recorded number instead of a latent contradiction.
    _all_floors = _walk_p_floors(art)
    _n_degenerate_zero_floors = sum(
        1 for _f in _leaf_floor_values(art) if isinstance(_f, (int, float)) and float(_f) <= 0.0
    )
    _claimed = ((art.get("scope_and_power") or {}).get("contention_ladder") or {}).get(
        "min_reachable_two_sided_p_at_this_support"
    )
    _min_p_reported = None

    def _walk_ps(node, out):
        if isinstance(node, dict):
            for k in ("p_two_sided", "p", "p_value"):
                v = node.get(k)
                if (
                    isinstance(v, (int, float))
                    and node.get("min_reachable_two_sided_p_at_this_support") is not None
                ):
                    out.append((float(v), float(node["min_reachable_two_sided_p_at_this_support"])))
            for x in node.values():
                _walk_ps(x, out)
        elif isinstance(node, list):
            for x in node:
                _walk_ps(x, out)

    _pairs: list = []
    _walk_ps(art, _pairs)
    _violations = [{"p_reported": pv, "own_floor": fl} for pv, fl in _pairs if pv < fl - 1e-12]
    if _pairs:
        _min_p_reported = round(min(pv for pv, _ in _pairs), 6)
    art["scope_and_power_self_consistency"] = {
        "n_tests_with_both_a_p_and_a_floor": len(_pairs),
        "min_p_reported_anywhere": _min_p_reported,
        "min_POSITIVE_floor_emitted_anywhere": round(min(_all_floors), 6) if _all_floors else None,
        "n_degenerate_zero_floors_excluded": _n_degenerate_zero_floors,
        "why_zero_floors_are_excluded": (
            "a floor of 0.0 is what a sign test emits on EMPTY support, not a reachable p. Including "
            "it drags the minimum to zero and makes this check meaningless."
        ),
        "scope_block_claimed_floor": _claimed,
        "scope_block_floor_is_LE_every_emitted_floor": (
            None
            if (_claimed is None or not _all_floors)
            else bool(_claimed <= min(_all_floors) + 1e-12)
        ),
        "n_tests_reporting_p_below_their_own_floor": len(_violations),
        "violations": _violations,
        "passed": bool(not _violations)
        and (_claimed is None or not _all_floors or _claimed <= min(_all_floors) + 1e-12),
        "principle": (
            "a scope block whose stated p-floor is ABOVE a p the same artifact reports is an "
            "internal contradiction that reads as a methodology violation; deriving the floor and "
            "checking it makes the drift fail the build instead of misleading a reader."
        ),
    }
    row_paths = [f["path"] for f in (p1.get("row_files") or []) + (p2.get("row_files") or [])]
    art["provenance"] = {
        "git_head": _git_head(),
        "code": [_file_fingerprint(p) for p in _code_dependencies()],
        "rows_sources": {
            "ladder_rows": [_file_fingerprint(p) for p in ladder_paths],
            "sweep_rows": [_file_fingerprint(p) for p in sweep_files],
            "exact_attribution_rows": [_file_fingerprint(p) for p in exact_files],
        },
        "rebuild_command": "python scripts/analyze_arc_max_actions_answer.py --out <this file>",
        "row_paths_used": row_paths,
    }
    art["reproducibility_checksum"] = hashlib.sha256(
        json.dumps(
            {
                k: v
                for k, v in art.items()
                # `build_timestamp_utc` MUST be excluded or the checksum changes on every rebuild and
                # can never answer the only question it exists for: "did any NUMBER change?" Caught by
                # a rebuild-determinism check on this file itself.
                if k
                not in (
                    "run_date",
                    "duration_s",
                    "build_timestamp_utc",
                    "provenance",
                    "reproducibility_checksum",
                )
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()[:16]
    art["reproducibility_checksum_note"] = (
        "sha256 over the artifact MINUS every clock field (run_date, duration_s, build_timestamp_utc) "
        "and provenance, so a rebuild that changes no NUMBER reproduces the checksum exactly. Verified "
        "by rebuilding twice and diffing."
    )
    # HONEST VERDICT, COMPUTED from the headline rather than written by hand -- a hand-written verdict
    # is how an artifact ends up contradicting its own table (a defect this project has shipped).
    hl0 = art["headline"]
    art["honest_verdict"] = (
        "complete_max_actions_answer_"
        f"largest_measured_budget_fitting_9h_110games={hl0['ANSWER_largest_MEASURED_budget_fitting_central']}_"
        f"conservative_all_levels={hl0['ANSWER_budget_conservative_across_all_three_cost_levels']}_"
        f"throughput_ratio_K2_vs_K1={hl0['throughput_ratio_by_K_vs_K1'].get('2')}_"
        f"latency_nearly_doubles_but_throughput_does_not_"
        f"wall_clock_binds_at_score_max={hl0['does_wall_clock_bind_at_the_score_maximising_budget']}_"
        f"generator_died_at_K={hl0['a_concurrency_CEILING_was_hit_before_any_wall_clock_ceiling']['Ks_fully_invalidated']}_"
        f"gates_passed={art['acceptance_gates_all_passed']}"
    ).replace(" ", "")
    art["honest_verdict_principle"] = (
        "A terminal-prefixed self-declared state lets the conductor's reconciler classify this without "
        "re-running it; the prefix is required because substrings like 'binds' and 'died' would "
        "otherwise trip the partial/blocked token matcher."
    )
    art["duration_s"] = round(time.time() - T_START, 3)
    art["duration_s_principle"] = (
        "This analyser's OWN runtime. It reads persisted JSON, so it is small by construction and is "
        "NOT evidence about how long the measurement took -- see measurement_wall_s."
    )
    with open(a.out, "w") as fh:
        json.dump(art, fh, indent=1, default=str)

    # Register so the freshness lint knows this artifact exists and which analyser rebuilds it.
    # `analyzer=` MUST be passed: the helper defaults to the module that DEFINES it, and the first
    # external reuser registered its artifact under the wrong analyser name.
    try:
        from pathlib import Path as _P

        import analyze_scored_path_lever_ab as _reg

        _reg.register_analyzed_artifact(_P(a.out), analyzer=_P(os.path.abspath(__file__)))
    except Exception as exc:  # pragma: no cover -- registration failure must be LOUD, not fatal
        print(
            f"WARNING: artifact registration failed ({type(exc).__name__}:{exc}) -- the freshness "
            f"lint will not cover this artifact until it is registered"
        )
    print(f"wrote {a.out}")
    hl = art["headline"]
    print(
        f"ANSWER central={hl['ANSWER_budget_central']} "
        f"conservative={hl['ANSWER_budget_conservative_across_all_three_cost_levels']} "
        f"score_max_budget={hl['score_maximising_budget_in_the_OFFLINE_unit_M0']} "
        f"wall_binds={hl['does_wall_clock_bind_at_the_score_maximising_budget']} "
        f"gates_passed={art['acceptance_gates_all_passed']} "
        f"failures={art['acceptance_gate_failures']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

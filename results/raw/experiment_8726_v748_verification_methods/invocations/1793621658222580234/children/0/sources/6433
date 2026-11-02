#!/usr/bin/env python3
"""DOES THE GENERATOR SURVIVE -- AND SPEED UP UNDER -- THE CONCURRENCY THE REAL EVAL CREATES?

WHY THIS EXISTS
===============
Two facts collided while measuring the MAX_ACTIONS wall-clock envelope:

1. The eval framework's `Swarm` starts **one thread per game** and joins them all
   (`ARC-AGI-3-Agents/agents/swarm.py:76-99`). With ~110 hidden games, ~110 agent threads are live
   simultaneously, so many induction requests are in flight at once.

2. The llama-server this project ships reports **`total_slots = 4`** (read live off `/props`,
   2026-07-26). This CORRECTED an earlier inference in this same session that the server had ONE
   slot because no `--parallel`/`-np` flag is passed -- the build's default is not 1, and asserting
   it from the absence of a flag would have been exactly the "assumed instead of measured" error
   this project keeps making.

Every LLM-on measurement this project has ever taken was issued at CONCURRENCY 1 (one dev process,
one request at a time). So two load-bearing properties of the scored path are untested:

  A. THE SPEEDUP. If 4 slots batch effectively on one GPU, the eval's total LLM wall clock is the
     serial sum DIVIDED by some factor S in [1, 4]. Every envelope projection this project holds
     assumes S = 1 (a pure serial sum). If S is really ~2-3, the affordable action budget is
     2-3x larger than any current estimate -- this is the single cheapest way to buy budget
     headroom, and nobody has measured it.

  B. THE CONTEXT HAZARD. In llama.cpp a server's context is divided among its slots. Our induction
     prompts are large (measured `tokens_prompt` 2072-11937 across real cells) and ask for up to
     4096 output tokens. If the per-slot allowance is the total context divided by 4, a prompt that
     fits comfortably at concurrency 1 may NOT fit when 4 slots are active -- a failure that
     single-threaded dev cannot possibly reveal.

     The reason B is dangerous rather than merely inconvenient: `LocalGGUFProposer.generate()`
     returns `(False, msg)` instead of raising when a request fails, so the agent logs
     `skipped: proposer_failed` and CARRIES ON. The run silently degrades to LLM-off while still
     being labelled an LLM-on run. That is this project's canonical "dead channel reads as a clean
     null" failure, and at eval scale it would look like the agent simply not benefiting from
     induction rather than like a capacity bug.

WHAT IT DOES
============
Issues the SAME synthetic induction-sized request N times, first strictly sequentially and then all
at once, against an ALREADY-RUNNING server, and compares aggregate wall clock and success. Nothing
is inferred from flags; both numbers are measured, and every failure body is recorded verbatim.

It does not submit anything, does not touch GPU 0, and changes no configuration.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import error, request


def one_request(port: int, prompt: str, n_predict: int, timeout: int) -> dict:
    """One /completion call, with the failure body captured rather than swallowed.

    Capturing the body matters: a context-overflow rejection and a model crash both surface as "no
    usable output" to the caller, and only the server's own message distinguishes them.
    """
    body = json.dumps(
        {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.7,
            "cache_prompt": False,  # a shared prefix cache would hide the real per-request cost
        }
    ).encode()
    req = request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t = time.time()
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            d = json.loads(resp.read())
        el = time.time() - t
        tim = d.get("timings") or {}
        return {
            "ok": True,
            "wall_s": round(el, 2),
            "tokens_predicted": d.get("tokens_predicted"),
            "tokens_evaluated": d.get("tokens_evaluated"),
            "truncated": d.get("truncated"),
            "stop_type": d.get("stop_type"),
            "predicted_per_second": tim.get("predicted_per_second"),
            "prompt_per_second": tim.get("prompt_per_second"),
        }
    except error.HTTPError as exc:
        return {
            "ok": False,
            "wall_s": round(time.time() - t, 2),
            "http_status": exc.code,
            "error_body": exc.read().decode(errors="replace")[:800],
        }
    except Exception as exc:
        return {
            "ok": False,
            "wall_s": round(time.time() - t, 2),
            "error": f"{type(exc).__name__}:{exc}",
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--port",
        type=int,
        required=True,
        help="an ALREADY-RUNNING generator server. This probe never spawns one: "
        "spawning would change the very GPU state being measured.",
    )
    ap.add_argument(
        "--concurrency", type=int, default=4, help="match the server's reported total_slots"
    )
    ap.add_argument(
        "--prompt-tokens",
        type=int,
        default=6000,
        help="approximate prompt size. Real induction prompts measured 2072-11937 "
        "tokens on this project's own LLM-on cells, so 6000 is mid-range and NOT "
        "an adversarially large probe.",
    )
    ap.add_argument(
        "--n-predict",
        type=int,
        default=400,
        help="kept well below the agent's real 4096 so the test isolates CONCURRENCY "
        "rather than becoming a long-generation benchmark",
    )
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # Digit-dense text, because that is what ARC grids are and this project's own notes record that
    # such text tokenizes at roughly one token per character for these models.
    prompt = (
        "/no_think\nGrid transition log:\n"
        + " ".join(str(i % 10) for i in range(a.prompt_tokens))
        + "\nDescribe the rule in one sentence."
    )

    props = {}
    try:
        with request.urlopen(f"http://127.0.0.1:{a.port}/props", timeout=30) as r:
            p = json.loads(r.read())
        props = {
            "total_slots": p.get("total_slots"),
            "slot_n_ctx": (p.get("default_generation_settings") or {}).get("n_ctx"),
            "model_path": p.get("model_path"),
        }
    except Exception as exc:
        props = {"error": f"{type(exc).__name__}:{exc}"}
    print(f"[props] {props}", flush=True)

    n = a.concurrency

    # SEQUENTIAL FIRST. This is the baseline the project's every prior LLM-on number was taken at.
    t0 = time.time()
    seq = [one_request(a.port, prompt, a.n_predict, a.timeout) for _ in range(n)]
    seq_wall = round(time.time() - t0, 2)
    print(
        f"[sequential] n={n} total={seq_wall}s ok={sum(1 for x in seq if x['ok'])}/{n}", flush=True
    )

    # THEN ALL AT ONCE. Same requests, same server, same GPU -- only the arrival pattern differs,
    # which is the one thing the eval changes relative to dev.
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        par = list(ex.map(lambda _: one_request(a.port, prompt, a.n_predict, a.timeout), range(n)))
    par_wall = round(time.time() - t0, 2)
    print(
        f"[parallel]   n={n} total={par_wall}s ok={sum(1 for x in par if x['ok'])}/{n}", flush=True
    )

    seq_ok = sum(1 for x in seq if x["ok"])
    par_ok = sum(1 for x in par if x["ok"])

    # A SPEEDUP IS ONLY A SPEEDUP IF THE WORK ACTUALLY HAPPENED.
    # The first version of this script computed seq_wall/par_wall unconditionally, and on the very
    # first real run that produced "3.041x -- CONCURRENCY HELPS" out of a parallel batch in which
    # ALL FOUR REQUESTS FAILED. They failed FAST, so the wall clock fell and the ratio rose. That is
    # a degenerate metric reading success out of total failure -- the same defect class as a gate
    # whose pass region is empty. The speedup is therefore gated on every request in BOTH arms
    # succeeding, and the ungated ratio is still reported, clearly labelled, so the failure mode is
    # visible rather than hidden.
    raw_ratio = round(seq_wall / par_wall, 3) if par_wall else None
    both_arms_complete = bool(seq_ok == n and par_ok == n)
    speedup = raw_ratio if both_arms_complete else None

    out = {
        "probe": "arc_generator_slot_concurrency",
        "run_date": "2026-07-26",
        "server_props": props,
        "concurrency": n,
        "approx_prompt_tokens_requested": a.prompt_tokens,
        "n_predict": a.n_predict,
        "sequential_total_wall_s": seq_wall,
        "parallel_total_wall_s": par_wall,
        "parallel_speedup_factor_S": speedup,
        "parallel_speedup_is_MEANINGLESS_unless_both_arms_completed": {
            "both_arms_complete": both_arms_complete,
            "raw_wall_ratio_DO_NOT_READ_AS_SPEEDUP_IF_INCOMPLETE": raw_ratio,
            "why": "Failed requests return fast. A wall-clock ratio computed across a failed "
            "parallel arm measures how quickly the server gave up, not how much work it "
            "batched. S is null unless every request in both arms succeeded.",
        },
        "sequential_requests": seq,
        "parallel_requests": par,
        "sequential_ok": seq_ok,
        "parallel_ok": par_ok,
        "any_parallel_request_failed": par_ok < n,
        "any_parallel_request_truncated": any(x.get("truncated") for x in par),
        "any_sequential_request_truncated": any(x.get("truncated") for x in seq),
        # The two verdicts, kept separate because they have different consequences: A changes the
        # affordable budget, B is a correctness/silent-degradation risk.
        "verdict_A_speedup": (
            "UNMEASURABLE_parallel_arm_did_not_COMPLETE_see_verdict_B"
            if not both_arms_complete
            else "NO_DATA"
            if speedup is None
            else "CONCURRENCY_HELPS_envelope_projections_assuming_serial_sum_are_PESSIMISTIC"
            if speedup >= 1.25
            else "CONCURRENCY_DOES_NOT_HELP_serial_sum_is_the_right_model"
        ),
        "verdict_B_context_hazard": (
            "FAILURES_OR_TRUNCATION_UNDER_CONCURRENCY_ONLY"
            if (
                par_ok < seq_ok
                or (
                    any(x.get("truncated") for x in par)
                    and not any(x.get("truncated") for x in seq)
                )
            )
            else "NO_CONCURRENCY_ONLY_FAILURE_AT_THIS_PROMPT_SIZE"
        ),
        "scope_limits": [
            "ONE prompt size and ONE n_predict. A larger induction prompt (real cells reached "
            "11937 prompt tokens) could still overflow a slot where 6000 does not.",
            "Concurrency equal to the slot count, not to the ~110 threads the eval creates. Beyond "
            "the slot count requests QUEUE, which is a different regime this does not measure.",
            "This box's GPU (RTX 3090, 24GB), not Kaggle's (L4 24GB / RTX 6000). Batching "
            "efficiency is hardware-dependent.",
            "A speedup measured on synthetic uniform requests is an UPPER bound on what "
            "heterogeneous real induction traffic would achieve.",
        ],
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"speedup_S={speedup}  A={out['verdict_A_speedup']}  B={out['verdict_B_context_hazard']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

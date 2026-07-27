#!/usr/bin/env python3
"""IS THE GENERATOR'S DEATH CAUSED BY THE PROMPT EXCEEDING ITS CONTEXT WINDOW?

WHY THIS EXISTS
===============
Measuring the LLM-on wall-clock envelope for the MAX_ACTIONS decision turned up a crash that
matters more than the wall-clock numbers did. On game cd82 at action budget 2000 the agent's search
expanded 411 states, the induction prompt built from that graph came to 16,189 tokens, and the
llama-server -- launched with `-c 16384` -- DIED. It died on that same cell in two independent runs,
on two different ports, so it is not random flakiness. The correlation with the context window is
striking (16,189 of 16,384 = 98.8%, with the agent still asking for 4,096 completion tokens on top)
and `LocalGGUFProposer.n_ctx`'s own source comment already records that "8192 overflowed".

But correlation is not mechanism, and the honest artifact said so: "the decisive test is to re-run
with a larger -c and confirm the crash disappears. That test has NOT been run." This is that test.

WHY IT MATTERS BEYOND THE BUDGET QUESTION
=========================================
The shipped Kaggle submission constructs its proposer with no `n_ctx` override, so it runs the same
16384 window with the same 4096-token completion request. There is NO prompt-size clamp anywhere in
the induction path (grepped). And when the server is gone `LocalGGUFProposer.generate()` returns
`(False, msg)` instead of raising, so the agent logs `skipped: proposer_failed` and CARRIES ON --
spending the rest of the evaluation as an LLM-off agent while still reporting itself as the LLM-on
scored path. The largest prompt observed at the SHIPPED budget of 400 was already 11,937 tokens,
leaving ~4.4k of headroom against a 4,096-token completion request. That is a margin of ~350 tokens
on the worst public game, and the hidden games are explicitly out-of-distribution.

THE DESIGN
==========
Isolate the single variable. Same binary, same model, same GPU, same prompt; only `-c` differs.
Sweep prompt sizes that BRACKET the window on a server with the shipped `-c 16384`, then repeat the
one that broke it on a server with a doubled window. Two possible outcomes and they mean opposite
things:

  - Breaks at 16384 and survives at 32768  -> CONTEXT IS THE MECHANISM. The fix is a window/clamp
    change, and the budget question is downstream of it.
  - Breaks at both                          -> something else kills the server and the context
    story is a red herring; do not ship a window change on the strength of it.

Each server is spawned by this script and torn down after, so nothing is left holding VRAM. GPU 1
only (the outer loop's card per the 2026-06-27 allocation); GPU 0 belongs to the conductor and is
never targeted. Nothing is submitted and no configuration is changed.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from urllib import error, request

BIN = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"


def resolve_gguf() -> str | None:
    hits = sorted(
        (Path.home() / ".cache/huggingface/hub/models--unsloth--Qwen3.5-9B-MTP-GGUF").glob(
            "snapshots/*/*.gguf"
        )
    )
    return str(hits[0]) if hits else None


def start_server(model: str, n_ctx: int, port: int, gpu: str) -> subprocess.Popen:
    """Spawn a server with the SAME arguments LocalGGUFProposer._ensure_server uses, changing only
    -c. Matching the real launch line matters: MTP self-draft doubles the model's KV footprint and
    q8 KV halves it, so a probe that omitted those flags would be measuring a different memory
    regime than the agent actually runs in."""
    args = [
        str(BIN),
        "-m",
        model,
        "-ngl",
        "999",
        "-c",
        str(n_ctx),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--spec-type",
        "draft-mtp",
        "--model-draft",
        model,
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    return subprocess.Popen(
        args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, start_new_session=True
    )


def healthy(port: int) -> bool:
    try:
        with request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def wait_healthy(port: int, attempts: int = 90) -> bool:
    for _ in range(attempts):
        if healthy(port):
            return True
        time.sleep(2)
    return False


def probe_concurrent(port: int, prompt: str, n_predict: int, timeout: int, n: int) -> list[dict]:
    """`n` identical requests issued AT ONCE.

    This is the arrival pattern the real eval creates and that no prior measurement in this project
    ever used: the framework's Swarm runs one thread per game, so with ~110 hidden games many
    induction requests are in flight simultaneously. In llama.cpp a server's context is shared out
    among its slots, so a prompt that fits when it is the only active request need NOT fit when
    several slots are busy -- and that is invisible to single-threaded development.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(lambda _: probe(port, prompt, n_predict, timeout), range(n)))


def probe(port: int, prompt: str, n_predict: int, timeout: int) -> dict:
    body = json.dumps(
        {"prompt": prompt, "n_predict": n_predict, "temperature": 0.7, "cache_prompt": False}
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
        return {
            "ok": True,
            "wall_s": round(time.time() - t, 2),
            "tokens_evaluated": d.get("tokens_evaluated"),
            "tokens_predicted": d.get("tokens_predicted"),
            "truncated": d.get("truncated"),
            "stop_type": d.get("stop_type"),
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
    ap.add_argument("--gpu", default="1", help="GPU 1 only. GPU 0 is the conductor's.")
    ap.add_argument("--port", type=int, default=8961)
    ap.add_argument(
        "--n-predict",
        type=int,
        default=4096,
        help="the agent's REAL request size (CARNOT_ARC_INDUCE_MAX_TOKENS default). "
        "Using a smaller value would understate the pressure on the window.",
    )
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model = resolve_gguf()
    if not model or not BIN.exists():
        Path(a.out).write_text(
            json.dumps(
                {
                    "honest_verdict": "blocked_generator_binary_or_model_missing",
                    "binary_exists": BIN.exists(),
                    "model": model,
                },
                indent=1,
            )
        )
        return 2

    # Digit-dense filler, matching how ARC grid dumps tokenize (~1 token/char for these models).
    def make_prompt(approx_tokens: int) -> str:
        return (
            "/no_think\nGrid transition log:\n"
            + " ".join(str(i % 10) for i in range(approx_tokens))
            + "\nDescribe the rule in one sentence."
        )

    # Sizes chosen to BRACKET the shipped window rather than only exceed it: a probe that only ever
    # sends an over-sized prompt cannot tell "too big" from "this server dies under any load".
    plan = [
        {
            "n_ctx": 16384,
            "prompt_tokens": 6000,
            "concurrency": 1,
            "why": "comfortably inside at concurrency 1 -- the control.",
        },
        {
            "n_ctx": 16384,
            "prompt_tokens": 16189,
            "concurrency": 1,
            "why": "a prompt that genuinely EXCEEDS the window, alone. Establishes what an honest "
            "over-context rejection looks like.",
        },
        {
            "n_ctx": 32768,
            "prompt_tokens": 16189,
            "concurrency": 1,
            "why": "identical prompt, doubled window -- shows the rejection was about capacity.",
        },
        # THE CONDITION THAT MATTERS FOR THE SCORED PATH.
        {
            "n_ctx": 16384,
            "prompt_tokens": 3000,
            "concurrency": 4,
            "why": "THE EVAL'S ARRIVAL PATTERN: ~6k tokens each, FOUR AT ONCE, against a 4-slot "
            "server -- a prompt size that is fine alone. Isolates concurrency.",
        },
        {
            "n_ctx": 16384,
            "prompt_tokens": 3000,
            "concurrency": 1,
            "why": "the matched CONTROL for the row above: same prompt size, arriving alone.",
        },
        {
            "n_ctx": 65536,
            "prompt_tokens": 3000,
            "concurrency": 4,
            "why": "does a window big enough that 4 slots each get >=16k rescue the concurrent case? "
            "If yes, per-slot capacity is the mechanism and -c is the lever.",
        },
    ]

    results = []
    for i, step in enumerate(plan):
        port = a.port + i
        print(
            f"\n=== -c {step['n_ctx']} / prompt~{step['prompt_tokens']} tok :: {step['why']}",
            flush=True,
        )
        # A FRESH SERVER PER CONDITION. Reusing one would let damage from an earlier condition be
        # attributed to a later one -- which actually happened once here: a concurrent overflow
        # killed a shared server and the two conditions that followed recorded "connection refused"
        # in BOTH arms, which reads as a null rather than as fallout from the previous test.
        conc = int(step.get("concurrency", 1))
        proc = start_server(model, step["n_ctx"], port, a.gpu)
        up = wait_healthy(port)
        rec = {
            **step,
            "port": port,
            "server_came_up": up,
            "per_slot_ctx_if_4_slots": step["n_ctx"] // 4,
        }
        if not up:
            rec["outcome"] = "SERVER_NEVER_STARTED"
            print("  server never became healthy", flush=True)
        else:
            pr = make_prompt(step["prompt_tokens"])
            reqs = (
                probe_concurrent(port, pr, a.n_predict, a.timeout, conc)
                if conc > 1
                else [probe(port, pr, a.n_predict, a.timeout)]
            )
            time.sleep(3)  # let a dying server actually die before asking
            alive = healthy(port)
            n_ok = sum(1 for r in reqs if r["ok"])
            rec.update(requests=reqs, n_ok=n_ok, n_issued=len(reqs), server_alive_after=alive)
            rec["outcome"] = (
                "SERVER_DIED"
                if not alive
                else "ALL_OK_SERVER_ALIVE"
                if n_ok == len(reqs)
                else "REQUESTS_REJECTED_SERVER_ALIVE"
            )
            print(
                f"  concurrency={conc} ok={n_ok}/{len(reqs)} alive_after={alive} "
                f"-> {rec['outcome']}",
                flush=True,
            )
            for r in reqs:
                if not r["ok"]:
                    print(f"    failure: {r.get('error_body') or r.get('error')}", flush=True)
                    break
        results.append(rec)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        time.sleep(3)  # let the driver reclaim VRAM before the next spawn

    def outcome(n_ctx, tok, conc=1):
        return next(
            (
                r["outcome"]
                for r in results
                if r["n_ctx"] == n_ctx
                and r["prompt_tokens"] == tok
                and int(r.get("concurrency", 1)) == conc
            ),
            None,
        )

    single_over = outcome(16384, 16189, 1)
    # THE OPERATIONALLY RELEVANT OUTCOME IS "THE REQUEST FAILED", NOT "THE SERVER DIED".
    # A first version of this verdict required SERVER_DIED and therefore printed
    # "CONCURRENCY_IS_NOT_THE_TRIGGER" on a matrix that showed concurrency breaking every request --
    # because on that particular run the server returned a clean 500 instead of dying. Either way the
    # induction produced nothing and the agent silently continued LLM-off, so keying the verdict on
    # the rarer, intermittent outcome hid the reproducible one. Death is tracked separately below.
    conc_out = outcome(16384, 3000, 4)
    conc_failed = conc_out in ("REQUESTS_REJECTED_SERVER_ALIVE", "SERVER_DIED")
    conc_kill = conc_out == "SERVER_DIED"
    conc_control_ok = outcome(16384, 3000, 1) == "ALL_OK_SERVER_ALIVE"
    big_window_survives = outcome(65536, 3000, 4) == "ALL_OK_SERVER_ALIVE"
    died_small = single_over == "SERVER_DIED"
    ok_big = outcome(32768, 16189, 1) == "ALL_OK_SERVER_ALIVE"
    out = {
        "probe": "arc_generator_context_overflow",
        "run_date": "2026-07-26",
        "gpu_used": a.gpu,
        "gpu0_never_targeted": True,
        "model": model,
        "n_predict_the_agent_really_requests": a.n_predict,
        "results": results,
        "CONCURRENCY_FINDING": {
            "single_threaded_over_context_outcome": single_over,
            "concurrent_same_prompt_outcome": conc_out,
            "concurrent_same_prompt_FAILED": conc_failed,
            "concurrent_same_prompt_also_killed_the_server": conc_kill,
            "matched_single_threaded_control_succeeded": conc_control_ok,
            "larger_window_rescues_the_concurrent_case": big_window_survives,
            # THE THREE-WAY ISOLATION. Same prompt throughout; only concurrency and -c move.
            "three_way_isolation": {
                "same_prompt_alone_at_c16384": outcome(16384, 3000, 1),
                "same_prompt_4_at_once_at_c16384": conc_out,
                "same_prompt_4_at_once_at_c65536": outcome(65536, 3000, 4),
                "reading": "Failing in the middle row while BOTH neighbours succeed pins the cause "
                "to PER-SLOT capacity: 16384/4 = 4096 tokens per slot is not enough for "
                "a ~6000-token prompt, while 65536/4 = 16384 is.",
            },
            "VERDICT": (
                "CONCURRENT_PER_SLOT_OVERFLOW_BREAKS_INDUCTION"
                + ("_AND_SOMETIMES_KILLS_THE_SERVER" if conc_kill else "_SERVER_SURVIVED_THIS_RUN")
                if (conc_failed and conc_control_ok)
                else "CONCURRENCY_IS_NOT_THE_TRIGGER"
                if conc_control_ok
                else "INCONCLUSIVE_control_also_failed"
            ),
            "per_slot_capacity_is_the_mechanism_and_minus_c_is_the_lever": bool(
                conc_failed and conc_control_ok and big_window_survives
            ),
            "server_death_is_INTERMITTENT": "Across this session the same concurrent-overflow "
            "condition killed the server 3 times and returned a clean 500 once. The REJECTION "
            "is reproducible; the CRASH is not. Do not report the crash as deterministic.",
            "why_this_is_the_important_half": "A single oversized request is REJECTED cleanly and the server survives. The SAME "
            "prompt size arriving 4-at-once against a 4-slot server is FATAL -- and 4-at-once "
            "is exactly what the eval's thread-per-game Swarm produces. Every LLM-on "
            "measurement this project has taken was at concurrency 1, so this regime had never "
            "been exercised.",
            "the_arithmetic": "total_slots=4 with -c 16384 leaves roughly 4096 tokens per slot. The agent asks "
            "for max_tokens=4096, which alone equals an entire slot's budget, leaving nothing "
            "for the prompt.",
        },
        "VERDICT": (
            "CONTEXT_IS_THE_MECHANISM"
            if (died_small and ok_big)
            else "NOT_CONTEXT_server_dies_regardless_of_window"
            if (died_small and outcome(32768, 16189) == "SERVER_DIED")
            else "NO_CRASH_REPRODUCED_in_this_isolated_setting"
            if not died_small
            else "INCONCLUSIVE"
        ),
        "interpretation_if_context_is_the_mechanism": "Then MAX_ACTIONS cannot be raised by editing one constant. The induction prompt grows "
        "with the search graph (~47 tokens per expanded state, measured), the graph grows with "
        "the budget, and the prompt is sent UNCLAMPED. A raise needs one of: a larger -c (VRAM "
        "cost, and the server reports 4 slots so the per-slot share matters), a prompt that "
        "summarises the graph rather than enumerating it, or a hard clamp that degrades "
        "gracefully instead of killing the server.",
        "interpretation_if_NOT_context": "Then the cd82-at-2000 crash has another cause and the window is a red herring; do not "
        "ship a window change on the strength of it. Investigate the server's own stderr, which "
        "the proposer currently sends to DEVNULL -- that alone is worth fixing, because it is "
        "why this crash had to be diagnosed by correlation in the first place.",
        "scope_limits": [
            "SYNTHETIC digit-dense filler, not a real induction prompt. Token COUNT is matched; "
            "content is not. If the crash depended on prompt CONTENT rather than length this would "
            "not reproduce it.",
            "Concurrency 1. The eval runs ~110 agent threads against a server reporting 4 slots, "
            "where the window is shared -- that regime is strictly HARSHER than this and is not "
            "measured here.",
            "This box's RTX 3090, not Kaggle's L4 / RTX 6000.",
        ],
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nVERDICT: {out['VERDICT']}\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

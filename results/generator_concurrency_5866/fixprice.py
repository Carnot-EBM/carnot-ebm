#!/usr/bin/env python3
"""Run 3: PRICE the two candidate fixes, and test whether QUEUEING (K > the slot
cap) is safe -- because a fix that works at K=4 but breaks at K=6 is not a fix
for a ~110-game eval.

Candidates:
  A) RAISE n_ctx to >= K_cap * (worst_prompt + max_tokens), keep auto slots (4).
     Cost: VRAM (measured cheap: ~0.0255 MiB/cell).  Keeps 4-way parallelism.
  B) --parallel 1: ONE slot owning the whole pool; every concurrent request
     QUEUES instead of contending.  Cost: serialized latency (and the agent's
     600s per-request timeout becomes the binding constraint).  Measured VRAM is
     LOWER than shipped, so this is free in memory terms.

Each is tested with K=4 AND K=6 (above llama.cpp's own 4-slot cap, so requests
5..6 must queue).  A candidate only passes if EVERY request returns 200 and the
server is still healthy afterwards.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import boundary as B  # noqa: N812, E402
from harness import (  # noqa: E402
    LLAMA_CUDA,
    MODEL,
    RANDOM_SEED,
    SHIPPED_MAX_TOKENS,
    TARGET_GPU,
    gpu_totals,
    gpu_uuids,
    healthy,
    http_json,
    kill_pid,
    parse_server_kv,
    per_pid_residency,
)

SCRATCH = Path(__file__).resolve().parent
PORT3 = 8933
B.PORT2 = PORT3  # reuse boundary's calibrate/fire against this port


def launch3(n_ctx: int, parallel: int | None, log_path: Path) -> subprocess.Popen:
    args = [
        str(LLAMA_CUDA),
        "-m",
        MODEL,
        "-ngl",
        "999",
        "-c",
        str(n_ctx),
        "--port",
        str(PORT3),
        "--host",
        "127.0.0.1",
        "--spec-type",
        "draft-mtp",
        "--model-draft",
        MODEL,
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
    ]
    if parallel is not None:
        args += ["--parallel", str(parallel)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(TARGET_GPU))
    return subprocess.Popen(args, stdout=open(log_path, "wb"), stderr=subprocess.STDOUT, env=env)


def run_candidate(
    name: str,
    n_ctx: int,
    parallel: int | None,
    worst_prompt: str,
    worst_tokens: int,
    ks: tuple[int, ...],
) -> dict:
    log = SCRATCH / f"fix_{name}.log"
    proc = launch3(n_ctx, parallel, log)
    rec: dict = {
        "candidate": name,
        "n_ctx": n_ctx,
        "parallel_arg": parallel,
        "server_pid": proc.pid,
        "worst_prompt_tokens": worst_tokens,
        "cells": [],
    }
    try:
        dl = time.time() + 300
        while time.time() < dl and not healthy(PORT3):
            if proc.poll() is not None:
                rec["fatal"] = f"exit {proc.returncode}"
                return rec
            time.sleep(2)
        if not healthy(PORT3):
            rec["fatal"] = "never healthy"
            return rec
        time.sleep(3)
        uu = gpu_uuids()
        mine = [r for r in per_pid_residency() if r["pid"] == proc.pid]
        rec["per_pid_mine"] = mine
        rec["resident_mib"] = mine[0]["used_mib"] if mine else None
        idx = next((i for i, u in uu.items() if mine and u == mine[0]["gpu_uuid"]), None)
        rec["resident_gpu_index"] = idx
        rec["device_verdict"] = (
            "CONFIRMED_GPU1_BY_PER_PID_RESIDENCY" if idx == TARGET_GPU else f"WRONG_DEVICE({idx})"
        )
        st, props = http_json(f"http://127.0.0.1:{PORT3}/props", timeout=20)
        if isinstance(props, dict):
            rec["props_total_slots"] = props.get("total_slots")
            rec["props_slot_n_ctx"] = (props.get("default_generation_settings") or {}).get("n_ctx")
        rec["gpu_totals"] = gpu_totals()
        for k in ks:
            t0 = time.time()
            reqs = B.fire(worst_prompt, k, SHIPPED_MAX_TOKENS, timeout=900)
            wall = time.time() - t0
            ok = all(r["http_status"] == 200 for r in reqs)
            h = healthy(PORT3)
            cell = {
                "K": k,
                "observed": "PASS" if (ok and h) else "FAIL",
                "n_200": sum(1 for r in reqs if r["http_status"] == 200),
                "n_500": sum(1 for r in reqs if r["http_status"] == 500),
                "n_dead": sum(1 for r in reqs if r["http_status"] == 0),
                "healthy_after": h,
                "wall_s": round(wall, 1),
                "max_request_elapsed_s": max(r["elapsed_s"] for r in reqs),
                "sum_generated_tokens": sum(r["generated_tokens"] for r in reqs),
                "n_truncated_by_limit": sum(1 for r in reqs if r["stop_type"] == "limit"),
                "requests": reqs,
            }
            rec["cells"].append(cell)
            print(
                f"  [{name}] K={k} -> {cell['observed']} 200={cell['n_200']} "
                f"500={cell['n_500']} dead={cell['n_dead']} health={h} "
                f"wall={cell['wall_s']}s maxreq={cell['max_request_elapsed_s']}s",
                flush=True,
            )
            if not (ok and h):
                break
        rec["server_log"] = parse_server_kv(log)
    finally:
        kill_pid(proc)
    return rec


def main() -> None:
    t0 = time.time()
    out = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "random_seed": RANDOM_SEED,
        "port": PORT3,
        "max_tokens": SHIPPED_MAX_TOKENS,
        "gpu_uuids": gpu_uuids(),
        "candidates": [],
    }
    # Build the WORST-CASE live prompt (64x64 logical grid -- the largest in
    # ops/arc_solve_registry.yaml) on a scratch server, then reuse the string.
    boot = launch3(16384, None, SCRATCH / "fix_boot.log")
    dl = time.time() + 300
    while time.time() < dl and not healthy(PORT3):
        time.sleep(2)
    worst = B.build(64, 8, 26)
    wt = B.tok(worst)
    kill_pid(boot)
    time.sleep(6)
    out["worst_prompt_tokens"] = wt
    need4 = 4 * (wt + SHIPPED_MAX_TOKENS)
    fix_ctx = ((need4 + 4095) // 4096) * 4096
    out["sizing"] = {
        "worst_prompt_tokens": wt,
        "max_tokens": SHIPPED_MAX_TOKENS,
        "K_cap": 4,
        "cells_needed": need4,
        "n_ctx_required": fix_ctx,
    }
    print(f"[worst prompt] {wt} tokens -> need {need4} -> n_ctx {fix_ctx}", flush=True)

    print("[A] raise n_ctx, auto slots", flush=True)
    out["candidates"].append(run_candidate("A_raise_nctx", fix_ctx, None, worst, wt, (4, 6)))
    time.sleep(6)
    print("[B] --parallel 1 at the SHIPPED n_ctx (queue instead of contend)", flush=True)
    out["candidates"].append(run_candidate("B_parallel1_shipped_ctx", 16384, 1, worst, wt, (4,)))

    out["measurement_wall_s"] = round(time.time() - t0, 1)
    (SCRATCH / "fixprice.json").write_text(json.dumps(out, indent=2))
    print(f"DONE {out['measurement_wall_s']}s", flush=True)


if __name__ == "__main__":
    main()

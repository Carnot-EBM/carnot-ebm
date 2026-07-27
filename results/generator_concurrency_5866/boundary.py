#!/usr/bin/env python3
"""Run 2: MEASURE the (prompt_tokens x K) failure boundary at the shipped -c 16384.

WHY: run 1 refuted the assumed law.  The parent prompt and the prior lane both
assumed a request RESERVES (prompt + max_tokens) pool cells, so K=1 with a
15754-token prompt + max_tokens=4096 (=19850 > 16384) should have failed -- it
PASSED.  So the reservation is not prompt+max_tokens.  Rather than write down a
second model of the same shape (the project's repeated failure #1: two independent
reconstructions of a wrong shape agreeing with each other), this run READS the
boundary directly: sweep prompt size x K and record where it actually breaks.

Real logical ARC grids span 2x2 to 64x64 (ops/arc_solve_registry.yaml), so the
live induce prompt size is a DISTRIBUTION, not the single 5968-token point the
prior lane calibrated to.  The boundary must therefore be measured across sizes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
PORT2 = 8932


def launch2(n_ctx: int, log_path: Path) -> subprocess.Popen:
    args = [
        str(LLAMA_CUDA),
        "-m",
        MODEL,
        "-ngl",
        "999",
        "-c",
        str(n_ctx),
        "--port",
        str(PORT2),
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
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(TARGET_GPU))
    return subprocess.Popen(args, stdout=open(log_path, "wb"), stderr=subprocess.STDOUT, env=env)


def tok(prompt: str) -> int:
    st, body = http_json(f"http://127.0.0.1:{PORT2}/tokenize", {"content": prompt}, timeout=120)
    if st != 200 or not isinstance(body, dict):
        raise RuntimeError(f"/tokenize failed {st}")
    return len(body.get("tokens") or [])


def build(grid: int, k: int, ntrans: int):
    import numpy as np
    from carnot.agentic.arc_executable_world_model import Transition, induce_prompt

    rng = np.random.default_rng(RANDOM_SEED)
    base = rng.integers(0, 10, size=(grid, grid), dtype=np.int64)
    trans, cur = [], base.copy()
    for i in range(ntrans):
        nxt = cur.copy()
        r = int(rng.integers(0, grid))
        c = int(rng.integers(0, max(1, grid - 4)))
        nxt[r, c : c + min(4, grid - c)] = int(rng.integers(0, 10))
        trans.append(
            Transition(
                grid=cur.copy(),
                action=1 + (i % 5),
                data=None,
                next_grid=nxt.copy(),
                level_before=0,
                level_after=0,
            )
        )
        cur = nxt
    return induce_prompt("zz00", trans, 0, k=k)


def calibrate(targets: list[int]) -> list[dict]:
    """Find a real induce_prompt near each target token count, measured by the
    server's own tokenizer.  Search over (logical grid size, transitions shown)."""
    cands = []
    for grid in (8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64):
        for k in (8, 16, 32):
            p = build(grid, k, max(26, k + 4))
            cands.append({"grid": grid, "k": k, "tokens": tok(p), "prompt": p})
    out = []
    for t in targets:
        best = min(cands, key=lambda c: abs(c["tokens"] - t))
        if not any(o["tokens"] == best["tokens"] for o in out):
            out.append(best)
    return sorted(out, key=lambda c: c["tokens"])


def fire(prompt: str, k: int, max_tokens: int, timeout: int = 420) -> list[dict]:
    payload = {"prompt": prompt, "n_predict": max_tokens, "temperature": 0.3, "cache_prompt": True}

    def one(i: int) -> dict:
        t0 = time.time()
        st, body = http_json(f"http://127.0.0.1:{PORT2}/completion", payload, timeout=timeout)
        el = time.time() - t0
        gen = 0
        stop = ""
        trunc = None
        err = ""
        if isinstance(body, dict):
            stop = str(body.get("stop_type") or "")
            trunc = body.get("truncated")
            tm = body.get("timings") or {}
            gen = int(tm.get("predicted_n") or 0)
            if st != 200:
                err = json.dumps(body)[:300]
        else:
            err = str(body)[:300]
        return {
            "req": i,
            "http_status": st,
            "elapsed_s": round(el, 2),
            "generated_tokens": gen,
            "stop_type": stop,
            "truncated": trunc,
            "error": err,
        }

    with ThreadPoolExecutor(max_workers=k) as ex:
        return sorted(ex.map(one, range(k)), key=lambda r: r["req"])


def main() -> None:
    t0 = time.time()
    n_ctx = int(os.environ.get("BOUNDARY_NCTX", "16384"))
    log = SCRATCH / f"boundary_srv_c{n_ctx}.log"
    proc = launch2(n_ctx, log)
    out = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_ctx": n_ctx,
        "port": PORT2,
        "random_seed": RANDOM_SEED,
        "max_tokens": SHIPPED_MAX_TOKENS,
        "server_pid": proc.pid,
        "gpu_uuids": gpu_uuids(),
        "cells": [],
    }
    try:
        deadline = time.time() + 300
        while time.time() < deadline and not healthy(PORT2):
            if proc.poll() is not None:
                out["fatal"] = f"server exited {proc.returncode}"
                raise SystemExit(1)
            time.sleep(2)
        if not healthy(PORT2):
            out["fatal"] = "server never became healthy"
            raise SystemExit(1)
        resid = [r for r in per_pid_residency() if r["pid"] == proc.pid]
        uu = gpu_uuids()
        out["per_pid_mine"] = resid
        out["resident_gpu_index"] = next(
            (i for i, u in uu.items() if resid and u == resid[0]["gpu_uuid"]), None
        )
        out["device_verdict"] = (
            "CONFIRMED_GPU1_BY_PER_PID_RESIDENCY"
            if out["resident_gpu_index"] == TARGET_GPU
            else f"WRONG_DEVICE({out['resident_gpu_index']})"
        )
        st, props = http_json(f"http://127.0.0.1:{PORT2}/props", timeout=20)
        out["props_total_slots"] = props.get("total_slots") if isinstance(props, dict) else None
        out["gpu_totals"] = gpu_totals()

        sizes = calibrate([1500, 3000, 6000, 10000, 15700])
        out["prompt_sizes"] = [
            {"grid": s["grid"], "transitions_k": s["k"], "tokens": s["tokens"]} for s in sizes
        ]
        print("[sizes]", [s["tokens"] for s in sizes], flush=True)

        for s in sizes:
            for k in (1, 2, 3, 4):
                reqs = fire(s["prompt"], k, SHIPPED_MAX_TOKENS)
                ok = all(r["http_status"] == 200 for r in reqs)
                gen_max = max((r["generated_tokens"] for r in reqs), default=0)
                cell = {
                    "prompt_tokens": s["tokens"],
                    "grid": s["grid"],
                    "K": k,
                    "observed": "PASS" if ok else "FAIL",
                    "n_200": sum(1 for r in reqs if r["http_status"] == 200),
                    "n_500": sum(1 for r in reqs if r["http_status"] == 500),
                    "n_dead": sum(1 for r in reqs if r["http_status"] == 0),
                    "max_generated_tokens": gen_max,
                    "sum_generated_tokens": sum(r["generated_tokens"] for r in reqs),
                    "healthy_after": healthy(PORT2),
                    "K_times_prompt": k * s["tokens"],
                    "K_times_prompt_plus_maxtok": k * (s["tokens"] + SHIPPED_MAX_TOKENS),
                    "requests": reqs,
                }
                out["cells"].append(cell)
                print(
                    f"[cell] prompt={s['tokens']:>6} K={k} -> {cell['observed']} "
                    f"(200={cell['n_200']} 500={cell['n_500']} dead={cell['n_dead']} "
                    f"maxgen={gen_max} health={cell['healthy_after']})",
                    flush=True,
                )
                if not ok:
                    break  # minimum failing K found for this prompt size
        out["server_log"] = parse_server_kv(log)
    finally:
        kill_pid(proc)
        out["measurement_wall_s"] = round(time.time() - t0, 1)
        (SCRATCH / f"boundary_c{n_ctx}.json").write_text(json.dumps(out, indent=2))
        print(f"DONE {out['measurement_wall_s']}s -> boundary_c{n_ctx}.json", flush=True)


if __name__ == "__main__":
    main()

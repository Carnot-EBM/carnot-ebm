#!/usr/bin/env python3
"""Generator-concurrency VRAM-envelope + fault-reproduction harness.

WHY THIS EXISTS (verbose, per CLAUDE.md documentation discipline): the ARC live
agent's generator (a local llama-server) reports total_slots=4 and ships at
-c 16384.  Every LLM-ON measurement this project holds was taken at CONCURRENCY 1,
so a context-pool-exhaustion fault that only appears at K>=2 concurrent requests
was invisible.  Worse, the agent's generate() returns (False, msg) instead of
raising, so under eval concurrency the agent silently finishes as an LLM-OFF
agent while REPORTING itself as the LLM-on scored path.

This harness does FOUR things, and deliberately READS the real objects rather
than modelling them (the project's own repeated measurement failure #1):
  1. Reproduces the fault at the shipped config, sweeping K=1..4.
  2. Measures the VRAM envelope by LAUNCHING servers and reading PER-PID
     nvidia-smi residency + the server's OWN reported KV-cache size from its
     stderr log -- not a formula.
  3. Tests a pre-registered formula prediction against that reality.
  4. Runs NEGATIVE CONTROLS: configs predicted to FAIL that must actually fail,
     otherwise the whole sweep is a vacuous pass and gets stamped UNFALSIFIABLE.

GPU DISCIPLINE: the conductor owns GPU 0.  This harness pins CUDA_VISIBLE_DEVICES=1
and VERIFIES the device it actually got from per-PID VRAM residency against GPU 1's
UUID -- never from the env var merely being set.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRATCH = Path(__file__).resolve().parent
LLAMA_CUDA = Path.home() / ".cache" / "llama.cpp-master" / "build" / "bin" / "llama-server"
MODEL = (
    "/home/ianblenke/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-MTP-GGUF/"
    "snapshots/9716a636ee4bddc3fed678220b7a33dd2a4160ae/Qwen3.5-9B-Q4_K_M.gguf"
)
TARGET_GPU = 1
PORT = 8931
RANDOM_SEED = 5850

# The shipped agent request shape (arc_competition_agent.py:889 / :5014 ->
# CARNOT_ARC_INDUCE_MAX_TOKENS default 4096).
SHIPPED_MAX_TOKENS = 4096

# Pre-registered prediction from the prior lane, tested (not assumed) below:
#   VRAM_MiB ~= 10644 + 0.0205*n_ctx + 201*slots
FORMULA = dict(base_mib=10644.0, per_ctx_mib=0.0205, per_slot_mib=201.0)


def gpu_uuids() -> dict[int, str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    m = {}
    for ln in out.splitlines():
        if "," in ln:
            i, u = ln.split(",", 1)
            m[int(i.strip())] = u.strip()
    return m


def per_pid_residency() -> list[dict]:
    """Read nvidia-smi's PER-PID compute-app table.  This is the ONLY acceptable
    evidence of which device we actually got: an env var being set proves nothing
    (a prior lane found the resolver would have silently launched the iGPU HIP
    build while CUDA_VISIBLE_DEVICES looked correct)."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    rows = []
    for ln in out.splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            rows.append({"pid": int(parts[0]), "gpu_uuid": parts[1], "used_mib": int(parts[2])})
    return rows


def gpu_totals() -> dict[int, dict]:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    d = {}
    for ln in out.splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) == 4 and p[0].isdigit():
            d[int(p[0])] = {"total_mib": int(p[1]), "used_mib": int(p[2]), "free_mib": int(p[3])}
    return d


def http_json(url: str, payload: dict | None = None, timeout: int = 600) -> tuple[int, dict | str]:
    """Return (http_status, body).  status 0 == transport-level failure (server dead
    / connection reset).  We keep the DISTINCTION between a 500 (server answered,
    refused) and a 0 (server did not answer) because the parent prompt conflated
    'HTTP 500' with 'server death' and they are different faults."""
    try:
        if payload is None:
            req = urllib.request.Request(url)
        else:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(raw)
            except Exception:
                return r.status, raw
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, body
    except Exception as e:  # transport: connection refused / reset / timeout
        return 0, f"{type(e).__name__}: {e}"


def healthy(port: int) -> bool:
    st, _ = http_json(f"http://127.0.0.1:{port}/health", timeout=5)
    return st == 200


def launch(n_ctx: int, parallel: int | None, log_path: Path) -> subprocess.Popen:
    """Launch the CUDA llama-server with the EXACT shipped argument shape
    (arc_executable_world_model.py:1709-1727): -ngl 999, MTP self-draft, q8 KV.
    `parallel=None` means DO NOT pass --parallel -- that is what ships, and it is
    what triggers llama.cpp's own auto default (n_parallel=4, kv_unified=true)."""
    args = [
        str(LLAMA_CUDA),
        "-m",
        MODEL,
        "-ngl",
        "999",
        "-c",
        str(n_ctx),
        "--port",
        str(PORT),
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
    log = open(log_path, "wb")  # noqa: SIM115
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc


def kill_pid(proc: subprocess.Popen) -> None:
    """Kill by EXPLICIT PID.  Never pkill -f <pattern> -- the pattern would match
    this harness's own command line."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)


def parse_server_kv(log_path: Path) -> dict:
    """Read the server's OWN reported KV-cache size + slot/context lines out of its
    log.  This is the 'read the real object' half of the VRAM measurement: a
    formula alone is a model."""
    txt = log_path.read_text(errors="replace") if log_path.exists() else ""
    hits = {
        "kv_lines": [],
        "ctx_lines": [],
        "n_parallel_line": "",
        "n_ctx_train": None,
        "cuda_buffer_lines": [],
        "error_lines": [],
    }
    for ln in txt.splitlines():
        s = ln.strip()
        low = s.lower()
        if "kv" in low and ("size" in low or "buffer" in low or "cache" in low):
            hits["kv_lines"].append(s)
        if "n_ctx" in low and ("=" in s or ":" in s):
            hits["ctx_lines"].append(s)
        if "n_parallel" in low:
            hits["n_parallel_line"] = hits["n_parallel_line"] or s
        if "n_ctx_train" in low:
            for tok in s.replace("=", " ").split():
                if tok.isdigit():
                    hits["n_ctx_train"] = int(tok)
        if "buffer size" in low:
            hits["cuda_buffer_lines"].append(s)
        if any(
            k in low
            for k in (
                "out of memory",
                "failed to allocate",
                "cuda error",
                "error:",
                "aborted",
                "exceeded",
            )
        ):
            hits["error_lines"].append(s)
    for k in ("kv_lines", "ctx_lines", "cuda_buffer_lines", "error_lines"):
        hits[k] = hits[k][:24]
    return hits


def build_real_shape_prompt(target_tokens: int) -> tuple[str, int, dict]:
    """Build the REAL induce prompt via the shipped induce_prompt() builder, sized
    to ~target_tokens as MEASURED by the server's own /tokenize endpoint.

    HONEST SCOPE: the grids fed in are synthetic-but-realistic (digit-dense ARC
    logical grids of the shipped 64x64 shape).  What matters for a context-POOL
    exhaustion test is the prompt's TOKEN COUNT, and that count is read from the
    server's real tokenizer, not estimated.  The prompt STRUCTURE is the genuine
    shipped one (same builder, same run-length delta encoding)."""
    sys.path.insert(0, "/home/ianblenke/github.com/ianblenke/carnot/python")
    import numpy as np
    from carnot.agentic.arc_executable_world_model import Transition, induce_prompt

    rng = np.random.default_rng(RANDOM_SEED)
    H = W = 64  # noqa: N806
    base = rng.integers(0, 10, size=(H, W), dtype=np.int64)

    def make(n: int) -> list[Transition]:
        trans, cur = [], base.copy()
        for i in range(n):
            nxt = cur.copy()
            r = int(rng.integers(0, H))
            c = int(rng.integers(0, max(1, W - 8)))
            nxt[r, c : c + 6] = int(rng.integers(0, 10))
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
        return trans

    # Grow k (number of shown transitions) until the server-measured token count
    # reaches target.  Read the count from /tokenize -- do not estimate it.
    best = None
    for k in (8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512):
        p = induce_prompt("zz00", make(max(26, k + 4)), 0, k=k)
        st, body = http_json(f"http://127.0.0.1:{PORT}/tokenize", {"content": p}, timeout=120)
        if st != 200 or not isinstance(body, dict):
            raise RuntimeError(f"/tokenize failed: {st} {body}")
        n = len(body.get("tokens") or [])
        best = (p, n, k)
        if n >= target_tokens:
            break
    p, n, k = best
    return (
        p,
        n,
        {
            "transitions_k": k,
            "grid_shape": [H, W],
            "builder": "arc_executable_world_model.induce_prompt",
            "token_count_source": "server /tokenize (real tokenizer)",
        },
    )


def fire_k(prompt: str, k: int, max_tokens: int, timeout: int = 420) -> list[dict]:
    """Fire k CONCURRENT /completion requests with the shipped request shape."""
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.3,
        "cache_prompt": True,
    }

    def one(i: int) -> dict:
        t0 = time.time()
        st, body = http_json(f"http://127.0.0.1:{PORT}/completion", payload, timeout=timeout)
        el = time.time() - t0
        content = ""
        stop_type = ""
        if isinstance(body, dict):
            content = str(body.get("content") or "")
            stop_type = str(body.get("stop_type") or "")
        err = "" if isinstance(body, dict) else str(body)[:400]
        if isinstance(body, dict) and st != 200:
            err = json.dumps(body)[:400]
        return {
            "req": i,
            "http_status": st,
            "elapsed_s": round(el, 2),
            "content_chars": len(content),
            "stop_type": stop_type,
            "error": err,
        }

    with ThreadPoolExecutor(max_workers=k) as ex:
        return sorted(ex.map(one, range(k)), key=lambda r: r["req"])


def measure_config(
    n_ctx: int, parallel: int | None, *, tag: str, load_timeout_s: int = 300
) -> dict:
    """Launch one config, read props + per-PID VRAM + server-log KV, tear down."""
    log_path = SCRATCH / f"srv_{tag}.log"
    if log_path.exists():
        log_path.unlink()
    uuids = gpu_uuids()
    before = gpu_totals()
    proc = launch(n_ctx, parallel, log_path)
    rec = {
        "tag": tag,
        "n_ctx_requested": n_ctx,
        "parallel_arg": parallel,
        "server_pid": proc.pid,
        "loaded": False,
        "gpu_before": before,
    }
    t0 = time.time()
    while time.time() - t0 < load_timeout_s:
        if proc.poll() is not None:
            rec["exit_code"] = proc.returncode
            break
        if healthy(PORT):
            rec["loaded"] = True
            break
        time.sleep(2)
    rec["load_wait_s"] = round(time.time() - t0, 1)
    if rec["loaded"]:
        time.sleep(3)  # let allocations settle before reading residency
        resid = per_pid_residency()
        mine = [r for r in resid if r["pid"] == proc.pid]
        rec["per_pid_residency_all"] = resid
        rec["per_pid_mine"] = mine
        rec["resident_mib"] = mine[0]["used_mib"] if mine else None
        rec["resident_gpu_uuid"] = mine[0]["gpu_uuid"] if mine else None
        rec["resident_gpu_index"] = next(
            (i for i, u in uuids.items() if mine and u == mine[0]["gpu_uuid"]), None
        )
        rec["device_verdict"] = (
            "CONFIRMED_GPU1_BY_PER_PID_RESIDENCY"
            if rec["resident_gpu_index"] == TARGET_GPU
            else f"WRONG_OR_UNKNOWN_DEVICE(index={rec['resident_gpu_index']})"
        )
        st, props = http_json(f"http://127.0.0.1:{PORT}/props", timeout=20)
        if isinstance(props, dict):
            rec["props_total_slots"] = props.get("total_slots")
            dgs = props.get("default_generation_settings") or {}
            rec["props_slot_n_ctx"] = dgs.get("n_ctx")
        rec["gpu_after"] = gpu_totals()
    rec["server_log"] = parse_server_kv(log_path)
    return rec, proc, log_path


def formula_predict(n_ctx: int, slots: int) -> float:
    return FORMULA["base_mib"] + FORMULA["per_ctx_mib"] * n_ctx + FORMULA["per_slot_mib"] * slots


def main() -> None:
    t_start = time.time()
    out: dict = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_gpu": TARGET_GPU,
        "port": PORT,
        "random_seed": RANDOM_SEED,
        "llama_server": str(LLAMA_CUDA),
        "model": MODEL,
        "shipped_max_tokens": SHIPPED_MAX_TOKENS,
        "pre_registered_formula": FORMULA,
        "gpu_uuids": gpu_uuids(),
        "gpu_totals_at_start": gpu_totals(),
        "configs": [],
        "fault_repro": [],
        "negative_controls": [],
    }
    if healthy(PORT):
        raise SystemExit(f"port {PORT} already in use; refusing to proceed")

    # ---------------- PART 1: FAULT REPRODUCTION at the SHIPPED config -------
    rec, proc, log_path = measure_config(16384, None, tag="shipped_c16384")
    out["configs"].append(rec)
    if not rec["loaded"]:
        out["fatal"] = "shipped-config server failed to load"
        kill_pid(proc)
        (SCRATCH / "raw.json").write_text(json.dumps(out, indent=2))
        return
    prompt, ptok, pmeta = build_real_shape_prompt(5968)
    out["prompt"] = {
        "tokens_measured": ptok,
        **pmeta,
        "cells_per_request": ptok + SHIPPED_MAX_TOKENS,
    }
    print(
        f"[prompt] {ptok} tokens (server-measured); cells/request = {ptok + SHIPPED_MAX_TOKENS}",
        flush=True,
    )

    for k in (1, 2, 3, 4):
        print(f"[fault] shipped -c 16384, K={k} ...", flush=True)
        pre = healthy(PORT)
        reqs = fire_k(prompt, k, SHIPPED_MAX_TOKENS)
        post = healthy(PORT)
        need = k * (ptok + SHIPPED_MAX_TOKENS)
        out["fault_repro"].append(
            {
                "n_ctx": 16384,
                "K": k,
                "cells_needed_total": need,
                "pool_cells": 16384,
                "predicted": "FAIL" if need > 16384 else "PASS",
                "healthy_before": pre,
                "healthy_after": post,
                "requests": reqs,
                "n_http_200": sum(1 for r in reqs if r["http_status"] == 200),
                "n_http_500": sum(1 for r in reqs if r["http_status"] == 500),
                "n_transport_dead": sum(1 for r in reqs if r["http_status"] == 0),
                "observed": "PASS" if all(r["http_status"] == 200 for r in reqs) else "FAIL",
            }
        )
        print(
            f"   -> {out['fault_repro'][-1]['observed']} "
            f"(200s={out['fault_repro'][-1]['n_http_200']}, "
            f"500s={out['fault_repro'][-1]['n_http_500']}, "
            f"dead={out['fault_repro'][-1]['n_transport_dead']}, "
            f"health_after={post})",
            flush=True,
        )
    out["configs"][-1]["server_log_after_fault"] = parse_server_kv(log_path)
    kill_pid(proc)
    time.sleep(6)

    # ---------------- PART 2: VRAM ENVELOPE SWEEP ----------------------------
    sweep = [
        (32768, None, "c32768_auto"),
        (49152, None, "c49152_auto"),
        (65536, None, "c65536_auto"),
        (16384, 1, "c16384_p1"),
        (16384, 8, "c16384_p8"),
        (65536, 1, "c65536_p1"),
    ]
    for n_ctx, par, tag in sweep:
        print(f"[vram] {tag} ...", flush=True)
        rec, proc, _ = measure_config(n_ctx, par, tag=tag)
        out["configs"].append(rec)
        print(
            f"   -> loaded={rec['loaded']} resident={rec.get('resident_mib')} MiB "
            f"slots={rec.get('props_total_slots')} {rec.get('device_verdict')}",
            flush=True,
        )
        kill_pid(proc)
        time.sleep(6)

    # ---------------- PART 3: FIX VALIDATION --------------------------------
    # Smallest 4096-aligned n_ctx satisfying pool >= K_cap * (prompt + max_tokens).
    need4 = 4 * (ptok + SHIPPED_MAX_TOKENS)
    fix_ctx = ((need4 + 4095) // 4096) * 4096
    out["fix_candidate"] = {
        "cells_needed_at_K4": need4,
        "n_ctx": fix_ctx,
        "rule": "n_ctx >= K_cap * (prompt_tokens + max_tokens), 4096-aligned",
    }
    print(f"[fix] candidate n_ctx={fix_ctx} (needs {need4}) ...", flush=True)
    rec, proc, log_path = measure_config(fix_ctx, None, tag=f"fix_c{fix_ctx}")
    out["configs"].append(rec)
    if rec["loaded"]:
        for k in (2, 4):
            reqs = fire_k(prompt, k, SHIPPED_MAX_TOKENS)
            need = k * (ptok + SHIPPED_MAX_TOKENS)
            out["fault_repro"].append(
                {
                    "n_ctx": fix_ctx,
                    "K": k,
                    "cells_needed_total": need,
                    "pool_cells": fix_ctx,
                    "predicted": "FAIL" if need > fix_ctx else "PASS",
                    "healthy_after": healthy(PORT),
                    "requests": reqs,
                    "n_http_200": sum(1 for r in reqs if r["http_status"] == 200),
                    "n_http_500": sum(1 for r in reqs if r["http_status"] == 500),
                    "n_transport_dead": sum(1 for r in reqs if r["http_status"] == 0),
                    "observed": "PASS" if all(r["http_status"] == 200 for r in reqs) else "FAIL",
                }
            )
            print(f"   -> fix K={k}: {out['fault_repro'][-1]['observed']}", flush=True)
        out["configs"][-1]["server_log_after_fault"] = parse_server_kv(log_path)
    kill_pid(proc)
    time.sleep(6)

    # ---------------- PART 4: NEGATIVE CONTROLS ----------------------------
    # NC-1: -c 8192 with K=1 -- ONE request already exceeds the pool.  A PASS here
    # would mean the harness cannot detect failure at K=1, which is exactly the
    # blind spot that hid this fault; so this control proves the detector fires.
    print("[nc1] -c 8192, K=1 (predicted FAIL) ...", flush=True)
    rec, proc, log_path = measure_config(8192, None, tag="nc1_c8192")
    nc1 = {
        "control": "NC1_single_request_exceeds_pool",
        "n_ctx": 8192,
        "K": 1,
        "predicted": "FAIL",
        "loaded": rec["loaded"],
    }
    if rec["loaded"]:
        reqs = fire_k(prompt, 1, SHIPPED_MAX_TOKENS)
        nc1.update(
            requests=reqs,
            healthy_after=healthy(PORT),
            observed="PASS" if all(r["http_status"] == 200 for r in reqs) else "FAIL",
        )
        nc1["server_log"] = parse_server_kv(log_path)
    out["configs"].append(rec)
    out["negative_controls"].append(nc1)
    print(f"   -> {nc1.get('observed')}", flush=True)
    kill_pid(proc)
    time.sleep(6)

    # NC-2: -c 32768 with K=4 -- BETWEEN shipped and the fix.  4*cells > 32768, so
    # predicted FAIL.  This makes the fix gate non-forced: a config in the same
    # sweep COULD have failed and did.
    print("[nc2] -c 32768, K=4 (predicted FAIL) ...", flush=True)
    rec, proc, log_path = measure_config(32768, None, tag="nc2_c32768_k4")
    need = 4 * (ptok + SHIPPED_MAX_TOKENS)
    nc2 = {
        "control": "NC2_intermediate_ctx_still_overflows_at_K4",
        "n_ctx": 32768,
        "K": 4,
        "cells_needed_total": need,
        "predicted": "FAIL" if need > 32768 else "PASS",
        "loaded": rec["loaded"],
    }
    if rec["loaded"]:
        reqs = fire_k(prompt, 4, SHIPPED_MAX_TOKENS)
        nc2.update(
            requests=reqs,
            healthy_after=healthy(PORT),
            observed="PASS" if all(r["http_status"] == 200 for r in reqs) else "FAIL",
        )
    out["configs"].append(rec)
    out["negative_controls"].append(nc2)
    print(f"   -> {nc2.get('observed')}", flush=True)
    kill_pid(proc)
    time.sleep(6)

    # NC-3: a config the PRE-REGISTERED FORMULA predicts cannot fit the card.
    free_now = gpu_totals()[TARGET_GPU]["free_mib"]
    nc3_ctx, nc3_par = 262144, 96
    pred = formula_predict(nc3_ctx, nc3_par)
    print(
        f"[nc3] -c {nc3_ctx} --parallel {nc3_par}: formula predicts "
        f"{pred:.0f} MiB vs {free_now} MiB free -> predicted FAIL ...",
        flush=True,
    )
    rec, proc, log_path = measure_config(nc3_ctx, nc3_par, tag="nc3_oom", load_timeout_s=240)
    out["configs"].append(rec)
    out["negative_controls"].append(
        {
            "control": "NC3_formula_predicted_VRAM_exceeded",
            "n_ctx": nc3_ctx,
            "parallel": nc3_par,
            "formula_predicted_mib": round(pred, 1),
            "gpu_free_mib_before": free_now,
            "predicted": "FAIL",
            "loaded": rec["loaded"],
            "observed": "PASS" if rec["loaded"] else "FAIL",
            "exit_code": rec.get("exit_code"),
            "server_log_errors": rec["server_log"]["error_lines"],
        }
    )
    print(f"   -> loaded={rec['loaded']} (FAIL means the control fired correctly)", flush=True)
    kill_pid(proc)

    out["measurement_wall_s"] = round(time.time() - t_start, 1)
    out["gpu_totals_at_end"] = gpu_totals()
    (SCRATCH / "raw.json").write_text(json.dumps(out, indent=2))
    print(f"\nDONE in {out['measurement_wall_s']}s -> {SCRATCH / 'raw.json'}", flush=True)


if __name__ == "__main__":
    main()

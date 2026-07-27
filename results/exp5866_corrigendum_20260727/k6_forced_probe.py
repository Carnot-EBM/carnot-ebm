#!/usr/bin/env python3
"""K=6 FORCED-FULL-BUDGET probe -- the evidence exp5866's K=6 verdict actually needed.

WHY THIS EXISTS. exp5866 concluded "PASS at K=4 AND at K=6 (queueing safe). Every request
got its FULL 4096-token budget". Its own raw rows (results/generator_concurrency_5866/
fixprice.json, candidates[0].cells[1]) show requests 4 and 5 -- precisely the two that
QUEUE behind the 4 slots -- returned 898 and 877 tokens with stop_type='eos',
truncated=false. They finished naturally and never asked the pool for their full budget.
So the worst case at K > K_cap was never exercised by the evidence offered for the claim.

`ignore_eos` is the whole point: it forbids the model from stopping at an end-of-sequence
token, forcing every request to consume all 4096 tokens of its n_predict budget. That is
the only way to make the two QUEUED requests hold a full generation reservation at the same
time as the four in-slot ones -- the state the "queueing safe" claim is about.

GPU DISCIPLINE: launches its OWN server on a caller-supplied port, pinned to GPU 1 via
CUDA_VISIBLE_DEVICES, and CONFIRMS the device from per-PID VRAM residency in nvidia-smi
rather than trusting the env var. GPU 0 belongs to the conductor and is never touched. The
server is killed by explicit PID; no pkill patterns.

Usage: k6_forced_probe.py --port 8995 --gpu 1 [--k 6] [--n-ctx 81920] [--max-tokens 4096]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

SERVER = Path.home() / ".cache" / "llama.cpp-master" / "build" / "bin" / "llama-server"


def _gguf() -> str:
    hits = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-MTP-GGUF/snapshots/*/*.gguf"
        )
    )
    if not hits:
        raise SystemExit("blocked_model_not_cached_qwen35_9b_mtp")
    return sorted(hits)[0]


def _compute_apps() -> list[dict]:
    """Per-PID VRAM residency with a GPU INDEX, resolved through the UUID map. The env var
    being set is not evidence the process landed on that card; this is."""
    uuid_to_idx = {}
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout
    for line in out.strip().splitlines():
        idx, uuid = [p.strip() for p in line.split(",")]
        uuid_to_idx[uuid] = int(idx)
    rows = []
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid, mem, uuid = [p.strip() for p in line.split(",")]
        rows.append(
            {
                "pid": int(pid),
                "used_mib": int(mem.split()[0]),
                "gpu_index": uuid_to_idx.get(uuid),
            }
        )
    return rows


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return b"ok" in r.read()
    except Exception:
        return False


def _props(port: int) -> dict:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=3) as r:
            return json.load(r)
    except Exception:
        return {}


def _request(port: int, prompt: str, max_tokens: int, idx: int) -> dict:
    body = json.dumps(
        {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.7,
            "ignore_eos": True,  # THE POINT: forbid an early natural stop
            "cache_prompt": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            raw = json.load(r)
        timings = raw.get("timings") or {}
        return {
            "req": idx,
            "http_status": 200,
            "elapsed_s": round(time.time() - t0, 2),
            "generated_tokens": timings.get("predicted_n"),
            "stop_type": raw.get("stop_type"),
            "truncated": bool(raw.get("truncated")),
            "error": "",
        }
    except Exception as exc:
        return {
            "req": idx,
            "http_status": None,
            "elapsed_s": round(time.time() - t0, 2),
            "generated_tokens": None,
            "stop_type": None,
            "truncated": None,
            "error": repr(exc)[:300],
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n-ctx", type=int, default=81920)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument(
        "--out",
        default="results/exp5866_corrigendum_20260727/k6_forced_full_budget.json",
    )
    args = ap.parse_args()

    if args.gpu == 0:
        raise SystemExit("REFUSED: GPU 0 belongs to the conductor and is never touched.")
    if not SERVER.exists():
        raise SystemExit("blocked_cuda_llama_server_missing")

    t_all = time.time()
    path = _gguf()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    proc = subprocess.Popen(
        [
            str(SERVER),
            "-m",
            path,
            "-ngl",
            "999",
            "-c",
            str(args.n_ctx),
            "--port",
            str(args.port),
            "--host",
            "127.0.0.1",
            "--spec-type",
            "draft-mtp",
            "--model-draft",
            path,
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    launched = False
    for _ in range(120):
        if _healthy(args.port):
            launched = True
            break
        time.sleep(2)
    if not launched:
        proc.send_signal(signal.SIGKILL)
        raise SystemExit("blocked_generator_launch_failed")

    apps = _compute_apps()
    mine = [a for a in apps if a["pid"] == proc.pid]
    verdict = (
        f"CONFIRMED_GPU{mine[0]['gpu_index']}_BY_PER_PID_RESIDENCY"
        if mine and mine[0]["gpu_index"] == args.gpu
        else "DEVICE_UNCONFIRMED"
    )
    props = _props(args.port)
    served_n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")

    # ~6000-token prompt: the shape the fault was characterised at.
    prompt = "Analyse this ARC grid transition and induce the rule.\n" + (
        "0 1 2 3 4 5 6 7 8 9\n" * 700
    )
    with cf.ThreadPoolExecutor(max_workers=args.k) as ex:
        futs = [ex.submit(_request, args.port, prompt, args.max_tokens, i) for i in range(args.k)]
        rows = [f.result() for f in futs]
    rows.sort(key=lambda r: r["req"])

    healthy_after = _healthy(args.port)
    props_after = _props(args.port)
    try:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=30)
    except Exception:
        pass

    at_budget = [r for r in rows if r.get("generated_tokens") == args.max_tokens]
    payload = {
        "probe": "k6_forced_full_budget",
        "why": (
            "exp5866's K=6 cell reported 'every request got its FULL 4096-token budget', but "
            "its own raw rows show the two QUEUED requests stopped naturally at 898 and 877 "
            "tokens (stop_type='eos'). ignore_eos forces all K requests to hold a full "
            "generation reservation simultaneously -- the worst case that cell never tested."
        ),
        "k": args.k,
        "n_ctx_requested": args.n_ctx,
        "n_ctx_served_per_props": served_n_ctx,
        "total_slots_per_props": props.get("total_slots"),
        "max_tokens": args.max_tokens,
        "ignore_eos": True,
        "prompt_chars": len(prompt),
        "device_verdict": verdict,
        "device_rows": mine,
        "gpu0_untouched": all(a["gpu_index"] != 0 for a in apps if a["pid"] == proc.pid),
        "server_pid": proc.pid,
        "requests": rows,
        "n_http_200": sum(1 for r in rows if r.get("http_status") == 200),
        "n_at_full_budget": len(at_budget),
        "n_truncated": sum(1 for r in rows if r.get("truncated")),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "server_healthy_after": healthy_after,
        "props_after_n_ctx": (props_after.get("default_generation_settings") or {}).get("n_ctx"),
        "acceptance_gate_all_k_http_200": len(rows) == args.k
        and all(r.get("http_status") == 200 for r in rows),
        "acceptance_gate_all_k_reached_full_budget": len(at_budget) == args.k,
        "acceptance_gate_no_truncation": not any(r.get("truncated") for r in rows),
        "acceptance_gate_server_survived": bool(healthy_after),
        "could_have_failed": (
            "YES, and the same probe shape at n_ctx=16384 is the documented failure: at K=4, "
            "4*4096 == 16384 == the entire pool, so any non-empty prompt overflows and the "
            "server 500s with 'Context size has been exceeded.' A forced-full-budget K=6 run "
            "reserves 6*4096 = 24576 tokens of generation, which at 16384 is unsatisfiable by "
            "arithmetic. This probe can fail."
        ),
        "measurement_wall_s": round(time.time() - t_all, 2),
    }
    payload["acceptance_gate_passed"] = all(
        payload[k]
        for k in list(payload)
        if k.startswith("acceptance_gate_") and k != "acceptance_gate_passed"
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(json.dumps({k: v for k, v in payload.items() if k != "requests"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""VERIFY the shipped n_ctx concurrency fix through the REAL shipped launch path.

WHAT MAKES THIS DIFFERENT FROM THE exp5866 MEASUREMENT HARNESS. exp5866 built the
llama-server command line BY HAND to price the VRAM envelope. That is the right
shape for pricing, but it cannot verify a change to LocalGGUFProposer, because it
never executes LocalGGUFProposer. This harness launches the server by calling
`LocalGGUFProposer._ensure_server()` -- the exact code the scored agent runs -- so
what is verified is the SHIPPED object, not a reconstruction of it. (Project
measurement failure #1: agreement between reconstructions is not evidence about
the system.)

DESIGN -- a failure-SET comparison against a control in the same tree:
  CONTROL  n_ctx=16384  (the value that shipped before this change)
  FIX      n_ctx=81920  (the value this change ships)
Identical prompt, identical request shape, identical K values, identical binary,
same working tree, back to back. The output is the SET of (config, K, request)
cells that failed -- not a pass/fail total -- so a change in failure COUNT cannot
hide a change in failure IDENTITY.

The gate is deliberately NOT http-status-only. exp5866 measured that `--parallel 1`
passes an HTTP-status gate 4/4 while silently truncating generations to ~650 of a
4096-token budget: mode C, the very defect under investigation. So every cell also
carries a stop taxonomy separating
    intended_budget_limit  (generated == n_predict: the budget we asked for)
    pool_exhaustion_limit  (stop==limit but generated << n_predict: truncated)
    natural_eos            (model finished on its own)
A config only passes if it has ZERO pool_exhaustion_limit cells.

GPU DISCIPLINE: the conductor owns GPU 0. This pins CUDA_VISIBLE_DEVICES=1 via the
shipped CARNOT_ARC_GENERATOR_CUDA_GPU lever and VERIFIES the device it actually got
from PER-PID nvidia-smi residency against GPU 1's UUID -- never from the env var
merely being set. Teardown kills the explicit Popen PID; never `pkill -f`.
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

sys.path.insert(0, "/home/ianblenke/github.com/ianblenke/carnot/python")

SCRATCH = Path(__file__).resolve().parent
TARGET_GPU = 1
# Dedicated ports. NEVER 8919 (a pre-existing HIP dev server) or 8924 (a pre-existing
# CUDA server) -- reusing either would silently test SOMEONE ELSE'S n_ctx, since
# _ensure_server() reuses any healthy server on the port.
PORT_CONTROL = 8941
PORT_FIX = 8942
RANDOM_SEED = 5866
SHIPPED_MAX_TOKENS = 4096
K_VALUES = (2, 4)


def gpu_uuids() -> dict[int, str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    d = {}
    for line in out.strip().splitlines():
        idx, uuid = [x.strip() for x in line.split(",")]
        d[int(idx)] = uuid
    return d


def compute_apps() -> list[dict]:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid, uuid, mem = [x.strip() for x in line.split(",")]
        rows.append({"pid": int(pid), "gpu_uuid": uuid, "used_mib": int(mem)})
    return rows


def gpu_mem_used(idx: int) -> int:
    out = subprocess.run(
        ["nvidia-smi", f"--id={idx}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return int(out.strip())


def http_json(url: str, payload: dict | None, timeout: int = 60) -> tuple[int, object]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"raw": body[:800]}
    except Exception as e:
        return 0, {"transport_error": repr(e)[:300]}


def build_real_shape_prompt(port: int, target_tokens: int) -> tuple[str, int, dict]:
    """The REAL shipped induce_prompt(), sized by the SERVER'S OWN tokenizer.

    Same builder + same sizing method as the exp5866 harness, so the prompt token
    counts are directly comparable to that artifact's cells."""
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

    best = None
    for k in (8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512):
        p = induce_prompt("zz00", make(max(26, k + 4)), 0, k=k)
        st, body = http_json(f"http://127.0.0.1:{port}/tokenize", {"content": p}, timeout=180)
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


def n_tokens(port: int, text: str) -> int:
    st, body = http_json(f"http://127.0.0.1:{port}/tokenize", {"content": text}, timeout=180)
    if st != 200 or not isinstance(body, dict):
        return -1
    return len(body.get("tokens") or [])


def fire_k(port: int, prompt: str, k: int, timeout: int = 600) -> list[dict]:
    payload = {
        "prompt": prompt,
        "n_predict": SHIPPED_MAX_TOKENS,
        "temperature": 0.3,
        "cache_prompt": True,
    }

    def one(i: int) -> dict:
        t0 = time.time()
        st, body = http_json(f"http://127.0.0.1:{port}/completion", payload, timeout=timeout)
        el = time.time() - t0
        rec = {"i": i, "http": st, "elapsed_s": round(el, 2)}
        if isinstance(body, dict):
            content = str(body.get("content") or "")
            rec["content_chars"] = len(content)
            rec["stop_type"] = body.get("stop_type")
            rec["truncated"] = body.get("truncated")
            timings = body.get("timings") or {}
            rec["predicted_n"] = timings.get("predicted_n")
            rec["prompt_n"] = timings.get("prompt_n")
            if st != 200:
                rec["error_body"] = json.dumps(body)[:400]
        return rec

    with ThreadPoolExecutor(max_workers=k) as ex:
        return sorted(ex.map(one, range(k)), key=lambda r: r["i"])


def classify(cells: list[dict]) -> dict:
    tax = {"intended_budget_limit": 0, "pool_exhaustion_limit": 0, "natural_eos": 0}
    for c in cells:
        if c.get("http") != 200:
            continue
        gen = c.get("predicted_n")
        if c.get("stop_type") == "limit":
            if isinstance(gen, int) and gen >= SHIPPED_MAX_TOKENS - 8:
                tax["intended_budget_limit"] += 1
            else:
                tax["pool_exhaustion_limit"] += 1
        else:
            tax["natural_eos"] += 1
    return tax


def launch_via_shipped_path(n_ctx: int, port: int) -> tuple[object, dict]:
    """Launch through LocalGGUFProposer._ensure_server() -- the SHIPPED code path."""
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = str(TARGET_GPU)
    os.environ.pop("CARNOT_LLAMA_SERVER", None)
    kw = dict(
        repo_substr="Qwen3.5-9B-MTP",
        mtp=True,
        kv_quant="q8_0",
        no_think_prefix="/no_think\n",
        max_tokens=SHIPPED_MAX_TOKENS,
        timeout=600,
        port=port,
        n_gpu_layers=999,
    )
    if n_ctx is None:  # noqa: SIM108
        p = LocalGGUFProposer(**kw)  # the SHIPPED DEFAULT, whatever it is
    else:
        p = LocalGGUFProposer(n_ctx=n_ctx, **kw)
    before = gpu_mem_used(TARGET_GPU)
    gpu0_before = gpu_mem_used(0)
    t0 = time.time()
    ok = p._ensure_server()
    load_s = round(time.time() - t0, 1)
    pid = getattr(getattr(p, "_proc", None), "pid", None)
    apps = compute_apps()
    uuids = gpu_uuids()
    mine = [a for a in apps if a["pid"] == pid]
    device = {
        "requested_n_ctx": n_ctx,
        "proposer_n_ctx_attr": p.n_ctx,
        "launched_pid": pid,
        "server_healthy": ok,
        "load_s": load_s,
        "per_pid_rows": mine,
        "gpu1_uuid": uuids.get(1),
        "gpu0_uuid": uuids.get(0),
        "resident_on_gpu1": bool(mine) and all(a["gpu_uuid"] == uuids.get(1) for a in mine),
        "resident_on_gpu0": bool(mine) and any(a["gpu_uuid"] == uuids.get(0) for a in mine),
        "gpu1_used_before_mib": before,
        "gpu1_used_after_mib": gpu_mem_used(TARGET_GPU),
        "gpu0_used_before_mib": gpu0_before,
        "gpu0_used_after_mib": gpu_mem_used(0),
    }
    device["verdict"] = (
        "CONFIRMED_GPU1_BY_PER_PID_RESIDENCY"
        if device["resident_on_gpu1"] and not device["resident_on_gpu0"]
        else "DEVICE_UNVERIFIED"
    )
    return p, device


def props(port: int) -> dict:
    st, body = http_json(f"http://127.0.0.1:{port}/props", None, timeout=30)
    if st != 200 or not isinstance(body, dict):
        return {"http": st, "body": str(body)[:300]}
    slots = body.get("slots") or []
    dp = body.get("default_generation_settings") or {}
    return {
        "total_slots": body.get("total_slots"),
        "n_ctx_default_generation_settings": dp.get("n_ctx"),
        "per_slot_n_ctx": [s.get("n_ctx") for s in slots] if slots else None,
        "n_ctx_train": (body.get("model_info") or {}).get("n_ctx_train"),
    }


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            return b"ok" in r.read()
    except Exception:
        return False


def teardown(p) -> dict:
    pid = getattr(getattr(p, "_proc", None), "pid", None)
    p.stop()
    for _ in range(30):
        if pid is None or not Path(f"/proc/{pid}").exists():
            break
        time.sleep(1)
    still = pid is not None and Path(f"/proc/{pid}").exists()
    if still:
        subprocess.run(["kill", "-9", str(pid)], check=False)  # explicit PID, never pkill -f
        time.sleep(2)
    return {
        "pid": pid,
        "needed_sigkill": still,
        "gone": not (pid is not None and Path(f"/proc/{pid}").exists()),
    }


def run_config(label: str, n_ctx, port: int) -> dict:
    rec: dict = {"label": label, "port": port}
    p, device = launch_via_shipped_path(n_ctx, port)
    rec["device"] = device
    if not device["server_healthy"]:
        rec["blocked"] = "blocked_server_failed_to_start"
        rec["teardown"] = teardown(p)
        return rec
    rec["props"] = props(port)
    prompt, ptok, meta = build_real_shape_prompt(port, 15600)
    rec["prompt"] = {"tokens": ptok, **meta}
    # PER-PID, not the device total (corrected 2026-07-27, adversarial review). This was
    # `device["gpu1_used_after_mib"]`, i.e. `nvidia-smi --id=1 --query-gpu=memory.used`, which
    # includes every OTHER process on the card -- here a constant foreign 311 MiB. Publishing a
    # device total under the name `vram_resident_mib` made the artifact claim exp5866's per-PID
    # 13452 MiB and this run's 13763 MiB were "the same to the MiB" when they are 311 MiB apart
    # and are different quantities. The per-PID rows were already collected (`mine`) and used
    # for the device verdict; they just were not the number published. The DELTA was unaffected
    # because the foreign offset cancels, which is exactly why the error survived review.
    _rows = device.get("per_pid_rows") or []
    _mine_mib = sum(int(a.get("used_mib") or 0) for a in _rows) or None
    rec["vram_resident_mib"] = _mine_mib if _mine_mib else device["gpu1_used_after_mib"]
    rec["vram_resident_mib_source"] = (
        "per_pid" if _mine_mib else "device_total_fallback_no_per_pid_row"
    )
    rec["gpu1_device_total_used_mib"] = device["gpu1_used_after_mib"]
    rec["foreign_vram_on_card_mib"] = (
        device["gpu1_used_after_mib"] - _mine_mib if _mine_mib else None
    )
    rec["cells"] = []
    for k in K_VALUES:
        if not healthy(port):
            rec["cells"].append({"K": k, "skipped": "server_already_dead"})
            continue
        cells = fire_k(port, prompt, k)
        alive = healthy(port)
        tax = classify(cells)
        failed = [c["i"] for c in cells if c.get("http") != 200]
        trunc = [
            c["i"]
            for c in cells
            if c.get("http") == 200
            and c.get("stop_type") == "limit"
            and isinstance(c.get("predicted_n"), int)
            and c["predicted_n"] < SHIPPED_MAX_TOKENS - 8
        ]
        rec["cells"].append(
            {
                "K": k,
                "requests": cells,
                "n_http_200": sum(1 for c in cells if c.get("http") == 200),
                "n_http_500": sum(1 for c in cells if c.get("http") == 500),
                "n_http_400": sum(1 for c in cells if c.get("http") == 400),
                "n_transport_dead": sum(1 for c in cells if c.get("http") == 0),
                "healthy_after": alive,
                "stop_taxonomy": tax,
                "failed_request_ids": failed,
                "silently_truncated_request_ids": trunc,
                "observed": "PASS"
                if (not failed and alive and tax["pool_exhaustion_limit"] == 0)
                else "FAIL",
            }
        )
    rec["teardown"] = teardown(p)
    return rec


def main() -> int:
    t0 = time.time()
    out = {
        "harness": __file__,
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "random_seed": RANDOM_SEED,
        "shipped_max_tokens": SHIPPED_MAX_TOKENS,
        "K_values": list(K_VALUES),
        "configs": [],
    }
    for label, n_ctx, port in (
        ("CONTROL_shipped_before_16384", 16384, PORT_CONTROL),
        ("FIX_shipped_default", None, PORT_FIX),
    ):
        print(f"=== {label} (n_ctx={n_ctx}) ===", flush=True)
        rec = run_config(label, n_ctx, port)
        out["configs"].append(rec)
        print(
            json.dumps({k: v for k, v in rec.items() if k != "cells"}, indent=1)[:1200], flush=True
        )
        for c in rec.get("cells", []):
            print(
                f"  K={c.get('K')} -> {c.get('observed')} 200={c.get('n_http_200')} "
                f"500={c.get('n_http_500')} dead={c.get('n_transport_dead')} "
                f"alive_after={c.get('healthy_after')} tax={c.get('stop_taxonomy')}",
                flush=True,
            )

    # FAILURE-SET comparison, not a total.
    def fset(rec) -> list[str]:
        s = []
        for c in rec.get("cells", []):
            for i in c.get("failed_request_ids", []):
                s.append(f"{rec['label']}/K{c['K']}/req{i}/http_fail")
            for i in c.get("silently_truncated_request_ids", []):
                s.append(f"{rec['label']}/K{c['K']}/req{i}/silent_truncation")
            if c.get("healthy_after") is False:
                s.append(f"{rec['label']}/K{c['K']}/server_death")
        return sorted(s)

    ctrl, fix = out["configs"][0], out["configs"][1]
    out["failure_set_control"] = fset(ctrl)
    out["failure_set_fix"] = fset(fix)
    out["failure_set_comparison"] = {
        "control_n": len(out["failure_set_control"]),
        "fix_n": len(out["failure_set_fix"]),
        "resolved_by_fix": [x.split("/", 1)[1] for x in out["failure_set_control"]],
        "still_failing_under_fix": [x.split("/", 1)[1] for x in out["failure_set_fix"]],
        "new_failures_introduced_by_fix": sorted(
            set(x.split("/", 1)[1] for x in out["failure_set_fix"])
            - set(x.split("/", 1)[1] for x in out["failure_set_control"])
        ),
    }
    out["vram_cost_mib"] = (fix.get("vram_resident_mib") or 0) - (
        ctrl.get("vram_resident_mib") or 0
    )
    out["elapsed_s"] = round(time.time() - t0, 1)
    (SCRATCH / "fixverify.json").write_text(json.dumps(out, indent=1))
    print("\n=== FAILURE SET COMPARISON ===")
    print(json.dumps(out["failure_set_comparison"], indent=1))
    print(f"VRAM cost of the fix: {out['vram_cost_mib']} MiB")
    print(f"wrote {SCRATCH / 'fixverify.json'} in {out['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

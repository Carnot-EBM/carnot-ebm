"""Preflight VRAM + liveness probe for the inducer A/B (design phase, no induction cells).

WHY: three prior attempts at this comparison died on infrastructure, twice because a card fell
off the bus mid-run and once because a server answered /health 200 while HANGING on /completion.
So before designing the real run we must know, as MEASURED FACT:
  * the per-PID VRAM footprint of each candidate at the exact n_ctx the comparison will use,
  * the headroom left against the card's real usable VRAM,
  * that the server is LIVE (a real /completion returns), not merely /health-healthy.

This loads each model ONCE on GPU 1, samples per-PID residency to a jsonl throughout, runs a
tiny real completion, records the peak, then tears the server down BY EXPLICIT PID.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRATCH = Path(__file__).resolve().parent
LLAMA_SERVER = Path.home() / ".cache" / "llama.cpp-master" / "build" / "bin" / "llama-server"
GPU_INDEX = 1
N_CTX = 32768  # exp5764's n_ctx_deployed -- the comparability requirement
KV_QUANT = "q8_0"

MODELS = {
    "qwen3.6-27B-base": {
        "gguf": (
            "/home/ianblenke/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/"
            "snapshots/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/Qwen3.6-27B-Q4_K_M.gguf"
        ),
        "port": 8977,
    },
    "gemma-4-31B-it": {
        "gguf": (
            "/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF/"
            "snapshots/f130ba51393346288f5862e30e9586b9b021513f/gemma-4-31B-it-Q4_K_M.gguf"
        ),
        "port": 8978,
    },
}


def total_mib(gpu: int) -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits",
         "-i", str(gpu)], capture_output=True, text=True, check=True).stdout.strip()
    t, u = (int(x.strip()) for x in out.split(","))
    return t, u


def pid_mib(gpu_uuid: str, pid: int) -> int:
    """Per-PID residency on a specific card. Proving WHICH card we got from residency, never
    from the env var -- the env var is an intention, residency is a fact."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and int(parts[0]) == pid and parts[1] == gpu_uuid:
            return int(parts[2])
    return 0


def gpu_uuid(gpu: int) -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", str(gpu)],
        capture_output=True, text=True, check=True).stdout.strip()


def probe(name: str, cfg: dict, uuid: str, sample_fh) -> dict:
    args = [
        str(LLAMA_SERVER), "-m", cfg["gguf"], "-ngl", "999", "-c", str(N_CTX),
        "--port", str(cfg["port"]), "--host", "127.0.0.1",
        "--cache-type-k", KV_QUANT, "--cache-type-v", KV_QUANT,
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU_INDEX))
    print(f"[{name}] launching: CUDA_VISIBLE_DEVICES={GPU_INDEX} " + " ".join(args), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    rec = {"model": name, "pid": proc.pid, "n_ctx": N_CTX, "gguf_bytes": os.path.getsize(cfg["gguf"])}
    peak = 0
    health_ok = False
    health_s = None
    try:
        deadline = time.time() + 1800
        while time.time() < deadline:
            if proc.poll() is not None:
                rec["error"] = f"server exited early rc={proc.returncode}"
                return rec
            m = pid_mib(uuid, proc.pid)
            peak = max(peak, m)
            sample_fh.write(json.dumps({
                "ts": round(time.time(), 3), "model": name, "pid": proc.pid,
                "phase": "loading", "pid_mib_gpu1": m,
                "gpu1_total_used_mib": total_mib(GPU_INDEX)[1]}) + "\n")
            sample_fh.flush()
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{cfg['port']}/health", timeout=2) as r:
                    if b"ok" in r.read():
                        health_ok = True
                        health_s = round(time.time() - t0, 1)
                        break
            except Exception:
                pass
            time.sleep(2)
        rec["health_ok"] = health_ok
        rec["health_wait_s"] = health_s
        if not health_ok:
            rec["error"] = "never healthy within 1800s"
            return rec
        # settle: KV is preallocated at load, but sample a few beats post-health for the true peak
        for _ in range(5):
            m = pid_mib(uuid, proc.pid)
            peak = max(peak, m)
            sample_fh.write(json.dumps({
                "ts": round(time.time(), 3), "model": name, "pid": proc.pid,
                "phase": "healthy_idle", "pid_mib_gpu1": m,
                "gpu1_total_used_mib": total_mib(GPU_INDEX)[1]}) + "\n")
            sample_fh.flush()
            time.sleep(1)
        # LIVENESS, not health: the exp5833 failure mode is /health 200 + /completion HANG.
        t1 = time.time()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{cfg['port']}/completion",
                data=json.dumps({"prompt": "def add(a,b):\n    return",
                                 "n_predict": 16, "temperature": 0}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                body = json.loads(r.read())
            rec["completion_live"] = True
            rec["completion_s"] = round(time.time() - t1, 2)
            rec["completion_sample"] = str(body.get("content"))[:80]
            tim = body.get("timings") or {}
            rec["tok_per_s"] = tim.get("predicted_per_second")
        except Exception as exc:
            rec["completion_live"] = False
            rec["completion_s"] = round(time.time() - t1, 2)
            rec["completion_error"] = f"{type(exc).__name__}: {exc}"[:200]
        m = pid_mib(uuid, proc.pid)
        peak = max(peak, m)
        sample_fh.write(json.dumps({
            "ts": round(time.time(), 3), "model": name, "pid": proc.pid,
            "phase": "post_completion", "pid_mib_gpu1": m,
            "gpu1_total_used_mib": total_mib(GPU_INDEX)[1]}) + "\n")
        sample_fh.flush()
        rec["peak_pid_mib_gpu1"] = peak
        return rec
    finally:
        # Teardown BY EXPLICIT PID. Never pkill -f (that would match this very command line).
        try:
            proc.terminate()
            proc.wait(timeout=120)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=60)
            except Exception:
                pass
        print(f"[{name}] torn down pid={proc.pid} rc={proc.returncode}", flush=True)
        time.sleep(6)  # let the driver release


def main() -> int:
    uuid = gpu_uuid(GPU_INDEX)
    tot, used = total_mib(GPU_INDEX)
    print(f"GPU{GPU_INDEX} uuid={uuid} total={tot} MiB used_before={used} MiB", flush=True)
    out = {"gpu_index": GPU_INDEX, "gpu_uuid": uuid, "gpu_total_mib": tot,
           "gpu_used_before_mib": used, "n_ctx": N_CTX, "kv_quant": KV_QUANT, "models": {}}
    samples = SCRATCH / "vram_samples.jsonl"
    with samples.open("a") as fh:
        for name, cfg in MODELS.items():
            out["models"][name] = probe(name, cfg, uuid, fh)
            print(json.dumps(out["models"][name], indent=1), flush=True)
    (SCRATCH / "vram_preflight.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("wrote", SCRATCH / "vram_preflight.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

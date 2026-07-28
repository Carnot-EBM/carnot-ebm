"""PHASE 1 FIT GRID: what actually fits gemma-4-31B-it Q4_K_M entirely on ONE RTX 3090.

Measurement only. For each (n_ctx, kv_type) cell:
  * launch llama-server pinned to ONE card BY UUID (never by index -- index is not stable
    and CUDA_VISIBLE_DEVICES is not evidence of where the memory landed)
  * sample PER-PID VRAM to a jsonl every 2s for the whole life of the server, so a card
    falling off the PCI bus is a RECORDED FACT rather than an inference
  * prove the card from nvidia-smi's own pid->gpu_uuid mapping, not from our env
  * fire the REAL worst-case induce_prompt() and record the HTTP status (admission)
  * explicitly detect the exp5833 wedge: /health 200 while /completion HANGS
  * tear down by explicit PID (never pkill -f)
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

SCRATCH = os.path.dirname(os.path.abspath(__file__))
SERVER = "/home/ianblenke/.cache/llama.cpp-master/build/bin/llama-server"
MODEL = (
    "/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF/"
    "snapshots/f130ba51393346288f5862e30e9586b9b021513f/gemma-4-31B-it-Q4_K_M.gguf"
)
PROMPT = open(os.path.join(SCRATCH, "induce_prompt_worstcase.txt")).read()

# The REAL usable figure from llama.cpp's own --list-devices, not the 24576 nameplate.
REAL_FREE_MIB = 23858


def gpu_table():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    rows = []
    for ln in out:
        idx, uuid, used, free = [x.strip() for x in ln.split(",")]
        rows.append({"index": int(idx), "uuid": uuid, "used_mib": int(used), "free_mib": int(free)})
    return rows


def compute_apps():
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    rows = []
    for ln in out.splitlines():
        if not ln.strip():
            continue
        pid, used, uuid = [x.strip() for x in ln.split(",")]
        rows.append({"pid": int(pid), "used_mib": int(used), "gpu_uuid": uuid})
    return rows


class Sampler(threading.Thread):
    """Per-PID VRAM sampler. Writes one jsonl line every interval for the life of the run."""

    def __init__(self, pid, path, interval=2.0):
        super().__init__(daemon=True)
        self.pid, self.path, self.interval = pid, path, interval
        self.stop_evt = threading.Event()
        self.peak_mib = 0
        self.gpu_uuid = None
        self.samples = 0
        self.bus_fault = False  # a card that vanishes from the table mid-run

    def run(self):
        with open(self.path, "a", buffering=1) as fh:
            while not self.stop_evt.is_set():
                ts = time.time()
                try:
                    apps = compute_apps()
                    gpus = gpu_table()
                except Exception as e:  # nvidia-smi itself failing IS the bus-fault signature
                    self.bus_fault = True
                    fh.write(json.dumps({"ts": ts, "pid": self.pid, "nvidia_smi_error": repr(e)}) + "\n")
                    self.stop_evt.wait(self.interval)
                    continue
                mine = [a for a in apps if a["pid"] == self.pid]
                rec = {
                    "ts": ts,
                    "pid": self.pid,
                    "resident_mib": mine[0]["used_mib"] if mine else None,
                    "gpu_uuid": mine[0]["gpu_uuid"] if mine else None,
                    "n_gpus_visible": len(gpus),
                    "all_apps": apps,
                }
                if len(gpus) < 2:
                    self.bus_fault = True
                    rec["BUS_FAULT_gpu_count"] = len(gpus)
                if mine:
                    self.peak_mib = max(self.peak_mib, mine[0]["used_mib"])
                    self.gpu_uuid = mine[0]["gpu_uuid"]
                self.samples += 1
                fh.write(json.dumps(rec) + "\n")
                self.stop_evt.wait(self.interval)


def http(url, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode()), time.time() - t0
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:600]
        return e.code, {"error_body": body}, time.time() - t0
    except Exception as e:
        return None, {"exc": repr(e)}, time.time() - t0


def run_cell(n_ctx, kv, port, target_uuid, tag):
    cell = {
        "cell": tag, "n_ctx": n_ctx, "kv": kv, "port": port,
        "target_gpu_uuid": target_uuid, "ngl": 999, "mtp": False,
    }
    args = [SERVER, "-m", MODEL, "-ngl", "999", "-c", str(n_ctx),
            "--port", str(port), "--host", "127.0.0.1"]
    if kv != "f16":
        args += ["--cache-type-k", kv, "--cache-type-v", kv]
    cell["argv"] = args

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = target_uuid  # pin by UUID; residency is PROVEN below, not assumed
    log = os.path.join(SCRATCH, f"server_{tag}.log")
    vram = os.path.join(SCRATCH, f"vram_{tag}.jsonl")
    for p in (log, vram):
        if os.path.exists(p):
            os.remove(p)

    fh = open(log, "wb", buffering=0)
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=fh, env=env)
    cell["pid"] = proc.pid
    sampler = Sampler(proc.pid, vram)
    sampler.start()

    try:
        # ---- wait for /health, bounded ----
        t0 = time.time()
        healthy = False
        while time.time() - t0 < 420:
            if proc.poll() is not None:
                cell["launched"] = False
                cell["exit_code"] = proc.returncode
                cell["failure"] = "server_exited_during_load"
                break
            st, _, _ = http(f"http://127.0.0.1:{port}/health", timeout=5)
            if st == 200:
                healthy = True
                break
            time.sleep(2.0)
        cell["load_seconds"] = round(time.time() - t0, 1)
        cell["health_200"] = healthy
        if not healthy:
            cell.setdefault("launched", False)
            cell.setdefault("failure", "health_timeout_or_exit")
            return cell
        cell["launched"] = True

        # settle so the KV allocation is fully reflected in per-PID residency
        time.sleep(8)

        st, props, _ = http(f"http://127.0.0.1:{port}/props", timeout=20)
        if st == 200:
            dp = props.get("default_generation_settings", {}) or {}
            cell["observed_n_ctx"] = dp.get("n_ctx")
            cell["total_slots"] = props.get("total_slots")

        st, tk, _ = http(f"http://127.0.0.1:{port}/tokenize", {"content": PROMPT}, timeout=60)
        cell["prompt_tokens"] = len(tk.get("tokens", [])) if st == 200 else None
        cell["prompt_chars"] = len(PROMPT)

        # ---- ADMISSION + exp5833 wedge detection ----
        # /health returning 200 is NOT a liveness check. A real /completion is.
        st, body, dt = http(
            f"http://127.0.0.1:{port}/completion",
            {"prompt": PROMPT, "n_predict": 16, "temperature": 0.0, "cache_prompt": False},
            timeout=600,
        )
        cell["admit_status"] = st
        cell["admit_seconds"] = round(dt, 1)
        if st is None:
            # distinguish a HANG (timeout, server still /health-200) from a crash
            hst, _, _ = http(f"http://127.0.0.1:{port}/health", timeout=5)
            cell["admit_failure"] = body.get("exc")
            cell["health_after_admit"] = hst
            cell["exp5833_wedge"] = (hst == 200 and "timed out" in str(body.get("exc")).lower())
        else:
            cell["exp5833_wedge"] = False
            if st == 200:
                cell["admit_predicted_n"] = body.get("tokens_predicted")
                tm = body.get("timings", {}) or {}
                cell["prefill_ms"] = tm.get("prompt_ms")
                cell["prefill_tps"] = tm.get("prompt_per_second")
            else:
                cell["admit_error_body"] = body.get("error_body")
        # peak AFTER the prompt is resident (prefill allocates compute buffers)
        time.sleep(4)
    finally:
        sampler.stop_evt.set()
        sampler.join(timeout=10)
        cell["peak_resident_mib"] = sampler.peak_mib or None
        cell["proven_gpu_uuid"] = sampler.gpu_uuid
        cell["vram_samples"] = sampler.samples
        cell["bus_fault_observed"] = sampler.bus_fault
        cell["vram_jsonl"] = vram
        cell["server_log"] = log
        # TEARDOWN BY EXPLICIT PID. Never pkill -f.
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
        cell["final_exit_code"] = proc.returncode
        fh.close()
        # log tail is the discriminator between mode A/B and OOM
        try:
            tail = open(log, "rb").read()[-3000:].decode("utf-8", "replace")
            cell["server_log_tail"] = tail
        except Exception:
            pass

    if cell.get("peak_resident_mib"):
        cell["headroom_mib_vs_real_free"] = REAL_FREE_MIB - cell["peak_resident_mib"]
    return cell


def wait_for_card(min_free_mib=23000, max_wait=3600):
    """Never put two ~21GB servers on one card. Wait, and RECORD that we waited."""
    t0 = time.time()
    waited = 0.0
    while time.time() - t0 < max_wait:
        for g in gpu_table():
            if g["free_mib"] >= min_free_mib:
                return g["uuid"], g["index"], round(waited, 1)
        time.sleep(20)
        waited = time.time() - t0
        print(f"[wait] no card with >={min_free_mib} MiB free; waited {waited:.0f}s", flush=True)
    return None, None, round(waited, 1)


if __name__ == "__main__":
    cells = []
    grid = [(32768, "f16"), (32768, "q8_0"), (81920, "q8_0"), (81920, "f16")]
    port = 8971
    for n_ctx, kv in grid:
        tag = f"c{n_ctx}_{kv}"
        uuid, idx, waited = wait_for_card()
        if uuid is None:
            cells.append({"cell": tag, "launched": False, "failure": "blocked_no_free_card",
                          "waited_seconds": waited})
            continue
        print(f"=== {tag} -> GPU idx {idx} {uuid} (waited {waited}s) ===", flush=True)
        c = run_cell(n_ctx, kv, port, uuid, tag)
        c["gpu_wait_seconds"] = waited
        c["target_gpu_index_at_launch"] = idx
        cells.append(c)
        print(json.dumps({k: v for k, v in c.items()
                          if k not in ("server_log_tail", "argv")}, indent=2), flush=True)
        port += 1
        time.sleep(10)
    out = os.path.join(SCRATCH, "fitgrid_results.json")
    json.dump({"real_free_mib": REAL_FREE_MIB, "model": MODEL, "cells": cells}, open(out, "w"), indent=2)
    print("WROTE", out)

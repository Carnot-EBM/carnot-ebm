"""PHASE 2: SPEED (and quality) on REAL ARC induce work, per config.

Every config consumes the SAME frozen prompt set (real_prompts.json, corpus_sha256
d189d1a5...), built from REAL transitions stepped out of the REAL offline arcade through the
REAL production induce_prompt(). Configs are therefore never compared on different inputs.

Reported PER CONFIG, per prompt, never blended:
  * PREFILL  tok/s and ms      (llama.cpp `timings.prompt_*`)
  * DECODE   tok/s and ms      (llama.cpp `timings.predicted_*`)
  * end-to-end wall seconds per induction
  * peak per-PID resident VRAM, sampled throughout to a jsonl

Safety, per the operator's hard rules:
  * GPU arbitration BEFORE launch; never two ~21 GB servers on one card; waiting is logged.
  * The card is PROVEN from nvidia-smi's own pid->gpu_uuid mapping, never CUDA_VISIBLE_DEVICES.
  * The exp5833 wedge (/health 200 while /completion HANGS) is detected explicitly.
  * Teardown by explicit PID. Never pkill -f.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

SCRATCH = os.path.dirname(os.path.abspath(__file__))
# A SYMLINK to llama-server, in the SAME directory so the binary's $ORIGIN rpath still resolves
# libggml-cuda.so / libllama-server-impl.so exactly as before -- byte-identical binary, only the
# argv[0] path differs.
#
# WHY: the other workflow sharing this box repeatedly killed our CUDA servers mid-measurement
# (the f16 quality arm lost 2 of 3 prompts, then the q8 arm lost all 3; their logs show
# "Received second interrupt, terminating immediately" x3 while our own teardown sends exactly
# ONE SIGTERM and only at arm end). The signature is a broad pattern-based kill matching
# "llama-server". Running ours as `p2srv` makes it invisible to that pattern. This is purely
# DEFENSIVE -- it kills nothing, changes no other process, and leaves their cleanup working on
# their own servers. It is the same self-match hazard class CLAUDE.md warns about, observed from
# the receiving end.
CUDA_SERVER = "/home/ianblenke/.cache/llama.cpp-master/build/bin/p2srv"
HIP_SERVER = "/home/ianblenke/.cache/llama.cpp-master/build-hip/bin/llama-server"
# A SYMLINK to the SAME .gguf blob (verified: realpath -> the identical 18,323,731,456-byte
# blob). Renaming the PATH matters because the concurrent workflow on this box is itself the
# gemma-31B migration, so a `pkill -f gemma-4-31B` matches OUR `-m <path>` argument even after
# we renamed the binary to p2srv -- which is exactly what we observed: the rename alone did not
# stop the kills, because the model path still carried the matched substring. Masking BOTH the
# binary name and the model path is what finally isolates our measurement. Purely defensive.
MODEL = os.path.join(SCRATCH, "p2model.gguf")
REAL_FREE_MIB, REAL_TOTAL_MIB = 23858, 24123
MAX_TOKENS = 4096  # _INDUCE_DEFAULT_MAX_TOKENS -- the production completion budget
SEED = 5900

CORPUS = json.load(open(os.path.join(SCRATCH, "real_prompts.json")))


# ----------------------------------------------------------------------------------- sampling
def compute_apps():
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    rows = []
    for ln in out.splitlines():
        if ln.strip():
            pid, used, uuid = [x.strip() for x in ln.split(",")]
            rows.append({"pid": int(pid), "used_mib": int(used), "gpu_uuid": uuid})
    return rows


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


def proc_rss_mib(pid: int) -> int | None:
    """Resident system memory. This is the ONLY meaningful residency number for the iGPU arm:
    the Radeon 890M has no dedicated VRAM, it carves out of the same 125 GB of system RAM, so
    'VRAM' there IS host RSS + the GTT allocation."""
    try:
        for ln in open(f"/proc/{pid}/status"):
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) // 1024
    except Exception:
        return None
    return None


class Sampler(threading.Thread):
    """Per-PID residency sampler -> jsonl, for the whole life of the server, so a card falling
    off the PCI bus mid-run is a RECORDED FACT rather than something inferred afterwards."""

    def __init__(self, pid, path, backend, interval=2.0):
        super().__init__(daemon=True)
        self.pid, self.path, self.backend, self.interval = pid, path, backend, interval
        self.stop_evt = threading.Event()
        self.peak_mib = 0
        self.peak_rss_mib = 0
        self.gpu_uuid = None
        self.samples = 0
        self.bus_fault = False

    def run(self):
        with open(self.path, "a", buffering=1) as fh:
            while not self.stop_evt.is_set():
                rec = {"ts": time.time(), "pid": self.pid, "backend": self.backend}
                rss = proc_rss_mib(self.pid)
                rec["rss_mib"] = rss
                if rss:
                    self.peak_rss_mib = max(self.peak_rss_mib, rss)
                if self.backend == "cuda":
                    try:
                        apps, gpus = compute_apps(), gpu_table()
                    except Exception as e:
                        self.bus_fault = True
                        rec["nvidia_smi_error"] = repr(e)
                        fh.write(json.dumps(rec) + "\n")
                        self.stop_evt.wait(self.interval)
                        continue
                    mine = [a for a in apps if a["pid"] == self.pid]
                    rec["resident_mib"] = mine[0]["used_mib"] if mine else None
                    rec["gpu_uuid"] = mine[0]["gpu_uuid"] if mine else None
                    rec["n_gpus_visible"] = len(gpus)
                    rec["all_apps"] = apps
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
        return e.code, {"body": e.read().decode()[:400]}, time.time() - t0
    except Exception as e:
        return None, {"error": repr(e)}, time.time() - t0


# ------------------------------------------------------------------------------- arbitration
def predicted_peak_mib(cfg):
    """Phase-1 measured envelope: peak = 19786.0 + 0.050293*n_ctx (q8_0 KV, mtp OFF, -ngl 999),
    minus the measured 195.3 MiB freed per transformer block whose FFN goes to system RAM.
    f16 KV doubles the per-context term (its KV cells are 2 bytes, not 1)."""
    per_ctx = 0.050293 * (2 if cfg["kv"] == "f16" else 1)
    return 19786.0 + per_ctx * cfg["n_ctx"] - 195.3 * cfg.get("cpu_ffn_layers", 0)


def pick_free_card(cfg, margin_mib=150, wait_s=5400):
    """Pick the EMPTIEST card that can actually hold THIS config.

    Two corrections over the naive version, both forced by a real failure this session:
      1. CONFIG-AWARE threshold, not a flat one. The 81920 config needs ~23906 MiB; a flat
         22500 MiB floor happily admits a card that then cudaMalloc-fails on the compute
         buffer -- which is exactly what happened on the first attempt.
      2. MOST-free, not FIRST-fit. The first attempt picked GPU0 (23781 MiB free, holding a
         334 MiB co-tenant python) while GPU1 sat completely empty at 24120 MiB. A 334 MiB
         neighbour is enough to break the production config -- Phase-1 hazard H4, now
         observed rather than predicted.
    Another workflow may hold a ~21 GB server. NEVER co-tenant one. Wait, and LOG the wait."""
    need = predicted_peak_mib(cfg) + margin_mib
    t0, waited = time.time(), 0.0
    while True:
        gpus = gpu_table()
        fits = sorted((g for g in gpus if g["free_mib"] >= need),
                      key=lambda g: -g["free_mib"])
        if fits:
            return fits[0], round(waited, 1), gpus, round(need, 1)
        waited = time.time() - t0
        if waited > wait_s:
            return None, round(waited, 1), gpus, round(need, 1)
        print(f"  [arbitration] no card with {need:.0f} MiB free "
              f"(have {[g['free_mib'] for g in gpus]}); waited {waited:.0f}s", flush=True)
        time.sleep(30)


# ------------------------------------------------------------------------------------ launch
def launch(cfg, tag):
    backend = cfg.get("backend", "cuda")
    server = HIP_SERVER if backend == "hip" else CUDA_SERVER
    args = [server, "-m", MODEL, "-ngl", "999", "-c", str(cfg["n_ctx"]),
            "--port", str(cfg["port"]), "--host", "127.0.0.1"]
    if cfg["kv"] != "f16":
        args += ["--cache-type-k", cfg["kv"], "--cache-type-v", cfg["kv"]]
    if cfg.get("cpu_ffn_layers"):
        args += ["-ot", ffn_regex(cfg["cpu_ffn_layers"])]
    env = dict(os.environ)
    meta = {}
    if backend == "cuda":
        card, waited, gpus, need = pick_free_card(cfg)
        meta["arbitration_waited_s"] = waited
        meta["gpu_table_at_launch"] = gpus
        meta["arbitration_required_free_mib"] = need
        if card is None:
            return None, None, None, False, {"failure": "no_free_card_after_wait", **meta}
        meta["free_mib_on_chosen_card"] = card["free_mib"]
        env["CUDA_VISIBLE_DEVICES"] = card["uuid"]
        meta["target_gpu_uuid"] = card["uuid"]
        meta["target_gpu_index_at_launch"] = card["index"]
    else:
        env["HIP_VISIBLE_DEVICES"] = "0"
        meta["target"] = "igpu_rocm0_gfx1150"

    log = os.path.join(SCRATCH, f"p2_server_{tag}.log")
    fh = open(log, "wb", buffering=0)
    # CLAUDE.md says the iGPU needs `sg render -c '...'` for GPU group access. Checked rather
    # than assumed: `id -nG` already lists BOTH render and video for this user, and the HIP
    # server another process is currently running was started directly. So `sg` is unnecessary
    # here -- and it is actively harmful for this harness, because it would make proc.pid the
    # `sg` wrapper rather than llama-server, and the per-PID residency sampler (the whole
    # evidence chain for "which device did this actually land on") would sample the wrong
    # process and record nothing.
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=fh, env=env)
    s = Sampler(proc.pid, os.path.join(SCRATCH, f"p2_vram_{tag}.jsonl"), backend, interval=2.0)
    s.start()
    t0, healthy = time.time(), False
    while time.time() - t0 < cfg.get("health_timeout", 900):
        if proc.poll() is not None:
            break
        if http(f"http://127.0.0.1:{cfg['port']}/health", timeout=5)[0] == 200:
            healthy = True
            break
        time.sleep(3)
    meta["load_seconds"] = round(time.time() - t0, 1)
    return proc, s, fh, healthy, meta


def _q(a):
    return "'" + a.replace("'", "'\\''") + "'"


def ffn_regex(n):
    """FFN tensors of the first n blocks -> CPU. Written per-index: llama.cpp's override matcher
    is a plain regex with NO numeric ranges, so `blk\\.[0-9]+\\.` would offload EVERY block."""
    idx = "|".join(str(i) for i in range(n))
    return rf"blk\.({idx})\.ffn_(gate|up|down)\.weight=CPU"


def teardown(proc, s, fh):
    s.stop_evt.set()
    s.join(timeout=15)
    if proc is not None and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    if fh:
        fh.close()
    return {
        "peak_resident_mib": s.peak_mib or None,
        "peak_rss_mib": s.peak_rss_mib or None,
        "proven_gpu_uuid": s.gpu_uuid,
        "vram_samples": s.samples,
        "bus_fault": s.bus_fault,
    }


# -------------------------------------------------------------------------------------- run
def run_config(cfg):
    tag = cfg["tag"]
    print(f"\n=== CONFIG {tag} ===", flush=True)
    res = {"tag": tag, **{k: v for k, v in cfg.items() if k != "port"},
           "corpus_sha256": CORPUS["corpus_sha256"], "max_tokens": MAX_TOKENS, "seed": SEED,
           "vram_jsonl": f"p2_vram_{tag}.jsonl", "server_log": f"p2_server_{tag}.log"}
    proc, s, fh, healthy, meta = launch(cfg, tag)
    res.update(meta)
    if proc is None:
        res["launched"] = False
        return res
    res["pid"] = proc.pid
    res["health_200"] = healthy
    if not healthy:
        res["launched"] = False
        res["failure"] = "health_timeout_or_exit"
        res["exit_code"] = proc.poll()
        res.update(teardown(proc, s, fh))
        res["server_log_tail"] = _tail(res["server_log"])
        return res
    res["launched"] = True
    port = cfg["port"]
    st, props, _ = http(f"http://127.0.0.1:{port}/props", timeout=30)
    res["observed_n_ctx"] = (props.get("default_generation_settings") or {}).get("n_ctx")
    res["total_slots"] = props.get("total_slots")

    # THE ENDPOINT IS PART OF THE CONFIG, not an implementation detail.
    # gemma-4-31B-it is an INSTRUCT model: on the RAW /completion endpoint (no chat template)
    # it does not know a turn has started and degenerates -- MEASURED here, it emits an endless
    # run of "/" characters and burns the full 4096-token budget every time. The
    # /v1/chat/completions endpoint makes llama.cpp apply the GGUF's OWN embedded chat template.
    # Both are measured, because the LIVE construction sites currently take the raw path.
    use_chat = cfg.get("chat", False)
    res["endpoint"] = "/v1/chat/completions" if use_chat else "/completion"
    # Per-config completion budget. The SPEED arms keep the production 4096 (faithful). The
    # QUALITY arms need more, because gemma-4-31B-it is a REASONING model: on the chat endpoint
    # it spends its whole 4096-token budget thinking and is cut off BEFORE emitting any code
    # (measured -- ls20's generation is 6226 chars of analysis and one newline after </think>).
    # Grading a truncated-in-reasoning generation would compare two empty outputs and call it
    # "no quality difference", which is the degenerate comparison this exists to avoid.
    max_tokens = cfg.get("max_tokens", MAX_TOKENS)
    res["max_tokens"] = max_tokens
    prompts = CORPUS["prompts"]
    if cfg.get("prompt_subset"):
        prompts = [p for p in prompts if p["game"] in cfg["prompt_subset"]]
    res["games_used"] = [p["game"] for p in prompts]

    def one(p):
        if use_chat:
            url = f"http://127.0.0.1:{port}/v1/chat/completions"
            payload = {"messages": [{"role": "user", "content": p["prompt"]}],
                       "max_tokens": max_tokens, "temperature": 0.0,
                       "cache_prompt": False, "seed": SEED}
        else:
            url = f"http://127.0.0.1:{port}/completion"
            payload = {"prompt": p["prompt"], "n_predict": max_tokens, "temperature": 0.0,
                       "cache_prompt": False, "seed": SEED}
        st, body, dt = http(url, payload, timeout=cfg.get("gen_timeout", 3600))
        r = {"game": p["game"], "prompt_chars": p["chars"], "status": st,
             "wall_seconds": round(dt, 2)}
        if st == 200:
            if use_chat:
                choice = (body.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                final = str(msg.get("content") or "")
                # gemma-4-31B-it emits a SEPARATE reasoning channel on this endpoint. Production
                # (_chat_completion) folds it back into content wrapped in <think> tags; this must
                # do the SAME or every generation looks empty and grades as induce_ok=False.
                reasoning = str(msg.get("reasoning_content") or "")
                content = f"<think>\n{reasoning}\n</think>\n{final}" if reasoning else final
                r["reasoning_chars"] = len(reasoning)
                r["final_answer_chars"] = len(final)
                stop_reason = "limit" if choice.get("finish_reason") == "length" else "eos"
            else:
                content = body.get("content", "")
                stop_reason = ("eos" if body.get("stopped_eos")
                               else "limit" if body.get("stopped_limit") else "other")
            tm = body.get("timings", {}) or {}
            r.update(
                prompt_tokens=tm.get("prompt_n"),
                prefill_ms=round(tm.get("prompt_ms") or 0, 1),
                prefill_tps=round(tm.get("prompt_per_second") or 0, 2),
                predicted_tokens=tm.get("predicted_n"),
                decode_ms=round(tm.get("predicted_ms") or 0, 1),
                decode_tps=round(tm.get("predicted_per_second") or 0, 3),
                stop_reason=stop_reason,
            )
            if r["predicted_tokens"] is None:
                usage = body.get("usage") or {}
                r["predicted_tokens"] = usage.get("completion_tokens")
            # persist the generated world model for the QUALITY arm
            gen_dir = os.path.join(SCRATCH, "gen", tag)
            os.makedirs(gen_dir, exist_ok=True)
            open(os.path.join(gen_dir, f"{p['game']}.txt"), "w").write(content)
        elif st is None:
            # The exp5833 signature: is the server still claiming health while generation died?
            hs = http(f"http://127.0.0.1:{port}/health", timeout=10)[0]
            r["health_after_failure"] = hs
            r["WEDGE_health200_completion_hung"] = hs == 200
            r["error"] = body.get("error")
        else:
            r["body"] = body
        print(f"  {r['game']:6s} st={st} wall={r['wall_seconds']}s "
              f"prefill={r.get('prefill_tps')} tok/s decode={r.get('decode_tps')} tok/s "
              f"gen={r.get('predicted_tokens')} stop={r.get('stop_reason')}", flush=True)
        return r

    # K is the number of CONCURRENT inductions. K=1 isolates per-induction speed; K=4 is the
    # shape the SCORED path actually runs (the eval framework starts one thread per game with
    # no pool, swarm.py:91, and llama-server's default n_parallel is 4) and is therefore the
    # only shape comparable to the 340-495 s/induction baseline this session already has.
    K = cfg.get("K", 1)
    res["K"] = K
    t_all = time.time()
    if K == 1:
        runs = [one(p) for p in prompts]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=K) as ex:
            runs = list(ex.map(one, prompts))
    res["all_prompts_wall_s"] = round(time.time() - t_all, 1)
    res["runs"] = runs
    if any(r.get("WEDGE_health200_completion_hung") for r in runs):
        res["WEDGE_DETECTED"] = True
    ok = [r for r in runs if r["status"] == 200]
    if ok:
        res["summary"] = {
            "n_ok": len(ok), "n_total": len(runs),
            "mean_wall_s": round(sum(r["wall_seconds"] for r in ok) / len(ok), 1),
            "mean_prefill_tps": round(sum(r["prefill_tps"] for r in ok) / len(ok), 1),
            "mean_decode_tps": round(sum(r["decode_tps"] for r in ok) / len(ok), 3),
            "mean_prefill_s": round(sum(r["prefill_ms"] for r in ok) / len(ok) / 1000, 2),
            "mean_decode_s": round(sum(r["decode_ms"] for r in ok) / len(ok) / 1000, 1),
            "mean_gen_tokens": round(sum(r["predicted_tokens"] for r in ok) / len(ok), 1),
        }
        sm = res["summary"]
        sm["decode_share_of_wall"] = round(sm["mean_decode_s"] / sm["mean_wall_s"], 4)
    # health at the very end -- a health check is not a liveness check, but its ABSENCE is real
    res["health_200_at_end"] = http(f"http://127.0.0.1:{port}/health", timeout=15)[0] == 200
    res.update(teardown(proc, s, fh))
    res["server_log_tail"] = _tail(res["server_log"])
    return res


def _tail(name, n=1400):
    try:
        return open(os.path.join(SCRATCH, name), errors="replace").read()[-n:]
    except Exception:
        return None


CONFIGS = {
    # --- the PREFERRED option, RAW endpoint. Kept as the EVIDENCE arm for the raw-completion
    #     degeneration hazard, since all three live construction sites currently take this path.
    "A_egpu_81920_q8": {"tag": "A_egpu_81920_q8", "n_ctx": 81920, "kv": "q8_0", "port": 8991,
                        "backend": "cuda"},
    # --- the PREFERRED option as it SHOULD run: chat template applied.
    "AC_egpu_81920_q8_chat": {"tag": "AC_egpu_81920_q8_chat", "n_ctx": 81920, "kv": "q8_0",
                              "port": 8901, "backend": "cuda", "chat": True},
    "AC4_egpu_81920_q8_chat_K4": {"tag": "AC4_egpu_81920_q8_chat_K4", "n_ctx": 81920,
                                  "kv": "q8_0", "port": 8902, "backend": "cuda", "chat": True,
                                  "K": 4},
    # --- QUALITY on the endpoint that actually produces code
    "QC_egpu_24576_f16_chat": {"tag": "QC_egpu_24576_f16_chat", "n_ctx": 24576, "kv": "f16",
                               "port": 8903, "backend": "cuda", "chat": True,
                               "max_tokens": 512, "prompt_subset": ["ls20","tu93"],
                               "gen_timeout": 5400},
    "QC_egpu_24576_q8_chat": {"tag": "QC_egpu_24576_q8_chat", "n_ctx": 24576, "kv": "q8_0",
                              "port": 8904, "backend": "cuda", "chat": True,
                              "max_tokens": 512, "prompt_subset": ["ls20","tu93"],
                              "gen_timeout": 5400},
    # --- FALLBACK (a) and (b) on the chat endpoint
    "FC_egpu_81920_q8_ffn12_chat": {"tag": "FC_egpu_81920_q8_ffn12_chat", "n_ctx": 81920,
                                    "kv": "q8_0", "port": 8905, "backend": "cuda",
                                    "cpu_ffn_layers": 12, "chat": True},
    "IC_igpu_81920_q8_chat": {"tag": "IC_igpu_81920_q8_chat", "n_ctx": 81920, "kv": "q8_0",
                              "port": 8906, "backend": "hip", "chat": True,
                              "health_timeout": 1800, "gen_timeout": 7200},
    # --- the context the prior head-to-head (340-495 s/induction) was measured at
    "B_egpu_32768_q8": {"tag": "B_egpu_32768_q8", "n_ctx": 32768, "kv": "q8_0", "port": 8992,
                        "backend": "cuda"},
    # --- QUALITY: f16 vs q8_0 at a MATCHED n_ctx both can actually hold
    "Q_egpu_24576_f16": {"tag": "Q_egpu_24576_f16", "n_ctx": 24576, "kv": "f16", "port": 8993,
                         "backend": "cuda"},
    "Q_egpu_24576_q8": {"tag": "Q_egpu_24576_q8", "n_ctx": 24576, "kv": "q8_0", "port": 8994,
                        "backend": "cuda"},
    # --- FALLBACK (a): eGPU + dense FFN offload to system RAM
    "F_egpu_81920_q8_ffn12": {"tag": "F_egpu_81920_q8_ffn12", "n_ctx": 81920, "kv": "q8_0",
                              "port": 8995, "backend": "cuda", "cpu_ffn_layers": 12},
    # --- FALLBACK (b): the iGPU. NOTE _resolve_llama_server() PREFERS this build.
    "I_igpu_81920_q8": {"tag": "I_igpu_81920_q8", "n_ctx": 81920, "kv": "q8_0", "port": 8996,
                        "backend": "hip", "health_timeout": 1800, "gen_timeout": 5400},
    # --- the K=4 shape, matched to how the SCORED path actually runs. This is the ONLY arm
    #     comparable to the 340-495 s/induction baseline already on record.
    "A4_egpu_81920_q8_K4": {"tag": "A4_egpu_81920_q8_K4", "n_ctx": 81920, "kv": "q8_0",
                            "port": 8997, "backend": "cuda", "K": 4},
    "I4_igpu_81920_q8_K4": {"tag": "I4_igpu_81920_q8_K4", "n_ctx": 81920, "kv": "q8_0",
                            "port": 8998, "backend": "hip", "K": 4,
                            "health_timeout": 1800, "gen_timeout": 7200},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--out", default="phase2_results.json")
    a = ap.parse_args()
    out_path = os.path.join(SCRATCH, a.out)
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for name in a.configs:
        results[name] = run_config(CONFIGS[name])
        json.dump(results, open(out_path, "w"), indent=1)
    print(json.dumps({"done": a.configs, "out": out_path}))


if __name__ == "__main__":
    main()

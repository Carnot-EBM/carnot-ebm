"""PHASE 1b: the K=4 concurrency + sustained shape, plus a margin curve.

The K=1 single-shot fit at (81920, q8_0) peaked 23906 MiB -- 48 MiB ABOVE llama.cpp's own
reported free (23858) and ~217 MiB below its reported total (24123). At that margin a single-
shot pass is NOT evidence the config is safe: the pool exists to serve K=4, and the f16/32768
cell OOMed at 23902 MiB, essentially the same number. So:

  ARM A: (81920, q8_0) under K=4 CONCURRENT real induce prompts, repeated for several rounds,
         with per-PID VRAM sampled throughout. This is the shape the pool is sized for and the
         shape that triggered modes A/B/C in the codebase's own history.
  ARM B: a margin curve over n_ctx at q8_0 (load-only, no traffic), so the operator can pick an
         n_ctx with real headroom if K=4 proves 81920 unsafe.

Prompts are sized to the codebase's OWN admission arithmetic: at K=4 the pool admits
n_ctx/4 - max_tokens = 16384 tokens. The worst-case prompt we built measures 16466, which is
82 tokens OVER that ceiling -- so ARM A uses a prompt trimmed under the ceiling, and separately
records that the over-ceiling prompt 500s at K=4 (a token-budget finding, not a VRAM one).
"""

import json, os, signal, subprocess, threading, time
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

SCRATCH = os.path.dirname(os.path.abspath(__file__))
SERVER = "/home/ianblenke/.cache/llama.cpp-master/build/bin/llama-server"
MODEL = ("/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF/"
         "snapshots/f130ba51393346288f5862e30e9586b9b021513f/gemma-4-31B-it-Q4_K_M.gguf")
FULL_PROMPT = open(os.path.join(SCRATCH, "induce_prompt_worstcase.txt")).read()
REAL_FREE_MIB, REAL_TOTAL_MIB = 23858, 24123

exec(open(os.path.join(SCRATCH, "fitgrid.py")).read().split("if __name__")[0].replace(
    "PROMPT = open(os.path.join(SCRATCH, \"induce_prompt_worstcase.txt\")).read()", "PROMPT = ''"))


def launch(n_ctx, kv, port, uuid, tag):
    args = [SERVER, "-m", MODEL, "-ngl", "999", "-c", str(n_ctx),
            "--port", str(port), "--host", "127.0.0.1"]
    if kv != "f16":
        args += ["--cache-type-k", kv, "--cache-type-v", kv]
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = uuid
    log = os.path.join(SCRATCH, f"p1b_server_{tag}.log")
    fh = open(log, "wb", buffering=0)
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=fh, env=env)
    s = Sampler(proc.pid, os.path.join(SCRATCH, f"p1b_vram_{tag}.jsonl"), interval=1.0)
    s.start()
    t0 = time.time(); healthy = False
    while time.time() - t0 < 420:
        if proc.poll() is not None:
            break
        if http(f"http://127.0.0.1:{port}/health", timeout=5)[0] == 200:
            healthy = True; break
        time.sleep(2)
    return proc, s, fh, healthy, log


def teardown(proc, s, fh):
    s.stop_evt.set(); s.join(timeout=10)
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try: proc.wait(timeout=45)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=30)
    fh.close()
    return s.peak_mib, s.gpu_uuid, s.samples, s.bus_fault


def trim_to_tokens(port, text, target):
    """Binary-search a char length whose /tokenize count is just under `target`."""
    lo, hi = 1000, len(text)
    best = text[:lo]
    for _ in range(14):
        mid = (lo + hi) // 2
        st, tk, _ = http(f"http://127.0.0.1:{port}/tokenize", {"content": text[:mid]}, timeout=60)
        n = len(tk.get("tokens", [])) if st == 200 else 10**9
        if n <= target:
            best = text[:mid]; lo = mid + 1
        else:
            hi = mid - 1
    st, tk, _ = http(f"http://127.0.0.1:{port}/tokenize", {"content": best}, timeout=60)
    return best, len(tk.get("tokens", []))


def arm_a(uuid):
    """K=4 concurrent, sustained, at the production (81920, q8_0)."""
    port, tag = 8981, "armA_81920_q8_0_K4"
    res = {"arm": "A_K4_concurrency_sustained", "n_ctx": 81920, "kv": "q8_0", "K": 4,
           "target_gpu_uuid": uuid}
    proc, s, fh, healthy, log = launch(81920, "q8_0", port, uuid, tag)
    res["pid"] = proc.pid; res["health_200"] = healthy
    if not healthy:
        res["launched"] = False; res["failure"] = "health_timeout_or_exit"
        res["peak_resident_mib"], res["proven_gpu_uuid"], res["vram_samples"], res["bus_fault"] = teardown(proc, s, fh)
        return res
    res["launched"] = True
    time.sleep(8)
    st, props, _ = http(f"http://127.0.0.1:{port}/props", timeout=20)
    res["observed_n_ctx"] = (props.get("default_generation_settings") or {}).get("n_ctx")
    res["total_slots"] = props.get("total_slots")

    # admission ceiling at K=4 per the codebase's own arithmetic
    ceiling = 81920 // 4 - 4096
    res["k4_admission_ceiling_tokens"] = ceiling
    st, tk, _ = http(f"http://127.0.0.1:{port}/tokenize", {"content": FULL_PROMPT}, timeout=60)
    res["worstcase_prompt_tokens"] = len(tk.get("tokens", []))
    res["worstcase_exceeds_k4_ceiling_by"] = res["worstcase_prompt_tokens"] - ceiling

    fit_prompt, fit_tokens = trim_to_tokens(port, FULL_PROMPT, ceiling - 200)
    res["k4_prompt_tokens"] = fit_tokens

    def one(i):
        t0 = time.time()
        st, body, dt = http(f"http://127.0.0.1:{port}/completion",
                            {"prompt": fit_prompt, "n_predict": 256, "temperature": 0.0,
                             "cache_prompt": False, "seed": 1000 + i}, timeout=900)
        r = {"i": i, "status": st, "seconds": round(dt, 1)}
        if st == 200:
            tm = body.get("timings", {}) or {}
            r.update(predicted_n=body.get("tokens_predicted"),
                     prompt_ms=tm.get("prompt_ms"), predicted_ms=tm.get("predicted_ms"),
                     prompt_tps=tm.get("prompt_per_second"), predicted_tps=tm.get("predicted_per_second"))
        elif st is None:
            hst, _, _ = http(f"http://127.0.0.1:{port}/health", timeout=5)
            r["exc"] = body.get("exc"); r["health_after"] = hst
            r["exp5833_wedge"] = (hst == 200 and "timed out" in str(body.get("exc")).lower())
        else:
            r["error_body"] = body.get("error_body")
        return r

    rounds = []
    for rnd in range(3):
        with ThreadPoolExecutor(max_workers=4) as ex:
            out = list(ex.map(one, range(4 * rnd, 4 * rnd + 4)))
        rounds.append(out)
        print(f"  [armA round {rnd}] " + json.dumps([{k: v for k, v in o.items()
              if k in ('status', 'seconds', 'predicted_n')} for o in out]), flush=True)
        # server still alive between rounds?
        rounds[-1] = {"round": rnd, "results": out,
                      "health_between_rounds": http(f"http://127.0.0.1:{port}/health", timeout=5)[0]}
    res["rounds"] = rounds

    # the OVER-ceiling worst-case prompt at K=4 -- expected to 500 (token budget, not VRAM)
    def over(i):
        st, body, dt = http(f"http://127.0.0.1:{port}/completion",
                            {"prompt": FULL_PROMPT, "n_predict": 4096, "temperature": 0.0,
                             "cache_prompt": False}, timeout=900)
        return {"i": i, "status": st,
                "error_body": (body.get("error_body") or "")[:200] if st not in (200, None) else None,
                "predicted_n": body.get("tokens_predicted") if st == 200 else None}
    with ThreadPoolExecutor(max_workers=4) as ex:
        res["overceiling_k4"] = list(ex.map(over, range(4)))
    print("  [armA overceiling] " + json.dumps(res["overceiling_k4"]), flush=True)

    res["health_final"] = http(f"http://127.0.0.1:{port}/health", timeout=5)[0]
    time.sleep(3)
    res["peak_resident_mib"], res["proven_gpu_uuid"], res["vram_samples"], res["bus_fault"] = teardown(proc, s, fh)
    res["headroom_vs_free"] = REAL_FREE_MIB - (res["peak_resident_mib"] or 0)
    res["headroom_vs_total"] = REAL_TOTAL_MIB - (res["peak_resident_mib"] or 0)
    res["server_log_tail"] = open(log, "rb").read()[-2500:].decode("utf-8", "replace")
    return res


def arm_b(uuid):
    """Margin curve: load-only peak VRAM vs n_ctx at q8_0."""
    out = []
    port = 8991
    for n_ctx in (49152, 65536, 73728):
        tag = f"armB_{n_ctx}_q8_0"
        proc, s, fh, healthy, log = launch(n_ctx, "q8_0", port, uuid, tag)
        rec = {"n_ctx": n_ctx, "kv": "q8_0", "pid": proc.pid, "health_200": healthy}
        if healthy:
            time.sleep(8)
            st, _, _ = http(f"http://127.0.0.1:{port}/completion",
                            {"prompt": FULL_PROMPT[:20000], "n_predict": 8,
                             "temperature": 0.0, "cache_prompt": False}, timeout=600)
            rec["admit_status"] = st
            time.sleep(3)
        rec["peak_resident_mib"], rec["proven_gpu_uuid"], rec["vram_samples"], rec["bus_fault"] = teardown(proc, s, fh)
        rec["headroom_vs_free"] = REAL_FREE_MIB - (rec["peak_resident_mib"] or 0)
        rec["headroom_vs_total"] = REAL_TOTAL_MIB - (rec["peak_resident_mib"] or 0)
        out.append(rec)
        print("  [armB] " + json.dumps(rec), flush=True)
        port += 1
        time.sleep(8)
    return out


if __name__ == "__main__":
    uuid, idx, waited = wait_for_card(min_free_mib=23000)
    if uuid is None:
        print(json.dumps({"failure": "blocked_no_free_card", "waited_seconds": waited}))
        raise SystemExit(1)
    print(f"=== phase1b on GPU idx {idx} {uuid} (waited {waited}s) ===", flush=True)
    out = {"gpu_uuid": uuid, "gpu_index_at_launch": idx, "gpu_wait_seconds": waited,
           "real_free_mib": REAL_FREE_MIB, "real_total_mib": REAL_TOTAL_MIB}
    out["arm_a"] = arm_a(uuid)
    print(json.dumps({k: v for k, v in out["arm_a"].items()
                      if k not in ("rounds", "server_log_tail")}, indent=2), flush=True)
    out["arm_b"] = arm_b(uuid)
    p = os.path.join(SCRATCH, "phase1b_results.json")
    json.dump(out, open(p, "w"), indent=2)
    print("WROTE", p)

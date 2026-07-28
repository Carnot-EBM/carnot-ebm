"""PHASE 1c: does the DEFAULT construction-site config fit?

arc_competition_agent.py:881, :5011 and arc_ige_cell_selector.py:158 all build the proposer with
    mtp=(os.environ.get("CARNOT_ARC_MTP", "1") != "0")
i.e. MTP is ON by default on the local/dev path; only scripts/kaggle/submission_kernel/main.py
forces it to 0. MTP passes `--spec-type draft-mtp --model-draft <same gguf>`, which loads a SECOND
copy of the weights (the VRAM envelope comment puts the 9B cost at ~6.1 GB). For a 17 GB 31B that
is not a tuning question, it is a hard fit question -- so it belongs in the Phase 1 fit grid.

Measures the exact default-shape launch. Fails fast either way; the failure MODE is the finding.
"""

import json, os, signal, subprocess, time

SCRATCH = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(SCRATCH, "fitgrid.py")).read().split("if __name__")[0].replace(
    'PROMPT = open(os.path.join(SCRATCH, "induce_prompt_worstcase.txt")).read()', "PROMPT = ''"))

SERVER = "/home/ianblenke/.cache/llama.cpp-master/build/bin/llama-server"
MODEL = ("/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF/"
         "snapshots/f130ba51393346288f5862e30e9586b9b021513f/gemma-4-31B-it-Q4_K_M.gguf")


def run(n_ctx, kv, mtp, port, uuid, tag):
    args = [SERVER, "-m", MODEL, "-ngl", "999", "-c", str(n_ctx),
            "--port", str(port), "--host", "127.0.0.1"]
    if mtp:
        args += ["--spec-type", "draft-mtp", "--model-draft", MODEL]
    if kv != "f16":
        args += ["--cache-type-k", kv, "--cache-type-v", kv]
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = uuid
    log = os.path.join(SCRATCH, f"p1c_{tag}.log")
    fh = open(log, "wb", buffering=0)
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=fh, env=env)
    s = Sampler(proc.pid, os.path.join(SCRATCH, f"p1c_vram_{tag}.jsonl"), interval=1.0)
    s.start()
    t0 = time.time(); healthy = False
    while time.time() - t0 < 420:
        if proc.poll() is not None:
            break
        if http(f"http://127.0.0.1:{port}/health", timeout=5)[0] == 200:
            healthy = True; break
        time.sleep(2)
    rec = {"cell": tag, "n_ctx": n_ctx, "kv": kv, "mtp": mtp, "argv": args,
           "pid": proc.pid, "health_200": healthy, "load_seconds": round(time.time() - t0, 1),
           "exit_code_during_load": proc.returncode}
    if healthy:
        time.sleep(8)
    s.stop_evt.set(); s.join(timeout=10)
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try: proc.wait(timeout=45)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=30)
    fh.close()
    rec.update(peak_resident_mib=s.peak_mib or None, proven_gpu_uuid=s.gpu_uuid,
               vram_samples=s.samples, bus_fault=s.bus_fault, final_exit_code=proc.returncode)
    rec["headroom_vs_total_24123"] = 24123 - (s.peak_mib or 0)
    tail = open(log, "rb").read()[-2500:].decode("utf-8", "replace")
    rec["log_tail"] = tail
    for marker in ("out of memory", "draft", "GGML_ASSERT", "error"):
        hits = [ln for ln in tail.splitlines() if marker.lower() in ln.lower()]
        if hits:
            rec.setdefault("log_markers", {})[marker] = hits[-3:]
    return rec


if __name__ == "__main__":
    uuid, idx, waited = wait_for_card(min_free_mib=23000)
    if uuid is None:
        print(json.dumps({"failure": "blocked_no_free_card", "waited_seconds": waited})); raise SystemExit(1)
    print(f"=== phase1c on GPU idx {idx} {uuid} (waited {waited}s) ===", flush=True)
    cells = []
    # the EXACT default construction-site shape for a 31B migration: mtp on, q8_0 kv, 81920
    cells.append(run(81920, "q8_0", True, 8961, uuid, "default_mtpON_81920_q8_0"))
    print(json.dumps({k: v for k, v in cells[-1].items() if k not in ("log_tail", "argv")}, indent=2), flush=True)
    json.dump({"gpu_uuid": uuid, "cells": cells},
              open(os.path.join(SCRATCH, "phase1c_results.json"), "w"), indent=2)
    print("WROTE phase1c_results.json")

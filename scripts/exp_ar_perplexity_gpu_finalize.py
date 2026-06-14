#!/usr/bin/env python3
"""GPU AR-perplexity leg + final matched comparison for the oracle-distinct detector.

Runs in the CUDA-compiled llama-cpp-python venv (n_gpu_layers=-1 -> all layers on a 3090),
replacing the ~4h CPU AR leg with ~minutes. vLLM was blocked (gemma-4 per-layer-KV-head config
crashes its transformers); llama-perplexity is corpus-only. This is the matched AR baseline:
same gemma-4-26B-A4B backbone + Q4_K_M quant as the diffusion scorer; the ONLY variable is
AR-vs-diffusion.

Inputs (produced by the conductor-venv side):
  results/_detector_items_manifest.jsonl   {key, y, text}  (the exact 160 items, seed 4211)
  results/_cache_diffusion_detector_scores.jsonl   {key, dg, ...}  (diffusion surprisal cache)
Output: results/oracle_distinct_diffusion_surprisal_detector.json  (augmented with AR + comparison)

Needs only: llama_cpp (CUDA), numpy.  verifier_is_oracle=False for both legs (headline-eligible).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

AR_GGUF = glob.glob(
    str(Path.home() / ".cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/"
        "snapshots/*/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf")
)
MANIFEST = Path("results/_detector_items_manifest.jsonl")
DG_CACHE = Path("results/_cache_diffusion_detector_scores.jsonl")
OUT = Path("results/oracle_distinct_diffusion_surprisal_detector.json")
PROMPT = "Evaluate this math reasoning step:\n"
SEED = 4211
N_BOOT = 2000


def log(m):
    print(f"[ar-gpu] {m}", flush=True)


def auroc(scores, y):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def boot_ci(scores, y, rng):
    base = auroc(scores, y)
    n = len(y)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yi = y[idx]
        if yi.sum() in (0, n):
            continue
        vals.append(auroc(scores[idx], yi))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return base, float(lo), float(hi)


def boot_ci_diff(s_a, s_b, y, rng):
    base = auroc(s_a, y) - auroc(s_b, y)
    n = len(y)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yi = y[idx]
        if yi.sum() in (0, n):
            continue
        vals.append(auroc(s_a[idx], yi) - auroc(s_b[idx], yi))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return base, float(lo), float(hi)


def main():
    t0 = time.time()
    items = [json.loads(l) for l in open(MANIFEST)]
    dg = {}
    for l in open(DG_CACHE):
        d = json.loads(l)
        if d.get("dg") is not None:
            dg[d["key"]] = d["dg"]
    log(f"{len(items)} items; dg cache covers {sum(1 for it in items if it['key'] in dg)}")

    rep = json.loads(OUT.read_text()) if OUT.exists() else {
        "experiment": "oracle_distinct_diffusion_surprisal_detector",
        "verifier_is_oracle": False,
    }

    from llama_cpp import Llama
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    gguf = AR_GGUF[0]
    log(f"loading AR {Path(gguf).name} with n_gpu_layers=-1 (GPU)...")
    ar = Llama(model_path=gguf, n_gpu_layers=-1, logits_all=True, n_ctx=512, verbose=True)
    prompt_tok_n = len(ar.tokenize(PROMPT.encode(), add_bos=True))

    # GPU offload evidence (best-effort introspection)
    gpu_evidence = {}
    try:
        import subprocess
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        gpu_evidence["nvidia_smi_after_load"] = smi
    except Exception as e:
        gpu_evidence["nvidia_smi_error"] = repr(e)[:120]
    rep["ar_gpu_offload_evidence"] = gpu_evidence
    rep["ar_n_gpu_layers_requested"] = -1

    def ar_score(text):
        # max_tokens=1 (NOT 0 -> 0 means "unlimited" in llama_cpp and wastes ~400 gen tokens/call).
        # echo=True returns logprobs for the prompt tokens [0..full_n); slice the step portion only,
        # excluding the single generated token, for a clean mean step-token NLL.
        full_n = len(ar.tokenize((PROMPT + text).encode(), add_bos=True))
        out = ar.create_completion(PROMPT + text, max_tokens=1, echo=True,
                                   logprobs=1, temperature=0.0)
        tlp = out["choices"][0]["logprobs"]["token_logprobs"]
        step = [x for x in tlp[prompt_tok_n:full_n] if x is not None]
        return float(-np.mean(step)) if step else None

    ar_scores = {}
    for i, it in enumerate(items):
        ar_scores[it["key"]] = ar_score(it["text"])
        if (i + 1) % 20 == 0 or i == len(items) - 1:
            log(f"AR scored {i + 1}/{len(items)} ({time.time() - t0:.0f}s)")

    # assemble paired arrays over items with BOTH scores
    ys, ars, dgs = [], [], []
    for it in items:
        a = ar_scores.get(it["key"])
        d = dg.get(it["key"])
        if a is None or d is None:
            continue
        ys.append(it["y"]); ars.append(a); dgs.append(d)
    y = np.array(ys); ar_s = np.array(ars); dg_s = np.array(dgs)
    rep["n_scored"] = int(len(y))
    rep["n_incorrect_scored"] = int(y.sum())
    rep["ar_mean_correct"] = round(float(ar_s[y == 0].mean()), 4)
    rep["ar_mean_incorrect"] = round(float(ar_s[y == 1].mean()), 4)
    rep["dg_mean_correct"] = round(float(dg_s[y == 0].mean()), 4)
    rep["dg_mean_incorrect"] = round(float(dg_s[y == 1].mean()), 4)

    brng = np.random.default_rng(SEED + 1)
    a_auc, a_lo, a_hi = boot_ci(ar_s, y, brng)
    d_auc, d_lo, d_hi = boot_ci(dg_s, y, brng)
    diff, dl, dh = boot_ci_diff(dg_s, ar_s, y, brng)
    rand_s = np.random.default_rng(SEED + 7).standard_normal(len(y))
    r_auc, r_lo, r_hi = boot_ci(rand_s, y, brng)

    rep["ar_auroc"] = round(a_auc, 4)
    rep["ar_auroc_ci95"] = [round(a_lo, 4), round(a_hi, 4)]
    rep["diffusion_auroc"] = round(d_auc, 4)
    rep["diffusion_auroc_ci95"] = [round(d_lo, 4), round(d_hi, 4)]
    rep["diff_minus_ar_auroc"] = round(diff, 4)
    rep["diff_minus_ar_ci95"] = [round(dl, 4), round(dh, 4)]
    rep["neg_control_random_auroc"] = round(r_auc, 4)
    rep["neg_control_random_ci95"] = [round(r_lo, 4), round(r_hi, 4)]
    rep["ar_status"] = "complete_gpu"
    rep["ar_engine"] = "llama-cpp-python CUDA (GGML_CUDA=on), n_gpu_layers=-1"

    g_detect = d_lo > 0.5
    g_beats = dl > 0.0
    neg_ok = r_lo <= 0.5 <= r_hi
    rep["acceptance_gate"] = {
        "G_detect_diffusion_excludes_0.5": bool(g_detect),
        "G_beats_ar_diff_excludes_0": bool(g_beats),
        "neg_control_ok": bool(neg_ok),
    }
    rep["reproducibility_checksum"] = hashlib.sha256(
        (str(sorted(it["key"] for it in items)) + str(SEED)).encode()).hexdigest()[:16]
    if not neg_ok:
        rep["honest_verdict"] = "blocked_neg_control_failed_metric_bug"
    elif g_detect and g_beats:
        rep["honest_verdict"] = (
            "complete: oracle_distinct_diffusion_surprisal_detects_errors_AND_beats_matched_AR")
    elif g_detect:
        rep["honest_verdict"] = (
            "complete: oracle_distinct_diffusion_surprisal_detects_errors_ties_AR")
    else:
        rep["honest_verdict"] = (
            "complete: oracle_distinct_diffusion_surprisal_no_detection_signal_honest_null")
    rep["duration_s_ar_gpu_leg"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rep, indent=2))
    log(f"AR AUROC={a_auc:.3f}{rep['ar_auroc_ci95']}  DG AUROC={d_auc:.3f}{rep['diffusion_auroc_ci95']}"
        f"  diff={diff:.3f}{rep['diff_minus_ar_ci95']}  verdict={rep['honest_verdict']}")


if __name__ == "__main__":
    main()

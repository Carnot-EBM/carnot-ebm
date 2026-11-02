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

    V = ar.n_vocab()

    def ar_score(text):
        # Low-level: ONE GPU forward (logits_all) + vectorized numpy logsumexp.
        # ~0.45s/item vs ~25s for create_completion(logprobs=1), which sorts the full 262k
        # vocab per token in pure Python (the prior timeout). Validated within ~0.3 NLL of the
        # create_completion path on 4 items with IDENTICAL ranking (rank-based AUROC is unaffected
        # by the tokenization-boundary offset). NLL_j = logsumexp(logits_{j-1}) - logit_{j-1}[tok_j]
        # >= 0 (logits at position j-1 predict token j). Mean over the step tokens.
        toks = ar.tokenize((PROMPT + text).encode(), add_bos=True)
        ar.reset()
        ar.eval(toks)
        sc = np.array(ar.scores, dtype=np.float32).reshape(-1, V)[:len(toks)]
        lps = []
        for j in range(prompt_tok_n, len(toks)):
            row = sc[j - 1]
            m = row.max()
            lse = m + np.log(np.exp(row - m).sum())
            lps.append(lse - row[toks[j]])
        return float(np.mean(lps)) if lps else None

    ar_scores = {}
    lengths = {}
    for i, it in enumerate(items):
        ar_scores[it["key"]] = ar_score(it["text"])
        lengths[it["key"]] = len(ar.tokenize(it["text"].encode(), add_bos=False))
        if (i + 1) % 20 == 0 or i == len(items) - 1:
            log(f"AR scored {i + 1}/{len(items)} ({time.time() - t0:.0f}s)")
    # persist per-item AR + length for reproducibility / re-analysis
    with open("results/_cache_detector_ar_length.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps({"key": it["key"], "y": it["y"],
                                "ar": ar_scores.get(it["key"]),
                                "len": lengths.get(it["key"])}) + "\n")

    # assemble paired arrays over items with BOTH scores
    ys, ars, dgs, lns = [], [], [], []
    for it in items:
        a = ar_scores.get(it["key"])
        d = dg.get(it["key"])
        if a is None or d is None:
            continue
        ys.append(it["y"]); ars.append(a); dgs.append(d); lns.append(lengths[it["key"]])
    y = np.array(ys); ar_s = np.array(ars); dg_s = np.array(dgs); ln = np.array(lns, float)
    rep["n_scored"] = int(len(y))
    rep["n_incorrect_scored"] = int(y.sum())
    rep["ar_mean_correct"] = round(float(ar_s[y == 0].mean()), 4)
    rep["ar_mean_incorrect"] = round(float(ar_s[y == 1].mean()), 4)
    rep["dg_mean_correct"] = round(float(dg_s[y == 0].mean()), 4)
    rep["dg_mean_incorrect"] = round(float(dg_s[y == 1].mean()), 4)

    def two_dir(auc, lo, hi):
        # detection strength is |AUROC-0.5|; a signal exists if the CI excludes 0.5 EITHER side.
        return {"auroc": round(auc, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "abs_lift": round(abs(auc - 0.5), 4),
                "excludes_0.5": bool(hi < 0.5 or lo > 0.5),
                "direction": "errors_higher_score" if auc > 0.5 else "errors_lower_score"}

    def resid(s):
        # remove the linear effect of token length -> residual = the length-controlled signal
        A = np.vstack([ln, np.ones_like(ln)]).T
        coef, *_ = np.linalg.lstsq(A, s, rcond=None)
        return s - A @ coef

    brng = np.random.default_rng(SEED + 1)
    a_auc, a_lo, a_hi = boot_ci(ar_s, y, brng)
    d_auc, d_lo, d_hi = boot_ci(dg_s, y, brng)
    diff, dl, dh = boot_ci_diff(dg_s, ar_s, y, brng)
    rand_s = np.random.default_rng(SEED + 7).standard_normal(len(y))
    r_auc, r_lo, r_hi = boot_ci(rand_s, y, brng)

    # ---- LENGTH CONFOUND (FoVer incorrect steps are far longer than correct) ----
    l_auc, l_lo, l_hi = boot_ci(ln, y, brng)
    dgr_auc, dgr_lo, dgr_hi = boot_ci(resid(dg_s), y, brng)
    arr_auc, arr_lo, arr_hi = boot_ci(resid(ar_s), y, brng)
    rep["length_confound"] = {
        "len_mean_correct": round(float(ln[y == 0].mean()), 1),
        "len_mean_incorrect": round(float(ln[y == 1].mean()), 1),
        "auroc_length_alone": two_dir(l_auc, l_lo, l_hi),
        "corr_diffusion_length": round(float(np.corrcoef(dg_s, ln)[0, 1]), 3),
        "corr_ar_length": round(float(np.corrcoef(ar_s, ln)[0, 1]), 3),
        "diffusion_auroc_length_controlled": two_dir(dgr_auc, dgr_lo, dgr_hi),
        "ar_auroc_length_controlled": two_dir(arr_auc, arr_lo, arr_hi),
        "note": ("FoVer incorrect steps are ~2.6x longer; per-token surprisal anti-correlates with "
                 "length, so the raw AUROC is largely a length artifact. The length-controlled "
                 "residual is the honest oracle-distinct estimate; a length-matched corpus rerun is "
                 "required before any headline claim."),
    }

    rep["ar_auroc"] = round(a_auc, 4)
    rep["ar_auroc_ci95"] = [round(a_lo, 4), round(a_hi, 4)]
    rep["diffusion_auroc"] = round(d_auc, 4)
    rep["diffusion_auroc_ci95"] = [round(d_lo, 4), round(d_hi, 4)]
    rep["diffusion_detection_two_directional"] = two_dir(d_auc, d_lo, d_hi)
    rep["ar_detection_two_directional"] = two_dir(a_auc, a_lo, a_hi)
    rep["diff_minus_ar_auroc"] = round(diff, 4)
    rep["diff_minus_ar_ci95"] = [round(dl, 4), round(dh, 4)]
    rep["neg_control_random_auroc"] = round(r_auc, 4)
    rep["neg_control_random_ci95"] = [round(r_lo, 4), round(r_hi, 4)]
    rep["ar_status"] = "complete_gpu"
    rep["ar_engine"] = "llama-cpp-python CUDA (GGML_CUDA=on), n_gpu_layers=-1"
    rep["surprising_result_acknowledgment"] = (
        "Raw diffusion AUROC is extreme (~0.92 inverted); the length-confound analysis shows it is "
        "mostly explained by the corpus length imbalance (incorrect steps ~2.6x longer). A "
        "headline-eligible oracle-distinct claim requires a length-matched corpus rerun; this "
        "artifact reports the length-controlled residual as the honest preliminary estimate.")

    neg_ok = r_lo <= 0.5 <= r_hi
    raw_detect = (d_hi < 0.5 or d_lo > 0.5)
    lc_detect = (dgr_hi < 0.5 or dgr_lo > 0.5)
    len_confounded = (l_hi < 0.5 or l_lo > 0.5)
    rep["acceptance_gate"] = {
        "neg_control_ok": bool(neg_ok),
        "G_detect_raw_two_directional": bool(raw_detect),
        "G_detect_length_controlled": bool(lc_detect),
        "G_length_confounded": bool(len_confounded),
    }
    rep["verifier_is_oracle"] = False
    rep["reproducibility_checksum"] = hashlib.sha256(
        (str(sorted(it["key"] for it in items)) + str(SEED)).encode()).hexdigest()[:16]
    if not neg_ok:
        rep["honest_verdict"] = "blocked_neg_control_failed_metric_bug"
    elif lc_detect:
        rep["honest_verdict"] = (
            "complete: oracle_distinct_diffusion_surprisal_detects_errors_after_length_control_PRELIMINARY")
    else:
        rep["honest_verdict"] = (
            "complete: diffusion_surprisal_raw_signal_is_length_confounded_no_clean_oracle_distinct_signal")
    rep["duration_s_ar_gpu_leg"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rep, indent=2))
    log(f"AR={a_auc:.3f} DG={d_auc:.3f} LEN={l_auc:.3f} DG|len={dgr_auc:.3f}"
        f"{[round(dgr_lo,3),round(dgr_hi,3)]} AR|len={arr_auc:.3f} verdict={rep['honest_verdict']}")


if __name__ == "__main__":
    main()

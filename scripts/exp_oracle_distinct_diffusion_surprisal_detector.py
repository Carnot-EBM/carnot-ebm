#!/usr/bin/env python3
"""ORACLE-DISTINCT verifier-as-detector: diffusion-surprisal vs AR-perplexity on FoVer.

WHY THIS EXPERIMENT (the open frontier, 2026-06-14).
The night's central finding is the EXECUTION-MOAT: Carnot's verifier adds value where it can
EXECUTE (an oracle), but TIES majority vote where it relies on a LEARNED/ENERGY signal with no
cheap oracle. Every execution win is CIRCULAR (verifier == the executable oracle). The conductor's
own headline gate (.390 A2/A3) just blocked on exactly this: the ARC candidate pool has NO
per-candidate correctness labels, so an oracle-distinct LEARNED verifier could not even be trained
there (results/experiment_4209...: blocked_arc_pool_no_candidate_labels).

FoVer is the UNBLOCKED version of that frontier: its reasoning steps ARE labeled correct/incorrect.
So we can ask the load-bearing oracle-distinct question on a labeled corpus, no executable oracle:

  Can a LEARNED energy signal -- the pretrained DiffusionGemma denoiser's pseudo-surprisal
  (SEDD arXiv:2310.16834: the denoiser's per-position logits = the concrete score = -grad E) --
  DETECT a wrong reasoning step better than chance, and better than a MATCHED autoregressive
  perplexity baseline on the SAME backbone?

verifier_is_oracle = FALSE for BOTH scorers: neither executes anything; both are frozen-LM signals.
A win here is therefore HEADLINE-ELIGIBLE (non-circular) per the Circularity/Oracle-Distinctness
Discipline -- the first oracle-distinct detection signal if the diffusion AUROC CI excludes 0.5.

MATCHED COMPARISON (isolates the variable).
  - Diffusion score: google/diffusiongemma-26B-A4B-it  Q4_K_M  (block-diffusion denoiser)
  - AR baseline:     unsloth/gemma-4-26B-A4B-it-GGUF    Q4_K_M  (same backbone, same quant, AR)
The ONLY difference is AR-vs-diffusion. If diffusion's bidirectional energy-prior beats matched AR
(paired bootstrap CI on the AUROC difference excludes 0), the diffusion energy-prior adds
detection value the AR model cannot.

SCORES (both: higher => more error-like).
  - AR perplexity:    mean over step tokens of -log p(token | prefix, left-context)   [llama_cpp]
  - Diffusion pseudo-surprisal: constant non-empty prompt as conditioning; place the step tokens
    in the 256-canvas; MASK every 3rd position; mean over masked positions of -log p(true token).
    Masking is REQUIRED -- an observed token has ~0 surprisal. Keeping 2/3 of positions observed
    gives the diffusion model its bidirectional-context advantage (the thing being tested).

GATE (declared up front, falsifiable).
  G_detect: diffusion-surprisal AUROC bootstrap CI95 EXCLUDES 0.5  -> oracle-distinct detection works
  G_beats:  (diffusion AUROC - AR AUROC) paired bootstrap CI95 EXCLUDES 0 -> diffusion > matched AR
  Negative control: a random score must give AUROC ~ 0.5 (CI brackets 0.5) -- guards a metric bug.

inference_substrate: live_llm_inference (two frozen 26B GGUF models).
Output: results/oracle_distinct_diffusion_surprisal_detector.json
Per-item score cache (resumable): results/_cache_diffusion_detector_scores.jsonl
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_DG = "google/diffusiongemma-26B-A4B-it"
DG_GGUF = glob.glob(
    str(Path.home() / ".cache/huggingface/hub/models--unsloth--diffusiongemma-26B-A4B-it-GGUF/"
        "snapshots/*/diffusiongemma-26B-A4B-it-Q4_K_M.gguf")
)
AR_GGUF = glob.glob(
    str(Path.home() / ".cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/"
        "snapshots/*/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf")
)
DG_EVAL = Path.home() / ".cache/llama.cpp-master/build/bin/llama-diffusion-gemma-eval"
CORPUS = Path("data/fover_corpus.jsonl")
OUT = Path("results/oracle_distinct_diffusion_surprisal_detector.json")
CACHE = Path("results/_cache_diffusion_detector_scores.jsonl")
WD = Path("/tmp/dgemma_detector")

VOCAB = 262144
CANVAS = 256
MASK = 4
PROMPT = "Evaluate this math reasoning step:\n"
N_PER_CLASS = 80           # 80 correct + 80 incorrect = 160 (>> CLT/robust sample floors)
MIN_TOK, MAX_TOK = 10, 256
MASK_EVERY = 3
SEED = 4211
N_BOOT = 2000


def log(m: str) -> None:
    print(f"[detector] {m}", flush=True)


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """AUROC = P(score[pos] > score[neg]); rank-based (Mann-Whitney), ties at 0.5."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
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
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def boot_ci(scores: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    base = auroc(scores, y)
    n = len(y)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yi = y[idx]
        if yi.sum() == 0 or yi.sum() == n:
            continue
        vals.append(auroc(scores[idx], yi))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return base, float(lo), float(hi)


def boot_ci_diff(s_a: np.ndarray, s_b: np.ndarray, y: np.ndarray,
                 rng: np.random.Generator) -> tuple[float, float, float]:
    """Paired bootstrap CI for AUROC(a) - AUROC(b) (resample items, both recomputed together)."""
    base = auroc(s_a, y) - auroc(s_b, y)
    n = len(y)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yi = y[idx]
        if yi.sum() == 0 or yi.sum() == n:
            continue
        vals.append(auroc(s_a[idx], yi) - auroc(s_b[idx], yi))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return base, float(lo), float(hi)


def main() -> None:
    t0 = time.time()
    rep: dict = {
        "experiment": "oracle_distinct_diffusion_surprisal_detector",
        "schema": "carnot.oracle_distinct_detector.v1",
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": False,
        "verifier_is_oracle_note": "BOTH scorers are frozen-LM signals; neither executes anything. "
                                   "A win is non-circular / headline-eligible.",
        "random_seed": SEED,
        "corpus": str(CORPUS),
        "n_per_class_target": N_PER_CLASS,
        "models": {
            "diffusion": f"{REPO_DG} Q4_K_M (block-diffusion denoiser)",
            "ar_baseline": "unsloth/gemma-4-26B-A4B-it-GGUF UD-Q4_K_M (same backbone+quant, AR)",
        },
        "score_design": {
            "ar": "mean over step tokens of -log p(token | prefix) via llama_cpp logprobs",
            "diffusion": f"constant prompt + step in canvas; mask every {MASK_EVERY}th position; "
                         "mean -log p(true) at masked positions (SEDD score)",
            "direction": "higher => more error-like for both",
        },
        "gate": {
            "G_detect": "diffusion AUROC bootstrap CI95 excludes 0.5",
            "G_beats_ar": "(diffusion - AR) AUROC paired bootstrap CI95 excludes 0",
            "neg_control": "random-score AUROC CI brackets 0.5",
        },
        "field_principles": {
            "verifier_is_oracle": "Declares circular(true)/oracle-distinct(false). Only false wins "
                                  "are headline-eligible (Circularity/Oracle-Distinctness Discipline).",
            "random_seed": "Deterministic corpus subset + bootstrap; third party can re-run.",
            "duration_s": "Real two-26B-model compute; fabrication floor for live_llm_inference.",
        },
    }
    try:
        # ---- PRECONDITIONS (Pre-Launch Preconditions Discipline) ----
        pre = []
        pre.append({"resource": "diffusiongemma Q4_K_M GGUF", "available": bool(DG_GGUF)})
        pre.append({"resource": "gemma-4-26B-A4B Q4_K_M GGUF", "available": bool(AR_GGUF)})
        pre.append({"resource": "llama-diffusion-gemma-eval binary", "available": DG_EVAL.exists()})
        pre.append({"resource": "fover corpus", "available": CORPUS.exists()})
        rep["preconditions_checked"] = pre
        if not all(p["available"] for p in pre):
            rep["honest_verdict"] = "blocked_preconditions_missing"
            OUT.write_text(json.dumps(rep, indent=2))
            log(f"BLOCKED preconditions: {pre}")
            return

        from llama_cpp import Llama
        from transformers import AutoTokenizer

        WD.mkdir(parents=True, exist_ok=True)
        dg_gguf, ar_gguf = DG_GGUF[0], AR_GGUF[0]
        tok = AutoTokenizer.from_pretrained(REPO_DG)

        # ---- balanced corpus subset ----
        rows = [json.loads(l) for l in open(CORPUS)]
        rng = np.random.default_rng(SEED)

        def eligible(r):
            n = len(tok(str(r.get("step_text", "")), add_special_tokens=False).input_ids)
            return MIN_TOK <= n <= MAX_TOK

        cor = [r for r in rows if r.get("label") == "correct" and eligible(r)]
        inc = [r for r in rows if r.get("label") == "incorrect" and eligible(r)]
        rng.shuffle(cor)
        rng.shuffle(inc)
        k = min(N_PER_CLASS, len(cor), len(inc))
        items = [(r["step_text"], 0) for r in cor[:k]] + [(r["step_text"], 1) for r in inc[:k]]
        # stable order independent of label for fair sequential scoring
        items.sort(key=lambda it: hashlib.md5(it[0].encode()).hexdigest())
        rep["n_correct"] = k
        rep["n_incorrect"] = k
        rep["n_total"] = len(items)
        rep["base_rate_incorrect"] = round(sum(y for _, y in items) / len(items), 3)
        log(f"corpus: {k} correct + {k} incorrect = {len(items)} items")

        # ---- resumable score cache ----
        done: dict[str, dict] = {}
        if CACHE.exists():
            for line in open(CACHE):
                try:
                    d = json.loads(line)
                    done[d["key"]] = d
                except Exception:
                    pass
        cache_fh = open(CACHE, "a")

        def itemkey(text):
            return hashlib.md5(text.encode()).hexdigest()

        # ---- AR scorer (resident; GPU 1 to keep GPU 0 free for conductor) ----
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
        log("loading AR gemma-4-26B-A4B (GPU 1)...")
        ar = Llama(model_path=ar_gguf, n_gpu_layers=-1, logits_all=True, n_ctx=512, verbose=False)
        prompt_tok_n = len(ar.tokenize(PROMPT.encode(), add_bos=True))

        def ar_score(text):
            full = PROMPT + text
            out = ar.create_completion(full, max_tokens=0, echo=True, logprobs=1, temperature=0.0)
            tlp = out["choices"][0]["logprobs"]["token_logprobs"]
            step_lp = [x for x in tlp[prompt_tok_n:] if x is not None]
            if not step_lp:
                return None
            return float(-np.mean(step_lp))

        # ---- Diffusion scorer (per-call CLI) ----
        p_ids = np.array(tok(PROMPT, add_special_tokens=True).input_ids, dtype=np.int32)
        p_ids.tofile(WD / "p.i32")

        def dg_score(text):
            ids = tok(text, add_special_tokens=False).input_ids[:CANVAS]
            n = len(ids)
            masked = [i for i in range(n) if i % MASK_EVERY == 0]
            if not masked:
                return None
            canvas = np.full(CANVAS, MASK, dtype=np.int32)
            canvas[:n] = ids
            for i in masked:
                canvas[i] = MASK
            canvas.tofile(WD / "c.i32")
            outb = WD / "o.bin"
            outb.unlink(missing_ok=True)
            r = subprocess.run(
                [str(DG_EVAL), dg_gguf, str(WD / "p.i32"), str(WD / "c.i32"), str(outb)],
                capture_output=True, text=True, timeout=600,
                env={"CUDA_VISIBLE_DEVICES": "0", "PATH": "/usr/bin:/bin"},
            )
            if not outb.exists():
                return None
            L = np.fromfile(outb, dtype=np.float32).reshape(-1, VOCAB)
            su = []
            for i in masked:
                row = L[i]
                m = row.max()
                lse = m + np.log(np.exp(row - m).sum())
                su.append(float(lse - row[ids[i]]))
            return float(np.mean(su))

        # ---- score loop (checkpointed) ----
        for idx, (text, y) in enumerate(items):
            key = itemkey(text)
            if key in done and done[key].get("ar") is not None and done[key].get("dg") is not None:
                continue
            rec = done.get(key, {"key": key, "y": y})
            if rec.get("ar") is None:
                rec["ar"] = ar_score(text)
            if rec.get("dg") is None:
                rec["dg"] = dg_score(text)
            done[key] = rec
            cache_fh.write(json.dumps(rec) + "\n")
            cache_fh.flush()
            if (idx + 1) % 10 == 0 or idx == len(items) - 1:
                log(f"scored {idx + 1}/{len(items)}  ({time.time() - t0:.0f}s elapsed)")

        cache_fh.close()

        # ---- assemble arrays ----
        ys, ars, dgs = [], [], []
        for text, y in items:
            rec = done[itemkey(text)]
            if rec.get("ar") is None or rec.get("dg") is None:
                continue
            ys.append(y)
            ars.append(rec["ar"])
            dgs.append(rec["dg"])
        y = np.array(ys)
        ar_s = np.array(ars)
        dg_s = np.array(dgs)
        rep["n_scored"] = int(len(y))
        rep["n_incorrect_scored"] = int(y.sum())

        # ---- sanity: per-class means (incorrect should be higher) ----
        rep["ar_mean_correct"] = round(float(ar_s[y == 0].mean()), 4)
        rep["ar_mean_incorrect"] = round(float(ar_s[y == 1].mean()), 4)
        rep["dg_mean_correct"] = round(float(dg_s[y == 0].mean()), 4)
        rep["dg_mean_incorrect"] = round(float(dg_s[y == 1].mean()), 4)

        # ---- AUROC + bootstrap CIs ----
        brng = np.random.default_rng(SEED + 1)
        a_auc, a_lo, a_hi = boot_ci(ar_s, y, brng)
        d_auc, d_lo, d_hi = boot_ci(dg_s, y, brng)
        diff, dl, dh = boot_ci_diff(dg_s, ar_s, y, brng)
        rng_neg = np.random.default_rng(SEED + 7)
        rand_s = rng_neg.standard_normal(len(y))
        r_auc, r_lo, r_hi = boot_ci(rand_s, y, brng)

        rep["ar_auroc"] = round(a_auc, 4)
        rep["ar_auroc_ci95"] = [round(a_lo, 4), round(a_hi, 4)]
        rep["diffusion_auroc"] = round(d_auc, 4)
        rep["diffusion_auroc_ci95"] = [round(d_lo, 4), round(d_hi, 4)]
        rep["diff_minus_ar_auroc"] = round(diff, 4)
        rep["diff_minus_ar_ci95"] = [round(dl, 4), round(dh, 4)]
        rep["neg_control_random_auroc"] = round(r_auc, 4)
        rep["neg_control_random_ci95"] = [round(r_lo, 4), round(r_hi, 4)]

        # ---- gate evaluation ----
        g_detect = d_lo > 0.5
        g_beats = dl > 0.0
        neg_ok = r_lo <= 0.5 <= r_hi
        rep["acceptance_gate"] = {
            "G_detect_diffusion_excludes_0.5": bool(g_detect),
            "G_beats_ar_diff_excludes_0": bool(g_beats),
            "neg_control_ok": bool(neg_ok),
        }
        rep["duration_s"] = round(time.time() - t0, 1)
        rep["reproducibility_checksum"] = hashlib.sha256(
            (str(sorted(itemkey(t) for t, _ in items)) + str(SEED)).encode()
        ).hexdigest()[:16]

        if not neg_ok:
            rep["honest_verdict"] = "blocked_neg_control_failed_metric_bug"
        elif g_detect and g_beats:
            rep["honest_verdict"] = (
                "complete: oracle_distinct_diffusion_surprisal_detects_errors_AND_beats_matched_AR"
            )
        elif g_detect:
            rep["honest_verdict"] = (
                "complete: oracle_distinct_diffusion_surprisal_detects_errors_ties_AR"
            )
        else:
            rep["honest_verdict"] = (
                "complete: oracle_distinct_diffusion_surprisal_no_detection_signal_honest_null"
            )
        log(f"AR AUROC={a_auc:.3f}{rep['ar_auroc_ci95']}  "
            f"DG AUROC={d_auc:.3f}{rep['diffusion_auroc_ci95']}  "
            f"diff={diff:.3f}{rep['diff_minus_ar_ci95']}")
    except Exception as e:
        rep["error"] = repr(e)[:300]
        rep["traceback"] = traceback.format_exc()[-1500:]
        rep["honest_verdict"] = "blocked_detector_error"
        rep["duration_s"] = round(time.time() - t0, 1)

    OUT.write_text(json.dumps(rep, indent=2))
    log(f"DONE verdict={rep.get('honest_verdict')} -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())

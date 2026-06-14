#!/usr/bin/env python3
"""DiffusionGemma energy-prior EXTRACTION (UC3 [REAL] feasibility — ACHIEVED 2026-06-14).

The pretrained discrete-diffusion denoiser IS a score/energy model (SEDD, arXiv:2310.16834):
its per-position logits = the concrete score (-grad E component) over token sequences. This
script extracts that score from the FROZEN DiffusionGemma weights -- the deep-EBM energy-prior
Phase-3 has been trying to train from scratch, now obtained for free.

WORKING PATH (after 4 released-runtime dead-ends: transformers meta-tensor/49GB-on-48GB,
llama.cpp-0.3.29 arch-missing, vLLM-0.23.0 no-native-runner, vLLM-recipe H100/H200-only):
  - llama.cpp built from PR #24423 (adds the diffusion-gemma arch + llama-diffusion-gemma-eval),
    CUDA 13.3 + gcc 16, GGML_CUDA=on.
  - Q4_K_M GGUF (16.8GB) loads on ONE RTX 3090 (-ngl 99) -- fits with headroom; no Hopper/fp8.
  - llama-diffusion-gemma-eval <gguf> <prompt_ids.i32> <canvas_ids.i32> <out_logits.bin> dumps
    the [canvas_length x vocab] float32 logits = the score. canvas MUST be diffusion.canvas_length
    (256) mask tokens (mask_token id = 4).

verifier_is_oracle: N/A (generator introspection, not a verifier-value claim).
Output: results/diffusiongemma_energy_prior_extracted.json
"""
from __future__ import annotations

import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

REPO = "google/diffusiongemma-26B-A4B-it"
BIN = Path.home() / ".cache/llama.cpp-master/build/bin/llama-diffusion-gemma-eval"
GGUF_GLOB = str(
    Path.home()
    / ".cache/huggingface/hub/models--unsloth--diffusiongemma-26B-A4B-it-GGUF/snapshots/*/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
)
VOCAB = 262144
CANVAS_LEN = 256
MASK_TOKEN = 4
WD = Path("/tmp/dgemma_score")
OUT = Path("/home/ianblenke/github.com/ianblenke/carnot/results/diffusiongemma_energy_prior_extracted.json")


def main() -> None:
    t0 = time.time()
    rep: dict = {
        "experiment": "diffusiongemma_energy_prior_extract",
        "repo": REPO,
        "inference_substrate": "live_llm_inference",
        "runtime": "llama.cpp PR#24423 (diffusion-gemma) CUDA, Q4_K_M GGUF on 1x RTX 3090",
        "verifier_is_oracle": None,
        "prompt": "def add(a, b):\n    return a + b\n# add(2,3) ==",
    }
    try:
        import glob

        gguf = glob.glob(GGUF_GLOB)[0]
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(REPO)
        WD.mkdir(parents=True, exist_ok=True)
        ids = tok(rep["prompt"], add_special_tokens=True).input_ids
        np.array(ids, dtype=np.int32).tofile(WD / "prompt_ids.i32")
        np.full(CANVAS_LEN, MASK_TOKEN, dtype=np.int32).tofile(WD / "canvas_ids.i32")
        out_logits = WD / "out_logits.bin"
        out_logits.unlink(missing_ok=True)

        proc = subprocess.run(
            [str(BIN), gguf, str(WD / "prompt_ids.i32"), str(WD / "canvas_ids.i32"), str(out_logits)],
            capture_output=True, text=True, timeout=300,
            env={"CUDA_VISIBLE_DEVICES": "0", "PATH": "/usr/bin:/bin"},
        )
        rep["eval_rc"] = proc.returncode
        if not out_logits.exists():
            rep["honest_verdict"] = "blocked_eval_no_output"
            rep["stderr_tail"] = proc.stderr[-400:]
            OUT.write_text(json.dumps(rep, indent=2))
            print("[extract] no out_logits", flush=True)
            return

        a = np.fromfile(out_logits, dtype=np.float32)
        L = a.reshape(-1, VOCAB)
        rep["score_shape"] = list(L.shape)
        rep["score_finite"] = bool(np.isfinite(L).all())
        rep["score_absmax"] = round(float(np.abs(L).max()), 3)
        top = np.argsort(-L[0])[:6]
        rep["pos0_top_tokens"] = [[tok.decode([int(t)]), round(float(L[0][t]), 2)] for t in top]
        rep["score_extracted"] = True
        rep["honest_verdict"] = "complete: energy_prior_score_extracted_from_frozen_diffusiongemma"
    except Exception as e:
        rep["score_extracted"] = False
        rep["error"] = repr(e)[:300]
        rep["traceback"] = traceback.format_exc()[-1200:]
        rep["honest_verdict"] = "blocked_extract_error"

    rep["duration_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rep, indent=2))
    print(f"[extract] DONE verdict={rep.get('honest_verdict')} shape={rep.get('score_shape')}", flush=True)


if __name__ == "__main__":
    main()

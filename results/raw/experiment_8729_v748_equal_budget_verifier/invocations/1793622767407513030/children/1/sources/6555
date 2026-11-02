#!/usr/bin/env python3
"""DiffusionGemma energy-prior probe via the pre-quantized Q4_K_M GGUF (llama.cpp).

Operator-directed 2026-06-14: the transformers 4-bit path loaded but the diffusion forward
API + device-map meta-tensor offload blocked score extraction. The unsloth Q4_K_M GGUF
(16.8GB, fits one 3090) loaded by llama.cpp sidesteps all of that AND exposes per-position
vocab logits directly -- which IS the score object (-grad E over tokens; the denoiser's
learned distribution). This probe confirms: (1) llama.cpp supports the diffusion-gemma arch
(loads), (2) we can extract a per-position score/logit distribution from the frozen weights.

Honest scope: this is the FEASIBILITY step for UC3 [REAL] (is the pretrained energy-prior
accessible?). It does NOT yet do proper noise-level denoising-score extraction or compose
with the Carnot verifier -- it confirms the score is extractable, the precondition for all of
that. verifier_is_oracle: N/A (generator introspection, not a verifier-value claim).

Output: results/diffusiongemma_energy_prior_gguf.json
"""
from __future__ import annotations

import glob
import json
import math
import time
import traceback
from pathlib import Path

GGUF_GLOB = "/home/ianblenke/.cache/huggingface/hub/models--unsloth--diffusiongemma-26B-A4B-it-GGUF/snapshots/*/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
OUT = Path("/home/ianblenke/github.com/ianblenke/carnot/results/diffusiongemma_energy_prior_gguf.json")


def main() -> None:
    t0 = time.time()
    rep: dict = {
        "experiment": "diffusiongemma_energy_prior_gguf",
        "inference_substrate": "live_llm_inference",
        "quant": "Q4_K_M_gguf",
        "verifier_is_oracle": None,
    }
    try:
        paths = glob.glob(GGUF_GLOB)
        if not paths:
            rep["honest_verdict"] = "blocked_gguf_not_found"
            OUT.write_text(json.dumps(rep, indent=2))
            print("[gguf probe] GGUF not found", flush=True)
            return
        gguf = paths[0]
        rep["gguf_path"] = gguf
        from llama_cpp import Llama

        # n_gpu_layers=-1 offloads all layers if the build is CUDA; falls back to CPU otherwise.
        llm = Llama(
            model_path=gguf,
            n_gpu_layers=-1,
            logits_all=True,
            n_ctx=512,
            verbose=False,
        )
        rep["model_loaded"] = True
        rep["arch_supported"] = True  # if it loaded, llama.cpp recognized diffusion-gemma
        rep["n_vocab"] = int(llm.n_vocab())

        prompt = "def add(a, b):\n    return a + b\n# add(2,3) =="
        # create_completion with logprobs surfaces the per-token distribution (the score) at
        # the generation position -- a version-stable way to confirm score extractability.
        out = llm.create_completion(prompt, max_tokens=1, logprobs=10, temperature=0.0)
        rep["completion_token"] = out["choices"][0].get("text", "")
        lp = out["choices"][0].get("logprobs", {}) or {}
        top = (lp.get("top_logprobs") or [{}])[0]
        # top is {token: logprob}; logprob = log p(x_i|context) = the per-position score component
        rep["top10_token_logprob"] = sorted(top.items(), key=lambda kv: -kv[1])[:10]
        rep["score_extracted"] = bool(top)

        # also pull the raw last-position logit vector if available (the full score vector)
        try:
            scores = getattr(llm, "scores", None)
            if scores is not None:
                import numpy as np

                arr = np.asarray(scores)
                # last non-zero row = last evaluated position
                rep["raw_logits_shape"] = list(arr.shape)
                last = arr[arr.any(axis=1)][-1] if arr.ndim == 2 and arr.any() else None
                if last is not None:
                    rep["raw_logits_finite"] = bool(np.isfinite(last).all())
                    rep["raw_logits_absmax"] = float(np.abs(last).max())
        except Exception as e_raw:
            rep["raw_logits_note"] = repr(e_raw)[:150]

        rep["honest_verdict"] = (
            "complete: energy_prior_accessible_via_gguf_score_extracted"
            if rep["score_extracted"]
            else "complete: gguf_loaded_score_extraction_incomplete"
        )
    except Exception as e:
        rep["model_loaded"] = False
        rep["error"] = repr(e)[:300]
        rep["traceback"] = traceback.format_exc()[-1500:]
        # distinguish "arch unsupported" from other load errors
        if "architecture" in repr(e).lower() or "unknown model" in repr(e).lower():
            rep["arch_supported"] = False
            rep["honest_verdict"] = "blocked_llamacpp_arch_unsupported_needs_update"
        else:
            rep["honest_verdict"] = "blocked_gguf_load_failed"

    rep["duration_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rep, indent=2))
    print(f"[gguf probe] DONE verdict={rep.get('honest_verdict')} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

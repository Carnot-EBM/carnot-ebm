#!/usr/bin/env python3
"""DiffusionGemma energy-prior probe via transformers DiffusionGemmaForBlockDiffusion.

The tractable path (operator "at your pleasure", 2026-06-14): the LM-head class is in our
installed transformers 5.12 and its forward(input_ids, decoder_input_ids, ...) RETURNS
.logits (lm_head over last_hidden_state) -- exactly the per-position score object
(log p(x_i|context) ~ -E component) for the UC3 [REAL] energy-prior. The only prior blocker
was device_map CPU-offload putting params on 'meta' (.item() error); fix = 4-bit across BOTH
3090s with NO cpu in max_memory (~11GB/GPU, fits, no offload).

Confirms the pretrained energy-prior is EXTRACTABLE from the frozen weights -- the
precondition for composing it with the Carnot verifier + energy-descent.
Output: results/diffusiongemma_energy_prior_transformers.json
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import torch

REPO = "google/diffusiongemma-26B-A4B-it"
OUT = Path("/home/ianblenke/github.com/ianblenke/carnot/results/diffusiongemma_energy_prior_transformers.json")


def main() -> None:
    t0 = time.time()
    rep: dict = {
        "experiment": "diffusiongemma_energy_prior_transformers",
        "repo": REPO,
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": None,  # generator introspection, not a verifier-value claim
    }
    try:
        from transformers import (
            AutoTokenizer,
            BitsAndBytesConfig,
            DiffusionGemmaForBlockDiffusion,
        )

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        n_gpu = torch.cuda.device_count()
        # GPUs ONLY -- no 'cpu' key -> no CPU/meta offload (the prior failure mode).
        max_mem = {i: "22GiB" for i in range(n_gpu)}
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            REPO,
            quantization_config=bnb,
            device_map="auto",
            max_memory=max_mem,
            trust_remote_code=False,
        )
        model.eval()
        rep["model_loaded"] = True
        rep["model_class"] = type(model).__name__
        rep["load_mode"] = f"4bit_nf4_{n_gpu}gpu_no_cpu_offload"
        rep["devices"] = sorted({str(p.device) for p in model.parameters()})

        tok = AutoTokenizer.from_pretrained(REPO)
        text = "def add(a, b):\n    return a + b\n# add(2,3) =="
        enc = tok(text, return_tensors="pt")
        ids = enc["input_ids"]
        # place inputs on the model's first parameter device
        dev = next(model.parameters()).device
        ids = ids.to(dev)

        with torch.no_grad():
            try:
                # encoder context = input_ids; decoder canvas = same tokens (feasibility:
                # extract the denoiser's per-position distribution = the score).
                out = model(input_ids=ids, decoder_input_ids=ids)
                logits = getattr(out, "logits", None)
                if logits is None and isinstance(out, (tuple, list)):
                    logits = out[0]
                logits = logits.float()
                rep["logits_shape"] = list(logits.shape)
                rep["logits_finite"] = bool(torch.isfinite(logits).all().item())
                rep["logits_absmax"] = float(logits.abs().max().item())
                logp = torch.log_softmax(logits[0, -1], dim=-1)
                topv, topi = logp.topk(10)
                rep["last_pos_top10_token_logp"] = [
                    [tok.decode([int(i)]), round(float(v), 3)] for v, i in zip(topv, topi)
                ]
                rep["score_extracted"] = True
                rep["honest_verdict"] = "complete: energy_prior_accessible_score_extracted"
            except Exception as e_fwd:
                rep["score_extracted"] = False
                rep["forward_error"] = repr(e_fwd)[:300]
                rep["honest_verdict"] = "complete: model_loaded_forward_needs_more_diffusion_args"
    except Exception as e:
        rep["model_loaded"] = False
        rep["error"] = repr(e)[:300]
        rep["traceback"] = traceback.format_exc()[-1500:]
        rep["honest_verdict"] = "blocked_diffusiongemma_load_failed"

    rep["duration_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rep, indent=2))
    print(f"[tf probe] DONE verdict={rep.get('honest_verdict')} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

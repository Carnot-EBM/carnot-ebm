#!/usr/bin/env python3
"""DiffusionGemma energy-prior FEASIBILITY probe (UC3 [REAL], 2026-06-14).

The operator-flagged groundbreaking prospect: a pretrained discrete-diffusion denoiser IS a
score/energy model by construction (SEDD, arXiv:2310.16834) — its output is the concrete
score = -grad E(x) over token sequences. If we can extract that score from DiffusionGemma's
frozen weights, we get a pretrained deep energy-prior over valid sequences for FREE — exactly
the object Phase-3 has been trying to TRAIN from scratch.

This probe is step 0: does the model LOAD, and can we EXTRACT a per-position score/logit
object from a forward pass? It does NOT yet compose with the Carnot verifier or do
energy-descent — it only confirms the score is accessible (the precondition for everything
downstream). Honest, bounded, no over-claim.

Output: results/diffusiongemma_energy_prior_probe.json
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import torch

REPO = "google/diffusiongemma-26B-A4B-it"
OUT = Path("/home/ianblenke/github.com/ianblenke/carnot/results/diffusiongemma_energy_prior_probe.json")


def main() -> None:
    t0 = time.time()
    rep: dict = {
        "experiment": "diffusiongemma_energy_prior_probe",
        "repo": REPO,
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": None,  # N/A: this is a generator-introspection probe, not a verifier-value claim
    }
    try:
        from transformers import AutoModel, AutoTokenizer

        load_mode = None
        model = None
        # Prefer 4-bit (fits one 3090 ~13-15GB); fall back to device_map across both GPUs.
        try:
            from transformers import BitsAndBytesConfig

            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModel.from_pretrained(
                REPO, quantization_config=bnb, device_map={"": 0}, trust_remote_code=False
            )
            load_mode = "4bit_nf4_gpu0"
        except Exception as e_bnb:
            rep["bnb_fallback_reason"] = repr(e_bnb)[:200]
            model = AutoModel.from_pretrained(
                REPO, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=False
            )
            load_mode = "bf16_device_map_auto"

        rep["load_mode"] = load_mode
        rep["model_class"] = type(model).__name__
        rep["model_loaded"] = True
        model.eval()

        tok = AutoTokenizer.from_pretrained(REPO)
        text = "def add(a, b):\n    return a + b\n# add(2,3) =="
        enc = tok(text, return_tensors="pt")
        dev = next(model.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}

        # Minimal forward: ask for logits. Discrete-diffusion denoisers expose per-position
        # vocab logits (the score is derived from these). Try the plain forward first; if the
        # block-diffusion API needs extra args, capture the signature for follow-up.
        score_extracted = False
        with torch.no_grad():
            try:
                out = model(**enc)
                logits = getattr(out, "logits", None)
                if logits is None and isinstance(out, (tuple, list)):
                    logits = out[0]
                if logits is not None:
                    logits = logits.float()
                    rep["logits_shape"] = list(logits.shape)
                    rep["logits_finite"] = bool(torch.isfinite(logits).all().item())
                    rep["logits_absmax"] = float(logits.abs().max().item())
                    # the "score" object: log-softmax over vocab = log p(x_i | context) per position
                    logp = torch.log_softmax(logits[0, -1], dim=-1)
                    topv, topi = logp.topk(5)
                    rep["last_pos_top5_token_logp"] = [
                        [tok.decode([int(i)]), round(float(v), 3)] for v, i in zip(topv, topi)
                    ]
                    score_extracted = True
            except Exception as e_fwd:
                rep["forward_error"] = repr(e_fwd)[:300]
                import inspect

                try:
                    rep["forward_signature"] = str(inspect.signature(model.forward))
                except Exception:
                    pass

        rep["score_extracted"] = score_extracted
        rep["honest_verdict"] = (
            "complete: energy_prior_accessible_score_extracted"
            if score_extracted
            else "complete: model_loaded_score_extraction_needs_diffusion_api"
        )
    except Exception as e:
        rep["model_loaded"] = False
        rep["error"] = repr(e)[:300]
        rep["traceback"] = traceback.format_exc()[-1500:]
        rep["honest_verdict"] = "blocked_diffusiongemma_load_failed"

    rep["duration_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rep, indent=2))
    print(f"[energy-prior probe] DONE verdict={rep.get('honest_verdict')} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

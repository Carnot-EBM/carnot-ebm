#!/usr/bin/env python3
"""DiffusionGemma energy-prior probe via vLLM (operator-directed 2026-06-14).

vLLM has native DiffusionGemma support (vllm.ai/blog/2026-06-10-diffusion-gemma), sidestepping
the transformers diffusion-API/meta-tensor issues AND the llama.cpp arch gap (released
llama-cpp-python 0.3.29 lacks 'diffusion-gemma'). vLLM exposes per-position PROMPT LOGPROBS,
which ARE the score object we want: logprob(x_i | context) = a component of -E(x). This probe
confirms the pretrained energy-prior is EXTRACTABLE from the frozen weights (UC3 [REAL]
feasibility) -- the precondition for composing it with the Carnot verifier + energy-descent.

Run with the ISOLATED vllm venv: /home/ianblenke/.cache/vllm-venv/bin/python.
Memory: 26B on 2x24GB -> tensor_parallel_size=2 + fp8 (online) + tiny max_model_len.
Output: results/diffusiongemma_energy_prior_vllm.json
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

REPO = "google/diffusiongemma-26B-A4B-it"
OUT = Path("/home/ianblenke/github.com/ianblenke/carnot/results/diffusiongemma_energy_prior_vllm.json")


def main() -> None:
    t0 = time.time()
    rep: dict = {
        "experiment": "diffusiongemma_energy_prior_vllm",
        "repo": REPO,
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": None,  # generator introspection, not a verifier-value claim
    }
    try:
        from vllm import LLM, SamplingParams

        import vllm

        rep["vllm_version"] = vllm.__version__
        load_mode = None
        llm = None
        # fp8 TP=2 fits 26B on 2x3090 (~13GB/GPU) with room for a tiny KV cache.
        for kw, tag in [
            (dict(tensor_parallel_size=2, quantization="fp8", max_model_len=1024,
                  gpu_memory_utilization=0.88, enforce_eager=True), "fp8_tp2"),
            (dict(tensor_parallel_size=2, max_model_len=512, gpu_memory_utilization=0.92,
                  enforce_eager=True), "bf16_tp2"),
        ]:
            try:
                llm = LLM(model=REPO, dtype="auto", trust_remote_code=False, **kw)
                load_mode = tag
                break
            except Exception as e_load:
                rep[f"load_attempt_{tag}_error"] = repr(e_load)[:200]
        if llm is None:
            rep["model_loaded"] = False
            rep["honest_verdict"] = "blocked_vllm_load_failed"
            OUT.write_text(json.dumps(rep, indent=2))
            print(f"[vllm probe] DONE verdict={rep['honest_verdict']}", flush=True)
            return

        rep["model_loaded"] = True
        rep["load_mode"] = load_mode

        # prompt_logprobs returns, per prompt position, a dict of {token_id: Logprob} = the
        # model's learned per-position distribution = the score component log p(x_i | context).
        prompt = "def add(a, b):\n    return a + b\n# add(2,3) =="
        sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=10)
        out = llm.generate([prompt], sp)
        o = out[0]
        plp = o.prompt_logprobs  # list[None | dict[int, Logprob]]
        nz = [d for d in (plp or []) if d]
        rep["n_prompt_positions_with_logprobs"] = len(nz)
        if nz:
            last = nz[-1]
            top = sorted(last.items(), key=lambda kv: -kv[1].logprob)[:10]
            rep["last_pos_top10"] = [
                [getattr(lp, "decoded_token", str(tid)), round(float(lp.logprob), 3)]
                for tid, lp in top
            ]
            rep["score_extracted"] = True
        else:
            rep["score_extracted"] = False
        rep["honest_verdict"] = (
            "complete: energy_prior_accessible_via_vllm_score_extracted"
            if rep.get("score_extracted")
            else "complete: vllm_loaded_prompt_logprobs_empty"
        )
    except Exception as e:
        rep["model_loaded"] = rep.get("model_loaded", False)
        rep["error"] = repr(e)[:300]
        rep["traceback"] = traceback.format_exc()[-1500:]
        rep["honest_verdict"] = "blocked_vllm_probe_error"

    rep["duration_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rep, indent=2))
    print(f"[vllm probe] DONE verdict={rep.get('honest_verdict')} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

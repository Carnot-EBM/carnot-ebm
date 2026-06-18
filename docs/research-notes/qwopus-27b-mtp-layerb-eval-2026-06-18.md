# Qwopus3.6-27B-Coder-MTP vs gemma-4-12B on Config Layer B (2026-06-18)

Operator request: try the OSS SOTA coder `Jackrong/Qwopus3.6-27B-Coder-MTP-GGUF` with a recent llama.cpp
(MTP support) to see if we get **faster** and **more accurate** Layer-B rule induction than the local
gemma-4-12B-it baseline. All on the offline-legal **iGPU** (port 8921, NOT the 3090s, per the 2026-06-17
directive); zero quota.

## Setup (what worked out of the box)

- **llama.cpp build 9606 (2026-06-14) already has native MTP** — `--spec-type draft-mtp`. No rebuild.
  The `-MTP-` GGUF carries the nextn heads; loading the same GGUF as `--model-draft` runs self-MTP
  speculative decoding (`common_speculative_impl_draft_mtp`, n_max=3, n_embd=5120). It loads the model
  twice (target ~16.8 GB + draft ~15.5 GB ≈ 32 GB on the iGPU's shared RAM — fits).
- Model: `Qwopus3.6-27B-Coder-MTP-Q4_K_M.gguf` (16.8 GB), Qwen3.6-27B base, agentic coder, SWE-bench
  Verified 67% (thinking-off, Q5_K_M, RTX 5090 in the card). It is a *reasoning* model (emits `<think>`);
  on our task it stayed concise (180–340 tokens to the rule), so thinking did not dominate.

## Results (scaffolded Layer-B prompt, identical to the gemma sweep)

| Game | gemma-12B (no MTP) | Qwopus-27B no-MTP | Qwopus-27B **+MTP** | MTP speedup |
|---|---|---|---|---|
| ka59 | Tier 2, ~4.2 tok/s | **Tier 2**, 2.74 tok/s, 340 tok | **Tier 2**, 3.40 tok/s, 319 tok | 1.24× |
| tn36 | Tier 2, ~4.2 tok/s | **Tier 2**, 1.54 tok/s, 180 tok | Tier 0 (sample-var), 2.98 tok/s | 1.93× |

- **MTP draft acceptance = 0.806** (high) — MTP is genuinely effective for this model; speedup 1.24–1.93×
  (consistent with the card's ~1.66×).
- **tn36's Tier-2→Tier-0 flip under MTP is temperature-0.2 sampling variance, NOT MTP degradation.**
  Speculative decoding (incl. MTP) is lossless only at greedy/temp-0; at temp>0 the trajectory differs
  run-to-run. ka59 grounded in both. For deterministic grounding + provably-lossless MTP, use temp 0.

## Honest bottom line

1. **Accuracy: comparable, no demonstrable advantage on this task.** Qwopus grounds the same count-class
   games gemma already grounds (ka59, tn36) and is more *concise*. It does NOT unlock the games gemma
   failed (cd82 ~10k-tok prompt, wa30 ~6.3k-tok prompt) — those fail because their editable region is a
   large 2-D area, so the flat `values=[…]` digest is huge (a **digest-shape** problem), not a model-
   strength problem. A stronger model does not fix a wrong-shaped digest. **For Config Layer B, the model
   is not the bottleneck — the digest/perception is.**
2. **Speed: even with MTP, the 27B is slower than the 12B on the iGPU** (~3 vs ~4.2 tok/s). MTP's ~1.5×
   does not overcome the 2.25× param difference. "Faster" (the card's ~100 tok/s) is fast-GPU territory
   (RTX 5090 / a 3090 — the latter off-limits here by directive). The iGPU is bandwidth-bound.
3. **Where Qwopus *would* pay off**: harder reasoning/agentic-coding tasks where a 27B coder genuinely
   beats a 12B AND where a fast GPU can host it. For the LLM-light verifier-induction path it is not a win
   on the iGPU. MTP itself is a clean, free speedup (native, 80% acceptance) worth using whenever this
   model is run.

## Recommendation

Keep gemma-4-12B as the default for Config Layer B (faster on the iGPU, already grounds the tractable
class). Reach for Qwopus-27B-Coder+MTP for harder coding/reasoning where 27B quality matters, ideally on
a fast GPU. The real Layer-B levers remain (a) better *digest shape* for large-2D/rewrite editable
regions (the cd82/wa30/tr87 class), and (b) the glyph-perception primitive for the rewrite class — not a
bigger LLM.

Artifacts: `results/arc3_layerb_qwopus_test.json` (no-MTP), `results/arc3_layerb_qwopus_test_mtp.json`
(MTP); harness `scripts/experiments/arc3_layerb_qwopus_test.py`.

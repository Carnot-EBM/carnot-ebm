# Config Layer-B model comparison: gemma-12B vs Qwopus-27B+MTP vs DiffusionGemma-26B-A4B (2026-06-18)

Operator-requested head-to-head on the Config Layer-B task (induce a grounded `is_win(grid)` predicate
from the scaffolded structured-extraction prompt). All three run on the offline-legal **iGPU** (ROCm/HIP),
never the 3090s (2026-06-17 directive); zero quota. Grounding = the induced predicate fires True on the
banked win and False on the non-wins (same verifier for all three). Reference games: ka59 + tn36 (the
count/relation class both small enough to fit the prompt; gemma grounds both).

## Results

| Model | Substrate / runner | Speed (iGPU) | ka59 | tn36 | Driveability |
|---|---|---|---|---|---|
| **gemma-4-12B-it** (AR) | llama-server, raw `/completion` | ~4.2 tok/s | **Tier 2 ✅** | **Tier 2 ✅** | easiest — no thinking, raw completion |
| **Qwopus3.6-27B-Coder** (AR coder) | llama-server `+--spec-type draft-mtp` | 2.74→**3.4** tok/s (MTP 1.24–1.93×, 80.6% accept) | **Tier 2 ✅** | Tier 2 (no-MTP) / Tier 0 (MTP, temp-0.2 variance) | easy; more concise (180–340 tok) |
| **DiffusionGemma-26B-A4B** (block-diffusion MoE) | `llama-diffusion-cli` (HIP, self-built) | **5–18 tok/s raw**, parallel denoise (120–203 tok/s in-step) | Tier 0 (reasons, no committed code) | Tier 0 | **hard** — reasoning-channel + fixed canvas |

## Per-model findings

**gemma-4-12B** — the baseline and still the best fit for Layer B: grounds the tractable class, simplest
to drive, fastest *effective* on the actual task. It is not a coder but the scaffolded digest does the
heavy lifting.

**Qwopus3.6-27B-Coder + MTP** — `llama.cpp` build 9606 already has native MTP (`--spec-type draft-mtp`,
no rebuild); self-MTP hit **80.6% draft acceptance → ~1.5× speedup** (matches the card's 1.66×). Grounds
the same games, more concisely. But even with MTP a 27B is **slower than the 12B on the iGPU** (~3 vs
4.2 tok/s) — MTP's 1.5× does not overcome the 2.25× param gap; the card's ~100 tok/s is fast-GPU
territory. No accuracy advantage on this task (gemma already grounds the tractable class). The tn36
MTP→Tier0 flip is temperature-0.2 sampling variance, not MTP loss (spec-decoding is lossless only at
greedy).

**DiffusionGemma-26B-A4B** — structurally the **fastest** (denoises the whole 256-token canvas in
parallel; 18 tok/s effective on a simple prompt, 4× the AR models) and a 26B-MoE/4B-active. We built
`llama-diffusion-cli` for the iGPU/HIP backend to run it (standard llama-server cannot generate from a
block-diffusion model). BUT on the Layer-B task it does **not** ground: it is a *reasoning* model
(`<|channel>thought`) and the diffusion canvas is **fixed-length**, so it fills the canvas with
chain-of-thought ("Wait, where does the 6 come from?") and never commits to the code. It DID write the
exact correct rule (`np.sum(grid[63,26:64]==4)==32`) on a *simple* "output only code" prompt — so the
capability is there; the failure is the **reasoning-vs-fixed-canvas mismatch**. Four mitigations failed
to fix it via `llama-diffusion-cli`: larger canvas (it loops longer), an explicit no-think directive (the
thought-channel is trained in, a raw-prompt instruction can't override it), and channel-marker priming
(the canvas still denoises into reasoning).

## The DiffusionGemma reframe (why this is not a knock on the model)

The project does **not** use DiffusionGemma as a generator — it uses it as an **energy-prior / score
signal** (per-position vocab logits = the denoiser's learned distribution = −∇E over tokens; see
`scripts/exp_diffusiongemma_energy_prior_gguf.py` and the pending
`diffusiongemma-energy-guided-diffusion-spec.md` gate). That is verifier-side — Carnot's actual thesis.
This Layer-B test ran it **off-label** as an autoregressive-style code generator, where its reasoning
canvas is a poor fit. Its real value here is the diffusion *energy* it exposes, not CoT code generation.

## DECISIVE constraint: the Kaggle validation environment (16 GB VRAM)

The speed/accuracy analysis below is secondary. The binding constraint is **deployment fit**: the ARC
Prize Kaggle validation environment has **16 GB VRAM**, and the model must fit **with adequate KV cache**.

| Model | Q4_K_M weights | Free on a 16 GB card | Verdict |
|---|---|---|---|
| **gemma-4-12B-it** | **6.7 GB** | ~9.3 GB for KV cache + activations | **FITS comfortably** |
| Qwopus-27B-Coder | 16.8 GB | **−0.8 GB** (weights alone exceed 16 GB) | does NOT fit |
| DiffusionGemma-26B-A4B | 16.8 GB | **−0.8 GB** | does NOT fit |

So **gemma-4-12B is the only one of the three deployable in the challenge environment**, independent of
its speed/accuracy merits. The 27B models are off the table for Kaggle deployment at Q4 (a smaller quant
would degrade quality and still leave little KV headroom). This is why gemma-4-12B is the model to build
the ARC pipeline around: it fits 16 GB with ~9 GB of KV-cache headroom, which is what the interactive
ARC-AGI-3 harness needs (long contexts, per-step verification). The 27B models remain useful for OFFLINE
development / research on the local rig, not for the validation submission.

## Bottom line + recommendation

- **For Config Layer B: keep gemma-4-12B as the default.** It grounds the tractable class, is the
  easiest to drive, and is the fastest *effective* on the task. The Layer-B wall is the **digest shape**
  (large-2D / rewrite editable regions → huge prompts) and **glyph perception** — not LLM strength, so a
  bigger generator does not move the needle.
- **Qwopus-27B+MTP** is the better pick for genuinely harder coding/reasoning where 27B quality matters —
  ideally on a fast GPU where MTP shines. MTP itself is a clean, free speedup (native, 80% accept).
- **DiffusionGemma** is worth pursuing as an **energy-prior / verifier signal** (its actual project role)
  and as a fast parallel generator *if* paired with thinking-suppression at the template level (Unsloth
  Studio auto-sets this) — not via the raw `llama-diffusion-cli` path on reasoning-heavy prompts.

Artifacts: `results/arc3_layerb_qwopus_test{,_mtp}.json`, `results/arc3_layerb_diffusiongemma_test.json`;
harnesses `scripts/experiments/arc3_layerb_{qwopus,diffusiongemma}_test.py`. iGPU diffusion runner built
at `~/.cache/llama.cpp-master/build-hip/bin/llama-diffusion-cli` (HIP, not the 3090s).

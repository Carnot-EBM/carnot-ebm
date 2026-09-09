# Qwen3.8-Flash-Next: evaluated 2026-09-09, NOT viable on this hardware

**Question (operator):** is `unsloth/Qwen3.8-Flash-Next-GGUF` or
`nvidia/Qwen3.8-Flash-Next-NVFP4` an option for us?

**Answer: no, for both, by a wide margin.** Recorded so it is not re-derived. Both repos are real
and current; the blocker is purely capacity.

## Sizes, measured from the HuggingFace API

| artifact | size | fits 48 GB VRAM? |
|---|---|---|
| current mandate `unsloth/Qwen3.8-27B-GGUF` | **16 GB** | yes, one card with KV headroom |
| `Flash-Next-GGUF` UD-IQ1_S (smallest quant) | **72.5 GB** | no |
| `Flash-Next-GGUF` UD-Q2_K_XL | 78.9 GB | no |
| `Flash-Next-GGUF` UD-Q4_K_XL | 111.3 GB | no |
| `Flash-Next-GGUF` BF16 | 354.0 GB | no |
| `Flash-Next-NVFP4` (11 safetensors shards) | **132.7 GB** | no |

BF16 at 354 GB puts Flash-Next around 177B parameters. **Even 1-bit quantisation is 1.5x our
entire rig.** The current mandate is 4.5x smaller than the smallest Flash-Next quant.

## Host facts

- 2x RTX 3090, **compute capability 8.6 (Ampere)**, 24 GB each, 48 GB total.
- System RAM 125 GB, 116 GB available.
- Installed: `transformers 5.12.0`. NOT installed: `tensorrt_llm`, `vllm`, `modelopt`.

## The NVFP4 variant fails on three independent counts

1. **Hardware.** NVFP4 needs Blackwell FP4 tensor cores. These GPUs are Ampere, SM 8.6.
2. **Size.** 132.7 GB against 48 GB.
3. **Runtime.** No NVFP4-capable inference stack is installed.

Any one blocks it. Note `unsloth/Qwen3.8-27B-NVFP4` is already cached on this host — **cached is
not runnable**, and its presence should not be read as evidence NVFP4 works here.

## The Kaggle escape route also fails

The Kaggle ARC eval GPU is an RTX 6000 Blackwell, 96 GB (per
`feedback_kaggle_gpu_far_faster_than_local` — a RECORDED figure, not re-measured today). Blackwell
*would* support NVFP4 natively, so the hardware objection lifts there. But 132.7 GB still exceeds
96 GB, and `feedback_kaggle_arc_resource_discipline` requires locally-verified forward progress
before a submission — impossible for a model that cannot load locally.

## Could CPU offload rescue the GGUF?

Technically the 72.5 GB IQ1_S fits in 116 GB of system RAM. In practice no:

- it is CPU inference on a ~177B model, and conductor tasks die at 1200 s of silence or a 4800 s
  hard cap;
- IQ1_S is severely degraded quantisation, unsuitable for headline-eligible results;
- it is a vision-language model (`image-text-to-text`) needing an extra ~0.9 GB mmproj file.

## Not evaluated, and why

Whether Flash-Next is *better* than Qwen3.8-27B for our tasks. That question is moot while it
cannot be loaded, and answering it would need a comparison we cannot run. If the rig ever gains
80+ GB of VRAM, revisit — the model is a vision-language model, which is interesting for ARC's
64x64 grids, and NVIDIA's AVO result notably acted on text-only grids rather than images.

**Also note the MTP files** (1.9-7.8 GB) in the GGUF repo are speculative-decoding draft models,
not standalone substitutes.

---

## Second worked example, same day: `unsloth/Kimi-K3-GGUF`

The operator applied the rule above and asked whether Kimi-K3 is "definitely not an option".
Confirmed by measurement, and it fails harder than Flash-Next:

| quant | size |
|---|---|
| UD-Q1_0 (smallest) | **466.4 GB** |
| UD-IQ1_S | 594.0 GB |
| UD-Q4_K_XL | 1,508.7 GB |
| UD-Q8_K_XL | 1,561.2 GB |

Against every executing surface:

    vs local VRAM             48 GB  ->  9.7x over
    vs local RAM             125 GB  ->  3.7x over   (so CPU offload is out too)
    vs Kaggle Blackwell       96 GB  ->  4.9x over

Flash-Next's smallest quant was 72.5 GB: 1.5x over VRAM but it DID fit in system RAM, so CPU
offload was at least arguable before being rejected on speed and quality. Kimi-K3 fits nowhere
that can execute it.

### The trap this example exposes: disk space is not a feasibility signal

The hub cache filesystem has **908 GB available**. The 466 GB quant would download cleanly and
land without error. It would then never run.

So there are now two shapes of the same mistake recorded here:

- **cached is not runnable** — `unsloth/Qwen3.8-27B-NVFP4` sits in the cache on hardware that
  cannot execute NVFP4;
- **downloadable is not runnable** — a 466 GB checkpoint fits comfortably on a 908 GB disk and
  exceeds every memory it would need to occupy.

A successful download proves storage, nothing more. **The feasibility question is smallest-quant
size against VRAM (and RAM, if offload is genuinely on the table), plus the numeric format
against compute capability.** Disk never enters it.

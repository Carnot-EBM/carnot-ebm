---
type: Claim
title: vLLM runs native FP4 on Blackwell at 651.8 tok/s aggregate (k=32), 12.5x the shipped config
description: Native SM120 cutlass FP4 works offline on Kaggle once the toolchain is coherent; fp8 KV is free; batching scales 11.6x.
tags: [arc-agi-3, vllm, nvfp4, blackwell, throughput, concurrency]
status: stable
resource: https://www.kaggle.com/code/iancblenke/carnot-vllm-blackwell-bench
sources:
  - id: bench-v13
    resource: /home/ianblenke/.claude/jobs/ad0c053d/tmp/kvllm13/vllm_bench.json
    author: outer-loop/claude-opus-5
  - id: bench-v15
    resource: /home/ianblenke/.claude/jobs/ad0c053d/tmp/kvllm15/vllm_bench.json
    author: outer-loop/claude-opus-5
generated: { by: outer-loop/claude-opus-5, at: 2026-08-18T19:35:00Z }
verified:
  - { by: outer-loop/claude-opus-5, at: 2026-08-18T19:30:00Z }
---

# The measurement

Kaggle RTX PRO 6000 Blackwell, vLLM 0.27.1, `unsloth/Qwen3.8-27B-NVFP4` safetensors, offline
(196-wheel `--no-index` install, 108 s). Same request shape as the llama.cpp arms: 22,000-token
prompt, 8,192 generated with `ignore_eos`, every finish `length`.

| k | aggregate tok/s | scaling |
|---|---|---|
| 1 | 56.1 | 1.00x |
| 4 | 181.0 | 3.23x |
| 16 | 483.5 | 8.62x |
| 32 | **651.8** | 11.62x |

References on the same card and shape: llama.cpp best (MTP off) 228.3 at k=16; the currently
shipped configuration ~52. So vLLM at k=32 is **2.85x the best llama.cpp config and ~12.5x what
ships today**. The induction budget (about 6.13M tokens) drops from does-not-fit to **~2.6 h
against the 11.5 h cap**.

# Every layer verified engaged, none assumed

- `Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM` -- native FP4, `marlin: False`.
- `Using fp8 data type to store kv cache` -- fp8 KV engaged, and FREE: k=1/4/16 match the bf16
  arms to within noise.
- Single-stream is a wash vs llama.cpp (56 vs 52) -- the entire win is batching + native FP4 at
  concurrency, exactly as scoped.

# The curve bends

k=16 -> 32 gains 1.35x, not 2x: bandwidth saturates. Practical plateau on this card is
~650-800 tok/s; ~1K tps would need k around 64 with diminishing returns. At capped production
session length (~54k tokens) fp8 KV supports ~22 concurrent sessions in the ~61 GB KV budget, so
a shipped config would run `--max-num-seqs` around 24.

# What it took: 15 kernel versions, two real root causes

Everything between was peeling layers, each failure specific and cheap after the first:
1. **pip resolved an incoherent CUDA stack** -- nvcc 13.3.73 against 13.0 headers. flashinfer's
   JIT of the SM120 FP4 GEMM died on "CUDA compiler and CUDA toolkit headers are incompatible".
   Fixed by pinning the coherent 13.0.x five (nvcc/crt/nvvm/cccl/runtime) in the wheels dataset.
2. **Missing linker dev-symlinks** -- `ld: cannot find -lcudart / -lcuda`. Wheels ship only
   `libcudart.so.13`, the driver only `libcuda.so.1`; `ld` wants unversioned names. Fixed with a
   symlink dir on `LIBRARY_PATH`, the driver located via `ldconfig -p` rather than guessed.

Notable dead ends, kept so they are not re-walked: the SM120-missing-from-FP4-dispatch GitHub
issues are STALE (the kernel exists and is selected); a pre-built AOT cache failed because
ninja's dep files embed build-container paths so everything reads dirty (and Kaggle's dataset
mount depth is not stable across runs -- resolve by rglob, never by fixed parents[N]); the
Marlin fallback arm died on a genuine vLLM bug (`ParallelLMHead` missing
`output_size_per_partition` on the FP8 lm_head).

# What this does NOT decide

Shipping vLLM on the scored path: ~1000-1200 lines in `LocalGGUFProposer` plus about 60 test
files that reference llama.cpp specifics (many are the pinning guards that caught real drift).
That is an operator decision, now with a measured 12.5x attached rather than a hunch. Interim
option that captures part of the win with zero migration: llama.cpp `--parallel 16` with MTP off,
measured 228.3.

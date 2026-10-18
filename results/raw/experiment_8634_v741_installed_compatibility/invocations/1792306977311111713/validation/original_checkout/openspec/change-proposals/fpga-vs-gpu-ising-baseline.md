# FPGA vs GPU Ising Sampler Benchmark — paper-blocking

> **2026-04-30 RE-SCOPE NOTICE:** This proposal was drafted before the
> Phase-3 architecture blind-spot audit (which identified 5 FATAL
> findings — see `docs/research-notes/phase3-architecture-blindspot-audit-results.md`)
> and before the user's direction to pivot future hardware bets to
> Extropic Z1 + photonic rather than continue the FPGA-deep-EBM
> rabbit hole.
>
> **The proposal scope is reduced.** Instead of a full FPGA-vs-GPU
> headline comparison, the .85+ task is now a **Phase-2 Sampler
> Correctness Audit + GPU Baseline** (see `ops/known-issues.md`
> revised entry). The audit Finding #2 means exp1081's
> synchronous-parallel-Glauber on arbitrary J doesn't preserve
> detailed balance — any FPGA-vs-GPU comparison must validate sampler
> correctness first OR explicitly caveat the comparison.
>
> The original spec below is preserved for reference but **superseded**
> by the new known-issues.md entry. Future hardware claims rest on
> Extropic Z1 + photonic, not on FPGA bitstream redesign.

---

## ORIGINAL PROPOSAL (SUPERSEDED, KEPT FOR REFERENCE)


## Why

Milestone .84's exp1081 "FPGA Scale Benchmark" shipped with a
`fpga_speedup_confirmed_measured` verdict claiming **12,788× speedup
at N=64 spins**. The headline is real (KV260 FPGA: 24.83 µs; CPU: 317.5
ms), but the comparison axis is wrong:

1. **Industry baseline for hardware-acceleration claims is GPU**, not
   CPU. Beating CPU is trivial for any specialized accelerator.
2. **The CPU column was overhead-bound, not compute-bound.** Latencies
   are nearly flat from N=64 to N=1024 (250-320 ms), confirming
   Python/JAX dispatch dominated — the benchmark didn't measure
   compute scaling.
3. **The position paper's Phase 2 hardware section currently rests on
   this comparison.** Submitting that to arxiv invites review-time
   takedown ("you're benchmarking against your own bottleneck, not
   the actual alternative hardware").

This proposal adds a proper FPGA vs GPU vs (compute-bound) CPU
benchmark to make the Phase 2 hardware claim defensible.

## What changes

A single experiment task `fpga-vs-gpu-ising-baseline` for milestone
.85, scope:

- **Hardware under test:**
  - FPGA: KV260 (board IP 192.168.51.98, anchored from exp1068)
  - GPU: RTX 3090 (host machine, onnxruntime-gpu 1.25.1 CUDA EP
    installed and verified working in commit `b53fe63a`)
  - CPU: re-measured with batched/jit'd inference (NOT raw JAX
    dispatch as in exp1081's CPU column)
- **Workload:** Ising sampling at N ∈ {64, 128, 256, 512, 1024}
  spins. Match the columns in exp1081 so artifacts are directly
  comparable.
- **Metrics:**
  - Per-sample latency (µs)
  - Throughput (samples/sec) at batch sizes 1, 32, 256
  - Power-per-sample if instrumentable (KV260 ≈ 5W TDP,
    RTX 3090 ≈ 350W TDP — even if GPU wins raw latency, FPGA may win
    perf/watt by 70× at single-sample)

## Required artifact schema

```json
{
  "experiment": <id>,
  "schema": "carnot.fpga_vs_gpu_baseline.v1",
  "fpga_latencies_us": {"64": ..., "128": ..., ...},
  "gpu_latencies_us": {"64": ..., "128": ..., ...},
  "cpu_latencies_us_compute_bound": {"64": ..., ...},
  "fpga_throughput_samples_per_sec": {"batch_1": ..., "batch_32": ..., "batch_256": ...},
  "gpu_throughput_samples_per_sec": {"batch_1": ..., "batch_32": ..., "batch_256": ...},
  "crossover_n_spins_fpga_beats_gpu": <int or null>,
  "perf_per_watt_advantage": {"fpga_vs_gpu_at_n_64": <float>, ...},
  "honest_verdict": <one of the tokens below>,
  ...
}
```

## Honest-verdict tokens

| Token | Meaning |
|---|---|
| `fpga_beats_gpu_below_N_X` | FPGA wins raw latency below crossover N=X |
| `gpu_dominates_all_N` | GPU wins everywhere on raw latency |
| `fpga_better_perf_per_watt_only` | GPU wins raw, FPGA wins watts/sample |
| `fpga_better_at_batch_size_1` | FPGA wins single-sample (no batching overhead), GPU wins batched |
| `fpga_synth_failed` | Vivado synthesis at N>64 failed; only anchor data |

## Acceptance criteria

The .85 milestone retro should mark this task `MET` only if:

1. At least one **measured** FPGA latency at N>64 (not extrapolation)
2. GPU column populated for all N
3. CPU column re-measured with compute-bound methodology
4. Honest verdict matches one of the tokens above (not the original
   `fpga_speedup_confirmed_measured`)
5. The position paper draft (next iteration after exp1078 v2)
   updates the Phase 2 section to reflect the actual comparison

## Decentralization implications

Per CLAUDE.md decentralization rules: the FPGA path is sovereignty
infrastructure. Even if GPU wins raw latency at large N, the FPGA
result has to stand on its own as "edge-portable energy verification
without a $700 NVIDIA GPU." The paper's Phase 2 framing should be
explicit about this — "specialized hardware can match GPU at
edge scales while running on a $200 board" is a defensible claim;
"FPGA beats GPU at all scales" probably isn't.

## Variance plan (mandatory per CLAUDE.md)

If the first attempt produces `fpga_synth_failed`:
- Attempt 2: skip Vivado synthesis, measure GPU + CPU only and
  compare against exp1081's existing FPGA anchor at N=64. Paper claim
  becomes "anchored at N=64, extrapolated for N>64 pending synthesis."
- Attempt 3: add the proposal to the permanent exclusion list
  (`retire_if_same_verdict: true`) and re-frame Phase 2 around
  edge-portability rather than scale-benchmarked speedup.

## Cost estimate

- Sonnet/Opus turns: ~50 (script writing, GPU EP wiring, KV260
  re-deploy if needed)
- Wall time: 30-60 min (Vivado synthesis on N>64 if attempted is the
  expensive step)
- GPU usage: minimal (Ising sampling at N=1024 is small for an RTX 3090)

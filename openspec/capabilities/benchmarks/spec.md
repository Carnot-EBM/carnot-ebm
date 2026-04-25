# Benchmarks Capability Specification

**Capability:** benchmarks
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-09

## Overview

Defines the live precision benchmarks that measure the end-to-end accuracy of Carnot's
verify-repair pipeline against real math reasoning datasets (GSM8K).  All benchmarks
must run on live GPU hardware; simulated fallbacks are explicitly labelled as such and
excluded from headline claims.

## Requirements

### REQ-BENCH-001: Live GPU Required for Headline Results

All benchmark runs that produce headline accuracy claims MUST set
`inference_mode="live_gpu"`.  Any run that falls back to simulated or CPU-only
inference MUST write `honest_verdict="simulated_no_verdict"` and MUST NOT be used
as a headline result.

**Rationale:** Exp 820 showed baseline=0 correct on HumanEval, which is suspicious.
That result was on a simulated/fallback path that did not actually run the model.
Headline results that claim a specific accuracy number must be traceable to a
confirmed live GPU run.

**Acceptance criteria:**
- `inference_mode` field present in every benchmark artifact.
- `honest_verdict` is never `"pipeline_improvement"` when `inference_mode != "live_gpu"`.

### REQ-BENCH-010: Live Precision Benchmark Must Run on Live GPU

A live precision benchmark MUST have `inference_mode="live_gpu"` to count as a
valid measurement.  When live GPU is not available (no CUDA/ROCm device, health
check fails, or `CARNOT_FORCE_LIVE` not set), the benchmark script MUST:

1. Write a blocked artifact with `honest_verdict="simulated_no_verdict"`.
2. Set `status="blocked"` in the artifact.
3. NOT produce accuracy numbers, because they would be fabricated.

**Rationale:** Silent simulation is the root cause of RETRO-033 and multiple false
improvements across milestones .30-.45.  The invariant check in
`assert_deliverable_written()` enforces this by rewriting the verdict when a
`pipeline_improvement` claim is accompanied by `inference_mode != "live_gpu"`.

**Acceptance criteria:**
- CPU-only machine produces `honest_verdict="simulated_no_verdict"`.
- Live GPU machine with all models healthy produces `inference_mode="live_gpu"`.
- `honest_verdict="pipeline_improvement"` is only written when both conditions hold:
  `inference_mode="live_gpu"` AND `signed_improvement > 0`.

### REQ-BENCH-011: Baseline Condition Must Run First

The BASELINE condition (raw LLM, no pipeline) MUST complete before any
pipeline-augmented condition (VR, FULL) begins.  The baseline accuracy
(`accuracy_baseline`) MUST be used as the denominator for all signed improvement
computations:

```
signed_improvement_vr   = accuracy_vr   - accuracy_baseline
signed_improvement_full = accuracy_full - accuracy_baseline
signed_improvement      = max(signed_improvement_vr, signed_improvement_full)
```

**Rationale:** If baseline does not run first, a pipeline that refuses to answer
any question (n_correct=0) would appear to improve on a zero baseline.
Running baseline first and computing signed improvement exposes this failure mode.

**Acceptance criteria:**
- Artifact contains `accuracy_baseline`, `accuracy_vr`, `accuracy_full`.
- `signed_improvement` equals `max(signed_improvement_vr, signed_improvement_full)`.
- Checkpoint files written after each baseline batch before VR begins.

## Scenarios

### SCENARIO-BENCH-020: Live GPU Baseline Match

**Given** a live GPU machine with Qwen3.5-0.8B loaded
**When** 50 GSM8K questions are run with no pipeline (BASELINE condition)
**Then** `n_correct_baseline >= 0` (sanity: result is non-negative and <= 50)
**And** `accuracy_baseline = n_correct_baseline / 50` is present in the artifact

**Spec traces:** REQ-BENCH-010, REQ-BENCH-011

### SCENARIO-BENCH-025: 50-Question GSM8K Three-Condition Benchmark

**Given** a live GPU machine with Qwen3.5-0.8B
**When** 50 GSM8K questions are run across three conditions:
  - BASELINE: raw LLM output only
  - VR: VerifyRepairPipeline applied to raw LLM output
  - FULL: VR + JEPA v24 if deployed (else VR only, noted in artifact)
**Then** the artifact contains all required accuracy and improvement fields
**And** each condition runs in 5 batches of 10 (checkpointed atomically)
**And** `honest_verdict="pipeline_improvement"` only when `inference_mode="live_gpu"` AND `signed_improvement > 0`
**And** `honest_verdict="pipeline_no_improvement"` when `inference_mode="live_gpu"` AND `signed_improvement <= 0`
**And** `honest_verdict="simulated_no_verdict"` when GPU is unavailable

**Spec traces:** REQ-BENCH-010, REQ-BENCH-011

### SCENARIO-BENCH-030: CPU Fallback Produces Honest Verdict

**Given** a machine with no GPU (or CARNOT_FORCE_LIVE unset)
**When** the benchmark script runs
**Then** the artifact has `status="blocked"`, `honest_verdict="simulated_no_verdict"`
**And** no accuracy numbers are included
**And** `assert_deliverable_written()` passes

**Spec traces:** REQ-BENCH-010

## Implementation Status

| Requirement | Status | Experiment |
|-------------|--------|------------|
| REQ-BENCH-001 | Implemented | Exp 427 |
| REQ-BENCH-010 | Implemented | Exp 840 |
| REQ-BENCH-011 | Implemented | Exp 840 |
| SCENARIO-BENCH-020 | Implemented | Exp 427 |
| SCENARIO-BENCH-025 | Implemented | Exp 840 |
| SCENARIO-BENCH-030 | Implemented | Exp 840 |

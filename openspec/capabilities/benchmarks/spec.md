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

### REQ-BENCH-1511: Product-Line Solver Oracle Benchmark

Carnot MUST provide a bounded product-line / feature-model benchmark that
parses semi-formal blueprints into mandatory, optional, requires, and excludes
constraints, evaluates small instances with a deterministic exhaustive oracle,
and records separate parse, feasibility, oracle-agreement, and false-accept
metrics for mandated local SOTA GGUF model outputs.

**Rationale:** CCTU-style executable constraints exercise local tool use, but
product-line analysis stresses structural parsing and feasibility reasoning in
a different domain while preserving an auditable oracle.

**Acceptance criteria:**
- The benchmark defines multiple semi-formal feature-model blueprints with
  bounded feature counts and expected analysis operations.
- The oracle enumerates the bounded configuration space deterministically and
  rejects infeasible configurations before checking optimality or answer
  agreement.
- Headline rows come only from mandated local SOTA GGUF models; missing or
  failed SOTA loading produces an honest terminal blocker rather than a legacy
  small-model headline.
- The artifact includes `parse_rate`, `feasibility_rate`,
  `oracle_agreement_rate`, `verifier_false_accept_rate`,
  `solver_oracle_ready`, `product_line_benchmark_ready`, and the JSONL
  manifest path.

### SCENARIO-BENCH-1511: Feature-Model Answers Are Checked Against Oracle

**Given** a bounded feature-model blueprint with mandatory, optional,
requires, and excludes clauses
**When** a mandated local SOTA GGUF model emits a JSON answer containing a
feature selection and self-verifier decision
**Then** Carnot parses the answer, checks selection feasibility with the
deterministic oracle, distinguishes infeasible from feasible-but-suboptimal
answers, and records any model self-verifier false accept in the manifest.

**Spec traces:** REQ-BENCH-1511

### REQ-BENCH-1523: Product-Line Staged Feedback Rescue

Carnot MUST provide a deterministic staged-feedback rescue path for the
product-line solver-oracle benchmark that:

- loads the Exp1511 JSONL rows and reproduces baseline parse, feasibility,
  oracle-agreement, and false-accept metrics before rescue;
- uses only mandated local SOTA GGUF provenance for headline rows and blocks
  rather than substituting legacy tiny models when the mandated cache path is
  unavailable;
- applies auditable syntax-parse, feature-model consistency, solver
  feasibility, and policy/compliance feedback stages to each row;
- records per-stage outcomes in a JSONL rescue manifest; and
- sets `product_line_rescue_ready=true` only when parse rate improves over
  Exp1511 and either feasibility or oracle agreement improves while
  `false_accept_rate=0.0`; otherwise it retires the branch with an explicit
  reason.

### SCENARIO-BENCH-1523: Staged Feedback Rescues Or Retires Product-Line Rows

**Given** Exp1511 product-line rows whose parse rate is 0.333333 and whose
feasibility and oracle-agreement rates are 0.0
**When** the staged rescue runner replays those rows through deterministic
parse, feature-model, feasibility, and policy feedback
**Then** the final artifact contains the required baseline and rescue metric
fields, the rescue manifest records every stage per row, and the branch is
marked ready only when the acceptance gate is satisfied with zero false
accepts.

**Spec traces:** REQ-BENCH-1523

### REQ-BENCH-1536: SATQuest CNF Verifier Benchmark

Carnot MUST provide a bounded SATQuest-style CNF verifier benchmark that
generates a deterministic mix of satisfiable and unsatisfiable SAT instances,
labels them with PySAT when available or a local exhaustive solver fallback
when PySAT is unavailable, emits machine, symbolic, and narrative prompt
formats for each instance, and scores mandated local SOTA GGUF answers against
the solver oracle rather than model self-verification.

The benchmark MUST record baseline answer accuracy, Carnot energy-ranked
candidate accuracy, repair-hint-assisted accuracy, and solver-oracle false
accepts separately from ordinary wrong answers.  Missing SOTA GGUF runtime
provenance MUST be recorded honestly without substituting legacy small models
as headline models.

**Decentralization implications:** The benchmark preserves local-first
operation by using local GGUF model provenance and a local deterministic SAT
oracle.

### SCENARIO-BENCH-1536: CNF Formats Are Reproducibly Labeled

**Given** the fixed Exp1536 run date and bounded CNF generator seed
**When** the SATQuest benchmark builds machine, symbolic, and narrative prompts
for each CNF instance
**Then** every format shares the same deterministic solver label, the manifest
records the solver backend used, and false accepts are counted only when a
model-declared verifier accepts a non-oracle answer.

**Spec traces:** REQ-BENCH-1536

## Implementation Status

| Requirement | Status | Experiment |
|-------------|--------|------------|
| REQ-BENCH-001 | Implemented | Exp 427 |
| REQ-BENCH-010 | Implemented | Exp 840 |
| REQ-BENCH-011 | Implemented | Exp 840 |
| REQ-BENCH-1511 | Implemented (`python/carnot/eval/product_line_solver_oracle_benchmark.py`) | Exp 1511 |
| REQ-BENCH-1523 | Planned (`python/carnot/eval/product_line_parser_feasibility_rescue.py`) | Exp 1523 |
| REQ-BENCH-1536 | Planned (`python/carnot/eval/satquest_cnf_verifier_benchmark.py`) | Exp 1536 |
| SCENARIO-BENCH-020 | Implemented | Exp 427 |
| SCENARIO-BENCH-025 | Implemented | Exp 840 |
| SCENARIO-BENCH-030 | Implemented | Exp 840 |
| SCENARIO-BENCH-1511 | Implemented (`tests/python/test_experiment_1511_product_line_solver_oracle_benchmark.py`) | Exp 1511 |
| SCENARIO-BENCH-1523 | Planned (`tests/python/test_experiment_1523_product_line_parser_feasibility_rescue.py`) | Exp 1523 |
| SCENARIO-BENCH-1536 | Planned (`tests/python/test_experiment_1536_satquest_cnf_verifier_benchmark.py`) | Exp 1536 |

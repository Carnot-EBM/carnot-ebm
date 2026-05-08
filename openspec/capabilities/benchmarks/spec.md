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

### REQ-BENCH-1540: Product-Line Staged Benchmark Scale

Carnot MUST provide a scaled product-line staged benchmark pack that extends
the Exp1523 bounded rescue beyond the six checked rows while keeping
deterministic validators as the only acceptance authority.  The benchmark MUST:

- write `results/experiment_1540_product_line_staged_benchmark_scale_v3.json`
  with `status="in_progress"` before source loading or model probing;
- load the Exp1511/Exp1523 product-line rows and record the syntax,
  feature-model, solver-feasibility, and policy/oracle fixes responsible for
  the bounded rescue;
- build or load at least 40 staged product-line cases unless the available
  corpus is smaller, in which case it MUST record the available count and
  blocker;
- evaluate syntax parsing, feature-model consistency, feasibility, and
  solver-oracle agreement as separate stages with the exhaustive product-line
  oracle as authority;
- run at least one mandated local SOTA GGUF model on benchmark prompts, using
  automata-style constrained JSON where useful while never allowing that
  automaton to replace the deterministic oracle;
- set `product_line_scale_ready=true` only when the scaled pack reaches the
  case-count gate, all staged rows agree with the oracle, and
  `false_accept_rate=0.0`; and
- set `branch_retired=true` with an explicit reason when the branch cannot
  scale beyond the bounded rescue, the live SOTA provenance gate fails, or
  `false_accept_rate` exceeds zero.

### SCENARIO-BENCH-1540: Scaled Product-Line Rows Preserve Oracle Labels

**Given** the Exp1523 rescue manifest and the fixed run date `20260508`
**When** the scaled staged benchmark pack generates product-line variants and
replays them through syntax, feature-model, feasibility, and policy feedback
**Then** every manifest row contains reproducible oracle labels, stage
outcomes, and a final policy decision that accepts only oracle-agreeing
selections
**And** the terminal artifact contains the required scale/retirement fields and
an honest verdict prefix accepted by the conductor.

**Spec traces:** REQ-BENCH-1540

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

### REQ-BENCH-1549: SATQuest Oracle Evidence Repair

Carnot MUST provide a SATQuest oracle repair gate that replays the Exp1536
false-accept rows and a bounded regression pack with proof or witness evidence
before downstream SATQuest or unified-contract work can treat SATQuest as an
acceptance authority.  The repaired gate MUST:

- prefer PySAT when available while preserving a deterministic exact fallback
  for bounded local CNFs;
- attach a checked full assignment witness to every SAT oracle success;
- attach a checked UNSAT certificate or exhaustive contradiction record to
  every UNSAT oracle success;
- reject malformed CNF rows, wrong labels, invalid SAT assignments, SAT answers
  without full assignment witnesses, and UNSAT answers when no proof or
  exhaustive contradiction evidence is available;
- replay the three known Exp1536 false-accept rows and report zero repaired
  oracle false accepts before setting `satquest_zero_false_accepts=true`; and
- write an artifact and repaired-case manifest that expose the prior
  false-accept count, rechecked case IDs, evidence counts, perturbation checks,
  and honest terminal verdict.

**Decentralization implications:** The repair preserves local-first operation:
PySAT is optional, exact exhaustive evidence is available locally for the
bounded benchmark, and no closed-weight model becomes an oracle.

### SCENARIO-BENCH-1549: False Accept Rows Are Replayed With Evidence

**Given** `results/experiment_1536_satquest_cnf_verifier_benchmark.json` and
`results/satquest_cnf_verifier_1536.jsonl`
**When** the Exp1549 repair gate extracts the rows where a model self-verifier
accepted a non-oracle answer
**Then** the repaired oracle rechecks each row with SAT assignment witnesses or
UNSAT contradiction evidence, rejects every known false accept, runs malformed
CNF and perturbation checks, and writes
`results/experiment_1549_satquest_oracle_false_accept_repair.json` with
`solver_oracle_false_accepts_after=0`.

**Spec traces:** REQ-BENCH-1549

### REQ-BENCH-1550: SATQuest SOTA Re-Evaluation Under Repaired Oracle

Carnot MUST provide a SATQuest SOTA re-evaluation workflow that runs only after
the Exp1549 repair artifact confirms `satquest_zero_false_accepts=true`.  The
workflow MUST use mandated local SOTA GGUF provenance for headline model rows
when available, honestly block without substituting legacy small GGUFs when
the mandated cache or llama.cpp runtime is unavailable, and keep the repaired
solver/proof oracle as the only acceptance authority.

The workflow MUST:

- write `results/experiment_1550_satquest_sota_reeval_zero_false_accepts.json`
  with `status="in_progress"` before model probing or evaluation;
- load the repaired SATQuest manifest from Exp1549 and refuse evaluation when
  the repaired zero-false-accept gate is not present;
- evaluate at least 30 bounded machine, symbolic, and narrative/natural-language
  CNF prompt cases when runtime permits;
- validate SAT assignment witnesses or UNSAT contradiction certificates before
  accepting any answer;
- report answer accuracy, witness-validity rate, false-accept rate, solver
  oracle false accepts, and Carnot energy/ranking AUC as diagnostics only; and
- include the required artifact fields for model specs, attempted models, case
  counts, format coverage, automata or format constraints, availability
  blockers, focused-test status, and an honest terminal verdict.

### SCENARIO-BENCH-1550: SOTA Rows Remain Solver-Grounded

**Given** Exp1549 has written a repaired manifest and
`satquest_zero_false_accepts=true`
**When** the Exp1550 runner resolves mandated local SOTA GGUF specs with the
`cached_sota_pair()` pattern and aggregates generated SATQuest answers
**Then** every accepted row is backed by a checked assignment witness or UNSAT
certificate, model self-verification never overrides the solver/proof oracle,
and the terminal artifact records zero solver-oracle false accepts or an honest
availability blocker when no mandated SOTA model can run.

**Spec traces:** REQ-BENCH-1550

## Implementation Status

| Requirement | Status | Experiment |
|-------------|--------|------------|
| REQ-BENCH-001 | Implemented | Exp 427 |
| REQ-BENCH-010 | Implemented | Exp 840 |
| REQ-BENCH-011 | Implemented | Exp 840 |
| REQ-BENCH-1511 | Implemented (`python/carnot/eval/product_line_solver_oracle_benchmark.py`) | Exp 1511 |
| REQ-BENCH-1523 | Planned (`python/carnot/eval/product_line_parser_feasibility_rescue.py`) | Exp 1523 |
| REQ-BENCH-1536 | Planned (`python/carnot/eval/satquest_cnf_verifier_benchmark.py`) | Exp 1536 |
| REQ-BENCH-1549 | Implemented (`python/carnot/eval/satquest_oracle_false_accept_repair.py`) | Exp 1549 |
| REQ-BENCH-1540 | Implemented (`python/carnot/eval/product_line_staged_benchmark_scale.py`) | Exp 1540 |
| REQ-BENCH-1550 | Planned (`python/carnot/eval/satquest_sota_reeval_zero_false_accepts.py`) | Exp 1550 |
| SCENARIO-BENCH-020 | Implemented | Exp 427 |
| SCENARIO-BENCH-025 | Implemented | Exp 840 |
| SCENARIO-BENCH-030 | Implemented | Exp 840 |
| SCENARIO-BENCH-1511 | Implemented (`tests/python/test_experiment_1511_product_line_solver_oracle_benchmark.py`) | Exp 1511 |
| SCENARIO-BENCH-1523 | Planned (`tests/python/test_experiment_1523_product_line_parser_feasibility_rescue.py`) | Exp 1523 |
| SCENARIO-BENCH-1536 | Planned (`tests/python/test_experiment_1536_satquest_cnf_verifier_benchmark.py`) | Exp 1536 |
| SCENARIO-BENCH-1540 | Implemented (`tests/python/test_experiment_1540_product_line_staged_benchmark_scale.py`) | Exp 1540 |
| SCENARIO-BENCH-1550 | Planned (`tests/python/test_experiment_1550_satquest_sota_reeval_zero_false_accepts.py`) | Exp 1550 |

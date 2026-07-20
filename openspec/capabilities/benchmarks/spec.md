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

### REQ-BENCH-2931: LLMEval-Logic-Style Local GGUF Z3 Mini Benchmark

Carnot MUST provide a bounded LLMEval-Logic-style mini benchmark that asks
mandated local GGUF models to translate natural-language logic items into a
structured formalization JSON, converts parseable formalizations into Z3, and
reports parseability, Z3 execution, answer correctness, and semantic
faithfulness as separate metrics.

The benchmark MUST call `cached_sota_pair(gpu_indices=(0, 1))` before any
fallback resolution and MUST only treat outputs from one of the mandated SOTA
GGUF model IDs as headline live inference.  If no mandated GGUF is cached, the
artifact MUST be honest and blocked rather than substituting legacy tiny
models.

**Acceptance criteria:**
- The item pack contains 12-20 logic items with gold answers and Z3-checkable
  gold formalizations, either loaded from a local LLMEval-Logic cache or
  clearly marked as a forward-authored LLMEval-Logic-style fixture.
- Each model prompt requests exactly one structured formalization JSON object.
- The evaluator separately records parseability, Z3 execution, answer
  accuracy, and rubric-like semantic faithfulness.
- The artifact includes a reproducibility checksum over items, prompts,
  model specs, raw outputs, parsed formulas, and Z3 results.
- `logic_verifier_mini_ready=true` only when mandated live GGUF inference
  produces at least one usable row and Z3 executes on every parsed row.

### SCENARIO-BENCH-2931: Formalized Logic Answers Are Checked By Z3

**Given** a bounded natural-language logic item with facts, Horn-style rules,
mutual exclusions, a query, and a gold modal answer
**When** a mandated local GGUF emits structured formalization JSON
**Then** Carnot parses the JSON, checks whether the query is possible and
necessary with Z3, derives the solver answer, compares it with the model answer
and the gold answer, and records semantic faithfulness separately from syntax
and Z3 execution.

**Spec traces:** REQ-BENCH-2931

### REQ-BENCH-2959: NL-To-Z3 Execution Repair Mini Benchmark

Carnot MUST provide a bounded natural-language-to-Z3 execution repair mini
benchmark that replays or collects 8-12 LLMEval-Logic-style local SOTA GGUF
formalization proposals, repairs only schema-fragment formatting into the
strict ground-logic JSON contract, executes Z3 on every parseable
formalization, and never treats model prose or model self-judgment as a proof.

The benchmark MUST include at least one of the mandated local GGUF model IDs in
`model_specs`, call `cached_sota_pair(gpu_indices=(0, 1))` during precondition
checking, and honestly record whether a headline model was resolved through the
pair helper or through a single mandated cached GGUF fallback.  The terminal
artifact MUST be written to
`results/experiment_2959_nl_to_z3_execution_repair_mini_v2.json` and include:
`honest_verdict`, `inference_substrate="live_llm_inference"`,
`preconditions_checked`, `model_specs`, `headline_models_used`, `z3_import_ok`,
`z3_execution_repaired`, `n_items`, `parseability_rate`,
`z3_execution_rate`, `solver_verified_accuracy`, `answer_accuracy`,
`failure_categories`, `formalization_manifest_sha256`, and `duration_s`.

**Acceptance criteria:**
- The selected item pack contains 8-12 small logic items with known
  Z3-checkable labels.
- The parser repairs only bounded schema fragments such as
  `formalization: {...}` plus an answer line into the strict JSON contract.
- Every parseable formalization is executed by Z3; Z3 exceptions are reported
  separately from parse failures.
- The artifact reports mutually exclusive counts for `unparseable`,
  `z3_exception`, `wrong_formula`, `wrong_answer`, and
  `solver_verified_correct`.
- `solver_verified_accuracy` is based on the Z3-derived label, while
  `answer_accuracy` is based on the model answer string; neither metric may be
  substituted for the other.

### SCENARIO-BENCH-2959: Fragmented Formalizations Are Accepted Or Rejected By Z3

**Given** a live local SOTA GGUF response that contains a repairable
formalization fragment and a model answer line
**When** Carnot converts the fragment to the strict ground-logic schema
**Then** Z3 derives the modal answer for the query, the model answer is scored
separately, and the row is categorized as solver-verified correct, wrong
formula, wrong answer, Z3 exception, or unparseable without using model prose as
proof.

**Spec traces:** REQ-BENCH-2959, SCENARIO-BENCH-2959

### REQ-BENCH-2966: Skill-Labeled Exact Logic Frontier Manifest

Carnot MUST provide a deterministic exact-verifier logic frontier that
materializes 20-30 compact logic items with known labels, reference Z3
formalizations, and explicit skill labels for downstream live LLM tasks.  This
frontier MUST NOT invoke a live model; it exists to separate formal reasoning
skills before another natural-language-to-Z3 generation attempt.

The materializer MUST check that Python can import Z3 before execution.  If Z3
is unavailable, it MUST write an honest blocked artifact with
`preconditions_checked` populated.  When Z3 is available, it MUST execute every
reference formalization, record per-item pass/fail evidence, write a downstream
manifest with expected solver status and labels, and write
`results/experiment_2966_logic_frontier_materializer_v1.json` with:
`honest_verdict`, `preconditions_checked`, `z3_import_ok`,
`logic_frontier_materialized`, `n_items`, `skill_labels`,
`reference_formalizations_executed`, `reference_z3_execution_rate`,
`reference_solver_accuracy`, `manifest_path`, `manifest_sha256`,
`model_specs_for_downstream_live_use`,
`inference_substrate="deterministic_wiring"`, and `duration_s`.

**Acceptance criteria:**
- The frontier contains 20-30 deterministic items spanning symbolization,
  quantifier handling, countermodel construction, satisfiability, validity,
  and answer extraction.
- Every item has a reference Z3 formalization, an expected solver status, a
  known label, and one or more skill labels.
- Z3 executes every reference formalization when the dependency is present, and
  `reference_solver_accuracy` is computed from solver outcomes rather than
  natural-language labels alone.
- The manifest includes the mandated downstream model specs:
  `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`.
- The manifest SHA-256 is computed over the stable manifest JSON that was
  written to disk.

### SCENARIO-BENCH-2966: Reference Logic Formalizations Execute Exactly

**Given** the deterministic skill-labeled logic frontier
**When** the materializer runs with Z3 importable
**Then** every reference formalization is parsed and checked by Z3, the
per-item result records the actual solver status and whether it matched the
expected status and answer-extraction fields, and the terminal artifact reports
`logic_frontier_materialized=true` with deterministic-wiring provenance.

**Spec traces:** REQ-BENCH-2966, SCENARIO-BENCH-2966

### REQ-BENCH-2967: SOTA DCCD NL-To-Z3 Frontier Formalization

Carnot MUST provide a live local-SOTA NL-to-Z3 formalization runner over the
Exp 2966 exact-verifier frontier.  The runner MUST load the materialized Exp
2966 manifest, check that `logic_frontier_materialized=true`, verify Python Z3
import, call `cached_sota_pair(gpu_indices=(0, 1))` before any single-model
fallback, and use at least one mandated headline GGUF model in `MODEL_SPECS`:
`unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, or
`unsloth/gemma-4-26B-A4B-it-GGUF`.

The live runner MUST use DCCD-style draft-then-structure prompting: first ask
the model for an unconstrained formalization draft, then condition a strict
structured output request on that draft.  The parseable structured proposal
schema MUST contain `variables`, `predicates`, `assertions`, `query`,
`expected_status`, and `answer_extraction`.  Carnot MUST execute Z3 on every
parseable proposal and MUST NOT treat model prose, model self-judgment, or the
model-provided expected status as proof.

The terminal artifact MUST be written to
`results/experiment_2967_sota_nl_to_z3_dccd_formalization_v1.json` and include:
`honest_verdict`, `inference_substrate="live_llm_inference"`,
`preconditions_checked`, `model_specs`, `headline_models_used`,
`legacy_models_only_for_smoke`, `n_items`, `parseability_rate`,
`z3_execution_rate`, `solver_verified_accuracy`, `answer_accuracy`,
`baseline_parseability_rate`, `baseline_solver_verified_accuracy`,
`skill_wise_metrics`, `failure_categories`, `formalization_delta_clean`,
`formalization_manifest_sha256`, and `duration_s`.

**Acceptance criteria:**
- If Exp 2966 is not materialized, Z3 is unavailable, or no mandated headline
  GGUF can be resolved, the runner writes a blocked artifact with populated
  `preconditions_checked` and zero rates.
- Every parseable proposal is compiled and checked by Z3; Z3 exceptions are
  categorized separately from unparseable outputs.
- Each item is categorized by failure mode and by every Exp 2966 skill label it
  carries.
- `formalization_delta_clean` is true only when
  `parseability_rate>=0.50`, `z3_execution_rate>=0.50`, and
  `solver_verified_accuracy>0.0`, compared against the .278 baseline
  `parseability_rate=0.083333` and `solver_verified_accuracy=0.0`.
- `formalization_manifest_sha256` is computed over the stable item manifest,
  structured proposals, Z3 results, skill labels, and failure categories.

### SCENARIO-BENCH-2967: DCCD Proposals Are Judged By Z3 Skill-Wise

**Given** the Exp 2966 exact-verifier frontier and at least one resolved
mandated local GGUF
**When** the live runner collects DCCD-style structured formalization proposals
**Then** every parseable proposal is executed by Z3, aggregate and skill-wise
parseability/Z3/solver/answer metrics are recorded, and the artifact marks
`formalization_delta_clean` only when the explicit .278 baseline gate is met.

**Spec traces:** REQ-BENCH-2967, SCENARIO-BENCH-2967

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

### REQ-BENCH-1744: Phase 4 EqM Latency-Accuracy Synthesis

Carnot MUST provide an impact analysis script for Exp 1744 that reads the large benchmark
results (Exp 1743) and synthesizes the tradeoff between EqM-guided decoding latency
and final accuracy gains.
The script MUST calculate and write a JSON summary to `results/experiment_1744_impact.json`
containing the fields `status`, `eqm_latency_overhead_ms`, `accuracy_gain_pct`,
and `honest_verdict`. Missing input files MUST NOT crash the process but produce
a blocked artifact or mock data for testing.

### SCENARIO-BENCH-1744: Impact Analysis Generates Scatter Data

**Given** the Exp 1744 analysis script
**When** executed against the Exp 1743 large benchmark logs
**Then** it parses token-latency and repair-success rates
**And** produces a valid `results/experiment_1744_impact.json` artifact containing the synthesized tradeoff metrics.

**Spec traces:** REQ-BENCH-1744

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
| REQ-BENCH-2959 | Implemented (`python/carnot/eval/nl_to_z3_execution_repair_mini_v2.py`) | Exp 2959 |
| SCENARIO-BENCH-2959 | Implemented (`tests/python/test_experiment_2959_nl_to_z3_execution_repair_mini_v2.py`) | Exp 2959 |
| REQ-BENCH-2966 | Implemented (`python/carnot/eval/logic_frontier_materializer.py`) | Exp 2966 |
| SCENARIO-BENCH-2966 | Implemented (`tests/python/test_experiment_2966_logic_frontier_materializer.py`) | Exp 2966 |
| REQ-BENCH-2967 | Implemented (`python/carnot/eval/sota_nl_to_z3_dccd_formalization.py`) | Exp 2967 |
| SCENARIO-BENCH-2967 | Implemented (`tests/python/test_experiment_2967_sota_nl_to_z3_dccd_formalization.py`) | Exp 2967 |

### REQ-DUALGPU-101: DualGPU System-2 EqM Benchmark

Carnot MUST provide a comprehensive DualGPU benchmark combining System-2 EqM, FourierCSP, and continuous learning.
The implementation MUST ensure SOTA models (`unsloth/Qwen3.6-35B-A3B-GGUF` and `unsloth/gemma-4-31B-it-GGUF`) are split or utilized effectively.
Results MUST be written to `results/experiment_1733_dualgpu.json`.

**Acceptance criteria:**
- Script `scripts/experiment_1733_dualgpu.py` runs the combined benchmark.
- SOTA models are referenced in the script.
- Artifact is written to `results/experiment_1733_dualgpu.json`.

### SCENARIO-DUALGPU-101: Comprehensive DualGPU execution

**Given** a DualGPU environment
**When** the Exp 1733 script runs
**Then** the System-2 EqM and FourierCSP pipelines execute using the specified SOTA models and write the required artifact.

**Spec traces:** REQ-DUALGPU-101

### REQ-BENCH-1742: SWE-Bench Lite Verify-Repair Harness

Carnot MUST provide an Exp 1742 SWE-Bench Lite harness that selects five
targeted `princeton-nlp/SWE-bench_Lite` test instances, prompts the mandated
SOTA GGUF model pair for unified-diff patches with EqM decoding disabled as the
baseline condition, verifies each patch through bounded Carnot patch checks,
and evaluates accepted patches through a SWE-Bench-compatible evaluator.

The experiment MUST write `results/experiment_1742_swe_bench.json` with the
selected instance IDs, model provenance, `eqm_decoding_enabled=false`,
baseline and verify-repair resolve metrics, per-case attempts, evaluator
provenance, blockers, and an honest verdict. Missing SOTA weights or missing
SWE-Bench evaluator dependencies MUST produce a blocked artifact instead of a
simulated headline result.

### SCENARIO-BENCH-1742: SWE-Bench Lite Baseline Artifact Is Honest

**Given** the Exp 1742 runner and the targeted SWE-Bench Lite cohort
**When** SOTA GGUF weights or the SWE-Bench evaluator are unavailable
**Then** the runner writes the required artifact with `status="blocked"`,
`eqm_decoding_enabled=false`, the selected instance metadata when fetchable,
and no fabricated resolve-rate headline.
**And** when injected model and evaluator backends are available, the runner
records baseline and verify-repair outcomes for each selected instance.

**Spec traces:** REQ-BENCH-1742

### REQ-BENCH-1773: Continuous Latent Constraint Benchmark

Carnot MUST provide a benchmark script to evaluate the continuous latent constraint modeling.
The implementation MUST use `unsloth/gemma-4-31B-it-GGUF` in MODEL_SPECS.
Results MUST be written to `/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/results/experiment_1773_latent_benchmark.json`.

**Acceptance criteria:**
- Script `scripts/experiment_1773_latent_benchmark.py` exists.
- Records energy convergence and validity rates.
- Uses `LatentOptimizer` from `python/carnot/models/latent_optimizer.py`.

### SCENARIO-BENCH-1773: Continuous Latent Constraint Execution

**Given** the LatentOptimizer and an energy function
**When** the benchmark script runs
**Then** it performs Langevin dynamics optimization
**And** produces a valid `results/experiment_1773_latent_benchmark.json` artifact containing energy convergence and validity rates.

**Spec traces:** REQ-BENCH-1773

### REQ-E2E-1782: End-to-end evaluation using Qwen 3.6 35B

Carnot MUST provide an E2E benchmark script to evaluate the flagship MoE model, integrating all Phase 1-3 improvements.
The implementation MUST use `unsloth/Qwen3.6-35B-A3B-GGUF` in MODEL_SPECS.
Results MUST be written to `/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/results/experiment_1782_e2e_qwen.json`.

**Acceptance criteria:**
- Script `scripts/experiment_1782_e2e_qwen.py` exists.
- Records latency, parse rate, and energy scores.
- Artifact is written to `results/experiment_1782_e2e_qwen.json`.

### SCENARIO-E2E-1782: E2E Qwen MoE Evaluation Execution

**Given** the complete Phase 1-3 integrated pipeline
**When** the benchmark script runs using `unsloth/Qwen3.6-35B-A3B-GGUF`
**Then** it performs the execution evaluation
**And** produces a valid `results/experiment_1782_e2e_qwen.json` artifact containing latency, parse rate, and energy scores.

**Spec traces:** REQ-E2E-1782

### REQ-E2E-1783: End-to-end evaluation using Gemma 3.1 31B

Carnot MUST provide an E2E benchmark script to evaluate the flagship dense model.
The implementation MUST use `unsloth/gemma-4-31B-it-GGUF` in MODEL_SPECS.
Results MUST be written to `/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/results/experiment_1783_e2e_gemma31.json`.

**Acceptance criteria:**
- Script `scripts/experiment_1783_e2e_gemma31.py` exists.
- Records latency, parse rate, and energy scores.
- Artifact is written to `results/experiment_1783_e2e_gemma31.json`.

### SCENARIO-E2E-1783: E2E Gemma 3.1 Dense Evaluation Execution

**Given** the complete Phase 1-3 integrated pipeline
**When** the benchmark script runs using `unsloth/gemma-4-31B-it-GGUF`
**Then** it performs the execution evaluation
**And** produces a valid `results/experiment_1783_e2e_gemma31.json` artifact containing latency, parse rate, and energy scores.

**Spec traces:** REQ-E2E-1783



### REQ-BENCH-1789: NRGPT Math Benchmark

Carnot MUST provide a benchmark script to validate the Aleph-style orchestrator and NRGPT exploration on a mathematical benchmark.
The implementation MUST use `unsloth/Qwen3.6-35B-A3B-GGUF` in MODEL_SPECS.
Results MUST be written to `results/experiment_1789_math_benchmark.json`.

**Acceptance criteria:**
- Script `scripts/experiment_1789_math_benchmark.py` exists.
- Records machine-checkable proof success rate.
- Artifact is written to `results/experiment_1789_math_benchmark.json`.

### SCENARIO-BENCH-1789: NRGPT Math Benchmark Execution

**Given** the NRGPTExplorer
**When** the benchmark script runs using `unsloth/Qwen3.6-35B-A3B-GGUF`
**Then** it performs the execution evaluation
**And** produces a valid `results/experiment_1789_math_benchmark.json` artifact containing machine-checkable proof success rate.

**Spec traces:** REQ-BENCH-1789

### REQ-E2E-1808: Capstone End-to-end evaluation using Qwen3.6-35B-A3B

Carnot MUST provide an E2E benchmark script to evaluate the flagship MoE model, incorporating the DPO adapter and KAN MILP verifier.
The implementation MUST use `unsloth/Qwen3.6-35B-A3B-GGUF` in MODEL_SPECS.
Results MUST be written to `results/experiment_1808_qwen.json`.

**Acceptance criteria:**
- Script `scripts/experiment_1808_capstone_qwen.py` exists.
- Records repair success rate.
- Artifact is written to `results/experiment_1808_qwen.json`.

### SCENARIO-E2E-1808: Capstone E2E Qwen MoE Evaluation Execution

**Given** the complete Phase 16 integrated pipeline with DPO adapter and KAN MILP verifier
**When** the benchmark script runs using `unsloth/Qwen3.6-35B-A3B-GGUF`
**Then** it performs the execution evaluation
**And** produces a valid `results/experiment_1808_qwen.json` artifact containing repair success rate.

**Spec traces:** REQ-E2E-1808

### REQ-E2E-1810: Capstone End-to-end evaluation using Gemma4-26B

Carnot MUST provide an E2E benchmark script to evaluate the flagship MoE model on the Phase 16 capstone pipeline.
The implementation MUST use `unsloth/gemma-4-26B-A4B-it-GGUF` in MODEL_SPECS.
Results MUST be written to `results/experiment_1810_gemma26.json`.
It MUST record accuracy and energy metrics.

**Acceptance criteria:**
- Script `scripts/experiment_1810_capstone_gemma26.py` exists.
- Records accuracy and energy metrics.
- Artifact is written to `results/experiment_1810_gemma26.json`.

### SCENARIO-E2E-1810: Capstone E2E Gemma4-26B Evaluation Execution

**Given** the complete Phase 16 integrated pipeline
**When** the benchmark script runs using `unsloth/gemma-4-26B-A4B-it-GGUF`
**Then** it performs the execution evaluation
**And** produces a valid `results/experiment_1810_gemma26.json` artifact containing accuracy and energy metrics.

**Spec traces:** REQ-E2E-1810

### REQ-E2E-1835: SOTA LLM Pipeline with EBRM and Zero-Violation loops Qwen3.6-35B-A3B

Carnot MUST provide an E2E benchmark script to evaluate the SOTA LLM pipeline integrated with EBRM and Zero-Violation loops.
The implementation MUST use `unsloth/Qwen3.6-35B-A3B-GGUF` in MODEL_SPECS.
Results MUST be written to `results/experiment_1835_qwen.json`.
It MUST record repair success rate.

**Acceptance criteria:**
- Script `scripts/experiment_1835_qwen.py` exists.
- Records repair success rate.
- Artifact is written to `results/experiment_1835_qwen.json`.

### SCENARIO-E2E-1835: SOTA LLM Pipeline Qwen MoE Evaluation Execution

**Given** the complete SOTA LLM pipeline integrated with EBRM and Zero-Violation loops
**When** the benchmark script runs using `unsloth/Qwen3.6-35B-A3B-GGUF`
**Then** it performs the verify-repair evaluation
**And** produces a valid `results/experiment_1835_qwen.json` artifact containing repair success rate.

**Spec traces:** REQ-E2E-1835

### REQ-E2E-1836: SOTA LLM Pipeline with EBRM and Zero-Violation loops Gemma4-31B-it

Carnot MUST provide an E2E benchmark script to evaluate the SOTA LLM pipeline integrated with EBRM and Zero-Violation loops.
The implementation MUST use `unsloth/gemma-4-31B-it-GGUF` in MODEL_SPECS.
Results MUST be written to `results/experiment_1836_gemma31.json`.
It MUST record repair success rate.

**Acceptance criteria:**
- Script `scripts/experiment_1836_gemma31.py` exists.
- Records repair success rate.
- Artifact is written to `results/experiment_1836_gemma31.json`.

### SCENARIO-E2E-1836: SOTA LLM Pipeline Gemma4 MoE Evaluation Execution

**Given** the complete SOTA LLM pipeline integrated with EBRM and Zero-Violation loops
**When** the benchmark script runs using `unsloth/gemma-4-31B-it-GGUF`
**Then** it performs the verify-repair evaluation
**And** produces a valid `results/experiment_1836_gemma31.json` artifact containing repair success rate.

**Spec traces:** REQ-E2E-1836

### REQ-E2E-1837: SOTA LLM Pipeline with EBRM and Zero-Violation loops Gemma4-26B-A4B-it

Carnot MUST provide an E2E benchmark script to evaluate the SOTA LLM pipeline integrated with EBRM and Zero-Violation loops.
The implementation MUST use `unsloth/gemma-4-26B-A4B-it-GGUF` in MODEL_SPECS.
Results MUST be written to `results/experiment_1837_gemma26.json`.
It MUST record repair success rate.

**Acceptance criteria:**
- Script `scripts/experiment_1837_gemma26.py` exists.
- Records repair success rate.
- Artifact is written to `results/experiment_1837_gemma26.json`.

### SCENARIO-E2E-1837: SOTA LLM Pipeline Gemma4-26B MoE Evaluation Execution

**Given** the complete SOTA LLM pipeline integrated with EBRM and Zero-Violation loops
**When** the benchmark script runs using `unsloth/gemma-4-26B-A4B-it-GGUF`
**Then** it performs the verify-repair evaluation
**And** produces a valid `results/experiment_1837_gemma26.json` artifact containing repair success rate.

**Spec traces:** REQ-E2E-1837



### REQ-E2E-1874: Triple Integration E2E on MoE and Dense SOTA models

Carnot MUST provide an E2E benchmark script to enforce ROCE, HILED, and continuous learning updates across all mandated SOTA models.
The implementation MUST use `unsloth/gemma-4-31B-it-GGUF` and `unsloth/gemma-4-26B-A4B-it-GGUF` in MODEL_SPECS.
Results MUST be written to `/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/results/experiment_1874_e2e.json`.
It MUST ensure cross-language equivalences, serialization, and sampling pipelines complete without error.

**Acceptance criteria:**
- Script `scripts/experiment_1874_e2e.py` exists.
- Records cross-language equivalences, serialization, and sampling pipelines status.
- Artifact is written to `results/experiment_1874_e2e.json`.

### SCENARIO-E2E-1874: Triple Integration E2E Evaluation Execution

**Given** the complete E2E test plan enforcing ROCE, HILED, and continuous learning updates
**When** the benchmark script runs using `unsloth/gemma-4-31B-it-GGUF` and `unsloth/gemma-4-26B-A4B-it-GGUF`
**Then** it performs the execution evaluation for cross-language equivalences, serialization, and sampling pipelines
**And** produces a valid `results/experiment_1874_e2e.json` artifact containing completion metrics.

**Spec traces:** REQ-E2E-1874

### REQ-E2E-1954: Tri-SOTA E2E v6

Carnot MUST provide an E2E benchmark script to evaluate verifiable reasoning dataset across the continuous latent optimization and multi-agent Ising tiers.
The implementation MUST use `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and `unsloth/gemma-4-26B-A4B-it-GGUF` in MODEL_SPECS.
Results MUST be written to `/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/results/experiment_1954_tri_sota_e2e_v6.json`.
It MUST evaluate complete end-to-end trace validity rates against the prior v5 baselines.

**Acceptance criteria:**
- Script `scripts/experiment_1954_tri_sota_e2e_v6.py` exists.
- Records end-to-end trace validity rates against the prior v5 baselines.
- Artifact is written to `results/experiment_1954_tri_sota_e2e_v6.json`.

### SCENARIO-E2E-1954: Tri-SOTA E2E v6 Evaluation Execution

**Given** the verifiable reasoning dataset and continuous latent optimization and multi-agent Ising tiers
**When** the benchmark script runs using `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and `unsloth/gemma-4-26B-A4B-it-GGUF`
**Then** it evaluates complete end-to-end trace validity rates against the prior v5 baselines
**And** produces a valid `results/experiment_1954_tri_sota_e2e_v6.json` artifact containing completion metrics.

**Spec traces:** REQ-E2E-1954

### REQ-BENCH-1771: Constraint-Aware Retrieval Module (CARM) Benchmark

Carnot MUST provide a benchmark suite of unstructured instruction prompts and their ground truth deterministic constraint representations for the Constraint-Aware Retrieval Module (CARM).
The suite MUST contain exactly 20 cases covering tool-use, arithmetic, and logic constraints.
The generator script MUST write the deliverable to `results/experiment_1771_care_test_suite.json` and `results/carm_benchmark_cases.json` with `schema="carnot.carm.benchmark.v1"` and `num_cases=20`.

**Acceptance criteria:**
- Script generates the benchmark cases.
- JSON output contains `schema`, `num_cases` (20), and an array of cases.
- Cases cover tool-use, arithmetic, and logic.

### SCENARIO-BENCH-1771: CARM Benchmark Suite Generation

**Given** the need for a CARM benchmark
**When** the benchmark generation script runs
**Then** it produces a valid JSON artifact with the correct schema and case count
**And** the generated test cases are successfully validated by the unit test suite.

**Spec traces:** REQ-BENCH-1771


### REQ-HARDWARE-1991: ROCm eGPU Detection and JAX Initialization

Carnot MUST provide a hardware verification script to detect the presence of the RX 7900 XTX eGPU via ROCm and verify its initialization in JAX over Thunderbolt.
The implementation MUST use `rocminfo` and `jax.devices()`.
Results MUST be written to `results/experiment_1991_egpu_rocm.json`.
It MUST record the hardware state and an honest verdict.

**Acceptance criteria:**
- Script `scripts/experiment_1991_egpu_rocm.py` exists.
- Records detected hardware state.
- Artifact is written to `results/experiment_1991_egpu_rocm.json`.

### SCENARIO-HARDWARE-1991: eGPU ROCm Detection Execution

**Given** the hardware verification script
**When** the benchmark script runs
**Then** it performs the hardware detection check using `rocminfo` and JAX
**And** produces a valid `results/experiment_1991_egpu_rocm.json` artifact containing the schema, experiment ID, and an honest verdict.

**Spec traces:** REQ-HARDWARE-1991

### REQ-HARDWARE-2008: ROCm Probe eGPU Detection
Carnot MUST provide a script `scripts/experiment_2008_rocm.py` that queries `rocminfo` and maps memory limits, generating a mock success if no hardware is found.
The artifact MUST be saved to `results/experiment_2008_rocm_probe.json`.

### SCENARIO-HARDWARE-2008: ROCm Probe Execution
**Given** the ROCm probe script is executed
**When** it queries `rocminfo`
**Then** it writes probe details or a mock success indicating "hardware missing but probe works" to `results/experiment_2008_rocm_probe.json`

**Spec traces:** REQ-HARDWARE-2008

### REQ-BENCH-2849: Local Evaluation Dataset Manifests

Carnot MUST materialize explicit local JSONL manifests for MBPP, HumanEval,
TruthfulQA generation, HaluEval, and FEVER before downstream corpus agents spend
model calls on those benchmarks.  The manifest builder MUST prefer already
available local `data/` or HuggingFace cache assets, MUST NOT synthesize
benchmark rows, and MUST mark a dataset readiness boolean false when the local
asset cannot meet the conservative count gate.

The terminal artifact MUST be written to
`results/experiment_2849_local_dataset_materialization_v1.json` and include:
`honest_verdict`, one readiness boolean per dataset, `dataset_status`,
`manifest_paths`, `manifest_counts`, `manifest_sha256`,
`preconditions_checked`, `synthetic_rows_created=false`, `duration_s`, and
`run_date="20260522"`.

**Acceptance criteria:**
- MBPP, HumanEval, and TruthfulQA manifests preserve stable IDs plus prompts,
  labels or executable tests, split names, and source paths.
- HaluEval and FEVER manifests preserve stable IDs plus prompt/input text,
  candidate or claim text, labels, split names, and source paths.
- Readiness is true only when the manifest reaches the configured count gate:
  MBPP >= 100, HumanEval >= 164, TruthfulQA generation >= 200, HaluEval >= 500,
  and FEVER >= 500.
- Every non-empty manifest has a recorded SHA256 digest computed from the JSONL
  bytes on disk.

### SCENARIO-BENCH-2849: Missing Local Corpus Blocks Honestly

**Given** a required dataset is absent from local `data/` and HuggingFace cache
locations
**When** the local manifest builder runs without a successful network download
**Then** the artifact records that dataset readiness as false
**And** `dataset_status` explains the exact missing resource without creating
synthetic rows.

**Spec traces:** REQ-BENCH-2849

### REQ-BENCH-2863: Stable Evaluation Manifest Resolver Contract

Carnot MUST expose a reusable resolver that treats
`results/experiment_2849_local_dataset_materialization_v1.json` as the source
of truth for local evaluation manifest filenames.  The resolver MUST map the
canonical corpus keys `halueval`, `fever`, `mbpp`, `humaneval`, and
`truthfulqa` to the actual dated JSONL manifest paths and SHA256 checksums
recorded by Exp 2849, rather than assuming plain filenames such as
`halueval.jsonl` or `fever.jsonl`.

The resolver contract artifact MUST be written to
`results/experiment_2863_eval_manifest_contract_v2.json` and include:
`honest_verdict`, `manifest_contract_ready`,
`manifest_source_artifact="results/experiment_2849_local_dataset_materialization_v1.json"`,
`resolved_manifest_paths`, `resolved_manifest_sha256`, one readiness boolean
per canonical corpus, `synthetic_rows_created=false`, `tests_run`,
`field_principles`, `duration_s`, and `run_date="20260522"`.

**Acceptance criteria:**
- HaluEval and FEVER consumers can resolve the dated Exp 2849 paths
  deterministically from the source artifact.
- Readiness is true for a corpus only when Exp 2849 marked the corpus ready,
  the resolved manifest file exists, and the file SHA256 matches Exp 2849.
- The resolver MUST NOT create synthetic rows, infer metrics, or silently
  replace dated manifest paths with plain aliases.

### SCENARIO-BENCH-2863: HaluEval and FEVER Resolve Dated Manifests

**Given** Exp 2849 records dated local manifest paths and SHA256 checksums for
HaluEval and FEVER
**When** the stable evaluation manifest resolver is called for those corpora
**Then** it returns the dated paths from Exp 2849
**And** it verifies the recorded checksums
**And** it reports both corpus readiness booleans without fabricating rows.

**Spec traces:** REQ-BENCH-2863, SCENARIO-BENCH-2863

### REQ-BENCH-2888: TruthfulQA InFi-Check-Style Taxonomy Manifest

Carnot MUST provide an Exp 2888 TruthfulQA taxonomy manifest builder that uses
only the local TruthfulQA manifest rows and existing Carnot artifacts. The
builder MUST NOT call a remote LLM, MUST NOT synthesize truth labels, and MUST
NOT cite or depend on the retired fabricated Exp 2823 artifact.

The terminal artifact MUST be written to
`results/experiment_2888_truthfulqa_inficheck_taxonomy_manifest_v1.json` and
include `honest_verdict`, `truthfulqa_taxonomy_ready`, `source_artifacts`,
`excluded_artifacts`, `manifest_paths`, `manifest_checksums`,
`n_rows_available`, `n_rows_materialized`, `taxonomy_fields`,
`error_type_counts`, `generated_answer_metrics_available`,
`headline_metric_claim_made=false`, `remote_llm_called=false`,
`synthetic_labels_created=false`, `tests_run`, `field_principles`,
`run_date="20260522"`, and a real `duration_s`.

**Acceptance criteria:**
- The builder resolves the dated TruthfulQA JSONL manifest from the Exp 2863
  resolver contract and verifies the recorded SHA256 digest before reading
  rows.
- The builder records every local TruthfulQA source path present in the
  materialized rows, including a checksum when the local source file exists.
- The bounded sample materializes at least 100 local rows when that many rows
  are available and records deterministic fields
  `factual_error_type`, `evidence_available`, `justification_available`,
  `correction_available`, `unsupported_reason`, and `metric_eligibility`.
- Generated-answer metrics remain null and
  `generated_answer_metrics_available=false` unless an existing clean
  generated-answer artifact supplies real generated answers and measured
  metrics.
- The retired Exp 2823 artifact is recorded only in `excluded_artifacts` with
  its checksum and exclusion reason; it is not used as source evidence.

### SCENARIO-BENCH-2888: TruthfulQA Local Labels Become Taxonomy Rows

**Given** the Exp 2863 manifest contract resolves a checksum-verified local
TruthfulQA JSONL manifest
**And** Exp 2831 and Exp 2840 do not contain clean generated-answer metrics
**When** the Exp 2888 taxonomy builder runs
**Then** it writes the required artifact with at least 100 materialized local
TruthfulQA rows when available
**And** every materialized row contains deterministic taxonomy fields derived
from existing TruthfulQA labels and source metadata
**And** headline metrics, remote-LLM usage, and synthetic-label creation remain
explicitly false or null.

**Spec traces:** REQ-BENCH-2888, SCENARIO-BENCH-2888

### REQ-BENCH-2864: HaluEval/FEVER Local Manifest Calibration

Carnot MUST provide an Exp 2864 HaluEval/FEVER full calibration runner that
depends on the Exp 2863 manifest-contract artifact rather than plain manifest
filenames. The runner MUST verify `halueval_ready=true` and `fever_ready=true`
from Exp 2863, load the resolved local JSONL manifests, recompute each manifest
SHA256 before scoring, and avoid any live LLM invocation.

The runner MUST score only dataset-provided labeled rows with deterministic
verifier-energy logic, compute AUROC and deterministic bootstrap 95% confidence
intervals when both binary labels are present, and write honest `null` values
when a metric is unavailable. The terminal artifact MUST be written to
`results/experiment_2864_halueval_fever_full_calibration_v3.json` and include:
`honest_verdict`, `halueval_fever_ready`, `full_benchmark_ready`,
`live_model_invoked=false`, `manifest_paths_used`, `manifest_sha256_used`,
`halueval_auroc`, `fever_auroc`, `halueval_n_examples`, `fever_n_examples`,
`auroc_ci95_by_dataset`, `label_counts_by_dataset`, `random_seed`,
`reproducibility_checksum`, `preconditions_checked`,
`adversarial_verify_passed`, `adversarial_verify_flags`, `field_principles`,
`run_date="20260522"`, and measured `duration_s`.

**Acceptance criteria:**
- The runner reads manifest paths and SHA256 values from Exp 2863 fields and
  never substitutes `halueval.jsonl` or `fever.jsonl` aliases.
- `halueval_fever_ready` is true only when both Exp 2863 readiness booleans are
  true and both manifest checksums verify on disk.
- `live_model_invoked` remains false, and the artifact contains no live-model
  provenance fields because this calibration uses dataset candidates only.
- If `scripts/adversarial_verify.py` is present, the runner verifies the
  written artifact and persists pass/fail plus flag details back into it.

### SCENARIO-BENCH-2864: Full Calibration Uses Resolved Local Manifests

**Given** Exp 2863 resolves dated HaluEval and FEVER manifests with valid
checksums
**When** the Exp 2864 calibration runner executes with a fixed random seed
**Then** it writes the required v3 artifact with the dated manifest paths and
checksums echoed from Exp 2863
**And** it reports AUROC and confidence intervals for every dataset that has
both labels
**And** it records `live_model_invoked=false` and adversarial-verification
results without claiming live-model provenance.

**Spec traces:** REQ-BENCH-2864, SCENARIO-BENCH-2864

### REQ-BENCH-2919: ConstraintBench Mini Direct Optimization

Carnot MUST provide a tiny local direct-optimization benchmark inspired by
ConstraintBench that materializes 15-30 deterministic tasks spanning linear
integer constraints, knapsack-like binary constraints, and graph/coloring
constraints.  Each task MUST be small enough for an exact local verifier
implemented by exhaustive enumeration, and the verifier MUST report syntax,
feasibility, objective value, and exact optimality as separate outcomes.

The runner MUST call `cached_sota_pair(gpu_indices=(0, 1))` before selecting
any model.  Headline rows MUST come only from at least one mandated local SOTA
GGUF (`unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, or
`unsloth/gemma-4-26B-A4B-it-GGUF`) through local llama.cpp inference.  When no
mandated GGUF is available, the runner MUST write
`honest_verdict="blocked_sota_gguf_cache_missing"` and MUST NOT report
headline model results.

The terminal artifact MUST be written to
`results/experiment_2919_constraintbench_mini_direct_optimization_v1.json` and
include: `honest_verdict`, `constraintbench_mini_ready`, `model_specs`,
`models_used`, `cached_sota_pair_used`, `random_seed=2919`,
`task_manifest_path`, `n_tasks`, `exact_verifier_types`,
`feasibility_rate`, `optimality_rate`, `syntax_valid_rate`,
`violation_classes`, `per_task_results`,
`inference_substrate="live_llm_inference_plus_exact_verifier"`,
`duration_s`, and `run_date="20260523"`.

**Acceptance criteria:**
- The manifest contains 15-30 tasks with at least one linear, knapsack, and
  graph/coloring task, and every task has a deterministic optimum from the
  local exact verifier.
- Structured JSON answers are parsed without trusting model self-verification.
- Feasibility and optimality are reported as independent rates, and violation
  classes distinguish syntax failures, feasibility violations, and
  feasible-but-suboptimal answers.

### SCENARIO-BENCH-2919: Feasibility And Optimality Are Separated

**Given** a manifest row from the ConstraintBench mini benchmark
**When** a local SOTA GGUF emits a structured JSON answer
**Then** Carnot parses the JSON answer, checks feasibility using the exact
verifier, computes the objective value, compares against the enumerated optimum,
and records feasible-but-suboptimal answers separately from infeasible and
syntax-invalid answers.

**Spec traces:** REQ-BENCH-2919, SCENARIO-BENCH-2919

### REQ-BENCH-2926: ConstraintBench Constrained-Output Rerun

Carnot MUST provide a bounded rerun of the Exp 2919 ConstraintBench mini direct
optimization row that repairs the prior methodology flags without weakening
the exact-verifier contract.  The rerun MUST use `random_seed=2926`, build at
least 30 deterministic tasks, call `cached_sota_pair(gpu_indices=(0, 1))`
before model selection, and use only mandated local SOTA GGUFs for live
headline rows.  Legacy tiny models MAY be used only for CPU smoke tests; if no
mandated GGUF is available, the runner MUST write
`honest_verdict="blocked_sota_gguf_cache_missing"`.

The runner MUST prefer an installed local constrained decoder such as
`llguidance`, and otherwise MUST record `constrained_decoder_available=false`
while using prompt-level JSON schema instructions plus deterministic parser
repair.  Each task result MUST capture raw model response provenance including
model path, GPU index, prompt hash, per-task seed, and raw-response file path.
The exact verifier MUST report syntax validity, feasibility, and optimality as
separate outcomes.  The terminal artifact MUST compute
`syntax_valid_rate` over all outputs, `feasibility_rate_overall` over all
outputs, `feasibility_rate_given_syntax` over syntax-valid outputs, and
`optimality_rate_given_feasible` over feasible outputs; it MUST NOT treat
syntax validity as feasibility.

The terminal artifact MUST be written to
`results/experiment_2926_constraintbench_constrained_output_rerun_v2.json` and
include: `honest_verdict`, `constraintbench_corrigendum_ready`,
`flagged_adversarial`, `random_seed=2926`, `reproducibility_checksum`,
`model_specs`, `models_used`, `cached_sota_pair_used`,
`constrained_decoder_available`, `n_tasks`, `syntax_valid_rate`,
`feasibility_rate_overall`, `feasibility_rate_given_syntax`,
`optimality_rate_given_feasible`, `syntax_feasibility_not_tautological`,
`per_task_results`, `raw_response_dir`, `live_inference_duration_s`,
`inference_substrate="live_llm_inference"`, `duration_s`, and
`run_date="20260523"`.  A live headline row MUST have
`live_inference_duration_s >= 60`; otherwise the artifact MUST honestly record
`honest_verdict="blocked_duration_gate_failed"`.

### SCENARIO-BENCH-2926: Rerun Metrics Are Non-Tautological

**Given** a ConstraintBench mini rerun task and a local SOTA GGUF raw response
**When** Carnot parses the structured answer and runs the exact verifier
**Then** the artifact records syntax-valid, feasible, and optimal rows as
separate sets
**And** feasible-but-suboptimal answers contribute to
`optimality_rate_given_feasible` but not to syntax invalidity
**And** syntax-valid but infeasible answers make
`feasibility_rate_overall` differ from `syntax_valid_rate` in the corrected
row unless the run is honestly blocked.

**Spec traces:** REQ-BENCH-2926, SCENARIO-BENCH-2926

### REQ-BENCH-3211: ConstraintBench Feasibility/Objective Exact Pilot

Carnot MUST provide a small ConstraintBench-style exact pilot that separates
hard-constraint feasibility from objective quality without claiming full
ConstraintBench coverage. The pilot MUST write a JSONL fixture at
`data/research/constraintbench_feasibility_objective_pilot_v1.jsonl` and a
terminal artifact at
`results/experiment_3211_constraintbench_feasibility_objective_pilot_v1.json`.

The fixture MUST contain at least 15 deterministic rows spanning two or more
small optimization families that are natural for Carnot's existing verification
surface, such as knapsack, assignment, scheduling, or graph coloring. Every row
MUST include instance data, human-readable constraints, an exact optimum or
feasible reference, and the checker backend used to validate candidates. Exact
references MUST come from local exhaustive enumeration or an equivalent exact
bounded solver, not from an LLM.

The checker MUST score response quality with separate fields for valid format,
hallucinated entity or variable, missing or violated constraint,
feasibility pass, objective value, and objective gap. If an optional LLM smoke
path is recorded, its model specs MUST include at least one mandated local SOTA
GGUF from `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, or
`unsloth/gemma-4-26B-A4B-it-GGUF`; legacy small models may be labeled
smoke-only but MUST NOT become headline evidence.

The terminal artifact MUST include `schema_version`, `experiment_id`,
`milestone`, `reference_papers`, `fixture_path`, `fixture_count`,
`optimization_families`, `exact_solver_backends`,
`feasibility_metric_defined`, `objective_gap_metric_defined`,
`hallucinated_entity_metric_defined`, `optional_llm_smoke`,
`ready_for_clean_verifier`, `conductor_file_modified`,
`active_roadmap_modified`, and `honest_verdict`.

### SCENARIO-BENCH-3211: Feasibility And Objective Gap Are Scored Separately

**Given** the Exp 3211 exact pilot fixture
**When** Carnot builds the terminal artifact
**Then** every fixture row has an exact bounded reference and checker backend
**And** the checker can distinguish invalid JSON, hallucinated entities,
constraint violations, feasible-but-suboptimal answers, and exact optimum
answers
**And** the artifact reports feasibility, objective gap, hallucinated-entity,
missing-constraint, and invalid-format metrics as separate quantities while
leaving `optional_llm_smoke` null unless a mandated local SOTA GGUF smoke run is
actually invoked.

**Spec traces:** REQ-BENCH-3211, SCENARIO-BENCH-3211

## Implementation Status (REQ-BENCH-3211)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-BENCH-3211 | Implemented (`python/carnot/eval/constraintbench_feasibility_objective_pilot_v1.py`) | Implemented (`tests/python/test_experiment_3211_constraintbench_feasibility_objective_pilot_v1.py`) |

### REQ-BENCH-5616: Exact Nonstationary Constraint-Stream Fixture

Carnot MUST provide a deterministic exact nonstationary constraint-stream
benchmark for Exp5616 that separates two independent axes before any learner is
fit:

- constraint-family or domain-space shifts, represented by explicit
  `space_shift_families`; and
- temporal predicate drift within a fixed family, represented by explicit
  `temporal_drift_types`.

The fixture MUST write a durable JSONL dataset at
`data/research/experiment_5616_exact_nonstationary_constraint_stream.jsonl` and
a terminal artifact at
`results/experiment_5616_exact_nonstationary_constraint_stream.json`. The
dataset schema SHALL be stable and row-level, with every state and update
labeled by an exact validator. Rows SHALL include old-rule, current-rule, and
future-rule labels so retention, adaptation, and leakage can be measured without
using circular model predictions.

The fixture SHALL include shared-rule, conflicting-rule, no-drift,
reversible-drift, and persistent-drift cases at preregistered durations
`1, 2, 4, 8, 16, 32`. Each condition SHALL contain at least 32 independent
stream instances generated from deterministic seeds, with disjoint train,
calibration, and held-out splits. The dataset SHALL publish stable stream
ordering, the seed list, family and duration counts, content hashes, split
receipts, and a replay loader.

The exact validator SHALL include corruption controls for wrong predicate,
wrong variable or entity binding, delayed label, and poison update. The fixture
SHALL prove that the validator rejects every injected violation and accepts
every known-valid control. The fixture SHALL NOT run an LLM, SHALL NOT fit a policy,
and SHALL NOT claim self-learning benefit.

The terminal artifact SHALL include required fields `field_principles`,
`dataset_path`, `schema_version`, `space_shift_families`,
`temporal_drift_types`, `task_durations`, `instances_per_condition`,
`split_receipts`, `exact_oracle_label_count`, `oracle_label_error_count`,
`corruption_controls`, `fixture_ready_score`, `inference_substrate`,
`random_seeds`, `reproducibility_checksum`, and `honest_verdict`.
`fixture_ready_score` SHALL be exactly `1.0` only when schema, counts, splits,
oracle controls, and replay validation pass. `inference_substrate` SHALL be `deterministic_verifier`,
and no other substrate value is valid for this fixture.

Required field principles:

- `field_principles`: principle "Every required evidence field states why it exists."
- `dataset_path`: principle "the fixture is durable"
- `schema_version`: principle "rows have a stable contract"
- `space_shift_families`: principle "domain changes are explicit"
- `temporal_drift_types`: principle "time changes are explicit"
- `task_durations`: principle "the duration axis is preregistered"
- `instances_per_condition`: principle "denominators meet the evidence floor"
- `split_receipts`: principle "leakage is excluded"
- `exact_oracle_label_count`: principle "authority is deterministic"
- `oracle_label_error_count`: principle "corrupted and valid controls are exact"
- `corruption_controls`: principle "unsafe cases are represented"
- `fixture_ready_score`: principle "only complete fixture gates can unlock learners"
- `inference_substrate`: principle "no LLM participated"
- `random_seeds`: principle "generation replays exactly"
- `reproducibility_checksum`: principle "generation replays exactly"
- `honest_verdict`: principle "a deficient fixture blocks learners"

#### SCENARIO-BENCH-5616-SCHEMA: Dataset Rows Carry Exact Rule Labels

**Given** the Exp5616 generator is run with the preregistered seed list,
durations, space-shift families, and temporal-drift types
**When** the JSONL dataset is replay-loaded
**Then** every row conforms to the stable schema, stream ordering is
deterministic, train/calibration/held-out splits are disjoint, and each state
and update has exact old-rule, current-rule, and future-rule labels.

#### SCENARIO-BENCH-5616-CONTROLS: Corruption Controls Fail Closed

**Given** known-valid controls and injected wrong-predicate,
wrong-binding, delayed-label, and poison-update controls
**When** the exact validator evaluates the rows
**Then** every valid control is accepted, every injected violation is rejected,
`oracle_label_error_count=0`, and `fixture_ready_score=1.0` only if the replay
loader reproduces the artifact hashes.

**Spec traces:** REQ-BENCH-5616, SCENARIO-BENCH-5616-SCHEMA,
SCENARIO-BENCH-5616-CONTROLS

## Implementation Status (REQ-BENCH-5616)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-BENCH-5616 | Planned (`python/carnot/experiment_5616_exact_nonstationary_constraint_stream.py`) | Planned (`tests/python/test_experiment_5616_exact_nonstationary_constraint_stream.py`) |

### REQ-BENCH-5757: Exp5746 Proposal Benchmark Scalar Bridge

Carnot MUST provide a read-only scalar bridge for the sealed Exp5746 exact
proposal-utility benchmark. The bridge MUST reuse the canonical Exp5746 JSON
artifact and JSONL manifest by hash/reference only; it MUST NOT regenerate rows,
change train/dev/science split membership, or replace exact receipts with model
or prose judgments.

The bridge MUST verify the Exp5746 manifest hash, every row hash, split hashes,
generator version, primary and independent solver versions, exact structure
receipts, exact solution receipts, validator-disagreement count, held-out split
disjointness, and adversarial controls before emitting any downstream readiness
score. It MUST derive top-level bare scalar fields
`benchmark_ready_score`, `structure_receipt_failure_count`,
`solution_receipt_failure_count`, `validator_disagreement_count`,
`heldout_partition_disjoint_score`, `adversarial_verification_clean_score`, and
`benchmark_bridge_ready_score`.

`benchmark_bridge_ready_score` MUST equal `1.0` only when the upstream
`benchmark_ready_score` is `1.0`, both receipt failure counts are zero,
`validator_disagreement_count` is zero, held-out partition disjointness is
`1.0`, adversarial verification cleanliness is `1.0`, all hashes reproduce,
and the planned Exp5759 conductor gates pass on the emitted artifact. Missing
fields, object-wrapped gate values, altered hashes, ambiguous nested values, and
false readiness MUST fail closed.

Required field principles:

- `benchmark_manifest_path`: principle "sealed manifest is reused by reference"
- `benchmark_manifest_hash`: principle "manifest bytes replay exactly"
- `split_manifest_hash`: principle "train/dev/science split commitment is immutable"
- `row_hash_count`: principle "all 180 row hashes are replayed"
- `benchmark_ready_score`: principle "upstream exact benchmark readiness is copied only after replay"
- `structure_receipt_failure_count`: principle "structure omissions block proposal utility"
- `solution_receipt_failure_count`: principle "missing feasibility/objective receipts block proposal utility"
- `validator_disagreement_count`: principle "exact validators must agree"
- `heldout_partition_disjoint_score`: principle "science split must stay disjoint"
- `adversarial_verification_clean_score`: principle "adversarial controls must all be detected"
- `benchmark_bridge_ready_score`: principle "downstream GPU panels unlock only after hash and gate replay"

#### SCENARIO-BENCH-5757: Sealed Benchmark Replays Without Row Mutation

**Given** the Exp5746 artifact and JSONL manifest are present
**When** the scalar bridge verifies the benchmark
**Then** every manifest row hash and split hash reproduces, the row count is
180, the science split remains unchanged, exact receipt failure counts are
zero, adversarial controls are clean, and the bridge readiness score is `1.0`.

#### SCENARIO-BENCH-5757-NEGATIVE-CONTROLS: Broken Evidence Cannot Unlock The Panel

**Given** a missing gate field, wrapper-object gate value, altered upstream or
manifest hash, ambiguous nested scalar value, or false readiness score
**When** benchmark bridge validation or conductor-gate replay runs
**Then** the condition fails closed and cannot satisfy the Exp5759 benchmark
gate.

**Spec traces:** REQ-BENCH-5757, SCENARIO-BENCH-5757,
SCENARIO-BENCH-5757-NEGATIVE-CONTROLS

## Implementation Status (REQ-BENCH-5757)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-BENCH-5757 | Planned (`scripts/experiment_5757_proposal_benchmark_scalar_bridge.py`) | Planned (`tests/python/test_experiment_5757_proposal_benchmark_scalar_bridge.py`) |

### REQ-BENCH-3389: ConstraintBench AR vs VGB Repair Evaluation

Carnot MUST provide an evaluation script that runs a subset of ConstraintBench tasks (at least 10) through standard autoregressive generation on unsloth/Qwen3.6-35B-A3B-GGUF and compares the validity ratio against candidates repaired by the VGB repair ladder.

#### SCENARIO-BENCH-3389-1: Run Evaluation
**Given** the script `scripts/experiment_3389_constraintbench.py` is executed
**When** the evaluation completes
**Then** it MUST write an artifact to `results/experiment_3389_constraintbench.json` containing the evaluation metrics for both AR and VGB repair loops.

### REQ-BENCH-3585: Realistic Factual Corpus with Headroom

- REQ-BENCH-3585-1: The corpus generation shall retrieve realistic QA pairs (e.g. HaluEval).
- REQ-BENCH-3585-2: The model confidence for each pair shall be generated using a SOTA GGUF model via `cached_sota_pair`.
- REQ-BENCH-3585-3: The sequence logprob / model-confidence AUROC shall be materially < 0.95 to ensure the corpus provides headroom for verifier benchmarking.

#### SCENARIO-BENCH-3585-1: Corpus Generation produces realistic headroom
**Given** the generation script `experiment_3585_realistic_factual_corpus.py`
**When** the corpus is generated and scored
**Then** the confidence baseline AUROC must be `< 0.95`.

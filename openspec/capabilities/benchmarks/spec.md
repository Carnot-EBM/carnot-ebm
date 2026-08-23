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

The planned Exp5759 conductor gate replay MUST evaluate the emitted bridge
artifact through `scripts/conductor_gates.evaluate_gates()` against exactly
these fields: `benchmark_bridge_ready_score`, `benchmark_ready_score`,
`structure_receipt_failure_count`, `solution_receipt_failure_count`,
`validator_disagreement_count`, `heldout_partition_disjoint_score`, and
`adversarial_verification_clean_score`. The bridge MUST preserve all seven as
top-level bare scalars and MUST keep any explanation in `field_principles`.

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
| REQ-BENCH-5757 | Implemented (`scripts/experiment_5757_proposal_benchmark_scalar_bridge.py`, `results/experiment_5757_proposal_benchmark_scalar_bridge.json`) | Implemented (`tests/python/test_experiment_5757_proposal_benchmark_scalar_bridge.py`) |

### REQ-BENCH-5759: SOTA Exact Proposal Utility Panel

Carnot MUST provide Exp5759 at
`python/carnot/experiment_5759_sota_exact_proposal_utility_panel.py` and write
`results/experiment_5759_sota_exact_proposal_utility_panel.json` by consuming
only the sealed Exp5757 scalar bridge and the frozen Exp5746 science split.
The panel MUST define `MODEL_SPECS` containing exactly the three mandated local
headline GGUF ids: `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Legacy tiny models MAY appear only in
explicit smoke-test receipts and MUST NOT populate headline metrics.

Before model access, the runner MUST verify bridge/upstream hashes, immutable
science rows, exact solver versions, candidate label bijections, candidate
ordering seeds, exact budgets, stopping rules, local GGUF paths and hashes,
llama.cpp CUDA build support, GPU visibility/offload, free VRAM/RAM/disk, and
checkpoint/resume space. If any model, runtime, or evidence provenance is not
authenticated, the runner MUST bail before inference and emit a terminal
blocked artifact rather than substitute stale or unauthenticated evidence.

The runner MUST use only the sealed finite-choice candidate-label interface:
native logits over one-token sealed labels MAY rank proposals, while generated
reasoning text, free-form answers, LLM judges, generated-text scoring, PHASE-D
reopening, grammar-runtime authority, and token scores as semantic authority
MUST remain absent. Exact deterministic validators are the only authority for
feasibility and objective values.

For each model, the panel MUST compare the model ordering with paired random
permutations, the deterministic Carnot energy heuristic, solver-native
branching, and exact-search-only controls under matched candidate pools,
validator-call budgets, seeds, and stopping rules. It MUST report top-1
feasible discovery, top-k exact optimum discovery, nodes/candidates to first
valid and first optimum, exact-validator calls, hard-violation rate, exact
objective gap, wall time, model tokens, model-family shortcut residuals, and
paired bootstrap intervals by row, family, model, and overall.

The artifact MUST preregister bare scalar `proposal_utility_delta_overall` as
a signed normalized composite where lower nodes/calls/gaps and higher feasible
or optimal discovery favor the proposal. Bare scalar `proposal_utility_lcb`
MUST be the paired 95 percent lower bootstrap bound. The downstream gate fields
`proposal_utility_lcb`, `flagship_nonregression_count`,
`validator_disagreement_count`, `authority_violation_count`, and
`proposal_utility_ready_score` MUST be top-level bare scalars and their
explanations MUST live only in `field_principles`. The ready score MAY equal
`1.0` only when the lower bound is non-negative, both flagship models have
non-negative per-family deltas, all three mandated models are authenticated and
used, and exact-authority violations are zero.

#### SCENARIO-BENCH-5759: Sealed Science Split Produces A Utility Panel

**Given** the Exp5757 bridge is ready, the Exp5746 manifest replays, and all
three mandated local GGUFs authenticate with llama.cpp CUDA
**When** Exp5759 scores the frozen science rows through the finite-choice label
interface
**Then** the artifact contains all required provenance, metric, bootstrap,
producer-gate, and no-forbidden-runtime fields, compares against all matched
controls, and emits a terminal `complete:` verdict without mutating the sealed
science split.

#### SCENARIO-BENCH-5759-BLOCKED-PRECONDITIONS: Unauthenticated Evidence Blocks Before Inference

**Given** a missing bridge scalar, altered upstream hash, missing GGUF path,
non-CUDA llama.cpp build, tokenizer label collision, insufficient resources, or
unreplayable checkpoint space
**When** Exp5759 preconditions are collected
**Then** inference is not invoked, the artifact status is `blocked`, the
producer ready score is `0.0`, and the blocked verdict names the mechanical
precondition rather than fabricating proposal utility.

**Spec traces:** REQ-BENCH-5759, SCENARIO-BENCH-5759,
SCENARIO-BENCH-5759-BLOCKED-PRECONDITIONS

## Implementation Status (REQ-BENCH-5759)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-BENCH-5759 | Planned (`python/carnot/experiment_5759_sota_exact_proposal_utility_panel.py`, `results/experiment_5759_sota_exact_proposal_utility_panel.json`) | Planned (`tests/python/test_experiment_5759_sota_exact_proposal_utility_panel.py`) |

### REQ-BENCH-5761: Exact Constraint Acquisition Benchmark

Carnot MUST provide Exp5761 at
`python/carnot/experiment_5761_exact_constraint_acquisition_benchmark.py` and
write
`results/experiment_5761_exact_constraint_acquisition_benchmark.json` plus a
sealed JSONL manifest. The benchmark SHALL be derived only from the immutable
Exp5746 artifact and manifest after replaying the upstream artifact hash,
manifest hash, every row hash, split hashes, generator version, solver
versions, family balance, deterministic seeds, free RAM/disk, local
license/provenance, and independent faithful-model structure reconstruction.
If faithful model structure cannot be reconstructed from the sealed Exp5746
receipts, Exp5761 SHALL fail closed with an `honest_verdict` beginning
`blocked:`.

Exp5761 SHALL select at least 120 disjoint source cases across
`finite_domain_csp`, `weighted_maxsat`, `hard_soft_packing`, and
`finite_state_planning`, preserving a family-stratified train/dev/science split
before any learner-facing benchmark rows are exposed. For every faithful typed
model, the generator SHALL emit hash-sealed faithful, incomplete, over-fit, and
mixed variants. Incomplete variants SHALL delete one or more non-implied
constraints, over-fit variants SHALL add spurious globally or locally
restrictive constraints, and mixed variants SHALL combine both operations.
Hard versus soft roles SHALL be preserved, and semantically equivalent no-op
mutations SHALL be rejected.

For every variant, Exp5761 SHALL generate exact positive and negative
assignments and minimal distinguishing membership queries. A primary exact
enumerator and an independent exact validator SHALL check model semantics,
satisfiability, objective values, query minimality, hard/soft roles, expected
repair operations, and adversarial controls. Duplicate, contradictory,
implied-constraint, infeasible-model, vacuous-global, objective-sign, leakage,
split-collision, and shortcut controls SHALL be present and detected. Domain
artifacts, model text and AST, assignments, queries, and expected lifecycle
operations SHALL be sealed by hash.

The terminal artifact MUST include top-level bare gate scalars
`ca_benchmark_ready_score`, `exact_validator_disagreement_count`, and
`train_dev_science_disjoint_score`, and it MUST list those names in
`producer_gate_fields`. The artifact MUST also include `field_principles`,
bare `status`, `preconditions_checked`, `spec_refs`,
`upstream_artifact_hashes`, `generator_version`, `generator_source_hash`,
`solver_versions`, `benchmark_manifest_path`, `benchmark_manifest_hash`,
`instance_count`, `family_counts`, `variant_counts`,
`faithful_model_hashes`, `incomplete_model_hashes`,
`overfit_model_hashes`, `mixed_model_hashes`, `domain_artifact_hashes`,
`positive_assignment_count`, `negative_assignment_count`,
`membership_query_count`, `distinguishing_query_receipts`,
`expected_repair_receipts`, `hard_soft_role_receipts`, `split_manifest`,
`science_row_hashes`, `adversarial_controls`,
`structure_receipt_failure_count`, `solution_receipt_failure_count`,
`llm_inference_used` with value false, `verifier_is_oracle` with value true,
`inference_substrate` with value
`deterministic_exact_solver_dataset_generation_no_llm`,
`random_seeds`, `test_commands`, `test_exit_codes`,
`reproducibility_checksum`, and an `honest_verdict` beginning `complete:` or
`blocked:`.

`ca_benchmark_ready_score` SHALL equal `1.0` only when upstream replay passes,
at least 120 selected cases are disjoint and balanced across the four required
families, train/dev/science split membership is disjoint, all variant mutations
are semantic changes except faithful no-op baselines, every variant has exact
positive and negative assignments, every required discriminating query is
minimal for its expected repair operation, independent validators agree, all
adversarial controls are detected, all required hashes replay, no LLM
inference is used, and the exact verifier is declared as the oracle. Otherwise
the ready score SHALL be `0.0`.

#### SCENARIO-BENCH-5761: Sealed Acquisition Rows Distinguish Missing And Over-Fit Rules

**Given** the Exp5746 exact benchmark artifact and manifest replay by hash
**When** Exp5761 derives the constraint-acquisition benchmark
**Then** at least 120 disjoint cases are selected across the four required
families, faithful/incomplete/over-fit/mixed variants are sealed, train/dev/
science split hashes remain disjoint, exact positive and negative assignments
are present for every variant, distinguishing membership queries expose the
expected add/remove repair operation, and the three producer gate fields are
top-level bare scalars.

#### SCENARIO-BENCH-5761-CONTROLS: Invalid Acquisition Evidence Fails Closed

**Given** duplicate rows, contradictory constraints, implied or vacuous
mutations, infeasible variants, objective-sign inversions, leakage markers,
split collisions, or shortcut acceptances
**When** Exp5761 validates controls
**Then** each control is detected and any undetected control blocks
`ca_benchmark_ready_score`.

**Spec traces:** REQ-BENCH-5761, SCENARIO-BENCH-5761,
SCENARIO-BENCH-5761-CONTROLS

## Implementation Status (REQ-BENCH-5761)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-BENCH-5761 | Planned (`python/carnot/experiment_5761_exact_constraint_acquisition_benchmark.py`, `results/experiment_5761_exact_constraint_acquisition_benchmark.json`) | Planned (`tests/python/test_experiment_5761_exact_constraint_acquisition_benchmark.py`) |

### REQ-BENCH-5785: Sealed Hardness/Surface Exact Fixture

Carnot MUST provide Exp5785 at
`python/carnot/experiment_5785_hardness_surface_fixture.py` and write
`results/experiment_5785_hardness_surface_fixture.json` plus
`results/experiment_5785_hardness_surface_fixture.rows.jsonl` without calling
an LLM. The fixture MUST replay the Exp5784 readiness gates before generation,
verify Z3 and deterministic exact-validator availability, verify free RAM/disk,
record deterministic seeds and versions, and fail closed with a terminal
`blocked:` artifact if either exact authority is unavailable.

The fixture MUST generate a chronological train/calibration/future-test panel
covering at least the three constraint families `finite_domain_scheduling`,
`logic_grid`, and `typed_finite_choice`. Within each family, canonical
independent units MUST cross exact SAT/UNSAT or feasible/infeasible labels,
preregistered solver conflict/time bins, two or more proof-preserving
symbol/order/paraphrase surface variants, and one meaning-changing canary.
Solver effort fields MUST be named as solver-hardness bins and MUST NOT be
called model hardness. The fixture MUST include a sample-size justification
with at least 30 independent canonical units per primary paired comparison, or
else an explicit power rationale; repeated turns or surface variants MUST NOT
be counted as independent units.

Immutable protected facts, mutable constraints, exact assignments or
certificates, protected-fact hashes, candidate domains, finite-choice labels,
surface-transform receipts, meaning-change canaries, chronological split
receipts, and every row hash MUST be sealed before downstream inference.
Protected-fact hashes are the authority for proof-preserving surface variants;
natural-language equivalence alone is insufficient. Candidate completeness
receipts MUST prove that the exact answer/certificate is present and that all
candidate labels are unique and bounded. Split isolation and disjoint row hashes
MUST prove that train, calibration, and future-test rows do not overlap.

The terminal artifact MUST include bare `fixture_ready_score`,
`exact_label_coverage`, and `parser_control_pass_rate`, and MUST list those
names in `producer_gate_fields`. It MUST also include `status`,
`preconditions_checked`, `spec_refs`, `fixture_schema`, `row_file`,
`row_file_sha256`, `family_counts`, `independent_unit_count`,
`sample_size_justification`, `chronological_split_receipts`,
`solver_hardness_bins`, `surface_variant_matrix`,
`proof_preserving_receipts`, `meaning_change_canary_receipts`,
`protected_fact_manifest`, `mutable_constraint_manifest`,
`candidate_completeness_receipts`, `parser_contract`,
`parser_negative_controls`, `exact_validator_receipts`, `leakage_checks`,
`inference_substrate` with value
`deterministic_local_fixture_generation_z3_and_exact_validators_no_llm`,
`test_commands`, `test_exit_codes`, `reproducibility_checksum`, and an
`honest_verdict` beginning `complete:` or `blocked:`.

`fixture_ready_score` SHALL equal `1.0` only when row hashes replay, exact
labels cover every fixture row, candidate completeness passes, chronological
split isolation passes, proof-preserving variants retain exact labels and
protected-fact hashes, meaning-changing canaries change exact labels or
certificates, parser controls pass, protected facts remain separate from
mutable constraints, and all exact validators agree. Otherwise the score SHALL
be `0.0`.

#### SCENARIO-BENCH-5785: Chronological Exact Fixture Seals Hardness And Surface Rows

**Given** Exp5784 has `evidence_index_ready_score=1.0`,
`next_range_collision_count=0`, `unresolved_canonical_count=0`, and
`history_mutation_count=0`
**When** Exp5785 generates the deterministic fixture
**Then** it writes the JSON artifact and JSONL rows, includes at least 30
independent canonical units, preserves train/calibration/future-test chronology,
seals disjoint row hashes, records solver-hardness bins separately from model
difficulty, and emits the three producer gate fields as top-level bare scalars.

#### SCENARIO-BENCH-5785-CONTROLS: Surface, Canary, Candidate, And Leakage Faults Block Readiness

**Given** a row-hash mutation, exact-label mismatch, incomplete candidate
domain, split collision, proof-preserving surface drift, missing
meaning-changing canary, parser-control failure, protected-fact leakage, or
exact-validator disagreement
**When** Exp5785 validates the fixture
**Then** the artifact remains terminal but `fixture_ready_score=0.0` and the
blocked verdict names the mechanical fault instead of treating parser failure as
exact wrongness.

**Spec traces:** REQ-BENCH-5785, SCENARIO-BENCH-5785,
SCENARIO-BENCH-5785-CONTROLS

## Implementation Status (REQ-BENCH-5785)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-BENCH-5785 | Planned (`python/carnot/experiment_5785_hardness_surface_fixture.py`, `results/experiment_5785_hardness_surface_fixture.json`) | Planned (`tests/python/test_experiment_5785_hardness_surface_fixture.py`) |

### REQ-BENCH-5786: Prospective Local-SOTA Constraint Response Stream

Carnot MUST provide Exp5786 at
`python/carnot/experiment_5786_sota_constraint_stream.py` and write
`results/experiment_5786_sota_constraint_stream.json` plus
`results/experiment_5786_sota_constraint_stream.rows.jsonl`. The experiment
MUST consume the sealed Exp5785 fixture chronologically and MUST NOT implement
constraint learning, update model weights, or use legacy tiny models for
headline cells.

Step 0 MUST replay the Exp5785 producer gates and row-file hash, call
`cached_sota_pair()`, verify both RTX 3090 devices, verify readable local GGUF
paths and SHA-256 hashes for exactly
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`, verify llama.cpp CUDA build support, verify
embedded GGUF chat-template availability, verify RAM/VRAM/disk budgets, and
verify output paths. If any mandated model cannot authenticate positive
llama.cpp CUDA offload, the terminal artifact MUST be `blocked:` and MUST NOT
substitute a smoke model into headline rows.

The stream MUST use matched prompt, sampling, stop, max-token, seed, and row
budgets for every mandated model. It MUST preserve canonical, symbol-relabel,
order-paraphrase, and meaning-change-canary rows, keep chronological order,
checkpoint after every model/row cell, and resume exactly without duplicate
rows. Each JSONL stream row MUST preserve lossless raw response text, a raw
response SHA-256 hash, prompt hash, fixture row hash, model identity, runtime
receipt, parser receipt, exact-validator label receipt, taxonomy labels, and a
row hash.

The artifact MUST report behavior by formal constraint family, solver-hardness
bin, surface rendering, satisfiability, and model family. It MUST include
proof-preserving paired deltas and model-identity interactions computed with
canonical fixture units as the independent cluster. At least 30 independent
canonical items per primary paired cell MUST be present or the artifact MUST
document a power calculation; repeated turns, model repeats, and surface
variants MUST NOT be counted as additional independent items.

The terminal artifact MUST include `status`, `preconditions_checked`,
`MODEL_SPECS`, `models_used`, `model_runtime_receipts`,
`gpu_offload_receipts`, `fixture_hash`, `row_file`, `row_file_sha256`,
`raw_response_receipts`, `checkpoint_resume_receipts`,
`sample_size_justification`, `independent_unit_count`,
`failure_taxonomy_counts`, `family_metrics`, `solver_hardness_metrics`,
`surface_sensitivity_metrics`, `proof_preserving_paired_deltas`,
`model_identity_interactions`, `parser_failure_rate`,
`protected_fact_distortion_count`, `satisfiable_drift_count`,
`exact_label_coverage`, `raw_response_coverage`, `leakage_checks`, bare
`real_sota_model_count`, bare `stream_ready_score`, `producer_gate_fields`,
`inference_substrate` with value
`real_local_llama_cpp_cuda_gguf_generation_plus_exact_z3_validation`,
`test_commands`, `test_exit_codes`, `reproducibility_checksum`, and an
`honest_verdict` beginning `complete:` or `blocked:`.

`stream_ready_score` SHALL equal `1.0` only when all three mandated local SOTA
families produce authenticated CUDA GGUF generation rows, raw-response coverage
and exact-label coverage are both `1.0`, parser failure remains below the
preregistered threshold, split/hash leakage checks pass, and the stream contains
enough satisfiable drift to test downstream learning. A complete stream without
that headroom MUST remain `complete:` but set `stream_ready_score=0.0`.

#### SCENARIO-BENCH-5786: Matched Chronological Stream Preserves Raw Responses

**Given** Exp5785 is ready, all three mandated local GGUFs are readable, and
llama.cpp authenticates CUDA offload on the dual RTX 3090 host
**When** Exp5786 runs the matched prospective stream
**Then** every model/fixture-row cell is checkpointed once, raw response text
and hashes are preserved, exact labels are assigned only by the fixture
validator boundary, and the artifact reports family, solver-bin, surface,
satisfiability, paired-delta, and model-interaction metrics.

#### SCENARIO-BENCH-5786-BLOCKERS: Missing SOTA Offload Blocks Headline Rows

**Given** a missing mandated GGUF, unreadable hash, missing chat template,
unauthenticated CUDA offload, split/hash leak, duplicate checkpoint row, parser
failure above threshold, or no satisfiable drift
**When** Exp5786 validates the stream
**Then** it writes a terminal artifact with a `blocked:` or complete-but-not-
ready verdict naming the mechanical fault, excludes tiny smoke models from
headline rows, and keeps raw responses immutable.

**Spec traces:** REQ-BENCH-5786, SCENARIO-BENCH-5786,
SCENARIO-BENCH-5786-BLOCKERS

## Implementation Status (REQ-BENCH-5786)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-BENCH-5786 | Planned (`python/carnot/experiment_5786_sota_constraint_stream.py`, `results/experiment_5786_sota_constraint_stream.json`) | Planned (`tests/python/test_experiment_5786_sota_constraint_stream.py`) |

### REQ-BENCH-5896: Typed ConstraintIR Exact Fixture

Carnot MUST provide Exp5896 at
`python/carnot/experiment_5896_typed_constraint_ir_fixture.py` and write
`results/experiment_5896_typed_constraint_ir_fixture.json` plus
`results/experiment_5896_typed_constraint_ir_fixture.rows.jsonl`. The fixture
SHALL define a minimal engine-neutral typed ConstraintIR schema with canonical
nodes for entities, finite domains, facts, Horn implications, predicates,
arithmetic relations, explicit negation, conjunction, and query goals. The
schema SHALL be versioned, parser-checked, type-checked, and fail closed on
unknown, ambiguous, unsupported, or ill-typed constructs.

The generator SHALL compile every supported valid IR row to the deterministic
Python finite-domain evaluator and the Z3 backend. If an optional Prolog
backend is installed, the artifact SHALL record its applicability; otherwise it
SHALL explicitly record that Prolog was not used. Whenever two exact backends
apply, their query behavior and satisfiability labels SHALL agree before the
row is accepted. Every row SHALL carry an exact certificate or counterexample,
and fixture labels SHALL be replayed by executable behavior over the finite
domain rather than by string equality, model judgment, or backend-specific
prompt tuning.

The row fixture SHALL contain at least three constraint families and include
held templates, natural-language paraphrases, symbol renamings, order
permutations, invalid IR, unsatisfiable IR, type errors, omitted constraints,
and semantic-non-equivalence controls. Paraphrases, renamings, and order
permutations of a single semantic problem SHALL share a group id and SHALL NOT
cross train/dev/held-out boundaries. The benchmark SHALL run no LLM and SHALL
keep the new path default-off.

The terminal artifact MUST include `status`, `preconditions_checked`,
`constraint_ir_schema_and_version`, `supported_and_rejected_fragments`,
`parser_and_typecheck_receipts`, `backend_compiler_receipts`,
`cross_backend_agreement`, `family_template_and_holdout_design`,
`exact_semantic_equivalence_contract`,
`invalid_unsat_and_nonequivalence_controls`,
`split_and_group_leakage_receipts`,
`label_certificate_and_balance_receipts`, `row_file_receipt`,
`deterministic_replay_receipt`, `protected_files_unchanged`,
`typed_constraint_ir_fixture_ready_score`, `duration_s`,
`inference_substrate`, `verifier_is_oracle`, `field_provenance`,
`test_commands`, `test_exit_codes`, `reproducibility_checksum`, and
`honest_verdict`.

Field principles:

- `constraint_ir_schema_and_version`: engine-neutral contract prevents backend-specific prompt tuning from defining success.
- `exact_semantic_equivalence_contract`: executable behavior and certificates own equivalence.
- `split_and_group_leakage_receipts`: paraphrases and renamings of one problem never cross held boundaries.
- `typed_constraint_ir_fixture_ready_score`: emit bare 1.0 only for deterministic exact replay, backend agreement, and nontrivial held headroom.
- `inference_substrate`: use `deterministic_exact_solver_labeled_dataset_no_llm`.
- `verifier_is_oracle`: true for fixture labels/certificates and never a learned-verifier claim.
- `honest_verdict`: use `ready:`, `complete_null:`, or `blocked:`.

#### SCENARIO-BENCH-5896-SCHEMA: Typed IR Rejects Ambiguous Constructs

**Given** an Exp5896 ConstraintIR object with the declared schema version
**When** the parser and typechecker inspect supported nodes, unknown fields,
unsupported node kinds, ambiguous expressions, and wrong-domain arguments
**Then** supported rows compile to typed IR receipts while invalid or ill-typed
rows fail closed with deterministic parser/typecheck receipts.

#### SCENARIO-BENCH-5896-CERTIFICATES: Exact Backends Own Labels

**Given** a supported valid or unsatisfiable Exp5896 IR row
**When** the Python finite-domain backend and Z3 backend replay the row
**Then** their satisfiability, query bindings, certificates, or
counterexamples agree exactly; any disagreement blocks
`typed_constraint_ir_fixture_ready_score`.

#### SCENARIO-BENCH-5896-LEAKAGE: Semantic Variants Stay Grouped

**Given** canonical rows, paraphrases, symbol renamings, order permutations,
held templates, omitted constraints, and semantic-non-equivalence controls
**When** Exp5896 builds train/dev/held-out splits
**Then** every semantic variant shares its group boundary, no group crosses a
split, semantic equivalence is decided by finite-domain behavior plus replayed
certificates, and held-out rows contain nontrivial valid and control examples.

**Spec traces:** REQ-BENCH-5896, SCENARIO-BENCH-5896-SCHEMA,
SCENARIO-BENCH-5896-CERTIFICATES, SCENARIO-BENCH-5896-LEAKAGE

## Implementation Status (REQ-BENCH-5896)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-BENCH-5896 | Planned (`python/carnot/experiment_5896_typed_constraint_ir_fixture.py`, `results/experiment_5896_typed_constraint_ir_fixture.json`) | Planned (`tests/python/test_experiment_5896_typed_constraint_ir_fixture.py`) |

### REQ-BENCH-5897: SOTA ConstraintIR Trace-Repair A/B

Carnot MUST provide Exp5897 at
`python/carnot/experiment_5897_sota_constraint_ir_repair_ab.py` and write
`results/experiment_5897_sota_constraint_ir_repair_ab.json`. The experiment
SHALL replay Exp5896's exact fixture gate before loading any model, SHALL use
only the three mandated GGUF headline families
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`, and SHALL resolve them through
`cached_sota_pair()` plus the cached third family without invoking
`AutoTokenizer.from_pretrained()` on a GGUF repo.

The runner SHALL block before model load when any headline precondition fails:
Exp5896 replay, GGUF files and hashes, llama.cpp CUDA offload support, two
healthy RTX 3090 receipts, adequate VRAM/RAM/disk, no protected training
workload, GPU-offload readiness, and atomic output readiness. A blocked run
SHALL still write the required artifact with `status="blocked_precondition"`
and an honest verdict prefix `blocked_precondition:`.

For each completed model/fixture-row cell, Exp5897 SHALL freeze prompts,
decoding, seeds, row groups, token budgets, and promotion thresholds before
generation, then run the following matched arms: single-pass typed IR
extraction, one exact parser/compiler/solver-trace-guided repair, a matched
second call with the same repair token budget and no diagnostic trace, and a
no-information trace-shaped control. A second call alone SHALL NOT be credited
as trace-guided repair. The trace-guided prompt SHALL expose only
deployment-available parser, type, compile, and solver diagnostics from the
candidate being repaired; it SHALL NOT expose hidden gold IR, answer labels,
held-family identities, certificate solutions, model self-scores, or generated
answer repair instructions.

The exact executor SHALL be the only evaluation authority. Exp5897 SHALL
measure parse validity, type validity, compilation, satisfiability handling,
exact semantic equivalence, query correctness, omitted and spurious
constraints, unsafe accepted constraints, token counts, latency, an energy
proxy, model/family/template metrics, and group-bootstrap lower bounds over
held groups. `trace_repair_mechanism_ready_score` SHALL be the bare value `1.0`
only when all three headline families complete, the held-group lower bounds
are positive for trace-guided repair over both no-trace controls, unsafe
acceptance does not increase, and the exact-verifier boundary remains intact;
otherwise it SHALL be `0.0`.

The terminal artifact MUST include `status`, `preconditions_checked`,
`upstream_gate_and_fixture_hashes`, `model_specs`, `model_file_hashes`,
`loader_and_cuda_receipts`, `frozen_prompts_decoding_seeds_and_budgets`,
`arm_definitions_and_compute_parity`, `trace_visibility_and_oracle_boundary`,
`per_model_family_and_template_metrics`,
`parse_type_compile_and_semantic_metrics`,
`omitted_spurious_and_unsafe_constraint_metrics`,
`group_bootstrap_lower_bounds`, `no_trace_and_no_information_controls`,
`gpu_utilization_vram_and_latency_receipts`, `raw_output_receipts`,
`protected_files_unchanged`, `trace_repair_mechanism_ready_score`,
`duration_s`, `inference_substrate`, `verifier_is_oracle`,
`field_provenance`, `test_commands`, `test_exit_codes`,
`reproducibility_checksum`, and `honest_verdict`.

Field principles:

- `arm_definitions_and_compute_parity`: a second call alone cannot be credited as trace-guided repair.
- `trace_visibility_and_oracle_boundary`: only deployment-available exact diagnostics may guide repair.
- `trace_repair_mechanism_ready_score`: emit bare 1.0 only for positive held lower bounds over both controls, zero unsafe increase, and all three headline families completed.
- `inference_substrate`: use `live_llm_inference`.
- `verifier_is_oracle`: true for exact evaluation; the model is only an IR proposer.
- `honest_verdict`: use `complete_positive:`, `complete_null:`, `unsafe:`, `blocked_precondition:`, or `blocked:`.

#### SCENARIO-BENCH-5897-PRECONDITIONS: Headline Resources Gate Model Load

**Given** Exp5896's checked-in fixture and mandated GGUF model ids
**When** Exp5897 checks replay, model paths, file hashes, llama.cpp CUDA
support, dual RTX 3090 health, resource budgets, GPU-offload readiness, and
protected training processes
**Then** it resolves all three headline families in frozen order or writes a
blocked-precondition artifact before any model loader is invoked.

#### SCENARIO-BENCH-5897-TRACE-BOUNDARY: Repair Sees Diagnostics Only

**Given** a model emits a candidate typed ConstraintIR object
**When** the trace-guided repair arm is prompted
**Then** the prompt includes only parser/type/compiler/solver diagnostics from
that candidate, with matched decoding and repair token budget, and excludes
gold IR, expected labels, held identities, certificate solutions, and
generated-answer repair content.

#### SCENARIO-BENCH-5897-EXACT-METRICS: Exact Executor Scores A/B Arms

**Given** completed model/row/arm outputs
**When** Exp5897 evaluates them through the Exp5896 parser, finite-domain
compiler, Z3 compiler, and exact semantic behavior hashes
**Then** it reports parse/type/compile/semantic/query, omitted/spurious,
unsafe-accept, token, latency, energy-proxy, per-group, and grouped-bootstrap
metrics, and only promotes trace-guided repair when it beats both no-trace
controls without increasing unsafe acceptance.

**Spec traces:** REQ-BENCH-5897, SCENARIO-BENCH-5897-PRECONDITIONS,
SCENARIO-BENCH-5897-TRACE-BOUNDARY, SCENARIO-BENCH-5897-EXACT-METRICS

## Implementation Status (REQ-BENCH-5897)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-BENCH-5897 | Planned (`python/carnot/experiment_5897_sota_constraint_ir_repair_ab.py`, `results/experiment_5897_sota_constraint_ir_repair_ab.json`) | Planned (`tests/python/test_experiment_5897_sota_constraint_ir_repair_ab.py`) |

### REQ-BENCH-5907: ConstraintIR Replay Contract Canonical Projection

Carnot MUST provide a shared, public, versioned canonical projection for the
Exp5896 ConstraintIR fixture artifact so the fixture producer and Exp5897
consumer compute the same replay checksum. The projection SHALL be owned by one
helper module rather than duplicated producer/consumer code. The helper SHALL
reject unknown projection versions, SHALL normalize JSON mappings, lists, and
numbers deterministically, SHALL exclude self-referential or post-hoc fields
such as wall-clock duration, test exit-code refreshes, reproducibility checksum
storage, protected-file receipts, and volatile RAM/disk availability, and SHALL
bind the actual row-file SHA-256 plus ConstraintIR artifact and row schema
versions.

The repaired contract SHALL NOT mutate historical Exp5896 or Exp5897 result
artifacts, SHALL NOT special-case the historical checksum value, SHALL NOT
weaken comparison by accepting either old or new projections, and SHALL NOT
claim model-science progress. Legacy Exp5896 evidence MAY be adjudicated under
the new projection only as immutable historical content: the old checksum
mismatch SHALL remain recorded as
`historical_checksum_mismatch_preserved`, while successful row binding,
certificate replay, schema binding, and shared-helper parity SHALL be recorded
separately as `new_contract_replay_ready`.

The terminal artifact MUST be
`results/experiment_5907_constraint_ir_replay_contract.json` and MUST include
`status`, `preconditions_checked`, `immutable_artifact_hashes`,
`mismatch_reproduction_and_root_cause`,
`canonical_projection_schema_and_version`, `excluded_and_bound_fields`,
`shared_helper_receipt`, `fresh_twin_producer_consumer_replay`,
`fresh_process_replay_receipt`, `tamper_detection_matrix`,
`legacy_exp5896_adjudication`, `historical_artifacts_unchanged`,
`protected_files_unchanged`, `constraint_ir_replay_contract_ready_score`,
`duration_s`, `inference_substrate`, `verifier_is_oracle`,
`field_provenance`, `test_commands`, `test_exit_codes`,
`reproducibility_checksum`, and `honest_verdict`.

Field principles:

- `canonical_projection_schema_and_version`: one explicit public projection owns every producer-consumer checksum.
- `historical_artifacts_unchanged`: a repaired contract cannot rewrite prior evidence.
- `constraint_ir_replay_contract_ready_score`: emit bare 1.0 only for shared-helper parity, fresh-process exact replay, row binding, and complete tamper rejection.
- `inference_substrate`: use `deterministic_artifact_replay_no_llm`.
- `verifier_is_oracle`: true only for checksum/schema/tamper adjudication.
- `honest_verdict`: use `complete_ready:`, `retired:`, or `blocked:`.

#### SCENARIO-BENCH-5907-CANONICAL: Producer And Consumer Share Projection

**Given** a freshly generated Exp5896 twin fixture
**When** the producer writes its checksum and Exp5897 replays the fixture gate
**Then** both entrypoints report the same
`carnot.constraint_ir.replay_contract_projection.v1` checksum from the shared
helper and any explicit unknown projection version is rejected.

#### SCENARIO-BENCH-5907-FRESH-PROCESS: Replay Is Independent

**Given** the freshly generated twin fixture rows and artifact on disk
**When** replay is invoked in a fresh Python process
**Then** row-file SHA-256, schema versions, certificates, and the public
projection checksum replay exactly without relying on in-process state.

#### SCENARIO-BENCH-5907-TAMPER: Bound Components Fail Closed

**Given** a producer-written twin fixture artifact
**When** the row bytes, row-file hash receipt, schema binding, projection
version, or reproducibility checksum is tampered with
**Then** replay fails closed for every tamper case.

#### SCENARIO-BENCH-5907-LEGACY: Historical Mismatch Is Preserved

**Given** the checked-in Exp5896 and Exp5897 artifacts
**When** Exp5907 reproduces the old byte projections and adjudicates Exp5896
under the new projection
**Then** the artifact records the old protected-file-hash root cause,
preserves the historical checksum mismatch, reports the new contract replay
receipt separately, and verifies the historical artifacts were not rewritten.

**Spec traces:** REQ-BENCH-5907, SCENARIO-BENCH-5907-CANONICAL,
SCENARIO-BENCH-5907-FRESH-PROCESS, SCENARIO-BENCH-5907-TAMPER,
SCENARIO-BENCH-5907-LEGACY

## Implementation Status (REQ-BENCH-5907)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-BENCH-5907 | Planned (`python/carnot/constraint_ir_replay_contract.py`, `python/carnot/experiment_5907_constraint_ir_replay_contract.py`, `results/experiment_5907_constraint_ir_replay_contract.json`) | Planned (`tests/python/test_experiment_5907_constraint_ir_replay_contract.py`) |

### REQ-BENCH-5908: VeriSynth Constraint Fixture Plan Replay

Carnot MUST provide a deterministic Exp5908 fixture planner at
`python/carnot/experiment_5908_verisynth_constraint_fixture.py` and write
`results/experiment_5908_verisynth_constraint_fixture.json` plus
`results/experiment_5908_verisynth_constraint_fixture.rows.jsonl`. Exp5908
SHALL invoke no LLM, SHALL perform no generated-answer repair, SHALL replay
Exp5907's ConstraintIR gate before constructing plans, and SHALL consume only
the checked-in Exp5896 typed ConstraintIR fixture rows.

The decomposition schema SHALL define engine-neutral units for finite
entities/domains, state facts, transition or implication relations,
invariants, explicit negation, arithmetic constraints, and query goals. Every
emitted unit SHALL map to supported Exp5896 typed IR nodes and SHALL replay
through the exact Exp5896 parser and Python/Z3 certificates.

The retrieval index SHALL use only train-visible typed structure and exact
metadata available at deployment. Paraphrases, symbol renamings,
order-permutations, held templates, omitted-constraint rows, and semantic
variants of the same template SHALL share one group identity so no held
semantic variant can enter the retrieval surface for held rows. Exp5908 SHALL
freeze prompt-plan arms for direct, semantic decomposition, decomposition plus
exact-example retrieval, wrong-family retrieval, shuffled decomposition,
omitted-component decomposition, and no-information retrieval. Treatment arms
SHALL match token envelopes and exemplar counts where applicable so
decomposition and retrieval are isolated mechanisms, not extra unbounded
context.

The terminal artifact MUST include `status`, `preconditions_checked`,
`upstream_gate_and_hashes`, `decomposition_schema_and_supported_units`,
`prompt_plan_arm_definitions`, `retrieval_index_and_visibility_contract`,
`family_template_and_group_holdouts`,
`wrong_family_shuffled_omitted_and_no_information_controls`,
`token_envelope_and_exemplar_count_parity`,
`exact_exemplar_and_component_replay`, `row_file_receipt`,
`consumer_stream_contract`, `protected_files_unchanged`,
`verisynth_constraint_fixture_ready_score`, `duration_s`,
`inference_substrate`, `verifier_is_oracle`, `field_provenance`,
`test_commands`, `test_exit_codes`, `reproducibility_checksum`, and
`honest_verdict`.

Field principles:

- `retrieval_index_and_visibility_contract`: held semantic variants cannot enter the retrieval surface.
- `prompt_plan_arm_definitions`: decomposition and retrieval must be isolated treatments rather than extra unbounded context.
- `verisynth_constraint_fixture_ready_score`: emit bare 1.0 only for exact replay, group isolation, nontrivial controls, and deterministic consumer hashes.
- `inference_substrate`: use `deterministic_exact_solver_labeled_dataset_no_llm`.
- `verifier_is_oracle`: true for fixture structure and replay labels.
- `honest_verdict`: use `ready:`, `complete_null:`, or `blocked:`.

#### SCENARIO-BENCH-5908-DECOMPOSITION: Supported Units Replay Exactly

**Given** an Exp5896 typed ConstraintIR row
**When** Exp5908 builds the semantic-decomposition plan
**Then** every component has a deterministic component hash, maps to a
supported typed IR unit, and replays through the exact parser/backend receipts.

#### SCENARIO-BENCH-5908-RETRIEVAL: Held Variants Stay Invisible

**Given** train/dev/held-out Exp5896 split groups
**When** Exp5908 builds exact-example retrieval
**Then** only train-visible typed structure and deployment metadata are indexed,
and no exemplar from the target group or a held split can be retrieved for a
held row.

#### SCENARIO-BENCH-5908-CONTROLS: Arms Are Matched Negative Controls

**Given** the frozen direct, decomposition, retrieval, wrong-family, shuffled,
omitted-component, and no-information arms
**When** Exp5908 emits row-level prompt plans
**Then** each row has deterministic arm hashes, exemplar-count parity for
retrieval arms, token-envelope parity for comparable arms, and nontrivial
wrong-family, shuffled, omitted, and no-information controls.

#### SCENARIO-BENCH-5908-STREAM: Consumer Rows Are Hash Stable

**Given** the terminal Exp5908 artifact and row JSONL
**When** a future Exp5909 consumer reads the stream contract
**Then** row-level hashes, artifact and row-file receipts, arm definitions,
visibility groups, and replay receipts are deterministic and sufficient to
replay without hidden in-process state.

**Spec traces:** REQ-BENCH-5908, SCENARIO-BENCH-5908-DECOMPOSITION,
SCENARIO-BENCH-5908-RETRIEVAL, SCENARIO-BENCH-5908-CONTROLS,
SCENARIO-BENCH-5908-STREAM

## Implementation Status (REQ-BENCH-5908)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-BENCH-5908 | Planned (`python/carnot/experiment_5908_verisynth_constraint_fixture.py`, `results/experiment_5908_verisynth_constraint_fixture.json`) | Planned (`tests/python/test_experiment_5908_verisynth_constraint_fixture.py`) |

### REQ-BENCH-5909: SOTA ConstraintIR Synthesis A/B Stream

Carnot MUST provide a live local-GGUF Exp5909 synthesis experiment at
`python/carnot/experiment_5909_sota_constraint_synthesis_ab.py` and write
`results/experiment_5909_sota_constraint_synthesis_ab.json` plus a sealed raw
proposal JSONL stream. Exp5909 SHALL replay the Exp5908 consumer gate before
any model load, SHALL use the three mandated GGUF families
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`, and SHALL resolve the first two headline
families through `cached_sota_pair(gpu_indices=(0, 1), model_indices=(0, 2))`
plus the cached third family. Missing cache, tokenizer, CUDA offload, dual RTX
3090, RAM, disk, atomic-output, or protected-workload preconditions SHALL write
an honest `blocked_precondition:` artifact before model generation begins.

The experiment SHALL run direct ConstraintIR synthesis, semantic-decomposition
synthesis, and decomposition plus exact-example retrieval on every Exp5908 row
for every completed model. It SHALL also run wrong-family retrieval and shuffled
decomposition on a preregistered confirmatory subset. All arms SHALL use matched
decoding, output-token, seed, call-count, and wall-clock budget rules so
decomposition and retrieval cannot win by unbounded context or extra calls.
Prompts SHALL NOT expose target hidden gold IR, target exact labels, held
identities, certificates, or diagnostic repair traces; retrieved examples may
include only deployment-visible exact examples selected by Exp5908's leakage
contract.

The exact Exp5896 parser, type checker, Python backend, Z3 backend, and
semantic replay labels SHALL own outcomes after each proposal is sealed.
Exp5909 SHALL report parse, type, compile, satisfiability, exact semantic
equivalence, query correctness, omitted/spurious constraints, unsafe accepted
constraints, token, latency, VRAM, GPU-utilization, and energy-proxy metrics by
model, family, template, arm, and group. It SHALL seal the chronological raw
proposal stream with raw output hashes, prompt hashes, exact labels,
deployment-visible diagnostics, group IDs, model file hashes, and row hashes for
downstream transactional learning.

The terminal artifact MUST include `status`, `preconditions_checked`,
`upstream_gate_and_fixture_hashes`, `model_specs`, `model_file_hashes`,
`embedded_tokenizer_and_loader_cuda_receipts`,
`frozen_prompts_decoding_seeds_and_budgets`,
`arm_definitions_and_compute_parity`, `retrieval_and_oracle_visibility`,
`per_model_family_template_metrics`,
`parse_type_compile_and_semantic_metrics`,
`omitted_spurious_and_unsafe_constraint_metrics`,
`group_bootstrap_lower_bounds`, `wrong_retrieval_and_shuffled_controls`,
`chronological_raw_stream_receipt`, `residual_error_and_diagnostic_headroom`,
`gpu_utilization_vram_latency_and_energy_receipts`,
`protected_files_unchanged`, `constraint_stream_ready_score`,
`verification_repair_admission_ready_score`, `duration_s`,
`inference_substrate`, `verifier_is_oracle`, `field_provenance`,
`test_commands`, `test_exit_codes`, `reproducibility_checksum`, and
`honest_verdict`.

Field principles:

- `arm_definitions_and_compute_parity`: decomposition and retrieval cannot win by unbounded context or calls.
- `chronological_raw_stream_receipt`: downstream learning sees only sealed model outputs and deployment-visible evidence in event order.
- `constraint_stream_ready_score`: emit bare 1.0 only for all three real models, exact labels, raw hashes, and zero authority violation.
- `verification_repair_admission_ready_score`: emit bare 1.0 only for sufficient exact residual headroom with usable non-oracle diagnostics.
- `inference_substrate`: use `live_llm_inference`.
- `verifier_is_oracle`: true for exact evaluation and never for model proposal credit.
- `honest_verdict`: use `complete_positive:`, `complete_null:`, `unsafe:`, `blocked_precondition:`, or `blocked:`.

#### SCENARIO-BENCH-5909-PRECONDITIONS: Gate Blocks Before Model Load

**Given** the checked-in Exp5908 fixture stream and mandated local GGUF ids
**When** any upstream replay, cache, embedded tokenizer, CUDA, dual-GPU, RAM,
disk, atomic-output, or protected-workload precondition fails
**Then** Exp5909 writes a `blocked_precondition:` artifact, records all
precondition receipts, and does not call the live model collector.

#### SCENARIO-BENCH-5909-PROMPTS: Prompt Arms Stay Leakage Safe

**Given** an Exp5908 plan row and its Exp5896 source row
**When** Exp5909 builds direct, decomposition, retrieval, wrong-retrieval, and
shuffled-control prompts
**Then** no prompt contains target hidden ConstraintIR JSON, target exact
labels, certificates, behavior hashes, held identities, or diagnostic repair
traces, and retrieval examples exclude target-group and held-split examples.

#### SCENARIO-BENCH-5909-STREAM: Raw Proposals Are Chronological And Exact Scored

**Given** three completed mandated model families and stubbed raw proposals
**When** Exp5909 scores the direct, decomposition, retrieval, and preregistered
control arms
**Then** the raw JSONL stream is chronological, every event has raw and prompt
hashes, every event receives exact parser/type/compile/semantic labels, and the
artifact exposes complete metric groups without accepting model outputs as an
oracle.

#### SCENARIO-BENCH-5909-HEADROOM: Repair Admission Requires Residual Diagnostics

**Given** a complete Exp5909 stream
**When** residual incorrect rows remain in every required fixture group
**Then** `verification_repair_admission_ready_score` is 1.0 only if those rows
carry deployment-visible diagnostics and no hidden oracle payload; otherwise it
is 0.0 even if `constraint_stream_ready_score` is 1.0.

**Spec traces:** REQ-BENCH-5909, SCENARIO-BENCH-5909-PRECONDITIONS,
SCENARIO-BENCH-5909-PROMPTS, SCENARIO-BENCH-5909-STREAM,
SCENARIO-BENCH-5909-HEADROOM

## Implementation Status (REQ-BENCH-5909)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-BENCH-5909 | Planned (`python/carnot/experiment_5909_sota_constraint_synthesis_ab.py`, `results/experiment_5909_sota_constraint_synthesis_ab.json`) | Planned (`tests/python/test_experiment_5909_sota_constraint_synthesis_ab.py`) |

### REQ-BENCH-6504: Exact Structural Benchmark Commitment SHALL Be Sealed Before Learned Work

The Exp6504 workflow SHALL create a procedural exact SAT/CSP benchmark before
any learned ranker, mutation stream, or downstream SOTA route can inspect held
rows. It SHALL generate random 3-CNF, pseudo-industrial 3-CNF, Tseitin,
pigeonhole, graph-coloring, and small scheduling instances with deterministic
local generators and fixed seeds. Each raw row SHALL preserve the formal
instance, source family, scale, surface relabeling, density, proof family,
lineage id, generator seed, and checksum before feature extraction.

Exp6504 SHALL label every accepted instance with exact local solver authority.
Accepted labels SHALL replay under exhaustive enumeration and Z3, SHALL include
a model witness for SAT or feasible rows, SHALL include an executable
refutation receipt for UNSAT or infeasible rows, and SHALL quarantine any
backend disagreement. No label SHALL be hand-corrected.

Exp6504 SHALL commit train, development, and held splits by base-instance
lineage before labels or features can affect the split. No base lineage or
duplicate structural hash SHALL cross split boundaries. The held set SHALL
contain at least 30 units in every planned headline cell. The headline cells
SHALL include family cells and one-axis label, scale, and surface cells.

Exp6504 SHALL freeze family, scale, surface, density, proof family, source,
structural hardness, and solver-effort strata. Solver effort MAY be recorded as
an audit stratum, but SHALL NOT be used as a model-difficulty proxy.

Exp6504 SHALL test identity, row order, serialization length, generator seed,
family, duplicate lineage, label balance, solver backend, and split leakage.
Every shortcut attack SHALL fail closed before
`base_structural_benchmark_ready_score` can equal `1.0`.

The Exp6504 artifact SHALL be written to
`results/experiment_6504_exact_structural_benchmark_commitment.json` with
`inference_substrate=procedural_formal_instances_and_exact_solver_labels_no_llm`
and `verifier_is_oracle=true` only for exact solver labels and executable
validity. The artifact SHALL include, at minimum, `status`, `verdict_class`,
`upstream_gate_receipts`, `benchmark_schema`, `raw_instance_rows`,
`exact_label_rows`, `exact_replay_rows`, `split_commitment`,
`stratum_balance_rows`, `minimum_held_cell_size`, `leakage_attack_matrix`,
`base_structural_benchmark_ready_score`, `per_unit_rows`,
`aggregate_row_recomputation`, `gate_check_summary`, `preconditions_checked`,
`protected_files_unchanged`, `inference_substrate`, `verifier_is_oracle`,
`field_principles`, `field_provenance`, `random_seed`, `duration_s`,
`tests_run`, `reproducibility_checksum`, and `honest_verdict`.

Field principles SHALL be:

- `status`: "Records the terminal benchmark state."
- `verdict_class`: "Closed enum: positive | circular_positive | null | blocked | disqualified | partial."
- `upstream_gate_receipts`: "Bind both same-roadmap gates and observed values."
- `benchmark_schema`: "Versions instances, labels, proofs, models, strata, and splits."
- `raw_instance_rows`: "Preserves every formal instance before feature extraction."
- `exact_label_rows`: "Provides solver-certified labels and authority receipts."
- `exact_replay_rows`: "Checks every accepted label with a deterministic replay."
- `split_commitment`: "Freezes lineage-separated train, development, and held sets."
- `stratum_balance_rows`: "Records family, scale, surface, hardness, source, and label counts."
- `minimum_held_cell_size`: "Enforces at least 30 units for percentage comparisons."
- `leakage_attack_matrix`: "Tests identity, order, length, seed, family, duplicate, backend, and split shortcuts."
- `base_structural_benchmark_ready_score`: "Same-roadmap gate for the base benchmark."
- `per_unit_rows`: "Carries instance, label, replay, split, stratum, and attack rows."
- `aggregate_row_recomputation`: "Recomputes counts and readiness from raw rows."
- `gate_check_summary`: "Names any failed gate, solver, cell-size, or replay check and observed value."
- `preconditions_checked`: "Records gates, solver tools, resources, repository, and storage checks."
- `protected_files_unchanged`: "Proves protected files stayed unchanged."
- `inference_substrate`: "Declares procedural generators and exact local solvers with no LLM."
- `verifier_is_oracle`: "True only for exact solver labels and executable validity checks."
- `field_principles`: "Explains why each commitment field exists."
- `field_provenance`: "Maps rows to generators, seeds, solver commands, and hashes."
- `random_seed`: "Records all generator and split seeds."
- `duration_s`: "Records measured wall time."
- `tests_run`: "Records commands and exit codes."
- `reproducibility_checksum`: "Hashes instances, labels, splits, strata, and attacks."
- `honest_verdict`: "Uses complete_* when the commitment is valid or blocked_* with gate_check_summary."

#### SCENARIO-BENCH-6504-GENERATION: Procedural Families Are Deterministic

**Given** the Exp6503 method contract names six SAT/CSP families
**When** Exp6504 generates benchmark rows
**Then** every family emits deterministic raw instances from fixed local seeds
and every raw row carries source, family, scale, surface, lineage, density, and
checksum fields before feature extraction.

#### SCENARIO-BENCH-6504-LABELS: Exact Backends Own Labels

**Given** raw formal instances
**When** Exp6504 labels and replays them
**Then** every accepted label agrees across exhaustive and Z3 backends, SAT rows
carry valid model witnesses, UNSAT rows carry executable refutation receipts,
and disagreements are quarantined without hand correction.

#### SCENARIO-BENCH-6504-SPLITS: Held Cells Are Lineage Separated

**Given** generated base-instance lineages
**When** Exp6504 commits train, development, and held splits
**Then** no lineage crosses splits, split rows predate feature extraction, and
every planned held headline cell contains at least 30 units.

#### SCENARIO-BENCH-6504-STRATA: Surface And Solver Effort Stay Separate

**Given** accepted labels and split rows
**When** Exp6504 emits strata
**Then** rows record family, scale, surface, structural hardness, density,
proof family, source, solver effort, and label counts while declaring that
solver effort is not a model-difficulty proxy.

#### SCENARIO-BENCH-6504-LEAKAGE: Shortcut Attacks Fail Closed

**Given** raw rows, label rows, split rows, and backend receipts
**When** Exp6504 runs shortcut attacks
**Then** identity, row order, serialization length, generator seed, family,
duplicate lineage, label balance, backend, and split shortcuts are blocked from
feature use and no false accept remains.

#### SCENARIO-BENCH-6504-SCHEMA: Artifact Is Ready And Checksummed

**Given** rows, receipts, field maps, protected hashes, and command receipts
**When** Exp6504 validates the artifact
**Then** every required field has a principle and provenance entry, the
checksum matches, exact replay passes, `base_structural_benchmark_ready_score`
equals `1.0`, and the terminal verdict starts with `complete_` or `blocked_`.

## Implementation Status (REQ-BENCH-6504)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-BENCH-6504 | Planned (`python/carnot/experiment_6504_exact_structural_benchmark_commitment.py`, `results/experiment_6504_exact_structural_benchmark_commitment.json`) | Planned (`tests/python/test_experiment_6504_exact_structural_benchmark_commitment.py`) |

### REQ-BENCH-6505: Local SOTA Formal Challenge Mutations SHALL Be One-Shot And Exact-Admitted

Carnot SHALL provide Exp6505 at
`python/carnot/experiment_6505_sota_formal_challenge_mutations.py`.
The command
`.venv/bin/python -m carnot.experiment_6505_sota_formal_challenge_mutations --date 20260822`
SHALL write
`results/experiment_6505_sota_formal_challenge_mutations.json`.

Exp6505 SHALL evaluate the Exp6504 gate before model loading. It SHALL record
the Exp6504 path, hash, `base_structural_benchmark_ready_score`, expected value,
and observed value. It SHALL check model cache paths, GGUF file sizes,
llama_cpp CUDA support, GPU inventory, free VRAM, disk space, repository state,
and protected-file hashes before loading any model. Every failed precondition
SHALL be named with its observed value.

Exp6505 SHALL use exactly these GGUF model families in `MODEL_SPECS`:
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. It SHALL resolve local GGUF files through the
canonical cache resolver. It SHALL load one model at a time through llama_cpp.
It SHALL use the tokenizer embedded in each GGUF. It SHALL NOT call
`AutoTokenizer` for a GGUF repository.

Exp6505 SHALL select development-only source instances from Exp6504 before
generation. For every selected source and every model family it SHALL issue one
request. It SHALL not retry a request after parsing or admission fails. It SHALL
persist exact request bytes, response bytes, hashes, model file identity, seed,
and terminal runtime receipt before parsing any response.

Exp6505 SHALL freeze a bounded edit DSL for clause, edge, color, job,
precedence, coefficient, and relabel operations. The prompt SHALL forbid
answers, labels, solver advice, free-form semantic translation, release
decisions, hidden held access, and model-identity shortcuts. The parser SHALL
reject those outputs before exact admission.

Exp6505 SHALL apply syntactically valid edit scripts to a copy of the formal
development instance only. Exact local tools SHALL parse, replay, solve, check
model or proof validity, compute novelty, classify label ambiguity, and accept
or quarantine each mutation. Invalid, duplicate, unchanged, answer-bearing,
label-bearing, semantic-translation, solver-advice, release-decision, or
ambiguous outputs SHALL be terminal quarantine rows. Models SHALL never be
oracles.

Exp6505 SHALL report valid yield, novelty, source family, source scale, exact
label, edit type, truncation, parser disposition, and runtime outcome separately
for each SOTA family. A model-family comparison SHALL contain one row for every
precommitted request. `challenge_generation_complete_score` SHALL be `1.0` when
every precommitted request has one terminal receipt, even when valid yield is
zero. `challenge_pool_ready_score` SHALL be separate and SHALL not block the
base Exp6504 benchmark.

The terminal artifact SHALL include `status`, `verdict_class`,
`upstream_gate_receipt`, `model_specs`, `model_runtime_receipts`,
`mutation_grammar`, `source_selection_receipt`,
`raw_request_response_receipts`, `rows`, `exact_admission_rows`,
`model_family_results`, `challenge_generation_complete_score`,
`challenge_pool_ready_score`, `prohibited_output_attack_matrix`,
`per_unit_rows`, `aggregate_row_recomputation`, `gate_check_summary`,
`preconditions_checked`, `protected_files_unchanged`, `inference_substrate`,
`verifier_is_oracle`, `field_principles`, `field_provenance`, `random_seed`,
`duration_s`, `tests_run`, `reproducibility_checksum`, and `honest_verdict`.
`inference_substrate` SHALL be
`local_llama_cpp_three_family_formal_mutation_generation_plus_exact_solver_admission`.
`verifier_is_oracle` SHALL be true only for exact parsing, solving, and validity.
`verdict_class` SHALL be one of `positive`, `circular_positive`, `null`,
`blocked`, `disqualified`, or `partial`.

- `status`: "Records terminal generation and admission state."
- `verdict_class`: "Closed enum: positive | circular_positive | null | blocked | disqualified | partial."
- `upstream_gate_receipt`: "Binds the base benchmark gate and observed value."
- `model_specs`: "Lists all three mandated GGUF repositories, files, quantization, and roles."
- `model_runtime_receipts`: "Records llama_cpp backend, tokenizer, offload, VRAM, and timing per model."
- `mutation_grammar`: "Freezes the bounded formal edit language."
- `source_selection_receipt`: "Proves all sources are development-only and chosen before generation."
- `raw_request_response_receipts`: "Preserves exact bytes and hashes before parsing."
- `rows`: "Provides one request, parse, edit, exact-check, and disposition row per unit."
- `exact_admission_rows`: "Records parse, replay, label, proof/model, novelty, and quarantine outcomes."
- `model_family_results`: "Reports valid yield and failure modes separately for each SOTA family."
- `challenge_generation_complete_score`: "Same-roadmap gate for complete request accounting."
- `challenge_pool_ready_score`: "Separates useful valid mutations from execution completeness."
- `prohibited_output_attack_matrix`: "Tests answers, labels, heuristics, semantic translation, retries, held access, and model identity shortcuts."
- `per_unit_rows`: "Carries every request and exact admission result."
- `aggregate_row_recomputation`: "Recomputes all yields, counts, and scores from rows."
- `gate_check_summary`: "Names any failed benchmark, cache, CUDA, GPU, parse, or runtime check and observed value."
- `preconditions_checked`: "Records gate, cache, GPU, runtime, tokenizer, disk, and repository checks."
- `protected_files_unchanged`: "Proves protected files stayed unchanged."
- `inference_substrate`: "Declares local llama_cpp GGUF generation plus exact CPU parsing and solving."
- `verifier_is_oracle`: "True only for exact formal admission checks; models are not oracles."
- `field_principles`: "Explains each generation, admission, and boundary field."
- `field_provenance`: "Maps rows to model files, raw hashes, parser functions, and solver receipts."
- `random_seed`: "Records source selection and generation seeds."
- `duration_s`: "Records measured wall time."
- `tests_run`: "Records commands and exit codes."
- `reproducibility_checksum`: "Hashes specs, raw bytes, parses, admissions, and rows."
- `honest_verdict`: "Uses complete_* for complete accounting, complete_null for zero useful yield, or blocked_* with gate_check_summary for unmet execution preconditions."

#### SCENARIO-BENCH-6505-PROVENANCE: Gate And Model Receipts Precede Loading

**Given** the Exp6504 benchmark artifact and local SOTA model registry
**When** Exp6505 starts
**Then** it records the Exp6504 gate, model cache paths, file sizes, llama_cpp
CUDA support, GPU inventory, free VRAM, disk, and protected hashes before any
model load event.

#### SCENARIO-BENCH-6505-ONE-SHOT: Development Requests Are Precommitted

**Given** Exp6504 development rows
**When** Exp6505 selects source instances
**Then** every selected source has `split=development`, selection occurs before
generation, and exactly one request is issued for each source and model pair.

#### SCENARIO-BENCH-6505-NO-ANSWER: Prohibited Outputs Fail Closed

**Given** model output that contains an answer, label, solver advice,
semantic translation, release decision, retry attempt, held access, or
model-identity shortcut
**When** Exp6505 parses the output
**Then** the row is rejected before exact admission and no accepted row contains
the prohibited content.

#### SCENARIO-BENCH-6505-ADMISSION: Exact Tools Own Mutation Admission

**Given** a syntactically valid bounded edit script
**When** Exp6505 applies it to a copy of a development formal instance
**Then** exact local parsing, replay, solving, model/proof validation, novelty,
duplicate, unchanged, and label-ambiguity checks determine acceptance or
quarantine without model judgment.

#### SCENARIO-BENCH-6505-SCORES: Complete Accounting Is Separate From Pool Readiness

**Given** all precommitted requests have terminal receipts
**When** Exp6505 recomputes aggregate rows
**Then** `challenge_generation_complete_score` is `1.0`, model-family results
include one row per request, and `challenge_pool_ready_score` is derived only
from accepted useful mutations.

#### SCENARIO-BENCH-6505-ARTIFACT: Artifact Is Checksummed And Boundary-Safe

**Given** raw bytes, exact admission rows, runtime receipts, protected hashes,
field principles, field provenance, tests, and attacks
**When** Exp6505 validates the artifact
**Then** every required field is present, the checksum matches,
`verifier_is_oracle` is scoped to exact tools, protected files are unchanged,
and the terminal verdict starts with `complete_`, `complete_null`, or
`blocked_`.

## Implementation Status (REQ-BENCH-6505)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-BENCH-6505 | Planned (`python/carnot/experiment_6505_sota_formal_challenge_mutations.py`, `results/experiment_6505_sota_formal_challenge_mutations.json`) | Planned (`tests/python/test_experiment_6505_sota_formal_challenge_mutations.py`) |

### REQ-BENCH-6506: V561 Evidence Corrigendum And V562 Lineage Lock SHALL Be Append-Only

Carnot SHALL provide Exp6506 at
`python/carnot/experiment_6506_v561_evidence_corrigendum_v562_lineage_lock.py`.
The command
`.venv/bin/python -m carnot.experiment_6506_v561_evidence_corrigendum_v562_lineage_lock --date 20260822`
SHALL write
`results/experiment_6506_v561_evidence_corrigendum_v562_lineage_lock.json`.

Exp6506 SHALL not modify Exp6504, Exp6505, historical roadmaps, or
`scripts/research_conductor.py`. It SHALL hash those files before and after the
corrigendum. A changed historical artifact SHALL block
`v562_exact_branch_ready_score`.

Exp6506 SHALL load Exp6504 from its checked-in artifact and recompute label,
replay, split, held-cell, stratum, leakage, aggregate, and checksum rows from
raw rows. It SHALL not trust Exp6504 aggregate fields. If row replay passes, it
MAY classify Exp6504 as operational benchmark-ready for exact branch-advice
work. It SHALL NOT reuse the Exp6504 `positive` verdict class as a scientific
verifier-value claim because `verifier_is_oracle=true`.

Exp6506 SHALL rerun adversarial verification against Exp6504, record the
observed `VERDICT_CLASS_MISMATCH`, and emit a separate corrigendum disposition.
The corrected disposition SHALL preserve provenance, declare that exact labels
and row checks are oracle checks, and keep any top-level Exp6506
`verdict_class` at `partial` or `null` unless an oracle-distinct scientific
comparison exists.

Exp6506 SHALL load Exp6505 as a terminal null. It SHALL recompute request
accounting and accepted-yield rows from Exp6505 rows. It SHALL freeze
`challenge_generation_complete_score=1.0` and
`challenge_pool_ready_score=0.0` as terminal evidence. It SHALL not invoke a
model and SHALL not make the empty mutation pool a downstream prerequisite.

Exp6506 SHALL authorize only the new exact branch-advice lineage. Downstream
V562 work MAY consume Exp6504 raw instances and exact labels. Downstream V562
work SHALL NOT consume Exp6505 challenge mutations, learned trajectory energy,
factor causal value, factor spawning, ARC policy work, hardware acceleration,
renamed retired scopes, aggregate-only corrections, missing upstream hashes, or
the old positive-class headline.

The artifact SHALL include `status`, `verdict_class`,
`cited_upstream_artifacts`, `exp6504_row_recomputation`,
`exp6504_corrigendum`, `exp6505_terminal_null_receipt`,
`lineage_decision_rows`, `forbidden_dependency_attack_matrix`,
`v562_exact_branch_ready_score`, `per_unit_rows`, `gate_check_summary`,
`preconditions_checked`, `protected_files_unchanged`, `inference_substrate`,
`verifier_is_oracle`, `field_principles`, `field_provenance`, `random_seed`,
`duration_s`, `tests_run`, `reproducibility_checksum`, and `honest_verdict`.
`inference_substrate` SHALL be
`independent_v561_artifact_replay_no_llm`. `verifier_is_oracle` SHALL be true
only for exact label, hash, and row checks.

`v562_exact_branch_ready_score` SHALL be an evidence-lineage gate. It SHALL
depend on Exp6504 row replay, the corrected class, historical-file hashes,
Exp6505 terminal-null accounting, and forbidden dependency checks. It SHALL NOT
turn a repository-wide test-suite failure into a scientific lineage failure.
Repository validation failures SHALL remain visible in `tests_run` and SHALL
NOT be rewritten as passing receipts.

Field principles SHALL be:

- `status`: "A terminal state distinguishes a valid corrigendum from an incomplete replay."
- `verdict_class`: "The closed enum prevents an exact-oracle self-check from becoming an unsupported positive science claim."
- `cited_upstream_artifacts`: "Paths, imported fields, and hashes bind every correction to immutable V561 evidence."
- `exp6504_row_recomputation`: "Independent row replay detects aggregate or classification errors without rewriting history."
- `exp6504_corrigendum`: "A separate correction preserves provenance and makes the eligible interpretation explicit."
- `exp6505_terminal_null_receipt`: "The receipt prevents a zero-yield SOTA stream from becoming a hidden prerequisite or a silent rerun."
- `lineage_decision_rows`: "One row per allowed or forbidden scope makes lineage decisions recheckable."
- `forbidden_dependency_attack_matrix`: "Adversarial cases ensure renamed or indirect dependencies fail closed."
- `v562_exact_branch_ready_score`: "A same-roadmap gate prevents downstream work from using quarantined evidence."
- `per_unit_rows`: "Unit rows let a third party recompute every headline and gate without rerunning V561."
- `gate_check_summary`: "A blocked verdict must name the failed check and observed value."
- `preconditions_checked`: "Explicit repository and artifact checks prevent synthesized success when inputs are missing."
- `protected_files_unchanged`: "Historical roadmaps, artifacts, and the conductor must remain immutable during correction."
- `inference_substrate`: "Declaring artifact replay with no LLM prevents substrate and SOTA-policy confusion."
- `verifier_is_oracle`: "Oracle disclosure prevents circular exact checks from supporting a verifier-value headline."
- `field_principles`: "The artifact must preserve why each evidence field is load-bearing."
- `field_provenance`: "JSON pointers and reducers make each corrected field independently traceable."
- `random_seed`: "A fixed attack order makes the fail-closed checks reproducible."
- `duration_s`: "Measured wall time helps detect implausible or fabricated replay work."
- `tests_run`: "Command and exit-code receipts show which validation actually ran."
- `reproducibility_checksum`: "A content hash detects later drift in inputs, rows, or decisions."
- `honest_verdict`: "A terminal complete_* or blocked_* prefix lets the conductor classify the result safely."

#### SCENARIO-BENCH-6506-ROW-REPLAY: Exp6504 Is Recomputed From Rows

**Given** the immutable Exp6504 artifact
**When** Exp6506 runs
**Then** it recomputes labels, exact replays, split rows, held-cell rows,
strata, leakage attacks, aggregates, and the checksum from raw rows and records
whether each recomputed value matches the historical artifact.

#### SCENARIO-BENCH-6506-CORRIGENDUM: Positive Oracle Reuse Is Corrected

**Given** Exp6504 declares `verdict_class=positive` with
`verifier_is_oracle=true`
**When** Exp6506 reruns adversarial verification
**Then** it records `VERDICT_CLASS_MISMATCH`, emits a separate corrected
disposition, and keeps the corrigendum from making a positive scientific
verifier-value claim.

#### SCENARIO-BENCH-6506-EXP6505-NULL: The Mutation Stream Is Terminal Null

**Given** the Exp6505 artifact
**When** Exp6506 recomputes request and admission accounting
**Then** `challenge_generation_complete_score` remains `1.0`,
`challenge_pool_ready_score` remains `0.0`, accepted mutation count remains
zero, and no model is invoked.

#### SCENARIO-BENCH-6506-LINEAGE-LOCK: V562 Dependencies Fail Closed

**Given** allowed exact branch-advice inputs and forbidden retired or null
dependencies
**When** Exp6506 evaluates lineage rows and attacks
**Then** only Exp6504 raw instances and exact labels are allowed, while
Exp6505 challenge rows, retired lineages, renamed scopes, missing hashes,
aggregate-only corrections, challenge-pool laundering, and positive-class reuse
fail closed before `v562_exact_branch_ready_score` can equal `1.0`.

## Implementation Status (REQ-BENCH-6506)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-BENCH-6506 | Implemented (`python/carnot/experiment_6506_v561_evidence_corrigendum_v562_lineage_lock.py`, `results/experiment_6506_v561_evidence_corrigendum_v562_lineage_lock.json`) | Implemented (`tests/python/test_experiment_6506_v561_evidence_corrigendum_v562_lineage_lock.py`) |

### REQ-BENCH-6510: V563 Independent Exact Root SHALL Qualify Immutable Evidence Without Retired Task Dependencies

Carnot SHALL provide Exp6510 at
`python/carnot/experiment_6510_v563_independent_exact_root.py`.
The command
`.venv/bin/python -m carnot.experiment_6510_v563_independent_exact_root --date 20260822`
SHALL write
`results/experiment_6510_v563_independent_exact_root.json`.

Exp6510 SHALL read Exp6504 and Exp6506 directly as immutable JSON files. It
SHALL record exact input paths, byte hashes, JSON pointers, solver
availability, CPU, RAM, disk, git status, exclusion-manifest state, and
protected-file hashes before qualification. It SHALL not create a bootstrap
stub. It SHALL not rerun the Exp6506 computation or require the retired
Exp6506 task ID.

Exp6510 SHALL recompute Exp6504 row counts, exact-label receipts, replay
receipts, split hashes, historical hashes, corrected class, and readiness from
file content. It SHALL not trust aggregate-only fields. It SHALL also
recompute the Exp6506 corrigendum contract from the Exp6506 artifact content:
`v562_exact_branch_ready_score`, corrected non-positive class, allowed fields,
forbidden lineage rows, attack rows, and protected historical hashes.

Exp6510 SHALL parse `ops/conductor-log.md` and preserve the V562 conductor
terminal result. It SHALL record the three
`artifact_not_updated_past_bootstrap` rows for Exp6506 and the downstream
Exp6507-Exp6509 gate-block cascade. It SHALL not reactivate Exp6506, Exp6507,
Exp6508, or Exp6509 as structured dependencies.

Exp6510 SHALL authorize only the new V563 exact branch-counterfactual path.
The only allowed historical content is Exp6504 raw instances, Exp6504 exact
labels, and Exp6506 content as immutable receipt evidence. It SHALL forbid
missing files, stale hashes, aggregate-only trust, historical mutation,
renamed retired dependencies, indirect Exp6505 challenge-pool use, and a
positive-class exact-oracle claim.

The artifact SHALL include `status`, `verdict_class`,
`prior_failure_receipt`, `historical_input_receipts`,
`independent_row_recomputation`, `lineage_decision_rows`,
`retired_dependency_attack_matrix`, `atomic_terminal_write_receipt`,
`v563_independent_root_ready_score`, `per_unit_rows`, `gate_check_summary`,
`preconditions_checked`, `protected_files_unchanged`, `inference_substrate`,
`verifier_is_oracle`, `field_principles`, `field_provenance`, `random_seed`,
`duration_s`, `tests_run`, `reproducibility_checksum`, and
`honest_verdict`. `inference_substrate` SHALL be
`bounded_independent_historical_artifact_replay_no_llm`.
`verifier_is_oracle` SHALL be true only for exact row and hash checks.
`verdict_class` SHALL be `partial`, `null`, or `blocked`; it SHALL NOT be
`positive`.

`v563_independent_root_ready_score` SHALL equal `1.0` only when direct row
checks pass, historical files are unchanged, the verdict class is null or
partial, every forbidden lineage attack fails closed, no structured dependency
names Exp6506, Exp6507, Exp6508, or Exp6509, and the target artifact is a
complete atomic terminal artifact rather than a bootstrap stub.

Field principles SHALL be:

- `status`: "A terminal state distinguishes the new root from a bootstrap or partial replay."
- `verdict_class`: "The closed class prevents exact-oracle readiness from becoming an unsupported positive result."
- `prior_failure_receipt`: "The receipt binds the changed technique to the Exp6506 conductor failure."
- `historical_input_receipts`: "Exact paths, hashes, and JSON pointers make direct immutable-file use auditable."
- `independent_row_recomputation`: "Row-derived checks prevent trust in stale or contradictory aggregates."
- `lineage_decision_rows`: "One row per allowed and forbidden scope proves retired dependencies fail closed."
- `retired_dependency_attack_matrix`: "Attacks detect renamed, indirect, or structured reuse of retired tasks."
- `atomic_terminal_write_receipt`: "The receipt addresses the prior bootstrap-update failure with a new bounded write path."
- `v563_independent_root_ready_score`: "This exact field is the same-roadmap gate for the new branch dataset."
- `per_unit_rows`: "Per-unit rows make all counts, decisions, and failures independently checkable."
- `gate_check_summary`: "A blocked result must name the failed path, hash, class, or lineage check and its observed value."
- `preconditions_checked`: "Explicit checks prevent a root-ready claim when files, solvers, or resources are absent."
- `protected_files_unchanged`: "Historical artifacts, research-roadmap.yaml, and the conductor must remain unchanged."
- `inference_substrate`: "Declaring bounded artifact replay with no LLM keeps substrate and SOTA policy explicit."
- `verifier_is_oracle`: "Oracle disclosure prevents exact consistency checks from supporting verifier-value claims."
- `field_principles`: "Load-bearing reasons help later tasks preserve the evidence contract."
- `field_provenance`: "Paths, hashes, reducers, and source lines make every field traceable."
- `random_seed`: "A fixed attack order makes the qualifier reproducible."
- `duration_s`: "Measured wall time supports fabrication and bounded-work checks."
- `tests_run`: "Command and exit-code receipts show which validation actually ran."
- `reproducibility_checksum`: "A content hash detects later drift in inputs, rows, or lineage decisions."
- `honest_verdict`: "A complete_* or blocked_* prefix gives the conductor a safe terminal result."

#### SCENARIO-BENCH-6510-DIRECT-IMMUTABLE: Historical Files Are Qualified Directly

**Given** the checked-in Exp6504 and Exp6506 artifacts
**When** Exp6510 runs
**Then** it records exact paths, byte hashes, JSON pointers, row counts, label
receipts, replay receipts, split hashes, protected hashes, solver state, and
resource preconditions from those files without invoking Exp6506.

#### SCENARIO-BENCH-6510-RETIRED-ISOLATION: Retired Task IDs Stay Out Of Dependencies

**Given** Exp6506 failed in the conductor as
`artifact_not_updated_past_bootstrap` and Exp6507-Exp6509 were blocked
**When** Exp6510 emits lineage rows
**Then** prior failures appear only in `prior_failure_receipt`, no structured
dependency names Exp6506, Exp6507, Exp6508, or Exp6509, and only the new
exact branch-counterfactual path is allowed.

#### SCENARIO-BENCH-6510-ATTACKS: Retired And Aggregate Shortcuts Fail Closed

**Given** missing files, stale hashes, aggregate-only fields, historical
mutation, renamed retired scopes, indirect Exp6505 challenge-pool use, and a
positive exact-oracle class claim
**When** Exp6510 evaluates attacks
**Then** each attack row fails closed before
`v563_independent_root_ready_score` can equal `1.0`.

#### SCENARIO-BENCH-6510-ATOMIC-TERMINAL: Artifact Write Is Complete And Atomic

**Given** all independent qualification gates pass
**When** Exp6510 writes its deliverable
**Then** the artifact is written in one atomic terminal operation, contains all
required fields, records the write receipt, is not a bootstrap stub, and has an
honest verdict with a `complete_` or `blocked_` prefix.

#### SCENARIO-BENCH-6510-VERDICT-CLASS: Exact Oracle Readiness Is Non-Positive

**Given** exact row and hash checks use oracle authority
**When** Exp6510 validates the artifact
**Then** `verdict_class` is `partial`, `null`, or `blocked`, never `positive`,
and `verifier_is_oracle=true` does not support a scientific performance claim.

## Implementation Status (REQ-BENCH-6510)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-BENCH-6510 | Implemented (`python/carnot/experiment_6510_v563_independent_exact_root.py`, `results/experiment_6510_v563_independent_exact_root.json`) | Implemented (`tests/python/test_experiment_6510_v563_independent_exact_root.py`) |

### REQ-BENCH-6512: Independent Branch Dataset Audit SHALL Close The Readiness Gate

Carnot SHALL provide Exp6512 at
`python/carnot/experiment_6512_branch_dataset_independent_audit.py`.
The command
`.venv/bin/python -m carnot.experiment_6512_branch_dataset_independent_audit --date 20260822`
SHALL write
`results/experiment_6512_branch_dataset_independent_audit.json`.

Exp6512 SHALL audit
`results/experiment_6511_exact_branch_counterfactual_dataset_v2.json` without
a conductor gate. It SHALL record whether the upstream path exists, its byte
hash, terminal status, row count, shard count, solver availability, resource
state, and protected-file hashes before the audit. If the upstream file is
missing, null, blocked, partial, unreadable, or invalid, Exp6512 SHALL still
write a complete terminal artifact with
`branch_dataset_audited_ready_score=0.0`.

Exp6512 SHALL recompute all dataset counts and aggregate values from dataset
rows. It SHALL not trust upstream aggregate fields. It SHALL replay exact
receipt fields per row, check immutable base-instance hashes, rebuild train,
development, and held lineage sets, and reject base-lineage overlap, post-held
repair, duplicate checkpoints, asymmetric budgets, missing terminal
dispositions, incomplete shards, missing resume receipts, broken hash chains,
bad censoring accounting, and features unavailable at decision time.

Exp6512 SHALL run deterministic shortcut attacks for unit IDs, row order,
serialization length, family, label, future effort, shard order, and censored
row removal. The audit SHALL report one per-unit audit row for each upstream
dataset unit that exists. It SHALL report empty per-unit rows with a named
missing-upstream failure when no upstream rows exist.

The artifact SHALL include `status`, `verdict_class`,
`upstream_artifact_receipt`, `independent_row_recomputation`,
`exact_receipt_replay_rows`, `split_and_lineage_audit`,
`shard_and_censoring_audit`, `feature_timing_audit`,
`shortcut_attack_matrix`, `branch_dataset_audited_ready_score`,
`per_unit_rows`, `aggregate_row_recomputation`, `gate_check_summary`,
`preconditions_checked`, `protected_files_unchanged`, `inference_substrate`,
`verifier_is_oracle`, `field_principles`, `field_provenance`, `random_seed`,
`duration_s`, `tests_run`, `reproducibility_checksum`, and
`honest_verdict`. `inference_substrate` SHALL be
`independent_branch_dataset_row_and_exact_receipt_replay_no_llm`.
`verifier_is_oracle` SHALL be true for label and receipt checks.
`verdict_class` SHALL be `null` only when readiness is valid. It SHALL be
`blocked` or `disqualified` for invalid inputs.

`branch_dataset_audited_ready_score` SHALL always be present and SHALL be
either `0.0` or `1.0`. It SHALL equal `1.0` only when exact receipts pass,
shards are complete, splits are sealed by base lineage, and all features are
available at decision time. Any failed check SHALL set the score to `0.0` and
name the failed check and observed value in `gate_check_summary`.

Field principles SHALL be:

- `status`: "A terminal audit status is required even when the upstream file is absent."
- `verdict_class`: "The class records null, blocked, disqualified, or partial readiness without claiming method value."
- `upstream_artifact_receipt`: "Path, existence, hash, status, and imported fields make the audit input explicit."
- `independent_row_recomputation`: "The audit must derive counts and metrics from rows rather than task aggregates."
- `exact_receipt_replay_rows`: "Per-row replay checks label and validity authority."
- `split_and_lineage_audit`: "Base-lineage separation prevents development and held leakage."
- `shard_and_censoring_audit`: "Manifest, resume, timeout, and terminal-count checks detect omitted hard units."
- `feature_timing_audit`: "Decision-time availability blocks future-effort and outcome leakage."
- `shortcut_attack_matrix`: "Attacks test identity, order, length, family, label, and censoring shortcuts."
- `branch_dataset_audited_ready_score`: "This exact closed field is the structured gate for Exp6513 and Exp6518."
- `per_unit_rows`: "One audit row per dataset unit makes readiness independently recheckable."
- `aggregate_row_recomputation`: "All audit summaries must derive from per-unit evidence."
- `gate_check_summary`: "Every score-0 or blocked result names the failed check and observed value."
- `preconditions_checked`: "Input, solver, and resource checks prevent invented audit results."
- `protected_files_unchanged`: "An audit cannot repair the source artifact or protected control files."
- `inference_substrate`: "Declaring independent exact replay with no LLM makes the evidence boundary explicit."
- `verifier_is_oracle`: "Oracle disclosure prevents exact dataset consistency from becoming a verifier-value claim."
- `field_principles`: "Reasons beside fields preserve the gate contract."
- `field_provenance`: "Paths, row IDs, reducers, solvers, and hashes make each audit result traceable."
- `random_seed`: "A fixed attack order makes the audit reproducible."
- `duration_s`: "Measured duration supports authenticity checks."
- `tests_run`: "Command receipts show which validation and E2E checks ran."
- `reproducibility_checksum`: "A content hash detects later changes to the audit decision."
- `honest_verdict`: "A complete_* or blocked_* prefix gives downstream gates a safe terminal state."

#### SCENARIO-BENCH-6512-MISSING-UPSTREAM: Missing Exp6511 Still Emits A Closed Gate

**Given** the Exp6511 dataset path is absent
**When** Exp6512 runs
**Then** it writes a complete terminal artifact with
`branch_dataset_audited_ready_score=0.0`, `verdict_class=blocked`,
`honest_verdict` beginning with `blocked_`, an upstream receipt with
`exists=false`, empty per-unit rows, and a gate-check row naming the missing
upstream path.

#### SCENARIO-BENCH-6512-ROW-REPLAY: Rows Own Counts And Exact Receipts

**Given** a present branch-counterfactual dataset
**When** Exp6512 audits it
**Then** row counts, labels, exact receipts, base hashes, aggregate metrics,
and terminal dispositions are recomputed from rows rather than imported
aggregate fields.

#### SCENARIO-BENCH-6512-SPLIT-LINEAGE: Base Lineage Is Sealed

**Given** train, development, and held rows
**When** Exp6512 rebuilds lineage sets
**Then** any base-lineage overlap, duplicate checkpoint, post-held repair,
asymmetric budget, or missing terminal disposition blocks readiness.

#### SCENARIO-BENCH-6512-SHARDS-CENSORING: Shards And Censoring Are Complete

**Given** a shard manifest, resume receipts, and censored rows
**When** Exp6512 recomputes shard and censoring state
**Then** missing shards, broken hash chains, missing resume receipts, timeout
omission, or mismatched terminal counts block readiness.

#### SCENARIO-BENCH-6512-LEAKAGE: Feature Timing And Shortcuts Fail Closed

**Given** decision-time features and adversarial shortcut probes
**When** Exp6512 audits feature timing
**Then** future effort, unit ID, row order, serialization length, family, label,
shard order, and censored-row-removal leakage block readiness.

## Implementation Status (REQ-BENCH-6512)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-BENCH-6512 | Planned (`python/carnot/experiment_6512_branch_dataset_independent_audit.py`, `results/experiment_6512_branch_dataset_independent_audit.json`) | Planned (`tests/python/test_experiment_6512_branch_dataset_independent_audit.py`) |

### REQ-BENCH-6513: V564 Terminal Handoff SHALL Preserve V563 Outcomes Without Retired Dependencies

Carnot SHALL provide Exp6513 at
`python/carnot/experiment_6513_v564_terminal_handoff_contract.py`.
The command
`.venv/bin/python -m carnot.experiment_6513_v564_terminal_handoff_contract --date 20260822`
SHALL write
`results/experiment_6513_v564_terminal_handoff_contract.json`.

Exp6513 SHALL build one bounded terminal handoff from immutable file content
and conductor rows. It SHALL read Exp6504, Exp6506, Exp6510, and Exp6512
directly by path and SHA-256 hash. It SHALL parse `ops/conductor-log.md` for
terminal rows. It SHALL record git status, input paths and hashes,
conductor-row identifiers, resources, exclusion state, and protected-file
hashes before it classifies readiness.

Exp6513 SHALL recompute Exp6504 raw, label, replay, split, held-cell,
stratum, leakage, aggregate, and checksum evidence from rows. It SHALL
recompute Exp6510 row counts, direct-input receipts, lineage decisions,
retired-dependency attacks, ready score, and non-positive verdict class from
Exp6510 rows and file content. It SHALL not trust aggregate-only fields from
either artifact.

Exp6513 SHALL preserve all historical determinations. It SHALL record the
three Exp6506 `artifact_not_updated_past_bootstrap` conductor rows, the three
Exp6510 `artifact_not_updated_past_bootstrap` conductor rows, the missing
Exp6511 dataset path, and the Exp6512 score-0 blocked audit. It SHALL
distinguish usable immutable content from eligible task dependencies. Exp6510
content MAY be read only as a direct path-and-hash input. Exp6510 SHALL NOT
be used as a structured dependency.

Exp6513 SHALL emit one row per historical task, immutable file, conductor
terminal event, allowed direct input, forbidden dependency, and retired-scope
attack. It SHALL attack renamed dependencies, indirect `requires` or
`gated_on` chains, stale paths, hash drift, positive exact-oracle framing, and
claims that terminal execution means scientific success.

The artifact SHALL include `status`, `honest_verdict`, `verdict_class`,
`prior_failure_receipts`, `historical_task_rows`,
`immutable_input_receipts`, `allowed_direct_input_rows`,
`forbidden_dependency_rows`, `retired_dependency_attack_matrix`,
`v564_handoff_ready_score`, `gate_check_summary`, `per_unit_rows`,
`aggregate_row_recomputation`, `preconditions_checked`,
`protected_files_unchanged`, `inference_substrate`, `verifier_is_oracle`,
`field_principles`, `field_provenance`, `random_seed`, `duration_s`,
`tests_run`, and `reproducibility_checksum`. `inference_substrate` SHALL be
`bounded_historical_artifact_and_conductor_replay_no_llm`.
`verifier_is_oracle` SHALL be true only for row, hash, and terminal-state
checks. `verdict_class` SHALL be `null`, `partial`, or `blocked`; it SHALL
NOT be `positive`.

`v564_handoff_ready_score` SHALL equal `1.0` only when Exp6504 row replay
passes, Exp6510 rows support its usable-content status, every historical
failure or block is preserved, Exp6511 is confirmed missing, Exp6512 is
confirmed as a blocked score-0 audit, protected files remain unchanged, every
forbidden dependency attack fails closed, and no planned structured dependency
or requires chain names Exp6506, Exp6507, Exp6508, Exp6509, Exp6510, or
Exp6511. A blocked verdict SHALL populate `gate_check_summary` with the
failed check, expected value, observed value, and source path.

Field principles SHALL be:

- `status`: "A terminal state distinguishes the handoff from a bootstrap or partial write."
- `honest_verdict`: "The verdict preserves terminal history without claiming scientific success."
- `verdict_class`: "A null or partial class prevents terminal governance from becoming a performance claim."
- `prior_failure_receipts`: "Receipts preserve Exp6506 and Exp6510 bootstrap failures, missing Exp6511, and the Exp6512 block without reactivating them."
- `historical_task_rows`: "One row per prior task separates task outcome from usable file content."
- `immutable_input_receipts`: "Path and hash receipts make every direct read auditable."
- `allowed_direct_input_rows`: "Direct input rows authorize content reads without structured dependencies."
- `forbidden_dependency_rows`: "Forbidden rows prove retired or missing IDs cannot gate V564 work."
- `retired_dependency_attack_matrix`: "Attacks catch aliases, indirect requires, stale paths, drift, positive framing, and terminal-success confusion."
- `v564_handoff_ready_score`: "The score opens only when all historical determinations are preserved and no retired structured dependency is planned."
- `gate_check_summary`: "Each gate records expected and observed values for replay."
- `per_unit_rows`: "Unit rows expose every task, file, event, input, dependency, and attack row."
- `aggregate_row_recomputation`: "The aggregate is recomputed from rows instead of imported totals."
- `preconditions_checked`: "Preconditions record git state, resources, solver, roadmap, exclusion, and required paths."
- `protected_files_unchanged`: "Historical files and conductor code must remain unchanged during the handoff."
- `inference_substrate`: "The declaration keeps the handoff in artifact and conductor replay with no LLM."
- `verifier_is_oracle`: "Oracle disclosure limits authority to row, hash, and terminal-state checks."
- `field_principles`: "Reasons beside fields help future tasks preserve the contract."
- `field_provenance`: "Provenance records path, hash, reducer, and constant sources for each field."
- `random_seed`: "A fixed attack order makes replay deterministic."
- `duration_s`: "Measured wall time supports authenticity checks."
- `tests_run`: "Command receipts show which verification actually ran."
- `reproducibility_checksum`: "A content hash detects drift in inputs, rows, and decisions."

#### SCENARIO-BENCH-6513-DIRECT-IMMUTABLE: Inputs Are Path-And-Hash Reads

**Given** the checked-in Exp6504, Exp6506, Exp6510, and Exp6512 artifacts
**When** Exp6513 builds immutable input receipts
**Then** every source records path, existence, SHA-256, imported pointers, and
`read_mode=direct_path_and_hash`, and no allowed direct input counts as a
structured dependency.

#### SCENARIO-BENCH-6513-ROW-REPLAY: Readiness Comes From Rows

**Given** Exp6504 and Exp6510 artifacts with aggregate fields
**When** Exp6513 recomputes readiness
**Then** Exp6504 counts and Exp6510 content status derive from per-row
evidence, lineage rows, attack rows, hashes, and non-positive classes rather
than aggregate-only fields.

#### SCENARIO-BENCH-6513-TERMINAL-HISTORY: V563 Terminal Outcomes Are Preserved

**Given** the conductor log, missing Exp6511 path, and Exp6512 artifact
**When** Exp6513 emits prior-failure and historical-task rows
**Then** it records three Exp6506 bootstrap failures, three Exp6510 bootstrap
failures, the missing Exp6511 deliverable, and the Exp6512 blocked score-0
audit without reclassifying terminal as scientifically successful.

#### SCENARIO-BENCH-6513-RETIRED-ISOLATION: Retired IDs Cannot Gate V564

**Given** planned V564 roadmap tasks and retired V562/V563 identifiers
**When** Exp6513 checks `requires` and `gated_on` chains
**Then** no allowed structured dependency names Exp6506, Exp6507, Exp6508,
Exp6509, Exp6510, or Exp6511, while direct path-and-hash reads of historical
files remain allowed.

#### SCENARIO-BENCH-6513-ATTACKS: Shortcut Dependencies Fail Closed

**Given** renamed dependencies, indirect requires chains, stale paths, hash
drift, positive exact-oracle framing, and terminal-success framing
**When** Exp6513 evaluates retired-dependency attacks
**Then** each attack fails closed before `v564_handoff_ready_score` can equal
`1.0`.

#### SCENARIO-BENCH-6513-ATOMIC-FINAL: The Handoff Is One Complete Artifact

**Given** all handoff gates pass
**When** Exp6513 writes its deliverable
**Then** the artifact is written atomically with all required fields, a stable
checksum, complete provenance, unchanged protected files, and an honest verdict
with a `complete_` or `blocked_` prefix.

## Implementation Status (REQ-BENCH-6513)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-BENCH-6513 | Planned (`python/carnot/experiment_6513_v564_terminal_handoff_contract.py`, `results/experiment_6513_v564_terminal_handoff_contract.json`) | Planned (`tests/python/test_experiment_6513_v564_terminal_handoff_contract.py`) |

### REQ-BENCH-6514: Atomic Shard Artifact Transaction SHALL Separate Resumable Rows From Terminal Artifacts

Carnot SHALL provide a reusable helper at
`python/carnot/atomic_shard_transaction.py` and an Exp6514 contract at
`python/carnot/experiment_6514_atomic_shard_artifact_transaction.py`.
The command
`.venv/bin/python -m carnot.experiment_6514_atomic_shard_artifact_transaction --date 20260822`
SHALL write
`results/experiment_6514_atomic_shard_artifact_transaction.json`.

The transaction helper SHALL support row-producing experiments that generate
long data streams. It SHALL write resumable unit rows as content-addressed
shards outside the conductor-visible final artifact. It SHALL record a planned
unit journal before terminal rows are finalized. It SHALL compute SHA-256
hashes for shard bytes and journal records. It SHALL verify a shard before
reuse, quarantine corrupt shard bytes, and reject a duplicate unit ID if a
verified terminal record already binds that ID to different content.

The helper SHALL resume only from verified journal records and verified shard
bytes. A planned unit SHALL count as terminal only when the journal has a
valid terminal disposition and its referenced content-addressed shard verifies.
Finalization SHALL fail closed while any planned unit is missing a terminal
disposition. Finalization SHALL write a complete temporary JSON artifact,
fsync the file, fsync the parent directory where supported, then atomically
replace the final path. A successful final path SHALL never contain a
bootstrap, running, partial, or incomplete status.

The helper SHALL refuse concurrent writers with a lock file. It SHALL recover
from a stale lock when the recorded process is gone and the lock age exceeds
the configured stale threshold. It SHALL check available disk space before
large writes and raise a closed insufficient-disk error before mutating the
final artifact when the precondition fails.

Exp6514 SHALL exercise crash injection before shard write, after shard write,
during journal update, before final replace, and after final replace. It SHALL
prove verified resume after each supported crash point. It SHALL test corrupt
shard quarantine, stale lock recovery, concurrent-writer refusal, missing unit
refusal, duplicate unit refusal, and insufficient-disk refusal.

The artifact SHALL include `status`, `honest_verdict`, `verdict_class`,
`transaction_schema`, `filesystem_capability_receipt`,
`crash_injection_rows`, `recovery_rows`, `shard_integrity_rows`,
`concurrency_attack_rows`, `terminal_write_receipt`,
`atomic_artifact_contract_ready_score`, `gate_check_summary`,
`per_unit_rows`, `aggregate_row_recomputation`, `preconditions_checked`,
`protected_files_unchanged`, `inference_substrate`, `verifier_is_oracle`,
`field_principles`, `field_provenance`, `random_seed`, `duration_s`,
`tests_run`, and `reproducibility_checksum`. `inference_substrate` SHALL be
`local_filesystem_transaction_and_crash_injection_no_llm`.
`verifier_is_oracle` SHALL be true only for local transaction invariants.
`verdict_class` SHALL be `null`, `partial`, `blocked`, or `disqualified`;
it SHALL NOT be `positive`.

`atomic_artifact_contract_ready_score` SHALL equal `1.0` only when every crash
injection row, recovery row, shard integrity row, concurrency attack row, and
terminal write gate passes on the supported local filesystem. A blocked or
partial result SHALL preserve a closed terminal failure artifact with
diagnostics in `gate_check_summary`.

Field principles SHALL be:

- `status`: "A terminal status proves the conductor sees a closed artifact, not a bootstrap."
- `honest_verdict`: "The verdict states whether the filesystem transaction contract is complete or bounded."
- `verdict_class`: "The class is an infrastructure result and must never be positive."
- `transaction_schema`: "The schema names the reusable helper and journal format."
- `filesystem_capability_receipt`: "Filesystem, permissions, disk, process, and fsync receipts bound the local preconditions."
- `crash_injection_rows`: "Each injected crash records where the process stopped and what survived."
- `recovery_rows`: "Recovery rows prove resume uses only verified shards and journal records."
- `shard_integrity_rows`: "Shard rows bind unit IDs to content hashes and quarantine corrupt bytes."
- `concurrency_attack_rows`: "Concurrency rows prove live locks refuse writers and stale locks recover."
- `terminal_write_receipt`: "The receipt proves complete-temp fsync and atomic replacement reached the final path."
- `atomic_artifact_contract_ready_score`: "The score opens only when every local transaction invariant passes."
- `gate_check_summary`: "Each failed gate records the expected and observed value."
- `per_unit_rows`: "Per-unit rows expose planned, terminal, crash, recovery, integrity, and concurrency evidence."
- `aggregate_row_recomputation`: "The aggregate recomputes readiness from rows."
- `preconditions_checked`: "Preconditions record git state, paths, disk, process, and protected hashes."
- `protected_files_unchanged`: "The conductor and historical inputs must remain byte-identical."
- `inference_substrate`: "The declaration prevents a filesystem contract from being read as model inference."
- `verifier_is_oracle`: "Oracle scope is limited to transaction invariants."
- `field_principles`: "Principles preserve why each artifact field exists."
- `field_provenance`: "Provenance maps each field to helper rows, local receipts, or deterministic reducers."
- `random_seed`: "A fixed seed makes crash and attack ordering reproducible."
- `duration_s`: "Measured duration supports authenticity checks."
- `tests_run`: "Command receipts show which validation actually ran."
- `reproducibility_checksum`: "A checksum detects later drift in rows, gates, or the terminal artifact."

#### SCENARIO-BENCH-6514-SHARD-IDENTITY: Unit Shards Are Content Addressed

**Given** planned units and terminal payloads
**When** the transaction writes shards
**Then** shard filenames, shard receipts, and terminal journal records use
SHA-256 content hashes, verified identical rewrites are idempotent, and a
duplicate unit ID with different verified content is rejected.

#### SCENARIO-BENCH-6514-PLANNED-TERMINAL: Planned Units Must All Close

**Given** a planned-unit journal with at least one missing terminal unit
**When** the producer tries to finalize the artifact
**Then** finalization fails closed before replacing the final path and names
the missing unit.

#### SCENARIO-BENCH-6514-RESUME-CRASHES: Crash Points Resume Safely

**Given** crash injection before shard write, after shard write, during journal
update, before replace, and after replace
**When** the transaction resumes
**Then** verified shards are reused, orphan shards do not count as terminal
without a journal record, and an after-replace crash leaves a complete
terminal final artifact.

#### SCENARIO-BENCH-6514-CORRUPT-QUARANTINE: Corrupt Shards Do Not Resume

**Given** a terminal journal record references a shard whose bytes no longer
match its hash
**When** the transaction resumes
**Then** the corrupt bytes move to quarantine, the unit is no longer terminal,
and the producer must rewrite a verified shard before finalization.

#### SCENARIO-BENCH-6514-ATOMIC-REPLACE: Final Artifact Replaces Atomically

**Given** all planned units have terminal dispositions
**When** finalization writes the terminal JSON
**Then** it fsyncs the complete temporary file, fsyncs the directory where
supported, atomically replaces the final path, and rejects bootstrap or running
success payloads.

#### SCENARIO-BENCH-6514-CONCURRENCY: Live Writers Are Refused

**Given** one process holds the transaction lock
**When** another writer opens the same transaction
**Then** the second writer is refused, while a stale lock with a dead process
is removed and recorded as recovered.

#### SCENARIO-BENCH-6514-CLOSED-FAILURE: Failures Write Or Preserve Closed Artifacts

**Given** disk, corruption, duplicate, or missing-unit preconditions fail
**When** the producer closes the run
**Then** it writes or preserves a terminal blocked or disqualified artifact
with diagnostics and never leaves a running artifact at the final path.

## Implementation Status (REQ-BENCH-6514)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-BENCH-6514 | Planned (`python/carnot/atomic_shard_transaction.py`, `python/carnot/experiment_6514_atomic_shard_artifact_transaction.py`, `results/experiment_6514_atomic_shard_artifact_transaction.json`) | Planned (`tests/python/test_atomic_shard_transaction.py`, `tests/python/test_experiment_6514_atomic_shard_artifact_transaction.py`) |

### REQ-BENCH-6516: Exact Branch Pilot SHALL Be Sealed Before Guidance Claims

Carnot SHALL provide Exp6516 at
`python/carnot/experiment_6516_exact_branch_pilot_dataset_v3.py`.
The command
`.venv/bin/python -m carnot.experiment_6516_exact_branch_pilot_dataset_v3 --date 20260823`
SHALL write
`results/experiment_6516_exact_branch_pilot_dataset_v3.json`.

Exp6516 SHALL evaluate the two V564 gates before pilot construction. It SHALL
read Exp6514 and Exp6515 by direct path and hash. It SHALL record expected and
observed gate values, artifact hashes, solver versions, CPU, RAM, disk,
output-space bounds, and protected-file hashes. Both gates SHALL equal `1.0`
before a complete readiness score can be reported.

Exp6516 SHALL read Exp6504 and Exp6510 by direct immutable path and hash. It
SHALL not use Exp6506, Exp6507, Exp6508, Exp6509, Exp6510, or Exp6511 as a
structured dependency. It SHALL record prior failure receipts from the direct
input artifacts. It SHALL use Exp6510 only as a historical receipt that
authorizes direct file reads.

Exp6516 SHALL freeze a bounded pilot before replay outcomes exist. The pilot
SHALL cover at least three procedural constraint families, two difficulty
scales, multiple seeds, and sealed train, development, and held base lineages.
It SHALL precommit base lineages, checkpoints, candidate values, equal exact
budgets, cell floors, and terminal dispositions. Checkpoint and feature
selection SHALL use only decision-time structural data. It SHALL not use unit
ID, label, held outcome, future effort, row order, serialization length,
family label, base label, or post-replay effort as a feature.

For every checkpoint, Exp6516 SHALL enumerate every eligible Boolean value. It
SHALL replay each value under the same exact budget. Each replay SHALL record
conflicts, propagations, decisions, restarts, model or proof receipt, timeout,
censoring, terminal disposition, exact answer equality across exact backends,
and a stable row hash.

Exp6516 SHALL write content-addressed row shards and a planned-unit journal
through the Exp6514 transaction helper. Resume SHALL accept only verified
journal records and verified shard bytes. Finalization SHALL occur only after
every planned unit has a terminal disposition. The final artifact SHALL be
written through the Exp6514 atomic finalizer.

Exp6516 SHALL attack row order, serialization length, family labels, base
labels, outcome-derived effort, duplicate checkpoints, split leakage,
asymmetric budgets, corrupt resume, and omitted hard rows. A false accept in
any attack SHALL set `branch_pilot_dataset_ready_score=0.0` and SHALL use
`verdict_class="disqualified"`.

The artifact SHALL include `status`, `honest_verdict`, `verdict_class`,
`upstream_gate_receipts`, `prior_failure_receipts`, `direct_input_receipts`,
`pilot_commitment`, `checkpoint_contract`, `structural_feature_schema`,
`branch_counterfactual_rows`, `exact_solver_receipts`, `split_commitment`,
`shard_manifest`, `planned_and_terminal_unit_counts`,
`censoring_and_budget_rows`, `leakage_attack_matrix`,
`branch_pilot_dataset_ready_score`, `gate_check_summary`, `per_unit_rows`,
`aggregate_row_recomputation`, `preconditions_checked`,
`protected_files_unchanged`, `inference_substrate`, `verifier_is_oracle`,
`field_principles`, `field_provenance`, `random_seed`, `duration_s`,
`tests_run`, and `reproducibility_checksum`. `inference_substrate` SHALL be
`procedural_exact_branch_replay_with_transactional_shards_no_llm`.
`verifier_is_oracle` SHALL be true only for exact labels and validity checks.
`verdict_class` SHALL be `null` for complete data readiness, `partial` for
bounded usable data below a claim gate, `blocked` for failed preconditions, or
`disqualified` for leakage or false terminal accounting. It SHALL NOT be
`positive`.

`branch_pilot_dataset_ready_score` SHALL equal `1.0` only when every admitted
row has an exact receipt, every planned unit is terminal, cell floors hold,
splits are sealed, no leakage attack false accepts, protected files remain
unchanged, and the final transaction verifies.

Field principles SHALL be:

- `status`: "Records the terminal pilot dataset state."
- `honest_verdict`: "The verdict states whether the bounded pilot is ready without claiming learned guidance value."
- `verdict_class`: "Null means complete data readiness. Partial, blocked, and disqualified preserve limits and failures."
- `upstream_gate_receipts`: "The two V564 gates are checked by path, hash, expected value, and observed value."
- `prior_failure_receipts`: "Historical bootstrap and skipped-dataset failures stay visible without reactivating retired roots."
- `direct_input_receipts`: "Exp6504 and Exp6510 are read as direct immutable inputs with hashes."
- `pilot_commitment`: "Families, scales, seeds, splits, budgets, floors, and candidate rules are fixed before replay."
- `checkpoint_contract`: "Checkpoint selection is deterministic and uses only decision-time structural data."
- `structural_feature_schema`: "Allowed features exclude identity, labels, row order, length, family labels, and future effort."
- `branch_counterfactual_rows`: "One row per base checkpoint and eligible value records the exact replay result."
- `exact_solver_receipts`: "Solver receipts expose model, proof, effort, timeout, censoring, and backend equality."
- `split_commitment`: "Train, development, and held base lineages are sealed and disjoint."
- `shard_manifest`: "Content-addressed shards, journal records, resume checks, and finalization state are auditable."
- `planned_and_terminal_unit_counts`: "Planned and terminal counts prevent omitted or duplicate rows."
- `censoring_and_budget_rows`: "Equal budgets and explicit timeout or censoring rows prevent asymmetric effort."
- `leakage_attack_matrix`: "Shortcut attacks detect row order, length, family, label, effort, checkpoint, split, budget, resume, and omission leaks."
- `branch_pilot_dataset_ready_score`: "The score opens only when rows, receipts, splits, floors, attacks, and transaction checks pass."
- `gate_check_summary`: "Each failed gate records expected and observed values."
- `per_unit_rows`: "Per-unit rows expose branch rows and audit rows for independent replay."
- `aggregate_row_recomputation`: "Aggregates are recomputed from rows, not imported totals."
- `preconditions_checked`: "Preconditions record direct files, gates, solvers, resources, output bounds, and protected hashes."
- `protected_files_unchanged`: "Historical inputs and the conductor stay byte-identical during the run."
- `inference_substrate`: "The declaration keeps the pilot on procedural exact replay with no LLM."
- `verifier_is_oracle`: "Oracle authority is limited to exact labels and validity checks."
- `field_principles`: "Principles explain why each artifact field exists."
- `field_provenance`: "Provenance maps fields to specs, direct inputs, solvers, rows, shards, and tests."
- `random_seed`: "Fixed seeds make pilot selection and attack order reproducible."
- `duration_s`: "Measured duration supports authenticity checks."
- `tests_run`: "Command receipts show which verification actually ran."
- `reproducibility_checksum`: "A content hash detects drift in gates, rows, shards, and decisions."

#### SCENARIO-BENCH-6516-DIRECT-IMMUTABLE: Inputs Are Path-And-Hash Reads

**Given** checked-in Exp6504, Exp6510, Exp6514, and Exp6515 artifacts
**When** Exp6516 evaluates gates and direct inputs
**Then** it records each path, hash, expected value, observed value, source
pointer, and direct-read mode without adding a retired structured dependency.

#### SCENARIO-BENCH-6516-CHECKPOINTS: Checkpoints Are Deterministic

**Given** selected base rows before replay outcomes exist
**When** Exp6516 freezes checkpoints
**Then** the selected variable and candidate values derive only from
decision-time structural features and every feature is declared in the schema.

#### SCENARIO-BENCH-6516-CANDIDATES: Every Eligible Value Is Replayed

**Given** a Boolean checkpoint
**When** Exp6516 builds branch units
**Then** both `false` and `true` values are planned and replayed under the
same exact budget for every selected base row.

#### SCENARIO-BENCH-6516-EXACT-REPLAY: Exact Receipts Own Outcomes

**Given** each forced branch formula
**When** Exp6516 solves it
**Then** exhaustive and Z3 backends agree on SAT or UNSAT, terminal model or
proof receipts are valid, and timeout or censoring is explicit.

#### SCENARIO-BENCH-6516-SPLIT-SEALING: Base Lineages Stay Disjoint

**Given** train, development, and held pilot rows
**When** Exp6516 recomputes split state
**Then** no base lineage crosses splits, every planned split/family/scale cell
meets its floor, and held rows are not repaired after held outcomes are read.

#### SCENARIO-BENCH-6516-BOUNDED-SHARDS: Shards And Censoring Are Complete

**Given** all planned branch units
**When** Exp6516 writes row shards
**Then** the shard manifest records verified resume receipts, terminal counts,
hashes, censoring counts, and no omitted hard row.

#### SCENARIO-BENCH-6516-RESUME-ATOMIC: Finalization Uses Exp6514

**Given** a verified transaction resume state
**When** Exp6516 finalizes the dataset
**Then** the final JSON is atomically replaced only after every planned unit is
terminal and the artifact checksum validates.

#### SCENARIO-BENCH-6516-ATTACKS: Shortcut And Accounting Attacks Fail Closed

**Given** row order, length, family, label, future-effort, duplicate
checkpoint, split-leakage, asymmetric-budget, corrupt-resume, and
omitted-hard-row attacks
**When** Exp6516 evaluates readiness
**Then** every attack fails closed before
`branch_pilot_dataset_ready_score` can equal `1.0`.

#### SCENARIO-BENCH-6516-READY: Readiness Is Row-Derived

**Given** all rows, receipts, shards, attacks, protected hashes, and tests
**When** Exp6516 validates the artifact
**Then** every required field has a principle and provenance entry, the
checksum matches, `verdict_class` is `null`, and the terminal verdict starts
with `complete_` or `blocked_`.

## Implementation Status (REQ-BENCH-6516)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-BENCH-6516 | Planned (`python/carnot/experiment_6516_exact_branch_pilot_dataset_v3.py`, `results/experiment_6516_exact_branch_pilot_dataset_v3.json`) | Planned (`tests/python/test_experiment_6516_exact_branch_pilot_dataset_v3.py`) |

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

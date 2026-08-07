# Code Verification Capability Specification

**Capability:** code-verification
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-12, FR-11

## Overview

Defines how Carnot verifies Python code correctness using Energy-Based Models. Python type constraints, test-pass constraints, and exception-free constraints are encoded as energy terms. A learned verifier (Gibbs model trained via NCE) complements handcoded energy checks. The autoresearch pipeline improves the verifier autonomously.

This capability connects all three Carnot pillars: EBM constraints (verifiable reasoning), learned verification (inference), and self-improvement (autoresearch).

## Requirements

### REQ-CODE-001: Python Type Constraint Encoding

The system shall encode Python function correctness constraints as energy terms:
- **ReturnTypeConstraint**: Energy = fraction of test inputs where the return type does not match the annotation. Zero energy means all return types match.
- **NoExceptionConstraint**: Energy = fraction of test inputs that raise exceptions. Zero energy means no exceptions.
- **TestPassConstraint**: Energy = fraction of test cases producing wrong output. Zero energy means all tests pass.
- Each constraint executes the actual Python code via safe_exec_function with timeout protection.

### REQ-CODE-002: Code-to-Embedding

The system shall convert Python source code to a fixed-size embedding vector:
- Uses stdlib `tokenize` to extract tokens from source
- Maps token identities (token type + token string) to indices via
  `hash(token_identity) % vocab_size`
- Returns a frequency vector of shape `(vocab_size,)` with float32 dtype
- Deterministic: same code always produces the same embedding
- Different code produces different embeddings (with high probability)

### REQ-CODE-003: Training Data Generation

The system shall generate labeled training data for the code verifier:
- Correct samples: known-good Python function templates
- Buggy samples: mutations of correct templates (removed returns, swapped operators, wrong values)
- Returns batches of `(correct_embeddings, buggy_embeddings)` for NCE training
- Deterministic given the same seed

### REQ-CODE-004: Learned Code Verifier

The system shall train a Gibbs model via NCE to distinguish correct from buggy Python code:
- Input: code embedding vectors from REQ-CODE-002
- Training: NCE loss with correct code as data and buggy code as noise
- Output: a GibbsModel that assigns lower energy to correct code
- Verification pipeline combines handcoded energy (REQ-CODE-001) with learned energy

### REQ-CODE-005: Self-Improving Code Verification

The system shall improve code verification via the autoresearch pipeline:
- Benchmark function evaluates verifier accuracy on held-out test data
- Baseline record captures initial verifier performance
- Hypothesis templates explore model width, depth, epoch count, and data volume
- The autoresearch loop (REQ-AUTO-003) orchestrates improvement

### REQ-CODE-006: Property-Based Code Verification

The system shall support property-based verification for Python functions:
- Property checks run execution-based randomized probes against candidate code
- Properties may be caller-provided or selected from built-in generators/checks
- Failures report the property name, sampled input, and observed error or value
- Property verification returns an energy-like failure rate and LLM-readable
  repair feedback

### REQ-CODE-007: Prompt-Derived Property Verification

The system shall derive additional code properties and invariants from the
available HumanEval-style problem description:
- Inputs include the prompt intent, function signature, docstrings/examples,
  and existing official tests
- The derivation path is deterministic and lightweight enough for live
  benchmark runs
- The verifier may synthesize fixed example regressions plus heuristic
  invariants such as determinism, non-mutation, ordering/permutation, reverse,
  or uniqueness when those are strongly implied by the prompt or examples
- The verifier records each derived property's source so repair feedback can
  explain why the property exists

### REQ-CODE-008: Structured Property Repair Feedback

The system shall turn prompt-derived property failures into structured repair
feedback compatible with the existing verify/repair flow:
- Each failure is convertible to a pipeline-compatible constraint/violation
  record
- The current execution-based code path augments static and dynamic
  instrumentation feedback with derived-property findings instead of replacing
  them
- Repair prompts can include the property name, source evidence, failing input,
  and actual/error outcome
- The added verifier remains additive and bounded so existing execution-based
  checks continue to run unchanged

### REQ-CODE-009: Hypothesis-Backed Property-Based Code Verification

The system shall support a Hypothesis-backed property verifier for Python code
candidates:
- Inputs include the generated code, HumanEval-style prompt, entry point, and
  official test harness
- The verifier derives bounded property checks from the prompt intent,
  signature, examples, and official tests
- Randomized input generation uses the `hypothesis` library with deterministic
  settings suitable for unit tests and live verification loops
- Generated checks cover type expectations, stability or determinism, edge
  cases such as empty inputs or negatives when the signature admits them, and
  prompt-intent invariants such as sorting, reversing, or non-mutation when the
  evidence is strong
- Each violating property records a concrete counterexample input plus the
  observed incorrect value or error

### REQ-CODE-010: VerifyRepairPipeline Generated-Code Verification Path

The system shall expose the Hypothesis-backed verifier through
`VerifyRepairPipeline` as an additive generated-code verification path:
- The pipeline accepts a code candidate plus the HumanEval-style prompt
  context needed to verify it
- Static code constraints and Hypothesis-backed property failures can both flow
  into one `VerificationResult`
- Property failures are returned as pipeline-compatible `ConstraintResult`
  records so existing repair feedback formatters can reuse them unchanged
- Existing `verify()` behavior for non-code responses remains backward
  compatible

### REQ-CODE-011: Deterministic HumanEval-Style PBT Comparison

The repository shall provide a deterministic five-problem HumanEval-style
comparison that validates the Hypothesis-backed verifier against the existing
execution-only path:
- The comparison uses the same prompt, entry point, and official tests for both
  execution-only and property-based verification
- The five-problem slice includes prompts where the official tests are known to
  under-specify the intended behavior
- The comparison demonstrates whether property-based verification surfaces extra
  wrong solutions beyond basic execution while keeping correct solutions clean
- The comparison stays deterministic in the checked-in test suite

### REQ-CODE-012: Full HumanEval PBT Benchmark

The repository shall provide a full official HumanEval benchmark workflow for
the Hypothesis-backed verifier:
- The workflow runs all 164 official `openai_humaneval` test problems with live
  GPU inference on `google/gemma-4-E4B-it`
- Each case records the baseline generation, official `check()` harness result,
  additive static or runtime findings, Hypothesis-backed PBT findings, and up
  to 3 repair attempts
- The benchmark keeps the same prompt, entry point, and official tests across
  baseline, PBT verification, and repair for a given case
- The primary pass@1 metrics remain the official HumanEval harness outcomes so
  the result stays comparable to published HumanEval baselines

### REQ-CODE-013: Resume-Aware Full-Benchmark Checkpointing

The full HumanEval PBT benchmark shall support long-running checkpointed
execution:
- The workflow checkpoints every 10 completed problems and writes a final
  checkpoint or artifact save at the end of the run
- Resume logic only reuses a checkpoint when the expected ordered cohort still
  matches
- Resumed runs skip completed cases, execute only the pending cases, and emit
  final ordered results in the original cohort order
- The persisted checkpoint stores enough metadata to report progress honestly
  and resume without silently dropping completed work

### REQ-CODE-014: Publishable Summary And Baseline Comparison

The full HumanEval PBT benchmark shall emit a publishable summary payload:
- The artifact reports baseline and verify-repair pass@1, 95% bootstrap
  confidence intervals, and the paired improvement delta confidence interval
- The artifact includes compare-to-published baseline entries for
  `Gemma4-E4B-it`, including the source title, URL, reported metric, and the
  local delta versus each published number
- The artifact includes a concise technical-report-ready summary that states the
  cohort size, local baseline, verify-repair outcome, improvement delta, repair
  yield, and relation to the published baseline(s)

### REQ-CODE-015: Seeded Qwen PBT Benchmark With Gemma Cohort Comparison

The repository shall provide a deterministic 30-problem HumanEval benchmark
workflow for `Qwen/Qwen3.5-0.8B` that reuses the checked-in Exp 208 Gemma
cohort:
- The workflow loads the exact ordered dataset indices and task ids from
  `results/experiment_208_results.json` instead of resampling HumanEval
- Each case runs live GPU baseline generation, official `check()` execution,
  additive Hypothesis-backed PBT verification, and up to 3 repair attempts
- The final artifact reports baseline and verify-repair pass@1 for the Qwen run
  and an explicit comparison against the Exp 208 `Gemma4-E4B-it` artifact on
  the same 30-problem cohort
- The comparison records per-metric deltas and an honest methodology note when
  the historical Gemma artifact used an earlier non-Hypothesis verify-repair
  path

### REQ-CODE-016: Verification Trace Analysis From Checked-In Benchmark Artifacts

The system shall extract learnable code-verification patterns from checked-in
benchmark artifacts such as Exp 225 and Exp 226:
- Inputs may mix artifacts with full per-problem verification traces and
  metadata-only artifacts that contain no usable trace history
- Trace ingestion normalizes each case into a stable structure carrying the
  source experiment, task id, baseline verifier outcomes, per-iteration repair
  history, PBT derived properties, PBT failure records, and final verify-repair
  outcome
- Artifacts without usable per-problem traces are skipped without raising, but
  are counted in the analysis summary so the caller can see what was ignored
- The analysis summary reports enough detail to answer which properties catch
  the most bugs, which repair families succeed, and which problem families gain
  the most from additive verification

### REQ-CODE-017: Rank PBT Properties By Observed Effectiveness

The system shall rank Hypothesis-backed PBT properties by observed
effectiveness on accumulated verification traces:
- Property statistics track raw failure support, number of affected cases,
  official-test misses caught beyond the harness, and repaired cases where the
  property was present in the failing baseline
- The ranking stays deterministic for the same trace corpus
- The ranking output includes enough structured fields for downstream reporting
  and later cumulative updates
- Ranking may be recomputed on prefixes or merged corpora so later traces can
  refine earlier property priorities

### REQ-CODE-018: Learn Repair Strategies From Verification Histories

The system shall learn which repair strategies work for which verification
error families:
- The learner infers coarse strategy labels from trace evidence, including
  syntax recovery, exception hardening, return-type alignment,
  immutability-preserving fixes, ordering fixes, and harness-only semantic
  fixes
- For each strategy and error family, the learner tracks attempts, successful
  repairs, partial recoveries, and supporting case ids
- The learner exposes ranked recommendations for a new failing trace based on
  accumulated evidence
- The module includes a learning-curve style demonstration that shows how the
  accumulated trace history changes property or repair recommendations over
  time

### REQ-CODE-019: Standalone Python Code Verification API

The system shall expose a standalone Python API for packaged code
verification:
- Users can import `verify_code` directly from `carnot.pipeline`
- The API accepts Python source code, an entry-point function name, and
  optional prompt and official-test context
- When prompt context is omitted, the API falls back to the provided source
  code so signature-derived PBT checks still run
- The API returns a pipeline-compatible `VerificationResult` whose certificate
  includes additive PBT summary metadata

### REQ-CODE-020: User-Facing CLI Command For Packaged Code Verification

The system shall expose packaged code verification through the CLI:
- The `carnot` CLI includes a `verify-code` subcommand
- The command accepts a Python source file plus `--func` to select the entry
  point
- The command can enable Hypothesis-backed verification with `--pbt`
- The command reports pass/fail, constraint counts, and PBT findings in
  terminal-friendly output

### REQ-CODE-021: MCP Tool For Hypothesis-Backed Code Verification

The system shall expose packaged code verification through MCP:
- The production MCP server registers a `verify_code_with_pbt` tool
- The tool accepts source code, entry point, and optional prompt or
  official-test context
- The tool returns structured verification results, repair feedback, and the
  additive PBT summary without crashing on bad input
- The server health response lists the new tool so MCP clients can discover it

### REQ-CODE-022: End-User Verification Workflow Documentation

The repository shall document the packaged code-verification workflow for end
users:
- Docs include runnable examples for the CLI command, the MCP tool, and the
  Python API
- Docs include a generate-verify-repair example showing how a buggy candidate
  is verified and then re-checked after repair
- The examples use the packaged surfaces added by REQ-CODE-019,
  REQ-CODE-020, and REQ-CODE-021 rather than research-only scripts

### REQ-CODE-023: Deterministic Explicit Code Spec Corpus From Benchmark Traces

The repository shall provide a checked-in workflow that converts checked-in
HumanEval-style benchmark cases into explicit code specs:
- The corpus is generated from checked-in benchmark artifacts such as Exp 226
  and Exp 227 rather than sampling live model outputs at generation time
- Each row is keyed deterministically by task identity and carries explicit
  `preconditions`, `postconditions`, `invariants`, `mutation_constraints`, and
  `oracle_hints`
- Derivation combines prompt intent, function signature, docstring examples,
  official tests, and trace-backed verification evidence from the source
  artifacts
- When the same task appears in multiple source artifacts, the workflow merges
  the task into one row while preserving per-trace provenance and observed
  failure or repair evidence

### REQ-CODE-024: Compact Schema And Provenance Summary For Code Specs

The repository shall emit a compact, machine-readable schema and summary for
the explicit code spec corpus:
- Each corpus row includes the task id, entry point, grouped spec families,
  normalized trace evidence, and provenance links back to the source benchmark
  artifact(s)
- The field layout and row ordering are deterministic for the same checked-in
  inputs so later verifier code can consume the corpus directly
- The workflow writes a companion summary artifact with counts by spec family
  and by source trace, plus aggregate counts for observed failures,
  official-test misses, and repaired traces
- The summary records the output paths and source artifact set honestly

### REQ-CODE-025: Spec-Aware Code Spec Ingestion And Matching

The system shall ingest the checked-in explicit code-spec corpus for
verification-time use:
- The verifier can load deterministic rows from
  `data/research/code_spec_corpus_236.jsonl` or a caller-provided corpus path
- Lookup may use the HumanEval-style task id directly and shall fall back to a
  deterministic entry-point or signature match when the task id is absent
- The loaded row preserves grouped clause families, oracle hints, trace
  summary, provenance links, and fixed artifact metadata such as the checked-in
  run date
- Missing corpus matches do not crash the verifier; they degrade to a
  structured “no explicit spec matched” result

### REQ-CODE-026: Aggregated Official-Test, PBT, And Explicit-Spec Verification

The system shall provide a structured spec-aware verifier for generated Python
code:
- Inputs include the candidate source code, prompt context, entry point,
  official test harness, and optional task id
- The verifier combines official harness execution, Hypothesis-backed PBT, and
  explicit spec checks derived from the checked-in code-spec corpus into one
  structured result
- Explicit spec checks may validate deterministic examples, typed outputs,
  immutability, ordering or reverse intent, no-exception behavior, and other
  corpus-backed clauses without replacing the existing PBT path
- Failures from the explicit-spec layer are convertible to
  pipeline-compatible `ConstraintResult` records and keep enough metadata to
  explain the violated clause, failing input, and observed result or error

### REQ-CODE-027: Trace-Ranked Repair Guidance And Additive Pipeline Integration

The system shall rank repair guidance using the checked-in code-learning path
and expose the spec-aware path additively:
- Ranking reuses the existing trace-learning logic from checked-in benchmark
  artifacts such as Exp 225, Exp 226, and Exp 227
- Observed PBT failures, explicit-spec failures, and official-harness outcomes
  are mapped onto the learned property and repair-strategy families when
  possible
- The verifier returns ranked repair hints with structured evidence including
  learned property support, recommended repair strategies, and supporting case
  ids or trace counts, and ranking remains deterministic for the same checked-in
  traces and failing candidate
- `VerifyRepairPipeline.verify_generated_code(...)` accepts an explicit opt-in
  for the spec-aware path, preserves the existing `VerificationResult`
  surface, and keeps the default generated-code behavior backward compatible
  unless the caller opts in
- The structured certificate carries official-test and explicit-spec summaries,
  ranked repair hints, and the fixed corpus run-date metadata `20260413` when a
  checked-in explicit spec row is used

### REQ-CODE-028: Identical-Stack Dual-Model HumanEval Cohort Reuse

The repository shall provide an identical-stack dual-model HumanEval benchmark
workflow for `Qwen/Qwen3.5-0.8B` and `google/gemma-4-E4B-it`:
- The workflow reuses one checked-in seeded 30-problem HumanEval cohort for
  both models instead of sampling separate slices
- The ordered case list, prompt text, official tests, entry point, and prompt
  seeds are shared across both models so the paired comparison stays
  apples-to-apples
- Each model runs the same verifier stack in the same stage order: baseline,
  official-tests verify-only, PBT verify-only, explicit-spec verify-only, and
  verify-repair
- The generation prompt format, repair prompt format, max token budget, and max
  repair attempts stay identical across both model runs

### REQ-CODE-029: Incremental Stage Summaries And Paired Model Comparison

The identical-stack dual-model benchmark shall emit publishable per-model and
cross-model summaries:
- Each model summary reports baseline pass@1, official-tests verify-only
  acceptance, PBT verify-only acceptance, explicit-spec verify-only
  acceptance, verify-repair pass@1, and paired bootstrap confidence intervals
  for the verify-repair minus baseline delta
- The summary records official-test misses caught by PBT, explicit-spec
  violations, and spec-only misses where the harness plus PBT still accepted
  the candidate
- The artifact includes an explicit same-cohort Gemma-vs-Qwen comparison block
  with paired deltas and confidence intervals for baseline, verify-only, and
  verify-repair outcomes plus paired outcome bucket counts
- The comparison includes a concise methodology note stating that both models
  used the same cohort, verifier stack, prompt formatting, and repair budget

### REQ-CODE-030: Checkpointed Dual-Model Benchmark Artifact With Honest Blockers

The identical-stack dual-model benchmark shall support resumable partial runs
and honest blocker reporting:
- Each model writes checkpointed per-case results often enough to preserve
  completed work during long live runs
- The final artifact includes the checkpoint location(s), run status, and any
  blockers encountered while loading a model, running a stage, or finishing the
  cohort
- When one model finishes and the other is interrupted, the artifact still
  preserves completed per-model results and marks the run as partial instead of
  silently dropping work
- New Exp 238 artifact metadata uses the fixed run-date `20260413`

### REQ-CODE-028: Shared-Cohort Dual-Model Spec-Aware HumanEval Benchmark

The repository shall provide one deterministic HumanEval benchmark workflow
that runs `Qwen/Qwen3.5-0.8B` and `google/gemma-4-E4B-it` on the same ordered
seeded cohort with the same verifier stack:
- The workflow samples or loads one ordered HumanEval cohort once and reuses
  the exact same case ids, task ids, prompts, official tests, and prompt seeds
  for both models
- Baseline prompt formatting, verify-only prompt formatting, repair prompt
  formatting, `max_new_tokens`, and repair budget stay identical across both
  models
- Each case records baseline generation, official-tests-only verify-only,
  additive PBT verify-only, additive explicit-spec verify-only, and
  verify-repair traces under one stable per-case schema
- The explicit-spec layer uses the checked-in code-spec corpus and the same
  trace-ranked repair-hint path for both models

### REQ-CODE-029: Publishable Stage-Wise And Cross-Model Comparison Summary

The paired spec-aware HumanEval benchmark shall emit a publishable summary that
shows the incremental value of each verifier layer and the apples-to-apples
cross-model comparison:
- Per-model summaries include baseline pass@1, official-tests-only verify-only
  acceptance, PBT verify-only acceptance, explicit-spec verify-only acceptance,
  and verify-repair pass@1
- The artifact reports paired 95% confidence intervals for within-model deltas
  such as PBT-over-official, spec-over-PBT, and verify-repair-over-baseline
- The artifact includes a direct Gemma-vs-Qwen comparison block on the shared
  cohort with paired outcome counts, per-stage pass@1 deltas, and paired
  confidence intervals for the main stage comparisons
- The technical-report-ready summary explicitly names the cohort size, shared
  verifier stack, shared repair budget, and which stage gained or failed to
  gain from the explicit spec layer

### REQ-CODE-030: Resume-Aware Blocker Reporting For Paired Live Benchmarks

The paired spec-aware HumanEval benchmark shall checkpoint progress and report
interruptions honestly:
- Checkpoints persist ordered cohort metadata plus per-model completed case
  results often enough to resume long live runs without losing completed work
- Resume logic reuses a checkpoint only when the expected ordered cohort still
  matches
- The final artifact reports `run_status`, completed and pending case counts,
  and any blocker records when hardware, runtime, or model-loading failures
  interrupt all or part of the benchmark
- Completed case traces remain in the final artifact even when the overall run
  is incomplete, and the comparison summary degrades cleanly to the completed
  paired subset instead of silently fabricating missing results

## Scenarios

### SCENARIO-CODE-001: Correct Function Passes Verification

Given a correct Python function `def add(a: int, b: int) -> int: return a + b` with test cases `[((1, 2), 3), ((0, 0), 0)]`, when the code verifier runs, then:
- ReturnTypeConstraint energy = 0.0
- NoExceptionConstraint energy = 0.0
- TestPassConstraint energy = 0.0
- Overall verification result: handcoded_verified = True

### SCENARIO-CODE-002: Buggy Function Fails Verification

Given a buggy function `def add(a: int, b: int) -> int: return a - b` with the same test cases, when the code verifier runs, then:
- TestPassConstraint energy > 0.0
- Overall verification result: handcoded_verified = False

### SCENARIO-CODE-003: Learned Verifier Discriminates

After training a Gibbs model on correct vs buggy code embeddings, the model assigns lower energy to correct code embeddings than to buggy code embeddings on average.

### SCENARIO-CODE-004: Autoresearch Improves Verifier

Running the autoresearch loop with hypothesis templates produces a LoopResult where at least some hypotheses are evaluated (iterations > 0).

### SCENARIO-CODE-005: Safe Execution Handles Errors

When safe_exec_function is called with syntactically invalid code, it returns (None, exception) without crashing the host process.

### SCENARIO-CODE-006: Prompt-Derived Properties Catch A Missed Bug

Given a prompt whose intent is to return a sorted list, and official tests that
only exercise already-sorted inputs, when a buggy identity implementation is
verified, then the official tests alone may pass but the prompt-derived
property verifier flags the code for violating sorted-output or
same-elements-different-order invariants on generated unsorted inputs.

### SCENARIO-CODE-007: Property Failures Become Repair Feedback

Given a candidate function with prompt-derived property failures, when the code
verification pipeline formats repair feedback, then the output includes
pipeline-compatible structured violations that mention the property's source,
the failing input, and the observed incorrect behavior.

### SCENARIO-CODE-008: Hypothesis PBT Finds A Counterexample The Official Tests Miss

Given a HumanEval-style prompt whose official tests only cover already-correct
inputs, when a buggy implementation is verified with the Hypothesis-backed
property verifier, then the official tests may pass while the verifier finds a
counterexample input that violates the prompt-implied property.

### SCENARIO-CODE-009: VerifyRepairPipeline Returns Structured PBT Violations

Given a generated Python code candidate with Hypothesis-backed property
failures, when `VerifyRepairPipeline` runs the generated-code verification
path, then the returned `VerificationResult` includes `ConstraintResult`
violations carrying the property name, source, failing input, and
actual/error details.

### SCENARIO-CODE-010: Five-Problem HumanEval Slice Shows Extra PBT Detection

Given a deterministic five-problem HumanEval-style slice with under-specified
official tests, when execution-only verification and Hypothesis-backed
property verification are compared on the same buggy candidates, then the
property-based path flags at least one additional wrong solution while the same
verifier keeps the matching correct solutions verified.

### SCENARIO-CODE-011: Checkpoint Resume Skips Completed Cases

Given a full-benchmark checkpoint whose ordered cohort metadata still matches
the current 164-problem run, when the workflow resumes, then only the pending
cases execute and the final ordered result list preserves the original case
order.

### SCENARIO-CODE-012: Full-Benchmark Summary Includes Delta CI And Published Comparison

Given full-benchmark baseline and verify-repair outcomes plus at least one
published `Gemma4-E4B-it` HumanEval baseline, when the artifact summary is
built, then it reports baseline and verify-repair pass@1, the paired bootstrap
delta confidence interval, published-baseline deltas, and a technical-report
summary suitable for direct quoting.

### SCENARIO-CODE-013: Seeded Qwen Run Reuses The Gemma Cohort

Given the checked-in Exp 208 Gemma artifact and a live Qwen run on the same
ordered 30-case cohort, when the Exp 227 artifact summary is built, then it
reports the reused cohort ids, Qwen baseline and verify-repair outcomes, the
Gemma baseline and verify-repair reference numbers, the Qwen-minus-Gemma
deltas, and the methodology note for the comparison.

### SCENARIO-CODE-014: Exp 225 Metadata Is Skipped And Exp 226 Traces Are Learned

Given the checked-in Exp 225 and Exp 226 benchmark artifacts, when the
trace-learning module analyzes them together, then it records Exp 225 as a
metadata-only artifact with no usable verification history, ingests Exp 226's
per-problem verification traces, and produces property and repair summaries
from the accumulated trace corpus.

### SCENARIO-CODE-015: Cumulative Trace Learning Sharpens Recommendations

Given an ordered stream of verification traces where some repair families
reliably succeed and others do not, when the learner recomputes its rankings on
larger and larger prefixes, then the recommended property and repair strategy
for the matching error family moves toward the empirically successful choice.

### SCENARIO-CODE-016: Packaged Python API Flags A PBT Violation From Source Alone

Given a Python function with enough signature information to derive bounded
PBT checks, when the standalone `carnot.pipeline.verify_code()` API verifies
the source without separate prompt files, then it returns a failing
`VerificationResult` whose certificate includes the additive PBT summary.

### SCENARIO-CODE-017: CLI verify-code Reports A Property Failure

Given a Python source file whose function violates a signature-implied
property, when a user runs `carnot verify-code <file> --func <name> --pbt`,
then the CLI exits non-zero and reports the failing packaged verification
finding in terminal output.

### SCENARIO-CODE-018: MCP Packaged Verification Returns Repair Feedback

Given buggy generated Python code plus the prompt or official-test context that
implies stronger properties than the visible examples, when an MCP client calls
`verify_code_with_pbt`, then the result includes `verified=False`, structured
violations, repair feedback, and the additive PBT summary.

### SCENARIO-CODE-019: Generated Code Is Verified, Repaired, And Re-Verified

Given an LLM-style generated function body that passes weak official tests but
violates a prompt-implied property, when the packaged verification workflow
verifies the candidate, applies a repair, and verifies again, then the initial
candidate fails packaged verification and the repaired candidate verifies cleanly.

### SCENARIO-CODE-020: Re-Running The Spec Corpus Generator Is Byte-Stable

Given the same checked-in Exp 226 and Exp 227 source artifacts, when the
explicit code-spec corpus workflow runs twice, then it emits the same ordered
rows and the same summary counts without introducing nondeterministic field
ordering or row ordering changes.

### SCENARIO-CODE-021: Overlapping Traces Merge Into One Spec Row With Provenance

Given a HumanEval task that appears in both Exp 226 and Exp 227, when the
explicit code-spec corpus is built, then the task appears once in the corpus,
its spec clauses preserve provenance links back to both source traces, and the
summary counts still record both contributing traces.

### SCENARIO-CODE-022: Spec Lookup Falls Back From Task Id To Entry Point

Given a checked-in explicit code-spec corpus row, when the verifier is called
without an explicit task id but with the matching entry point and signature
context, then it resolves the same row deterministically and preserves the
row’s fixed run-date metadata and grouped clause families.

### SCENARIO-CODE-023: Aggregated Spec-Aware Verification Flags A Harness-Passing Bug

Given a generated Python candidate that still passes the weak official
HumanEval harness but violates a corpus-backed ordering or mutation clause,
when the spec-aware verifier runs, then the structured result reports the
official harness as passing while the combined PBT and explicit-spec layers
still reject the candidate and expose pipeline-compatible failures.

### SCENARIO-CODE-024: Trace-Learned Ranking Prioritizes The Higher-Yield Repair Family

Given failing code whose observed verifier signals match a repair family with
documented wins in the checked-in trace-learning corpus, when ranked repair
guidance is produced, then the top hint prefers that empirically successful
repair family ahead of lower-yield generic guidance.

### SCENARIO-CODE-025: VerifyRepairPipeline Adds Spec Summaries Only On Opt-In

Given the existing generated-code verification surface, when callers leave the
spec-aware path disabled, then the default behavior stays unchanged; and when
callers opt in, then the returned `VerificationResult.certificate` includes the
official-test summary, explicit-spec summary, and ranked repair hints.

### SCENARIO-CODE-026: Dual-Model Benchmark Reuses One Ordered Seeded Cohort

Given the checked-in seeded 30-problem HumanEval cohort used for Exp 238, when
the benchmark loads both model runs, then Qwen and Gemma receive the same
ordered case ids, prompts, official tests, entry points, and prompt seeds for
baseline and repair generation.

### SCENARIO-CODE-027: Spec-Aware Stage Summary Shows Incremental Value Beyond PBT

Given a shared-cohort dual-model benchmark run, when the per-model summary is
built, then it reports official-tests verify-only, PBT verify-only,
explicit-spec verify-only, and verify-repair side by side and records whether
the explicit spec layer caught any harness-passing cases that PBT alone did not
reject.

### SCENARIO-CODE-028: Partial Dual-Model Run Preserves Completed Work And Blockers

Given a dual-model benchmark run where one model finishes and the other
interrupts after some completed cases, when the artifact is written, then it
keeps the finished model's results, preserves the interrupted model's completed
checkpointed cases, marks `run_status` as partial, and records the blocker with
the failing model and stage.

### SCENARIO-CODE-026: Shared Seeded Cohort Reuses The Exact Same Cases For Both Models

Given the paired spec-aware benchmark workflow, when it builds the HumanEval
cohort for one run, then both models receive the same ordered case ids, task
ids, prompts, official tests, and prompt seeds instead of independently sampled
 slices.

### SCENARIO-CODE-027: Stage-Wise Summary Shows The Explicit Spec Increment

Given paired per-case traces that include official-tests-only, additive PBT,
 additive explicit-spec verification, and verify-repair outcomes, when the
 artifact summary is built, then the per-model summary and the Gemma-vs-Qwen
 comparison block both report stage-wise pass rates, paired outcome counts, and
 paired confidence intervals that isolate the explicit spec layer's effect.

### SCENARIO-CODE-028: Interrupted Run Preserves Completed Traces And Reports Blockers

Given a paired live benchmark that fails partway through one model or stage,
when the final artifact is written, then it keeps completed traces plus resume
 checkpoint metadata, marks the run incomplete, and records the blocker without
 claiming missing cases ran successfully.

### REQ-CODE-031: Two-Round Iterative Code Repair Pipeline

The system shall implement a TwoRoundCodeRepairPipeline that:
- Generates initial code from a problem prompt via an LLM caller.
- Executes the generated code against test cases, capturing full tracebacks on failure.
- Injects the full traceback, expected output, actual output, and original problem into a repair prompt.
- Re-generates code for up to 2 repair rounds when failures occur.
- Classifies each failure as one of: syntax_error, assertion_error, name_error, timeout, other.
- Returns a TwoRoundResult with round0_pass, round1_pass, round2_pass, and error_types per round.

### REQ-CODE-032: Per-Round Pass Rate Reporting

The system shall report iterative code repair results with per-round pass@1 metrics:
- pass_round0: fraction of problems passing before any repair.
- pass_round1: fraction passing after at most 1 repair (cumulative).
- pass_round2: fraction passing after at most 2 repairs (cumulative).
- round1_improvement: pass_round1 - pass_round0.
- round2_improvement: pass_round2 - pass_round1.
- total_improvement: pass_round2 - pass_round0.
- honest_verdict: "two_round_repair_confirmed" (>=0.02), "two_round_repair_marginal" (>0, <0.02),
  or "two_round_repair_no_improvement" (<=0).

### SCENARIO-CODE-029: Two-Round Repair Fixes a Bug in Round 1

Given a problem where round 0 produces wrong output, when TwoRoundCodeRepairPipeline runs,
then the repair prompt includes the full traceback, expected output, and actual output,
and the round 1 code is re-executed with the repaired code, recording round1_pass=True.

### SCENARIO-CODE-030: All Rounds Tracked in Pass Rate Computation

Given 4 problems where 1 passes in round 0, 1 in round 1, 1 in round 2, and 1 never,
when compute_pass_rates is called, then pass_round0=0.25, pass_round1=0.50, pass_round2=0.75
and error_type_breakdown correctly attributes each failed round to its error type.

### REQ-REPAIR-020: Iterative Repair Prompt Must Include Full Traceback and Test Case Inputs

The iterative 2-round code repair experiment (Exp 759) shall build repair prompts that:
- Include the FULL traceback string from the failing execution.
- Include the failing test case call expression (input) explicitly.
- Include the expected output for that test case.
- Are structurally different from the original generation prompt (repair prompt adds error context).

Rationale: arXiv 2604.10508 shows error message quality is the primary driver of
self-repair gains. A repair prompt that omits the traceback or test input gives the
model insufficient signal to fix the bug.

Spec: REQ-REPAIR-020, SCENARIO-REPAIR-040

### REQ-REPAIR-021: Live GPU Execution Required for Code Repair Experiments

The code repair live experiment (Exp 759) shall:
- Require CARNOT_FORCE_LIVE=1 to be set before running GPU inference.
- Write a blocked artifact with honest_verdict="blocked_no_live_gpu" if CARNOT_FORCE_LIVE is not set.
- Never simulate inference results when CARNOT_FORCE_LIVE=1 is absent.

Spec: REQ-REPAIR-021, SCENARIO-REPAIR-041

### SCENARIO-REPAIR-040: Repair Prompt Contains Traceback and Test Case Call

Given a code execution that fails with an AssertionError, when build_repair_prompt_759() is called,
then the returned prompt string contains the full traceback text AND the failing test case
call expression (the exact function call that returned wrong output).

### SCENARIO-REPAIR-041: Blocked Artifact Written When CARNOT_FORCE_LIVE Not Set

Given that CARNOT_FORCE_LIVE is not set in the environment, when Exp 759 runs,
then it writes a JSON artifact with honest_verdict="blocked_no_live_gpu" and exits
without attempting GPU inference.

### REQ-REPAIR-022: SOTA GGUF Model Required for Headline Code Repair Results

The SOTA GGUF code repair experiment (Exp 769) MUST use unsloth/Qwen3.6-35B-A3B-GGUF or
an equivalent model >= 7B parameters for all headline code repair results.
Qwen3.5-0.8B MUST NOT be used as a headline code repair model (it is too small to
generate valid Python code — Exp 759 confirmed pass@1=0.0 with this model).

Rationale: arXiv 2604.10508 tested only models >= 7B. Qwen3.6-35B-A3B (MoE with
~3B active params, Q4_K_M fits in 24GB VRAM) is the mandatory minimum for credible
headline code repair numbers from Carnot.

Spec: REQ-REPAIR-022, SCENARIO-REPAIR-042

### REQ-REPAIR-023: signed_improvement Computed From Round1 vs Round2 Pass@1

The signed_improvement metric in code repair experiments MUST be computed as:
  signed_improvement = pass_at_1_round2 - pass_at_1_round1

Where:
- pass_at_1_round1: fraction of N problems passing in initial (round 1) generation.
- pass_at_1_round2: cumulative fraction passing after at most one repair (round 2).
- n_repaired: count of problems that fail round 1 but pass round 2.

This definition must be consistent across Exp 759 and Exp 769 to allow direct comparison.

Spec: REQ-REPAIR-023, SCENARIO-REPAIR-043

### SCENARIO-REPAIR-042: SOTA GGUF Smoke Test Confirms Model Loads

Given the Qwen3.6-35B-A3B-GGUF Q4_K_M file is available in the HuggingFace cache,
when the experiment loads the model via llama-cpp-python, then a smoke test generation
completes successfully and the model is used for all 50 HumanEval problems.

### SCENARIO-REPAIR-043: n_repaired Count Matches pass@1 Delta

Given N problems with known round1_pass and round2_pass values, when compute_repair_metrics()
is called, then n_repaired == count(NOT round1_pass AND round2_pass), and
signed_improvement == pass_at_1_round2 - pass_at_1_round1 to 4 decimal places.

### REQ-REPAIR-024: SOTA GGUF v2 Must Kill GPU Zombies Before Model Load, Limit to 25 Problems

The SOTA GGUF code repair v2 experiment (Exp 785) MUST call kill_gpu_zombies(gpu_index=0)
from carnot.pipeline.gpu_zombie_killer BEFORE any model loading attempt. The experiment MUST
limit the benchmark to 25 HumanEval problems with a per-problem timeout of 3 minutes.
The overall experiment MUST use a 90-minute hard cap via ExperimentTimeoutWatchdog.

Rationale: Exp 769 timed out (RETRO-SOTA-GGUF-TIMEOUT) for two reasons: (1) 15 GiB of zombie
VRAM occupied GPU 0 before model load, causing OOM; (2) 50 problems × 3-4 min/problem = 150-200
min exceeds the 120-min budget. Killing zombies first and halving the problem count fixes both.

Spec: REQ-REPAIR-024, SCENARIO-REPAIR-044

### REQ-REPAIR-025: VRAM-Based Model Fallback — 7B If Qwen3.6-35B OOM After Zombie Kill

If free VRAM after kill_gpu_zombies() is < 20000 MB, Exp 785 MUST fall back to
Qwen3.5-7B-Instruct-GGUF Q4_K_M (~4 GiB) instead of Qwen3.6-35B-A3B-GGUF Q4_K_M (~20 GiB).
Both models are acceptable headline models for code repair results.
The artifact MUST record model_used and free_vram_mb_after_kill to document which path ran.

Rationale: The 7B fallback ensures the experiment produces a live_gpu result even when VRAM
is constrained after zombie processes have been killed. Known from Exp 759: Qwen3.5-7B
generates valid Python code (unlike 0.8B which produced pass@1=0.0).

Spec: REQ-REPAIR-025, SCENARIO-REPAIR-045

### SCENARIO-REPAIR-044: kill_gpu_zombies() Called Before Model Load in Exp 785

Given Exp 785 is launched with CARNOT_FORCE_LIVE=1, when the experiment initialises,
then kill_gpu_zombies(gpu_index=0) from carnot.pipeline.gpu_zombie_killer is called
BEFORE any llama-cpp model load attempt, and free_vram_mb_after_kill is recorded in artifact.

### SCENARIO-REPAIR-045: Exp 785 Falls Back to 7B When free_vram_mb < 20000

Given free VRAM after zombie kill is < 20000 MB, when Exp 785 selects a model,
then it uses Qwen3.5-7B-Instruct-GGUF Q4_K_M and records model_used="Qwen3.5-7B-Instruct-GGUF"
in the artifact. The signed_improvement calculation is identical regardless of which model runs.

### REQ-CODE-033: IterativeSelfRepair Pipeline — Execution-Feedback-Driven Code Repair

The system shall implement an IterativeSelfRepair pipeline that bypasses
ArithmeticExtractor and uses execution tracebacks as the correction signal:
- Generates initial code from a problem prompt via an LLM runner (attempt 0).
- Executes the generated code in a subprocess sandbox with a configurable timeout.
- On execution failure, feeds the full traceback back to the LLM inside a
  correction prompt that includes the original problem, the failing code, and
  the error text (arXiv 2604.10508 pattern).
- Retries up to max_retries times (default 3).
- After all attempts, selects the best attempt using the Carnot energy scorer:
  passing attempts are preferred over failing ones; among passing (or failing)
  attempts, the lowest-energy attempt is selected.
- Returns a RepairResult with the best attempt, all attempts, n_retries, and
  a boolean recording whether the energy scorer selected a passing attempt.
- Supports optional gVisor sandbox isolation via CARNOT_USE_SANDBOX=1.

### SCENARIO-CODE-031: Iterative Repair Retries With Execution Feedback Until Passing Or Budget

Given a problem where the first attempt fails, when IterativeSelfRepair.repair() runs,
then the correction prompt includes the original code, the full traceback, and the
problem description, and subsequent attempts use the repaired code, continuing until
a passing attempt is found or max_retries is exhausted.

### REQ-CODE-034: HumanEval Prompt NSVIF DSL Extraction Evaluation

The repository shall provide a deterministic Exp 1607 evaluation that applies
the NSVIF instruction-to-constraint DSL to HumanEval prompts:
- The workflow loads HumanEval prompts, extracts bounded code-generation
  constraints such as entry-point presence, markdown-free code output, placeholder
  avoidance, and return-statement expectations from each prompt.
- Each extracted constraint pack compiles through the existing safe local DSL
  compiler without arbitrary code execution.
- The workflow reports attempted prompts, valid extractions, extraction failures,
  constraints extracted, valid extraction rate, and compiled validator rate.
- The artifact records the mandated model specs
  `unsloth/Qwen3.6-35B-A3B-GGUF` and `unsloth/gemma-4-31B-it-GGUF` as the target
  code-verification model cohort without requiring live model inference for this
  deterministic DSL-only extraction step.
- The workflow writes `results/experiment_1607_dsl_humaneval.json` with complete
  summary metrics, per-model extraction summaries, sample rows, tests run, and an
  honest verdict.

### SCENARIO-CODE-032: HumanEval Prompts Compile To Safe DSL Packs

Given a deterministic HumanEval prompt cohort and the mandated Exp 1607 model
spec list,
When the Exp 1607 NSVIF extraction workflow runs,
Then every prompt with supported prompt-derived code constraints produces a
schema-valid DSL pack, the pack compiles to a local validator and PySAT-compatible
CNF, no arbitrary code execution path is introduced, and the terminal artifact
records valid extraction and compiled validator rates for both model specs.

### REQ-CODE-2879: Manifest-Only MBPP/HumanEval Execution Pilot

The repository shall provide a deterministic manifest-only MBPP/HumanEval code
execution pilot that does not generate code with an LLM. The pilot MUST:

- resolve MBPP and HumanEval manifest paths and checksums through the checked-in
  evaluation manifest contract artifact;
- select a tiny deterministic sample whose rows contain canonical/reference code
  plus local test data, and record the selection rule and checksums;
- execute only the canonical/reference code against the local tests through the
  existing sandboxed Python execution path;
- write `blocked_sandbox` and no pilot rows when gVisor/runsc sandbox isolation
  is unavailable or the execution wrapper would fall back to in-process exec;
- report per-row pass/fail/test metadata and verifier feature coverage; and
- set `headline_metric_claim_made=false` because the pilot does not contain
  generated-code labels for AUROC or headline pass@k claims.

### SCENARIO-CODE-2879: Canonical Code Rows Run Through The Manifest Contract

Given the eval manifest contract resolves MBPP and HumanEval manifests with
verified checksums,
When the manifest-only pilot runs with gVisor sandbox execution available,
Then one deterministic MBPP row and one deterministic HumanEval row are selected
from rows with canonical/reference code and tests, both rows are executed through
the sandbox wrapper, per-row pass/fail metadata is recorded, and no headline
metric is claimed.

### REQ-CODE-2889: Smallest Defensible MBPP/HumanEval Generated-Code Row

The repository shall provide a bounded MBPP/HumanEval generated-code row that
uses the local mandated SOTA GGUF runtime selected by Exp 2874 and the existing
gVisor/runsc sandbox for deterministic local tests. The runner MUST:

- gate on Exp 2874's `sota_runtime_clean=True` with a non-empty resolved model
  path, on visible CUDA/GPU offload, on the eval manifest contract being ready,
  and on a runsc sandbox being available, writing a `blocked_*` artifact and
  stopping if any precondition is unmet;
- select a tiny deterministic sample of MBPP and HumanEval rows (target five
  each when manifests/tests support them, fewer when only fewer are safe),
  record the selection rule, fingerprints, and stable IDs;
- generate candidate code with the cached SOTA GGUF through a fixed
  instruction prompt, deterministic seed, greedy decoding, and a bounded token
  budget, capturing the raw generated text per row;
- execute generated outputs only through the existing sandbox wrapper with
  `allow_fallback=False`, capturing pass/fail, timeout, exception, and
  unsupported reasons per row without any in-process fallback;
- record `generated_code_row_clean=True` only when generation, sandbox
  execution, manifests, and per-row outputs are all clean, and otherwise mark
  the artifact as pilot-only or blocked with a row-level explanation; and
- never claim a headline pass@k or AUROC metric — `headline_metric_claim_made`
  MUST be `False` for this row.

### SCENARIO-CODE-2889: Bounded SOTA GGUF Generation With Deterministic Local Tests

Given Exp 2874 recorded `sota_runtime_clean=True` with a resolved cached SOTA
GGUF, the eval manifest contract resolves both MBPP and HumanEval manifests
with verified checksums, and the runsc sandbox is available,
When the Exp 2889 generated-code row runs with the cached SOTA GGUF as the
generator and the existing sandbox wrapper as the executor,
Then candidate code is generated for each selected row from a fixed prompt and
deterministic seed, the generated function is executed through the sandbox
against the row's local tests, the artifact records the model fingerprint,
selection rule, per-row pass/fail metadata, and a row status of `pilot_only` or
`blocked_*` when any output is not clean, and no headline metric is claimed.

### REQ-CODE-2905: Bounded-Budget k=8 SOTA Code Generation Expansion

The repository shall provide a live MBPP/HumanEval generated-code expansion
runner for Exp 2905 that keeps the Exp 2889 safety gates while increasing the
per-prompt token budget and sampling exactly `k=8` candidates per selected task.
The runner MUST:

- gate on the same local SOTA GGUF, eval manifest, GPU/offload, and runsc
  sandbox preconditions as Exp 2889, writing a terminal `blocked_*` artifact
  instead of fabricating live inference when any precondition is unmet;
- select a deterministic bounded sample from each corpus and record
  `n_tasks_per_corpus`, `k_candidates_per_task`, per-task stable IDs, and every
  random seed used for candidate sampling;
- use `inference_substrate="live_llm_inference"` and a `model_specs.principle`
  stating that the task actually invokes the live SOTA model and that the 60s
  duration floor applies;
- execute every non-empty generated candidate only through the runsc sandbox
  test harness, with no in-process fallback;
- report `per_task_pass_at_1` and `per_task_pass_at_k` lists where each entry
  is `1.0` when the corresponding task passes on the first candidate or any of
  its `k` candidates, otherwise `0.0`; and
- write `results/experiment_2905_sota_code_generation_bounded_budget_expansion_v1.json`
  with the required schema fields, a reproducibility checksum over manifests,
  selected tasks, seeds, token budget, and model fingerprint, and measured
  wall-clock `duration_s`.

### SCENARIO-CODE-2905: k=8 Candidate Sampling Gives The Verifier A Choice Set

Given Exp 2874 recorded clean live SOTA runtime evidence, the code manifests
verify, and runsc is available,
When the Exp 2905 runner samples `k=8` candidates for each selected MBPP and
HumanEval task using the expanded token budget,
Then the artifact records all candidate seeds and outcomes, computes per-task
pass@1/pass@k from sandbox execution results, identifies whether pass@k exceeds
pass@1 on at least one task, and preserves live-inference provenance in
`model_specs` and `inference_substrate`.

### REQ-CODE-2910: SOTA Code-Generation Corrigendum v2

The repository shall provide an Exp 2910 corrigendum runner for the
adversarially flagged Exp 2905 SOTA code-generation row. The runner MUST:

- call `cached_sota_pair(gpu_indices=(0, 1))` before selecting any model and
  record whether that pair was usable;
- use only mandated SOTA GGUF model paths for headline generation, with legacy
  tiny models allowed only for CPU smoke tests; when no mandated SOTA GGUF is
  cached, write `blocked_sota_gguf_cache_missing` and leave headline pass-rate
  metrics unreported;
- set top-level `random_seed=2910`, sample at least 20 MBPP and 20
  HumanEval-style local manifest rows for non-blocked runs, and sample exactly
  `k_candidates_per_task=8` bounded candidates per task;
- capture every raw response to a raw-response directory and record per-candidate
  seed, model path, GPU index, prompt hash, extraction status, syntax success,
  runtime success, timeout, and pass/fail outcome;
- compute aggregate and per-task pass@1/pass@k only from sandbox execution
  outcomes, never from generated text inspection; and
- include a methodology note explaining why pass@1 and pass@k are distinct
  metrics, failing the row when aggregate pass@1 and pass@k match exactly
  without per-task explanations.

### SCENARIO-CODE-2910: Corrigendum Writes Clean Or Honest Blocked Artifact

Given local MBPP and HumanEval-style manifests are available,
When the Exp 2910 runner resolves SOTA GGUF cache state, samples rows, generates
k=8 candidates, extracts code, and executes candidates through the sandbox,
Then it writes `results/experiment_2910_sota_code_generation_corrigendum_v2.json`
with the required corrigendum schema, top-level seed provenance, raw-response
paths, model provenance, per-task pass vectors, and either a clean corrected row
or an honest blocked verdict without fabricated headline pass rates.

### REQ-CODE-2911: Deterministic Exp 2910 Code-Hallucination Taxonomy Verifier

The repository shall provide a deterministic taxonomy verifier for the checked-in
Exp 2910 code-generation corrigendum artifact. The verifier MUST:

- block with `honest_verdict="blocked_codegen_corrigendum_missing"` when
  `results/experiment_2910_sota_code_generation_corrigendum_v2.json` is absent
  or does not set `codegen_corrigendum_ready=True`;
- load Exp 2910 `per_task_results`, candidate rows, and raw-response files, then
  re-parse every candidate with Python AST before assigning taxonomy labels;
- statically and orthogonally label invented or unavailable imports, undefined
  names, invented attributes or methods, and invalid argument counts or
  incompatible call signatures;
- preserve Exp 2910 sandbox task-test outcomes so failures can be separated into
  true assertion-test failures, syntax errors, runtime errors, and code
  hallucination categories without rerunning live generation; and
- write
  `results/experiment_2911_code_hallucination_taxonomy_verifier_v1.json` with
  per-candidate labels, category rates, grouped summaries by model, corpus, task,
  and pass/fail status, verifier source paths, deterministic substrate metadata,
  run date `20260523`, and measured duration.

### SCENARIO-CODE-2911: Taxonomy Keeps Hallucination Categories Orthogonal

Given the checked-in Exp 2910 artifact contains generated MBPP/HumanEval
candidates with raw-response paths and sandbox task-test outcomes,
When the Exp 2911 verifier parses each candidate and runs its static taxonomy
checks,
Then one candidate may carry multiple independent labels such as
`invented_import`, `undefined_name`, `invented_attribute_or_method`, and
`invalid_argument`, while assertion-only failures remain labeled as
`true_test_failure`, and the terminal artifact reports rates and grouped
summaries without fabricating missing upstream evidence.

### REQ-CODE-5445: Deterministic AST/KB Witness Constraints For Code/API Hallucinations

The repository shall provide a deterministic Exp 5445 AST/KB witness fixture
for code/API hallucination rows. The verifier MUST:

- use a bounded fixture set covering valid API calls, nonexistent methods,
  wrong-module aliases, bare calls with missing imports, unavailable imported
  symbols, argument-intent mismatches, and benign controls;
- parse every fixture through Python AST and emit the parse status before any
  API judgment;
- resolve simple `import X as Y` and `from X import Y as Z` aliases into a
  row-level alias map;
- validate imported symbols and fully qualified call sites against a safe
  local knowledge base built from whitelisted installed-library introspection
  with committed fallback metadata when imports are unavailable;
- emit inspectable witness fields for each row, including source checksum,
  alias map, fully qualified call sites, imported-symbol checks, KB lookup
  results, semantic intent evidence, deterministic accept/reject outcome,
  reject reasons, and a witness checksum; and
- write `results/experiment_5445_static_ast_kb_witness_constraints_v495.json`
  with `fixture_count`, `api_family_counts`, `ast_parse_success_rate`,
  `kb_source_paths`, `witness_field_names`, `nonexistent_call_reject_rate`,
  `valid_call_accept_rate`, `unsafe_false_accepts`,
  `row_provenance_checksum`, `ast_kb_witness_ready`,
  `inference_substrate="deterministic_ast_kb_verifier_no_llm"`, and a terminal
  `honest_verdict` beginning with `complete:` or `blocked:`.

### SCENARIO-CODE-5445: Witness Rows Reject Hallucinated API Structure And Keep Controls Clean

Given the Exp 5445 bounded fixture set, when the AST/KB verifier evaluates the
rows, then alias-resolved valid calls are accepted, nonexistent methods and
unavailable imported symbols are rejected, bare calls without matching imports
are rejected, argument-intent mismatches are rejected even when the API exists,
benign controls are accepted, per-row witness checksums are reproducible, and
the terminal artifact is ready only when all invalid hallucination rows are
rejected with zero unsafe false accepts and no live LLM inference.

### REQ-CODE-2946: AUPRC-Gated SOTA Code-Generation Continuation

The repository shall provide an Exp 2946 continuation runner that reads Exp
2940's `paper_v6_recommendation` before deciding whether any further
code-generation pass-rate claim is allowed. The runner MUST:

- require the checked-in Exp 2940 artifact and record the normalized
  `exp2940_recommendation_used`;
- verify local torch CUDA availability before live continuation protocols and
  block honestly when CUDA is unavailable;
- execute the Exp 2910 live GGUF, k=8, MBPP/HumanEval sandbox protocol with
  `n_tasks=50` when Exp 2940 recommends `retain`;
- execute the same protocol with `n_tasks=20` plus explicit limitation framing
  when Exp 2940 recommends `narrow`;
- execute only failure-mode analysis, with no pass-rate claim, when Exp 2940
  recommends `retract`;
- write `results/experiment_2946_sota_code_generation_continuation_v1.json`
  with top-level `honest_verdict`, `inference_substrate`,
  `exp2940_recommendation_used`, `protocol_executed`, `pass_at_1`,
  `pass_at_k`, `failure_mode_analysis`, `random_seeds_used`,
  `reproducibility_checksum`, and measured `duration_s`.

### SCENARIO-CODE-2946: Retain Verdict Expands The Live Protocol To 50 Tasks

Given Exp 2940 reports `paper_v6_recommendation.value="retain"` and torch CUDA
is available,
When the Exp 2946 runner builds the continuation artifact,
Then it executes the Exp 2910-style live protocol on 50 total MBPP/HumanEval
tasks with k=8 candidates per task, records pass@1/pass@k only from the nested
sandbox outcomes, carries every candidate seed into `random_seeds_used`, and
sets `failure_mode_analysis=null`.

### REQ-CODE-2950: Code Taxonomy Repair Prompt Manifest

The repository shall provide an Exp 2950 deterministic repair-prompt manifest
builder for the post-.277 SOTA code-generation failure modes. The builder MUST:

- require checked-in Exp 2940, Exp 2943, and Exp 2946 artifacts before emitting
  a ready manifest, and block honestly when any prerequisite is absent;
- aggregate failed and low-scoring candidate evidence from Exp 2946 and
  code-corpus verifier artifacts without running live generation and without
  claiming a pass-rate improvement;
- define taxonomy labels and repair prompt templates for syntax errors,
  missing symbols, wrong return type, failed tests, unsafe imports, and
  unsupported API hallucinations;
- map every taxonomy label to deterministic post-repair checks including parser
  checks, tests where present, static import/name checks, and the Exp 2940
  verifier threshold; and
- write `results/experiment_2950_code_taxonomy_repair_prompt_manifest_v1.json`
  with `honest_verdict`, `repair_prompt_manifest_ready`, `source_artifacts`,
  `model_specs`, `legacy_models_only_for_smoke`, `taxonomy_labels`,
  `repair_prompt_templates`, `deterministic_checks`, `downstream_eval_plan`,
  `inference_substrate="aggregation_from_upstream_artifacts"`, and measured
  `duration_s`.

### SCENARIO-CODE-2950: Repair Manifest Uses Upstream Failures Without New Pass-Rate Claim

Given the checked-in Exp 2940, Exp 2943, and Exp 2946 artifacts are present,
When the Exp 2950 builder aggregates candidate failures and verifier gates,
Then the artifact records sample IDs, taxonomy labels, repair templates,
deterministic checks, acceptance criteria, mandated SOTA GGUF model specs, and
a downstream live-evaluation plan while preserving
`inference_substrate="aggregation_from_upstream_artifacts"` and making no
claim that pass@1 or pass@k improved.

### REQ-CODE-2951: Structured Candidate Manifest Adapter

The repository shall provide an Exp 2951 local-first structured candidate
manifest adapter for the .278 code-repair evaluation. The adapter MUST:

- define a reusable candidate manifest schema with `task_id`, `prompt_id`,
  `model_id`, `raw_completion_ref`, `repaired_code`, `failure_taxonomy`,
  `parser_status`, `test_status`, `verifier_score`, and
  `provenance_checksums`;
- probe local JSON-schema validation support, `llguidance`, and llama.cpp
  grammar support without installing packages and without calling a live LLM;
- prefer locally available grammar metadata for downstream structured decoding
  while retaining a deterministic schema-validation fallback when grammar
  backends are absent;
- validate at least three synthetic candidate records, including one valid
  candidate, one syntax failure, and one unsupported import/API hallucination;
- include downstream live-use model specs for
  `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`; and
- write `results/experiment_2951_structured_candidate_manifest_adapter_v1.json`
  with `honest_verdict`, `structured_decode_manifest_ready`,
  `schema_version`, `schema_fields`, `local_backends_checked`,
  `llguidance_available`, `llama_cpp_grammar_available`,
  `validation_fixture_count`, `validation_fixture_passed`,
  `model_specs_for_downstream_live_use`,
  `inference_substrate="deterministic_wiring"`, and measured `duration_s`.

### SCENARIO-CODE-2951: Synthetic Candidate Manifests Validate Locally

Given no live model call is made,
When the Exp 2951 adapter builds its schema, probes local structured-output
backends, and validates the three synthetic candidate records,
Then the result artifact records the required schema fields, local backend
availability, deterministic validation results for all fixtures, downstream
SOTA GGUF model specs, and readiness only when the deterministic fallback
accepts the valid row and rejects no schema-compliant failure rows.

### REQ-CODE-2952: SOTA Taxonomy-Guided Code-Repair Evaluation

The repository shall provide an Exp 2952 bounded live repair evaluator for the
post-.277 generated-code failures. The evaluator MUST:

- require checked-in Exp 2940, Exp 2946, Exp 2950, and Exp 2951 artifacts and
  preserve the selected Exp 2946 task IDs, candidate seeds, and original
  failure categories;
- check the dual RTX 3090 host, local llama.cpp GGUF runtime, runsc sandbox,
  Exp 2950 repair manifest, Exp 2951 structured candidate schema, and at least
  one cached mandated headline GGUF before live repair;
- run a baseline no-taxonomy repair prompt and a taxonomy-guided repair prompt
  under the same bounded sample budget using only mandated local SOTA GGUF
  models for headline results, while legacy small models remain smoke-only;
- validate every generated repair through the Exp 2951 structured manifest
  schema, Python parser/static checks, available MBPP/HumanEval tests, and the
  Exp 2940 code-verifier threshold;
- report baseline and taxonomy-guided pass@1/pass@k, syntax failure rate,
  schema failure rate, verifier acceptance rate, false verifier accepts, and
  audit notes; and
- write `results/experiment_2952_sota_taxonomy_guided_code_repair_eval_v1.json`
  with top-level `honest_verdict`, `inference_substrate="live_llm_inference"`,
  `preconditions_checked`, `model_specs`, `headline_models_used`,
  `legacy_models_only_for_smoke`, `n_tasks`, `baseline_pass_at_1`,
  `repair_pass_at_1`, `pass_at_1_delta`, `baseline_pass_at_k`,
  `repair_pass_at_k`, `pass_at_k_delta`, `syntax_failure_rate_delta`,
  `schema_failure_rate_delta`, `false_accept_delta`,
  `taxonomy_repair_delta_pass`, `candidate_manifest_sha256`,
  `reproducibility_checksum`, and measured `duration_s`.

### SCENARIO-CODE-2952: Taxonomy Repair Is Accepted Only With Cleaner Metrics

Given Exp 2946 contains failed generated-code candidates and Exp 2950/2951 are
ready,
When Exp 2952 runs equal-budget baseline and taxonomy-guided repairs through a
mandated cached local GGUF,
Then the artifact records the same selected task IDs for both modes, validates
each repair candidate through the structured manifest and deterministic
checks, computes deltas as taxonomy-guided minus baseline, and sets
`taxonomy_repair_delta_pass=true` only when `pass_at_1_delta > 0` or
`syntax_failure_rate_delta < 0` while `false_accept_delta <= 0`.

### REQ-CODE-2953: Code Verifier Threshold Policy

The repository shall provide an Exp 2953 deterministic threshold-policy builder
for the Exp 2940 code-corpus verifier evidence. The builder MUST:

- require checked-in Exp 2940 and Exp 2943 artifacts before emitting a ready
  policy;
- use Exp 2940 score-distribution or precision-recall summary fields to derive
  conservative, balanced, and permissive operating points without inventing
  missing score or label rows;
- compute expected PPV, recall, and false-accept rate for each operating point
  from the measured Exp 2940 positive count, candidate count, and
  precision-recall curve;
- select a default threshold for code repair/generation filtering only when
  the false-accept boundary is explicit; and
- write `results/experiment_2953_code_verifier_threshold_policy_v1.json` with
  `honest_verdict`, `threshold_policy_ready`, `source_artifacts`,
  `operating_points`, `selected_default_threshold`,
  `expected_ppv_at_default`, `expected_recall_at_default`,
  `expected_false_accept_rate_at_default`, `deployment_boundary`,
  `missing_score_distribution`,
  `inference_substrate="aggregation_from_upstream_artifacts"`, and measured
  `duration_s`.

### SCENARIO-CODE-2953: Threshold Policy Defines Repair Filtering Boundaries

Given Exp 2940 contains code-corpus AUPRC evidence, a precision-recall curve,
candidate counts, positive counts, and score-distribution provenance, and Exp
2943 carries the cross-corpus boundary row,
When Exp 2953 builds the threshold-policy artifact,
Then the artifact reports conservative, balanced, and permissive thresholds
with PPV/recall/false-accept-rate tradeoffs, selects the conservative threshold
as the default for automated candidate filtering, and states that the threshold
is scoped to code-repair triage rather than standalone correctness acceptance.

### SCENARIO-CODE-2953-BLOCKED: Missing Score Distribution Produces Partial Artifact

Given Exp 2940 is present but omits the score-distribution field needed to audit
threshold provenance,
When Exp 2953 builds the threshold-policy artifact,
Then it sets `threshold_policy_ready=false`, `missing_score_distribution=true`,
names the missing Exp 2940 field, proposes the next local command to regenerate
Exp 2940, and does not emit invented operating scores.

### REQ-CODE-2963: DCCD Structured-Repair Protocol Manifest

The repository shall provide an Exp 2963 deterministic DCCD structured-repair
protocol manifest for the downstream live SOTA replication. The builder MUST:

- require checked-in Exp 2950, Exp 2951, Exp 2952, and Exp 2953 artifacts before
  marking the protocol ready, and block honestly when any prerequisite is absent
  or malformed;
- extract the Exp 2950 failure taxonomy and deterministic repair checks, the
  Exp 2951 structured candidate schema and backend status, the Exp 2952
  flagged repair delta and false-accept evidence, and the Exp 2953 conservative
  verifier threshold policy;
- define a DCCD flow that separates unconstrained semantic draft generation,
  taxonomy-conditioned repair, constrained manifest emission, parser/static/test
  checks, verifier-threshold checking, and a false-accept audit;
- define a downstream replication cohort with at least 20 planned tasks, fixed
  seeds, and separate accounting for pass@1/pass@k, syntax/schema failures,
  test failures, and verifier-only accepts;
- record local structured-output backends to check downstream, including
  `llguidance`, llama.cpp grammar, and deterministic JSON-schema validation
  fallback;
- include downstream live-use model specs for
  `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`, while keeping legacy small models
  smoke-only; and
- write `results/experiment_2963_dccd_repair_protocol_manifest_v1.json` with
  `honest_verdict`, `dccd_repair_protocol_ready`, `source_artifacts`,
  `model_specs`, `legacy_models_only_for_smoke`, `n_tasks_planned_min`,
  `fixed_seed_plan`, `dccd_steps`, `structured_backends_to_check`,
  `deterministic_acceptance_checks`, `false_accept_audit_plan`,
  `downstream_gate`, `inference_substrate="aggregation_from_upstream_artifacts"`,
  and measured `duration_s`.

### SCENARIO-CODE-2963: DCCD Protocol Is Pre-Registered Without A Pass-Rate Claim

Given the checked-in Exp 2950, Exp 2951, Exp 2952, and Exp 2953 artifacts are
present,
When Exp 2963 builds the DCCD protocol manifest,
Then the artifact records a ready protocol that cites the upstream taxonomy,
schema, repair-delta warning, threshold policy, false-accept constraints,
downstream SOTA GGUF model specs, fixed seeds, and deterministic acceptance
gates while preserving `inference_substrate="aggregation_from_upstream_artifacts"`
and explicitly making no claim that DCCD improves pass@1 or pass@k.

### REQ-CODE-2964: SOTA DCCD Repair Replication

The repository shall provide an Exp 2964 bounded live replication for DCCD
structured code repair. The replication MUST:

- check the dual RTX 3090 host, local llama.cpp runtime, runsc sandbox,
  cached mandated headline GGUF availability, checked-in Exp 2946 failed
  candidates, checked-in Exp 2952/2953 references, and
  `results/experiment_2963_dccd_repair_protocol_manifest_v1.json` with
  `dccd_repair_protocol_ready=true` before attempting live generation;
- select at least 20 failed or low-scoring Exp 2946 code candidates when
  preconditions pass, preserving stable task IDs, candidate seeds, raw-response
  references, and original failure categories;
- run baseline no-taxonomy repair, Exp 2952 taxonomy-guided repair, and DCCD
  structured repair under the same sample budget using mandated local SOTA GGUF
  models for headline measurements, while legacy small models remain smoke-only;
- validate every candidate through structured schema validation, Python parser
  and static checks, available MBPP/HumanEval task tests, the Exp 2953
  verifier threshold, and a false-accept audit that does not count verifier-only
  accepts as pass@1/pass@k;
- set `dccd_repair_replication_clean=true` only when `n_tasks>=20`,
  `pass_at_1_delta > 0` or `syntax_failure_rate_delta < 0`, and
  `false_accept_delta <= 0`; and
- write `results/experiment_2964_sota_dccd_repair_replication_v1.json` with
  `honest_verdict`, `inference_substrate="live_llm_inference"`,
  `preconditions_checked`, `model_specs`, `headline_models_used`,
  `legacy_models_only_for_smoke`, `n_tasks`, `baseline_pass_at_1`,
  `taxonomy_repair_pass_at_1`, `dccd_repair_pass_at_1`,
  `pass_at_1_delta`, `baseline_pass_at_k`, `dccd_repair_pass_at_k`,
  `pass_at_k_delta`, `syntax_failure_rate_delta`,
  `schema_failure_rate_delta`, `false_accept_delta`,
  `dccd_repair_replication_clean`, `candidate_manifest_sha256`,
  `reproducibility_checksum`, and measured `duration_s`.

### SCENARIO-CODE-2964: DCCD Replication Promotes Only Clean Live Results

Given Exp 2963 is ready and at least one mandated local SOTA GGUF can run on
the dual RTX 3090 host,
When Exp 2964 runs equal-budget baseline, taxonomy-guided, and DCCD structured
repair across at least 20 failed or low-scoring Exp 2946 candidates,
Then the artifact records all selected task IDs and failure labels, validates
every generated candidate through schema/parser/static/tests/verifier gates,
computes DCCD-minus-baseline deltas, and keeps
`dccd_repair_replication_clean=false` unless the pre-registered sample-size,
improvement, and false-accept gates all clear.

### REQ-CODE-2990: Verifier-Backed Hard-Code Stress Manifest

The repository shall provide a deterministic hard-code stress manifest for the
next SOTA repair rerun without running a headline LLM repair experiment. The
builder MUST:

- write a JSONL manifest such as `datasets/repair_hard/manifest_v1.jsonl`
  containing 20-40 executable code-repair items;
- include, for every item, prompt provenance, an executable test suite, a
  baseline incorrect candidate, a reference solution or expected behavior, and
  edge-case rationale;
- execute each item's verifier tests against both the baseline candidate and
  the reference solution, accepting only items where the baseline fails at
  least one deterministic test and the reference passes every test;
- reject flaky, nondeterministic, or non-distinguishing items instead of
  including them in the ready set;
- record replayable verifier transcripts or pass/fail hashes for every item;
  and
- write
  `results/experiment_2990_verifier_backed_hard_code_stress_manifest_v1.json`
  with `hard_code_stress_set_ready`, `manifest_path`, `n_items`,
  `all_items_have_tests`, `all_baseline_candidates_fail`,
  `all_reference_solutions_pass`, `flaky_items`,
  `verifier_transcript_paths`, `hard_generation_sources`, `honest_verdict`,
  validation commands, and measured `duration_s`.

### SCENARIO-CODE-2990: Hard Stress Items Are Independently Executable

Given a curated or synthesized set of code-repair stress items,
When the Exp 2990 manifest builder validates the set,
Then it writes the manifest and terminal result artifact only when every
included item has executable deterministic tests, every baseline candidate
fails at least one verifier assertion, every reference solution passes, no
flaky item remains in the ready set, and the artifact points to replayable
transcripts for the verifier evidence.

### REQ-CODE-2991: Gated SOTA Intent-Preserving Repair on the Hard Set

The repository shall provide an Exp 2991 rerun harness that applies the
intent-preserving, trace-aware repair protocol to the Exp 2990 hard-code stress
manifest using mandated local headline GGUF models. The harness MUST:

- require
  `results/experiment_2989_sota_gguf_cache_provenance_preflight_v1.json` with
  `sota_headline_ready=true` and
  `results/experiment_2990_verifier_backed_hard_code_stress_manifest_v1.json`
  with `hard_code_stress_set_ready=true` before any live repair generation;
- validate the hard manifest SHA, item count, deterministic tests, baseline
  failures, and reference-solution passes before selecting repair tasks;
- call the repo-local SOTA cache helper, accepting only mandated headline GGUF
  model IDs for headline measurements while keeping tiny legacy models
  smoke-only;
- execute baseline verifier tests before LLM repair, then preserve draft
  intent, runtime trace, failing assertion evidence, final patch, transcript
  path, candidate patch path, and verifier log path for every generated repair
  candidate;
- compute pass@1/pass@k, schema-failure, syntax-failure, verifier
  false-accept, and trace-coverage deltas against the baseline candidates,
  keeping small-model smoke evidence separate from headline metrics;
- set `repair_rerun_clean=true` only when `headline_result=true`,
  `n_tasks>=20`, at least one mandated headline model produced live results,
  pass@1 improves, pass@k does not regress, schema and syntax failure rates do
  not rise, verifier false accepts do not rise, and trace coverage is complete
  enough for trace-aware repair; and
- write
  `results/experiment_2991_gated_sota_intent_preserving_repair_hard_set_v1.json`
  with `repair_rerun_clean`, `headline_result`, `n_tasks`, `model_specs`,
  `headline_models_used`, `pass_at_1_delta`, `pass_at_k_delta`,
  `schema_failure_rate_delta`, `syntax_failure_rate_delta`,
  `verifier_false_accept_delta`, `trace_coverage`, `transcript_paths`,
  `inference_substrate`, and `honest_verdict`.

### SCENARIO-CODE-2991: Hard-Set Repair Promotion Is Machine-Gated

Given Exp 2989 proves at least one mandated headline GGUF can produce a live
local transcript and Exp 2990 proves the hard-code stress manifest is ready,
When Exp 2991 runs equal-budget baseline and intent-preserving repair on the
hard set,
Then the artifact reports headline evidence only from actual headline GGUF
models, records replayable transcripts, patches, verifier logs, and metric
deltas against the failing baselines, and marks the result clean only when the
sample-size, pass-rate, schema, syntax, false-accept, and trace gates all pass.

### REQ-CODE-3002: Metamorphic Hard-Set Repair Oracle Audit

The repository shall provide an Exp 3002 deterministic audit that strengthens
the Exp 2990/2991 hard-code stress oracle without running live LLM repair. The
audit MUST:

- load `datasets/repair_hard/manifest_v1.jsonl`, or reconstruct an inspectable
  hard-set manifest from Exp 2991 artifacts and checked-in hard-set fixtures
  when the original manifest has moved;
- generate a bounded inspectable metamorphic manifest containing only
  relation-preserving variants, including conservative alpha-renaming,
  equivalent boundary cases, input permutation where order is irrelevant, and
  oracle-preserving refactors;
- reject and report generated variants when executable validation shows the
  reference solution no longer satisfies the intended semantics or the probe is
  tautological;
- replay executable validators on original and metamorphic cases against the
  reference solution, baseline candidate, and any cached Exp 2991 repair
  candidates found in local artifacts;
- produce false-accept probes that pass the original hard-set tests with a
  visible-test-overfit candidate but fail the independent metamorphic verifier,
  plus tautology probes that demonstrate vacuous tests would be rejected; and
- write
  `results/experiment_3002_metamorphic_repair_oracle_audit_v1.json` with
  `metamorphic_oracle_ready`, `hard_set_manifest_path`,
  `metamorphic_manifest_path`, `n_source_items`, `n_metamorphic_variants`,
  `relation_types`, `false_accept_probe_ready`, `tautology_probe_ready`,
  `rejected_variants`, `verifier_transcript_paths`, and `honest_verdict`.

### SCENARIO-CODE-3002: Metamorphic Oracle Evidence Is Replayable

Given the Exp 2990 hard-set manifest and any cached Exp 2991 repair patches,
When Exp 3002 runs the deterministic metamorphic oracle audit,
Then every accepted variant passes the reference solution under executable
validation, rejected variants remain visible with reasons, original and variant
validators write replayable transcripts, false-accept and tautology probes are
ready, and the terminal artifact marks downstream repair promotion as ready,
flagged, or blocked without invoking live LLM repair.

### REQ-CODE-3003: Gated SOTA Repair Rerun With Metamorphic False-Accept Evidence

The repository shall provide an Exp 3003 claim-repair harness that reruns the
hard-set repair evidence against the Exp 3002 metamorphic oracle before any
repair row can be promoted. The harness MUST:

- require
  `results/experiment_3001_sota_gguf_cache_carry_forward_checksum_refresh_v1.json`
  with `sota_headline_ready=true` and
  `results/experiment_3002_metamorphic_repair_oracle_audit_v1.json` with
  `metamorphic_oracle_ready=true` before evaluating repair promotion;
- verify CUDA/cache status, headline GGUF checksum availability, hard-set
  integrity, and metamorphic manifest integrity as explicit preconditions;
- load the original hard-set manifest and the metamorphic manifest, then run
  deterministic baseline validators before evaluating SOTA repair candidates;
- preserve headline model identity, prompt, draft intent, failing trace, final
  patch, generation duration, source transcript hash, candidate patch path, and
  verifier log path for every evaluated repair candidate;
- compute pass@1/pass@k, syntax/schema failure, verifier false-accept, and
  tautology deltas against the baseline and include comparable Exp 2991 deltas
  when available, while keeping smoke-only model evidence separate from
  headline evidence;
- set `repair_rerun_clean=false` when false accepts rise, tautology probes pass
  vacuously, schema or syntax failures rise, `n_tasks` is too small, no
  metamorphic variants are available, or only smoke models ran; and
- write
  `results/experiment_3003_gated_sota_repair_metamorphic_false_accept_rerun_v1.json`
  with `repair_rerun_clean`, `headline_result`, `preconditions_checked`,
  `n_tasks`, `n_metamorphic_variants`, `model_specs`, `headline_models_used`,
  `model_checksums`, `pass_at_1_delta`, `pass_at_k_delta`,
  `false_accept_delta`, `tautology_gate_clean`,
  `syntax_failure_rate_delta`, `live_transcript_paths`,
  `verifier_log_paths`, and `honest_verdict`.

### SCENARIO-CODE-3003: Metamorphic Repair Promotion Is Terminally Gated

Given Exp 3001 reports a live headline SOTA GGUF transcript and Exp 3002
reports a ready metamorphic oracle,
When Exp 3003 evaluates headline repair candidates across the original hard
set and every accepted metamorphic variant,
Then the artifact records replayable live transcripts, candidate patches,
verifier logs, model checksums, baseline and repair metrics, and marks the
result `clean`, `flagged`, or `blocked` without allowing smoke-only, vacuous,
syntax-regressed, schema-regressed, or false-accept-regressed evidence to
promote.

### REQ-CODE-3014: Cached Repair-Failure Taxonomy For Exp 3003

The repository shall provide an Exp 3014 deterministic repair-failure taxonomy
for Exp 3003 candidates and validators without running live LLM inference. The
taxonomy builder MUST:

- load the Exp 3002 oracle artifact, Exp 3002 metamorphic manifest, Exp 3003
  repair artifact, candidate transcripts, verifier logs, cached candidate patch
  files, and the repair-hard manifest when present;
- replay deterministic parser/schema checks and executable original plus
  metamorphic validators over cached candidates only;
- classify candidate and validator failures into root causes including
  prompt-format pressure, parser/schema mismatch, invalid patch shape, oracle
  ambiguity, intent drift, false accept, and tautology;
- count syntax, schema, false-accept, tautology, and intent-preservation
  failures explicitly so downstream controllers can gate on diagnosed failure
  modes rather than only aggregate Exp 3003 deltas;
- write an inspectable taxonomy table under `results/` or
  `datasets/repair_hard/` with per-candidate and per-validator evidence rows;
- produce a minimal Exp 3015/3016 acceptance-rule plan that rejects diagnosed
  unsafe modes before another live repair rerun; and
- write
  `results/experiment_3014_repair_syntax_schema_failure_taxonomy_v1.json`
  with `repair_failure_taxonomy_ready`, `taxonomy_table_path`,
  `n_cached_candidates_audited`, `syntax_failure_count`,
  `schema_failure_count`, `false_accept_count`, `tautology_failure_count`,
  `intent_drift_count`, `recommended_acceptance_rules`,
  `halluguard_ntk_claim_made=false`, and `honest_verdict`.

### SCENARIO-CODE-3014: Cached Failure Classifications Are Inspectable

Given Exp 3002 and Exp 3003 artifacts with cached candidate patches,
transcripts, and verifier logs,
When Exp 3014 runs the taxonomy builder,
Then it does not invoke live inference, replays parser/schema checks and
validators over the cached patches, writes an inspectable taxonomy table, emits
the required counts and acceptance rules, and uses any data-driven versus
reasoning-driven taxonomy language only as plain taxonomy language without
claiming an NTK or HalluGuard implementation.

### REQ-CODE-3015: Offline Cactus-Style Repair Acceptance Controller

The repository shall provide an Exp 3015 offline repair-candidate acceptance
controller over cached Exp 3003 candidates, Exp 3014 taxonomy rows, verifier
logs, and Exp 3002 metamorphic oracle evidence. The controller MUST:

- load cached candidates, taxonomy rows, verifier outcomes, the metamorphic
  oracle artifact and manifest, and optional telemetry artifacts when present;
- derive deterministic candidate features for schema validity, syntax validity,
  false-accept probe status, tautology probe status, intent-drift class,
  original-test pass/fail, metamorphic-variant pass/fail, and optional
  telemetry availability without invoking a live LLM or black-box judge;
- grid-search a tiny transparent accept/reject rule over those features and
  record the selected rule in an inspectable controller config;
- evaluate the selected rule against accept-all and conservative accept-only-
  clean policies, reporting offline deltas for false accepts, syntax failures,
  schema failures, and pass@1 against accept-all;
- reject candidates whose apparent usefulness is explained only by increased
  syntax/schema failures, false accepts, or tautology exposure; and
- write
  `results/experiment_3015_cactus_style_repair_acceptance_controller_v1.json`
  with `acceptance_controller_ready`, `controller_config_path`,
  `n_candidates_evaluated`, `false_accept_delta_offline`,
  `syntax_failure_delta_offline`, `schema_failure_delta_offline`,
  `pass_at_1_delta_offline`, `rejected_candidate_table_path`,
  `llm_judge_used=false`, and `honest_verdict`.

`acceptance_controller_ready` shall be true only when the selected controller
is inspectable, evaluates at least one cached candidate, does not increase
false accepts, syntax failures, or schema failures against accept-all, and has
usable tautology-probe evidence from Exp 3002.

### SCENARIO-CODE-3015: Deterministic Cached Evidence Selects An Inspectable Rule

Given Exp 3002, Exp 3003, and Exp 3014 artifacts are present with cached
candidate rows and taxonomy evidence,
When Exp 3015 builds the offline acceptance controller,
Then it writes an inspectable controller config and rejected-candidate table,
marks `llm_judge_used=false`, reports sample size and offline deltas, and only
marks the controller ready when the selected transparent rule rejects unsafe
syntax/schema, false-accept, tautology, or intent-drift evidence without
promoting a regression over accept-all.

### REQ-CODE-3016: SOTA Repair Rerun With Acceptance Controller

The repository shall provide an Exp 3016 local SOTA GGUF repair rerun that uses
the Exp 3015 acceptance controller as a machine gate over hard-set and
metamorphic repair candidates. The rerun MUST:

- require
  `results/experiment_3013_sota_gguf_logprob_telemetry_preflight_v1.json`
  with `sota_logprob_ready=true` and
  `results/experiment_3015_cactus_style_repair_acceptance_controller_v1.json`
  with `acceptance_controller_ready=true` before evaluating repair promotion;
- verify CUDA/cache status, model checksum availability, controller-config
  availability, hard-set integrity, and metamorphic-manifest integrity as
  explicit preconditions, writing a terminal blocked artifact when any
  precondition fails;
- load the original hard-set manifest, metamorphic manifest, and acceptance
  controller config, then run deterministic baseline validators before
  evaluating live headline repair candidates;
- generate or replay repair candidates from at least one mandated headline GGUF
  model while preserving prompt, draft intent, failing trace, final patch,
  generation duration, transcript hash, candidate patch path, verifier log path,
  and available telemetry;
- apply the acceptance controller before final promotion and execute
  deterministic original plus metamorphic validators for both accepted and
  rejected candidates;
- compute pass@1/pass@k, syntax/schema failure, false-accept, and tautology
  deltas against baseline, Exp 3003, and accept-all where comparable, keeping
  smoke-only model evidence separate from headline evidence; and
- write
  `results/experiment_3016_sota_repair_rerun_with_acceptance_controller_v1.json`
  with `repair_controller_clean`, `headline_result`, `preconditions_checked`,
  `n_tasks`, `n_metamorphic_variants`, `model_specs`,
  `headline_models_used`, `model_checksums`,
  `acceptance_controller_config_path`, `pass_at_1_delta`,
  `pass_at_k_delta`, `false_accept_delta`, `tautology_gate_clean`,
  `syntax_failure_rate_delta`, `schema_failure_rate_delta`,
  `live_transcript_paths`, `verifier_log_paths`, `inference_substrate`,
  `duration_s`, and `honest_verdict`.

`repair_controller_clean` shall be false when false accepts rise, tautology
probes pass vacuously, syntax/schema failures rise, `n_tasks` is below the
headline minimum, no accepted metamorphic variants are available, no mandated
headline model produced candidate evidence, or only smoke-only models ran.

### SCENARIO-CODE-3016: Acceptance-Controlled Repair Promotion Is Replayable

Given Exp 3013 reports ready SOTA telemetry and Exp 3015 reports a ready
transparent acceptance controller,
When Exp 3016 reruns hard-set repair over original and metamorphic validators,
Then every candidate is deterministically tested before and after controller
classification, accepted and rejected rows keep replayable patches,
transcripts, verifier logs, telemetry summaries, and controller reasons, and
the terminal artifact marks the result complete, complete-flagged, or blocked
without allowing smoke-only, vacuous, syntax-regressed, schema-regressed, or
false-accept-regressed evidence to promote.

### REQ-CODE-3028: Clean-Methodology SOTA Repair Evidence Rerun

The repository shall provide an Exp 3028 clean-methodology repair evidence
builder that writes
`results/experiment_3028_sota_repair_clean_methodology_rerun_v2.json`. The
builder MUST:

- read Exp 3013, Exp 3015, Exp 3016, Exp 3027, the repair-hard manifest, and
  metamorphic oracle evidence before deciding whether existing live transcripts
  are reconstructable or a small live panel is required;
- record CUDA/GPU availability, free VRAM, repo commit, Python environment,
  GGUF cache paths, model checksum feasibility, and the Exp 3027
  `repair_rerun_required` decision before any model load;
- preserve draft intent, failing trace, candidate patch path and hash, final
  patch text/hash, validator output, generation duration, transcript path/hash,
  random seed, reproducibility checksum, and headline GGUF model specs for every
  reconstructed or live repair task;
- apply the Exp 3015 acceptance controller with MARCH-style information
  asymmetry, keeping repair candidate collection separate from deterministic
  syntax/schema, original-test, metamorphic, false-accept, tautology, and intent
  checks;
- reject candidates whose repair gain depends on schema drift, syntax drift,
  false accepts, tautology exposure, missing checker evidence, legacy smoke-only
  models, or non-headline model provenance; and
- write `clean_repair_rerun_ready`, `repair_controller_clean`,
  `clean_repair_claim_promotable_candidate`, `n_tasks`,
  `n_live_transcripts`, `model_specs`, `legacy_smoke_only_used`,
  `pass_at_1_delta`, `pass_at_k_delta`, `syntax_failure_rate_delta`,
  `schema_failure_rate_delta`, `false_accept_delta`, `tautology_gate_clean`,
  `intent_drift_count`, `reproducibility_checksum`, `inference_substrate`, and
  `honest_verdict`.

`clean_repair_rerun_ready` shall be true only when at least one mandated
headline GGUF supplied the repair candidates, every accepted candidate has
independent checker evidence, pass-rate deltas are positive or non-negative as
appropriate, syntax/schema/false-accept deltas do not rise, tautology checks
are non-vacuous, and intent drift is zero. If neither complete transcript
reconstruction nor a local headline GGUF path is available, the artifact SHALL
block with `blocked_sota_headline_model_unavailable`.

### SCENARIO-CODE-3028: Clean Reconstruction Produces Promotion Candidate

Given Exp 3027 reports the Exp 3016 top-level metadata gap and the Exp 3016
candidate rows, transcripts, patch files, verifier logs, model checksums, and
seeds are otherwise present,
When Exp 3028 reconstructs evidence with the Exp 3015 controller and independent
deterministic checks,
Then it writes the v2 artifact with all required methodology fields, records
the Exp 3027 rerun decision, counts live transcripts separately from
reconstruction, uses only mandated headline GGUF model specs for headline
evidence, reports clean repair deltas, sets
`clean_repair_rerun_ready=true`, and emits an honest verdict starting with
`complete:`.

### REQ-CODE-3031: Tiny DCCD Structured Repair Panel

The repository shall provide an Exp 3031 tiny DCCD-style structured repair
panel over repair-hard cases. The panel builder MUST:

- record GPU status, model-cache status, repo commit, selected headline GGUF,
  and repair-hard fixture availability before any model load;
- block honestly with an `honest_verdict` starting with
  `blocked_sota_headline_model_unavailable` when no mandated headline GGUF can
  be loaded for live generation;
- select a small fixed repair-hard panel with known failing baseline traces and
  executable validators, preserving item ids, prompts, expected behavior,
  failing test ids, and validator evidence;
- capture an unconstrained semantic draft from at least one mandated headline
  GGUF before applying schema or syntax constraints;
- compare three conditions: unconstrained draft repair, acceptance-only
  constrained acceptance using the Exp 3015 controller rule, and a simple
  draft-conditioned constrained repair pass that emits a schema-bearing patch
  candidate conditioned on the draft;
- measure strict validity, pass rate, syntax failures, schema failures, false
  accepts, and intent drift for every condition, with explicit deltas for
  draft-conditioned repair versus the acceptance-only baseline;
- keep legacy small-model evidence smoke-only and exclude it from headline
  panel metrics; and
- write `results/experiment_3031_dccd_structured_repair_panel_v1.json` with
  `dccd_panel_ready`, `n_cases`, `model_specs`, `legacy_smoke_only_used`,
  `baseline_acceptance_metrics`, `dccd_metrics`, `intent_drift_delta`,
  `false_accept_delta`, `changed_files`, `tests_run`, `inference_substrate`,
  and `honest_verdict`.

`dccd_panel_ready` shall be true only when at least one mandated headline GGUF
generated live drafts for the fixed panel, every selected case has executable
validators, the acceptance-only baseline and draft-conditioned condition are
both measured, and false-accept delta does not increase.

### SCENARIO-CODE-3031: Draft-Conditioned Panel Compares Against Acceptance Baseline

Given repair-hard fixtures, Exp 3015 controller evidence, and at least one
loadable mandated headline GGUF are available,
When Exp 3031 runs the fixed DCCD panel,
Then the artifact records preconditions, the selected headline model, live
draft transcript evidence, per-condition validity metrics, acceptance-only
baseline metrics, draft-conditioned metrics, intent-drift and false-accept
deltas, changed files, tests run, and an honest verdict starting with
`complete:` or `complete_flagged:`.

### REQ-CODE-2965: BEAVER-Style Structured-Repair Certificate Audit

The repository shall provide an Exp 2965 deterministic certificate audit for
structured code-repair manifests. The audit MUST:

- require `results/experiment_2963_dccd_repair_protocol_manifest_v1.json` with
  `dccd_repair_protocol_ready=true` before marking the certificate ready;
- consume the Exp 2951 structured candidate schema, Exp 2952 .278 repair
  candidate manifests when present, and the Exp 2953 verifier threshold policy;
- define prefix-closed acceptance constraints for schema validity, code-block
  completeness, import allow-list checks, function-name preservation when an
  expected entry point is available, parser validity, test/verifier status
  fields, and false-accept accounting;
- add per-candidate certificate fields `explored_prefix_count`,
  `blocked_prefix_count`, `schema_valid`, `parser_valid`,
  `verifier_threshold_used`, and `false_accept_audit`;
- validate the certificate logic on at least five synthetic candidate records
  and every available .278 repair candidate manifest from Exp 2952;
- record local `llguidance` and llama.cpp grammar support while retaining a
  deterministic fallback; and
- write `results/experiment_2965_beaver_style_repair_certificate_v1.json` with
  `honest_verdict`, `beaver_style_certificate_ready`,
  `full_beaver_claim=false`, `source_artifacts`,
  `certificate_schema_version`, `prefix_closed_constraints`,
  `validation_fixture_count`, `validation_fixture_passed`,
  `local_backends_checked`, `llguidance_available`,
  `llama_cpp_grammar_available`, `false_accept_audit_fields`,
  `files_changed`, `inference_substrate="deterministic_wiring"`, and measured
  `duration_s`.

The audit MUST NOT claim full BEAVER probability bounds unless such bounds are
actually computed. The Exp 2965 artifact is a bounded certificate over the
manifest and deterministic repair gates, not a probabilistic verifier proof.

### SCENARIO-CODE-2965: Certificate Audit Rejects Verifier-Only Accept Paths

Given Exp 2963 is ready and Exp 2951/2952/2953 artifacts are available,
When Exp 2965 audits synthetic and checked-in .278 structured repair manifests,
Then the artifact records ready BEAVER-style certificate fields, reports
grammar-backend availability, keeps `full_beaver_claim=false`, validates at
least five synthetic fixtures, audits all available Exp 2952 candidate
manifests, and flags any verifier-threshold accept that lacks schema, parser,
allowed-import, preserved-function, and available-test evidence as a
false-accept path rather than as a pass.

### REQ-CODE-2925: Exp 2911 Taxonomy Provenance Corrigendum v2

The repository shall provide an Exp 2925 deterministic provenance corrigendum
for the Exp 2911 code-hallucination taxonomy artifact. The corrigendum MUST:

- block with `honest_verdict="blocked_upstream_artifact_missing"` when either
  `results/experiment_2910_sota_code_generation_corrigendum_v2.json` or
  `results/experiment_2911_code_hallucination_taxonomy_verifier_v1.json` is
  absent or malformed;
- verify that Exp 2911 candidate counts, per-candidate label rows, taxonomy
  rate denominators, and Exp 2910 per-task candidate totals all match the Exp
  2910 raw candidate inventory;
- preserve Exp 2911 taxonomy rates and per-category counts without rerunning
  local GGUF inference;
- compute source checksums for Exp 2910, Exp 2911, and any raw-response
  manifest explicitly referenced by Exp 2911;
- add top-level deterministic-verifier provenance fields including
  `random_seed=2925`, `reproducibility_checksum`, `upstream_model_specs`,
  `upstream_models_used`, `no_new_llm_call=true`, and
  `deterministic_verifier_no_new_llm_call=true`; and
- rerun the local adversarial artifact audit when available and record the
  exact command outcome in `adversarial_audit_rerun`.

### SCENARIO-CODE-2925: Provenance Corrigendum Re-Emits The Taxonomy Row

Given Exp 2910 and Exp 2911 artifacts are present,
When the Exp 2925 corrigendum builds
`results/experiment_2925_code_hallucination_taxonomy_provenance_corrigendum_v2.json`,
Then the artifact contains the required provenance fields, source checksums,
candidate-count validation, original taxonomy rates and counts, audit rerun
metadata, `inference_substrate="deterministic_verifier"`, run date `20260523`,
and no claim that new local GGUF inference was performed.

### REQ-CODE-2890: MBPP/HumanEval Structural Dependency Verifier

The repository shall provide a deterministic structural-dependency verifier for
MBPP/HumanEval-like code rows. The verifier MUST NOT run live generation or make
headline pass@k/AUROC claims. It MUST:

- build serializable contracts for rows selected by Exp 2879 and, when present,
  Exp 2889, including required inputs, function signature, dependency edges,
  forbidden imports or side effects, test prerequisites, and output obligations;
- verify canonical/reference code and available generated outputs with static
  AST-based checks that localize violations to a code line, contract field, and
  violation type where possible;
- record unsupported reasons instead of inferring structure when source code is
  absent or cannot be parsed;
- aggregate violation counts and localization examples for downstream matrix v7
  reuse; and
- write `results/experiment_2890_code_structural_dependency_verifier_v1.json`
  with `honest_verdict`, `structural_dependency_verifier_ready`,
  `source_artifacts`, `contract_schema_version`, `n_contracts_built`,
  `n_rows_verified`, `violation_types`, `localization_examples`,
  `generated_outputs_consumed`, `tests_run`, `field_principles`,
  `run_date="20260523"`, and measured `duration_s`.

### SCENARIO-CODE-2890: Static Contracts Localize Generated-Code Failures

Given Exp 2879 selected MBPP/HumanEval canonical rows and Exp 2889 may provide
generated outputs for the same manifest rows,
When the structural dependency verifier builds contracts and statically checks
the canonical/reference and generated code candidates,
Then canonical/reference rows pass the structural contract, generated or
malformed rows surface deterministic violation types such as parse errors,
missing function definitions, wrong signatures, forbidden imports, or missing
test prerequisites, unsupported rows are labeled explicitly, and no live
generation or headline metric is claimed.

## Implementation Status

| Requirement | Status |
|-------------|--------|
| REQ-CODE-001 | Implemented |
| REQ-CODE-002 | Implemented |
| REQ-CODE-003 | Implemented |
| REQ-CODE-004 | Implemented |
| REQ-CODE-005 | Implemented |
| REQ-CODE-006 | Implemented |
| REQ-CODE-007 | Implemented |
| REQ-CODE-008 | Implemented |
| REQ-CODE-009 | Implemented |
| REQ-CODE-010 | Implemented |
| REQ-CODE-011 | Implemented |
| REQ-CODE-012 | Implemented |
| REQ-CODE-013 | Implemented |
| REQ-CODE-014 | Implemented |
| REQ-CODE-015 | Implemented |
| REQ-CODE-016 | Implemented |
| REQ-CODE-017 | Implemented |
| REQ-CODE-018 | Implemented |
| REQ-CODE-019 | Implemented |
| REQ-CODE-020 | Implemented |
| REQ-CODE-021 | Implemented |
| REQ-CODE-022 | Implemented |
| REQ-CODE-023 | Implemented |
| REQ-CODE-024 | Implemented |
| REQ-CODE-025 | Implemented |
| REQ-CODE-026 | Implemented |
| REQ-CODE-027 | Implemented |
| REQ-CODE-028 | Not Started |
| REQ-CODE-029 | Not Started |
| REQ-CODE-030 | Not Started |
| REQ-CODE-028 | Implemented |
| REQ-CODE-029 | Implemented |
| REQ-CODE-030 | Implemented |
| REQ-CODE-031 | Implemented |
| REQ-CODE-032 | Implemented |
| REQ-CODE-033 | Implemented |
| REQ-CODE-034 | Implemented (`python/carnot/verifiers/dsl.py`) |
| REQ-CODE-2879 | Implemented |
| REQ-CODE-2889 | Implemented |
| REQ-CODE-2905 | Implemented (`python/carnot/eval/sota_code_generation_bounded_budget_expansion.py`) |
| REQ-CODE-2910 | Implemented (`python/carnot/eval/sota_code_generation_corrigendum.py`) |
| REQ-CODE-2911 | Implemented (`python/carnot/eval/code_hallucination_taxonomy_verifier.py`) |
| REQ-CODE-5445 | Implemented (`python/carnot/experiment_5445_static_ast_kb_witness_constraints_v495.py`) |
| REQ-CODE-2925 | Implemented (`python/carnot/eval/code_hallucination_taxonomy_provenance_corrigendum.py`) |
| REQ-CODE-2946 | Implemented (`python/carnot/eval/sota_code_generation_continuation.py`) |
| REQ-CODE-2950 | Implemented (`python/carnot/eval/code_taxonomy_repair_prompt_manifest.py`) |
| REQ-CODE-2951 | Implemented (`python/carnot/eval/structured_candidate_manifest_adapter.py`) |
| REQ-CODE-2952 | Implemented (`python/carnot/eval/sota_taxonomy_guided_code_repair_eval.py`) |
| REQ-CODE-2953 | Implemented (`python/carnot/eval/code_verifier_threshold_policy.py`) |
| REQ-CODE-2963 | Implemented (`python/carnot/eval/dccd_repair_protocol_manifest.py`) |
| REQ-CODE-2964 | Implemented (`python/carnot/eval/sota_dccd_repair_replication.py`) |
| REQ-CODE-2965 | Implemented (`python/carnot/eval/beaver_style_repair_certificate.py`) |
| REQ-CODE-2990 | Implemented (`python/carnot/eval/hard_code_stress_manifest.py`) |
| REQ-CODE-2991 | Implemented (`python/carnot/eval/gated_sota_intent_preserving_repair_hard_set.py`) |
| REQ-CODE-3002 | Planned (`python/carnot/eval/metamorphic_repair_oracle_audit.py`) |
| REQ-CODE-3003 | Implemented (`python/carnot/eval/gated_sota_repair_metamorphic_false_accept_rerun.py`) |
| REQ-CODE-3014 | Planned (`python/carnot/eval/repair_failure_taxonomy.py`) |
| REQ-CODE-3015 | Implemented (`python/carnot/eval/repair_acceptance_controller.py`) |
| REQ-CODE-2890 | Implemented (`python/carnot/verify/code_structural_dependency_verifier.py`) |
| REQ-REPAIR-020 | Implemented |
| REQ-REPAIR-021 | Implemented |
| REQ-REPAIR-022 | Implemented |
| REQ-REPAIR-023 | Implemented |
| REQ-REPAIR-024 | Implemented |
| REQ-REPAIR-025 | Implemented |

### REQ-EXTRACT-035: ASTKnowledgeVerifier Must Parse Python Code to AST and Validate API Calls

The ASTKnowledgeVerifier MUST parse Python code to an Abstract Syntax Tree (AST) and
validate all attribute-access API calls (obj.attr patterns) against a KnowledgeBase
built by introspecting the actual Python module via import + dir() + inspect.getmembers().
Precision MUST be 1.0 — zero false positives are tolerated.  Every flagged violation
MUST correspond to an attribute that genuinely does not exist in the introspected module.

Rationale: arXiv 2601.19106 shows 100% precision, 87.6% recall on 200 Python snippets,
with 77% auto-correction.  Execution-free detection is safe for untrusted code.

Spec: REQ-EXTRACT-035, SCENARIO-EXTRACT-070, SCENARIO-EXTRACT-071

### REQ-EXTRACT-036: ASTKnowledgeVerifier Tier 0d Integration

ASTKnowledgeVerifier, when wired as a Tier 0d pre-filter, MUST:
- Return violation_detected=True only when the call target attr is absent from the
  KnowledgeBase's introspected module dict.
- Never flag attributes that exist in the module (precision=1.0 invariant).
- Allow downstream callers to skip expensive Ising verification on high-confidence
  Knowledge Conflicting Hallucinations (KCHs), since 100% precision means every
  detection is a real error.

Spec: REQ-EXTRACT-036, SCENARIO-EXTRACT-072

### SCENARIO-EXTRACT-070: Correct API Call Is Not Flagged (No False Positive)

Given code "import json; data = json.loads(text)" where json.loads exists in the
Python standard library, when ASTKnowledgeVerifier.verify() runs, then the returned
violations list MUST be empty — no false positive is generated.

### SCENARIO-EXTRACT-071: Hallucinated API Call Is Flagged as Violation

Given code "import json; data = json.parse(text)" where json.parse does NOT exist
in the Python standard library (the real method is json.loads), when
ASTKnowledgeVerifier.verify() runs, then exactly one ASTKnowledgeViolation is returned
with module="json", attr="parse", violation_type="missing_attr".

### SCENARIO-EXTRACT-072: Tier 0d Returns violation_detected=True for KCH

Given a snippet that calls os.read_file(path) where os.read_file does not exist,
when ASTKnowledgeVerifier.verify() is called and violations are non-empty, then
the caller concludes violation_detected=True and skips Ising verification,
because 100% precision guarantees this is a real error.

### REQ-CODE-035: Ising-Guided Fuzzing for Code Verification
The system shall verify code execution via Ising-guided fuzzing on live GPU (Exp 1999):
- Run 50 HumanEval questions generating code.
- Execute structural CodeExtractor instrumentation.
- Record baseline vs. repair improvements.
- Write the results artifact to `results/experiment_1999_code_verification_humaneval.json`.

### SCENARIO-CODE-033: HumanEval Fuzzing Run Completes
Given a run of 50 HumanEval questions, when Ising-guided fuzzing is applied, then structural constraints are extracted via CodeExtractor, and baseline and repair improvements are tracked and saved to a JSON artifact.

## S* Energy Pre-Ranking Requirements (arXiv 2502.14382)

### REQ-RANK-001: SStarEnergyRanker Candidate Energy Computation

The SStarEnergyRanker MUST compute Carnot energy for each code candidate using
static structural analysis (bag-of-tokens embedding). It MUST select the
lowest-energy candidate as the predicted-best candidate before any execution
tests are run. This implements the pre-filtering stage from the S* test-time
scaling paper (arXiv 2502.14382), replacing expensive oracle execution with
energy-based selection.

### REQ-RANK-002: Energy Pre-Filtering Metrics

The system MUST report the following metrics for energy pre-ranking experiments:
- energy_correct_rank_pct: fraction of problems where the energy-selected
  candidate (lowest energy) matches the correct candidate (passes execution test)
- tests_saved_pct: estimated fraction of execution tests avoided by using energy
  pre-filtering. Reported as (1 - 1/n_candidates) when energy_correct_rank_pct
  >= 0.60; zero otherwise (energy must be reliable enough to justify skipping)

### SCENARIO-RANK-001: Energy Rank Returns Sorted Candidate List

Given a list of code candidates with different operator usage, when
SStarEnergyRanker.rank_by_energy() is called, then the returned list MUST be
sorted ascending by computed energy (lowest energy first), and all input
candidates MUST appear in the output (no candidates dropped).

### SCENARIO-RANK-002: Tests-Saved Threshold Gate

Given energy_correct_rank_pct = 0.45 (below the 0.60 threshold), when
tests_saved_pct is computed, then it MUST equal 0.0 — the energy signal is
not reliable enough to skip execution tests at this accuracy level.

## Implementation Status

| Requirement | Status |
|-------------|--------|
| REQ-EXTRACT-035 | Implemented |
| REQ-EXTRACT-036 | Implemented |
| REQ-RANK-001 | Implemented (Exp 787) |
| REQ-RANK-002 | Implemented (Exp 787) |
| REQ-CODE-3573 | Implemented |
| REQ-CODE-4197 | Implemented |
| REQ-CODE-4198 | Implemented (`python/carnot/experiment_4198_verifier_reward_3arm_rft_launch.py`) |
| REQ-CODE-4211 | Planned (`python/carnot/experiment_4211_verifier_as_reward_finish_synchronous.py`) |
| REQ-CODE-4222 | Planned (`python/carnot/experiment_4222_verifier_reward_lora_harness_fix_smoke.py`) |
| REQ-CODE-4223 | Implemented (`python/carnot/experiment_4223_verifier_as_reward_3arm_synchronous.py`) |
| REQ-CODE-4234 | Planned (`python/carnot/experiment_4234_verifier_reward_lora_harness_real_training_smoke.py`) |
| REQ-CODE-4247 | Implemented (`python/carnot/experiment_4247_verifier_reward_offline_harness_retire_livelora.py`) |

### REQ-CODE-3573: Verifier Ensemble Generalization to Code
The verifier ensemble MUST be evaluated for generalization on code domains (correct vs buggy execution-labeled completions).
- Compare AUROC to model-confidence and best single verifier.
- Produce `results/experiment_3573_verifier_code_bug_error_detection.json`.

### REQ-CODE-4197: Code Verifier-Reward Phase-0 And 3-Arm LoRA-RFT Harness

The repository shall provide an Exp 4197 workflow that establishes whether the
execution verifier is clean enough to become a code-training reward and builds
the on-policy 3-arm LoRA-RFT runner for the decisive A2 training run:
- Preconditions must check for a cached non-Qwen trainable base, CUDA, a
  loadable code corpus, and the restricted execution sandbox before any
  measurement or harness launch.
- Phase-0 measurement must compute `phase0_precision` as a bare float
  P(hidden-pass | visible-perfect) and `youden_j` as a bare float TPR-FPR from
  visible-test verifier labels against hidden-test ground truth.
- Training suitability must report a bare boolean
  `training_headroom_present`, own-sample visible-perfect rate, truncation rate,
  and no-answer rate for the selected code operating point.
- The result artifact must include bare top-level fields
  `phase0_precision`, `youden_j`, `training_headroom_present`, and
  `harness_ready`, plus `operating_point`, `model_specs`, `random_seed`, and a
  reproducibility checksum.
- The on-policy runner must construct N-matched Arm A verifier-certified
  traces, Arm B same-generator random-label controls, Arm C hidden-test-gold
  traces, and Arm D cold-base evaluation with identical LoRA configuration for
  trained arms.

### SCENARIO-CODE-4197-PHASE0: Visible-Perfect Labels Produce Bare Metrics

Given cached code candidates with visible and hidden execution outcomes, when
the Exp 4197 Phase-0 scorer runs, then it computes certification precision,
TPR, FPR, and Youden-J from the raw labels and exposes `phase0_precision` and
`youden_j` as bare JSON numbers.

### SCENARIO-CODE-4197-HARNESS: 3-Arm Runner Builds Matched Corpora

Given code tasks with same-generator candidate traces, when the Exp 4197 runner
builds training corpora, then Arm A contains visible-perfect traces, Arm B
contains an equal number of non-certified same-generator traces selected with a
deterministic seed, Arm C contains hidden-pass gold traces, and the smoke path
validates two tasks without launching a full training run.

### REQ-CODE-4198: Detached 3-Arm LoRA-RFT Launch With Stable Resume Key

The repository shall provide an Exp 4198 workflow that launches the decisive
code verifier-as-reward 3-arm LoRA-RFT run only after the Exp 4197 operating
point gate clears:
- Preconditions must read the Exp 4197 artifact and require
  `phase0_precision >= 0.85`, `youden_j > 0`, `harness_ready=true`, and CUDA
  availability before launching training.
- If the gate is missing or below threshold, the result must honestly defer with
  `honest_verdict` equal to
  `complete_verifier_reward_train_deferred_no_clean_operating_point`; if CUDA is
  unavailable, the verdict must be `blocked_cuda_unavailable`.
- The workflow must build Arm A verifier-certified traces, Arm B same-generator
  non-certified/random-label controls with `|B|=|A|`, Arm C hidden-test-gold
  traces, and Arm D cold-base metadata from the same code-generation checkpoint
  and deterministic seed.
- The workflow must derive a stable checkpoint path from the corpus, trainable
  base, LoRA config, and seed rather than from a transient experiment id, so
  future windows resume the same run instead of restarting it.
- When no prior live run is present, the workflow must launch the existing
  3-arm LoRA-SFT runner detached with per-arm checkpoint directories and write a
  launch artifact containing `training_launched`, `stable_checkpoint_path`,
  `arm_corpus_sizes`, `gold_control_early_read`, `model_specs`, `random_seed`,
  and `reproducibility_checksum`.

### SCENARIO-CODE-4198-GATED-LAUNCH: Clean Gate Produces Detached Live Artifact

Given a clean Exp 4197 operating point and CUDA availability, when Exp 4198
runs, then it writes a result artifact whose `training_launched` field is a bare
boolean true, whose stable checkpoint path is corpus/base/config keyed, whose
Arm A and Arm B corpus sizes match, and whose recorded process is alive or
already checkpointing.

### SCENARIO-CODE-4198-HONEST-DEFERRAL: Missing Gate Does Not Launch Training

Given a missing or sub-threshold Exp 4197 gate, when Exp 4198 runs, then it
writes the terminal honest deferral verdict and leaves `training_launched` false
without spawning any training process.

### REQ-CODE-4211: Synchronous Resume Of Verifier-As-Reward A-vs-B

The repository shall provide Exp 4211 at
`python/carnot/experiment_4211_verifier_as_reward_finish_synchronous.py` and
`results/experiment_4211_verifier_as_reward_finish_synchronous.py` to finish
the `.389` verifier-as-reward A-vs-B contrast from the stable Exp 4198
checkpoint. The runner MUST:

- check a cached non-Qwen trainable base before training and block with
  `honest_verdict="blocked_no_nonqwen_base_cached"` when none of
  `google/gemma-4-E4B-it`, `google/gemma-4-12B-it`, or
  `openbmb/MiniCPM5-1B` is readable;
- require CUDA before training and block with
  `honest_verdict="blocked_cuda_unavailable"` when `torch.cuda.is_available()`
  is false;
- read the stable checkpoint
  `code_verifier_reward_lora_rft_a83b52882c198954`, its manifest, and the
  Arm A/B/C corpora, preserving the Exp 4198 N-matching guarantee `|A|=|B|`;
- resume and accumulate per-arm progress synchronously in the current process,
  not by `setsid`, `Popen`, or detached background launch, and print
  per-arm/step/loss progress at least every approximately 30 seconds while
  training is active;
- write a terminal artifact containing `honest_verdict`,
  `verifier_label_carries_signal`, `a_vs_b_delta`, `a_vs_b_ci95`,
  `positive_control_confirmed`, `youden_j`, `accumulated_n`,
  `verifier_is_oracle`, `model_specs`, `random_seed`, and
  `reproducibility_checksum`, with principles for each required field; and
- treat a clean label-carries-signal result, a clean A~=B distillation null, an
  honest blocked precondition, or an honest accumulating/invalid/retired result
  as terminal rather than fabricating a headline.

The A-vs-B decision SHALL be based on held-out hidden-test pass@1 for Arm A
minus Arm B and a task-level bootstrap CI95 using at least 2000 resamples when
held-out evaluation exists. `verifier_label_carries_signal` SHALL be the bare
boolean `true` only when the CI excludes zero and the delta is positive. The
positive control SHALL be the bare boolean `true` only when Arm C is at least
the cold base and every arm's truncation rate is below 5%; a failed positive
control invalidates A-vs-B and must produce an honest invalid verdict.
`verifier_is_oracle` SHALL be the bare boolean `true` because the reward label
is the executable correctness oracle.

### SCENARIO-CODE-4211-BLOCKED-PRECONDITION: Missing Live Resource Writes Terminal Block

Given the stable Exp 4198 artifact and corpora are present but either no
approved non-Qwen trainable base is cached or CUDA is unavailable, when Exp
4211 runs, then it writes
`results/experiment_4211_verifier_as_reward_finish_synchronous.json` with the
corresponding blocked `honest_verdict`, leaves the A-vs-B metrics null, sets
`verifier_label_carries_signal=false`, and records the failed precondition
without launching any training process.

### SCENARIO-CODE-4211-SYNC-ACCUMULATE: Checkpoint Resumes In Process

Given CUDA, an approved non-Qwen base, the stable Exp 4198 checkpoint, and
N-matched Arm A/B corpora, when Exp 4211 starts training, then it calls the
3-arm LoRA-SFT runner in-process, records synchronous progress for each arm,
updates accumulated per-arm checkpoints, and does not use detached process
launching.

### SCENARIO-CODE-4211-VERDICT-GATES: A-vs-B Is Reported Only After Controls

Given held-out hidden-test pass@1 measurements for Arms A, B, C, and cold base,
when Exp 4211 builds the final artifact, then it first applies the gold-control
and truncation gates, reports `a_vs_b_delta`, `a_vs_b_ci95`,
`positive_control_confirmed`, and `verifier_label_carries_signal` as bare
fields, marks failed controls as invalid rather than headline findings, and
includes Youden-J plus the memorization-shortcut diagnostic.

### REQ-CODE-4222: Verifier-Reward LoRA Attach Harness Smoke

The repository shall provide Exp 4222 at
`python/carnot/experiment_4222_verifier_reward_lora_harness_fix_smoke.py` and
`results/experiment_4222_verifier_reward_lora_harness_fix_smoke.py` to prove
the verifier-as-reward LoRA harness attaches before another live A-vs-B
measurement. The smoke MUST:

- block with `honest_verdict="blocked_cuda_unavailable"` before model loading
  when CUDA is unavailable;
- block with `honest_verdict="blocked_no_nonqwen_base_cached"` when the cached
  standard HuggingFace base `google/gemma-4-E4B-it` is absent, and must never
  substitute Qwen as the trained base;
- build an 8-16 example fixture from the stable
  `code_verifier_reward_lora_rft_a83b52882c198954` A/B/C corpora when readable,
  or from the same operating-point prompt shape when those corpora are absent;
- load the base through standard
  `transformers.AutoModelForCausalLM.from_pretrained("google/gemma-4-E4B-it")`,
  not the custom Gemma4 quantized wrapper, and attach PEFT LoRA to standard
  projection module names `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, and `down_proj`, or retry the same projection modules' inner
  `.linear` children when the standard HF class exposes `Gemma4ClippableLinear`
  wrappers;
- assert the attached model exposes a bare positive `trainable_param_count`,
  executes one optimizer step on the fixture, and reports a finite loss before
  setting the bare `harness_smoke_passed` field to true; and
- write `results/experiment_4222_verifier_reward_lora_harness_fix_smoke.json`
  with `honest_verdict`, `harness_smoke_passed`, `lora_attach_path`,
  `trainable_param_count`, `verifier_is_oracle`, `model_specs`, `random_seed`,
  and `reproducibility_checksum` plus principles for each required field.

### SCENARIO-CODE-4222-BLOCKED-PRECONDITION: Missing Live Resource Stops Smoke

Given the stable verifier-reward corpora may or may not be present, when Exp
4222 runs without CUDA or without the cached `google/gemma-4-E4B-it` base, then
it writes the corresponding blocked artifact, leaves
`harness_smoke_passed=false`, leaves `trainable_param_count=0`, and does not
attempt LoRA attachment.

### SCENARIO-CODE-4222-STANDARD-LORA-ATTACH: Positive Control Attaches And Steps

Given CUDA, the cached non-Qwen Gemma base, and an 8-16 example fixture, when
Exp 4222 runs, then PEFT attaches LoRA through the standard
`AutoModelForCausalLM` path with either the projection target modules or their
inner `.linear` children, records `lora_attach_path`, records
`trainable_param_count > 0`, executes one training step with finite loss, sets
`harness_smoke_passed=true`, and includes a checksum over the fixture plus
working LoRA configuration.

### REQ-CODE-4223: B1-Gated Synchronous Verifier-As-Reward 3-Arm Finish

The repository shall provide Exp 4223 at
`python/carnot/experiment_4223_verifier_as_reward_3arm_synchronous.py` and
`results/experiment_4223_verifier_as_reward_3arm_synchronous.py` to finish the
operator's 2026-06-11 verifier-as-reward pivot using the B1-proven LoRA harness
configuration from Exp 4222. The runner MUST:

- read `results/experiment_4222_verifier_reward_lora_harness_fix_smoke.json`
  first and stop with
  `honest_verdict="complete_verifier_reward_deferred_harness_not_smoked"` when
  `harness_smoke_passed` is not the bare boolean true or when
  `lora_attach_path`/`model_specs.lora_config` are missing;
- require CUDA before live training and stop with
  `honest_verdict="blocked_cuda_unavailable"` when `torch.cuda.is_available()`
  is false;
- read the stable
  `code_verifier_reward_lora_rft_a83b52882c198954` checkpoint and the .389/.390
  Arm A/B/C corpora, preserving the size-matched same-generator random-label
  control guarantee `|B|=|A|`;
- resume and accumulate per-arm LoRA-SFT synchronously in the current process,
  not by `setsid`, `Popen`, or detached background launch, while recording the
  B1-proven LoRA config in `model_specs` and in the reproducibility checksum;
- write `results/experiment_4223_verifier_as_reward_3arm_synchronous.json`
  with `honest_verdict`, `verifier_label_carries_signal`, `a_vs_b_delta`,
  `a_vs_b_ci95`, `positive_control_confirmed`, `youden_j`, `accumulated_n`,
  `verifier_is_oracle`, `model_specs`, `random_seed`, and
  `reproducibility_checksum`, with principles for each required field; and
- treat a clean label-carries-signal result, a clean A~=B distillation null, an
  honest blocked/deferred precondition, or an honest accumulating/invalid/retired
  result as complete rather than fabricating a headline.

The A-vs-B decision SHALL use held-out hidden-test pass@1 for Arm A minus Arm B
and a task-level bootstrap CI95 with at least 2000 resamples when held-out
evaluation exists. `verifier_label_carries_signal` SHALL be the bare boolean
`true` only when the CI excludes zero and the delta is positive. The positive
control SHALL be the bare boolean `true` only when Arm C is at least the cold base
and every arm's truncation rate is below 5%; a failed positive control invalidates
A-vs-B and must produce an honest invalid verdict. `verifier_is_oracle` SHALL be
the bare boolean `true` because the reward label is the executable correctness
oracle.

### SCENARIO-CODE-4223-DEFERRED-HARNESS: Missing B1 Smoke Stops The Finish

Given the Exp 4222 smoke artifact is absent, unreadable, false, or missing its
working LoRA attach path/config, when Exp 4223 runs, then it writes the terminal
deferred verdict, leaves A-vs-B metrics null, sets
`verifier_label_carries_signal=false`, and does not launch training.

### SCENARIO-CODE-4223-SYNC-ACCUMULATE: B1 Config Resumes In Process

Given B1 passed, CUDA is available, an approved non-Qwen base is cached, and the
stable checkpoint has N-matched Arm A/B corpora, when Exp 4223 starts training,
then it delegates to the existing 3-arm LoRA-SFT runner in-process, records
synchronous progress and accumulated per-arm checkpoints, records
`used_detached_process=false`, and carries the B1-proven LoRA config into the
artifact and checksum.

### SCENARIO-CODE-4223-VERDICT-GATES: De-Confounded A-vs-B Is Reported Only After Controls

Given held-out hidden-test pass@1 measurements for Arms A, B, C, and cold base,
when Exp 4223 builds the final artifact, then it applies the gold-control and
truncation gates before headline interpretation, reports `a_vs_b_delta`,
`a_vs_b_ci95`, `positive_control_confirmed`, and
`verifier_label_carries_signal` as bare fields, marks failed controls as invalid,
and includes Youden-J plus the memorization-shortcut diagnostic.

### REQ-CODE-4234: Verifier-Reward LoRA Harness Real-Training Smoke

The repository shall provide Exp 4234 at
`python/carnot/experiment_4234_verifier_reward_lora_harness_real_training_smoke.py`
and `results/experiment_4234_verifier_reward_lora_harness_real_training_smoke.py`
to prove the verifier-as-reward LoRA harness performs real optimizer work before
B2 can launch another 3-arm A/B/C/D run. The smoke MUST:

- check CUDA first and write `honest_verdict="blocked_cuda_unavailable"` without
  attempting model load or attachment when CUDA is unavailable;
- select a cached non-Qwen standard HuggingFace trainable base, preferring
  `unsloth/gemma-4-12B-it` and falling back only to `google/gemma-4-E4B-it`,
  and write `honest_verdict="blocked_no_nonqwen_base_cached"` when neither base
  is cached or fetchable in-window;
- build a 16-32 example fixture from the stable
  `code_verifier_reward_lora_rft_a83b52882c198954` A/B/C corpora when readable,
  or from the same operating-point prompt shape when the stable corpora are
  absent;
- load the selected base with
  `transformers.AutoModelForCausalLM.from_pretrained(...)`, never through the
  custom Gemma4 quantized wrapper or a GGUF tokenizer path, and attach PEFT LoRA
  to standard projection module names `q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, and `down_proj`, retrying those modules' inner
  `.linear` children only if the standard path exposes unsupported
  `Gemma4ClippableLinear` wrappers;
- set bare `harness_smoke_passed=true` only when LoRA exposes
  `trainable_param_count > 0`, at least 20 real optimizer steps run, the final
  step loss is lower than the first step loss by a non-trivial margin, and the
  wall clock exceeds the configured plausibility floor; and
- write
  `results/experiment_4234_verifier_reward_lora_harness_real_training_smoke.json`
  with bare fields `honest_verdict`, `harness_smoke_passed`, `steps_run`,
  `loss_initial`, `loss_final`, `lora_attach_path`, `trainable_param_count`,
  `verifier_is_oracle`, `model_specs`, `random_seed`, and
  `reproducibility_checksum`, plus field principles and the per-step loss trace.

`verifier_is_oracle` SHALL be the bare boolean `true` because the reward label is
the executable correctness oracle. A smoke that cannot perform real training in
the current window SHALL write
`honest_verdict="blocked_lora_training_cannot_run_in_window"` rather than a
`progress_*` verdict.

### SCENARIO-CODE-4234-BLOCKED-PRECONDITION: Missing Live Resource Stops Real Smoke

Given the stable verifier-reward corpora may or may not be present, when Exp
4234 runs without CUDA or without an approved cached non-Qwen base, then it
writes the corresponding blocked artifact, leaves `harness_smoke_passed=false`,
sets `steps_run=0`, sets `trainable_param_count=0`, and does not attempt LoRA
attachment.

### SCENARIO-CODE-4234-REAL-TRAINING: Positive Control Requires Sustained Optimizer Work

Given CUDA, an approved cached non-Qwen base, and a 16-32 example fixture, when
Exp 4234 runs, then PEFT attaches LoRA through the standard
`AutoModelForCausalLM` path with either the projection target modules or their
inner `.linear` children, prints and records a positive trainable parameter
count, prints every optimizer step and loss, records at least 20 step/loss
events, requires `loss_final < loss_initial`, requires wall-clock duration at or
above the plausibility floor, and sets `harness_smoke_passed=true` only when all
of those checks pass.

### REQ-CODE-4247: Offline Reward-Weighted SFT Harness Retires Live LoRA

The repository shall provide Exp 4247 at
`python/carnot/experiment_4247_verifier_reward_offline_harness_retire_livelora.py`
and `results/experiment_4247_verifier_reward_offline_harness_retire_livelora.py`
to retire the failed live-LoRA verifier-as-reward path and prove the bounded
offline reward-weighted SFT harness can perform real optimizer work before B2
launches the full A/B/C contrast. The smoke MUST:

- set the bare boolean `live_lora_retired=true` with a one-line rationale that
  the live path accumulated six infrastructure failures and met the
  accumulate-floor retirement condition;
- check CUDA before model loading and write `honest_verdict` equal to
  `blocked_cuda_unavailable` when CUDA is unavailable;
- select a cached non-Qwen standard HuggingFace trainable base, preferring
  `google/gemma-4-E4B-it` and falling back only to
  `unsloth/gemma-4-12B-it`, never Qwen and never a GGUF training repo;
- build a 16-32 example fixture from the stable
  `code_verifier_reward_lora_rft_a83b52882c198954` Arm A/B/C corpora when
  readable, or rebuild a tiny same-operating-point fixture when the corpora are
  absent;
- precompute deterministic per-example reward weights for offline SFT so
  verifier-certified and gold rows receive the positive reward weight while
  ablation/control rows receive the configured control weight;
- load the selected base only through
  `transformers.AutoModelForCausalLM.from_pretrained(..., local_files_only=True)`
  and attach PEFT LoRA only to standard projection module names `q_proj`,
  `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`;
- run a finite, fixed-step, reward-weighted supervised loss with no live
  generation and no RL loop; and
- write
  `results/experiment_4247_verifier_reward_offline_harness_retire_livelora.json`
  with bare fields `honest_verdict`, `live_lora_retired`,
  `harness_smoke_passed`, `steps_run`, `loss_initial`, `loss_final`,
  `lora_attach_path`, `trainable_param_count`, `verifier_is_oracle`,
  `model_specs`, `random_seed`, and `reproducibility_checksum`, plus field
  principles and the per-step loss trace.

`harness_smoke_passed` SHALL be the bare boolean `true` only when LoRA exposes
`trainable_param_count > 0`, at least 20 real optimizer steps run, the final
step loss is lower than the first step loss by a non-trivial margin, and the
wall clock exceeds the configured plausibility floor. `verifier_is_oracle`
SHALL be the bare boolean `true` because the reward label is the executable
correctness oracle. A smoke that cannot perform real offline reward-weighted
training in the current window SHALL write
`honest_verdict="blocked_offline_reward_weighted_training_cannot_run_in_window"`
rather than a `progress_*` verdict.

### SCENARIO-CODE-4247-BLOCKED-PRECONDITION: Missing Live Resource Stops Offline Smoke

Given the stable verifier-reward corpora may or may not be present, when Exp
4247 runs without CUDA or without an approved cached non-Qwen base, then it
writes the corresponding blocked artifact, leaves `harness_smoke_passed=false`,
sets `steps_run=0`, sets `trainable_param_count=0`, sets
`live_lora_retired=true`, and does not attempt LoRA attachment or training.

### SCENARIO-CODE-4247-OFFLINE-REAL-TRAINING: Reward-Weighted SFT Requires Sustained Optimizer Work

Given CUDA, an approved cached non-Qwen base, and a 16-32 example weighted
fixture, when Exp 4247 runs, then PEFT attaches LoRA through the standard
`AutoModelForCausalLM` projection targets, prints and records a positive
trainable parameter count, prints every offline optimizer step and weighted
loss, records at least 20 step/loss events, requires `loss_final <
loss_initial`, requires wall-clock duration at or above the plausibility floor,
and sets `harness_smoke_passed=true` only when all of those checks pass.


### REQ-CODE-VERIFY-3602: FoVer Math-Trained PRM Transfer to Code Benchmark

Evaluate if a math-trained Process Reward Model (PRM) transfers its discriminator capability to a code-generation benchmark (such as HumanEval). Measures whether the discriminative verifier is robust to out-of-domain evaluation or if discriminative verifiers are OOD-FRAGILE. Required for determining whether the 0.44 AUROC on code was a fundamental limit or a wiring failure.

### REQ-CODE-3641: Cached Code-Corpus Verifier Transfer Artifact

The repository shall provide an offline Exp 3641 workflow that builds a
machine-readable labeled code-candidate corpus and evaluates verifier transfer
without live LLM inference:
- The workflow first inspects Exp 1999 and records whether it contains both
  correctness labels and candidate source text. If Exp 1999 lacks candidate
  source, the workflow may fall back to an on-disk MBPP/HumanEval candidate
  artifact that carries per-candidate source code and execution labels.
- The workflow writes `data/code_verification_corpus_v1.jsonl` with at least
  `candidate_code`, `label`, and `test_outcome` for every usable row.
- The workflow imports the configured code-applicable verifier modules and
  records whether each verifier actually scored candidate code, distinguishing
  import success from an inert interface mismatch.
- The top-level artifact writes `code_verifiers_fire` as a bare boolean that is
  true only when at least one execution-applicable verifier produced nonconstant
  scores (`n_scored > 0` and variance > 0).
- The workflow reports execution-verifier AUROC, FoVer-math-derived signal
  AUROC, and a confidence/self-consistency baseline AUROC with deterministic
  bootstrap confidence intervals over at least three random seeds.
- A no-transfer verdict is permitted only when the corpus has headroom and the
  execution verifiers fired; otherwise the artifact must use a blocked or inert
  terminal verdict.

### SCENARIO-CODE-3641: Code Transfer Artifact Distinguishes Corpus, Inert, And Transfer Verdicts

Given synthetic code-candidate fixtures that cover a blocked corpus, inert
verifiers, a firing/no-transfer result, and a firing/transfer result, when the
Exp 3641 workflow builds its artifact, then the terminal `honest_verdict`
matches the honest fixture state, `code_verifiers_fire` remains a bare boolean,
the corpus JSONL is written only when labeled source rows exist, and the AUROC
fields include deterministic confidence intervals.

### REQ-CODE-3658: Balanced Second-Corpus Code Generalization Replication

The repository shall provide an offline Exp 3658 workflow that re-tests the
Exp 3641 math-to-code verifier transfer claim on a second, balanced code
corpus without live LLM inference:
- The workflow builds `data/code_verification_corpus_v2.jsonl` from a cached
  MBPP, LiveCodeBench, or fresh HumanEval split with per-candidate correctness
  labels and at least `candidate_code`, `label`, and `test_outcome` fields.
- The corpus must contain at least 50 labeled examples and both correct and
  error classes must each represent at least 20% of the rows; otherwise the
  artifact must report `complete: blocked_no_second_code_corpus`.
- The four execution-applicable verifier modules from Exp 3641 must be imported
  and diagnosed, with inert per-candidate interfaces recorded without aborting
  the run.
- The artifact must report execution-verifier AUROC, FoVer-math-derived code
  AUROC, and a confidence baseline AUROC with deterministic bootstrap
  confidence intervals over at least three random seeds.
- The top-level `code_generalization_replicates` field must be a bare boolean
  and may be true only when the balanced second corpus meets the gate, the code
  verifiers fire with nonconstant scores, and the math-derived signal is
  consistent with the Exp 3641 transfer result.

### SCENARIO-CODE-3658: Second Corpus Verdicts Cover Replication, Null, Inert, And Blocked

Given synthetic second-corpus fixtures that cover a balanced replication, a
balanced non-replication, inert verifier scores, and a missing or imbalanced
corpus, when the Exp 3658 workflow builds its artifact, then the terminal
`honest_verdict` matches the honest fixture state, `class_balance` records the
class fractions, `code_generalization_replicates` remains a bare top-level
boolean, and AUROC fields include deterministic confidence intervals whenever
the corpus is accepted.

### REQ-CODE-3705: Code-Native Leak Audit And Held-Out Replication

The repository shall provide an offline Exp 3705 workflow that audits the
Exp 3695 code-native AUROC=1.0 result before the code signal may remain a
shipped headline:
- The workflow must reproduce or read the Exp 3695 in-corpus code-native AUROC
  and record why the number is suspicious, including separability,
  train/holdout contamination, and label-correlated metadata or feature checks.
- The workflow must score a held-out code corpus that is distinct from the
  Exp 3658 tuned corpus, using AST and execution-trace features rather than a
  label proxy or constant score.
- The artifact must report held-out AUROC, CI, calibration Brier/ECE,
  recall-at-fixed-FPR, five bootstrap seeds, held-out sample size, and a
  reproducibility checksum.
- The top-level `leak_detected` and `code_signal_survives_heldout` fields must
  be bare booleans. `code_signal_survives_heldout` must be false whenever
  held-out AUROC is at least 0.99, even if the CI excludes chance.
- Cached-corpus-only runs must set `inference_substrate` exactly to
  `verifier_ensemble_against_cached_candidates` and must not include live GGUF
  markers. Live generation is out of scope unless a mandated cached GGUF and
  CUDA precondition are both satisfied.

### SCENARIO-CODE-3705: Leak Audit Verdicts Cover Survives, Leak, And Blocked

Given synthetic Exp 3705 fixtures covering a genuine held-out signal, a
label-correlated or implausibly perfect leak case, and a missing held-out
corpus, when the workflow builds the artifact, then the terminal
`honest_verdict` matches the honest fixture state, required AUROC/CI/audit
fields are present, `leak_detected` and `code_signal_survives_heldout` remain
bare booleans, and the acceptance gate passes only when the suspicious
in-corpus result is explained and the held-out metric plus leak audit are
recorded.

### REQ-CODE-6186: Cached LiveCodeBench Bank Preregistration

The repository shall provide an offline Exp6186 workflow that freezes a
LiveCodeBench code bank before any generation, selector tuning, hidden-state
extraction, or continuous-learning outcome can influence task choice:
- The workflow shall load only an already cached LiveCodeBench snapshot and
  fail closed if the cache is absent, revised during the run, or would require
  a runtime dataset download.
- The workflow shall derive stable task IDs and deterministic eligibility from
  pre-generation metadata only: platform, contest date, difficulty, metadata
  tags, prompt size, and supported Python runtime. candidate correctness,
  model scores, hidden states, selector features, or oracle outcomes shall not
  participate in eligibility, stratification, or split assignment.
- The workflow shall freeze exactly 120 unique task IDs into four disjoint
  splits: 36 calibration, 36 held-selector, 18 continuous-learning seed, and
  30 continuous-learning prospective tasks.
- The workflow shall seal exact prompt, public-test, private-test, metadata,
  and stable-task hashes for each selected task while keeping public prompt
  rows separate from the private-test vault index.
- The public prompt bank and selector feature rows shall not contain private
  test source, assertion text, expected output, oracle traces, candidate
  source, model output, hidden states, or private-test-derived features.
- The private-test vault shall be access controlled to executor-only use and
  shall record cache coordinates plus hashes rather than exposing private test
  text to prompts or selectors.
- The workflow shall dry-run only deterministic executor fixtures or
  maintainer references, never candidate solutions, and shall record timeout,
  process, filesystem, network, resource, nondeterminism, and unsupported-task
  policies.
- The terminal artifact shall be
  `results/experiment_6186_livecodebench_bank_preregistration.json` and shall
  include the required Exp6186 schema fields, set
  `candidate_and_model_access_count` to bare `0`, set
  `inference_substrate` to
  `deterministic_cached_livecodebench_bank_preregistration`, and set
  `bank_ready_score` to bare `1` only when all split, overlap, hash,
  private-test-isolation, executor-fixture, protected-file, and cache
  immutability checks pass.

### SCENARIO-CODE-6186-CACHED-SNAPSHOT-FAIL-CLOSED: Runtime Downloads Are Forbidden

Given an expected cached LiveCodeBench snapshot identity and file hash,
when Exp6186 loads the bank,
then it reads local Arrow cache files directly, records the dataset
name/revision/cache path/hash/task count, and returns a blocked artifact rather
than downloading or accepting silent cache revision.

### SCENARIO-CODE-6186-DISJOINT-DETERMINISTIC-SPLITS: Stable Metadata Defines The Bank

Given a cached 400-task LiveCodeBench snapshot,
when Exp6186 applies its seed and deterministic metadata-only stratification,
then it emits exactly 36 calibration, 36 held-selector, 18 continuous-learning
seed, and 30 continuous-learning prospective unique IDs with a zero overlap
matrix and a reproducible split checksum.

### SCENARIO-CODE-6186-PRIVATE-TEST-NONINTERFERENCE: Prompts And Features Cannot See Oracles

Given Exp6186 must hash private tests for future execution,
when it writes the public prompt bank, selector feature rows, split manifest,
and terminal artifact,
then none of those public surfaces contain private test source, assertion text,
expected output, oracle traces, candidate source, model output, or hidden
states, and private-test access is limited to the executor vault receipt.

### SCENARIO-CODE-6186-EXECUTOR-FIXTURE-ONLY: No Candidate Generation During Preregistration

Given Exp6186 validates executor readiness,
when it performs executor dry-runs,
then it runs deterministic fixtures or maintainer references only, records the
sandbox policy dimensions, classifies unsupported task forms, and keeps
`candidate_and_model_access_count` equal to bare `0`.

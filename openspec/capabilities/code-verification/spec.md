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

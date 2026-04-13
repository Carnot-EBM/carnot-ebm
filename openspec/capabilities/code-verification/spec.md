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
- Maps token strings to indices via `hash(token) % vocab_size`
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

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

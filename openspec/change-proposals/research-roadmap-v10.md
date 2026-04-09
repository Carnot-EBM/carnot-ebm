# Carnot Research Roadmap v10: Learn to Verify — Self-Improving Constraints + Differentiable Reasoning

**Created:** 2026-04-09
**Milestone:** 2026.04.5
**Status:** Planned (activates when milestone 2026.04.4 completes)
**Supersedes:** research-roadmap-v9.md (milestone 2026.04.4)
**Informed by:** Experiments 66–85, beta release, pipeline benchmarks, dogfooding results
**Target Models:** Qwen3.5-0.8B, google/gemma-4-E4B-it

## What v9 Proved

| Approach | Experiments | Finding |
|----------|------------|---------|
| Unified constraint API | 74 | Pluggable extractors (arithmetic, code, logic, NL) with auto-detection |
| VerifyRepairPipeline | 75 | Single-class API: verify + repair in 5 lines of Python |
| MCP server hardening | 76 | 6 tools, 30s timeout, 10K char limit, structured errors |
| CLI overhaul | 77 | `carnot verify`, `carnot score`, `carnot pipeline` subcommands |
| Error handling | 82 | Structured error hierarchy, wall-clock timeout, graceful degradation |
| PyPI packaging | 78 | Pure-Python install, optional Rust bindings, extras system |
| Integration examples | 79 | 5 production-ready examples covering common workflows |
| Documentation | 80 | Getting started, concepts, API reference — 5-minute onboarding |
| Integration tests | 81 | Full pipeline E2E, CLI subprocess tests, install smoke tests |
| Performance benchmarks | 83 | Sub-ms p99, 36,887 calls/s, zero memory growth |
| Dogfooding | 84 | Pipeline verifies its own source code |
| Beta release prep | 85 | 0.1.0-beta1 ready with release script |

**The gap:** The pipeline is shipped and usable, but it doesn't LEARN. Constraint extractors are hand-coded pattern matchers. The Ising models are trained once and frozen. The system cannot improve itself from its own failures. And we've never run the pipeline with real LLM inference on published benchmarks — only simulated/fallback outputs.

## 3 Gaps Addressed

### Gap 1: No Self-Improvement (FR-11)
The autoresearch loop proposes hypotheses and evaluates energy/time/memory, but the constraint pipeline doesn't feed back into itself. When the pipeline fails to catch a hallucination, that failure is lost — no one learns from it. v10 closes this loop: pipeline failures become training data for better constraint models and extractors.

### Gap 2: No Real Benchmark Validation (FR-12)
GSM8K (Exp 67) and HumanEval (Exp 68) used synthetic fallback data when datasets weren't available. We've never measured actual accuracy improvement from constraint verification on a real model generating real answers. v10 runs live inference on Qwen3.5-0.8B and Gemma4-E4B-it against published benchmarks.

### Gap 3: Differentiable Reasoning Deferred Twice (Exp 66)
Exp 66 (end-to-end differentiable constraint reasoning) was deferred from v8 and v9. The continuous relaxation (Exp 64) and embedding-space verification (Exp 65) are both done — the prerequisites are met. v10 completes the arc from discrete Ising to continuous differentiable constraint reasoning, which is the bridge to Kona-style latent-space verification.

## v10 Architecture: Self-Improving Constraint Pipeline

```
Pipeline Failure Log
     │
     ▼
┌──────────────────────┐
│  Failure Analyzer     │ ← Exp 88: mine missing constraint patterns
│  (what did we miss?)  │ ← from GSM8K/MATH/HumanEval failures
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│  Self-Bootstrap       │ ← Exp 89: verified outputs → training data
│  Trainer              │ ← CD on (correct, wrong) from pipeline's OWN runs
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│  Improved Ising       │ ← Exp 90: autoresearch proposes extractor changes
│  + Extractors         │ ← evaluate on held-out failures → accept/reject
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│  Differentiable       │ ← Exp 66: end-to-end gradient through verification
│  Constraint Layer     │ ← Exp 86: learned energy weights via validation
│  (continuous Ising)   │ ← Exp 87: gradient repair in embedding space
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│  Real Benchmarks      │ ← Exp 91: GSM8K live (Qwen3.5 + Gemma4)
│  (live inference)     │ ← Exp 92: MATH subset with CoT tracing
│                       │ ← Exp 93: Multi-model systematic comparison
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│  Rust Pipeline Port   │ ← Exp 94: carnot-pipeline crate
│  + PyO3 Bridge        │ ← Exp 95: Python calls Rust pipeline via PyO3
└──────────────────────┘
```

## Phase 17: Differentiable Constraint Reasoning (experiments 66, 86, 87)

Bridge discrete Ising verification with continuous gradient-based reasoning. This completes the arc from Exp 64 (continuous relaxation) and Exp 65 (embedding-space constraints) into a unified differentiable verification layer.

### Exp 66: End-to-end differentiable constraint reasoning
The twice-deferred experiment. Combine continuous Ising relaxation (Exp 64) with embedding-space verification (Exp 65) into a single differentiable forward pass:
- Input: LLM response text → sentence embedding (384-dim) + constraint features (N-dim)
- Continuous Ising layer: relaxed spins ∈ [0,1], JAX-differentiable energy
- Joint embedding-constraint energy via small Gibbs EBM
- End-to-end gradient: ∂(verification_loss)/∂(constraint_weights)
- Train on Exp 58 multi-domain benchmark data (500 Q/A pairs)
- **Metric:** Verification AUROC vs discrete pipeline, gradient stability
- **Models:** Qwen3.5-0.8B, google/gemma-4-E4B-it

### Exp 86: Learned energy composition weights
Auto-tune the weights in ComposedEnergy via gradient descent on validation accuracy:
- Current: hand-tuned weights per constraint type (all 1.0)
- Proposed: learn weights via held-out validation split (minimize misclassification)
- Use Exp 66's differentiable pipeline for gradients
- Compare: uniform weights vs learned weights vs per-domain learned weights
- **Metric:** Verification accuracy improvement, weight interpretability
- **Key question:** Do some constraint types matter more than others?

### Exp 87: Gradient-based repair in continuous constraint space
Replace the discrete verify→feedback→regenerate repair loop with continuous gradient descent:
- Relax LLM response embedding to continuous space
- Gradient descent on constraint energy (minimize violations)
- Project repaired embedding back to text via nearest-neighbor decoding
- Compare: gradient repair vs discrete repair loop (Exp 57) vs combined
- **Metric:** Repair success rate, iteration count, wall-clock time
- **Key question:** Is continuous repair faster/better than discrete LLM re-prompting?

## Phase 18: Self-Improving Constraint Pipeline (experiments 88, 89, 90)

Close the FR-11 autonomous self-learning loop. The pipeline learns from its own failures.

### Exp 88: Failure-driven constraint mining
Analyze pipeline failures to discover missing constraint patterns:
- Run Exp 58 benchmark (500 questions) and collect all cases where pipeline MISSED a hallucination (false negatives)
- Categorize failures: what type of claim was missed? (arithmetic chain, implicit logic, world knowledge, code semantics)
- For each failure category, propose new extraction patterns
- Generate candidate regex/AST patterns for the top 3 failure modes
- Test new patterns on held-out failures
- **Metric:** False negative reduction on held-out set
- **Deliverable:** `python/carnot/pipeline/mining.py` with FailureAnalyzer class

### Exp 89: Self-bootstrapped constraint training
Use the pipeline's own verified outputs as training data for better Ising models:
- Run pipeline on 1000 questions (200 per domain) with Qwen3.5-0.8B
- Collect (question, response, verified, constraints, energy) tuples
- Split: verified-correct as positive, verified-wrong as negative
- Train domain-specific discriminative Ising models via CD (extending Exp 62)
- Compare: self-bootstrapped Ising vs original hand-coded Ising vs Exp 62 models
- **Metric:** AUROC improvement on held-out verification accuracy
- **Key question:** Can the pipeline train its own successor?

### Exp 90: Autoresearch constraint improvement loop
Full FR-11 loop: propose extractor improvements → test → evaluate → incorporate:
- Define extractor improvement as an autoresearch hypothesis type
- Hypothesis: "Add pattern X to ArithmeticExtractor" or "Train Ising with features Y"
- Sandbox: test proposed changes on held-out benchmark
- Evaluation gate: improvement in verification AUROC on validation set
- Accept/reject: only incorporate changes that improve validation accuracy
- Run 20 iterations of the improvement loop
- **Metric:** Cumulative accuracy improvement over 20 iterations
- **Deliverable:** `scripts/experiment_90_autoresearch_constraints.py`

## Phase 19: Real Model Benchmarks (experiments 91, 92, 93)

First-ever real LLM inference benchmarks. No more simulated/fallback outputs.

### Exp 91: GSM8K live benchmark (Qwen3.5-0.8B + Gemma4-E4B-it)
Re-run Exp 67 with REAL model inference:
- Load 200 GSM8K test-split questions via HuggingFace datasets
- Generate answers with Qwen3.5-0.8B (greedy decoding, no CoT prompt)
- Generate answers with google/gemma-4-E4B-it (same questions)
- Run full verify-repair pipeline (max 3 iterations) on both
- **Metrics:** Baseline accuracy, post-verify accuracy, post-repair accuracy, per-model
- **Comparison:** Qwen3.5-0.8B vs Gemma4-E4B-it, with and without Carnot
- **Hardware:** CPU inference (both models fit in 4GB)

### Exp 92: MATH benchmark subset with chain-of-thought constraint tracing
Competition math problems require multi-step reasoning — test constraint extraction on chains:
- Load 100 MATH problems (Level 1-3, algebra/number theory) via HuggingFace
- Generate chain-of-thought answers with Qwen3.5-0.8B and Gemma4-E4B-it
- Extract constraints from EACH reasoning step (not just final answer)
- Verify step-by-step: does each intermediate arithmetic claim hold?
- **Metrics:** Step-level accuracy, end-to-end accuracy, constraint coverage per step
- **Key question:** Does step-by-step verification catch errors earlier than final-answer-only?

### Exp 93: Multi-model systematic comparison
Head-to-head Qwen3.5-0.8B vs Gemma4-E4B-it across all 5 domains:
- 50 questions per domain × 5 domains × 2 models × 3 modes = 1500 evaluations
- Modes: baseline, verify-only, verify-repair
- **Metrics per cell:** accuracy, hallucination rate, repair success, latency, constraint coverage
- Generate comparison table and identify model-specific strengths/weaknesses
- **Key question:** Which model benefits MORE from Carnot verification?
- **Deliverable:** `ops/multi-model-comparison.md` with full results table

## Phase 20: Rust Pipeline + Cross-Language (experiments 94, 95)

Port the verify-repair pipeline to Rust for NFR-01 (10x performance).

### Exp 94: Rust VerifyRepairPipeline
Port the core pipeline to the `carnot-constraints` crate:
- Port `ConstraintExtractor` trait with `ArithmeticExtractor`, `LogicExtractor`
- Port `VerifyRepairPipeline` struct with `verify()` and `extract_constraints()`
- Use `tree-sitter` for code constraint extraction (replaces Python AST)
- Reuse existing `BoundConstraint`, `EqualityConstraint`, `IsingConstraint`
- **Metric:** Feature parity with Python pipeline, performance comparison (target: 10x)
- **Deliverable:** `crates/carnot-constraints/src/pipeline.rs`

### Exp 95: PyO3 pipeline bridge
Expose the Rust pipeline through PyO3 bindings:
- Add `RustVerifyRepairPipeline` to `carnot-python` crate
- Python API: `from carnot.bindings import RustVerifyRepairPipeline`
- Cross-language conformance: same 100 inputs → identical verification results
- Benchmark: Python-only vs Rust-via-PyO3 latency
- Auto-select: if `RUST_AVAILABLE`, use Rust backend; else Python
- **Metric:** Latency improvement, conformance test pass rate
- **Deliverable:** `crates/carnot-python/src/pipeline.rs`

## Dependencies

```
2026.04.4 outputs (done):
  Exp 74 ✓ (ConstraintExtractor API)
  Exp 75 ✓ (VerifyRepairPipeline)
  Exp 78 ✓ (PyPI package)
  Exp 82 ✓ (Error handling)
  Exp 83 ✓ (Performance benchmarks)

Phase 17 (differentiable):
  Exp 66 ← Exp 64 (continuous relaxation) + Exp 65 (embedding constraints)
  Exp 86 ← Exp 66 (differentiable pipeline)
  Exp 87 ← Exp 66 + Exp 65 (embedding repair)

Phase 18 (self-improvement):
  Exp 88 ← Exp 58 (benchmark results) + Exp 74 (extractor API)
  Exp 89 ← Exp 88 (failure analysis) + Exp 62 (domain CD training)
  Exp 90 ← Exp 89 (self-bootstrap) + autoresearch orchestrator

Phase 19 (real benchmarks):
  Exp 91 ← Exp 75 (pipeline) + Qwen3.5-0.8B + Gemma4-E4B-it
  Exp 92 ← Exp 91 (GSM8K infra) + Exp 74 (arithmetic extractor)
  Exp 93 ← Exp 91 + Exp 92 (both benchmark infra)

Phase 20 (Rust):
  Exp 94 ← Exp 70 (Rust constraints) + Exp 74 (extractor design)
  Exp 95 ← Exp 94 (Rust pipeline)
```

## Execution Order

```
1. exp66   -- Differentiable constraint reasoning (foundation for Phase 17)
2. exp86   -- Learned energy composition (uses Exp 66 gradients)
3. exp87   -- Gradient-based repair (uses Exp 66 + Exp 65)
4. exp88   -- Failure-driven constraint mining (analyze past results)
5. exp89   -- Self-bootstrapped constraint training (learn from pipeline outputs)
6. exp91   -- GSM8K live benchmark (first real inference)
7. exp92   -- MATH benchmark subset (step-level verification)
8. exp93   -- Multi-model systematic comparison (full matrix)
9. exp90   -- Autoresearch constraint improvement loop (20 iterations)
10. exp94  -- Rust VerifyRepairPipeline (port to Rust)
11. exp95  -- PyO3 pipeline bridge (expose to Python)
```

## Hardware Requirements

| Experiment | Compute | Memory | Time est. |
|-----------|---------|--------|-----------|
| 66 | CPU + JAX | 4GB | 30-60 min |
| 86 | CPU + JAX | 4GB | 30 min |
| 87 | CPU + JAX + sentence-transformers | 4GB | 30 min |
| 88 | CPU | 2GB | 15 min |
| 89 | CPU + Qwen3.5-0.8B | 4GB | 1-2 hours |
| 90 | CPU | 2GB | 1-2 hours (20 iterations) |
| 91 | CPU + Qwen3.5 + Gemma4 | 8GB | 1-2 hours |
| 92 | CPU + Qwen3.5 + Gemma4 | 8GB | 1-2 hours |
| 93 | CPU + Qwen3.5 + Gemma4 | 8GB | 2-3 hours |
| 94 | CPU (Rust build) | 2GB | 30-60 min |
| 95 | CPU (Rust + PyO3) | 2GB | 30-60 min |

## Success Criteria

- Differentiable constraint verification achieves ≥ discrete pipeline AUROC
- Self-bootstrapped Ising models improve verification accuracy by ≥ 5%
- Autoresearch loop produces ≥ 1 accepted extractor improvement in 20 iterations
- GSM8K live accuracy improvement ≥ 3% over baseline (with Carnot verify-repair)
- MATH step-level verification catches ≥ 50% of intermediate errors
- Multi-model comparison completes for all 1500 cells
- Rust pipeline achieves ≥ 10x throughput vs Python
- PyO3 bridge passes 100% cross-language conformance

## What's Explicitly NOT in Scope

- **Activation-based EBMs** — proven insufficient (experiments 8-38)
- **New model architectures** — Qwen3.5-0.8B and Gemma4-E4B-it are sufficient
- **TSU hardware integration** — hardware not available; TsuBackend stub is enough
- **PyPI publish** — beta release prep is done (Exp 85), actual publish is a release task
- **Kona full implementation** — Exp 66/87 are stepping stones, not the full Kona vision
- **RLHF/DPO training** — we verify and repair, not fine-tune

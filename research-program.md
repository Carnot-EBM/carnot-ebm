# Carnot Research Program

**Purpose:** This document declares the high-level research goals, constraints,
and priorities for the autonomous research conductor and planning agent.
Humans edit this file to steer research direction. The planning agent reads
it when designing new milestones.

## Mission

Escape LLM hallucinations via verifiable constraint reasoning, and enable
autonomous directed self-learning where the energy function is ground truth.

## Current Strategic Goals (in priority order)

1. **Fix live model loading in experiments** — ENGINEERING, NOT RESEARCH.
   Qwen3.5-0.8B loads fine when run directly (proved in Exp 56: 19/20 with
   live inference). But the conductor's `claude -p` subprocess falls back to
   simulated inference inconsistently. Fix: detect available memory, use
   torch_dtype=float32 on CPU, retry on failure. Until this works reliably,
   all benchmark numbers are synthetic. This unblocks goals #5 and #6.

2. **Multi-turn / agentic verification** — BIGGEST UNTAPPED OPPORTUNITY.
   Verify not just single Q&A but multi-step agent workflows. An agent that
   plans → acts → observes should have each step constraint-verified. No one
   else is doing this. Concrete research needed:
   - Constraint propagation across steps (step 1's output constraints become
     step 2's input constraints)
   - State accumulation (track which facts have been verified vs assumed)
   - Rollback semantics (when step 3 fails, which earlier step to repair?)
   The verify-repair loop (Exp 57) works for single turns; extend it to
   chains of reasoning. This is critical for autonomous agents and is the
   most differentiating product feature Carnot could offer.

3. **Factual constraint extractor** — CLOSES BIGGEST COVERAGE GAP.
   Exp 88 showed factual and scheduling domains have near-zero constraint
   coverage (100% false negative rate). No amount of constraint learning
   helps if we can't extract constraints from factual claims. Research
   needed: knowledge-base-backed extractor that verifies claims against
   Wikidata/Wikipedia. This is fundamentally different from arithmetic/code/
   logic constraints and requires a new approach.

4. **Guided decoding latency benchmark** — QUICK EXPERIMENT, HIGH SIGNAL.
   Exp 66 proved differentiable constraints work (1.0 AUROC). The remaining
   question: can a single constraint check run fast enough during token
   generation? If <1ms → viable for real-time guided decoding (Kona). If
   >10ms → too slow, need approximations. One experiment answers this.

5. **Apple GSM8K adversarial benchmark** — THE CREDIBILITY EXPERIMENT.
   Apple (arxiv 2410.05229) proved LLMs can't do math — they pattern-match.
   Swapping numbers or adding one irrelevant sentence drops accuracy up to
   65%. Even o1-preview drops from 92.7% → 77.4%. 8-shot doesn't help.
   RUN CARNOT ON THE ADVERSARIAL VARIANT. Show that:
   - LLM accuracy drops (as Apple showed)
   - Carnot verify-repair MAINTAINS accuracy (Ising catches arithmetic
     errors regardless of irrelevant context)
   - Improvement is LARGER on adversarial vs standard (more errors to catch)
   This is the single most compelling experiment we can run. It directly
   proves the value proposition: external constraint verification succeeds
   where internal reasoning fails. See research-references.md for details.

6. **Real benchmark validation at scale** — NEEDS GOAL #1 FIRST.
   GSM8K full 1,319 test set + HumanEval full 164 problems with LIVE model
   inference. Exp 91 showed +14-15% on 200 questions with simulated
   inference. Report confidence intervals, compare to published baselines.
   Also TruthfulQA for factual domain. These are the credibility numbers
   but are blocked until live model loading is reliable.

7. **Scale constraint learning** — GOOD PROGRESS, CONTINUE.
   Exp 55/62/63/88/89 built the foundation. Next: use failure mining results
   (Exp 88) to build the `intermediate_result` extractor that would catch
   44.8% of current false negatives. Then self-bootstrap on the expanded
   constraint set.

8. **KAN-based energy tier** — NEW ARCHITECTURE, HIGH IMPACT.
   Kolmogorov-Arnold Networks as a new energy function tier between Ising
   (quadratic, interpretable) and Gibbs (MLP, opaque). KAN edges have
   learnable spline activations — strictly more expressive than Ising while
   remaining interpretable. Addresses the constraint learning ceiling from
   Exp 62/88 where linear Ising features can't capture nonlinear
   relationships. Differentiable (slots into Exp 66 pipeline). Potentially
   hardware-mappable (spline lookup tables in FPGA). Create `carnot-kan`
   crate and `carnot.models.kan` Python module.

9. **LNN-based adaptive constraints for agentic verification** — PAIRS WITH #2.
   Liquid Neural Networks for constraint models that adapt during multi-turn
   agent workflows. Static Ising can't update as new facts emerge during
   agent execution. LNN coupling strengths evolve via differential equations
   in response to observations. Also improves noise robustness for
   constraint extraction from adversarial LLM outputs (Exp 88 failure mode).

10. **Bridge to continuous reasoning (Kona direction)** — VALIDATED, DEPENDS ON #4.
   Continuous Ising relaxation (Exp 64 ✅) → embedding-space constraints
   (Exp 65 ✅) → end-to-end differentiable (Exp 66 ✅, 1.0 AUROC) →
   gradient repair (Exp 87 ✅, 44% energy reduction). The math works.
   Practicality depends on latency benchmark (#4). If fast enough, pursue
   energy-guided decoding. If not, focus on post-hoc verify-repair which
   is already proven effective. KAN energy tier (#7) may improve the
   differentiable pipeline's expressiveness.

11. **FPGA Ising machine as TSU stand-in** — DON'T WAIT FOR HARDWARE.
   SamplerBackend abstraction built (Exp 71). Instead of waiting for the
   Z1, implement a parallel Ising sampler on FPGA (1k-10k p-bits on
   Kria/DE10-Nano, up to 256k on large FPGAs). Create `FpgaBackend` that
   sends couplings over PCIe/AXI and reads back spins. Benchmark vs CPU
   ParallelIsingSampler on 5000-var SAT. This validates the hardware path
   and gives real latency numbers for guided decoding feasibility.
   See research-references.md for prior art (Tohoku, Microsoft, Fujitsu).

## Continuous Self-Learning (CORE ARCHITECTURAL GOAL)

Carnot must get smarter over time. Every query, every verification, every
repair should make the next one faster and more accurate. This is the path
to autonomous directed self-learning (the PRD's FR-11).

**The key constraint:** Every learning mechanism must have a hardware
acceleration path. We're building for hybrid compute: CPU + system memory
for orchestration, GPU/NPU/APU for batch training, FPGA/TSU for sampling.
Avoid the classic LLM problem where training is 1000x slower than inference.
Learning should happen AT inference speed, not as a separate offline phase.

### Tier 1: Online Constraint Learning (NEAR-TERM, PRACTICAL)
- Track which constraints fire and which errors they catch across queries
- Upweight constraints that catch real errors, downweight noisy ones
- Running average of per-constraint precision, updated after each cycle
- **Hardware path:** Pure CPU — just counter updates, no matrix ops
- **Implementation:** Add to VerifyRepairPipeline as opt-in mode
- **Learning speed:** Instant (one counter update per verification)

### Tier 2: Constraint Memory / Trace2Skill (MEDIUM-TERM, PAIRS WITH AGENTIC)
- Cache verified facts across sessions, not just within a chain
- Learn per-user and per-domain error patterns ("this codebase always has
  off-by-one errors" → auto-add loop bound constraints)
- ConstraintStateMachine (Exp 125) persists across agent sessions
- Consolidate learned patterns into reusable constraint templates
- **Hardware path:** CPU + system memory for storage, FPGA for fast
  pattern matching against constraint template library
- **Learning speed:** Minutes (accumulate across session, consolidate overnight)

### Tier 3: JEPA-Style Predictive Verification (RESEARCH FRONTIER)
- Train a model to predict constraint violations BEFORE the LLM finishes
- Input: partial response (first N tokens) → predict which constraints
  will be violated in the full response
- Enables preemptive guided decoding: steer away from violations before
  they happen, instead of checking after each token
- Combines with guided decoding (Exp 110) — predict instead of check
- **Hardware path:** Small predictor model on GPU/NPU (batch inference),
  Ising sampling on FPGA/TSU for energy evaluation
- **Learning speed:** CD training on accumulated (partial, violation) pairs.
  With FPGA-accelerated sampling, training could run continuously in
  background without blocking inference.

### Tier 4: Adaptive Energy Landscapes (LONG-TERM, KONA)
- Energy function itself evolves based on the distribution of queries
- Not just adapting weights (that's Tier 1-2) but adapting the STRUCTURE
  of the energy function — adding new constraint types, removing obsolete
  ones, merging redundant constraints
- KAN splines naturally support this: add knots where energy landscape
  is complex, remove where it's smooth (adaptive mesh refinement)
- **Hardware path:** FPGA reconfiguration for structural changes to the
  Ising/KAN coupling graph. TSU hardware would need reprogramming for
  new coupling topologies.
- **Learning speed:** Hours (structural changes are expensive but rare).
  The goal: one structural update per day based on accumulated statistics.

### Hardware Acceleration Principle
Every learning tier must answer: "How does this run 100x faster on
hardware?" If the answer is "it can't," redesign it until it can.

| Tier | Learning | Hardware | Speed Target |
|------|----------|----------|-------------|
| 1: Online weights | Counter updates | CPU | <1μs per update |
| 2: Constraint memory | Pattern matching | CPU+FPGA | <1ms per lookup |
| 3: Predictive verify | Small model inference | GPU/NPU | <10ms per prediction |
| 4: Adaptive structure | Graph reconfiguration | FPGA/TSU | <1s per restructure |

### What LNN Taught Us (Exp 116)
LNN's continuous-time adaptation (dJ/dt = f(J, obs)) hurt performance
(10% vs 100% for static Ising) because constraint structures don't change
within a single reasoning chain. But the IDEA of adaptive couplings is
right — it just needs to operate at the right timescale:
- **Within a chain:** Static couplings (Ising/KAN) — proven effective
- **Across chains:** Online weight updates (Tier 1) — fast and cheap
- **Across sessions:** Constraint memory (Tier 2) — persistent learning
- **Across domains:** Predictive verification (Tier 3) — transfer learning
- **Across architectures:** Adaptive structure (Tier 4) — evolution

## Completed Goals

- ~~**Ship a usable product**~~ — ✅ DONE (2026.04.4). pip install carnot,
  VerifyRepairPipeline, CLI, MCP server, examples, docs, 0.1.0-beta1.

## Model Choices

- **Primary LLM:** Qwen3.5-0.8B (latest small Qwen, runs on CPU)
- **Secondary LLM:** google/gemma-4-E4B-it (latest small Gemma)
- **Do NOT use:** Older models (Llama, Phi) unless results demand architectural
  diversity. Prefer latest small SoTA for relevance.
- **Inference optimization:** Consider RotorQuant for larger models when memory
  is the bottleneck. Consider HISA for long-context verification.

## Research Constraints

- **JAX_PLATFORMS=cpu** is mandatory — ROCm GPU is slower than CPU on this machine
- **100% test coverage** required for all code changes
- **100% spec coverage** — every test traces to REQ-* or SCENARIO-*
- **Never modify research_conductor.py** from within experiment prompts
- **Experiments must be self-contained** — each produces a deliverable file
- **Dual language** — research in Python/JAX, production path in Rust

## What Works (do more of this)

- Ising constraint verification: 100% hallucination detection
- Verify-repair loop: +27% accuracy improvement
- Parallel Ising sampler: 183x faster than thrml
- CD training: learned couplings generalize to unseen instances
- Runtime instrumentation: catches bugs static analysis misses
- HumanEval pass@1+repair: 96% (up from 90% baseline)

## What Doesn't Work (don't repeat)

- Activation-based EBMs: detect confidence, not correctness (50% practical)
- Cross-model activation transfer: ~50% = chance
- Temperature diversity / domain mixing: hurts performance
- Normalization: destroys signal
- Logit lens: dynamics identical for correct/wrong
- Sentence/NLI encoders: embed topic, not truth

## Quality Standards

- Every experiment must run end-to-end and produce results
- Simulated fallbacks are acceptable but live LLM results preferred
- Doc reconciliation after every experiment (changelog, status, traceability)
- Adversarial review for non-trivial changes
- Never remove existing content from ops/spec docs

## HuggingFace Publishing Milestones

Artifacts to publish at huggingface.co/Carnot-EBM as research progresses:

1. **NOW: Update existing model READMEs** — Clarify that the 16 per-token
   activation EBMs are Phase 1 research artifacts (detect confidence, not
   correctness). Point users to `pip install carnot` for the actual product.

2. **NOW: Publish Exp 66 joint model** — The differentiable constraint
   model (embedding + Ising → score, 1.0 AUROC). Small proof-of-concept
   with clear README: "demonstrates the approach, not production quality."

3. **AFTER Goal #4 (latency benchmark):** If guided decoding is viable,
   publish the first **energy-guided decoding adapter** — a small module
   that adjusts LLM token probabilities based on constraint energy. Novel
   artifact, nothing like it exists on HuggingFace. Model-agnostic.

4. **AFTER Goal #2 (agentic verification):** Publish **constraint
   propagation models** — learned Ising models encoding how constraints
   flow across multi-step reasoning chains. Domain-specific variants for
   math, code, and planning.

5. **AFTER Goal #3 (factual extractor):** Publish **knowledge-grounded
   constraint models** — encode factual verification against knowledge
   bases. Larger, more broadly useful.

6. **KONA PARITY (6-12 months):** Publish the **differentiable reasoning
   module** — attach to any HuggingFace LLM for constraint-verified
   outputs. Like a LoRA but for reasoning verification. Model-agnostic,
   small, solves hallucination. This is the artifact that gets attention.

## When to Propose New Research vs Production Work

- If the pipeline has proven capabilities not yet shipped → production first
- If a fundamental limitation blocks progress → research to overcome it
- If external benchmarks show weakness in a domain → targeted research
- Default: alternate research and production milestones

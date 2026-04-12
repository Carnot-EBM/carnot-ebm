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

8. **KAN-based energy tier** — DONE (Exp 108-109). DEPLOYMENT GUIDANCE:
   KAN achieves 0.994 AUROC with 8.7x fewer params than Ising on the same
   task. Use the right tier for the right job:
   - **KAN = default for verification** — best accuracy/cost ratio, nonlinear
     edge detection, interpretable spline shapes, differentiable
   - **Ising = hardware and real-time sampling** — direct FPGA/TSU mapping,
     fastest parallel Gibbs (183x speedup relies on quadratic structure),
     coupling matrix is just wire strengths in hardware
   - **Gibbs MLP = research/complex patterns** — most expressive, opaque
   - **Boltzmann = large-scale generation** — deep residual, attention
   KAN and Ising complement each other: KAN for accuracy, Ising for speed.

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

- **GPU available:** 2x RTX 3090 (48GB VRAM total) via CUDA + iGPU (Radeon 890M,
  shares system RAM for large models). JAX_PLATFORMS=cpu no longer required.
  Use GPU for LLM inference and training. Use CPU for Ising sampling (fastest).
- **Agent allocation:** Codex (20x pro) for autoresearch conductor. Claude (20x max)
  for interactive work, planning, and code review. This prevents the conductor's
  high token consumption from exhausting the Claude weekly quota.
- **100% test coverage** required for all code changes
- **100% spec coverage** — every test traces to REQ-* or SCENARIO-*
- **Never modify research_conductor.py** from within experiment prompts
- **Experiments must be self-contained** — each produces a deliverable file
- **Dual language** — research in Python/JAX, production path in Rust

## Planning Phase Requirements

When designing new milestones, the planning agent MUST:
1. **Do arxiv research first** — Search for recent papers (2025-2026) on
   EBMs, constraint verification, Ising models, KANs, guided decoding,
   hardware-accelerated sampling, and LLM hallucination mitigation.
   Add promising findings to research-references.md BEFORE designing
   experiments. This ensures we don't miss accelerating ideas.
2. **Include at least one self-learning experiment** — Every milestone
   should advance the continuous self-learning architecture (Tiers 1-4).
3. **Check the hardware wishlist** — If an experiment would benefit from
   hardware in research-hardware-wishlist.md, note it in the experiment's
   description so we can prioritize acquisition.
4. **Update research-references.md** — Any new papers or tools discovered
   during planning must be filed for future reference.

## Lessons Learned: Large Benchmark Experiments

Experiments 181-183 (live GPU benchmarks) repeatedly failed because they
were too large for a single `claude -p` session (50 turns). Pattern:
- Claude writes 50-80KB script successfully
- Gets stuck in test-fix loop trying to reach 100% coverage
- Hits max turns, conductor retries, same result, exhausted after 3 failures

**Rule for future milestones:** Break large benchmarks into phases:
1. **Script generation** — write the benchmark script (no tests needed,
   deliverable = script file)
2. **Test coverage** — add tests for the new script (separate experiment)
3. **Execution** — run the script and capture results (deliverable = results JSON)
4. **Analysis** — analyze results and update docs (separate experiment)

Each phase is small enough for 50 turns. The checkpoint system preserves
work between phases.

**Also:** Dirty checkpoint files (exp181_ckpt, exp182_ckpt, exp183_ckpt)
need cleanup — consolidate partial results into proper experiment_NNN_results.json
files or remove if data is invalid.

## Next Milestone Focus (2026.04.10)

The planning agent MUST prioritize these for the next milestone:

1. **Make Tier 1 self-learning actually work** — Exp 134 showed that
   precision-based REWEIGHTING didn't improve accuracy (fixed=adaptive
   across all 500 questions). The infrastructure works (tracker, adaptive
   weighter) but the approach is wrong. The fix: constraint ADDITION from
   memory patterns, not just weight changes. When memory detects "arithmetic
   carry errors are common," ADD a carry-check constraint — don't just
   upweight existing ones. Tier 2 memory (Exp 135) has the pattern data;
   wire it into constraint generation, not just weight adjustment.

2. **JEPA predictive verification (Tier 3)** — Predict constraint violations
   from partial LLM responses BEFORE generation completes. Train a small
   predictor on (partial_response, final_violation) pairs from accumulated
   verify-repair logs. If it predicts "high energy likely," trigger full
   Ising verification; otherwise fast-path skip. This is the key to making
   guided decoding practical at scale.

3. **Actually upload to HuggingFace** — Exp 137 prepared the guided decoding
   adapter artifacts but didn't push them. Run `huggingface-cli upload` to
   publish. Also update the existing 16 model READMEs to point users to
   `pip install carnot` for the actual product.

4. **Integrate arxiv findings from Exp 139** — Whatever papers the scan
   discovers, incorporate the most promising as concrete experiments.

5. **AMD XDNA NPU experimentation** — The current machine has an unused
   NPU. Install the AMD XDNA driver + SDK and test whether small constraint
   models can run on the NPU. This is $0 cost and could unlock Tier 3
   (JEPA predictor on NPU while LLM runs on CPU). See
   research-hardware-wishlist.md for setup steps.

## What Works (do more of this)

- Ising constraint verification: 100% hallucination detection
- Verify-repair loop: +27% accuracy improvement, +26.5% on adversarial
- Adversarial robustness: Ising ignores irrelevant context (74% correctly silent)
- KAN energy tier: 0.994 AUROC with 8.7x fewer params than Ising
- Parallel Ising sampler: 183x faster than thrml
- CD training: learned couplings generalize to unseen instances
- Runtime instrumentation: catches bugs static analysis misses
- HumanEval pass@1+repair: 96% (up from 90% baseline)
- Guided decoding: 0.006ms per constraint check, 100% CSR
- Agentic verification: 60/60 violations caught in multi-step workflows
- Constraint state machine with rollback: works for multi-turn agents

## What Doesn't Work (don't repeat)

- Activation-based EBMs: detect confidence, not correctness (50% practical)
- Cross-model activation transfer: ~50% = chance
- Temperature diversity / domain mixing: hurts performance
- Normalization: destroys signal
- Logit lens: dynamics identical for correct/wrong
- Sentence/NLI encoders: embed topic, not truth
- LNN adaptive couplings within a chain: 10% vs 100% for static Ising (Exp 116)
- Precision-based constraint REWEIGHTING: no improvement over fixed weights (Exp 134)
  — need constraint ADDITION instead

## Research Sources

The planning agent searches these sources when designing milestones:

**Primary — arxiv.org:** EBMs, constraint satisfaction, Ising models, KANs,
guided decoding, FPGA/thermodynamic sampling, continual learning.

**Secondary:**
- **OpenReview.net** — NeurIPS/ICML/ICLR submissions (3-6 months before arxiv)
- **extropic.ai/writing** — TSU hardware updates and thermodynamic computing
- **Semantic Scholar** — Papers citing our key references (EBT, ARM-EBM bijection)
- **HuggingFace papers** (huggingface.co/papers) — verification/hallucination work
- **GitHub trending** — New EBM/constraint/KAN repos (Python + Rust)
- **logicalintelligence.com** — Kona architecture updates
- **FPGA conference proceedings** (FCCM, FPL, DAC) — Ising machine implementations
- **AMD developer forums** — Ryzen AI NPU SDK, XDNA updates, onnxruntime VitisAI

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

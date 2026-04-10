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

5. **Real benchmark validation at scale** — NEEDS GOAL #1 FIRST.
   GSM8K full 1,319 test set + HumanEval full 164 problems with LIVE model
   inference. Exp 91 showed +14-15% on 200 questions with simulated
   inference. Report confidence intervals, compare to published baselines.
   Also TruthfulQA for factual domain. These are the credibility numbers
   but are blocked until live model loading is reliable.

6. **Scale constraint learning** — GOOD PROGRESS, CONTINUE.
   Exp 55/62/63/88/89 built the foundation. Next: use failure mining results
   (Exp 88) to build the `intermediate_result` extractor that would catch
   44.8% of current false negatives. Then self-bootstrap on the expanded
   constraint set.

7. **Bridge to continuous reasoning (Kona direction)** — VALIDATED, DEPENDS ON #4.
   Continuous Ising relaxation (Exp 64 ✅) → embedding-space constraints
   (Exp 65 ✅) → end-to-end differentiable (Exp 66 ✅, 1.0 AUROC) →
   gradient repair (Exp 87 ✅, 44% energy reduction). The math works.
   Practicality depends on latency benchmark (#4). If fast enough, pursue
   energy-guided decoding. If not, focus on post-hoc verify-repair which
   is already proven effective.

8. **Prepare for Extropic TSU** — WAITING ON HARDWARE.
   SamplerBackend abstraction built (Exp 71). Nothing to do until hardware
   ships.

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

## When to Propose New Research vs Production Work

- If the pipeline has proven capabilities not yet shipped → production first
- If a fundamental limitation blocks progress → research to overcome it
- If external benchmarks show weakness in a domain → targeted research
- Default: alternate research and production milestones

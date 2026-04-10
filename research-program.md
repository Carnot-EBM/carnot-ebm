# Carnot Research Program

**Purpose:** This document declares the high-level research goals, constraints,
and priorities for the autonomous research conductor and planning agent.
Humans edit this file to steer research direction. The planning agent reads
it when designing new milestones.

## Mission

Escape LLM hallucinations via verifiable constraint reasoning, and enable
autonomous directed self-learning where the energy function is ground truth.

## Current Strategic Goals (in priority order)

1. **Ship a usable product** — `pip install carnot`, VerifyRepairPipeline as
   the main API, MCP server for Claude Code integration. Users should be
   able to verify LLM outputs in 5 lines of Python.

2. **Real benchmark validation at scale** — GSM8K and HumanEval scripts exist
   (Exp 67, 68, 91) but used simulated LLM inference. Run them with LIVE
   model inference at FULL scale to produce publishable numbers:
   - GSM8K: full 1,319 test set (not 200 subset), live Qwen3.5 + Gemma4
   - HumanEval: full 164 problems, live model generation
   - Report confidence intervals, compare to published baselines
   - Also TruthfulQA for factual domain
   Exp 91 showed +14-15% improvement on 200 questions with simulated
   inference — now prove it's real with live models on the full set.

3. **Multi-turn / agentic verification** — Verify not just single Q&A but
   multi-step agent workflows. An agent that plans → acts → observes should
   have each step constraint-verified. The verify-repair loop (Exp 57) works
   for single turns; extend it to chains of reasoning where each step's
   constraints propagate to the next. This is critical for autonomous agents.

4. **Scale constraint learning** — Move from hand-coded constraints to learned
   constraint structures. Domain-specific Ising models trained from data.

5. **Bridge to continuous reasoning (Kona direction)** — Continuous Ising
   relaxation (Exp 64 ✅) → embedding-space constraints (Exp 65 ✅) →
   **end-to-end differentiable constraint reasoning (Exp 66 — NEXT)**.
   Exp 66 is the critical next research experiment: backpropagate energy
   gradients through constraints to LLM logits, adjusting the sampling
   distribution toward constraint-satisfying tokens in real-time. This is
   the holy grail — energy-guided decoding where constraints steer
   generation, not just verify it post-hoc. See research-roadmap-v7.md
   Phase 8 for the full design.

6. **Prepare for Extropic TSU** — SamplerBackend abstraction is built (Exp 71).
   When hardware ships, plug it in.

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

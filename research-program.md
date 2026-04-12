# Carnot Research Program

**Purpose:** This document declares the high-level research goals, constraints,
and priorities for the autonomous research conductor and planning agent.
Humans edit this file to steer research direction. The planning agent reads
it when designing new milestones.

## Mission

Escape LLM hallucinations via verifiable constraint reasoning, and enable
autonomous directed self-learning where the energy function is ground truth.

## CRITICAL FINDING (2026-04-11): Simulation vs Reality

**ALL previous positive results were simulation artifacts.** Live GPU
testing revealed:
- Base model (Qwen3.5-0.8B): 25% GSM8K accuracy, 0% verify-repair improvement
- Base model (Qwen3-4B): 63% accuracy, -2% to -13% (verify-repair HARMFUL)
- Instruction-tuned (Gemma4-E4B-it): 80% accuracy, ZERO violations detected

**Root cause:** Simulated inference was calibrated to instruction-tuned
benchmarks (~65-70%) but experiments loaded BASE models. Additionally,
the ArithmeticExtractor uses regex pattern matching (`a + b = c`) which
is too crude — IT models don't write equations in that format, and base
models make errors the regex can't parse.

**What we now know works:**
- The constraint VERIFICATION infrastructure is solid (energy computation,
  Ising sampling, KAN, JEPA, pipeline architecture)
- The constraint EXTRACTION is the bottleneck — too crude for real models
- The self-learning, dogfooding, and conductor infrastructure works well

## Current Strategic Goals (in priority order)

1. **Rebuild constraint extraction for real models** — HIGHEST PRIORITY.
   The ArithmeticExtractor's regex is useless on instruction-tuned models
   (0 violations found on Gemma4-E4B-it). Two approaches:
   a. **NSVIF/Z3 SMT approach** (arxiv 2601.17789): Formalize constraints
      as first-order logic, solve with Z3. Zero false positives by design.
      Parse the model's chain-of-thought into logical steps, verify each.
   b. **LLM-as-extractor**: Use a second LLM call to extract verifiable
      claims from the response, then verify those claims with Ising/KAN.
      More flexible than regex, handles any response format.
   Both approaches must be tested on instruction-tuned models with LIVE
   GPU inference. No more simulated results.

2. **Establish REAL baselines with instruction-tuned models** — CREDIBILITY.
   All future experiments MUST:
   - Use instruction-tuned models (Gemma4-E4B-it, Qwen3.5 if IT available)
   - Use LIVE GPU inference (CARNOT_FORCE_LIVE=1, no simulation fallback)
   - Report inference_mode="live_gpu" in all results
   - Compare to published baselines for the same models
   Run 200 GSM8K + 50 HumanEval with new extraction on live IT models.

3. **Code verification (HumanEval)** — MOST LIKELY TO STILL WORK.
   Code verification uses structural tests (execute code, check output),
   NOT regex extraction. The CodeExtractor + runtime instrumentation +
   Ising-guided fuzzing pipeline may still show improvement because it
   verifies via EXECUTION, not pattern matching. Test on live GPU with
   Gemma4-E4B-it generating code.

4. **Multi-turn agentic verification** — STILL VALUABLE.
   The global consistency checker (Exp 172: 100% detection, 0% FP) works
   on logical consistency across steps, not arithmetic regex. This may
   be the domain where Carnot adds the most value — catching contradictions
   that span multiple reasoning steps.

5. **FPGA Ising machine** — HARDWARE PATH STILL VALID.
   The energy computation and Ising sampling infrastructure is proven
   (0.006ms per check, 183x faster than thrml). The hardware path is
   independent of extraction quality. KV260 arriving in 4 days.

6. **Bridge to Kona** — LONG-TERM, DEPENDS ON #1.
   Continuous reasoning, guided decoding, and differentiable constraints
   all depend on having constraint extraction that works on real models.
   Fix extraction first, then these become viable.

## What Was Invalidated

- ~~All GSM8K improvement numbers (+10-28%)~~ — simulation artifacts
- ~~Adversarial recovery claims~~ — tested on simulated, not live inference
- ~~Self-learning improvement (67→97%)~~ — trained on simulated error patterns
- ~~Full-scale benchmark results~~ — all simulated

## What Remains Valid

- Constraint verification infrastructure (Ising, KAN, Gibbs, Boltzmann)
- Energy computation speed (0.006ms per constraint check)
- Parallel Ising sampler (183x faster than thrml)
- Global consistency checker (100% detection on logical contradictions)
- Self-learning architecture (tracker, memory, JEPA — need real data)
- Dogfooding system (brace auto-fix works, CodeExtractor runs)
- Pipeline architecture (VerifyRepairPipeline, extractors, MCP server)
- FPGA/TSU hardware path (SamplerBackend abstraction ready)

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

4. **Investigate the model-size precision ceiling** — CRITICAL FINDING.
   Exp 184 showed verify-repair has 0% net improvement on 3B models
   (6 fixed, 6 broken — false positives cancel true fixes). This means
   our constraints are too coarse for stronger models. Experiments needed:

   a. **False positive analysis** — On the 6 broken cases from Exp 184,
      what did the ArithmeticExtractor flag? Were the violations real
      errors or valid intermediate steps that looked wrong? This tells
      us whether the problem is extraction precision or repair quality.

   b. **Constraint precision by model size** — Run the SAME 200 questions
      on 0.8B, 3B, and 13B. For each, measure: true positive rate,
      false positive rate, and net improvement. Plot the precision ceiling.
      Hypothesis: FP rate crosses TP rate between 1B and 3B.

   c. **Confidence-weighted constraints** — Instead of binary
      violated/not-violated, weight constraint violations by confidence.
      A "47+28=76" violation is high-confidence (definitely wrong).
      A "the intermediate result is approximately 150" is low-confidence.
      Only repair high-confidence violations. This should reduce FP on
      larger models while preserving TP.

   d. **Model-adaptive constraint thresholds** — Use the self-learning
      tracker (Exp 132) to learn per-model FP rates. When FP rate > TP
      rate for a constraint type, disable that constraint for that model.
      The system self-calibrates to each model's error patterns.

   e. **Semantic constraints for larger models** — Larger models make
      semantic errors (wrong problem setup, misinterpreted quantities),
      not arithmetic errors. We need constraint types that check LOGIC
      not just MATH. The global consistency checker (Exp 172, 100%
      detection) is a start — apply it to single-response verification.

5. **Integrate arxiv findings from Exp 139** — Whatever papers the scan
   discovers, incorporate the most promising as concrete experiments.

5. **AMD XDNA NPU experimentation** — The current machine has an unused
   NPU. Install the AMD XDNA driver + SDK and test whether small constraint
   models can run on the NPU. This is $0 cost and could unlock Tier 3
   (JEPA predictor on NPU while LLM runs on CPU). See
   research-hardware-wishlist.md for setup steps.

## What Works (verified on real models)

- KAN energy tier: 0.994 AUROC with 8.7x fewer params than Ising
- Parallel Ising sampler: 183x faster than thrml
- Energy computation speed: 0.006ms per constraint check
- Global consistency checker: 100% detection of cross-step contradictions
- CD training: learned couplings generalize to unseen instances
- Runtime instrumentation: catches bugs static analysis misses
- Dogfooding: auto-brace-fix, CodeExtractor on generated code
- Self-learning architecture: tracker, memory, JEPA (needs real training data)
- Pipeline infrastructure: VerifyRepairPipeline, MCP server, CLI
- FPGA/TSU hardware path: SamplerBackend abstraction ready

## What Was Invalidated (simulation artifacts)

- ~~Verify-repair +27% on GSM8K~~ — simulated inference, not live
- ~~Adversarial +26.5% recovery~~ — simulated inference
- ~~HumanEval 90→96%~~ — simulated inference (code execution MAY still work)
- ~~Self-learning 67→97%~~ — trained on simulated error patterns
- ~~Full-scale benchmark numbers~~ — all simulated baselines

## What Doesn't Work (don't repeat)

- **Regex-based constraint extraction on IT models** — ArithmeticExtractor
  found ZERO violations on Gemma4-E4B-it (0/20 questions, including 4 wrong)
- **Verify-repair on base models** — 0% improvement at 0.8B, -2% to -13% at 3B
- Activation-based EBMs: detect confidence, not correctness (50% practical)
- Cross-model activation transfer: ~50% = chance
- LNN adaptive couplings within a chain: 10% vs 100% for static Ising
- Precision-based constraint REWEIGHTING: 0% improvement
- Simulated inference as a proxy for live: produces unreliable results

## Outer Loop / Inner Loop Agent Architecture

- **Claude (outer loop):** Continuously researches novel ideas, ranks them by
  impact on current research state, and queues the most promising into the
  next roadmap milestone. Tracks findings in `research-studying.md` with
  scored rankings (relevance × novelty × feasibility × urgency).
- **Codex (inner loop):** Executes the current experiments and plans the
  future roadmap when empty. Follows AGENTS.md/CODEX.md workflow.
- **Principle:** Claude stays a step ahead — by the time Codex finishes
  a milestone, Claude has already identified the most promising next
  experiments from online research. Only top-ranked ideas get promoted
  to the roadmap; lower-ranked ideas stay in research-studying.md.

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
